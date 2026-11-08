"""Evaluate Milestone 7 transparent medication-ranking baselines."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import duckdb
import numpy as np

from pipeline.artifact_metadata import infer_consistent_version
from pipeline.config import (
    BASELINE_VERSION,
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_TEMP_DIR,
    DUCKDB_THREADS,
    EVALUATION_VERSION,
    FEATURE_VERSION,
    FEATURES_ROOT,
    LABEL_VERSION,
    MILESTONE7_EVALUATION_ROOT,
    RANDOM_SEED,
    REPORTS_ROOT,
    SPLIT_VERSION,
    TRAINING_ROOT,
)
from pipeline.extract_utils import (
    configure_duckdb_connection,
    parquet_scan,
    safe_error_message,
    sql_string,
)
from pipeline.features import copy_query_to_parquet, fetch_dict_rows
from pipeline.learned_baselines import (
    LEARNED_BASELINES,
    artifact_paths,
    combine_score_tables,
    fit_linear_model,
    fit_preprocessor,
    fit_xgboost_model,
    learned_manifest_section,
    materialize_learned_scores,
    resolve_feature_spec,
    save_learned_artifacts,
    training_sample_query,
    write_training_sample,
)


SCHEMA_VERSION = "milestone7-baseline-evaluation-v1"
COVERAGE_SCHEMA_VERSION = "milestone7-coverage-report-v1"
DEFAULT_COVERAGE_REPORT_PATH = REPORTS_ROOT / "milestone7_coverage_report.json"
DEFAULT_EVALUATION_REPORT_PATH = REPORTS_ROOT / "milestone7_baseline_evaluation.json"
DEFAULT_TOP_K = (1, 3, 5, 10)
NON_LEARNED_BASELINES = ("random", "global_popularity", "condition_popularity")
ALL_BASELINES = NON_LEARNED_BASELINES + LEARNED_BASELINES
MIN_SUBGROUP_POSITIVE_GROUPS = 25


@dataclass(frozen=True)
class BaselineEvaluationConfig:
    """Configuration for Milestone 7 baseline scoring and evaluation."""

    features_root: Path = FEATURES_ROOT
    training_root: Path = TRAINING_ROOT
    evaluation_root: Path = MILESTONE7_EVALUATION_ROOT
    coverage_report_path: Path = DEFAULT_COVERAGE_REPORT_PATH
    evaluation_report_path: Path = DEFAULT_EVALUATION_REPORT_PATH
    training_manifest_path: Path = REPORTS_ROOT / "training_table_manifest.json"
    top_k: tuple[int, ...] = DEFAULT_TOP_K
    mode: str = "development"
    frozen_selection: bool = False
    seed: int = RANDOM_SEED
    condition_tokens: tuple[str, ...] = ()
    baselines: tuple[str, ...] = NON_LEARNED_BASELINES
    evaluation_version: str = EVALUATION_VERSION
    baseline_version: str = BASELINE_VERSION
    feature_version: str | None = None
    label_version: str = LABEL_VERSION
    split_version: str = SPLIT_VERSION
    min_subgroup_positive_groups: int = MIN_SUBGROUP_POSITIVE_GROUPS
    duckdb_temp_directory: Path | None = DUCKDB_TEMP_DIR
    duckdb_memory_limit: str | None = DUCKDB_MEMORY_LIMIT
    duckdb_threads: int | None = DUCKDB_THREADS

    @property
    def patient_stay_features_path(self) -> Path:
        return self.features_root / "patient_stay_features.parquet"

    @property
    def patient_condition_medication_path(self) -> Path:
        return self.training_root / "patient_condition_medication.parquet"

    @property
    def candidate_catalog_path(self) -> Path:
        return self.training_root / "candidate_catalog.parquet"

    @property
    def score_output_path(self) -> Path:
        return self.evaluation_root / "baseline_scores.parquet"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_json_if_present(path: Path) -> dict[str, Any] | None:
    """Load a JSON file when available."""

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def configure_connection(
    config: BaselineEvaluationConfig,
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Apply shared memory-safe DuckDB settings."""

    configure_duckdb_connection(
        connection,
        temp_directory=config.duckdb_temp_directory,
        memory_limit=config.duckdb_memory_limit,
        threads=config.duckdb_threads,
    )


def missing_input_tables(config: BaselineEvaluationConfig) -> list[dict[str, str]]:
    """Return missing Milestone 6 inputs required for evaluation."""

    required = (
        ("patient_stay_features", config.patient_stay_features_path),
        ("patient_condition_medication", config.patient_condition_medication_path),
        ("candidate_catalog", config.candidate_catalog_path),
    )
    return [
        {"table_name": table_name, "path": str(path)}
        for table_name, path in required
        if not path.exists()
    ]


def resolve_feature_version(
    config: BaselineEvaluationConfig,
) -> BaselineEvaluationConfig:
    """Return config stamped from feature and ranking-table inputs."""

    version = infer_consistent_version(
        (config.patient_stay_features_path, config.patient_condition_medication_path),
        column_name="feature_version",
        declared_version=config.feature_version,
        fallback_version=FEATURE_VERSION,
    )
    return replace(config, feature_version=version)


def parse_top_k(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated top-k list."""

    values: list[int] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value <= 0:
            raise ValueError("top-k values must be positive")
        values.append(value)
    if not values:
        raise ValueError("at least one top-k value is required")
    return tuple(sorted(set(values)))


def parse_repeated_csv(values: Sequence[str] | None) -> tuple[str, ...]:
    """Parse repeated CLI values that may each contain comma-separated tokens."""

    if not values:
        return ()
    parsed: list[str] = []
    for value in values:
        parsed.extend(token.strip() for token in value.split(",") if token.strip())
    return tuple(dict.fromkeys(parsed))


def condition_filter_sql(config: BaselineEvaluationConfig, alias: str = "pcm") -> str:
    """Return an optional condition-token filter predicate."""

    if not config.condition_tokens:
        return "TRUE"
    tokens = ", ".join(sql_string(token) for token in config.condition_tokens)
    return f"{alias}.index_condition_token IN ({tokens})"


def scoring_scope_sql(config: BaselineEvaluationConfig, alias: str = "pcm") -> str:
    """Return the source/split scope used for row-level scoring."""

    if config.mode == "development":
        return (
            f"{alias}.source = 'mimiciv' AND {alias}.split IN ('train', 'validation')"
        )
    return (
        f"(({alias}.source = 'mimiciv' "
        f"AND {alias}.split IN ('train', 'validation', 'test')) "
        f"OR {alias}.source = 'eicu_crd')"
    )


def baseline_choice_sql(config: BaselineEvaluationConfig) -> str:
    """Return a SQL literal list for selected baselines."""

    return ", ".join(sql_string(baseline) for baseline in config.baselines)


def selected_baselines(config: BaselineEvaluationConfig) -> tuple[str, ...]:
    """Validate and return selected baselines."""

    unknown = sorted(set(config.baselines) - set(ALL_BASELINES))
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"unsupported baselines: {joined}")
    return config.baselines


def nonlearned_baselines(config: BaselineEvaluationConfig) -> tuple[str, ...]:
    """Return selected non-learned baselines."""

    return tuple(
        baseline
        for baseline in selected_baselines(config)
        if baseline in NON_LEARNED_BASELINES
    )


def learned_baselines(config: BaselineEvaluationConfig) -> tuple[str, ...]:
    """Return selected learned baselines."""

    return tuple(
        baseline
        for baseline in selected_baselines(config)
        if baseline in LEARNED_BASELINES
    )


def scores_query(config: BaselineEvaluationConfig, *, generated_at: str) -> str:
    """Build the row-level non-learned baseline score query."""

    baselines = nonlearned_baselines(config)
    if not baselines:
        raise ValueError("no non-learned baselines selected")
    table = config.patient_condition_medication_path
    scope = scoring_scope_sql(config)
    condition_filter = condition_filter_sql(config)
    seed = sql_string(str(config.seed))
    baseline_selects: list[str] = []

    if "random" in baselines:
        baseline_selects.append(
            f"""
SELECT
    source,
    split,
    ranking_group_id,
    index_condition_token,
    candidate_medication_token,
    candidate_rank,
    label_prescribed,
    'random' AS baseline_name,
    CAST(
        HASH(
            {seed} || '|' || ranking_group_id || '|'
            || candidate_medication_token
        ) AS DOUBLE
    ) / 18446744073709551615.0 AS score
FROM evaluation_rows
"""
        )

    if "global_popularity" in baselines:
        baseline_selects.append(
            """
SELECT
    rows.source,
    rows.split,
    rows.ranking_group_id,
    rows.index_condition_token,
    rows.candidate_medication_token,
    rows.candidate_rank,
    rows.label_prescribed,
    'global_popularity' AS baseline_name,
    CASE
        WHEN totals.total_positive_train_rows > 0
            THEN COALESCE(global_counts.positive_train_rows, 0)::DOUBLE
                / totals.total_positive_train_rows
        ELSE 0.0
    END AS score
FROM evaluation_rows AS rows
CROSS JOIN global_totals AS totals
LEFT JOIN global_counts
    ON rows.candidate_medication_token
        = global_counts.candidate_medication_token
"""
        )

    if "condition_popularity" in baselines:
        baseline_selects.append(
            """
SELECT
    rows.source,
    rows.split,
    rows.ranking_group_id,
    rows.index_condition_token,
    rows.candidate_medication_token,
    rows.candidate_rank,
    rows.label_prescribed,
    'condition_popularity' AS baseline_name,
    CASE
        WHEN condition_totals.total_positive_train_rows > 0
            THEN COALESCE(condition_counts.positive_train_rows, 0)::DOUBLE
                / condition_totals.total_positive_train_rows
        ELSE 0.0
    END AS score
FROM evaluation_rows AS rows
LEFT JOIN condition_totals
    ON rows.index_condition_token = condition_totals.index_condition_token
LEFT JOIN condition_counts
    ON rows.index_condition_token = condition_counts.index_condition_token
    AND rows.candidate_medication_token
        = condition_counts.candidate_medication_token
"""
        )

    unioned = (
        "\nUNION ALL\n".join(baseline_selects)
        or """
SELECT
    source,
    split,
    ranking_group_id,
    index_condition_token,
    candidate_medication_token,
    candidate_rank,
    label_prescribed,
    'none' AS baseline_name,
    0.0 AS score
FROM evaluation_rows
WHERE FALSE
"""
    )
    return f"""
WITH all_rows AS (
    SELECT
        source,
        split,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        candidate_rank,
        label_prescribed
    FROM {parquet_scan(table)} AS pcm
    WHERE {condition_filter}
),
evaluation_rows AS (
    SELECT *
    FROM all_rows AS pcm
    WHERE {scope}
),
train_rows AS (
    SELECT *
    FROM all_rows
    WHERE source = 'mimiciv'
        AND split = 'train'
),
global_counts AS (
    SELECT
        candidate_medication_token,
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END)
            AS positive_train_rows
    FROM train_rows
    GROUP BY candidate_medication_token
),
global_totals AS (
    SELECT
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END)
            AS total_positive_train_rows
    FROM train_rows
),
condition_counts AS (
    SELECT
        index_condition_token,
        candidate_medication_token,
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END)
            AS positive_train_rows
    FROM train_rows
    GROUP BY index_condition_token, candidate_medication_token
),
condition_totals AS (
    SELECT
        index_condition_token,
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END)
            AS total_positive_train_rows
    FROM train_rows
    GROUP BY index_condition_token
),
scored AS (
{unioned}
)
SELECT
    source,
    split,
    ranking_group_id,
    index_condition_token,
    candidate_medication_token,
    candidate_rank,
    label_prescribed,
    baseline_name,
    score,
    {config.seed} AS seed,
    {sql_string(config.baseline_version)} AS baseline_version,
    {sql_string(config.evaluation_version)} AS evaluation_version,
    {sql_string(generated_at)} AS generated_at
FROM scored
WHERE baseline_name IN ({", ".join(sql_string(baseline) for baseline in baselines)})
"""


def coverage_summary_query(config: BaselineEvaluationConfig) -> str:
    """Return aggregate coverage and evaluability counts."""

    table = config.patient_condition_medication_path
    condition_filter = condition_filter_sql(config)
    return f"""
WITH rows AS (
    SELECT
        source,
        split,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        label_prescribed
    FROM {parquet_scan(table)} AS pcm
    WHERE {condition_filter}
),
groups AS (
    SELECT
        source,
        split,
        ranking_group_id,
        MAX(CASE WHEN label_prescribed THEN 1 ELSE 0 END)
            AS has_in_catalog_positive,
        COUNT(*) AS group_candidate_count
    FROM rows
    GROUP BY source, split, ranking_group_id
),
row_summary AS (
    SELECT
        source,
        split,
        COUNT(*) AS candidate_row_count,
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END)
            AS in_catalog_positive_row_count,
        COUNT(DISTINCT index_condition_token) AS index_condition_count,
        COUNT(DISTINCT candidate_medication_token) AS candidate_medication_count
    FROM rows
    GROUP BY source, split
),
group_summary AS (
    SELECT
        source,
        split,
        COUNT(*) AS ranking_group_count,
        SUM(CASE WHEN has_in_catalog_positive > 0 THEN 1 ELSE 0 END)
            AS positive_ranking_group_count,
        SUM(CASE WHEN has_in_catalog_positive = 0 THEN 1 ELSE 0 END)
            AS no_positive_ranking_group_count,
        MIN(group_candidate_count) AS min_candidates_per_group,
        AVG(group_candidate_count) AS mean_candidates_per_group,
        MAX(group_candidate_count) AS max_candidates_per_group
    FROM groups
    GROUP BY source, split
)
SELECT
    row_summary.source,
    row_summary.split,
    row_summary.candidate_row_count,
    row_summary.in_catalog_positive_row_count,
    row_summary.index_condition_count,
    row_summary.candidate_medication_count,
    group_summary.ranking_group_count,
    group_summary.positive_ranking_group_count,
    group_summary.no_positive_ranking_group_count,
    group_summary.min_candidates_per_group,
    group_summary.mean_candidates_per_group,
    group_summary.max_candidates_per_group
FROM row_summary
INNER JOIN group_summary
    ON row_summary.source = group_summary.source
    AND row_summary.split = group_summary.split
ORDER BY row_summary.source, row_summary.split
"""


def candidate_catalog_summary_query(config: BaselineEvaluationConfig) -> str:
    """Return aggregate candidate catalog coverage."""

    catalog = config.candidate_catalog_path
    if config.condition_tokens:
        tokens = ", ".join(sql_string(token) for token in config.condition_tokens)
        where = f"WHERE index_condition_token IN ({tokens})"
    else:
        where = ""
    return f"""
SELECT
    COUNT(DISTINCT index_condition_token) AS index_condition_count,
    COUNT(*) AS candidate_count,
    MIN(candidate_rank) AS min_candidate_rank,
    MAX(candidate_rank) AS max_candidate_rank,
    SUM(positive_train_stay_count) AS positive_train_stay_count
FROM {parquet_scan(catalog)}
{where}
"""


def training_manifest_coverage(
    config: BaselineEvaluationConfig,
) -> dict[str, Any]:
    """Return aggregate coverage details copied from the Milestone 6 manifest."""

    manifest = load_json_if_present(config.training_manifest_path)
    if manifest is None:
        return {
            "training_manifest_path": str(config.training_manifest_path),
            "status": "missing",
            "out_of_catalog_positives": [],
        }
    return {
        "training_manifest_path": str(config.training_manifest_path),
        "status": "available",
        "out_of_catalog_positives": manifest.get("out_of_catalog_positives", []),
        "training_rows_by_source_split": manifest.get(
            "training_rows_by_source_split",
            [],
        ),
    }


def add_evaluability_status(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach performance-evaluability labels to source/split coverage rows."""

    updated: list[dict[str, Any]] = []
    for row in rows:
        positive_groups = int(row.get("positive_ranking_group_count") or 0)
        source = row.get("source")
        split = row.get("split")
        if positive_groups == 0:
            status = "coverage_only_no_in_catalog_positive_groups"
        elif source == "eicu_crd":
            status = "external_evaluable_when_final_mode_is_frozen"
        elif split == "train":
            status = "diagnostic_only_train"
        else:
            status = "evaluable"
        enriched = dict(row)
        enriched["performance_status"] = status
        updated.append(enriched)
    return updated


def build_coverage_report(
    connection: duckdb.DuckDBPyConnection,
    config: BaselineEvaluationConfig,
    *,
    generated_at: str,
) -> dict[str, Any]:
    """Build aggregate-only Milestone 7 candidate coverage report."""

    coverage_rows = fetch_dict_rows(connection, coverage_summary_query(config))
    return {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed",
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_level_scores": False,
            "local_score_artifacts_contain_patient_level_rows": True,
        },
        "parameters": {
            "condition_tokens": list(config.condition_tokens),
            "development_source": "mimiciv",
            "external_validation_source": "eicu_crd",
        },
        "candidate_catalog": fetch_dict_rows(
            connection,
            candidate_catalog_summary_query(config),
        )[0],
        "source_split_coverage": add_evaluability_status(coverage_rows),
        "milestone6_manifest_coverage": training_manifest_coverage(config),
        "notes": [
            (
                "Performance metrics are averaged over ranking groups with at "
                "least one in-catalog positive."
            ),
            (
                "eICU splits with zero in-catalog positive groups are coverage "
                "checks only, not external performance estimates."
            ),
        ],
    }


def base_manifest(
    config: BaselineEvaluationConfig,
    *,
    status: str,
    generated_at: str,
    learned_baselines_section: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common evaluation manifest shell."""

    if learned_baselines_section is None:
        if learned_baselines(config):
            learned_baselines_section = {
                "status": "pending",
                "reason": "Learned baseline training did not complete.",
            }
        else:
            learned_baselines_section = {
                "status": "not_requested",
                "reason": "No learned baselines were selected.",
            }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "generated_at": generated_at,
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_level_scores": False,
            "local_score_artifacts_contain_patient_level_rows": True,
            "local_score_artifact_storage": "ignored Dataset/processed/evaluation",
        },
        "parameters": {
            "mode": config.mode,
            "frozen_selection": config.frozen_selection,
            "top_k": list(config.top_k),
            "seed": config.seed,
            "condition_tokens": list(config.condition_tokens),
            "baselines": list(config.baselines),
            "min_subgroup_positive_groups": config.min_subgroup_positive_groups,
        },
        "versions": {
            "baseline_version": config.baseline_version,
            "evaluation_version": config.evaluation_version,
            "feature_version": config.feature_version or FEATURE_VERSION,
            "label_version": config.label_version,
            "split_version": config.split_version,
        },
        "inputs": {
            "patient_stay_features": str(config.patient_stay_features_path),
            "patient_condition_medication": str(
                config.patient_condition_medication_path,
            ),
            "candidate_catalog": str(config.candidate_catalog_path),
            "training_manifest": str(config.training_manifest_path),
        },
        "coverage_report": str(config.coverage_report_path),
        "artifacts": {},
        "learned_baselines": learned_baselines_section,
        "observational_label_caveat": (
            "Labels are observed historical prescriptions in the label window. "
            "Unobserved catalog candidates are weak negatives, not confirmed "
            "clinical non-indications."
        ),
    }


def score_learned_baselines(
    connection: duckdb.DuckDBPyConnection,
    config: BaselineEvaluationConfig,
    *,
    generated_at: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Train learned baselines and materialize local row-level scores."""

    selected = learned_baselines(config)
    artifacts = artifact_paths(config.evaluation_root)
    feature_spec = resolve_feature_spec(connection, config.patient_stay_features_path)
    train_query = training_sample_query(
        patient_condition_medication_path=config.patient_condition_medication_path,
        patient_stay_features_path=config.patient_stay_features_path,
        feature_spec=feature_spec,
        condition_filter_sql=condition_filter_sql(config),
        seed=config.seed,
    )
    write_training_sample(
        connection,
        query=train_query,
        output_path=artifacts.training_sample_path,
    )
    preprocessor, training_counts = fit_preprocessor(
        artifacts.training_sample_path,
        feature_spec,
    )
    models: dict[str, Any] = {}
    if "linear" in selected:
        models["linear"] = fit_linear_model(
            artifacts.training_sample_path,
            feature_spec,
            preprocessor,
            config.seed,
        )
    if "xgboost" in selected:
        models["xgboost"] = fit_xgboost_model(
            artifacts.training_sample_path,
            feature_spec,
            preprocessor,
            config.seed,
        )
    save_learned_artifacts(
        artifacts,
        preprocessor=preprocessor,
        models=models,
        feature_spec=feature_spec,
    )
    learned_scores_path = config.evaluation_root / "_baseline_scores_learned.parquet"
    row_count = materialize_learned_scores(
        connection,
        output_path=learned_scores_path,
        patient_condition_medication_path=config.patient_condition_medication_path,
        patient_stay_features_path=config.patient_stay_features_path,
        feature_spec=feature_spec,
        condition_filter_sql=condition_filter_sql(config),
        scoring_scope_sql=scoring_scope_sql(config),
        baselines=selected,
        preprocessor=preprocessor,
        models=models,
        seed=config.seed,
        baseline_version=config.baseline_version,
        evaluation_version=config.evaluation_version,
        generated_at=generated_at,
    )
    manifest_section = learned_manifest_section(
        status="completed",
        feature_spec=feature_spec,
        training_counts=training_counts,
        artifacts=artifacts,
        baselines=selected,
        seed=config.seed,
    )
    record = {
        "table_name": "baseline_scores_learned",
        "output_path": str(learned_scores_path),
        "status": "completed",
        "row_count": row_count,
    }
    return record, manifest_section, learned_scores_path


def score_artifact_record(
    connection: duckdb.DuckDBPyConnection,
    config: BaselineEvaluationConfig,
    *,
    generated_at: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Materialize local row-level score artifacts and return aggregate records."""

    score_parts: list[Path] = []
    table_records: list[dict[str, Any]] = []
    learned_section: dict[str, Any] | None = None

    if nonlearned_baselines(config):
        nonlearned_path = config.evaluation_root / "_baseline_scores_nonlearned.parquet"
        try:
            row_count = copy_query_to_parquet(
                connection,
                scores_query(config, generated_at=generated_at),
                nonlearned_path,
            )
        except Exception as error:
            return (
                {
                    "table_name": "baseline_scores",
                    "output_path": str(config.score_output_path),
                    "status": "failed",
                    "row_count": None,
                    "reason": safe_error_message(error),
                },
                None,
            )
        score_parts.append(nonlearned_path)
        table_records.append(
            {
                "table_name": "baseline_scores_nonlearned",
                "output_path": str(nonlearned_path),
                "status": "completed",
                "row_count": row_count,
            }
        )

    if learned_baselines(config):
        try:
            learned_record, learned_section, learned_path = score_learned_baselines(
                connection,
                config,
                generated_at=generated_at,
            )
        except Exception as error:
            return (
                {
                    "table_name": "baseline_scores",
                    "output_path": str(config.score_output_path),
                    "status": "failed",
                    "row_count": None,
                    "reason": safe_error_message(error),
                },
                {
                    "status": "failed",
                    "reason": safe_error_message(error),
                },
            )
        score_parts.append(learned_path)
        table_records.append(learned_record)

    try:
        row_count = combine_score_tables(
            connection,
            score_paths=score_parts,
            output_path=config.score_output_path,
        )
    except Exception as error:
        return (
            {
                "table_name": "baseline_scores",
                "output_path": str(config.score_output_path),
                "status": "failed",
                "row_count": None,
                "reason": safe_error_message(error),
            },
            learned_section,
        )

    return (
        {
            "table_name": "baseline_scores",
            "output_path": str(config.score_output_path),
            "status": "completed",
            "row_count": row_count,
            "tables": table_records,
        },
        learned_section,
    )


def metric_slice_predicate(slice_row: dict[str, Any]) -> str:
    """Return a SQL predicate restricting a metric query to one score slice.

    Metric window functions sort each ``(baseline_name, source, split)``
    partition. In final mode a single partition spans tens of millions of rows
    across every baseline and split, so running one global query forces DuckDB
    to sort the entire combined score table at once and exhausts the temp/memory
    budget. Filtering to a single slice bounds every sort to one partition.
    """

    return (
        f"baseline_name = {sql_string(str(slice_row['baseline_name']))} "
        f"AND source = {sql_string(str(slice_row['source']))} "
        f"AND split = {sql_string(str(slice_row['split']))}"
    )


def metric_slices(
    connection: duckdb.DuckDBPyConnection,
    config: BaselineEvaluationConfig,
) -> list[dict[str, Any]]:
    """Return the distinct ``(baseline_name, source, split)`` score slices."""

    scores = config.score_output_path
    query = f"""
SELECT DISTINCT baseline_name, source, split
FROM {parquet_scan(scores)}
ORDER BY baseline_name, source, split
"""
    return fetch_dict_rows(connection, query)


def row_level_metric_query(
    config: BaselineEvaluationConfig,
    *,
    slice_predicate: str = "TRUE",
) -> str:
    """Return aggregate row-level AP, ROC-AUC, Brier, and ECE metrics.

    ``slice_predicate`` restricts the scan to a single
    ``(baseline_name, source, split)`` slice so the window sorts stay bounded.
    """

    scores = config.score_output_path
    return f"""
WITH rows AS (
    SELECT
        baseline_name,
        source,
        split,
        candidate_rank,
        candidate_medication_token,
        CASE WHEN label_prescribed THEN 1 ELSE 0 END AS label_int,
        score
    FROM {parquet_scan(scores)}
    WHERE {slice_predicate}
),
counts AS (
    SELECT
        baseline_name,
        source,
        split,
        COUNT(*) AS row_count,
        SUM(label_int) AS positive_row_count,
        COUNT(*) - SUM(label_int) AS negative_row_count
    FROM rows
    GROUP BY baseline_name, source, split
),
ranked_desc AS (
    SELECT
        rows.*,
        ROW_NUMBER() OVER (
            PARTITION BY baseline_name, source, split
            ORDER BY score DESC, candidate_rank, candidate_medication_token
        ) AS rank_desc,
        SUM(label_int) OVER (
            PARTITION BY baseline_name, source, split
            ORDER BY score DESC, candidate_rank, candidate_medication_token
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_positives
    FROM rows
),
average_precision AS (
    SELECT
        ranked_desc.baseline_name,
        ranked_desc.source,
        ranked_desc.split,
        CASE
            WHEN counts.positive_row_count > 0
                AND counts.negative_row_count > 0
                THEN SUM(
                    CASE
                        WHEN ranked_desc.label_int = 1
                            THEN ranked_desc.cumulative_positives::DOUBLE
                                / ranked_desc.rank_desc
                        ELSE 0.0
                    END
                ) / counts.positive_row_count
            ELSE NULL
        END AS average_precision
    FROM ranked_desc
    INNER JOIN counts
        ON ranked_desc.baseline_name = counts.baseline_name
        AND ranked_desc.source = counts.source
        AND ranked_desc.split = counts.split
    GROUP BY
        ranked_desc.baseline_name,
        ranked_desc.source,
        ranked_desc.split,
        counts.positive_row_count,
        counts.negative_row_count
),
ranked_asc AS (
    SELECT
        rows.*,
        RANK() OVER (
            PARTITION BY baseline_name, source, split
            ORDER BY score ASC
        ) AS min_rank_asc,
        COUNT(*) OVER (
            PARTITION BY baseline_name, source, split, score
        ) AS tie_count
    FROM rows
),
roc_auc AS (
    SELECT
        ranked_asc.baseline_name,
        ranked_asc.source,
        ranked_asc.split,
        CASE
            WHEN counts.positive_row_count > 0
                AND counts.negative_row_count > 0
                THEN (
                    SUM(
                        CASE
                            WHEN ranked_asc.label_int = 1
                                THEN ranked_asc.min_rank_asc
                                    + (ranked_asc.tie_count - 1) / 2.0
                            ELSE 0.0
                        END
                    )
                    - counts.positive_row_count
                        * (counts.positive_row_count + 1) / 2.0
                )
                / (counts.positive_row_count * counts.negative_row_count)
            ELSE NULL
        END AS roc_auc
    FROM ranked_asc
    INNER JOIN counts
        ON ranked_asc.baseline_name = counts.baseline_name
        AND ranked_asc.source = counts.source
        AND ranked_asc.split = counts.split
    GROUP BY
        ranked_asc.baseline_name,
        ranked_asc.source,
        ranked_asc.split,
        counts.positive_row_count,
        counts.negative_row_count
),
brier AS (
    SELECT
        rows.baseline_name,
        rows.source,
        rows.split,
        CASE
            WHEN counts.positive_row_count > 0
                AND counts.negative_row_count > 0
                THEN AVG(POWER(rows.score - rows.label_int, 2))
            ELSE NULL
        END AS brier_score
    FROM rows
    INNER JOIN counts
        ON rows.baseline_name = counts.baseline_name
        AND rows.source = counts.source
        AND rows.split = counts.split
    GROUP BY
        rows.baseline_name,
        rows.source,
        rows.split,
        counts.positive_row_count,
        counts.negative_row_count
),
ece_bins AS (
    SELECT
        baseline_name,
        source,
        split,
        LEAST(9, FLOOR(GREATEST(0.0, LEAST(score, 1.0)) * 10)) AS bin_id,
        COUNT(*) AS bin_count,
        AVG(label_int) AS bin_accuracy,
        AVG(score) AS bin_confidence
    FROM rows
    GROUP BY baseline_name, source, split, bin_id
),
ece AS (
    SELECT
        ece_bins.baseline_name,
        ece_bins.source,
        ece_bins.split,
        CASE
            WHEN counts.positive_row_count > 0
                AND counts.negative_row_count > 0
                THEN SUM(
                    ece_bins.bin_count::DOUBLE / counts.row_count
                    * ABS(ece_bins.bin_accuracy - ece_bins.bin_confidence)
                )
            ELSE NULL
        END AS expected_calibration_error_10bin
    FROM ece_bins
    INNER JOIN counts
        ON ece_bins.baseline_name = counts.baseline_name
        AND ece_bins.source = counts.source
        AND ece_bins.split = counts.split
    GROUP BY
        ece_bins.baseline_name,
        ece_bins.source,
        ece_bins.split,
        counts.row_count,
        counts.positive_row_count,
        counts.negative_row_count
)
SELECT
    counts.baseline_name,
    counts.source,
    counts.split,
    counts.row_count,
    counts.positive_row_count,
    counts.negative_row_count,
    average_precision.average_precision,
    roc_auc.roc_auc,
    brier.brier_score,
    ece.expected_calibration_error_10bin
FROM counts
LEFT JOIN average_precision
    ON counts.baseline_name = average_precision.baseline_name
    AND counts.source = average_precision.source
    AND counts.split = average_precision.split
LEFT JOIN roc_auc
    ON counts.baseline_name = roc_auc.baseline_name
    AND counts.source = roc_auc.source
    AND counts.split = roc_auc.split
LEFT JOIN brier
    ON counts.baseline_name = brier.baseline_name
    AND counts.source = brier.source
    AND counts.split = brier.split
LEFT JOIN ece
    ON counts.baseline_name = ece.baseline_name
    AND counts.source = ece.source
    AND counts.split = ece.split
ORDER BY counts.baseline_name, counts.source, counts.split
"""


def ranking_metric_query(
    config: BaselineEvaluationConfig,
    *,
    k: int,
    slice_predicate: str = "TRUE",
) -> str:
    """Return aggregate ranking metrics for one K.

    ``slice_predicate`` restricts the scan to a single
    ``(baseline_name, source, split)`` slice so the window sorts stay bounded.
    """

    scores = config.score_output_path
    return f"""
WITH ranked AS (
    SELECT
        baseline_name,
        source,
        split,
        ranking_group_id,
        candidate_medication_token,
        candidate_rank,
        CASE WHEN label_prescribed THEN 1 ELSE 0 END AS label_int,
        score,
        ROW_NUMBER() OVER (
            PARTITION BY baseline_name, source, split, ranking_group_id
            ORDER BY score DESC, candidate_rank, candidate_medication_token
        ) AS rank_position,
        COUNT(*) OVER (
            PARTITION BY baseline_name, source, split, ranking_group_id
        ) AS group_size,
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END) OVER (
            PARTITION BY baseline_name, source, split, ranking_group_id
        ) AS group_positive_count
    FROM {parquet_scan(scores)}
    WHERE {slice_predicate}
),
group_metrics AS (
    SELECT
        baseline_name,
        source,
        split,
        ranking_group_id,
        MIN(group_size) AS group_size,
        MIN(group_positive_count) AS group_positive_count,
        SUM(CASE
            WHEN rank_position <= {k} AND label_int = 1 THEN 1
            ELSE 0
        END) AS hits_at_k,
        SUM(CASE
            WHEN rank_position <= {k} AND label_int = 1
                THEN 1.0 / LOG(2, rank_position + 1)
            ELSE 0.0
        END) AS dcg_at_k,
        MAX(CASE
            WHEN rank_position <= {k} AND label_int = 1
                THEN 1.0 / rank_position
            ELSE 0.0
        END) AS reciprocal_rank_at_k
    FROM ranked
    GROUP BY baseline_name, source, split, ranking_group_id
),
positive_group_metrics AS (
    SELECT
        *,
        (
            SELECT SUM(1.0 / LOG(2, rank_value + 1))
            FROM range(
                1,
                CAST(LEAST(group_positive_count, {k}) AS BIGINT) + 1
            ) AS ideal(rank_value)
        ) AS ideal_dcg_at_k
    FROM group_metrics
    WHERE group_positive_count > 0
),
group_counts AS (
    SELECT
        baseline_name,
        source,
        split,
        COUNT(*) AS ranking_group_count,
        SUM(CASE WHEN group_positive_count > 0 THEN 1 ELSE 0 END)
            AS positive_ranking_group_count
    FROM group_metrics
    GROUP BY baseline_name, source, split
)
SELECT
    group_counts.baseline_name,
    group_counts.source,
    group_counts.split,
    {k} AS k,
    group_counts.ranking_group_count,
    group_counts.positive_ranking_group_count,
    AVG(
        positive_group_metrics.hits_at_k::DOUBLE
        / LEAST({k}, positive_group_metrics.group_size)
    ) AS precision_at_k,
    AVG(
        positive_group_metrics.hits_at_k::DOUBLE
        / positive_group_metrics.group_positive_count
    ) AS recall_at_k,
    AVG(CASE WHEN positive_group_metrics.hits_at_k > 0 THEN 1.0 ELSE 0.0 END)
        AS hit_rate_at_k,
    AVG(CASE
        WHEN positive_group_metrics.ideal_dcg_at_k > 0
            THEN positive_group_metrics.dcg_at_k
                / positive_group_metrics.ideal_dcg_at_k
        ELSE NULL
    END) AS ndcg_at_k,
    AVG(positive_group_metrics.reciprocal_rank_at_k) AS mrr_at_k
FROM group_counts
LEFT JOIN positive_group_metrics
    ON group_counts.baseline_name = positive_group_metrics.baseline_name
    AND group_counts.source = positive_group_metrics.source
    AND group_counts.split = positive_group_metrics.split
GROUP BY
    group_counts.baseline_name,
    group_counts.source,
    group_counts.split,
    group_counts.ranking_group_count,
    group_counts.positive_ranking_group_count
ORDER BY group_counts.baseline_name, group_counts.source, group_counts.split
"""


def condition_metric_query(
    config: BaselineEvaluationConfig,
    *,
    k: int,
    slice_predicate: str = "TRUE",
) -> str:
    """Return suppressed per-condition ranking metrics for one K.

    ``slice_predicate`` restricts the scan to a single
    ``(baseline_name, source, split)`` slice so the window sorts stay bounded.
    """

    scores = config.score_output_path
    threshold = config.min_subgroup_positive_groups
    return f"""
WITH ranked AS (
    SELECT
        baseline_name,
        source,
        split,
        index_condition_token,
        ranking_group_id,
        candidate_medication_token,
        candidate_rank,
        CASE WHEN label_prescribed THEN 1 ELSE 0 END AS label_int,
        score,
        ROW_NUMBER() OVER (
            PARTITION BY baseline_name, source, split, ranking_group_id
            ORDER BY score DESC, candidate_rank, candidate_medication_token
        ) AS rank_position,
        COUNT(*) OVER (
            PARTITION BY baseline_name, source, split, ranking_group_id
        ) AS group_size,
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END) OVER (
            PARTITION BY baseline_name, source, split, ranking_group_id
        ) AS group_positive_count
    FROM {parquet_scan(scores)}
    WHERE {slice_predicate}
),
group_metrics AS (
    SELECT
        baseline_name,
        source,
        split,
        index_condition_token,
        ranking_group_id,
        MIN(group_size) AS group_size,
        MIN(group_positive_count) AS group_positive_count,
        SUM(CASE
            WHEN rank_position <= {k} AND label_int = 1 THEN 1
            ELSE 0
        END) AS hits_at_k
    FROM ranked
    GROUP BY
        baseline_name,
        source,
        split,
        index_condition_token,
        ranking_group_id
)
SELECT
    baseline_name,
    source,
    split,
    index_condition_token,
    {k} AS k,
    COUNT(*) AS ranking_group_count,
    SUM(CASE WHEN group_positive_count > 0 THEN 1 ELSE 0 END)
        AS positive_ranking_group_count,
    CASE
        WHEN SUM(CASE WHEN group_positive_count > 0 THEN 1 ELSE 0 END)
            >= {threshold}
            THEN AVG(
                CASE
                    WHEN group_positive_count > 0
                        THEN hits_at_k::DOUBLE / LEAST({k}, group_size)
                    ELSE NULL
                END
            )
        ELSE NULL
    END AS precision_at_k,
    CASE
        WHEN SUM(CASE WHEN group_positive_count > 0 THEN 1 ELSE 0 END)
            >= {threshold}
            THEN AVG(
                CASE
                    WHEN group_positive_count > 0
                        THEN hits_at_k::DOUBLE / group_positive_count
                    ELSE NULL
                END
            )
        ELSE NULL
    END AS recall_at_k,
    CASE
        WHEN SUM(CASE WHEN group_positive_count > 0 THEN 1 ELSE 0 END)
            >= {threshold}
            THEN AVG(
                CASE
                    WHEN group_positive_count > 0
                        THEN CASE WHEN hits_at_k > 0 THEN 1.0 ELSE 0.0 END
                    ELSE NULL
                END
            )
        ELSE NULL
    END AS hit_rate_at_k,
    CASE
        WHEN SUM(CASE WHEN group_positive_count > 0 THEN 1 ELSE 0 END)
            >= {threshold}
            THEN 'reported'
        ELSE 'suppressed_low_positive_group_count'
    END AS reporting_status
FROM group_metrics
GROUP BY baseline_name, source, split, index_condition_token
ORDER BY baseline_name, source, split, index_condition_token
"""


def append_metric_summaries(
    connection: duckdb.DuckDBPyConnection,
    config: BaselineEvaluationConfig,
    manifest: dict[str, Any],
) -> None:
    """Attach aggregate metric summaries to the evaluation manifest.

    Metrics are computed one ``(baseline_name, source, split)`` slice at a time.
    Each slice query only sorts a single window partition, so peak DuckDB memory
    and temp spill stay bounded even when the combined score table holds tens of
    millions of rows in final mode. The small per-slice result rows are unioned
    in Python.
    """

    row_level_metrics: list[dict[str, Any]] = []
    ranking_metrics: list[dict[str, Any]] = []
    per_condition_metrics: list[dict[str, Any]] = []
    for slice_row in metric_slices(connection, config):
        predicate = metric_slice_predicate(slice_row)
        row_level_metrics.extend(
            fetch_dict_rows(
                connection,
                row_level_metric_query(config, slice_predicate=predicate),
            ),
        )
        for k in config.top_k:
            ranking_metrics.extend(
                fetch_dict_rows(
                    connection,
                    ranking_metric_query(config, k=k, slice_predicate=predicate),
                ),
            )
            per_condition_metrics.extend(
                fetch_dict_rows(
                    connection,
                    condition_metric_query(config, k=k, slice_predicate=predicate),
                ),
            )
    manifest["row_level_metrics"] = row_level_metrics
    manifest["ranking_metrics"] = ranking_metrics
    manifest["per_condition_metrics"] = per_condition_metrics


def build_baseline_evaluation(
    config: BaselineEvaluationConfig = BaselineEvaluationConfig(),
) -> dict[str, Any]:
    """Build Milestone 7 coverage and non-learned baseline evaluation reports."""

    generated_at = datetime.now(UTC).isoformat()
    config.evaluation_root.mkdir(parents=True, exist_ok=True)
    missing = missing_input_tables(config)
    if missing:
        manifest = base_manifest(
            config,
            status="failed_missing_inputs",
            generated_at=generated_at,
        )
        manifest["missing_inputs"] = missing
        write_json(config.evaluation_report_path, manifest)
        return manifest

    try:
        config = resolve_feature_version(config)
    except ValueError as error:
        manifest = base_manifest(
            config,
            status="failed_feature_version_mismatch",
            generated_at=generated_at,
        )
        manifest["reason"] = safe_error_message(error)
        write_json(config.evaluation_report_path, manifest)
        return manifest

    with duckdb.connect(database=":memory:") as connection:
        configure_connection(config, connection)
        coverage_report = build_coverage_report(
            connection,
            config,
            generated_at=generated_at,
        )
        write_json(config.coverage_report_path, coverage_report)

        if config.mode == "final" and not config.frozen_selection:
            manifest = base_manifest(
                config,
                status="blocked_final_requires_frozen_selection",
                generated_at=generated_at,
            )
            manifest["coverage_summary"] = coverage_report["source_split_coverage"]
            manifest["blocker"] = (
                "Final/test evaluation requires --frozen-selection after "
                "validation-based choices are locked."
            )
            write_json(config.evaluation_report_path, manifest)
            return manifest

        manifest = base_manifest(config, status="completed", generated_at=generated_at)
        manifest["coverage_summary"] = coverage_report["source_split_coverage"]
        record, learned_section = score_artifact_record(
            connection,
            config,
            generated_at=generated_at,
        )
        if learned_section is not None:
            manifest["learned_baselines"] = learned_section
        manifest["artifacts"]["baseline_scores"] = (
            str(config.score_output_path) if record["status"] == "completed" else None
        )
        manifest["tables"] = record.get("tables", [record])
        if record["status"] != "completed":
            manifest["status"] = "failed"
        else:
            append_metric_summaries(connection, config, manifest)

    write_json(config.evaluation_report_path, manifest)
    return manifest


def group_ranking_metrics(
    labels: Sequence[bool | int],
    scores: Sequence[float],
    *,
    top_k: Iterable[int] = DEFAULT_TOP_K,
) -> dict[int, dict[str, float]]:
    """Compute ranking metrics for one positive group.

    This helper is used by synthetic tests. The CLI computes aggregate metrics
    in DuckDB so protected-data runs do not need to load score tables into
    Python memory.
    """

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    rows = sorted(
        zip(labels, scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    positives = sum(1 for label, _score in rows if bool(label))
    metrics: dict[int, dict[str, float]] = {}
    for k in top_k:
        window = rows[: min(k, len(rows))]
        hits = sum(1 for label, _score in window if bool(label))
        denominator = min(k, len(rows))
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, (label, _score) in enumerate(window, start=1)
            if bool(label)
        )
        ideal_hits = min(positives, k, len(rows))
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        first_positive_rank = next(
            (
                rank
                for rank, (label, _score) in enumerate(window, start=1)
                if bool(label)
            ),
            None,
        )
        metrics[int(k)] = {
            "precision_at_k": hits / denominator if denominator else 0.0,
            "recall_at_k": hits / positives if positives else 0.0,
            "hit_rate_at_k": 1.0 if hits else 0.0,
            "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else 0.0,
            "mrr_at_k": 1.0 / first_positive_rank
            if first_positive_rank is not None
            else 0.0,
        }
    return metrics


def row_level_classification_metrics(
    labels: Sequence[bool | int],
    scores: Sequence[float],
) -> dict[str, float | None]:
    """Compute row-level AP, ROC-AUC, Brier, and ECE for synthetic tests."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    label_array = np.array([int(bool(label)) for label in labels], dtype=float)
    score_array = np.array(scores, dtype=float)
    positive_count = int(label_array.sum())
    negative_count = int(len(label_array) - positive_count)
    if positive_count == 0 or negative_count == 0:
        return {
            "average_precision": None,
            "roc_auc": None,
            "brier_score": None,
            "expected_calibration_error_10bin": None,
        }

    order = np.argsort(-score_array, kind="mergesort")
    sorted_labels = label_array[order]
    cumulative_positives = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(sorted_labels) + 1)
    precision_terms = cumulative_positives[ranks - 1] / ranks
    average_precision = float(
        np.sum(precision_terms[sorted_labels == 1]) / positive_count
    )

    sorted_asc = np.argsort(score_array, kind="mergesort")
    ranks_asc = np.empty(len(score_array), dtype=float)
    ranks_asc[sorted_asc] = np.arange(1, len(score_array) + 1)
    positive_rank_sum = float(np.sum(ranks_asc[label_array == 1]))
    roc_auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )

    brier_score = float(np.mean((score_array - label_array) ** 2))
    bin_ids = np.minimum(9, np.floor(np.clip(score_array, 0.0, 1.0) * 10).astype(int))
    ece = 0.0
    for bin_id in range(10):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        bin_accuracy = float(np.mean(label_array[mask]))
        bin_confidence = float(np.mean(score_array[mask]))
        ece += mask.mean() * abs(bin_accuracy - bin_confidence)

    return {
        "average_precision": average_precision,
        "roc_auc": float(roc_auc),
        "brier_score": brier_score,
        "expected_calibration_error_10bin": float(ece),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Milestone 7 transparent medication-ranking baselines.",
    )
    parser.add_argument(
        "--features-root",
        type=Path,
        default=FEATURES_ROOT,
        help="Root directory containing Milestone 6 feature artifacts.",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=TRAINING_ROOT,
        help="Root directory containing Milestone 6 training artifacts.",
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=MILESTONE7_EVALUATION_ROOT,
        help="Output directory for local ignored evaluation artifacts.",
    )
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=DEFAULT_COVERAGE_REPORT_PATH,
        help="Output path for the aggregate coverage report.",
    )
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=DEFAULT_EVALUATION_REPORT_PATH,
        help="Output path for the aggregate evaluation report.",
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=REPORTS_ROOT / "training_table_manifest.json",
        help="Milestone 6 aggregate training-table manifest path.",
    )
    parser.add_argument(
        "--top-k",
        default="1,3,5,10",
        help="Comma-separated ranking cutoffs.",
    )
    parser.add_argument(
        "--mode",
        choices=("development", "final"),
        default="development",
        help="Development evaluates train diagnostics and validation; final adds held-out splits.",
    )
    parser.add_argument(
        "--frozen-selection",
        action="store_true",
        help="Required with --mode final before held-out test metrics are produced.",
    )
    parser.add_argument(
        "--condition-token",
        action="append",
        default=[],
        help="Optional condition token filter; may be repeated or comma-separated.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        choices=ALL_BASELINES,
        help="Optional baseline to run; may be repeated.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Seed used for deterministic random baseline scores.",
    )
    parser.add_argument(
        "--feature-version",
        default=None,
        help=(
            "Optional expected feature version. By default it is inferred from "
            "the input artifacts."
        ),
    )
    parser.add_argument(
        "--duckdb-temp-dir",
        type=Path,
        default=DUCKDB_TEMP_DIR,
        help="Directory DuckDB may use to spill larger-than-memory operators.",
    )
    parser.add_argument(
        "--duckdb-memory-limit",
        default=DUCKDB_MEMORY_LIMIT,
        help="Optional DuckDB memory ceiling, e.g. '24GB'.",
    )
    parser.add_argument(
        "--duckdb-threads",
        type=int,
        default=DUCKDB_THREADS,
        help="Optional DuckDB thread cap.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        top_k = parse_top_k(args.top_k)
        baselines = tuple(args.baseline) if args.baseline else NON_LEARNED_BASELINES
        manifest = build_baseline_evaluation(
            BaselineEvaluationConfig(
                features_root=args.features_root,
                training_root=args.training_root,
                evaluation_root=args.evaluation_root,
                coverage_report_path=args.coverage_report,
                evaluation_report_path=args.evaluation_report,
                training_manifest_path=args.training_manifest,
                top_k=top_k,
                mode=args.mode,
                frozen_selection=args.frozen_selection,
                seed=args.seed,
                condition_tokens=parse_repeated_csv(args.condition_token),
                baselines=baselines,
                feature_version=args.feature_version,
                duckdb_temp_directory=args.duckdb_temp_dir,
                duckdb_memory_limit=args.duckdb_memory_limit,
                duckdb_threads=args.duckdb_threads,
            ),
        )
    except ValueError as error:
        print(f"Invalid Milestone 7 evaluation arguments: {error}")
        return 2

    print(
        "Wrote Milestone 7 baseline evaluation report: "
        f"status={manifest['status']}, "
        f"report={args.evaluation_report}"
    )
    if manifest["status"] in {
        "failed_missing_inputs",
        "blocked_final_requires_frozen_selection",
    }:
        return 2
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
