"""Evaluate Milestone 8B graph-aware ranking ablations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.compose import ColumnTransformer

from pipeline.config import (
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_TEMP_DIR,
    DUCKDB_THREADS,
    FEATURE_VERSION,
    FEATURES_ROOT,
    GRAPH_ABLATION_VERSION,
    GRAPH_VERSION,
    LABEL_VERSION,
    MILESTONE7_EVALUATION_ROOT,
    MILESTONE8_GRAPH_ROOT,
    MILESTONE8B_EVALUATION_ROOT,
    MILESTONE8B_REPORT_VERSION,
    RANDOM_SEED,
    REPORTS_ROOT,
    SPLIT_VERSION,
    TRAINING_ROOT,
)
from pipeline.evaluate_baselines import (
    DEFAULT_TOP_K,
    add_evaluability_status,
    append_metric_summaries,
    coverage_summary_query,
    load_json_if_present,
    parse_repeated_csv,
    parse_top_k,
    write_json,
)
from pipeline.extract_utils import (
    configure_duckdb_connection,
    parquet_scan,
    safe_error_message,
    sql_string,
)
from pipeline.features import copy_query_to_parquet, fetch_dict_rows
from pipeline.io_utils import quote_identifier
from pipeline.learned_baselines import (
    NEGATIVE_TO_POSITIVE_RATIO,
    SCORING_BATCH_COUNT,
    XGBOOST_V1_HYPERPARAMETERS,
    LearnedFeatureSpec,
    fit_preprocessor,
    fit_xgboost_model,
    predict_scores,
    resolve_feature_spec,
    write_training_sample,
)


SCHEMA_VERSION = MILESTONE8B_REPORT_VERSION
FEATURE_MANIFEST_SCHEMA_VERSION = "milestone8b-graph-feature-manifest-v1"
FROZEN_SELECTION_SCHEMA_VERSION = "milestone8b-frozen-selection-v1"
DEFAULT_FEATURE_MANIFEST_PATH = REPORTS_ROOT / "milestone8b_graph_feature_manifest.json"
DEFAULT_EVALUATION_REPORT_PATH = REPORTS_ROOT / "milestone8b_ablation_evaluation.json"
DEFAULT_FROZEN_SELECTION_PATH = REPORTS_ROOT / "milestone8b_frozen_selection.json"
GRAPH_SELECTION_DELTA = 0.005
PRIMARY_METRIC_MAX_DROP = 0.01
FUSION_WEIGHT_GRID = tuple(round(index / 20, 2) for index in range(21))

GRAPH_NUMERIC_FEATURE_COLUMNS = (
    "graph_condition_medication_support_count",
    "graph_condition_medication_log_support",
    "graph_condition_medication_support_share",
    "graph_condition_total_medication_support",
    "graph_condition_medication_degree",
    "graph_condition_lab_degree",
    "graph_condition_vital_degree",
    "graph_condition_intervention_degree",
    "graph_condition_total_degree",
    "graph_condition_total_support",
    "graph_candidate_medication_degree",
    "graph_candidate_medication_support",
    "graph_candidate_coprescription_degree",
    "graph_candidate_coprescription_support",
    "graph_condition_in_graph",
    "graph_candidate_in_graph",
    "graph_direct_edge_present",
)

SCORE_OUTPUT_COLUMNS = (
    "source",
    "split",
    "ranking_group_id",
    "index_condition_token",
    "candidate_medication_token",
    "candidate_rank",
    "label_prescribed",
    "baseline_name",
    "score",
    "seed",
    "baseline_version",
    "evaluation_version",
    "generated_at",
)


@dataclass(frozen=True)
class GraphAblationConfig:
    """Configuration for Milestone 8B graph-aware ablation evaluation."""

    features_root: Path = FEATURES_ROOT
    training_root: Path = TRAINING_ROOT
    graph_root: Path = MILESTONE8_GRAPH_ROOT
    milestone7_evaluation_root: Path = MILESTONE7_EVALUATION_ROOT
    evaluation_root: Path = MILESTONE8B_EVALUATION_ROOT
    feature_manifest_path: Path = DEFAULT_FEATURE_MANIFEST_PATH
    evaluation_report_path: Path = DEFAULT_EVALUATION_REPORT_PATH
    frozen_selection_path: Path = DEFAULT_FROZEN_SELECTION_PATH
    milestone6_feature_manifest_path: Path = (
        REPORTS_ROOT / "milestone6_feature_manifest.json"
    )
    training_manifest_path: Path = REPORTS_ROOT / "training_table_manifest.json"
    milestone7_evaluation_report_path: Path = (
        REPORTS_ROOT / "milestone7_baseline_evaluation.json"
    )
    milestone8_suitability_report_path: Path = (
        REPORTS_ROOT / "milestone8_graph_suitability.json"
    )
    top_k: tuple[int, ...] = DEFAULT_TOP_K
    mode: str = "development"
    frozen_selection: bool = False
    seed: int = RANDOM_SEED
    condition_tokens: tuple[str, ...] = ()
    graph_ablation_version: str = GRAPH_ABLATION_VERSION
    report_version: str = MILESTONE8B_REPORT_VERSION
    graph_version: str = GRAPH_VERSION
    feature_version: str = FEATURE_VERSION
    label_version: str = LABEL_VERSION
    split_version: str = SPLIT_VERSION
    min_subgroup_positive_groups: int = 25
    allow_development_milestone7_reference: bool = False
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
    def graph_edges_path(self) -> Path:
        return self.graph_root / "graph_edges.parquet"

    @property
    def milestone7_scores_path(self) -> Path:
        return self.milestone7_evaluation_root / "baseline_scores.parquet"

    @property
    def graph_feature_matrix_path(self) -> Path:
        return self.evaluation_root / "graph_feature_matrix.parquet"

    @property
    def graph_only_training_sample_path(self) -> Path:
        return self.evaluation_root / "graph_only_training_sample.parquet"

    @property
    def graph_augmented_training_sample_path(self) -> Path:
        return self.evaluation_root / "graph_augmented_training_sample.parquet"

    @property
    def reference_scores_path(self) -> Path:
        return self.evaluation_root / "_scores_reference.parquet"

    @property
    def model_scores_path(self) -> Path:
        return self.evaluation_root / "_scores_graph_models.parquet"

    @property
    def fusion_scores_path(self) -> Path:
        return self.evaluation_root / "_scores_fusion.parquet"

    @property
    def score_output_path(self) -> Path:
        return self.evaluation_root / "graph_ablation_scores.parquet"

    @property
    def models_root(self) -> Path:
        return self.evaluation_root / "models"

    @property
    def graph_only_model_path(self) -> Path:
        return self.models_root / "graph_only_xgboost_model.json"

    @property
    def graph_augmented_model_path(self) -> Path:
        return self.models_root / "graph_augmented_xgboost_model.json"

    @property
    def graph_feature_preprocessor_path(self) -> Path:
        return self.models_root / "graph_feature_preprocessor.joblib"

    @property
    def fusion_weights_path(self) -> Path:
        return self.evaluation_root / "fusion_weights.json"

    @property
    def selection_k(self) -> int:
        return 10 if 10 in self.top_k else max(self.top_k)


@dataclass(frozen=True)
class FittedExperiment:
    """Fitted graph ablation model and preprocessing metadata."""

    experiment_name: str
    feature_spec: LearnedFeatureSpec
    preprocessor: ColumnTransformer
    model: xgb.Booster
    training_counts: dict[str, int]
    training_sample_path: Path
    model_path: Path
    include_stay_features: bool


def configure_connection(
    config: GraphAblationConfig,
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Apply shared memory-safe DuckDB settings."""

    configure_duckdb_connection(
        connection,
        temp_directory=config.duckdb_temp_directory,
        memory_limit=config.duckdb_memory_limit,
        threads=config.duckdb_threads,
    )


def missing_input_tables(config: GraphAblationConfig) -> list[dict[str, str]]:
    """Return missing inputs required for Milestone 8B ablations."""

    required = (
        ("patient_stay_features", config.patient_stay_features_path),
        ("patient_condition_medication", config.patient_condition_medication_path),
        ("candidate_catalog", config.candidate_catalog_path),
        ("graph_edges", config.graph_edges_path),
        ("milestone7_baseline_scores", config.milestone7_scores_path),
        ("milestone6_feature_manifest", config.milestone6_feature_manifest_path),
        ("training_table_manifest", config.training_manifest_path),
        ("milestone7_baseline_evaluation", config.milestone7_evaluation_report_path),
        ("milestone8_graph_suitability", config.milestone8_suitability_report_path),
    )
    return [
        {"table_name": table_name, "path": str(path)}
        for table_name, path in required
        if not path.exists()
    ]


def condition_filter_sql(config: GraphAblationConfig, alias: str = "gf") -> str:
    """Return an optional condition-token filter predicate."""

    if not config.condition_tokens:
        return "TRUE"
    tokens = ", ".join(sql_string(token) for token in config.condition_tokens)
    return f"{alias}.index_condition_token IN ({tokens})"


def scoring_scope_sql(config: GraphAblationConfig, alias: str = "gf") -> str:
    """Return the source/split scope used for Milestone 8B scoring."""

    mimic_splits = "'train', 'validation'"
    if config.mode == "final":
        mimic_splits = "'train', 'validation', 'test'"
    return (
        f"(({alias}.source = 'mimiciv' "
        f"AND {alias}.split IN ({mimic_splits})) "
        f"OR {alias}.source = 'eicu_crd')"
    )


def column_ref(alias: str, column_name: str) -> str:
    """Return a quoted SQL column reference."""

    return f"{alias}.{quote_identifier(column_name)}"


def feature_select_sql(
    feature_spec: LearnedFeatureSpec,
    *,
    row_alias: str,
    stay_alias: str,
) -> str:
    """Return feature select expressions for a fitted feature spec."""

    selects: list[str] = []
    for column_name in (*feature_spec.stay_numeric, *feature_spec.stay_categorical):
        selects.append(
            f"{column_ref(stay_alias, column_name)} AS {quote_identifier(column_name)}"
        )
    for column_name in (*feature_spec.row_numeric, *feature_spec.row_categorical):
        selects.append(
            f"{column_ref(row_alias, column_name)} AS {quote_identifier(column_name)}"
        )
    return ",\n    ".join(selects)


def graph_feature_query(config: GraphAblationConfig) -> str:
    """Return SQL that joins train-fit graph features onto candidate rows."""

    pcm = config.patient_condition_medication_path
    edges = config.graph_edges_path
    scope = scoring_scope_sql(config, alias="pcm")
    condition_filter = condition_filter_sql(config, alias="pcm")
    return f"""
WITH graph_edges AS (
    SELECT
        src_id,
        dst_id,
        src_type,
        dst_type,
        relation_type,
        support_count
    FROM {parquet_scan(edges)}
    WHERE fit_source = 'mimiciv'
        AND fit_split = 'train'
),
condition_medication_edges AS (
    SELECT
        REPLACE(src_id, 'condition|', '') AS index_condition_token,
        REPLACE(dst_id, 'medication|', '') AS candidate_medication_token,
        support_count
    FROM graph_edges
    WHERE relation_type = 'condition_medication_train_positive'
),
condition_medication_totals AS (
    SELECT
        index_condition_token,
        SUM(support_count) AS total_support,
        COUNT(*) AS medication_degree
    FROM condition_medication_edges
    GROUP BY index_condition_token
),
condition_relation_summary AS (
    SELECT
        src_id,
        SUM(CASE WHEN dst_type = 'lab' THEN 1 ELSE 0 END) AS lab_degree,
        SUM(CASE WHEN dst_type = 'vital' THEN 1 ELSE 0 END) AS vital_degree,
        SUM(CASE WHEN dst_type = 'intervention' THEN 1 ELSE 0 END)
            AS intervention_degree,
        COUNT(*) AS total_degree,
        SUM(support_count) AS total_support
    FROM graph_edges
    WHERE src_type = 'condition'
    GROUP BY src_id
),
medication_relation_summary AS (
    SELECT
        medication_id,
        COUNT(*) AS medication_degree,
        SUM(support_count) AS medication_support,
        SUM(CASE WHEN relation_type = 'medication_medication_train_coprescribed'
            THEN 1 ELSE 0 END) AS coprescription_degree,
        SUM(CASE WHEN relation_type = 'medication_medication_train_coprescribed'
            THEN support_count ELSE 0 END) AS coprescription_support
    FROM (
        SELECT src_id AS medication_id, relation_type, support_count
        FROM graph_edges
        WHERE src_type = 'medication'
        UNION ALL
        SELECT dst_id AS medication_id, relation_type, support_count
        FROM graph_edges
        WHERE dst_type = 'medication'
    )
    GROUP BY medication_id
),
graph_nodes AS (
    SELECT src_id AS node_id FROM graph_edges
    UNION
    SELECT dst_id AS node_id FROM graph_edges
),
candidate_rows AS (
    SELECT
        pcm.source,
        pcm.split,
        pcm.stay_uid,
        pcm.ranking_group_id,
        pcm.index_condition_token,
        pcm.candidate_medication_token,
        pcm.candidate_rank,
        pcm.label_prescribed,
        'condition|' || pcm.index_condition_token AS condition_node_id,
        'medication|' || pcm.candidate_medication_token AS medication_node_id
    FROM {parquet_scan(pcm)} AS pcm
    WHERE {scope}
        AND {condition_filter}
)
SELECT
    rows.source,
    rows.split,
    rows.stay_uid,
    rows.ranking_group_id,
    rows.index_condition_token,
    rows.candidate_medication_token,
    rows.candidate_rank,
    rows.label_prescribed,
    COALESCE(direct.support_count, 0) AS graph_condition_medication_support_count,
    LN(1 + COALESCE(direct.support_count, 0))
        AS graph_condition_medication_log_support,
    CASE
        WHEN condition_med_totals.total_support > 0
            THEN COALESCE(direct.support_count, 0)::DOUBLE
                / condition_med_totals.total_support
        ELSE 0.0
    END AS graph_condition_medication_support_share,
    COALESCE(condition_med_totals.total_support, 0)
        AS graph_condition_total_medication_support,
    COALESCE(condition_med_totals.medication_degree, 0)
        AS graph_condition_medication_degree,
    COALESCE(condition_summary.lab_degree, 0) AS graph_condition_lab_degree,
    COALESCE(condition_summary.vital_degree, 0) AS graph_condition_vital_degree,
    COALESCE(condition_summary.intervention_degree, 0)
        AS graph_condition_intervention_degree,
    COALESCE(condition_summary.total_degree, 0) AS graph_condition_total_degree,
    COALESCE(condition_summary.total_support, 0) AS graph_condition_total_support,
    COALESCE(medication_summary.medication_degree, 0)
        AS graph_candidate_medication_degree,
    COALESCE(medication_summary.medication_support, 0)
        AS graph_candidate_medication_support,
    COALESCE(medication_summary.coprescription_degree, 0)
        AS graph_candidate_coprescription_degree,
    COALESCE(medication_summary.coprescription_support, 0)
        AS graph_candidate_coprescription_support,
    CASE WHEN condition_nodes.node_id IS NULL THEN 0 ELSE 1 END
        AS graph_condition_in_graph,
    CASE WHEN medication_nodes.node_id IS NULL THEN 0 ELSE 1 END
        AS graph_candidate_in_graph,
    CASE WHEN direct.support_count IS NULL THEN 0 ELSE 1 END
        AS graph_direct_edge_present
FROM candidate_rows AS rows
LEFT JOIN condition_medication_edges AS direct
    ON rows.index_condition_token = direct.index_condition_token
    AND rows.candidate_medication_token = direct.candidate_medication_token
LEFT JOIN condition_medication_totals AS condition_med_totals
    ON rows.index_condition_token = condition_med_totals.index_condition_token
LEFT JOIN condition_relation_summary AS condition_summary
    ON rows.condition_node_id = condition_summary.src_id
LEFT JOIN medication_relation_summary AS medication_summary
    ON rows.medication_node_id = medication_summary.medication_id
LEFT JOIN graph_nodes AS condition_nodes
    ON rows.condition_node_id = condition_nodes.node_id
LEFT JOIN graph_nodes AS medication_nodes
    ON rows.medication_node_id = medication_nodes.node_id
"""


def graph_feature_summary_query(config: GraphAblationConfig) -> str:
    """Return aggregate graph-feature coverage summaries."""

    features = config.graph_feature_matrix_path
    feature_columns = ",\n        ".join(
        f"AVG(CASE WHEN {quote_identifier(column)} IS NULL THEN 1.0 ELSE 0.0 END) "
        f"AS {quote_identifier(column + '_null_rate')}"
        for column in GRAPH_NUMERIC_FEATURE_COLUMNS
    )
    return f"""
SELECT
    source,
    split,
    COUNT(*) AS candidate_row_count,
    SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END) AS positive_row_count,
    AVG(graph_condition_in_graph) AS condition_in_graph_rate,
    AVG(graph_candidate_in_graph) AS candidate_in_graph_rate,
    AVG(graph_direct_edge_present) AS direct_edge_present_rate,
    {feature_columns}
FROM {parquet_scan(features)}
GROUP BY source, split
ORDER BY source, split
"""


def training_sample_query(
    config: GraphAblationConfig,
    feature_spec: LearnedFeatureSpec,
    *,
    include_stay_features: bool,
) -> str:
    """Build a deterministic train-only sample for graph ablation models."""

    feature_sql = feature_select_sql(
        feature_spec,
        row_alias="gf",
        stay_alias="psf",
    )
    join_sql = ""
    if include_stay_features:
        join_sql = f"""
INNER JOIN {parquet_scan(config.patient_stay_features_path)} AS psf
    ON gf.stay_uid = psf.stay_uid
    AND gf.source = psf.source
"""
    else:
        join_sql = "CROSS JOIN (SELECT 1 AS no_stay_features) AS psf"
    hash_uniform = (
        "CAST("
        "HASH("
        f"{sql_string(str(config.seed))} || '|' || train_rows.ranking_group_id "
        "|| '|' || train_rows.candidate_medication_token"
        ") AS DOUBLE"
        ") / 18446744073709551615.0"
    )
    return f"""
WITH train_rows AS (
    SELECT *
    FROM {parquet_scan(config.graph_feature_matrix_path)} AS gf
    WHERE gf.source = 'mimiciv'
        AND gf.split = 'train'
        AND {condition_filter_sql(config, alias="gf")}
),
condition_counts AS (
    SELECT
        index_condition_token,
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END)
            AS positive_row_count,
        SUM(CASE WHEN NOT label_prescribed THEN 1 ELSE 0 END)
            AS negative_row_count
    FROM train_rows
    GROUP BY index_condition_token
),
selected_rows AS (
    SELECT train_rows.*
    FROM train_rows
    INNER JOIN condition_counts
        ON train_rows.index_condition_token = condition_counts.index_condition_token
    WHERE train_rows.label_prescribed
        OR (
            NOT train_rows.label_prescribed
            AND condition_counts.positive_row_count > 0
            AND {hash_uniform} < LEAST(
                1.0,
                (
                    condition_counts.positive_row_count
                    * {NEGATIVE_TO_POSITIVE_RATIO}
                )::DOUBLE
                / NULLIF(condition_counts.negative_row_count, 0)
            )
        )
)
SELECT
    gf.source,
    gf.split,
    gf.ranking_group_id,
    gf.index_condition_token,
    gf.candidate_medication_token,
    gf.candidate_rank,
    gf.label_prescribed,
    {feature_sql}
FROM selected_rows AS gf
{join_sql}
"""


def scoring_rows_query(
    config: GraphAblationConfig,
    feature_spec: LearnedFeatureSpec,
    *,
    include_stay_features: bool,
    batch_id: int,
    batch_count: int,
) -> str:
    """Return one graph-ablation scoring batch."""

    feature_sql = feature_select_sql(
        feature_spec,
        row_alias="gf",
        stay_alias="psf",
    )
    join_sql = ""
    if include_stay_features:
        join_sql = f"""
INNER JOIN {parquet_scan(config.patient_stay_features_path)} AS psf
    ON gf.stay_uid = psf.stay_uid
    AND gf.source = psf.source
"""
    else:
        join_sql = "CROSS JOIN (SELECT 1 AS no_stay_features) AS psf"
    return f"""
SELECT
    gf.source,
    gf.split,
    gf.ranking_group_id,
    gf.index_condition_token,
    gf.candidate_medication_token,
    gf.candidate_rank,
    gf.label_prescribed,
    {feature_sql}
FROM {parquet_scan(config.graph_feature_matrix_path)} AS gf
{join_sql}
WHERE {scoring_scope_sql(config, alias="gf")}
    AND {condition_filter_sql(config, alias="gf")}
    AND ABS(HASH(gf.ranking_group_id)) % {batch_count} = {batch_id}
"""


def reference_scores_query(config: GraphAblationConfig, *, generated_at: str) -> str:
    """Return frozen Milestone 7 XGBoost scores aligned to 8B graph rows."""

    scope = scoring_scope_sql(config, alias="scores")
    condition_filter = condition_filter_sql(config, alias="scores")
    return f"""
SELECT
    scores.source,
    scores.split,
    scores.ranking_group_id,
    scores.index_condition_token,
    scores.candidate_medication_token,
    scores.candidate_rank,
    scores.label_prescribed,
    'xgboost_frozen_reference' AS baseline_name,
    scores.score,
    {config.seed} AS seed,
    {sql_string(config.graph_ablation_version)} AS baseline_version,
    {sql_string(config.report_version)} AS evaluation_version,
    {sql_string(generated_at)} AS generated_at
FROM {parquet_scan(config.milestone7_scores_path)} AS scores
INNER JOIN {parquet_scan(config.graph_feature_matrix_path)} AS gf
    ON scores.source = gf.source
    AND scores.split = gf.split
    AND scores.ranking_group_id = gf.ranking_group_id
    AND scores.index_condition_token = gf.index_condition_token
    AND scores.candidate_medication_token = gf.candidate_medication_token
WHERE scores.baseline_name = 'xgboost'
    AND {scope}
    AND {condition_filter}
"""


def score_union_sql(score_paths: Sequence[Path]) -> str:
    """Return a SQL union over score parquet artifacts."""

    return "\nUNION ALL\n".join(
        f"SELECT * FROM {parquet_scan(path)}" for path in score_paths
    )


def fusion_validation_metric_query(
    *,
    score_sql: str,
    graph_weight: float,
    k: int,
) -> str:
    """Return validation ranking metrics for one fusion weight."""

    weight = f"{graph_weight:.6f}"
    return f"""
WITH all_scores AS (
{score_sql}
),
paired AS (
    SELECT
        ref.source,
        ref.split,
        ref.ranking_group_id,
        ref.candidate_medication_token,
        ref.candidate_rank,
        ref.label_prescribed,
        ((1.0 - {weight}) * ref.score + {weight} * graph.score) AS score
    FROM all_scores AS ref
    INNER JOIN all_scores AS graph
        ON ref.source = graph.source
        AND ref.split = graph.split
        AND ref.ranking_group_id = graph.ranking_group_id
        AND ref.candidate_medication_token = graph.candidate_medication_token
    WHERE ref.baseline_name = 'xgboost_frozen_reference'
        AND graph.baseline_name = 'graph_only_xgboost'
        AND ref.source = 'mimiciv'
        AND ref.split = 'validation'
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY ranking_group_id
            ORDER BY score DESC, candidate_rank, candidate_medication_token
        ) AS rank_position,
        COUNT(*) OVER (PARTITION BY ranking_group_id) AS group_size,
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END) OVER (
            PARTITION BY ranking_group_id
        ) AS group_positive_count
    FROM paired
),
group_metrics AS (
    SELECT
        ranking_group_id,
        MIN(group_size) AS group_size,
        MIN(group_positive_count) AS group_positive_count,
        SUM(CASE
            WHEN rank_position <= {k} AND label_prescribed THEN 1
            ELSE 0
        END) AS hits_at_k,
        SUM(CASE
            WHEN rank_position <= {k} AND label_prescribed
                THEN 1.0 / LOG(2, rank_position + 1)
            ELSE 0.0
        END) AS dcg_at_k,
        MAX(CASE
            WHEN rank_position <= {k} AND label_prescribed
                THEN 1.0 / rank_position
            ELSE 0.0
        END) AS reciprocal_rank_at_k
    FROM ranked
    GROUP BY ranking_group_id
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
)
SELECT
    {weight}::DOUBLE AS graph_weight,
    COUNT(*) AS positive_ranking_group_count,
    AVG(CASE WHEN hits_at_k > 0 THEN 1.0 ELSE 0.0 END) AS hit_rate_at_k,
    AVG(CASE
        WHEN ideal_dcg_at_k > 0 THEN dcg_at_k / ideal_dcg_at_k
        ELSE NULL
    END) AS ndcg_at_k,
    AVG(reciprocal_rank_at_k) AS mrr_at_k
FROM positive_group_metrics
"""


def fusion_scores_query(
    config: GraphAblationConfig,
    *,
    reference_path: Path,
    model_scores_path: Path,
    graph_weight: float,
    generated_at: str,
) -> str:
    """Return late-fusion and simple-ensemble score rows."""

    weighted = f"{graph_weight:.12f}"
    score_sql = score_union_sql((reference_path, model_scores_path))
    return f"""
WITH all_scores AS (
{score_sql}
),
paired AS (
    SELECT
        ref.source,
        ref.split,
        ref.ranking_group_id,
        ref.index_condition_token,
        ref.candidate_medication_token,
        ref.candidate_rank,
        ref.label_prescribed,
        ref.score AS reference_score,
        graph.score AS graph_score
    FROM all_scores AS ref
    INNER JOIN all_scores AS graph
        ON ref.source = graph.source
        AND ref.split = graph.split
        AND ref.ranking_group_id = graph.ranking_group_id
        AND ref.index_condition_token = graph.index_condition_token
        AND ref.candidate_medication_token = graph.candidate_medication_token
    WHERE ref.baseline_name = 'xgboost_frozen_reference'
        AND graph.baseline_name = 'graph_only_xgboost'
),
fusion_rows AS (
    SELECT
        source,
        split,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        candidate_rank,
        label_prescribed,
        'late_fusion_validation_weighted' AS baseline_name,
        ((1.0 - {weighted}) * reference_score + {weighted} * graph_score) AS score
    FROM paired
    UNION ALL
    SELECT
        source,
        split,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        candidate_rank,
        label_prescribed,
        'simple_ensemble_mean' AS baseline_name,
        0.5 * reference_score + 0.5 * graph_score AS score
    FROM paired
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
    {sql_string(config.graph_ablation_version)} AS baseline_version,
    {sql_string(config.report_version)} AS evaluation_version,
    {sql_string(generated_at)} AS generated_at
FROM fusion_rows
"""


def graph_only_feature_spec() -> LearnedFeatureSpec:
    """Return the graph-only feature spec."""

    return LearnedFeatureSpec(
        stay_numeric=(),
        stay_categorical=(),
        row_numeric=GRAPH_NUMERIC_FEATURE_COLUMNS,
        row_categorical=(),
    )


def graph_augmented_feature_spec(
    connection: duckdb.DuckDBPyConnection,
    config: GraphAblationConfig,
) -> LearnedFeatureSpec:
    """Return the Milestone 7 feature family augmented with graph features."""

    base = resolve_feature_spec(connection, config.patient_stay_features_path)
    return LearnedFeatureSpec(
        stay_numeric=base.stay_numeric,
        stay_categorical=base.stay_categorical,
        row_numeric=(*base.row_numeric, *GRAPH_NUMERIC_FEATURE_COLUMNS),
        row_categorical=base.row_categorical,
    )


def fit_graph_experiment(
    connection: duckdb.DuckDBPyConnection,
    config: GraphAblationConfig,
    *,
    experiment_name: str,
    feature_spec: LearnedFeatureSpec,
    training_sample_path: Path,
    model_path: Path,
    include_stay_features: bool,
) -> FittedExperiment:
    """Fit one XGBoost graph ablation experiment."""

    query = training_sample_query(
        config,
        feature_spec,
        include_stay_features=include_stay_features,
    )
    write_training_sample(
        connection,
        query=query,
        output_path=training_sample_path,
    )
    preprocessor, training_counts = fit_preprocessor(
        training_sample_path,
        feature_spec,
    )
    model = fit_xgboost_model(
        training_sample_path,
        feature_spec,
        preprocessor,
        config.seed,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    return FittedExperiment(
        experiment_name=experiment_name,
        feature_spec=feature_spec,
        preprocessor=preprocessor,
        model=model,
        training_counts=training_counts,
        training_sample_path=training_sample_path,
        model_path=model_path,
        include_stay_features=include_stay_features,
    )


def score_experiment_batches(
    connection: duckdb.DuckDBPyConnection,
    config: GraphAblationConfig,
    *,
    experiment: FittedExperiment,
    output_path: Path,
    generated_at: str,
    batch_count: int = SCORING_BATCH_COUNT,
) -> int:
    """Score one fitted graph ablation model in hash batches."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        for batch_id in range(batch_count):
            query = scoring_rows_query(
                config,
                experiment.feature_spec,
                include_stay_features=experiment.include_stay_features,
                batch_id=batch_id,
                batch_count=batch_count,
            )
            frame = connection.execute(query).fetchdf()
            if frame.empty:
                continue
            scores = predict_scores(
                "xgboost",
                model=experiment.model,
                preprocessor=experiment.preprocessor,
                frame=frame,
                feature_spec=experiment.feature_spec,
            )
            batch_output = pd.DataFrame(
                {
                    "source": frame["source"],
                    "split": frame["split"],
                    "ranking_group_id": frame["ranking_group_id"],
                    "index_condition_token": frame["index_condition_token"],
                    "candidate_medication_token": frame["candidate_medication_token"],
                    "candidate_rank": frame["candidate_rank"],
                    "label_prescribed": frame["label_prescribed"],
                    "baseline_name": experiment.experiment_name,
                    "score": scores,
                    "seed": config.seed,
                    "baseline_version": config.graph_ablation_version,
                    "evaluation_version": config.report_version,
                    "generated_at": generated_at,
                }
            )
            table = pa.Table.from_pandas(batch_output, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            row_count += len(batch_output)
    finally:
        if writer is not None:
            writer.close()
    if row_count == 0:
        raise ValueError(f"{experiment.experiment_name} produced no score rows")
    return row_count


def materialize_model_scores(
    connection: duckdb.DuckDBPyConnection,
    config: GraphAblationConfig,
    *,
    experiments: Sequence[FittedExperiment],
    generated_at: str,
) -> tuple[int, dict[str, int]]:
    """Score graph models and combine their score artifacts."""

    temp_paths: list[Path] = []
    row_counts: dict[str, int] = {}
    for experiment in experiments:
        temp_path = config.evaluation_root / f"_{experiment.experiment_name}.parquet"
        row_counts[experiment.experiment_name] = score_experiment_batches(
            connection,
            config,
            experiment=experiment,
            output_path=temp_path,
            generated_at=generated_at,
        )
        temp_paths.append(temp_path)
    combined_count = combine_score_tables(
        connection,
        score_paths=temp_paths,
        output_path=config.model_scores_path,
    )
    return combined_count, row_counts


def combine_score_tables(
    connection: duckdb.DuckDBPyConnection,
    *,
    score_paths: Sequence[Path],
    output_path: Path,
) -> int:
    """Union score parquet tables and return the row count."""

    existing_paths = [path for path in score_paths if path.exists()]
    if not existing_paths:
        raise ValueError("no graph ablation score tables were produced")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unions = " UNION ALL ".join(
        f"SELECT * FROM {parquet_scan(path)}" for path in existing_paths
    )
    connection.execute(f"COPY ({unions}) TO {sql_string(output_path)} (FORMAT PARQUET)")
    row = connection.execute(
        f"SELECT COUNT(*) FROM {parquet_scan(output_path)}"
    ).fetchone()
    return int(row[0]) if row is not None else 0


def select_fusion_weight(
    connection: duckdb.DuckDBPyConnection,
    config: GraphAblationConfig,
    *,
    score_paths: Sequence[Path],
) -> dict[str, Any]:
    """Select a late-fusion weight using MIMIC validation only."""

    score_sql = score_union_sql(score_paths)
    candidates: list[dict[str, Any]] = []
    for graph_weight in FUSION_WEIGHT_GRID:
        rows = fetch_dict_rows(
            connection,
            fusion_validation_metric_query(
                score_sql=score_sql,
                graph_weight=graph_weight,
                k=config.selection_k,
            ),
        )
        if rows:
            candidates.append(rows[0])
    candidates = [
        row
        for row in candidates
        if row.get("positive_ranking_group_count") and row.get("ndcg_at_k") is not None
    ]
    if not candidates:
        return {
            "status": "default_no_validation_pairs",
            "selected_graph_weight": 0.5,
            "selection_split": "mimiciv_validation",
            "selection_k": config.selection_k,
            "candidates": [],
        }
    candidates.sort(
        key=lambda row: (
            row["ndcg_at_k"] or -1.0,
            row["mrr_at_k"] or -1.0,
            row["hit_rate_at_k"] or -1.0,
            -abs(float(row["graph_weight"]) - 0.5),
        ),
        reverse=True,
    )
    best = candidates[0]
    return {
        "status": "selected",
        "selected_graph_weight": float(best["graph_weight"]),
        "selection_split": "mimiciv_validation",
        "selection_k": config.selection_k,
        "selected_metrics": best,
        "candidates": candidates,
    }


def write_fusion_weights(config: GraphAblationConfig, payload: dict[str, Any]) -> None:
    """Write local fusion-weight metadata."""

    config.fusion_weights_path.parent.mkdir(parents=True, exist_ok=True)
    config.fusion_weights_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_fusion_weight_from_selection(config: GraphAblationConfig) -> dict[str, Any]:
    """Load a previously frozen fusion weight for final-mode scoring."""

    selection = load_json_if_present(config.frozen_selection_path)
    if selection is None:
        return {
            "status": "default_missing_frozen_selection",
            "selected_graph_weight": 0.5,
            "selection_split": "mimiciv_validation",
            "selection_k": config.selection_k,
            "candidates": [],
        }
    return selection.get(
        "fusion_weight",
        {
            "status": "default_missing_fusion_weight",
            "selected_graph_weight": 0.5,
            "selection_split": "mimiciv_validation",
            "selection_k": config.selection_k,
            "candidates": [],
        },
    )


def metric_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    """Return a stable key for a ranking metric row."""

    return (
        str(row["baseline_name"]),
        str(row["source"]),
        str(row["split"]),
        int(row["k"]),
    )


def select_ablation_model(
    config: GraphAblationConfig,
    manifest: dict[str, Any],
    *,
    generated_at: str,
    fusion_weight: dict[str, Any],
) -> dict[str, Any]:
    """Select the validation winner under the Milestone 8B gate."""

    ranking_by_key = {metric_key(row): row for row in manifest["ranking_metrics"]}
    row_metrics_by_experiment = {
        row["baseline_name"]: row
        for row in manifest["row_level_metrics"]
        if row["source"] == "mimiciv" and row["split"] == "validation"
    }
    reference_key = (
        "xgboost_frozen_reference",
        "mimiciv",
        "validation",
        config.selection_k,
    )
    reference = ranking_by_key.get(reference_key)
    if reference is None or reference.get("ndcg_at_k") is None:
        return {
            "schema_version": FROZEN_SELECTION_SCHEMA_VERSION,
            "status": "frozen",
            "frozen_at": generated_at,
            "selected_experiment": "xgboost_frozen_reference",
            "reason": "No evaluable MIMIC validation ranking metric was available.",
            "selection_basis": {
                "split": "mimiciv_validation",
                "k": config.selection_k,
                "primary_metric": "ndcg_at_k",
            },
            "fusion_weight": fusion_weight,
            "observational_label_caveat": manifest["observational_label_caveat"],
        }

    candidates = [
        row
        for row in manifest["ranking_metrics"]
        if row["source"] == "mimiciv"
        and row["split"] == "validation"
        and int(row["k"]) == config.selection_k
        and row["baseline_name"] != "xgboost_frozen_reference"
        and row.get("ndcg_at_k") is not None
    ]
    candidates.sort(
        key=lambda row: (
            row["ndcg_at_k"] or -1.0,
            row["mrr_at_k"] or -1.0,
            row["hit_rate_at_k"] or -1.0,
            (
                row_metrics_by_experiment.get(row["baseline_name"], {}).get(
                    "average_precision"
                )
                or -1.0
            ),
            (
                row_metrics_by_experiment.get(row["baseline_name"], {}).get("roc_auc")
                or -1.0
            ),
            row["baseline_name"],
        ),
        reverse=True,
    )
    selected = reference
    reason = (
        "Frozen XGBoost retained because no graph-aware candidate cleared the "
        "validation lift gate."
    )
    if candidates:
        best = candidates[0]
        reference_ndcg = float(reference["ndcg_at_k"] or 0.0)
        reference_mrr = float(reference["mrr_at_k"] or 0.0)
        reference_hit = float(reference["hit_rate_at_k"] or 0.0)
        best_ndcg = float(best["ndcg_at_k"] or 0.0)
        best_mrr = float(best["mrr_at_k"] or 0.0)
        best_hit = float(best["hit_rate_at_k"] or 0.0)
        if (
            best_ndcg >= reference_ndcg + GRAPH_SELECTION_DELTA
            and best_mrr >= reference_mrr - PRIMARY_METRIC_MAX_DROP
            and best_hit >= reference_hit - PRIMARY_METRIC_MAX_DROP
        ):
            selected = best
            reason = (
                "Graph-aware candidate cleared the validation lift gate over "
                "the frozen XGBoost reference."
            )

    return {
        "schema_version": FROZEN_SELECTION_SCHEMA_VERSION,
        "status": "frozen",
        "frozen_at": generated_at,
        "selected_experiment": selected["baseline_name"],
        "reference_experiment": "xgboost_frozen_reference",
        "reason": reason,
        "selection_basis": {
            "split": "mimiciv_validation",
            "k": config.selection_k,
            "primary_metric": "ndcg_at_k",
            "tie_breakers": ["mrr_at_k", "hit_rate_at_k", "average_precision"],
            "minimum_ndcg_lift": GRAPH_SELECTION_DELTA,
            "maximum_primary_metric_drop": PRIMARY_METRIC_MAX_DROP,
            "reference_metrics": reference,
            "reference_row_level_metrics": row_metrics_by_experiment.get(
                "xgboost_frozen_reference"
            ),
            "candidate_metrics": candidates,
            "candidate_row_level_metrics": {
                row["baseline_name"]: row_metrics_by_experiment.get(
                    row["baseline_name"]
                )
                for row in candidates
            },
            "selected_metrics": selected,
            "selected_row_level_metrics": row_metrics_by_experiment.get(
                selected["baseline_name"]
            ),
        },
        "fusion_weight": fusion_weight,
        "observational_label_caveat": manifest["observational_label_caveat"],
        "clinical_claim_boundary": manifest["clinical_claim_boundary"],
    }


def preflight_gate_status(config: GraphAblationConfig) -> tuple[str, str] | None:
    """Return a blocking status and reason when upstream gates are not satisfied."""

    milestone7 = load_json_if_present(config.milestone7_evaluation_report_path) or {}
    milestone7_parameters = milestone7.get("parameters", {})
    final_gate_ok = (
        milestone7.get("status") == "completed"
        and milestone7_parameters.get("mode") == "final"
        and milestone7_parameters.get("frozen_selection") is True
    )
    development_gate_ok = (
        config.allow_development_milestone7_reference
        and config.mode == "development"
        and milestone7.get("status") == "completed"
        and milestone7_parameters.get("mode") == "development"
    )
    if not (final_gate_ok or development_gate_ok):
        return (
            "blocked_milestone7_final_missing",
            (
                "Milestone 8B requires a completed Milestone 7 final evaluation "
                "with frozen XGBoost selection, or a completed development "
                "evaluation when --allow-development-milestone7-reference is set."
            ),
        )

    suitability = load_json_if_present(config.milestone8_suitability_report_path) or {}
    gate = suitability.get("gate_review", {})
    audit = suitability.get("leakage_audit", {})
    if gate.get("result") != "pass_for_graph_ablation":
        return (
            "blocked_graph_gate_not_passed",
            "Milestone 8 graph gate did not pass for graph ablation.",
        )
    if audit.get("status") != "pass":
        return (
            "blocked_graph_leakage_audit_failed",
            "Milestone 8 graph leakage audit did not pass.",
        )

    if config.mode == "final" and (
        not config.frozen_selection or not config.frozen_selection_path.exists()
    ):
        return (
            "blocked_final_requires_frozen_selection",
            (
                "Final/test graph-ablation evaluation requires "
                "--frozen-selection and reports/milestone8b_frozen_selection.json."
            ),
        )
    return None


def base_manifest(
    config: GraphAblationConfig,
    *,
    status: str,
    generated_at: str,
) -> dict[str, Any]:
    """Build the common Milestone 8B report shell."""

    return {
        "schema_version": config.report_version,
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
            "selection_k": config.selection_k,
            "seed": config.seed,
            "condition_tokens": list(config.condition_tokens),
            "minimum_ndcg_lift": GRAPH_SELECTION_DELTA,
            "maximum_primary_metric_drop": PRIMARY_METRIC_MAX_DROP,
        },
        "versions": {
            "graph_ablation_version": config.graph_ablation_version,
            "report_version": config.report_version,
            "graph_version": config.graph_version,
            "feature_version": config.feature_version,
            "label_version": config.label_version,
            "split_version": config.split_version,
        },
        "inputs": {
            "patient_stay_features": str(config.patient_stay_features_path),
            "patient_condition_medication": str(
                config.patient_condition_medication_path
            ),
            "candidate_catalog": str(config.candidate_catalog_path),
            "graph_edges": str(config.graph_edges_path),
            "milestone7_scores": str(config.milestone7_scores_path),
            "milestone7_evaluation_report": str(
                config.milestone7_evaluation_report_path
            ),
            "milestone8_suitability_report": str(
                config.milestone8_suitability_report_path
            ),
        },
        "artifacts": {},
        "observational_label_caveat": (
            "Labels are observed historical prescriptions in the label window. "
            "Unobserved catalog candidates are weak negatives, not confirmed "
            "clinical non-indications."
        ),
        "clinical_claim_boundary": (
            "Milestone 8B reports graph-aware ablations only. It is not a "
            "validated clinical recommendation, external validation result, "
            "or full Transformer-GNN model."
        ),
    }


def blocked_reports(
    config: GraphAblationConfig,
    *,
    generated_at: str,
    status: str,
    reason: str,
    missing: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Write aggregate-only blocked reports."""

    feature_manifest = {
        "schema_version": FEATURE_MANIFEST_SCHEMA_VERSION,
        "status": status,
        "generated_at": generated_at,
        "reason": reason,
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "local_feature_artifacts_contain_patient_level_rows": True,
        },
    }
    manifest = base_manifest(config, status=status, generated_at=generated_at)
    manifest["reason"] = reason
    if missing is not None:
        feature_manifest["missing_inputs"] = missing
        manifest["missing_inputs"] = missing
    write_json(config.feature_manifest_path, feature_manifest)
    write_json(config.evaluation_report_path, manifest)
    return manifest


def build_graph_feature_manifest(
    connection: duckdb.DuckDBPyConnection,
    config: GraphAblationConfig,
    *,
    generated_at: str,
    row_count: int,
) -> dict[str, Any]:
    """Build the aggregate graph-feature manifest."""

    return {
        "schema_version": FEATURE_MANIFEST_SCHEMA_VERSION,
        "status": "completed",
        "generated_at": generated_at,
        "artifacts": {
            "graph_feature_matrix": str(config.graph_feature_matrix_path),
        },
        "tables": [
            {
                "table_name": "graph_feature_matrix",
                "status": "completed",
                "row_count": row_count,
            }
        ],
        "feature_columns": list(GRAPH_NUMERIC_FEATURE_COLUMNS),
        "feature_summary_by_source_split": fetch_dict_rows(
            connection,
            graph_feature_summary_query(config),
        ),
        "leakage_policy": (
            "Graph features are derived from Milestone 8 graph edges fit from "
            "MIMIC train only. Validation, test, and eICU rows are scored for "
            "coverage/evaluation only."
        ),
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "local_feature_artifacts_contain_patient_level_rows": True,
            "local_feature_artifact_storage": "ignored Dataset/processed/evaluation",
        },
        "versions": {
            "graph_ablation_version": config.graph_ablation_version,
            "graph_version": config.graph_version,
            "feature_version": config.feature_version,
            "label_version": config.label_version,
            "split_version": config.split_version,
        },
    }


def experiment_report_section(
    experiments: Sequence[FittedExperiment],
    config: GraphAblationConfig,
    *,
    model_row_counts: dict[str, int],
) -> dict[str, Any]:
    """Return aggregate metadata for graph ablation model experiments."""

    return {
        "status": "completed",
        "negative_to_positive_ratio": NEGATIVE_TO_POSITIVE_RATIO,
        "hyperparameters": {
            "xgboost": {
                **XGBOOST_V1_HYPERPARAMETERS,
                "seed": config.seed,
            },
        },
        "experiments": [
            {
                "experiment_name": experiment.experiment_name,
                "include_stay_features": experiment.include_stay_features,
                "training_sample": experiment.training_counts,
                "training_sample_path": str(experiment.training_sample_path),
                "model_path": str(experiment.model_path),
                "score_row_count": model_row_counts.get(
                    experiment.experiment_name,
                    0,
                ),
                "feature_columns": list(experiment.feature_spec.model_columns),
                "graph_feature_columns": list(GRAPH_NUMERIC_FEATURE_COLUMNS),
            }
            for experiment in experiments
        ],
        "preprocessor_artifact": str(config.graph_feature_preprocessor_path),
    }


def build_graph_ablation(
    config: GraphAblationConfig = GraphAblationConfig(),
) -> dict[str, Any]:
    """Build Milestone 8B graph-aware ablation artifacts and reports."""

    generated_at = datetime.now(UTC).isoformat()
    config.evaluation_root.mkdir(parents=True, exist_ok=True)
    missing = missing_input_tables(config)
    if missing:
        return blocked_reports(
            config,
            generated_at=generated_at,
            status="failed_missing_inputs",
            reason="Required Milestone 6/7/8 graph-ablation inputs are missing.",
            missing=missing,
        )
    blocker = preflight_gate_status(config)
    if blocker is not None:
        status, reason = blocker
        return blocked_reports(
            config,
            generated_at=generated_at,
            status=status,
            reason=reason,
        )

    with duckdb.connect(database=":memory:") as connection:
        configure_connection(config, connection)
        try:
            graph_feature_rows = copy_query_to_parquet(
                connection,
                graph_feature_query(config),
                config.graph_feature_matrix_path,
            )
            feature_manifest = build_graph_feature_manifest(
                connection,
                config,
                generated_at=generated_at,
                row_count=graph_feature_rows,
            )
            write_json(config.feature_manifest_path, feature_manifest)

            graph_only = fit_graph_experiment(
                connection,
                config,
                experiment_name="graph_only_xgboost",
                feature_spec=graph_only_feature_spec(),
                training_sample_path=config.graph_only_training_sample_path,
                model_path=config.graph_only_model_path,
                include_stay_features=False,
            )
            graph_augmented = fit_graph_experiment(
                connection,
                config,
                experiment_name="xgboost_graph_augmented",
                feature_spec=graph_augmented_feature_spec(connection, config),
                training_sample_path=config.graph_augmented_training_sample_path,
                model_path=config.graph_augmented_model_path,
                include_stay_features=True,
            )
            experiments = (graph_only, graph_augmented)
            config.models_root.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    experiment.experiment_name: {
                        "feature_spec": experiment.feature_spec,
                        "preprocessor": experiment.preprocessor,
                    }
                    for experiment in experiments
                },
                config.graph_feature_preprocessor_path,
            )

            reference_row_count = copy_query_to_parquet(
                connection,
                reference_scores_query(config, generated_at=generated_at),
                config.reference_scores_path,
            )
            model_row_count, model_row_counts = materialize_model_scores(
                connection,
                config,
                experiments=experiments,
                generated_at=generated_at,
            )
            if config.mode == "final":
                fusion_weight = load_fusion_weight_from_selection(config)
            else:
                fusion_weight = select_fusion_weight(
                    connection,
                    config,
                    score_paths=(
                        config.reference_scores_path,
                        config.model_scores_path,
                    ),
                )
            write_fusion_weights(config, fusion_weight)
            selected_graph_weight = float(
                fusion_weight.get("selected_graph_weight", 0.5)
            )
            fusion_row_count = copy_query_to_parquet(
                connection,
                fusion_scores_query(
                    config,
                    reference_path=config.reference_scores_path,
                    model_scores_path=config.model_scores_path,
                    graph_weight=selected_graph_weight,
                    generated_at=generated_at,
                ),
                config.fusion_scores_path,
            )
            final_score_count = combine_score_tables(
                connection,
                score_paths=(
                    config.reference_scores_path,
                    config.model_scores_path,
                    config.fusion_scores_path,
                ),
                output_path=config.score_output_path,
            )

            manifest = base_manifest(
                config,
                status="completed",
                generated_at=generated_at,
            )
            manifest["coverage_summary"] = add_evaluability_status(
                fetch_dict_rows(connection, coverage_summary_query(config))
            )
            manifest["artifacts"] = {
                "graph_feature_matrix": str(config.graph_feature_matrix_path),
                "graph_ablation_scores": str(config.score_output_path),
                "reference_scores": str(config.reference_scores_path),
                "model_scores": str(config.model_scores_path),
                "fusion_scores": str(config.fusion_scores_path),
                "fusion_weights": str(config.fusion_weights_path),
            }
            manifest["tables"] = [
                {
                    "table_name": "graph_feature_matrix",
                    "status": "completed",
                    "row_count": graph_feature_rows,
                },
                {
                    "table_name": "xgboost_frozen_reference_scores",
                    "status": "completed",
                    "row_count": reference_row_count,
                },
                {
                    "table_name": "graph_model_scores",
                    "status": "completed",
                    "row_count": model_row_count,
                },
                {
                    "table_name": "fusion_scores",
                    "status": "completed",
                    "row_count": fusion_row_count,
                },
                {
                    "table_name": "graph_ablation_scores",
                    "status": "completed",
                    "row_count": final_score_count,
                },
            ]
            manifest["graph_features"] = {
                "manifest_path": str(config.feature_manifest_path),
                "feature_columns": list(GRAPH_NUMERIC_FEATURE_COLUMNS),
                "graph_source": "Milestone 8 train-fit concept graph",
            }
            manifest["graph_ablation_models"] = experiment_report_section(
                experiments,
                config,
                model_row_counts=model_row_counts,
            )
            manifest["fusion"] = fusion_weight
            append_metric_summaries(connection, config, manifest)
            if config.mode == "development":
                selection = select_ablation_model(
                    config,
                    manifest,
                    generated_at=generated_at,
                    fusion_weight=fusion_weight,
                )
                write_json(config.frozen_selection_path, selection)
            else:
                selection = load_json_if_present(config.frozen_selection_path)
            manifest["frozen_selection"] = selection
            manifest["eicu_policy"] = (
                "eICU remains coverage-only unless in-catalog positive groups "
                "exist; no external performance claim is made from zero-positive "
                "splits."
            )
        except Exception as error:
            manifest = base_manifest(config, status="failed", generated_at=generated_at)
            manifest["reason"] = safe_error_message(error)

    write_json(config.evaluation_report_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Milestone 8B graph-aware ranking ablations.",
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
        "--graph-root",
        type=Path,
        default=MILESTONE8_GRAPH_ROOT,
        help="Root directory containing Milestone 8 graph artifacts.",
    )
    parser.add_argument(
        "--milestone7-evaluation-root",
        type=Path,
        default=MILESTONE7_EVALUATION_ROOT,
        help="Root directory containing Milestone 7 local score artifacts.",
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=MILESTONE8B_EVALUATION_ROOT,
        help="Output directory for local ignored Milestone 8B artifacts.",
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=DEFAULT_FEATURE_MANIFEST_PATH,
        help="Output path for the aggregate graph-feature manifest.",
    )
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=DEFAULT_EVALUATION_REPORT_PATH,
        help="Output path for the aggregate Milestone 8B evaluation report.",
    )
    parser.add_argument(
        "--frozen-selection-report",
        type=Path,
        default=DEFAULT_FROZEN_SELECTION_PATH,
        help="Path for the Milestone 8B frozen-selection report.",
    )
    parser.add_argument(
        "--milestone6-feature-manifest",
        type=Path,
        default=REPORTS_ROOT / "milestone6_feature_manifest.json",
        help="Milestone 6 aggregate feature manifest path.",
    )
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=REPORTS_ROOT / "training_table_manifest.json",
        help="Milestone 6 aggregate training-table manifest path.",
    )
    parser.add_argument(
        "--milestone7-evaluation-report",
        type=Path,
        default=REPORTS_ROOT / "milestone7_baseline_evaluation.json",
        help="Milestone 7 final baseline evaluation report path.",
    )
    parser.add_argument(
        "--milestone8-suitability-report",
        type=Path,
        default=REPORTS_ROOT / "milestone8_graph_suitability.json",
        help="Milestone 8 graph suitability report path.",
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
        help="Development selects on validation; final adds held-out test metrics.",
    )
    parser.add_argument(
        "--frozen-selection",
        action="store_true",
        help="Required with --mode final before held-out test metrics are produced.",
    )
    parser.add_argument(
        "--allow-development-milestone7-reference",
        action="store_true",
        help=(
            "Allow Milestone 8B development scoring when the Milestone 7 "
            "evaluation report is development-mode only. Intended for isolated "
            "Phase 8 P0 ablations."
        ),
    )
    parser.add_argument(
        "--condition-token",
        action="append",
        default=[],
        help="Optional condition token filter; may be repeated or comma-separated.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Seed used for deterministic sampling.",
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
        manifest = build_graph_ablation(
            GraphAblationConfig(
                features_root=args.features_root,
                training_root=args.training_root,
                graph_root=args.graph_root,
                milestone7_evaluation_root=args.milestone7_evaluation_root,
                evaluation_root=args.evaluation_root,
                feature_manifest_path=args.feature_manifest,
                evaluation_report_path=args.evaluation_report,
                frozen_selection_path=args.frozen_selection_report,
                milestone6_feature_manifest_path=args.milestone6_feature_manifest,
                training_manifest_path=args.training_manifest,
                milestone7_evaluation_report_path=args.milestone7_evaluation_report,
                milestone8_suitability_report_path=args.milestone8_suitability_report,
                top_k=parse_top_k(args.top_k),
                mode=args.mode,
                frozen_selection=args.frozen_selection,
                allow_development_milestone7_reference=(
                    args.allow_development_milestone7_reference
                ),
                seed=args.seed,
                condition_tokens=parse_repeated_csv(args.condition_token),
                duckdb_temp_directory=args.duckdb_temp_dir,
                duckdb_memory_limit=args.duckdb_memory_limit,
                duckdb_threads=args.duckdb_threads,
            )
        )
    except ValueError as error:
        print(f"Invalid Milestone 8B graph-ablation arguments: {error}")
        return 2

    print(
        "Wrote Milestone 8B graph-ablation report: "
        f"status={manifest['status']}, "
        f"report={args.evaluation_report}"
    )
    if manifest["status"].startswith("blocked_") or manifest["status"].startswith(
        "failed_"
    ):
        return 2
    return 1 if manifest["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
