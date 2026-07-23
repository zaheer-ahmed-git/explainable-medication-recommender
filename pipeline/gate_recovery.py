"""Run the gate-first Phase 8 P0 rank-aware structured recovery experiment."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

from pipeline.config import (
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_TEMP_DIR,
    DUCKDB_THREADS,
    PROCESSED_DATA_ROOT,
    PROJECT_ROOT,
    RANDOM_SEED,
    REPORTS_ROOT,
)
from pipeline.evaluate_baselines import (
    BaselineEvaluationConfig,
    append_metric_summaries,
    parse_top_k,
)
from pipeline.extract_utils import (
    configure_duckdb_connection,
    parquet_scan,
    safe_error_message,
    sql_string,
)
from pipeline.features import copy_query_to_parquet
from pipeline.graph_ablation import (
    GRAPH_NUMERIC_FEATURE_COLUMNS,
    GraphAblationConfig,
    graph_feature_query,
)
from pipeline.io_utils import quote_identifier
from pipeline.learned_baselines import (
    XGBOOST_V1_HYPERPARAMETERS,
    LearnedFeatureSpec,
    build_preprocessor,
    resolve_feature_spec,
)
from pipeline.training_contract import (
    DEFAULT_OUTPUT_PATH as DEFAULT_CONTRACT_LOCK,
)
from pipeline.training_contract import (
    approved_model_projection,
    load_json,
    schema_columns,
    sha256_file,
    validate_requested_columns,
    write_json,
)


SCHEMA_VERSION = "phase8-p0-gate-recovery-evaluation-v1"
SELECTION_SCHEMA_VERSION = "phase8-p0-gate-recovery-selection-v1"
EXPERIMENT_VERSION = "phase8-p0-rank-recovery-v1"
PHASE8_P0_ROOT = PROCESSED_DATA_ROOT / "phase8_p0"
DEFAULT_FEATURES_ROOT = PHASE8_P0_ROOT / "features"
DEFAULT_TRAINING_ROOT = PHASE8_P0_ROOT / "training"
DEFAULT_GRAPH_ROOT = PHASE8_P0_ROOT / "graph" / "milestone8"
DEFAULT_REFERENCE_ROOT = PHASE8_P0_ROOT / "evaluation" / "milestone8b"
DEFAULT_REFERENCE_SCORES = DEFAULT_REFERENCE_ROOT / "_scores_reference.parquet"
DEFAULT_REFERENCE_SELECTION = (
    REPORTS_ROOT / "phase8_p0_milestone8b_frozen_selection.json"
)
DEFAULT_FEATURE_MANIFEST = REPORTS_ROOT / "phase8_p0_milestone6_feature_manifest.json"
DEFAULT_EVALUATION_ROOT = PHASE8_P0_ROOT / "evaluation" / "gate_recovery"
DEFAULT_EVALUATION_REPORT = REPORTS_ROOT / "phase8_p0_gate_recovery_evaluation.json"
DEFAULT_SELECTION_REPORT = REPORTS_ROOT / "phase8_p0_gate_recovery_selection.json"

FOLD_COUNT = 3
SELECTION_K = 10
MINIMUM_NDCG_LIFT = 0.005
MAXIMUM_SECONDARY_DROP = 0.01
FROZEN_REFERENCE_NDCG_AT_10 = 0.374899
CONDITION_CAPS = (0, 20, 40)
GRAPH_SUPPORT_THRESHOLDS = (1, 5, 10)
FUSION_WEIGHT_GRID = tuple(round(index / 20, 2) for index in range(21))

DIRECT_GRAPH_FEATURES = (
    "graph_condition_medication_support_count",
    "graph_condition_medication_log_support",
    "graph_condition_medication_support_share",
    "graph_condition_total_medication_support",
    "graph_direct_edge_present",
)
CONTEXT_GRAPH_FEATURES = tuple(
    column
    for column in GRAPH_NUMERIC_FEATURE_COLUMNS
    if column not in DIRECT_GRAPH_FEATURES
)
GRAPH_FAMILIES = {
    "none": (),
    "direct": DIRECT_GRAPH_FEATURES,
    "context": CONTEXT_GRAPH_FEATURES,
    "all": GRAPH_NUMERIC_FEATURE_COLUMNS,
}

METADATA_COLUMNS = (
    "source",
    "split",
    "ranking_group_id",
    "index_condition_token",
    "candidate_medication_token",
    "candidate_rank",
    "label_prescribed",
)


@dataclass(frozen=True)
class RankerHyperparameters:
    """Locked XGBoost ranker parameters used by one recovery experiment."""

    max_depth: int
    learning_rate: float
    min_child_weight: float
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    max_boost_rounds: int = 1000
    early_stopping_rounds: int = 50

    @property
    def name(self) -> str:
        return (
            f"d{self.max_depth}_eta{self.learning_rate:g}_mcw{self.min_child_weight:g}"
        )


SCREENING_HYPERPARAMETERS = RankerHyperparameters(
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=10.0,
)
HYPERPARAMETER_GRID = (
    RankerHyperparameters(4, 0.05, 10.0),
    RankerHyperparameters(6, 0.05, 10.0),
    RankerHyperparameters(4, 0.03, 1.0),
    RankerHyperparameters(6, 0.03, 1.0),
)


@dataclass(frozen=True)
class RecoveryExperiment:
    """Feature choices for one train-fold recovery experiment."""

    condition_cap: int
    graph_support_threshold: int
    graph_family: str
    include_candidate_rank: bool

    @property
    def name(self) -> str:
        rank_name = "rank" if self.include_candidate_rank else "no_rank"
        return (
            f"condition_{self.condition_cap}_graph_{self.graph_family}_"
            f"support_{self.graph_support_threshold}_{rank_name}"
        )


@dataclass(frozen=True)
class GateRecoveryConfig:
    """Configuration for the Phase 8 P0 rank-aware recovery runner."""

    features_root: Path = DEFAULT_FEATURES_ROOT
    training_root: Path = DEFAULT_TRAINING_ROOT
    graph_root: Path = DEFAULT_GRAPH_ROOT
    reference_scores_path: Path = DEFAULT_REFERENCE_SCORES
    reference_selection_path: Path = DEFAULT_REFERENCE_SELECTION
    contract_lock_path: Path = DEFAULT_CONTRACT_LOCK
    feature_manifest_path: Path = DEFAULT_FEATURE_MANIFEST
    evaluation_root: Path = DEFAULT_EVALUATION_ROOT
    evaluation_report_path: Path = DEFAULT_EVALUATION_REPORT
    selection_report_path: Path = DEFAULT_SELECTION_REPORT
    mode: str = "development"
    frozen_selection: bool = False
    top_k: tuple[int, ...] = (1, 3, 5, 10)
    seed: int = RANDOM_SEED
    fold_count: int = FOLD_COUNT
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
    def cache_root(self) -> Path:
        return self.evaluation_root / "cache"

    @property
    def models_root(self) -> Path:
        return self.evaluation_root / "models"

    @property
    def selected_model_path(self) -> Path:
        return self.models_root / "xgboost_rank_ndcg_gate_recovery.json"

    @property
    def selected_preprocessor_path(self) -> Path:
        return self.models_root / "ranker_preprocessor.joblib"

    @property
    def selected_config_path(self) -> Path:
        return self.models_root / "selected_recovery_config.json"

    @property
    def score_root(self) -> Path:
        return (
            self.evaluation_root
            if self.mode == "development"
            else self.evaluation_root / "final"
        )

    @property
    def score_output_path(self) -> Path:
        return self.score_root / "baseline_scores.parquet"

    def graph_feature_path(self, support_threshold: int) -> Path:
        scope = "development" if self.mode == "development" else "final"
        return (
            self.cache_root
            / f"graph_features_{scope}_support_{support_threshold}.parquet"
        )


def configure_connection(
    config: GateRecoveryConfig,
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Apply bounded DuckDB settings."""

    configure_duckdb_connection(
        connection,
        temp_directory=config.duckdb_temp_directory,
        memory_limit=config.duckdb_memory_limit,
        threads=config.duckdb_threads,
    )


def required_inputs(config: GateRecoveryConfig) -> dict[str, Path]:
    """Return inputs required before any model fit."""

    return {
        "training_contract_lock": config.contract_lock_path,
        "feature_manifest": config.feature_manifest_path,
        "reference_selection": config.reference_selection_path,
        "reference_scores": config.reference_scores_path,
        "patient_stay_features": config.patient_stay_features_path,
        "patient_condition_medication": config.patient_condition_medication_path,
        "candidate_catalog": config.candidate_catalog_path,
        "graph_edges": config.graph_edges_path,
    }


def _locked_input_errors(contract: dict[str, Any]) -> list[dict[str, str]]:
    """Check the lock's file metadata and aggregate-manifest hashes."""

    errors: list[dict[str, str]] = []
    for artifact_name, lock in (
        contract.get("contract", {}).get("artifacts", {}).items()
    ):
        path = Path(lock["path"])
        if not path.exists():
            errors.append(
                {
                    "code": "locked_artifact_missing",
                    "artifact_name": artifact_name,
                }
            )
            continue
        stat = path.stat()
        if int(lock["file_size_bytes"]) != stat.st_size or int(
            lock["modified_time_ns"]
        ) != int(stat.st_mtime_ns):
            errors.append(
                {
                    "code": "locked_artifact_changed",
                    "artifact_name": artifact_name,
                }
            )
    for manifest_name, lock in (
        contract.get("contract", {}).get("manifests", {}).items()
    ):
        path = Path(lock["path"])
        if not path.exists() or sha256_file(path) != lock["sha256"]:
            errors.append(
                {
                    "code": "locked_manifest_changed",
                    "artifact_name": manifest_name,
                }
            )
    return errors


def _git_revision() -> str:
    """Return the current repository revision without making it a hard dependency."""

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _package_lock_digest() -> str:
    lock_path = PROJECT_ROOT / "uv.lock"
    return sha256_file(lock_path) if lock_path.exists() else "unavailable"


def _frozen_model_artifacts(config: GateRecoveryConfig) -> dict[str, dict[str, str]]:
    return {
        artifact_name: {"path": str(path), "sha256": sha256_file(path)}
        for artifact_name, path in (
            ("selected_model", config.selected_model_path),
            ("selected_preprocessor", config.selected_preprocessor_path),
            ("selected_config", config.selected_config_path),
        )
        if path.exists()
    }


def preflight_errors(config: GateRecoveryConfig) -> list[dict[str, str]]:
    """Return fail-closed aggregate preflight errors."""

    errors = [
        {"code": "missing_input", "artifact_name": name, "path": str(path)}
        for name, path in required_inputs(config).items()
        if not path.exists()
    ]
    if errors:
        return errors

    try:
        contract = load_json(config.contract_lock_path)
    except (OSError, json.JSONDecodeError):
        return [
            {
                "code": "invalid_contract_lock",
                "detail": "training contract lock is not valid JSON",
            }
        ]
    if contract.get("status") != "completed" or not contract.get("contract_digest"):
        errors.append(
            {
                "code": "invalid_contract_lock",
                "detail": "training contract lock must be completed and contain a digest",
            }
        )
    else:
        try:
            errors.extend(_locked_input_errors(contract))
        except (KeyError, OSError, TypeError, ValueError):
            errors.append(
                {
                    "code": "invalid_contract_lock",
                    "detail": "training contract lock metadata is malformed",
                }
            )
    try:
        reference = load_json(config.reference_selection_path)
    except (OSError, json.JSONDecodeError):
        errors.append(
            {
                "code": "invalid_reference_selection",
                "detail": "frozen reference selection is not valid JSON",
            }
        )
        return errors
    if reference.get("selected_experiment") != "xgboost_frozen_reference":
        errors.append(
            {
                "code": "invalid_reference_selection",
                "detail": "Phase 8 P0 frozen reference must remain xgboost_frozen_reference",
            }
        )
    reference_metric = reference.get("selection_basis", {}).get("reference_metrics", {})
    reference_ndcg = reference_metric.get("ndcg_at_k")
    try:
        reference_anchor_matches = reference_ndcg is not None and round(
            float(reference_ndcg), 6
        ) == round(FROZEN_REFERENCE_NDCG_AT_10, 6)
    except (TypeError, ValueError):
        reference_anchor_matches = False
    if reference.get("status") != "frozen" or not reference_anchor_matches:
        errors.append(
            {
                "code": "reference_metric_mismatch",
                "detail": "frozen validation NDCG@10 anchor must remain 0.374899",
            }
        )
    if config.mode == "final":
        if not config.frozen_selection:
            errors.append(
                {
                    "code": "final_requires_frozen_selection",
                    "detail": "final mode requires --frozen-selection",
                }
            )
        if not config.selection_report_path.exists():
            errors.append(
                {
                    "code": "missing_recovery_selection",
                    "detail": "development recovery selection report is missing",
                }
            )
        else:
            try:
                selection = load_json(config.selection_report_path)
            except (OSError, json.JSONDecodeError):
                errors.append(
                    {
                        "code": "invalid_recovery_selection",
                        "detail": "development selection is not valid JSON",
                    }
                )
                selection = {}
            if selection.get("status") != "frozen" or not selection.get(
                "neural_training_authorized", False
            ):
                errors.append(
                    {
                        "code": "recovery_gate_not_passed",
                        "detail": "final scoring is blocked until recovery passes",
                    }
                )
            if selection.get("contract_digest") != contract.get("contract_digest"):
                errors.append(
                    {
                        "code": "selection_contract_mismatch",
                        "detail": "development selection does not match the input lock",
                    }
                )
            frozen_artifacts = selection.get("frozen_artifacts", {})
            for artifact_name, lock in frozen_artifacts.items():
                try:
                    path = Path(lock["path"])
                    matches = path.exists() and sha256_file(path) == lock["sha256"]
                except (KeyError, OSError, TypeError, ValueError):
                    matches = False
                if not matches:
                    errors.append(
                        {
                            "code": "frozen_recovery_artifact_changed",
                            "artifact_name": artifact_name,
                        }
                    )
            if not frozen_artifacts:
                errors.append(
                    {
                        "code": "missing_frozen_artifact_lock",
                        "detail": "development selection does not lock model artifacts",
                    }
                )
        for artifact_name, path in (
            ("selected_model", config.selected_model_path),
            ("selected_preprocessor", config.selected_preprocessor_path),
            ("selected_config", config.selected_config_path),
        ):
            if not path.exists():
                errors.append(
                    {
                        "code": "missing_frozen_recovery_artifact",
                        "artifact_name": artifact_name,
                        "path": str(path),
                    }
                )
    return errors


def condition_columns(config: GateRecoveryConfig) -> tuple[str, ...]:
    """Read the train-ordered Phase 8 condition columns from its aggregate manifest."""

    manifest = load_json(config.feature_manifest_path)
    return tuple(str(value) for value in manifest.get("condition_columns_added", []))


def graph_columns(graph_family: str) -> tuple[str, ...]:
    """Return one registered graph feature family."""

    try:
        return tuple(GRAPH_FAMILIES[graph_family])
    except KeyError as error:
        raise ValueError(f"unsupported graph family: {graph_family}") from error


def resolve_recovery_feature_spec(
    connection: duckdb.DuckDBPyConnection,
    config: GateRecoveryConfig,
    experiment: RecoveryExperiment,
) -> LearnedFeatureSpec:
    """Resolve a leakage-reviewed feature specification for one experiment."""

    base = resolve_feature_spec(connection, config.patient_stay_features_path)
    schema = schema_columns(connection, config.patient_stay_features_path)
    approved = set(approved_model_projection("patient_stay_features", schema))
    all_condition_columns = condition_columns(config)
    selected_conditions = set(all_condition_columns[: experiment.condition_cap])
    stay_numeric = tuple(
        column
        for column in base.stay_numeric
        if column in approved
        and (column not in all_condition_columns or column in selected_conditions)
    )
    stay_categorical = tuple(
        column for column in base.stay_categorical if column in approved
    )
    validate_requested_columns(
        "patient_stay_features",
        (*stay_numeric, *stay_categorical),
        schema,
    )
    return LearnedFeatureSpec(
        stay_numeric=stay_numeric,
        stay_categorical=stay_categorical,
        row_numeric=(
            *(("candidate_rank",) if experiment.include_candidate_rank else ()),
            *graph_columns(experiment.graph_family),
        ),
        row_categorical=("index_condition_token", "candidate_medication_token"),
    )


def materialize_graph_features(
    connection: duckdb.DuckDBPyConnection,
    config: GateRecoveryConfig,
    support_threshold: int,
) -> Path:
    """Materialize one consistently support-filtered graph feature matrix."""

    output_path = config.graph_feature_path(support_threshold)
    graph_config = GraphAblationConfig(
        features_root=config.features_root,
        training_root=config.training_root,
        graph_root=config.graph_root,
        evaluation_root=config.cache_root,
        mode=config.mode,
        seed=config.seed,
        feature_version="temporal-features-v2",
        minimum_graph_support=support_threshold,
        duckdb_temp_directory=config.duckdb_temp_directory,
        duckdb_memory_limit=config.duckdb_memory_limit,
        duckdb_threads=config.duckdb_threads,
    )
    split_scope = "'train', 'validation'"
    if config.mode == "final":
        split_scope = "'train', 'validation', 'test'"
    query = graph_feature_query(graph_config)
    copy_query_to_parquet(
        connection,
        f"""
SELECT *
FROM ({query}) AS graph_features
WHERE source = 'mimiciv' AND split IN ({split_scope})
""",
        output_path,
    )
    return output_path


def patient_fold_sql(*, seed: int, fold_count: int, alias: str = "pcm") -> str:
    """Return a deterministic patient-level train-fold expression."""

    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    return (
        "CAST(HASH("
        f"{sql_string(str(seed))} || '|' || {alias}.patient_uid"
        f") % {int(fold_count)} AS INTEGER)"
    )


def _feature_select_sql(
    feature_spec: LearnedFeatureSpec,
    *,
    row_alias: str,
    stay_alias: str,
) -> str:
    columns: list[str] = []
    for name in (*feature_spec.stay_numeric, *feature_spec.stay_categorical):
        columns.append(
            f"{stay_alias}.{quote_identifier(name)} AS {quote_identifier(name)}"
        )
    for name in feature_spec.row_numeric:
        if name == "candidate_rank":
            continue
        columns.append(
            f"{row_alias}.{quote_identifier(name)} AS {quote_identifier(name)}"
        )
    return ",\n    ".join(columns)


def _positive_group_rows_sql(
    config: GateRecoveryConfig,
    *,
    split: str,
    sampled: bool,
) -> str:
    pcm = config.patient_condition_medication_path
    source_rows = f"""
SELECT
    source,
    split,
    patient_uid,
    stay_uid,
    ranking_group_id,
    index_condition_token,
    candidate_medication_token,
    candidate_rank,
    label_prescribed
FROM {parquet_scan(pcm)}
WHERE source = 'mimiciv' AND split = {sql_string(split)}
"""
    if not sampled:
        return f"""
WITH source_rows AS ({source_rows}),
positive_groups AS (
    SELECT ranking_group_id
    FROM source_rows
    GROUP BY ranking_group_id
    HAVING SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END) > 0
)
SELECT source_rows.*
FROM source_rows
INNER JOIN positive_groups USING (ranking_group_id)
"""

    return f"""
WITH source_rows AS ({source_rows}),
group_counts AS (
    SELECT
        ranking_group_id,
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END) AS positive_count,
        SUM(CASE WHEN NOT label_prescribed THEN 1 ELSE 0 END) AS negative_count
    FROM source_rows
    GROUP BY ranking_group_id
),
positive_rows AS (
    SELECT source_rows.*
    FROM source_rows
    INNER JOIN group_counts USING (ranking_group_id)
    WHERE label_prescribed AND positive_count > 0
),
ranked_negative_rows AS (
    SELECT
        source_rows.*,
        ROW_NUMBER() OVER (
            PARTITION BY source_rows.ranking_group_id
            ORDER BY HASH(
                {sql_string(str(config.seed))} || '|'
                || source_rows.ranking_group_id || '|'
                || source_rows.candidate_medication_token
            )
        ) AS negative_rank,
        group_counts.positive_count,
        group_counts.negative_count
    FROM source_rows
    INNER JOIN group_counts USING (ranking_group_id)
    WHERE NOT source_rows.label_prescribed AND group_counts.positive_count > 0
),
selected_negative_rows AS (
    SELECT * EXCLUDE (negative_rank, positive_count, negative_count)
    FROM ranked_negative_rows
    WHERE negative_rank <= LEAST(
        negative_count,
        GREATEST(10, 5 * positive_count)
    )
)
SELECT * FROM positive_rows
UNION ALL
SELECT * FROM selected_negative_rows
"""


def recovery_frame_query(
    config: GateRecoveryConfig,
    experiment: RecoveryExperiment,
    feature_spec: LearnedFeatureSpec,
    *,
    split: str,
    sampled: bool,
) -> str:
    """Return a bounded model-frame query for fitting or scoring."""

    candidates = _positive_group_rows_sql(config, split=split, sampled=sampled)
    psf = config.patient_stay_features_path
    use_graph = experiment.graph_family != "none"
    row_alias = "gf" if use_graph else "pcm"
    feature_sql = _feature_select_sql(
        feature_spec,
        row_alias=row_alias,
        stay_alias="psf",
    )
    graph_join = ""
    if use_graph:
        graph_path = config.graph_feature_path(experiment.graph_support_threshold)
        graph_join = f"""
INNER JOIN {parquet_scan(graph_path)} AS gf
    ON pcm.source = gf.source
    AND pcm.split = gf.split
    AND pcm.stay_uid = gf.stay_uid
    AND pcm.ranking_group_id = gf.ranking_group_id
    AND pcm.index_condition_token = gf.index_condition_token
    AND pcm.candidate_medication_token = gf.candidate_medication_token
"""
    extra_features = f",\n    {feature_sql}" if feature_sql else ""
    fold_sql = patient_fold_sql(
        seed=config.seed,
        fold_count=config.fold_count,
        alias="pcm",
    )
    return f"""
WITH pcm AS ({candidates})
SELECT
    pcm.source,
    pcm.split,
    pcm.ranking_group_id,
    pcm.index_condition_token,
    pcm.candidate_medication_token,
    pcm.candidate_rank,
    pcm.label_prescribed,
    {fold_sql} AS patient_fold_id
    {extra_features}
FROM pcm
INNER JOIN {parquet_scan(psf)} AS psf
    ON pcm.source = psf.source AND pcm.stay_uid = psf.stay_uid
{graph_join}
ORDER BY pcm.ranking_group_id, pcm.candidate_rank, pcm.candidate_medication_token
"""


def _prepare_matrix(
    frame: pd.DataFrame,
    feature_spec: LearnedFeatureSpec,
) -> pd.DataFrame:
    return frame.loc[:, list(feature_spec.model_columns)]


def _sort_for_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["ranking_group_id", "candidate_rank", "candidate_medication_token"],
        kind="mergesort",
    ).reset_index(drop=True)


def _ranking_dmatrix(
    frame: pd.DataFrame,
    feature_spec: LearnedFeatureSpec,
    preprocessor: Any,
) -> xgb.DMatrix:
    ordered = _sort_for_ranking(frame)
    transformed = preprocessor.transform(_prepare_matrix(ordered, feature_spec))
    qid = pd.factorize(ordered["ranking_group_id"], sort=False)[0]
    labels = ordered["label_prescribed"].astype(int).to_numpy()
    return xgb.DMatrix(transformed, label=labels, qid=qid)


def _score_ranker(
    model: xgb.Booster,
    preprocessor: Any,
    frame: pd.DataFrame,
    feature_spec: LearnedFeatureSpec,
) -> np.ndarray:
    transformed = preprocessor.transform(_prepare_matrix(frame, feature_spec))
    return model.predict(xgb.DMatrix(transformed))


def _fit_ranker_fold(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_spec: LearnedFeatureSpec,
    hyperparameters: RankerHyperparameters,
    *,
    seed: int,
) -> tuple[xgb.Booster, Any, int, np.ndarray]:
    preprocessor = build_preprocessor(feature_spec)
    preprocessor.fit(_prepare_matrix(train_frame, feature_spec))
    train_ordered = _sort_for_ranking(train_frame)
    validation_ordered = _sort_for_ranking(validation_frame)
    train_matrix = _ranking_dmatrix(train_ordered, feature_spec, preprocessor)
    validation_matrix = _ranking_dmatrix(
        validation_ordered,
        feature_spec,
        preprocessor,
    )
    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "ndcg_exp_gain": False,
        "tree_method": "hist",
        "max_depth": hyperparameters.max_depth,
        "eta": hyperparameters.learning_rate,
        "min_child_weight": hyperparameters.min_child_weight,
        "subsample": hyperparameters.subsample,
        "colsample_bytree": hyperparameters.colsample_bytree,
        "seed": seed,
    }
    model = xgb.train(
        params,
        train_matrix,
        num_boost_round=hyperparameters.max_boost_rounds,
        evals=[(validation_matrix, "train_fold_validation")],
        early_stopping_rounds=hyperparameters.early_stopping_rounds,
        verbose_eval=False,
    )
    best_rounds = int(getattr(model, "best_iteration", 0)) + 1
    raw_scores = _score_ranker(
        model,
        preprocessor,
        validation_frame,
        feature_spec,
    )
    return model, preprocessor, best_rounds, raw_scores


def _fit_binary_reference_fold(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_spec: LearnedFeatureSpec,
    *,
    seed: int,
) -> np.ndarray:
    preprocessor = build_preprocessor(feature_spec)
    preprocessor.fit(_prepare_matrix(train_frame, feature_spec))
    train_features = preprocessor.transform(_prepare_matrix(train_frame, feature_spec))
    labels = train_frame["label_prescribed"].astype(int).to_numpy()
    params = {**XGBOOST_V1_HYPERPARAMETERS, "seed": seed}
    rounds = int(params.pop("n_estimators"))
    model = xgb.train(
        params,
        xgb.DMatrix(train_features, label=labels),
        num_boost_round=rounds,
    )
    validation_features = preprocessor.transform(
        _prepare_matrix(validation_frame, feature_spec)
    )
    return model.predict(xgb.DMatrix(validation_features))


def ranking_metrics_at_k(
    frame: pd.DataFrame,
    *,
    score_column: str = "score",
    k: int = SELECTION_K,
) -> dict[str, float | int]:
    """Compute vectorized aggregate ranking metrics for positive groups."""

    if frame.empty:
        return {
            "positive_ranking_group_count": 0,
            "ndcg_at_k": 0.0,
            "mrr_at_k": 0.0,
            "hit_rate_at_k": 0.0,
        }
    ordered = frame.sort_values(
        [
            "ranking_group_id",
            score_column,
            "candidate_rank",
            "candidate_medication_token",
        ],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).copy()
    ordered["rank_position"] = ordered.groupby("ranking_group_id").cumcount() + 1
    ordered["label_int"] = ordered["label_prescribed"].astype(int)
    group_positive = ordered.groupby("ranking_group_id")["label_int"].sum()
    positive_ids = group_positive[group_positive > 0].index
    ordered = ordered[ordered["ranking_group_id"].isin(positive_ids)]
    top = ordered[ordered["rank_position"] <= k].copy()
    top["discounted_gain"] = top["label_int"] / np.log2(top["rank_position"] + 1)
    grouped = top.groupby("ranking_group_id").agg(
        hits=("label_int", "sum"),
        dcg=("discounted_gain", "sum"),
    )
    first_hits = (
        top[top["label_int"] == 1].groupby("ranking_group_id")["rank_position"].min()
    )
    group_sizes = ordered.groupby("ranking_group_id").size()
    ideal_lookup = {
        count: sum(1.0 / math.log2(rank + 1) for rank in range(1, count + 1))
        for count in range(1, k + 1)
    }
    ideal_dcg = group_positive.loc[positive_ids].clip(upper=k).map(ideal_lookup)
    grouped = grouped.reindex(positive_ids, fill_value=0.0)
    reciprocal = (1.0 / first_hits).reindex(positive_ids, fill_value=0.0)
    precision = grouped["hits"] / np.minimum(k, group_sizes.loc[positive_ids])
    recall = grouped["hits"] / group_positive.loc[positive_ids]
    return {
        "positive_ranking_group_count": int(len(positive_ids)),
        "precision_at_k": float(precision.mean()),
        "recall_at_k": float(recall.mean()),
        "hit_rate_at_k": float((grouped["hits"] > 0).mean()),
        "ndcg_at_k": float((grouped["dcg"] / ideal_dcg).mean()),
        "mrr_at_k": float(reciprocal.mean()),
    }


def _score_frame(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    output = frame.loc[:, list(METADATA_COLUMNS)].copy()
    output["score"] = scores
    return output


def cross_validate_ranker(
    train_sample: pd.DataFrame,
    train_scoring: pd.DataFrame,
    feature_spec: LearnedFeatureSpec,
    experiment: RecoveryExperiment,
    hyperparameters: RankerHyperparameters,
    *,
    seed: int,
    fold_count: int,
    capture_scores: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Evaluate one experiment exclusively on patient-held-out train folds."""

    fold_rows: list[dict[str, Any]] = []
    score_parts: list[pd.DataFrame] = []
    best_rounds: list[int] = []
    for fold_id in range(fold_count):
        fit_rows = train_sample[train_sample["patient_fold_id"] != fold_id]
        heldout_rows = train_scoring[train_scoring["patient_fold_id"] == fold_id]
        if fit_rows.empty or heldout_rows.empty:
            raise ValueError(f"patient fold {fold_id} has no fit or scoring rows")
        _model, _preprocessor, rounds, raw_scores = _fit_ranker_fold(
            fit_rows,
            heldout_rows,
            feature_spec,
            hyperparameters,
            seed=seed + fold_id,
        )
        scored = _score_frame(heldout_rows, raw_scores)
        metrics = ranking_metrics_at_k(scored, k=SELECTION_K)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "fit_row_count": int(len(fit_rows)),
                "scoring_row_count": int(len(heldout_rows)),
                "best_boost_rounds": rounds,
                **metrics,
            }
        )
        best_rounds.append(rounds)
        if capture_scores:
            score_parts.append(scored)
    result = {
        "experiment_name": experiment.name,
        "experiment": asdict(experiment),
        "hyperparameters": asdict(hyperparameters),
        "feature_count": len(feature_spec.model_columns),
        "mean_ndcg_at_10": float(np.mean([row["ndcg_at_k"] for row in fold_rows])),
        "mean_mrr_at_10": float(np.mean([row["mrr_at_k"] for row in fold_rows])),
        "mean_hit_rate_at_10": float(
            np.mean([row["hit_rate_at_k"] for row in fold_rows])
        ),
        "median_best_boost_rounds": int(median(best_rounds)),
        "folds": fold_rows,
    }
    scores = pd.concat(score_parts, ignore_index=True) if score_parts else None
    return result, scores


def cross_validate_binary_reference(
    train_sample: pd.DataFrame,
    train_scoring: pd.DataFrame,
    feature_spec: LearnedFeatureSpec,
    *,
    seed: int,
    fold_count: int,
) -> pd.DataFrame:
    """Produce fold-matched reference scores for fusion selection."""

    score_parts: list[pd.DataFrame] = []
    for fold_id in range(fold_count):
        fit_rows = train_sample[train_sample["patient_fold_id"] != fold_id]
        heldout_rows = train_scoring[train_scoring["patient_fold_id"] == fold_id]
        scores = _fit_binary_reference_fold(
            fit_rows,
            heldout_rows,
            feature_spec,
            seed=seed + fold_id,
        )
        score_parts.append(_score_frame(heldout_rows, scores))
    return pd.concat(score_parts, ignore_index=True)


def select_best_result(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Select by ranking metrics, then simplicity and stable name."""

    if not results:
        raise ValueError("no cross-validation results were produced")
    return sorted(
        results,
        key=lambda row: (
            -float(row["mean_ndcg_at_10"]),
            -float(row["mean_mrr_at_10"]),
            -float(row["mean_hit_rate_at_10"]),
            int(row["feature_count"]),
            str(row["experiment_name"]),
            str(row["hyperparameters"]),
        ),
    )[0]


def _rank_normalized(frame: pd.DataFrame, score_column: str) -> pd.Series:
    ordered = frame.sort_values(
        [
            "ranking_group_id",
            score_column,
            "candidate_rank",
            "candidate_medication_token",
        ],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).copy()
    ordered["_rank"] = ordered.groupby("ranking_group_id").cumcount()
    ordered["_size"] = ordered.groupby("ranking_group_id")["_rank"].transform("size")
    ordered["_normalized"] = np.where(
        ordered["_size"] > 1,
        1.0 - ordered["_rank"] / (ordered["_size"] - 1),
        1.0,
    )
    return ordered["_normalized"].reindex(frame.index)


def select_oof_fusion_weight(
    candidate_scores: pd.DataFrame,
    reference_scores: pd.DataFrame,
) -> dict[str, Any]:
    """Choose a candidate weight from train out-of-fold scores only."""

    keys = list(METADATA_COLUMNS)
    paired = candidate_scores.merge(
        reference_scores,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_candidate", "_reference"),
    )
    if paired.empty:
        raise ValueError("no paired out-of-fold scores were available for fusion")
    paired["candidate_normalized"] = _rank_normalized(paired, "score_candidate")
    paired["reference_normalized"] = _rank_normalized(paired, "score_reference")
    candidates: list[dict[str, Any]] = []
    for weight in FUSION_WEIGHT_GRID:
        paired["score"] = (
            weight * paired["candidate_normalized"]
            + (1.0 - weight) * paired["reference_normalized"]
        )
        metrics = ranking_metrics_at_k(paired, k=SELECTION_K)
        candidates.append({"candidate_weight": weight, **metrics})
    selected = sorted(
        candidates,
        key=lambda row: (
            -float(row["ndcg_at_k"]),
            -float(row["mrr_at_k"]),
            -float(row["hit_rate_at_k"]),
            -float(row["candidate_weight"]),
        ),
    )[0]
    return {
        "status": "selected_from_mimic_train_oof",
        "selected_candidate_weight": float(selected["candidate_weight"]),
        "selection_k": SELECTION_K,
        "selected_metrics": selected,
        "candidates": candidates,
    }


def _load_frame(
    connection: duckdb.DuckDBPyConnection,
    query: str,
) -> pd.DataFrame:
    frame = connection.execute(query).fetchdf()
    if frame.empty:
        raise ValueError("recovery model frame is empty")
    return frame


def _fit_final_ranker(
    train_frame: pd.DataFrame,
    feature_spec: LearnedFeatureSpec,
    hyperparameters: RankerHyperparameters,
    *,
    boost_rounds: int,
    seed: int,
) -> tuple[xgb.Booster, Any]:
    preprocessor = build_preprocessor(feature_spec)
    preprocessor.fit(_prepare_matrix(train_frame, feature_spec))
    ordered = _sort_for_ranking(train_frame)
    matrix = _ranking_dmatrix(ordered, feature_spec, preprocessor)
    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "ndcg_exp_gain": False,
        "tree_method": "hist",
        "max_depth": hyperparameters.max_depth,
        "eta": hyperparameters.learning_rate,
        "min_child_weight": hyperparameters.min_child_weight,
        "subsample": hyperparameters.subsample,
        "colsample_bytree": hyperparameters.colsample_bytree,
        "seed": seed,
    }
    model = xgb.train(params, matrix, num_boost_round=max(1, boost_rounds))
    return model, preprocessor


def _write_score_frame(
    frame: pd.DataFrame,
    *,
    output_path: Path,
    baseline_name: str,
    seed: int,
    generated_at: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.loc[:, list(METADATA_COLUMNS)].copy()
    output["baseline_name"] = baseline_name
    output["score"] = frame["score"].astype(float)
    output["seed"] = seed
    output["baseline_version"] = EXPERIMENT_VERSION
    output["evaluation_version"] = SCHEMA_VERSION
    output["generated_at"] = generated_at
    ordered_columns = (
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
    pq.write_table(
        pa.Table.from_pandas(
            output.loc[:, list(ordered_columns)], preserve_index=False
        ),
        output_path,
    )
    return int(len(output))


def _combine_with_reference(
    connection: duckdb.DuckDBPyConnection,
    config: GateRecoveryConfig,
    *,
    candidate_path: Path,
    split: str,
) -> int:
    config.score_output_path.parent.mkdir(parents=True, exist_ok=True)
    query = f"""
SELECT * FROM {parquet_scan(candidate_path)}
UNION ALL
SELECT * FROM {parquet_scan(config.reference_scores_path)}
WHERE source = 'mimiciv'
    AND split = {sql_string(split)}
    AND baseline_name = 'xgboost_frozen_reference'
"""
    return copy_query_to_parquet(connection, query, config.score_output_path)


def _reference_frame(
    connection: duckdb.DuckDBPyConnection,
    config: GateRecoveryConfig,
    *,
    split: str,
) -> pd.DataFrame:
    return connection.execute(
        f"""
SELECT
    source,
    split,
    ranking_group_id,
    index_condition_token,
    candidate_medication_token,
    candidate_rank,
    label_prescribed,
    score
FROM {parquet_scan(config.reference_scores_path)}
WHERE source = 'mimiciv'
    AND split = {sql_string(split)}
    AND baseline_name = 'xgboost_frozen_reference'
"""
    ).fetchdf()


def _apply_fusion(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    candidate_weight: float,
) -> pd.DataFrame:
    paired = candidate.merge(
        reference,
        on=list(METADATA_COLUMNS),
        how="inner",
        validate="one_to_one",
        suffixes=("_candidate", "_reference"),
    )
    if paired.empty:
        raise ValueError("no paired candidate/reference scores were available")
    paired["candidate_normalized"] = _rank_normalized(paired, "score_candidate")
    paired["reference_normalized"] = _rank_normalized(paired, "score_reference")
    paired["score"] = (
        candidate_weight * paired["candidate_normalized"]
        + (1.0 - candidate_weight) * paired["reference_normalized"]
    )
    return paired.loc[:, [*METADATA_COLUMNS, "score"]]


def _evaluation_manifest(
    connection: duckdb.DuckDBPyConnection,
    config: GateRecoveryConfig,
    *,
    generated_at: str,
    contract_digest: str,
    selected_experiment: RecoveryExperiment,
    hyperparameters: RankerHyperparameters,
    cv_results: Sequence[dict[str, Any]],
    fusion: dict[str, Any],
    boost_rounds: int,
    score_row_count: int,
    selected_baseline_name: str,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "generated_at": generated_at,
        "mode": config.mode,
        "contract_digest": contract_digest,
        "seed": config.seed,
        "fold_count": config.fold_count,
        "selection_split": "mimiciv_train_patient_folds",
        "validation_policy": "single_locked_candidate_evaluation",
        "reproducibility": {
            "git_revision": _git_revision(),
            "package_lock_digest": _package_lock_digest(),
            "model_version": EXPERIMENT_VERSION,
        },
        "frozen_artifacts": _frozen_model_artifacts(config),
        "cohort_window_semantics": {
            "development_source": "mimiciv",
            "feature_cutoff_hours_from_admit": 24,
            "label_window": "(24h, 48h]",
            "zero_positive_groups_used_for_fitting": False,
        },
        "selected_baseline_name": selected_baseline_name,
        "selected_experiment": asdict(selected_experiment),
        "selected_hyperparameters": asdict(hyperparameters),
        "training_iterations": {
            "boost_rounds": boost_rounds,
            "maximum_boost_rounds": hyperparameters.max_boost_rounds,
            "early_stopping_rounds": hyperparameters.early_stopping_rounds,
        },
        "fusion": fusion,
        "cross_validation_results": list(cv_results),
        "artifacts": {"baseline_scores": str(config.score_output_path)},
        "tables": [
            {
                "table_name": "baseline_scores",
                "status": "completed",
                "row_count": score_row_count,
            }
        ],
        "label_caveat": (
            "Labels are observed historical prescriptions in (24h, 48h]. "
            "Unobserved catalog candidates are weak observational negatives."
        ),
        "clinical_claim_boundary": (
            "This is an offline research gate, not a validated medication recommender."
        ),
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "local_scores_contain_restricted_group_keys": True,
            "local_scores_are_ignored_and_protected": True,
        },
    }
    metric_config = BaselineEvaluationConfig(
        features_root=config.features_root,
        training_root=config.training_root,
        evaluation_root=config.score_root,
        top_k=config.top_k,
        mode=config.mode,
        frozen_selection=config.frozen_selection,
        seed=config.seed,
        feature_version="temporal-features-v2",
        duckdb_temp_directory=config.duckdb_temp_directory,
        duckdb_memory_limit=config.duckdb_memory_limit,
        duckdb_threads=config.duckdb_threads,
    )
    append_metric_summaries(connection, metric_config, manifest)
    manifest["ranking_group_coverage"] = _ranking_group_coverage(
        connection,
        config,
    )
    return manifest


def _ranking_group_coverage(
    connection: duckdb.DuckDBPyConnection,
    config: GateRecoveryConfig,
) -> list[dict[str, Any]]:
    """Report aggregate ranking-group exclusions without exposing group keys."""

    split_scope = "'train', 'validation'"
    if config.mode == "final":
        split_scope = "'test'"
    return [
        dict(
            zip(
                (
                    "source",
                    "split",
                    "ranking_group_count",
                    "positive_group_count",
                    "zero_positive_group_count",
                ),
                row,
                strict=True,
            )
        )
        for row in connection.execute(
            f"""
WITH group_labels AS (
    SELECT
        source,
        split,
        ranking_group_id,
        MAX(CASE WHEN label_prescribed THEN 1 ELSE 0 END) AS has_positive
    FROM {parquet_scan(config.patient_condition_medication_path)}
    WHERE source = 'mimiciv' AND split IN ({split_scope})
    GROUP BY source, split, ranking_group_id
)
SELECT
    source,
    split,
    COUNT(*) AS ranking_group_count,
    SUM(has_positive) AS positive_group_count,
    SUM(CASE WHEN has_positive = 0 THEN 1 ELSE 0 END) AS zero_positive_group_count
FROM group_labels
GROUP BY source, split
ORDER BY source, split
"""
        ).fetchall()
    ]


def _metric_row(
    report: dict[str, Any],
    *,
    baseline_name: str,
    split: str,
    k: int = SELECTION_K,
) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in report.get("ranking_metrics", [])
            if row.get("baseline_name") == baseline_name
            and row.get("source") == "mimiciv"
            and row.get("split") == split
            and int(row.get("k", -1)) == k
        ),
        None,
    )


def gate_decision(
    *,
    candidate: dict[str, Any],
    reference: dict[str, Any],
    minimum_ndcg_lift: float = MINIMUM_NDCG_LIFT,
    maximum_secondary_drop: float = MAXIMUM_SECONDARY_DROP,
) -> dict[str, Any]:
    """Apply the neural-readiness gate against Phase 8 P0 frozen XGBoost."""

    ndcg_delta = float(candidate["ndcg_at_k"]) - float(reference["ndcg_at_k"])
    mrr_delta = float(candidate["mrr_at_k"]) - float(reference["mrr_at_k"])
    hit_delta = float(candidate["hit_rate_at_k"]) - float(reference["hit_rate_at_k"])
    passed = (
        ndcg_delta >= minimum_ndcg_lift
        and mrr_delta >= -maximum_secondary_drop
        and hit_delta >= -maximum_secondary_drop
    )
    return {
        "decision": "promote_to_neural_prototype"
        if passed
        else "retain_frozen_xgboost",
        "neural_training_authorized": passed,
        "primary_metric": "mimic_validation_ndcg_at_10",
        "reference_ndcg_at_10": float(reference["ndcg_at_k"]),
        "required_candidate_ndcg_at_10": float(reference["ndcg_at_k"])
        + minimum_ndcg_lift,
        "candidate_ndcg_at_10": float(candidate["ndcg_at_k"]),
        "ndcg_delta": ndcg_delta,
        "mrr_delta": mrr_delta,
        "hit_rate_delta": hit_delta,
        "minimum_ndcg_lift": minimum_ndcg_lift,
        "maximum_secondary_drop": maximum_secondary_drop,
    }


def _selection_report(
    config: GateRecoveryConfig,
    evaluation: dict[str, Any],
    *,
    generated_at: str,
    selected_baseline_name: str,
) -> dict[str, Any]:
    candidate = _metric_row(
        evaluation,
        baseline_name=selected_baseline_name,
        split="validation",
    )
    reference = _metric_row(
        evaluation,
        baseline_name="xgboost_frozen_reference",
        split="validation",
    )
    if candidate is None or reference is None:
        return {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "status": "failed_missing_validation_metrics",
            "generated_at": generated_at,
            "neural_training_authorized": False,
            "decision": "retain_frozen_xgboost",
        }
    decision = gate_decision(candidate=candidate, reference=reference)
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "frozen",
        "generated_at": generated_at,
        "selected_experiment": selected_baseline_name,
        "reference_experiment": "xgboost_frozen_reference",
        "selection_basis": {
            "model_selection": "mimic_train_patient_folds",
            "gate_split": "mimiciv_validation",
            "k": SELECTION_K,
            "candidate_metrics": candidate,
            "reference_metrics": reference,
        },
        **decision,
        "contract_digest": evaluation["contract_digest"],
        "reproducibility": evaluation["reproducibility"],
        "seed": config.seed,
        "fold_count": config.fold_count,
        "feature_configuration": evaluation["selected_experiment"],
        "hyperparameters": evaluation["selected_hyperparameters"],
        "training_iterations": evaluation["training_iterations"],
        "calibration": {
            "status": "not_applied_to_rank_selection",
            "ranking_input": "xgboost_margin_or_oof_rank_normalized_fusion",
        },
        "cohort_window_semantics": evaluation["cohort_window_semantics"],
        "ranking_group_coverage": evaluation["ranking_group_coverage"],
        "frozen_artifacts": evaluation["frozen_artifacts"],
        "clinical_claim_boundary": evaluation["clinical_claim_boundary"],
        "label_caveat": evaluation["label_caveat"],
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
        },
    }


def _experiment_from_dict(payload: dict[str, Any]) -> RecoveryExperiment:
    return RecoveryExperiment(
        condition_cap=int(payload["condition_cap"]),
        graph_support_threshold=int(payload["graph_support_threshold"]),
        graph_family=str(payload["graph_family"]),
        include_candidate_rank=bool(payload["include_candidate_rank"]),
    )


def _hyperparameters_from_dict(payload: dict[str, Any]) -> RankerHyperparameters:
    return RankerHyperparameters(**payload)


def _run_development(
    connection: duckdb.DuckDBPyConnection,
    config: GateRecoveryConfig,
    *,
    generated_at: str,
    contract_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for threshold in GRAPH_SUPPORT_THRESHOLDS:
        materialize_graph_features(connection, config, threshold)

    results: list[dict[str, Any]] = []
    frame_cache: dict[tuple[str, bool], pd.DataFrame] = {}
    cached_experiment_name: str | None = None

    def frames(
        experiment: RecoveryExperiment,
        feature_spec: LearnedFeatureSpec,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        nonlocal cached_experiment_name
        if cached_experiment_name != experiment.name:
            frame_cache.clear()
            cached_experiment_name = experiment.name
        key = (experiment.name, True)
        if key not in frame_cache:
            frame_cache[key] = _load_frame(
                connection,
                recovery_frame_query(
                    config,
                    experiment,
                    feature_spec,
                    split="train",
                    sampled=True,
                ),
            )
        scoring_key = (experiment.name, False)
        if scoring_key not in frame_cache:
            frame_cache[scoring_key] = _load_frame(
                connection,
                recovery_frame_query(
                    config,
                    experiment,
                    feature_spec,
                    split="train",
                    sampled=False,
                ),
            )
        return frame_cache[key], frame_cache[scoring_key]

    def evaluate(
        experiment: RecoveryExperiment,
        hyperparameters: RankerHyperparameters,
    ) -> dict[str, Any]:
        feature_spec = resolve_recovery_feature_spec(connection, config, experiment)
        train_sample, train_scoring = frames(experiment, feature_spec)
        result, _scores = cross_validate_ranker(
            train_sample,
            train_scoring,
            feature_spec,
            experiment,
            hyperparameters,
            seed=config.seed,
            fold_count=config.fold_count,
        )
        results.append(result)
        return result

    condition_results = [
        evaluate(
            RecoveryExperiment(cap, 1, "none", True),
            SCREENING_HYPERPARAMETERS,
        )
        for cap in CONDITION_CAPS
    ]
    best_condition = int(
        select_best_result(condition_results)["experiment"]["condition_cap"]
    )

    support_results = [
        evaluate(
            RecoveryExperiment(best_condition, threshold, "all", True),
            SCREENING_HYPERPARAMETERS,
        )
        for threshold in GRAPH_SUPPORT_THRESHOLDS
    ]
    best_support = int(
        select_best_result(support_results)["experiment"]["graph_support_threshold"]
    )

    family_results = [
        evaluate(
            RecoveryExperiment(best_condition, best_support, family, True),
            SCREENING_HYPERPARAMETERS,
        )
        for family in ("direct", "context", "all")
    ]
    best_family = str(select_best_result(family_results)["experiment"]["graph_family"])

    rank_results = [
        evaluate(
            RecoveryExperiment(best_condition, best_support, best_family, include_rank),
            SCREENING_HYPERPARAMETERS,
        )
        for include_rank in (True, False)
    ]
    selected_features = _experiment_from_dict(
        select_best_result(rank_results)["experiment"]
    )

    hyper_results = [
        evaluate(selected_features, hyperparameters)
        for hyperparameters in HYPERPARAMETER_GRID
    ]
    best_hyper_result = select_best_result(hyper_results)
    selected_hyperparameters = _hyperparameters_from_dict(
        best_hyper_result["hyperparameters"]
    )

    selected_spec = resolve_recovery_feature_spec(
        connection,
        config,
        selected_features,
    )
    train_sample, train_scoring = frames(selected_features, selected_spec)
    final_cv, candidate_oof = cross_validate_ranker(
        train_sample,
        train_scoring,
        selected_spec,
        selected_features,
        selected_hyperparameters,
        seed=config.seed,
        fold_count=config.fold_count,
        capture_scores=True,
    )
    assert candidate_oof is not None

    reference_experiment = RecoveryExperiment(40, 1, "none", True)
    reference_spec = resolve_recovery_feature_spec(
        connection,
        config,
        reference_experiment,
    )
    reference_sample, reference_scoring = frames(reference_experiment, reference_spec)
    reference_oof = cross_validate_binary_reference(
        reference_sample,
        reference_scoring,
        reference_spec,
        seed=config.seed,
        fold_count=config.fold_count,
    )
    fusion = select_oof_fusion_weight(candidate_oof, reference_oof)
    candidate_oof_metrics = ranking_metrics_at_k(candidate_oof)
    fusion_metrics = fusion["selected_metrics"]
    use_fusion = (
        float(fusion_metrics["ndcg_at_k"]),
        float(fusion_metrics["mrr_at_k"]),
        float(fusion_metrics["hit_rate_at_k"]),
    ) > (
        float(candidate_oof_metrics["ndcg_at_k"]),
        float(candidate_oof_metrics["mrr_at_k"]),
        float(candidate_oof_metrics["hit_rate_at_k"]),
    )
    fusion["selected_variant"] = "late_fusion" if use_fusion else "ranker_only"

    boost_rounds = int(final_cv["median_best_boost_rounds"])
    final_model, final_preprocessor = _fit_final_ranker(
        train_sample,
        selected_spec,
        selected_hyperparameters,
        boost_rounds=boost_rounds,
        seed=config.seed,
    )
    config.models_root.mkdir(parents=True, exist_ok=True)
    final_model.save_model(config.selected_model_path)
    joblib.dump(final_preprocessor, config.selected_preprocessor_path)
    selected_payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "contract_digest": contract_digest,
        "experiment": asdict(selected_features),
        "hyperparameters": asdict(selected_hyperparameters),
        "feature_spec": {
            "stay_numeric": list(selected_spec.stay_numeric),
            "stay_categorical": list(selected_spec.stay_categorical),
            "row_numeric": list(selected_spec.row_numeric),
            "row_categorical": list(selected_spec.row_categorical),
        },
        "boost_rounds": boost_rounds,
        "fusion": fusion,
        "seed": config.seed,
        "reproducibility": {
            "git_revision": _git_revision(),
            "package_lock_digest": _package_lock_digest(),
        },
    }
    write_json(config.selected_config_path, selected_payload)

    validation_frame = _load_frame(
        connection,
        recovery_frame_query(
            config,
            selected_features,
            selected_spec,
            split="validation",
            sampled=False,
        ),
    )
    raw_scores = _score_ranker(
        final_model,
        final_preprocessor,
        validation_frame,
        selected_spec,
    )
    candidate_frame = _score_frame(
        validation_frame,
        1.0 / (1.0 + np.exp(-np.clip(raw_scores, -30.0, 30.0))),
    )
    selected_baseline_name = "xgboost_rank_ndcg_gate_recovery"
    if use_fusion:
        candidate_frame = _apply_fusion(
            candidate_frame,
            _reference_frame(connection, config, split="validation"),
            candidate_weight=float(fusion["selected_candidate_weight"]),
        )
        selected_baseline_name = "xgboost_rank_ndcg_oof_late_fusion"

    candidate_path = config.score_root / "_candidate_scores.parquet"
    candidate_row_count = _write_score_frame(
        candidate_frame,
        output_path=candidate_path,
        baseline_name=selected_baseline_name,
        seed=config.seed,
        generated_at=generated_at,
    )
    combined_count = _combine_with_reference(
        connection,
        config,
        candidate_path=candidate_path,
        split="validation",
    )
    evaluation = _evaluation_manifest(
        connection,
        config,
        generated_at=generated_at,
        contract_digest=contract_digest,
        selected_experiment=selected_features,
        hyperparameters=selected_hyperparameters,
        cv_results=[*results, final_cv],
        fusion=fusion,
        boost_rounds=boost_rounds,
        score_row_count=combined_count,
        selected_baseline_name=selected_baseline_name,
    )
    evaluation["candidate_score_row_count"] = candidate_row_count
    selection = _selection_report(
        config,
        evaluation,
        generated_at=generated_at,
        selected_baseline_name=selected_baseline_name,
    )
    return evaluation, selection


def _run_final(
    connection: duckdb.DuckDBPyConnection,
    config: GateRecoveryConfig,
    *,
    generated_at: str,
    contract_digest: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = load_json(config.selected_config_path)
    if selected.get("contract_digest") != contract_digest:
        raise ValueError("selected recovery artifacts do not match the contract lock")
    experiment = _experiment_from_dict(selected["experiment"])
    hyperparameters = _hyperparameters_from_dict(selected["hyperparameters"])
    materialize_graph_features(connection, config, experiment.graph_support_threshold)
    feature_spec = LearnedFeatureSpec(
        stay_numeric=tuple(selected["feature_spec"]["stay_numeric"]),
        stay_categorical=tuple(selected["feature_spec"]["stay_categorical"]),
        row_numeric=tuple(selected["feature_spec"]["row_numeric"]),
        row_categorical=tuple(selected["feature_spec"]["row_categorical"]),
    )
    model = xgb.Booster()
    model.load_model(config.selected_model_path)
    preprocessor = joblib.load(config.selected_preprocessor_path)
    test_frame = _load_frame(
        connection,
        recovery_frame_query(
            config,
            experiment,
            feature_spec,
            split="test",
            sampled=False,
        ),
    )
    raw_scores = _score_ranker(model, preprocessor, test_frame, feature_spec)
    candidate_frame = _score_frame(
        test_frame,
        1.0 / (1.0 + np.exp(-np.clip(raw_scores, -30.0, 30.0))),
    )
    fusion = selected["fusion"]
    selected_baseline_name = "xgboost_rank_ndcg_gate_recovery"
    if fusion.get("selected_variant") == "late_fusion":
        candidate_frame = _apply_fusion(
            candidate_frame,
            _reference_frame(connection, config, split="test"),
            candidate_weight=float(fusion["selected_candidate_weight"]),
        )
        selected_baseline_name = "xgboost_rank_ndcg_oof_late_fusion"
    candidate_path = config.score_root / "_candidate_scores.parquet"
    _write_score_frame(
        candidate_frame,
        output_path=candidate_path,
        baseline_name=selected_baseline_name,
        seed=config.seed,
        generated_at=generated_at,
    )
    combined_count = _combine_with_reference(
        connection,
        config,
        candidate_path=candidate_path,
        split="test",
    )
    evaluation = _evaluation_manifest(
        connection,
        config,
        generated_at=generated_at,
        contract_digest=contract_digest,
        selected_experiment=experiment,
        hyperparameters=hyperparameters,
        cv_results=[],
        fusion=fusion,
        boost_rounds=int(selected["boost_rounds"]),
        score_row_count=combined_count,
        selected_baseline_name=selected_baseline_name,
    )
    selection = load_json(config.selection_report_path)
    return evaluation, selection


def build_gate_recovery(
    config: GateRecoveryConfig = GateRecoveryConfig(),
) -> dict[str, Any]:
    """Run development selection or frozen final scoring."""

    generated_at = datetime.now(UTC).isoformat()
    errors = preflight_errors(config)
    if errors:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked_preflight",
            "generated_at": generated_at,
            "mode": config.mode,
            "errors": errors,
            "data_safety": {
                "report_contains_patient_rows": False,
                "report_contains_row_samples": False,
            },
        }
        write_json(config.evaluation_report_path, report)
        return report

    contract = load_json(config.contract_lock_path)
    contract_digest = str(contract["contract_digest"])
    config.evaluation_root.mkdir(parents=True, exist_ok=True)
    try:
        with duckdb.connect(database=":memory:") as connection:
            configure_connection(config, connection)
            if config.mode == "development":
                evaluation, selection = _run_development(
                    connection,
                    config,
                    generated_at=generated_at,
                    contract_digest=contract_digest,
                )
                write_json(config.selection_report_path, selection)
            else:
                evaluation, selection = _run_final(
                    connection,
                    config,
                    generated_at=generated_at,
                    contract_digest=contract_digest,
                )
                evaluation["frozen_development_selection"] = {
                    "decision": selection.get("decision"),
                    "selected_experiment": selection.get("selected_experiment"),
                    "contract_digest": selection.get("contract_digest"),
                }
    except Exception as error:
        evaluation = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "generated_at": generated_at,
            "mode": config.mode,
            "reason": safe_error_message(error),
            "contract_digest": contract_digest,
            "data_safety": {
                "report_contains_patient_rows": False,
                "report_contains_row_samples": False,
            },
        }
    write_json(config.evaluation_report_path, evaluation)
    return evaluation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 8 P0 rank-aware structured gate recovery.",
    )
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument(
        "--reference-scores", type=Path, default=DEFAULT_REFERENCE_SCORES
    )
    parser.add_argument(
        "--reference-selection",
        type=Path,
        default=DEFAULT_REFERENCE_SELECTION,
    )
    parser.add_argument("--contract-lock", type=Path, default=DEFAULT_CONTRACT_LOCK)
    parser.add_argument(
        "--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST
    )
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument(
        "--evaluation-report", type=Path, default=DEFAULT_EVALUATION_REPORT
    )
    parser.add_argument(
        "--selection-report", type=Path, default=DEFAULT_SELECTION_REPORT
    )
    parser.add_argument(
        "--mode", choices=("development", "final"), default="development"
    )
    parser.add_argument("--frozen-selection", action="store_true")
    parser.add_argument("--top-k", default="1,3,5,10")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--fold-count", type=int, default=FOLD_COUNT)
    parser.add_argument("--duckdb-temp-dir", type=Path, default=DUCKDB_TEMP_DIR)
    parser.add_argument("--duckdb-memory-limit", default=DUCKDB_MEMORY_LIMIT)
    parser.add_argument("--duckdb-threads", type=int, default=DUCKDB_THREADS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = GateRecoveryConfig(
        features_root=args.features_root,
        training_root=args.training_root,
        graph_root=args.graph_root,
        reference_scores_path=args.reference_scores,
        reference_selection_path=args.reference_selection,
        contract_lock_path=args.contract_lock,
        feature_manifest_path=args.feature_manifest,
        evaluation_root=args.evaluation_root,
        evaluation_report_path=args.evaluation_report,
        selection_report_path=args.selection_report,
        mode=args.mode,
        frozen_selection=args.frozen_selection,
        top_k=parse_top_k(args.top_k),
        seed=args.seed,
        fold_count=args.fold_count,
        duckdb_temp_directory=args.duckdb_temp_dir,
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_threads=args.duckdb_threads,
    )
    report = build_gate_recovery(config)
    print(
        "Wrote Phase 8 P0 gate recovery report: "
        f"status={report['status']}, mode={report['mode']}"
    )
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
