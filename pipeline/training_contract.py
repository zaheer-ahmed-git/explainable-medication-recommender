"""Audit and lock the Phase 8 P0 model-training input contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import duckdb

from pipeline.config import (
    DATASET_ROOT,
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_TEMP_DIR,
    DUCKDB_THREADS,
    REPORTS_ROOT,
)
from pipeline.extract_utils import (
    configure_duckdb_connection,
    parquet_scan,
    safe_error_message,
)
from pipeline.features import fetch_dict_rows


SCHEMA_VERSION = "phase8-p0-training-contract-lock-v1"
DEFAULT_PACKAGE_MANIFEST = REPORTS_ROOT / "phase8_p0_model_ready_manifest.json"
DEFAULT_DATA_DICTIONARY = REPORTS_ROOT / "phase8_p0_model_ready_data_dictionary.json"
DEFAULT_PRIMARY_TRAINING_MANIFEST = (
    REPORTS_ROOT / "phase8_p0_training_table_manifest.json"
)
DEFAULT_SENSITIVITY_TRAINING_MANIFEST = (
    REPORTS_ROOT / "phase8_p0_atc3_training_table_manifest.json"
)
DEFAULT_FEATURE_MANIFEST = REPORTS_ROOT / "phase8_p0_milestone6_feature_manifest.json"
DEFAULT_PREPROCESSING_MANIFEST = REPORTS_ROOT / "phase8_p0_preprocessing_manifest.json"
DEFAULT_SUBGRAPHS_MANIFEST = REPORTS_ROOT / "phase8_p0_patient_subgraphs_manifest.json"
DEFAULT_GRAPH_SUITABILITY_REPORT = (
    REPORTS_ROOT / "phase8_p0_milestone8_graph_suitability.json"
)
DEFAULT_OUTPUT_PATH = REPORTS_ROOT / "phase8_p0_training_contract_lock.json"

PINNED_VERSIONS = {
    "feature_version": "temporal-features-v2",
    "graph_version": "graph-suitability-v1",
    "label_version": "observed-medication-label-v1",
    "split_version": "patient-split-v1",
}

IDENTIFIER_COLUMNS = frozenset(
    {
        "patient_uid",
        "encounter_uid",
        "stay_uid",
        "source_patient_id",
        "source_encounter_id",
        "source_stay_id",
        "source_event_id",
        "ranking_group_id",
        "subgraph_id",
    }
)
RAW_OR_UNSAFE_COLUMNS = frozenset(
    {
        *IDENTIFIER_COLUMNS,
        "source_text",
        "value_text",
        "source_code",
        "hospital_id",
        "ward_id",
        "eligibility_status",
        "primary_training_eligible",
        "stay_end_hours_from_admit",
        "label_first_observed_hours_from_admit",
        "label_event_count",
    }
)
UNSAFE_COLUMN_SUFFIXES = ("_uid", "_id", "_code", "_text")
OUTCOME_COLUMN_TOKENS = (
    "death",
    "discharge",
    "hospital_mortality",
    "length_of_stay",
    "outcome",
)
PROVENANCE_COLUMNS = frozenset(
    {
        "source_version",
        "cohort_version",
        "extraction_version",
        "mapping_version",
        "harmonization_version",
        "feature_version",
        "graph_version",
        "label_version",
        "split_version",
        "generated_at",
    }
)
LOW_CARDINALITY_STAY_CATEGORICAL = frozenset(
    {
        "sex",
        "race_or_ethnicity",
        "admission_type",
        "admission_source",
        "unit_type",
        "last_unit_type",
        "stay_type",
    }
)

EXPLICIT_PROJECTIONS = {
    "event_sequences": (
        "event_sequence_position",
        "event_type",
        "event_time_hours_from_admit",
        "event_token",
        "value_numeric",
        "normalized_unit",
    ),
    "patient_condition_medication": (
        "index_condition_token",
        "candidate_medication_token",
        "candidate_rank",
    ),
    "graph_edges": (
        "src_id",
        "dst_id",
        "src_type",
        "dst_type",
        "relation_type",
        "support_count",
    ),
    "subgraph_nodes": (
        "node_index",
        "node_id",
        "node_type",
        "node_role",
        "observed_predecision",
        "in_train_graph",
        "cold_start",
    ),
    "subgraph_edges": (
        "src_node_index",
        "dst_node_index",
        "src_type",
        "dst_type",
        "relation_type",
        "support_count",
    ),
    "subgraph_candidates": (
        "index_condition_token",
        "candidate_medication_token",
        "candidate_node_index",
        "candidate_rank",
        "in_train_graph",
        "cold_start",
    ),
}

REQUIRED_SCHEMA_COLUMNS = {
    "patient_stay_features": {
        "source",
        "patient_uid",
        "stay_uid",
        "split",
        "prediction_time_hours_from_admit",
        "label_window_end_hours_from_admit",
        "feature_version",
        "split_version",
    },
    "event_sequences": {
        "source",
        "patient_uid",
        "stay_uid",
        "split",
        "event_sequence_position",
        "event_type",
        "event_time_hours_from_admit",
        "event_token",
        "value_numeric",
        "normalized_unit",
        "feature_version",
    },
    "patient_condition_medication": {
        "source",
        "patient_uid",
        "stay_uid",
        "split",
        "ranking_group_id",
        "index_condition_token",
        "candidate_medication_token",
        "candidate_rank",
        "label_prescribed",
        "label_first_observed_hours_from_admit",
        "prediction_time_hours_from_admit",
        "label_window_end_hours_from_admit",
        "feature_version",
        "label_version",
        "split_version",
    },
    "split_manifest": {"source", "patient_uid", "split", "split_version"},
    "candidate_catalog": {
        "index_condition_token",
        "candidate_medication_token",
        "candidate_rank",
        "feature_version",
        "label_version",
        "split_version",
    },
    "graph_edges": {
        "src_id",
        "dst_id",
        "src_type",
        "dst_type",
        "relation_type",
        "support_count",
        "fit_source",
        "fit_split",
        "graph_version",
        "feature_version",
        "label_version",
        "split_version",
    },
    "subgraph_index": {
        "source",
        "split",
        "stay_uid",
        "subgraph_id",
        "index_condition_token",
        "node_count",
        "edge_count",
        "candidate_count",
        "positive_count",
        "feature_version",
        "graph_version",
        "label_version",
        "split_version",
    },
    "subgraph_nodes": {
        "source",
        "split",
        "stay_uid",
        "subgraph_id",
        "node_index",
        "node_id",
        "node_type",
        "node_role",
        "observed_predecision",
        "in_train_graph",
        "cold_start",
        "feature_version",
        "graph_version",
        "split_version",
    },
    "subgraph_edges": {
        "source",
        "split",
        "stay_uid",
        "subgraph_id",
        "src_node_index",
        "dst_node_index",
        "src_id",
        "dst_id",
        "src_type",
        "dst_type",
        "relation_type",
        "support_count",
        "fit_source",
        "fit_split",
        "feature_version",
        "graph_version",
    },
    "subgraph_candidates": {
        "source",
        "split",
        "stay_uid",
        "subgraph_id",
        "index_condition_token",
        "candidate_medication_token",
        "candidate_node_index",
        "candidate_rank",
        "label_prescribed",
        "in_train_graph",
        "cold_start",
        "feature_version",
        "graph_version",
        "label_version",
        "split_version",
    },
}

VOCABULARY_ARTIFACTS = (
    "condition_vocabulary",
    "candidate_medication_vocabulary",
    "event_vocabulary",
    "graph_node_vocabulary",
)
REQUIRED_PACKAGE_ARTIFACTS = frozenset(
    {
        "cohort_stays",
        "cohort_decision_times",
        "patient_stay_features",
        "patient_condition_medication",
        "event_sequences",
        "split_manifest",
        "candidate_catalog",
        "graph_edges",
        "subgraph_index",
        "subgraph_nodes",
        "subgraph_edges",
        "subgraph_candidates",
        *VOCABULARY_ARTIFACTS,
        "train_fitted_preprocessor",
        "preprocessing_manifest",
        "patient_subgraphs_manifest",
        "data_dictionary",
    }
)


@dataclass(frozen=True)
class TrainingContractConfig:
    """Configuration for one Phase 8 P0 training-contract audit."""

    package_manifest_path: Path = DEFAULT_PACKAGE_MANIFEST
    data_dictionary_path: Path = DEFAULT_DATA_DICTIONARY
    primary_training_manifest_path: Path = DEFAULT_PRIMARY_TRAINING_MANIFEST
    sensitivity_training_manifest_path: Path = DEFAULT_SENSITIVITY_TRAINING_MANIFEST
    feature_manifest_path: Path = DEFAULT_FEATURE_MANIFEST
    preprocessing_manifest_path: Path = DEFAULT_PREPROCESSING_MANIFEST
    subgraphs_manifest_path: Path = DEFAULT_SUBGRAPHS_MANIFEST
    graph_suitability_report_path: Path = DEFAULT_GRAPH_SUITABILITY_REPORT
    output_path: Path = DEFAULT_OUTPUT_PATH
    expected_lock_path: Path | None = None
    dataset_root: Path = DATASET_ROOT
    require_protected_storage: bool = True
    duckdb_temp_directory: Path | None = DUCKDB_TEMP_DIR
    duckdb_memory_limit: str | None = DUCKDB_MEMORY_LIMIT
    duckdb_threads: int | None = DUCKDB_THREADS

    def manifest_paths(self) -> dict[str, Path]:
        return {
            "model_ready_manifest": self.package_manifest_path,
            "model_ready_data_dictionary": self.data_dictionary_path,
            "primary_training_manifest": self.primary_training_manifest_path,
            "sensitivity_training_manifest": self.sensitivity_training_manifest_path,
            "feature_manifest": self.feature_manifest_path,
            "preprocessing_manifest": self.preprocessing_manifest_path,
            "patient_subgraphs_manifest": self.subgraphs_manifest_path,
            "graph_suitability_report": self.graph_suitability_report_path,
        }


def load_json(path: Path) -> dict[str, Any]:
    """Load one aggregate JSON document."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one stable aggregate JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    """Hash a small aggregate manifest without reading patient artifacts."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(payload: dict[str, Any]) -> str:
    """Return a deterministic digest for a JSON-compatible contract payload."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def configure_connection(
    config: TrainingContractConfig,
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Apply shared bounded DuckDB settings."""

    configure_duckdb_connection(
        connection,
        temp_directory=config.duckdb_temp_directory,
        memory_limit=config.duckdb_memory_limit,
        threads=config.duckdb_threads,
    )


def schema_columns(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
) -> tuple[tuple[str, str], ...]:
    """Return a Parquet schema from footer metadata only."""

    rows = connection.execute(f"DESCRIBE SELECT * FROM {parquet_scan(path)}").fetchall()
    return tuple((str(row[0]), str(row[1])) for row in rows)


def approved_model_projection(
    artifact_name: str,
    columns: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    """Return the approved model matrix projection for one artifact."""

    available = {name: dtype.upper() for name, dtype in columns}
    if artifact_name in EXPLICIT_PROJECTIONS:
        return tuple(
            name for name in EXPLICIT_PROJECTIONS[artifact_name] if name in available
        )
    if artifact_name != "patient_stay_features":
        return ()

    approved: list[str] = []
    for name, dtype in available.items():
        if is_unsafe_model_column(name) or name in PROVENANCE_COLUMNS:
            continue
        if name in {"source", "split"}:
            continue
        if name in LOW_CARDINALITY_STAY_CATEGORICAL:
            approved.append(name)
            continue
        if any(
            token in dtype for token in ("BOOL", "INT", "DOUBLE", "FLOAT", "DECIMAL")
        ):
            approved.append(name)
    return tuple(approved)


def is_unsafe_model_column(name: str) -> bool:
    """Return whether a dynamic model column is identifier, raw, or post-outcome data."""

    normalized = name.lower()
    return (
        normalized in RAW_OR_UNSAFE_COLUMNS
        or normalized.startswith("source_")
        or normalized.endswith(UNSAFE_COLUMN_SUFFIXES)
        or "hospital_id" in normalized
        or "ward_id" in normalized
        or any(token in normalized for token in OUTCOME_COLUMN_TOKENS)
    )


def validate_requested_columns(
    artifact_name: str,
    requested: Sequence[str],
    columns: Sequence[tuple[str, str]],
) -> None:
    """Fail when a model requests unavailable or disallowed columns."""

    approved = set(approved_model_projection(artifact_name, columns))
    rejected = sorted(
        name
        for name in set(requested)
        if name not in approved or is_unsafe_model_column(name)
    )
    if rejected:
        raise ValueError(
            f"unsafe or unavailable {artifact_name} model columns: "
            + ", ".join(rejected)
        )


def _error(
    code: str, detail: str, *, artifact_name: str | None = None
) -> dict[str, str]:
    row = {"code": code, "detail": detail}
    if artifact_name is not None:
        row["artifact_name"] = artifact_name
    return row


def _is_protected_dataset_root(path: Path) -> bool:
    return "protected" in path.resolve().parts


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _table_row_counts(package: dict[str, Any]) -> dict[str, int]:
    return {
        str(row["table_name"]): int(row["row_count"])
        for row in package.get("tables", [])
        if row.get("row_count") is not None
    }


def _base_report(*, status: str, generated_at: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "generated_at": generated_at,
        "versions": dict(PINNED_VERSIONS),
        "clinical_claim_boundary": (
            "The locked inputs encode observed historical prescribing in (24h, 48h]. "
            "They are research artifacts, not validated treatment recommendations."
        ),
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "report_contains_identifier_values": False,
            "manifest_and_schema_metadata_only": True,
        },
    }


def build_training_contract_lock(
    config: TrainingContractConfig = TrainingContractConfig(),
) -> dict[str, Any]:
    """Audit Phase 8 P0 inputs and write an aggregate deterministic lock."""

    generated_at = datetime.now(UTC).isoformat()
    report = _base_report(status="completed", generated_at=generated_at)
    missing_manifests = [
        {"manifest_name": name, "path": str(path)}
        for name, path in config.manifest_paths().items()
        if not path.exists()
    ]
    if missing_manifests:
        report["status"] = "failed_missing_inputs"
        report["missing_inputs"] = missing_manifests
        write_json(config.output_path, report)
        return report

    try:
        manifests = {
            name: load_json(path) for name, path in config.manifest_paths().items()
        }
    except (OSError, json.JSONDecodeError) as error:
        report["status"] = "failed_invalid_manifest"
        report["reason"] = safe_error_message(error)
        write_json(config.output_path, report)
        return report

    errors: list[dict[str, str]] = []
    package = manifests["model_ready_manifest"]
    primary = manifests["primary_training_manifest"]
    sensitivity = manifests["sensitivity_training_manifest"]
    features = manifests["feature_manifest"]
    preprocessing = manifests["preprocessing_manifest"]
    subgraphs = manifests["patient_subgraphs_manifest"]
    graph = manifests["graph_suitability_report"]

    for name, manifest in manifests.items():
        if manifest.get("status") != "completed":
            errors.append(
                _error(
                    "manifest_not_completed",
                    f"{name} status must be completed",
                    artifact_name=name,
                )
            )

    for name, expected in PINNED_VERSIONS.items():
        actual = package.get("versions", {}).get(name)
        if actual != expected:
            errors.append(
                _error(
                    "version_mismatch",
                    f"{name} must equal {expected!r}",
                    artifact_name="model_ready_manifest",
                )
            )

    for manifest_name, manifest in manifests.items():
        declared_versions = manifest.get("versions", {})
        for version_name, actual in declared_versions.items():
            expected = PINNED_VERSIONS.get(version_name)
            if expected is not None and actual != expected:
                errors.append(
                    _error(
                        "upstream_version_mismatch",
                        f"{version_name} must equal {expected!r}",
                        artifact_name=manifest_name,
                    )
                )

    parameters = primary.get("parameters", {})
    if parameters.get("development_source") != "mimiciv":
        errors.append(
            _error("invalid_development_source", "MIMIC must be development source")
        )
    if parameters.get("candidate_token_strategy") != "rxnorm_or_atc":
        errors.append(
            _error("invalid_primary_catalog", "primary catalog must be rxnorm_or_atc")
        )
    if parameters.get("prediction_offset_hours") != 24:
        errors.append(
            _error("invalid_prediction_offset", "prediction offset must be 24h")
        )
    if parameters.get("label_window_hours") != 24:
        errors.append(_error("invalid_label_window", "label window width must be 24h"))
    if primary.get("split_integrity", {}).get("patients_with_multiple_splits") != 0:
        errors.append(
            _error(
                "manifest_split_overlap", "training manifest reports patient overlap"
            )
        )
    primary_external = primary.get("external_validation", {})
    if (
        primary_external.get("status") != "coverage_only_no_in_catalog_positive_groups"
        or primary_external.get("positive_ranking_group_count") != 0
        or primary_external.get("performance_claims_allowed") is not False
    ):
        errors.append(
            _error(
                "invalid_primary_external_scope",
                "primary eICU catalog must remain coverage-only with zero positive groups",
            )
        )

    sensitivity_parameters = sensitivity.get("parameters", {})
    if sensitivity_parameters.get("candidate_token_strategy") != "atc3_or_rxnorm":
        errors.append(
            _error(
                "invalid_sensitivity_catalog",
                "sensitivity catalog must be atc3_or_rxnorm",
            )
        )
    sensitivity_external = sensitivity.get("external_validation", {})
    if (
        sensitivity_external.get("status") != "externally_evaluable"
        or int(sensitivity_external.get("positive_ranking_group_count") or 0) <= 0
        or sensitivity_external.get("performance_claims_allowed") is not True
    ):
        errors.append(
            _error(
                "invalid_sensitivity_external_scope",
                "ATC-3 sensitivity must have externally evaluable positive groups",
            )
        )
    if (
        features.get("parameters", {}).get("include_predecision_medications")
        is not False
    ):
        errors.append(
            _error(
                "medication_history_enabled", "event sequences must exclude medications"
            )
        )
    if features.get("parameters", {}).get("prediction_offset_hours") != 24:
        errors.append(_error("feature_cutoff_mismatch", "feature cutoff must be 24h"))

    fit_scope = preprocessing.get("fit_scope", {})
    if fit_scope.get("source") != "mimiciv" or fit_scope.get("split") != "train":
        errors.append(
            _error("preprocessor_fit_scope", "preprocessor must fit on MIMIC train")
        )
    if any(
        row.get("fit_source") != "mimiciv" or row.get("fit_split") != "train"
        for row in package.get("train_fit_scope", [])
    ):
        errors.append(
            _error("graph_fit_scope", "all package graph scopes must be MIMIC train")
        )
    if any(
        row.get("fit_source") != "mimiciv" or row.get("fit_split") != "train"
        for row in subgraphs.get("graph_fit_scope", [])
    ):
        errors.append(
            _error("subgraph_fit_scope", "subgraph edges must be MIMIC train fit")
        )
    if graph.get("leakage_audit", {}).get("status") != "pass":
        errors.append(_error("graph_leakage_audit", "graph leakage audit must pass"))
    if graph.get("gate_review", {}).get("result") != "pass_for_graph_ablation":
        errors.append(
            _error("graph_suitability_gate", "graph must pass for graph ablation")
        )

    artifacts = {
        str(name): Path(path) for name, path in package.get("artifacts", {}).items()
    }
    missing_artifact_names = sorted(REQUIRED_PACKAGE_ARTIFACTS - artifacts.keys())
    if missing_artifact_names:
        errors.append(
            _error(
                "missing_package_artifacts",
                "model-ready package is missing artifacts: "
                + ", ".join(missing_artifact_names),
            )
        )
    missing_artifacts = [
        {"artifact_name": name, "path": str(path)}
        for name, path in artifacts.items()
        if not path.exists()
    ]
    if missing_artifacts:
        report["status"] = "failed_missing_inputs"
        report["missing_inputs"] = missing_artifacts
        write_json(config.output_path, report)
        return report

    if config.require_protected_storage and not _is_protected_dataset_root(
        config.dataset_root
    ):
        errors.append(
            _error(
                "unprotected_dataset_root",
                "DATASET_ROOT must resolve to protected storage",
            )
        )

    row_counts = _table_row_counts(package)
    artifact_locks: dict[str, dict[str, Any]] = {}
    projections: dict[str, dict[str, Any]] = {}
    split_summary: list[dict[str, Any]] = []
    split_alignment_summary: list[dict[str, Any]] = []
    candidate_scope_summary: dict[str, Any] = {}
    temporal_summary: dict[str, Any] = {}
    graph_scope_summary: list[dict[str, Any]] = []
    vocabulary_scope_summary: list[dict[str, Any]] = []

    try:
        with duckdb.connect(database=":memory:") as connection:
            configure_connection(config, connection)
            for artifact_name, path in artifacts.items():
                stat = path.stat()
                artifact_locks[artifact_name] = {
                    "path": str(path),
                    "file_size_bytes": int(stat.st_size),
                    "modified_time_ns": int(stat.st_mtime_ns),
                    "row_count": row_counts.get(artifact_name),
                    "actual_row_count": None,
                    "schema_digest": None,
                }
                patient_level_artifact = (
                    path.suffix == ".parquet"
                    or artifact_name == "train_fitted_preprocessor"
                )
                if (
                    config.require_protected_storage
                    and patient_level_artifact
                    and not _is_under(path, config.dataset_root)
                ):
                    errors.append(
                        _error(
                            "artifact_outside_dataset_root",
                            "patient-level artifact must remain under DATASET_ROOT",
                            artifact_name=artifact_name,
                        )
                    )
                if path.suffix != ".parquet":
                    continue
                actual_row_count = int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {parquet_scan(path)}"
                    ).fetchone()[0]
                )
                artifact_locks[artifact_name]["actual_row_count"] = actual_row_count
                declared_row_count = row_counts.get(artifact_name)
                if (
                    declared_row_count is None
                    or int(declared_row_count) != actual_row_count
                ):
                    errors.append(
                        _error(
                            "artifact_row_count_mismatch",
                            "artifact row count differs from package manifest",
                            artifact_name=artifact_name,
                        )
                    )
                columns = schema_columns(connection, path)
                column_names = {name for name, _dtype in columns}
                missing_columns = sorted(
                    REQUIRED_SCHEMA_COLUMNS.get(artifact_name, set()) - column_names
                )
                if missing_columns:
                    errors.append(
                        _error(
                            "missing_schema_columns",
                            "missing required columns: " + ", ".join(missing_columns),
                            artifact_name=artifact_name,
                        )
                    )
                projection = approved_model_projection(artifact_name, columns)
                rejected_projection = sorted(set(projection) & RAW_OR_UNSAFE_COLUMNS)
                if rejected_projection:
                    errors.append(
                        _error(
                            "unsafe_projection",
                            "unsafe projected columns: "
                            + ", ".join(rejected_projection),
                            artifact_name=artifact_name,
                        )
                    )
                artifact_locks[artifact_name]["schema_digest"] = stable_digest(
                    {
                        "columns": [
                            {"name": name, "dtype": dtype} for name, dtype in columns
                        ]
                    }
                )
                projections[artifact_name] = {
                    "approved_column_count": len(projection),
                    "approved_columns": list(projection),
                    "source_unsafe_column_count": len(
                        column_names & RAW_OR_UNSAFE_COLUMNS
                    ),
                }
            split_path = artifacts["split_manifest"]
            split_summary = fetch_dict_rows(
                connection,
                f"""
WITH patient_splits AS (
    SELECT source, patient_uid, COUNT(DISTINCT split) AS split_count
    FROM {parquet_scan(split_path)}
    GROUP BY source, patient_uid
)
SELECT
    source,
    COUNT(*) AS patient_count,
    SUM(CASE WHEN split_count > 1 THEN 1 ELSE 0 END)
        AS patients_with_multiple_splits
FROM patient_splits
GROUP BY source
ORDER BY source
""",
            )
            if any(row["patients_with_multiple_splits"] for row in split_summary):
                errors.append(
                    _error(
                        "patient_split_overlap", "a patient appears in multiple splits"
                    )
                )

            for artifact_name in (
                "patient_stay_features",
                "event_sequences",
                "patient_condition_medication",
            ):
                path = artifacts[artifact_name]
                alignment = fetch_dict_rows(
                    connection,
                    f"""
WITH artifact_splits AS (
    SELECT DISTINCT source, patient_uid, split
    FROM {parquet_scan(path)}
), canonical_splits AS (
    SELECT source, patient_uid, MIN(split) AS split
    FROM {parquet_scan(split_path)}
    GROUP BY source, patient_uid
)
SELECT
    {artifact_name!r} AS artifact_name,
    COUNT(*) AS distinct_patient_split_count,
    SUM(CASE
        WHEN artifact_splits.patient_uid IS NULL
            OR canonical_splits.patient_uid IS NULL
        THEN 1 ELSE 0
    END) AS missing_patient_split_count,
    SUM(CASE
        WHEN canonical_splits.patient_uid IS NOT NULL
            AND artifact_splits.split <> canonical_splits.split
        THEN 1 ELSE 0
    END) AS mismatched_patient_split_count
FROM artifact_splits
LEFT JOIN canonical_splits USING (source, patient_uid)
""",
                )[0]
                split_alignment_summary.append(alignment)
            if any(
                int(row["missing_patient_split_count"] or 0) > 0
                or int(row["mismatched_patient_split_count"] or 0) > 0
                for row in split_alignment_summary
            ):
                errors.append(
                    _error(
                        "artifact_split_mismatch",
                        "an input artifact disagrees with the patient split manifest",
                    )
                )

            catalog_path = artifacts["candidate_catalog"]
            candidate_scope_summary = fetch_dict_rows(
                connection,
                f"""
WITH train_positives AS (
    SELECT DISTINCT index_condition_token, candidate_medication_token
    FROM {parquet_scan(artifacts["patient_condition_medication"])}
    WHERE source = 'mimiciv' AND split = 'train' AND label_prescribed
)
SELECT
    COUNT(*) AS candidate_count,
    SUM(CASE
        WHEN train_positives.candidate_medication_token IS NULL
        THEN 1 ELSE 0
    END) AS candidates_without_train_positive_count
FROM {parquet_scan(catalog_path)} AS catalog
LEFT JOIN train_positives USING (
    index_condition_token,
    candidate_medication_token
)
""",
            )[0]
            if (
                int(
                    candidate_scope_summary["candidates_without_train_positive_count"]
                    or 0
                )
                > 0
            ):
                errors.append(
                    _error(
                        "candidate_fit_scope",
                        "candidate catalog contains entries not derived from MIMIC train positives",
                    )
                )

            event_path = artifacts["event_sequences"]
            event_rows = fetch_dict_rows(
                connection,
                f"""
SELECT
    COUNT(*) AS row_count,
    SUM(CASE
        WHEN event_time_hours_from_admit IS NULL
            OR event_time_hours_from_admit < 0
        THEN 1 ELSE 0
    END)
        AS pre_admission_row_count,
    SUM(CASE WHEN event_time_hours_from_admit > 24 THEN 1 ELSE 0 END)
        AS post_cutoff_row_count,
    SUM(CASE WHEN event_type = 'medication' THEN 1 ELSE 0 END)
        AS medication_row_count
FROM {parquet_scan(event_path)}
""",
            )[0]
            temporal_summary["event_sequences"] = event_rows
            if any(
                int(event_rows.get(name) or 0) > 0
                for name in (
                    "pre_admission_row_count",
                    "post_cutoff_row_count",
                    "medication_row_count",
                )
            ):
                errors.append(
                    _error(
                        "event_temporal_leakage",
                        "event sequence timing or medication exclusion failed",
                    )
                )

            ranking_path = artifacts["patient_condition_medication"]
            ranking_rows = fetch_dict_rows(
                connection,
                f"""
SELECT
    COUNT(*) AS row_count,
    SUM(CASE WHEN prediction_time_hours_from_admit <> 24 THEN 1 ELSE 0 END)
        AS invalid_prediction_time_rows,
    SUM(CASE WHEN label_window_end_hours_from_admit <> 48 THEN 1 ELSE 0 END)
        AS invalid_label_window_end_rows,
    SUM(CASE
        WHEN label_prescribed
            AND (
                label_first_observed_hours_from_admit IS NULL
                OR label_first_observed_hours_from_admit <= 24
                OR label_first_observed_hours_from_admit > 48
            )
            THEN 1 ELSE 0
        END) AS invalid_positive_label_time_rows
FROM {parquet_scan(ranking_path)}
""",
            )[0]
            temporal_summary["patient_condition_medication"] = ranking_rows
            if any(
                int(ranking_rows.get(name) or 0) > 0
                for name in (
                    "invalid_prediction_time_rows",
                    "invalid_label_window_end_rows",
                    "invalid_positive_label_time_rows",
                )
            ):
                errors.append(
                    _error(
                        "label_temporal_contract",
                        "ranking label timing contract failed",
                    )
                )

            for artifact_name in ("graph_edges", "subgraph_edges"):
                graph_path = artifacts[artifact_name]
                rows = fetch_dict_rows(
                    connection,
                    f"""
SELECT fit_source, fit_split, COUNT(*) AS edge_count
FROM {parquet_scan(graph_path)}
GROUP BY fit_source, fit_split
ORDER BY fit_source, fit_split
""",
                )
                graph_scope_summary.extend(
                    {"artifact_name": artifact_name, **row} for row in rows
                )
            if any(
                row["fit_source"] != "mimiciv" or row["fit_split"] != "train"
                for row in graph_scope_summary
            ):
                errors.append(
                    _error(
                        "graph_fit_scope_rows",
                        "graph edge rows are not exclusively MIMIC train fit",
                    )
                )

            for artifact_name in VOCABULARY_ARTIFACTS:
                path = artifacts[artifact_name]
                rows = fetch_dict_rows(
                    connection,
                    f"""
SELECT fit_source, fit_split, COUNT(*) AS token_count
FROM {parquet_scan(path)}
GROUP BY fit_source, fit_split
ORDER BY fit_source, fit_split
""",
                )
                vocabulary_scope_summary.extend(
                    {"artifact_name": artifact_name, **row} for row in rows
                )
            if any(
                row["fit_source"] != "mimiciv" or row["fit_split"] != "train"
                for row in vocabulary_scope_summary
            ):
                errors.append(
                    _error(
                        "vocabulary_fit_scope", "vocabularies must be MIMIC train fit"
                    )
                )
    except (duckdb.Error, OSError, KeyError, ValueError) as error:
        errors.append(_error("audit_query_failed", safe_error_message(error)))

    manifest_locks = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in config.manifest_paths().items()
    }
    contract = {
        "versions": dict(PINNED_VERSIONS),
        "manifests": manifest_locks,
        "artifacts": artifact_locks,
        "model_projections": projections,
        "split_integrity": split_summary,
        "artifact_split_alignment": split_alignment_summary,
        "candidate_fit_scope": candidate_scope_summary,
        "temporal_integrity": temporal_summary,
        "graph_fit_scope": graph_scope_summary,
        "vocabulary_fit_scope": vocabulary_scope_summary,
        "external_validation": {
            "primary": primary.get("external_validation", {}),
            "sensitivity": sensitivity.get("external_validation", {}),
        },
        "label_semantics": "observed medication starts in (24h, 48h]",
    }
    contract_digest = stable_digest(contract)
    report["contract_digest"] = contract_digest
    report["contract"] = contract

    if config.expected_lock_path is not None:
        if not config.expected_lock_path.exists():
            errors.append(
                _error("missing_expected_lock", "expected contract lock is missing")
            )
        else:
            try:
                expected = load_json(config.expected_lock_path).get("contract_digest")
            except (OSError, json.JSONDecodeError):
                expected = None
                errors.append(
                    _error(
                        "invalid_expected_lock",
                        "expected contract lock is not valid JSON",
                    )
                )
            if expected is not None and expected != contract_digest:
                errors.append(
                    _error(
                        "contract_digest_mismatch",
                        "input contract differs from expected lock",
                    )
                )

    if errors:
        report["status"] = "failed_contract_audit"
        report["errors"] = errors
    write_json(config.output_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and lock the Phase 8 P0 training input contract.",
    )
    parser.add_argument(
        "--package-manifest", type=Path, default=DEFAULT_PACKAGE_MANIFEST
    )
    parser.add_argument("--data-dictionary", type=Path, default=DEFAULT_DATA_DICTIONARY)
    parser.add_argument(
        "--primary-training-manifest",
        type=Path,
        default=DEFAULT_PRIMARY_TRAINING_MANIFEST,
    )
    parser.add_argument(
        "--sensitivity-training-manifest",
        type=Path,
        default=DEFAULT_SENSITIVITY_TRAINING_MANIFEST,
    )
    parser.add_argument(
        "--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST
    )
    parser.add_argument(
        "--preprocessing-manifest",
        type=Path,
        default=DEFAULT_PREPROCESSING_MANIFEST,
    )
    parser.add_argument(
        "--subgraphs-manifest", type=Path, default=DEFAULT_SUBGRAPHS_MANIFEST
    )
    parser.add_argument(
        "--graph-suitability-report",
        type=Path,
        default=DEFAULT_GRAPH_SUITABILITY_REPORT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--expected-lock", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument(
        "--allow-unprotected-storage",
        action="store_true",
        help="Allow synthetic fixture roots outside protected storage.",
    )
    parser.add_argument("--duckdb-temp-dir", type=Path, default=DUCKDB_TEMP_DIR)
    parser.add_argument("--duckdb-memory-limit", default=DUCKDB_MEMORY_LIMIT)
    parser.add_argument("--duckdb-threads", type=int, default=DUCKDB_THREADS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_training_contract_lock(
        TrainingContractConfig(
            package_manifest_path=args.package_manifest,
            data_dictionary_path=args.data_dictionary,
            primary_training_manifest_path=args.primary_training_manifest,
            sensitivity_training_manifest_path=args.sensitivity_training_manifest,
            feature_manifest_path=args.feature_manifest,
            preprocessing_manifest_path=args.preprocessing_manifest,
            subgraphs_manifest_path=args.subgraphs_manifest,
            graph_suitability_report_path=args.graph_suitability_report,
            output_path=args.output,
            expected_lock_path=args.expected_lock,
            dataset_root=args.dataset_root,
            require_protected_storage=not args.allow_unprotected_storage,
            duckdb_temp_directory=args.duckdb_temp_dir,
            duckdb_memory_limit=args.duckdb_memory_limit,
            duckdb_threads=args.duckdb_threads,
        )
    )
    print(
        "Wrote Phase 8 P0 training contract lock: "
        f"status={report['status']}, digest={report.get('contract_digest', 'none')}"
    )
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
