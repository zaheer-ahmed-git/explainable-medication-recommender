"""Assemble the complete Phase 8 P0 model-ready artifact package."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import duckdb

from pipeline.artifact_metadata import infer_consistent_version
from pipeline.config import (
    DUCKDB_MEMORY_LIMIT,
    DUCKDB_TEMP_DIR,
    DUCKDB_THREADS,
    FEATURE_VERSION,
    FEATURES_ROOT,
    GRAPH_VERSION,
    LABEL_VERSION,
    MILESTONE8_GRAPH_ROOT,
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
from pipeline.features import (
    PHASE8_P0_FEATURE_VERSION,
    copy_query_to_parquet,
    fetch_dict_rows,
)
from pipeline.patient_subgraphs import DEFAULT_SUBGRAPHS_ROOT


SCHEMA_VERSION = "phase8-p0-model-ready-package-v1"
DATA_DICTIONARY_VERSION = "phase8-p0-model-ready-data-dictionary-v1"
DEFAULT_PACKAGE_ROOT = TRAINING_ROOT / "model_ready"
DEFAULT_DATA_DICTIONARY_PATH = (
    REPORTS_ROOT / "phase8_p0_model_ready_data_dictionary.json"
)
DEFAULT_MANIFEST_PATH = REPORTS_ROOT / "phase8_p0_model_ready_manifest.json"
DEFAULT_PRIMARY_TRAINING_MANIFEST = (
    REPORTS_ROOT / "phase8_p0_training_table_manifest.json"
)
DEFAULT_SENSITIVITY_TRAINING_MANIFEST = (
    REPORTS_ROOT / "phase8_p0_atc3_training_table_manifest.json"
)
DEFAULT_PREPROCESSING_MANIFEST = REPORTS_ROOT / "phase8_p0_preprocessing_manifest.json"
DEFAULT_SUBGRAPHS_MANIFEST = REPORTS_ROOT / "phase8_p0_patient_subgraphs_manifest.json"

PATIENT_IDENTIFIER_COLUMNS = frozenset(
    {
        "patient_uid",
        "encounter_uid",
        "stay_uid",
        "source_patient_id",
        "source_encounter_id",
        "source_stay_id",
        "ranking_group_id",
        "subgraph_id",
        "source_event_id",
    }
)
TOKEN_COLUMNS = frozenset(
    {
        "index_condition_token",
        "candidate_medication_token",
        "event_token",
        "node_id",
        "src_id",
        "dst_id",
    }
)


@dataclass(frozen=True)
class ModelReadyPackageConfig:
    """Configuration for vocabulary, dictionary, and package manifest outputs."""

    features_root: Path = FEATURES_ROOT
    training_root: Path = TRAINING_ROOT
    graph_root: Path = MILESTONE8_GRAPH_ROOT
    subgraphs_root: Path = DEFAULT_SUBGRAPHS_ROOT
    preprocessing_root: Path = TRAINING_ROOT / "preprocessing"
    package_root: Path = DEFAULT_PACKAGE_ROOT
    data_dictionary_path: Path = DEFAULT_DATA_DICTIONARY_PATH
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    primary_training_manifest_path: Path = DEFAULT_PRIMARY_TRAINING_MANIFEST
    sensitivity_training_manifest_path: Path = DEFAULT_SENSITIVITY_TRAINING_MANIFEST
    preprocessing_manifest_path: Path = DEFAULT_PREPROCESSING_MANIFEST
    subgraphs_manifest_path: Path = DEFAULT_SUBGRAPHS_MANIFEST
    feature_version: str | None = None
    graph_version: str | None = None
    label_version: str = LABEL_VERSION
    split_version: str = SPLIT_VERSION
    duckdb_temp_directory: Path | None = DUCKDB_TEMP_DIR
    duckdb_memory_limit: str | None = DUCKDB_MEMORY_LIMIT
    duckdb_threads: int | None = DUCKDB_THREADS

    @property
    def vocabularies_root(self) -> Path:
        return self.package_root / "vocabularies"

    @property
    def condition_vocabulary_path(self) -> Path:
        return self.vocabularies_root / "condition_vocabulary.parquet"

    @property
    def candidate_medication_vocabulary_path(self) -> Path:
        return self.vocabularies_root / "candidate_medication_vocabulary.parquet"

    @property
    def event_vocabulary_path(self) -> Path:
        return self.vocabularies_root / "event_vocabulary.parquet"

    @property
    def graph_node_vocabulary_path(self) -> Path:
        return self.vocabularies_root / "graph_node_vocabulary.parquet"

    @property
    def preprocessor_path(self) -> Path:
        return self.preprocessing_root / "train_fitted_preprocessor.joblib"

    def parquet_artifacts(self) -> dict[str, Path]:
        """Return final Parquet artifact names and paths."""

        return {
            "cohort_stays": self.training_root / "cohort_stays.parquet",
            "cohort_decision_times": (
                self.features_root / "cohort_decision_times.parquet"
            ),
            "patient_stay_features": (
                self.features_root / "patient_stay_features.parquet"
            ),
            "patient_condition_medication": (
                self.training_root / "patient_condition_medication.parquet"
            ),
            "event_sequences": self.features_root / "event_sequences.parquet",
            "split_manifest": self.training_root / "split_manifest.parquet",
            "candidate_catalog": self.training_root / "candidate_catalog.parquet",
            "graph_edges": self.graph_root / "graph_edges.parquet",
            "subgraph_index": self.subgraphs_root / "subgraph_index.parquet",
            "subgraph_nodes": self.subgraphs_root / "subgraph_nodes.parquet",
            "subgraph_edges": self.subgraphs_root / "subgraph_edges.parquet",
            "subgraph_candidates": (
                self.subgraphs_root / "subgraph_candidates.parquet"
            ),
        }

    def vocabulary_artifacts(self) -> dict[str, Path]:
        """Return generated local vocabulary paths."""

        return {
            "condition_vocabulary": self.condition_vocabulary_path,
            "candidate_medication_vocabulary": (
                self.candidate_medication_vocabulary_path
            ),
            "event_vocabulary": self.event_vocabulary_path,
            "graph_node_vocabulary": self.graph_node_vocabulary_path,
        }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable aggregate JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    """Load an aggregate JSON manifest."""

    return json.loads(path.read_text(encoding="utf-8"))


def configure_connection(
    config: ModelReadyPackageConfig,
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Apply shared memory-safe DuckDB settings."""

    configure_duckdb_connection(
        connection,
        temp_directory=config.duckdb_temp_directory,
        memory_limit=config.duckdb_memory_limit,
        threads=config.duckdb_threads,
    )


def required_inputs(config: ModelReadyPackageConfig) -> dict[str, Path]:
    """Return all artifacts required for a completed package."""

    return {
        **config.parquet_artifacts(),
        "train_fitted_preprocessor": config.preprocessor_path,
        "primary_training_manifest": config.primary_training_manifest_path,
        "sensitivity_training_manifest": config.sensitivity_training_manifest_path,
        "preprocessing_manifest": config.preprocessing_manifest_path,
        "patient_subgraphs_manifest": config.subgraphs_manifest_path,
    }


def missing_inputs(config: ModelReadyPackageConfig) -> list[dict[str, str]]:
    """Return missing final-package inputs."""

    return [
        {"artifact_name": name, "path": str(path)}
        for name, path in required_inputs(config).items()
        if not path.exists()
    ]


def dependency_manifest_statuses(
    config: ModelReadyPackageConfig,
) -> list[dict[str, str | None]]:
    """Return aggregate dependency completion statuses."""

    paths = {
        "primary_training_manifest": config.primary_training_manifest_path,
        "sensitivity_training_manifest": config.sensitivity_training_manifest_path,
        "preprocessing_manifest": config.preprocessing_manifest_path,
        "patient_subgraphs_manifest": config.subgraphs_manifest_path,
    }
    return [
        {
            "manifest_name": name,
            "status": load_json(path).get("status"),
        }
        for name, path in paths.items()
    ]


def train_fit_scope_summary(
    connection: duckdb.DuckDBPyConnection,
    config: ModelReadyPackageConfig,
) -> list[dict[str, Any]]:
    """Return aggregate fit-scope counts for graph and subgraph edges."""

    paths = {
        "graph_edges": config.graph_root / "graph_edges.parquet",
        "subgraph_edges": config.subgraphs_root / "subgraph_edges.parquet",
    }
    rows: list[dict[str, Any]] = []
    for artifact_name, path in paths.items():
        artifact_rows = fetch_dict_rows(
            connection,
            f"""
SELECT fit_source, fit_split, COUNT(*) AS edge_count
FROM {parquet_scan(path)}
GROUP BY fit_source, fit_split
ORDER BY fit_source, fit_split
""",
        )
        rows.extend({"artifact_name": artifact_name, **row} for row in artifact_rows)
    return rows


def train_fit_scope_is_valid(rows: Sequence[dict[str, Any]]) -> bool:
    """Return whether all graph statistics are MIMIC-train fitted."""

    return all(
        row["fit_source"] == "mimiciv" and row["fit_split"] == "train" for row in rows
    )


def resolve_versions(config: ModelReadyPackageConfig) -> ModelReadyPackageConfig:
    """Return config stamped from all version-bearing input artifacts."""

    paths = tuple(config.parquet_artifacts().values())
    feature_version = infer_consistent_version(
        paths,
        column_name="feature_version",
        declared_version=config.feature_version,
        fallback_version=FEATURE_VERSION,
    )
    if feature_version != PHASE8_P0_FEATURE_VERSION:
        raise ValueError(
            "Phase 8 P0 package requires feature_version "
            f"{PHASE8_P0_FEATURE_VERSION!r}; found {feature_version!r}"
        )
    graph_version = infer_consistent_version(
        (config.graph_root / "graph_edges.parquet",),
        column_name="graph_version",
        declared_version=config.graph_version,
        fallback_version=GRAPH_VERSION,
    )
    return replace(
        config,
        feature_version=feature_version,
        graph_version=graph_version,
    )


def vocabulary_queries(
    config: ModelReadyPackageConfig,
    *,
    generated_at: str,
) -> dict[str, tuple[str, Path]]:
    """Return train-fit vocabulary queries and output paths."""

    catalog = config.training_root / "candidate_catalog.parquet"
    events = config.features_root / "event_sequences.parquet"
    edges = config.graph_root / "graph_edges.parquet"
    feature_version = sql_string(config.feature_version or FEATURE_VERSION)
    graph_version = sql_string(config.graph_version or GRAPH_VERSION)
    timestamp = sql_string(generated_at)
    return {
        "condition_vocabulary": (
            f"""
WITH tokens AS (
    SELECT DISTINCT index_condition_token AS condition_token
    FROM {parquet_scan(catalog)}
    WHERE index_condition_token IS NOT NULL
)
SELECT
    ROW_NUMBER() OVER (ORDER BY condition_token) - 1 AS token_index,
    condition_token,
    'mimiciv' AS fit_source,
    'train' AS fit_split,
    {feature_version} AS feature_version,
    {timestamp} AS generated_at
FROM tokens
""",
            config.condition_vocabulary_path,
        ),
        "candidate_medication_vocabulary": (
            f"""
WITH tokens AS (
    SELECT DISTINCT candidate_medication_token
    FROM {parquet_scan(catalog)}
    WHERE candidate_medication_token IS NOT NULL
)
SELECT
    ROW_NUMBER() OVER (ORDER BY candidate_medication_token) - 1 AS token_index,
    candidate_medication_token,
    'mimiciv' AS fit_source,
    'train' AS fit_split,
    {feature_version} AS feature_version,
    {timestamp} AS generated_at
FROM tokens
""",
            config.candidate_medication_vocabulary_path,
        ),
        "event_vocabulary": (
            f"""
WITH tokens AS (
    SELECT DISTINCT event_type, event_token
    FROM {parquet_scan(events)}
    WHERE source = 'mimiciv'
        AND split = 'train'
        AND event_token IS NOT NULL
        AND event_time_hours_from_admit >= 0
        AND event_time_hours_from_admit <= 24
)
SELECT
    ROW_NUMBER() OVER (ORDER BY event_type, event_token) - 1 AS token_index,
    event_type,
    event_token,
    event_type || '|' || event_token AS node_id,
    'mimiciv' AS fit_source,
    'train' AS fit_split,
    {feature_version} AS feature_version,
    {timestamp} AS generated_at
FROM tokens
""",
            config.event_vocabulary_path,
        ),
        "graph_node_vocabulary": (
            f"""
WITH nodes AS (
    SELECT src_id AS node_id, src_type AS node_type
    FROM {parquet_scan(edges)}
    UNION
    SELECT dst_id AS node_id, dst_type AS node_type
    FROM {parquet_scan(edges)}
)
SELECT
    ROW_NUMBER() OVER (ORDER BY node_type, node_id) - 1 AS node_index,
    node_id,
    node_type,
    'mimiciv' AS fit_source,
    'train' AS fit_split,
    {feature_version} AS feature_version,
    {graph_version} AS graph_version,
    {timestamp} AS generated_at
FROM nodes
""",
            config.graph_node_vocabulary_path,
        ),
    }


def column_role(column_name: str) -> str:
    """Return the model-ready role for a schema column."""

    if column_name in PATIENT_IDENTIFIER_COLUMNS:
        return "restricted_local_identifier"
    if column_name in TOKEN_COLUMNS or column_name.endswith("_token"):
        return "concept_token"
    if column_name == "split":
        return "partition"
    if column_name.startswith("label_") or column_name == "label_prescribed":
        return "observed_label"
    if "time" in column_name or "timestamp" in column_name:
        return "temporal_boundary_or_value"
    if column_name.endswith("_version") or column_name == "generated_at":
        return "provenance"
    if column_name.endswith("_id"):
        return "local_identifier_or_graph_key"
    return "feature_or_metadata"


def artifact_dictionary_metadata(artifact_name: str) -> dict[str, str]:
    """Return branch, temporal, and leakage annotations for an artifact."""

    if artifact_name in {"event_sequences", "patient_stay_features"}:
        return {
            "branch_use": "patient_context",
            "temporal_window": "predecision through 24 hours from admission",
            "leakage_notes": "future and default medication-history events excluded",
        }
    if artifact_name == "patient_condition_medication":
        return {
            "branch_use": "ranking_head_and_evaluation",
            "temporal_window": "features through 24h; observed labels in (24h, 48h]",
            "leakage_notes": "candidate catalog fit on MIMIC train only",
        }
    if artifact_name.startswith("subgraph_") or artifact_name == "graph_edges":
        return {
            "branch_use": "graph_context",
            "temporal_window": "train-fit graph with patient context through 24h",
            "leakage_notes": "graph statistics fit on MIMIC train only",
        }
    if artifact_name.endswith("vocabulary"):
        return {
            "branch_use": "token_encoding",
            "temporal_window": "train-fit predecision scope",
            "leakage_notes": "vocabulary values are local and train-derived",
        }
    return {
        "branch_use": "shared_model_input_or_provenance",
        "temporal_window": "artifact-specific boundaries recorded in columns",
        "leakage_notes": "patient split and version provenance preserved",
    }


def describe_artifact(
    connection: duckdb.DuckDBPyConnection,
    *,
    artifact_name: str,
    path: Path,
) -> dict[str, Any]:
    """Describe one Parquet schema without reading or sampling rows."""

    cursor = connection.execute(f"DESCRIBE SELECT * FROM {parquet_scan(path)}")
    columns = [
        {
            "column_name": str(row[0]),
            "dtype": str(row[1]),
            "key_role": column_role(str(row[0])),
            "restricted_local_identifier": str(row[0]) in PATIENT_IDENTIFIER_COLUMNS,
        }
        for row in cursor.fetchall()
    ]
    return {
        "artifact_path": str(path),
        "columns": columns,
        **artifact_dictionary_metadata(artifact_name),
    }


def build_data_dictionary(
    connection: duckdb.DuckDBPyConnection,
    config: ModelReadyPackageConfig,
    *,
    generated_at: str,
) -> dict[str, Any]:
    """Build a disclosure-safe dictionary from schemas only."""

    artifacts = {
        **config.parquet_artifacts(),
        **config.vocabulary_artifacts(),
    }
    return {
        "schema_version": DATA_DICTIONARY_VERSION,
        "status": "completed",
        "generated_at": generated_at,
        "artifacts": {
            name: describe_artifact(
                connection,
                artifact_name=name,
                path=path,
            )
            for name, path in artifacts.items()
        },
        "data_safety": {
            "contains_patient_rows": False,
            "contains_row_samples": False,
            "contains_note_text": False,
            "contains_raw_source_text": False,
            "schema_names_may_identify_restricted_local_columns": True,
        },
    }


def external_validation_entry(path: Path) -> dict[str, Any]:
    """Read one aggregate training manifest's eICU readiness entry."""

    manifest = load_json(path)
    external = manifest.get("external_validation", {})
    return {
        "candidate_token_strategy": manifest.get("parameters", {}).get(
            "candidate_token_strategy"
        ),
        "manifest_status": manifest.get("status"),
        "status": external.get(
            "status",
            "coverage_only_no_in_catalog_positive_groups",
        ),
        "positive_ranking_group_count": int(
            external.get("positive_ranking_group_count") or 0
        ),
        "performance_claims_allowed": bool(
            external.get("performance_claims_allowed", False)
        ),
    }


def external_validation_readiness(
    primary_manifest_path: Path,
    sensitivity_manifest_path: Path,
) -> dict[str, Any]:
    """Combine primary and ATC-3 sensitivity readiness without clinical claims."""

    strategies = [
        external_validation_entry(primary_manifest_path),
        external_validation_entry(sensitivity_manifest_path),
    ]
    evaluable = [
        row
        for row in strategies
        if row["positive_ranking_group_count"] > 0
        and row["manifest_status"] == "completed"
    ]
    return {
        "source": "eicu_crd",
        "status": (
            "externally_evaluable"
            if evaluable
            else "coverage_only_no_in_catalog_positive_groups"
        ),
        "performance_claims_allowed": bool(evaluable),
        "evaluable_candidate_token_strategies": [
            row["candidate_token_strategy"] for row in evaluable
        ],
        "strategy_reviews": strategies,
        "claim_boundary": (
            "External performance may be reported only for a completed strategy "
            "with positive eICU ranking groups; otherwise report coverage only."
        ),
    }


def base_manifest(
    config: ModelReadyPackageConfig,
    *,
    status: str,
    generated_at: str,
) -> dict[str, Any]:
    """Return the aggregate package manifest shell."""

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "generated_at": generated_at,
        "versions": {
            "feature_version": config.feature_version or FEATURE_VERSION,
            "graph_version": config.graph_version or GRAPH_VERSION,
            "label_version": config.label_version,
            "split_version": config.split_version,
        },
        "artifacts": {},
        "tables": [],
        "data_safety": {
            "manifest_contains_patient_rows": False,
            "manifest_contains_row_samples": False,
            "local_patient_level_artifacts_are_ignored": True,
            "public_reports_are_aggregate_or_schema_only": True,
        },
        "clinical_claim_boundary": (
            "Artifacts represent historical prescribing for research and clinician "
            "review; they are not validated treatment recommendations."
        ),
    }


def build_model_ready_package(
    config: ModelReadyPackageConfig = ModelReadyPackageConfig(),
) -> dict[str, Any]:
    """Build vocabularies, schema dictionary, and final completion manifest."""

    generated_at = datetime.now(UTC).isoformat()
    missing = missing_inputs(config)
    if missing:
        manifest = base_manifest(
            config,
            status="failed_missing_inputs",
            generated_at=generated_at,
        )
        manifest["missing_inputs"] = missing
        write_json(config.manifest_path, manifest)
        return manifest

    try:
        config = resolve_versions(config)
    except ValueError as error:
        manifest = base_manifest(
            config,
            status="failed_version_mismatch",
            generated_at=generated_at,
        )
        manifest["reason"] = safe_error_message(error)
        write_json(config.manifest_path, manifest)
        return manifest

    dependency_statuses = dependency_manifest_statuses(config)
    if any(row["status"] != "completed" for row in dependency_statuses):
        manifest = base_manifest(
            config,
            status="failed_dependency_status",
            generated_at=generated_at,
        )
        manifest["dependency_manifests"] = dependency_statuses
        write_json(config.manifest_path, manifest)
        return manifest

    config.vocabularies_root.mkdir(parents=True, exist_ok=True)
    manifest = base_manifest(config, status="completed", generated_at=generated_at)
    manifest["dependency_manifests"] = dependency_statuses
    with duckdb.connect(database=":memory:") as connection:
        configure_connection(config, connection)
        fit_scope = train_fit_scope_summary(connection, config)
        manifest["train_fit_scope"] = fit_scope
        if not train_fit_scope_is_valid(fit_scope):
            manifest["status"] = "failed_graph_fit_scope"
            write_json(config.manifest_path, manifest)
            return manifest
        for table_name, (query, output_path) in vocabulary_queries(
            config,
            generated_at=generated_at,
        ).items():
            try:
                row_count = copy_query_to_parquet(connection, query, output_path)
            except Exception as error:
                manifest["status"] = "failed"
                manifest["reason"] = safe_error_message(error)
                break
            manifest["artifacts"][table_name] = str(output_path)
            manifest["tables"].append(
                {
                    "table_name": table_name,
                    "status": "completed",
                    "row_count": row_count,
                }
            )

        if manifest["status"] == "completed":
            dictionary = build_data_dictionary(
                connection,
                config,
                generated_at=generated_at,
            )
            write_json(config.data_dictionary_path, dictionary)
            all_parquet = {
                **config.parquet_artifacts(),
                **config.vocabulary_artifacts(),
            }
            row_counts = {row["table_name"]: row for row in manifest["tables"]}
            for table_name, path in all_parquet.items():
                if table_name in row_counts:
                    continue
                row_count = int(
                    fetch_dict_rows(
                        connection,
                        f"SELECT COUNT(*) AS row_count FROM {parquet_scan(path)}",
                    )[0]["row_count"]
                )
                manifest["tables"].append(
                    {
                        "table_name": table_name,
                        "status": "completed",
                        "row_count": row_count,
                    }
                )
            manifest["artifacts"].update(
                {name: str(path) for name, path in config.parquet_artifacts().items()}
            )
            manifest["artifacts"].update(
                {
                    "train_fitted_preprocessor": str(config.preprocessor_path),
                    "preprocessing_manifest": str(config.preprocessing_manifest_path),
                    "patient_subgraphs_manifest": str(config.subgraphs_manifest_path),
                    "data_dictionary": str(config.data_dictionary_path),
                }
            )
            manifest["external_validation_readiness"] = external_validation_readiness(
                config.primary_training_manifest_path,
                config.sensitivity_training_manifest_path,
            )
            manifest["completion_contract"] = {
                "mimic_development_artifacts_complete": True,
                "patient_level_split_required": True,
                "graph_fit_train_only": True,
                "deferred_p1": [
                    "neural_transformer_gnn_training",
                    "external_ddi_and_ontology_edges",
                    "note_features",
                    "pooled_mimic_eicu_training",
                ],
            }

    write_json(config.manifest_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Assemble the complete Phase 8 P0 model-ready package.",
    )
    parser.add_argument("--features-root", type=Path, default=FEATURES_ROOT)
    parser.add_argument("--training-root", type=Path, default=TRAINING_ROOT)
    parser.add_argument("--graph-root", type=Path, default=MILESTONE8_GRAPH_ROOT)
    parser.add_argument("--subgraphs-root", type=Path, default=DEFAULT_SUBGRAPHS_ROOT)
    parser.add_argument(
        "--preprocessing-root",
        type=Path,
        default=TRAINING_ROOT / "preprocessing",
    )
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument(
        "--data-dictionary",
        type=Path,
        default=DEFAULT_DATA_DICTIONARY_PATH,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
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
        "--preprocessing-manifest",
        type=Path,
        default=DEFAULT_PREPROCESSING_MANIFEST,
    )
    parser.add_argument(
        "--subgraphs-manifest",
        type=Path,
        default=DEFAULT_SUBGRAPHS_MANIFEST,
    )
    parser.add_argument("--feature-version", default=None)
    parser.add_argument("--graph-version", default=None)
    parser.add_argument("--duckdb-temp-dir", type=Path, default=DUCKDB_TEMP_DIR)
    parser.add_argument("--duckdb-memory-limit", default=DUCKDB_MEMORY_LIMIT)
    parser.add_argument("--duckdb-threads", type=int, default=DUCKDB_THREADS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    manifest = build_model_ready_package(
        ModelReadyPackageConfig(
            features_root=args.features_root,
            training_root=args.training_root,
            graph_root=args.graph_root,
            subgraphs_root=args.subgraphs_root,
            preprocessing_root=args.preprocessing_root,
            package_root=args.package_root,
            data_dictionary_path=args.data_dictionary,
            manifest_path=args.manifest,
            primary_training_manifest_path=args.primary_training_manifest,
            sensitivity_training_manifest_path=(args.sensitivity_training_manifest),
            preprocessing_manifest_path=args.preprocessing_manifest,
            subgraphs_manifest_path=args.subgraphs_manifest,
            feature_version=args.feature_version,
            graph_version=args.graph_version,
            duckdb_temp_directory=args.duckdb_temp_dir,
            duckdb_memory_limit=args.duckdb_memory_limit,
            duckdb_threads=args.duckdb_threads,
        )
    )
    print(
        "Wrote Phase 8 P0 model-ready manifest: "
        f"status={manifest['status']}, tables={len(manifest.get('tables', []))}"
    )
    if manifest["status"] == "failed_missing_inputs":
        return 2
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
