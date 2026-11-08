"""Build Milestone 8 graph-readiness artifacts and aggregate reports."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    MILESTONE8_REPORT_VERSION,
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


SCHEMA_REPORT_VERSION = "milestone8-graph-schema-v1"
ABLATION_PLAN_VERSION = "milestone8-ablation-plan-v1"
DEFAULT_SCHEMA_REPORT_PATH = REPORTS_ROOT / "milestone8_graph_schema.json"
DEFAULT_SUITABILITY_REPORT_PATH = REPORTS_ROOT / "milestone8_graph_suitability.json"
DEFAULT_ABLATION_PLAN_PATH = REPORTS_ROOT / "milestone8_ablation_plan.json"

NODE_TYPES = ("condition", "medication", "lab", "vital", "intervention")
RELATION_TYPES = (
    "condition_medication_train_positive",
    "condition_lab_predecision",
    "condition_vital_predecision",
    "condition_intervention_predecision",
    "medication_medication_train_coprescribed",
)
PUBLIC_REPORT_BLOCKED_COLUMNS = frozenset(
    {
        "patient_uid",
        "encounter_uid",
        "stay_uid",
        "ranking_group_id",
        "source_event_id",
        "source_text",
        "value_text",
    }
)


@dataclass(frozen=True)
class GraphSuitabilityConfig:
    """Configuration for Milestone 8 graph-readiness analysis."""

    features_root: Path = FEATURES_ROOT
    training_root: Path = TRAINING_ROOT
    graph_root: Path = MILESTONE8_GRAPH_ROOT
    schema_report_path: Path = DEFAULT_SCHEMA_REPORT_PATH
    suitability_report_path: Path = DEFAULT_SUITABILITY_REPORT_PATH
    ablation_plan_path: Path = DEFAULT_ABLATION_PLAN_PATH
    seed: int = RANDOM_SEED
    graph_version: str = GRAPH_VERSION
    report_version: str = MILESTONE8_REPORT_VERSION
    feature_version: str | None = None
    label_version: str = LABEL_VERSION
    split_version: str = SPLIT_VERSION
    duckdb_temp_directory: Path | None = DUCKDB_TEMP_DIR
    duckdb_memory_limit: str | None = DUCKDB_MEMORY_LIMIT
    duckdb_threads: int | None = DUCKDB_THREADS

    @property
    def event_sequences_path(self) -> Path:
        return self.features_root / "event_sequences.parquet"

    @property
    def patient_condition_medication_path(self) -> Path:
        return self.training_root / "patient_condition_medication.parquet"

    @property
    def candidate_catalog_path(self) -> Path:
        return self.training_root / "candidate_catalog.parquet"

    @property
    def graph_edges_path(self) -> Path:
        return self.graph_root / "graph_edges.parquet"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def configure_connection(
    config: GraphSuitabilityConfig,
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Apply shared memory-safe DuckDB settings."""

    configure_duckdb_connection(
        connection,
        temp_directory=config.duckdb_temp_directory,
        memory_limit=config.duckdb_memory_limit,
        threads=config.duckdb_threads,
    )


def missing_input_tables(config: GraphSuitabilityConfig) -> list[dict[str, str]]:
    """Return missing Milestone 6 inputs required for graph readiness."""

    required = (
        ("event_sequences", config.event_sequences_path),
        ("patient_condition_medication", config.patient_condition_medication_path),
        ("candidate_catalog", config.candidate_catalog_path),
    )
    return [
        {"table_name": table_name, "path": str(path)}
        for table_name, path in required
        if not path.exists()
    ]


def resolve_feature_version(
    config: GraphSuitabilityConfig,
) -> GraphSuitabilityConfig:
    """Return config stamped from sequence and ranking-table inputs."""

    version = infer_consistent_version(
        (config.event_sequences_path, config.patient_condition_medication_path),
        column_name="feature_version",
        declared_version=config.feature_version,
        fallback_version=FEATURE_VERSION,
    )
    return replace(config, feature_version=version)


def schema_report_shell(
    config: GraphSuitabilityConfig,
    *,
    generated_at: str,
) -> dict[str, Any]:
    """Return the aggregate graph schema report shell."""

    return {
        "schema_version": SCHEMA_REPORT_VERSION,
        "status": "running",
        "generated_at": generated_at,
        "graph_artifacts": {
            "graph_edges": str(config.graph_edges_path),
        },
        "node_types": [
            {"node_type": node_type, "node_id_format": f"{node_type}|<token>"}
            for node_type in NODE_TYPES
        ],
        "relation_types": list(RELATION_TYPES),
        "edge_schema": [
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
            "generated_at",
        ],
        "leakage_policy": {
            "fit_source": "mimiciv",
            "fit_split": "train",
            "validation_test_external_use": "coverage_only",
            "predecision_event_policy": "uses Milestone 6 event_sequences only",
        },
        "eicu_policy": (
            "eICU remains coverage-only until in-catalog positive groups are "
            "available for external performance evaluation."
        ),
        "deferred_sources": [
            "external_ddi",
            "ontology_edges",
            "note_embeddings",
            "clinical_rule_edges",
        ],
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "local_graph_artifacts_are_patient_level": False,
            "local_graph_artifact_storage": "ignored Dataset/processed/graph",
        },
        "versions": {
            "graph_version": config.graph_version,
            "report_version": config.report_version,
            "feature_version": config.feature_version or FEATURE_VERSION,
            "label_version": config.label_version,
            "split_version": config.split_version,
        },
    }


def graph_edges_query(config: GraphSuitabilityConfig, *, generated_at: str) -> str:
    """Return SQL for train-only concept-level graph edges."""

    pcm = config.patient_condition_medication_path
    events = config.event_sequences_path
    return f"""
WITH train_positive_rows AS (
    SELECT
        source,
        split,
        stay_uid,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token
    FROM {parquet_scan(pcm)}
    WHERE source = 'mimiciv'
        AND split = 'train'
        AND label_prescribed
        AND index_condition_token IS NOT NULL
        AND candidate_medication_token IS NOT NULL
),
train_stay_conditions AS (
    SELECT DISTINCT
        source,
        split,
        stay_uid,
        index_condition_token
    FROM {parquet_scan(pcm)}
    WHERE source = 'mimiciv'
        AND split = 'train'
        AND index_condition_token IS NOT NULL
),
condition_medication_edges AS (
    SELECT
        'condition|' || index_condition_token AS src_id,
        'medication|' || candidate_medication_token AS dst_id,
        'condition' AS src_type,
        'medication' AS dst_type,
        'condition_medication_train_positive' AS relation_type,
        COUNT(DISTINCT ranking_group_id) AS support_count
    FROM train_positive_rows
    GROUP BY index_condition_token, candidate_medication_token
),
medication_pairs AS (
    SELECT
        a.candidate_medication_token AS src_medication_token,
        b.candidate_medication_token AS dst_medication_token,
        a.ranking_group_id
    FROM train_positive_rows AS a
    INNER JOIN train_positive_rows AS b
        ON a.ranking_group_id = b.ranking_group_id
        AND a.candidate_medication_token < b.candidate_medication_token
),
medication_medication_edges AS (
    SELECT
        'medication|' || src_medication_token AS src_id,
        'medication|' || dst_medication_token AS dst_id,
        'medication' AS src_type,
        'medication' AS dst_type,
        'medication_medication_train_coprescribed' AS relation_type,
        COUNT(DISTINCT ranking_group_id) AS support_count
    FROM medication_pairs
    GROUP BY src_medication_token, dst_medication_token
),
predecision_events AS (
    SELECT DISTINCT
        source,
        split,
        stay_uid,
        event_type,
        event_token
    FROM {parquet_scan(events)}
    WHERE source = 'mimiciv'
        AND split = 'train'
        AND event_type IN ('lab', 'vital', 'intervention')
        AND event_time_hours_from_admit >= 0
        AND event_time_hours_from_admit <= 24.0
        AND event_token IS NOT NULL
),
condition_event_edges AS (
    SELECT
        'condition|' || c.index_condition_token AS src_id,
        e.event_type || '|' || e.event_token AS dst_id,
        'condition' AS src_type,
        e.event_type AS dst_type,
        CASE e.event_type
            WHEN 'lab' THEN 'condition_lab_predecision'
            WHEN 'vital' THEN 'condition_vital_predecision'
            ELSE 'condition_intervention_predecision'
        END AS relation_type,
        COUNT(DISTINCT c.stay_uid) AS support_count
    FROM train_stay_conditions AS c
    INNER JOIN predecision_events AS e
        ON c.source = e.source
        AND c.split = e.split
        AND c.stay_uid = e.stay_uid
    GROUP BY c.index_condition_token, e.event_type, e.event_token
),
all_edges AS (
    SELECT * FROM condition_medication_edges
    UNION ALL
    SELECT * FROM condition_event_edges
    UNION ALL
    SELECT * FROM medication_medication_edges
)
SELECT
    src_id,
    dst_id,
    src_type,
    dst_type,
    relation_type,
    CAST(support_count AS BIGINT) AS support_count,
    'mimiciv' AS fit_source,
    'train' AS fit_split,
    {sql_string(config.graph_version)} AS graph_version,
    {sql_string(config.feature_version or FEATURE_VERSION)} AS feature_version,
    {sql_string(config.label_version)} AS label_version,
    {sql_string(config.split_version)} AS split_version,
    {sql_string(generated_at)} AS generated_at
FROM all_edges
"""


def edge_counts_query(config: GraphSuitabilityConfig) -> str:
    """Return edge-count aggregates by relation."""

    edges = config.graph_edges_path
    return f"""
SELECT
    relation_type,
    src_type,
    dst_type,
    COUNT(*) AS edge_count,
    SUM(support_count) AS total_support_count,
    MIN(support_count) AS min_support_count,
    MAX(support_count) AS max_support_count,
    AVG(support_count) AS mean_support_count
FROM {parquet_scan(edges)}
GROUP BY relation_type, src_type, dst_type
ORDER BY relation_type
"""


def node_counts_query(config: GraphSuitabilityConfig) -> str:
    """Return node-count aggregates by node type."""

    edges = config.graph_edges_path
    return f"""
WITH nodes AS (
    SELECT src_type AS node_type, src_id AS node_id
    FROM {parquet_scan(edges)}
    UNION
    SELECT dst_type AS node_type, dst_id AS node_id
    FROM {parquet_scan(edges)}
)
SELECT node_type, COUNT(DISTINCT node_id) AS node_count
FROM nodes
GROUP BY node_type
ORDER BY node_type
"""


def degree_summary_query(config: GraphSuitabilityConfig) -> str:
    """Return node degree summaries without exposing node IDs."""

    edges = config.graph_edges_path
    return f"""
WITH endpoints AS (
    SELECT src_type AS node_type, src_id AS node_id, dst_id AS neighbor_id
    FROM {parquet_scan(edges)}
    UNION ALL
    SELECT dst_type AS node_type, dst_id AS node_id, src_id AS neighbor_id
    FROM {parquet_scan(edges)}
),
degrees AS (
    SELECT
        node_type,
        node_id,
        COUNT(DISTINCT neighbor_id) AS degree_count
    FROM endpoints
    GROUP BY node_type, node_id
)
SELECT
    node_type,
    COUNT(*) AS node_count,
    MIN(degree_count) AS min_degree,
    MAX(degree_count) AS max_degree,
    AVG(degree_count) AS mean_degree
FROM degrees
GROUP BY node_type
ORDER BY node_type
"""


def coverage_query(config: GraphSuitabilityConfig) -> str:
    """Return split-level graph coverage and cold-start aggregates."""

    edges = config.graph_edges_path
    pcm = config.patient_condition_medication_path
    return f"""
WITH graph_nodes AS (
    SELECT src_type AS node_type, src_id AS node_id
    FROM {parquet_scan(edges)}
    UNION
    SELECT dst_type AS node_type, dst_id AS node_id
    FROM {parquet_scan(edges)}
),
condition_medication_edges AS (
    SELECT src_id, dst_id
    FROM {parquet_scan(edges)}
    WHERE relation_type = 'condition_medication_train_positive'
),
rows AS (
    SELECT
        source,
        split,
        index_condition_token,
        candidate_medication_token,
        label_prescribed
    FROM {parquet_scan(pcm)}
    WHERE index_condition_token IS NOT NULL
        AND candidate_medication_token IS NOT NULL
)
SELECT
    rows.source,
    rows.split,
    COUNT(*) AS candidate_row_count,
    COUNT(DISTINCT rows.index_condition_token) AS condition_count,
    COUNT(DISTINCT rows.candidate_medication_token) AS candidate_medication_count,
    COUNT(DISTINCT CASE
        WHEN condition_nodes.node_id IS NULL THEN rows.index_condition_token
    END) AS unseen_condition_count,
    COUNT(DISTINCT CASE
        WHEN medication_nodes.node_id IS NULL THEN rows.candidate_medication_token
    END) AS unseen_candidate_medication_count,
    SUM(CASE WHEN rows.label_prescribed THEN 1 ELSE 0 END) AS positive_row_count,
    SUM(CASE
        WHEN rows.label_prescribed
            AND condition_medication_edges.src_id IS NOT NULL
            THEN 1
        ELSE 0
    END) AS in_graph_positive_row_count
FROM rows
LEFT JOIN graph_nodes AS condition_nodes
    ON condition_nodes.node_type = 'condition'
    AND condition_nodes.node_id = 'condition|' || rows.index_condition_token
LEFT JOIN graph_nodes AS medication_nodes
    ON medication_nodes.node_type = 'medication'
    AND medication_nodes.node_id = 'medication|' || rows.candidate_medication_token
LEFT JOIN condition_medication_edges
    ON condition_medication_edges.src_id =
        'condition|' || rows.index_condition_token
    AND condition_medication_edges.dst_id =
        'medication|' || rows.candidate_medication_token
GROUP BY rows.source, rows.split
ORDER BY rows.source, rows.split
"""


def leakage_audit_query(config: GraphSuitabilityConfig) -> str:
    """Return the source/split used by graph edges."""

    edges = config.graph_edges_path
    return f"""
SELECT fit_source, fit_split, COUNT(*) AS edge_count
FROM {parquet_scan(edges)}
GROUP BY fit_source, fit_split
ORDER BY fit_source, fit_split
"""


def graph_columns(
    connection: duckdb.DuckDBPyConnection,
    config: GraphSuitabilityConfig,
) -> list[str]:
    """Return materialized graph-edge column names."""

    cursor = connection.execute(
        f"DESCRIBE SELECT * FROM {parquet_scan(config.graph_edges_path)}"
    )
    return [str(row[0]) for row in cursor.fetchall()]


def connected_component_summary(
    connection: duckdb.DuckDBPyConnection,
    config: GraphSuitabilityConfig,
) -> dict[str, int]:
    """Compute connected-component aggregate statistics from concept edges."""

    rows = connection.execute(
        f"""
SELECT src_id, dst_id
FROM {parquet_scan(config.graph_edges_path)}
"""
    ).fetchall()
    adjacency: dict[str, set[str]] = defaultdict(set)
    for src_id, dst_id in rows:
        src = str(src_id)
        dst = str(dst_id)
        adjacency[src].add(dst)
        adjacency[dst].add(src)

    visited: set[str] = set()
    component_sizes: list[int] = []
    for node_id in adjacency:
        if node_id in visited:
            continue
        queue: deque[str] = deque([node_id])
        visited.add(node_id)
        size = 0
        while queue:
            node = queue.popleft()
            size += 1
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        component_sizes.append(size)

    return {
        "component_count": len(component_sizes),
        "connected_node_count": sum(component_sizes),
        "largest_component_node_count": max(component_sizes, default=0),
        "singleton_component_count": sum(1 for size in component_sizes if size == 1),
    }


def sparsity_summary(
    node_counts: list[dict[str, Any]],
    edge_counts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute coarse density summaries for each relation type."""

    nodes = {row["node_type"]: int(row["node_count"]) for row in node_counts}
    rows: list[dict[str, Any]] = []
    for edge_row in edge_counts:
        relation = str(edge_row["relation_type"])
        src_type = str(edge_row["src_type"])
        dst_type = str(edge_row["dst_type"])
        edge_count = int(edge_row["edge_count"])
        if src_type == dst_type:
            n = nodes.get(src_type, 0)
            possible_edges = n * (n - 1) // 2
        else:
            possible_edges = nodes.get(src_type, 0) * nodes.get(dst_type, 0)
        density = (edge_count / possible_edges) if possible_edges else None
        rows.append(
            {
                "relation_type": relation,
                "edge_count": edge_count,
                "possible_edge_count": possible_edges,
                "density": density,
            }
        )
    return rows


def coverage_with_rates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add coverage rates to split-level cold-start aggregates."""

    rated: list[dict[str, Any]] = []
    for row in rows:
        candidate_med_count = int(row["candidate_medication_count"] or 0)
        condition_count = int(row["condition_count"] or 0)
        positive_count = int(row["positive_row_count"] or 0)
        unseen_med_count = int(row["unseen_candidate_medication_count"] or 0)
        unseen_condition_count = int(row["unseen_condition_count"] or 0)
        in_graph_positive_count = int(row["in_graph_positive_row_count"] or 0)
        enriched = dict(row)
        enriched["candidate_medication_cold_start_rate"] = (
            unseen_med_count / candidate_med_count if candidate_med_count else None
        )
        enriched["condition_cold_start_rate"] = (
            unseen_condition_count / condition_count if condition_count else None
        )
        enriched["positive_graph_coverage_rate"] = (
            in_graph_positive_count / positive_count if positive_count else None
        )
        rated.append(enriched)
    return rated


def leakage_audit_status(
    rows: list[dict[str, Any]],
    columns: Sequence[str],
) -> dict[str, Any]:
    """Return a disclosure-conscious leakage audit summary."""

    train_only = all(
        row["fit_source"] == "mimiciv" and row["fit_split"] == "train" for row in rows
    )
    blocked_columns = sorted(PUBLIC_REPORT_BLOCKED_COLUMNS.intersection(columns))
    return {
        "fit_source_split_counts": rows,
        "train_only_graph_fit": train_only,
        "blocked_identifier_columns_present": blocked_columns,
        "status": "pass" if train_only and not blocked_columns else "fail",
    }


def graph_gate(
    *,
    edge_row_count: int,
    edge_counts: list[dict[str, Any]],
    components: dict[str, int],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Return a conservative graph-readiness gate result."""

    relation_counts = {
        row["relation_type"]: int(row["edge_count"]) for row in edge_counts
    }
    required_relation_count = relation_counts.get(
        "condition_medication_train_positive",
        0,
    )
    passes = (
        edge_row_count > 0
        and required_relation_count > 0
        and components["largest_component_node_count"] > 1
        and audit["status"] == "pass"
    )
    if passes:
        result = "pass_for_graph_ablation"
        next_action = "Proceed to Milestone 8B graph-only and fusion ablation planning."
    else:
        result = "blocked_graph_not_ready"
        next_action = (
            "Stop at graph-readiness reporting and prioritize mapping, catalog, "
            "or temporal-artifact fixes."
        )
    return {
        "result": result,
        "next_action": next_action,
        "criteria": {
            "has_edges": edge_row_count > 0,
            "has_condition_medication_edges": required_relation_count > 0,
            "has_connected_component": components["largest_component_node_count"] > 1,
            "leakage_audit_passed": audit["status"] == "pass",
        },
        "clinical_claim_boundary": (
            "This gate supports graph-readiness only. It is not a clinical "
            "recommendation or Transformer-GNN performance result."
        ),
    }


def build_ablation_plan(
    config: GraphSuitabilityConfig,
    *,
    generated_at: str,
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Return the aggregate Milestone 8B ablation plan shell."""

    return {
        "schema_version": ABLATION_PLAN_VERSION,
        "status": "completed",
        "generated_at": generated_at,
        "graph_gate_result": gate["result"],
        "recommended_next_action": gate["next_action"],
        "ablation_sequence": [
            "xgboost_frozen_baseline_reference",
            "graph_only_relation_features_or_gnn",
            "tabular_sequence_transformer_only",
            "late_fusion_graph_plus_context",
            "simple_ensemble_against_xgboost",
        ],
        "required_before_training": [
            "final Milestone 7 test evaluation recorded",
            "graph leakage audit passes",
            "candidate and condition coverage reviewed",
            "no eICU performance claim unless in-catalog positives exist",
        ],
        "deferred_inputs": [
            "external DDI edges",
            "ontology edges",
            "note embeddings",
            "rule-source contraindication edges",
        ],
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
        },
        "versions": {
            "graph_version": config.graph_version,
            "report_version": config.report_version,
        },
    }


def failed_reports(
    config: GraphSuitabilityConfig,
    *,
    generated_at: str,
    missing: list[dict[str, str]],
) -> dict[str, Any]:
    """Write safe failed reports for missing-input cases."""

    schema = schema_report_shell(config, generated_at=generated_at)
    schema["status"] = "failed_missing_inputs"
    schema["missing_inputs"] = missing
    suitability = {
        "schema_version": config.report_version,
        "status": "failed_missing_inputs",
        "generated_at": generated_at,
        "missing_inputs": missing,
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
        },
    }
    ablation = build_ablation_plan(
        config,
        generated_at=generated_at,
        gate={
            "result": "blocked_missing_inputs",
            "next_action": "Build Milestone 6 feature and training artifacts first.",
        },
    )
    ablation["status"] = "blocked_missing_inputs"
    write_json(config.schema_report_path, schema)
    write_json(config.suitability_report_path, suitability)
    write_json(config.ablation_plan_path, ablation)
    return suitability


def failed_feature_version_report(
    config: GraphSuitabilityConfig,
    *,
    generated_at: str,
    reason: str,
) -> dict[str, Any]:
    """Write aggregate-only reports for inconsistent provenance stamps."""

    schema = schema_report_shell(config, generated_at=generated_at)
    schema.update(status="failed_feature_version_mismatch", reason=reason)
    suitability = {
        "schema_version": config.report_version,
        "status": "failed_feature_version_mismatch",
        "generated_at": generated_at,
        "reason": reason,
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
        },
    }
    ablation = build_ablation_plan(
        config,
        generated_at=generated_at,
        gate={
            "result": "blocked_feature_version_mismatch",
            "next_action": "Rebuild inputs with one consistent feature version.",
        },
    )
    ablation["status"] = "blocked_feature_version_mismatch"
    write_json(config.schema_report_path, schema)
    write_json(config.suitability_report_path, suitability)
    write_json(config.ablation_plan_path, ablation)
    return suitability


def build_graph_suitability(config: GraphSuitabilityConfig) -> dict[str, Any]:
    """Build Milestone 8 graph-readiness artifacts and aggregate reports."""

    generated_at = datetime.now(UTC).isoformat()
    missing = missing_input_tables(config)
    if missing:
        return failed_reports(config, generated_at=generated_at, missing=missing)

    try:
        config = resolve_feature_version(config)
    except ValueError as error:
        return failed_feature_version_report(
            config,
            generated_at=generated_at,
            reason=safe_error_message(error),
        )

    schema = schema_report_shell(config, generated_at=generated_at)
    config.graph_root.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(database=":memory:") as connection:
        configure_connection(config, connection)
        try:
            edge_row_count = copy_query_to_parquet(
                connection,
                graph_edges_query(config, generated_at=generated_at),
                config.graph_edges_path,
            )
            edge_counts = fetch_dict_rows(connection, edge_counts_query(config))
            node_counts = fetch_dict_rows(connection, node_counts_query(config))
            degree_summary = fetch_dict_rows(connection, degree_summary_query(config))
            split_coverage = coverage_with_rates(
                fetch_dict_rows(connection, coverage_query(config))
            )
            components = connected_component_summary(connection, config)
            audit = leakage_audit_status(
                fetch_dict_rows(connection, leakage_audit_query(config)),
                graph_columns(connection, config),
            )
            sparsity = sparsity_summary(node_counts, edge_counts)
            gate = graph_gate(
                edge_row_count=edge_row_count,
                edge_counts=edge_counts,
                components=components,
                audit=audit,
            )
            suitability = {
                "schema_version": config.report_version,
                "status": "completed",
                "generated_at": generated_at,
                "artifacts": {
                    "graph_edges": str(config.graph_edges_path),
                },
                "tables": [
                    {
                        "table_name": "graph_edges",
                        "status": "completed",
                        "row_count": edge_row_count,
                    }
                ],
                "node_counts": node_counts,
                "edge_counts_by_relation": edge_counts,
                "degree_summary": degree_summary,
                "connected_components": components,
                "sparsity": sparsity,
                "split_coverage_and_cold_start": split_coverage,
                "leakage_audit": audit,
                "gate_review": gate,
                "data_safety": {
                    "report_contains_patient_rows": False,
                    "report_contains_row_samples": False,
                    "local_graph_artifact_storage": "ignored Dataset/processed/graph",
                    "local_graph_artifacts_contain_patient_rows": False,
                },
                "versions": {
                    "graph_version": config.graph_version,
                    "report_version": config.report_version,
                    "feature_version": config.feature_version or FEATURE_VERSION,
                    "label_version": config.label_version,
                    "split_version": config.split_version,
                },
            }
            schema["status"] = "completed"
            schema["edge_artifact_row_count"] = edge_row_count
            ablation = build_ablation_plan(
                config,
                generated_at=generated_at,
                gate=gate,
            )
        except Exception as error:
            schema["status"] = "failed"
            schema["reason"] = safe_error_message(error)
            suitability = {
                "schema_version": config.report_version,
                "status": "failed",
                "generated_at": generated_at,
                "reason": safe_error_message(error),
                "data_safety": {
                    "report_contains_patient_rows": False,
                    "report_contains_row_samples": False,
                },
            }
            ablation = build_ablation_plan(
                config,
                generated_at=generated_at,
                gate={
                    "result": "blocked_graph_build_failed",
                    "next_action": "Fix graph suitability build failure first.",
                },
            )
            ablation["status"] = "blocked_graph_build_failed"

    write_json(config.schema_report_path, schema)
    write_json(config.suitability_report_path, suitability)
    write_json(config.ablation_plan_path, ablation)
    return suitability


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Build Milestone 8 graph-readiness artifacts and reports.",
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
        help="Output directory for local ignored Milestone 8 graph artifacts.",
    )
    parser.add_argument(
        "--schema-report",
        "--graph-schema-report",
        type=Path,
        default=DEFAULT_SCHEMA_REPORT_PATH,
        dest="schema_report",
        help="Output path for the aggregate graph-schema report.",
    )
    parser.add_argument(
        "--suitability-report",
        type=Path,
        default=DEFAULT_SUITABILITY_REPORT_PATH,
        help="Output path for the aggregate graph-suitability report.",
    )
    parser.add_argument(
        "--ablation-plan",
        type=Path,
        default=DEFAULT_ABLATION_PLAN_PATH,
        help="Output path for the aggregate Milestone 8B ablation plan.",
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
    """CLI entrypoint."""

    args = parse_args(argv)
    report = build_graph_suitability(
        GraphSuitabilityConfig(
            features_root=args.features_root,
            training_root=args.training_root,
            graph_root=args.graph_root,
            schema_report_path=args.schema_report,
            suitability_report_path=args.suitability_report,
            ablation_plan_path=args.ablation_plan,
            feature_version=args.feature_version,
            duckdb_temp_directory=args.duckdb_temp_dir,
            duckdb_memory_limit=args.duckdb_memory_limit,
            duckdb_threads=args.duckdb_threads,
        )
    )
    print(
        "Wrote Milestone 8 graph-suitability report: "
        f"status={report['status']}, "
        f"report={args.suitability_report}"
    )
    if report["status"] == "failed_missing_inputs":
        return 2
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
