"""Build leakage-safe patient query subgraphs from train-fit concept edges."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import duckdb

from pipeline.artifact_metadata import infer_consistent_version
from pipeline.config import (
    DEFAULT_MODELING_PARAMETERS,
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
from pipeline.features import copy_query_to_parquet, fetch_dict_rows
from pipeline.features import parquet_scan_paths


SCHEMA_VERSION = "patient-subgraph-manifest-v1"
DEFAULT_SUBGRAPHS_ROOT = MILESTONE8_GRAPH_ROOT / "patient_subgraphs"
DEFAULT_MANIFEST_PATH = REPORTS_ROOT / "patient_subgraphs_manifest.json"
CONTEXT_EVENT_TYPES = ("lab", "vital", "intervention")
DEFAULT_SUBGRAPH_BATCH_COUNT = 8
DEFAULT_SUBGRAPH_JOIN_SHARDS = 8
DEFAULT_SUBGRAPH_EDGE_THREADS = 1


@dataclass(frozen=True)
class PatientSubgraphBuildConfig:
    """Configuration for one query subgraph per ranking group."""

    features_root: Path = FEATURES_ROOT
    training_root: Path = TRAINING_ROOT
    graph_root: Path = MILESTONE8_GRAPH_ROOT
    subgraphs_root: Path = DEFAULT_SUBGRAPHS_ROOT
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    prediction_offset_hours: int = int(
        DEFAULT_MODELING_PARAMETERS["prediction_offset_hours"]
    )
    feature_version: str | None = None
    graph_version: str | None = None
    label_version: str = LABEL_VERSION
    split_version: str = SPLIT_VERSION
    duckdb_temp_directory: Path | None = DUCKDB_TEMP_DIR
    duckdb_memory_limit: str | None = DUCKDB_MEMORY_LIMIT
    duckdb_threads: int | None = DUCKDB_THREADS
    subgraph_batch_count: int = DEFAULT_SUBGRAPH_BATCH_COUNT
    subgraph_join_shards: int = DEFAULT_SUBGRAPH_JOIN_SHARDS
    edge_duckdb_threads: int = DEFAULT_SUBGRAPH_EDGE_THREADS

    @property
    def event_sequences_path(self) -> Path:
        return self.features_root / "event_sequences.parquet"

    @property
    def patient_condition_medication_path(self) -> Path:
        return self.training_root / "patient_condition_medication.parquet"

    @property
    def graph_edges_path(self) -> Path:
        return self.graph_root / "graph_edges.parquet"

    @property
    def subgraph_index_path(self) -> Path:
        return self.subgraphs_root / "subgraph_index.parquet"

    @property
    def subgraph_nodes_path(self) -> Path:
        return self.subgraphs_root / "subgraph_nodes.parquet"

    @property
    def subgraph_edges_path(self) -> Path:
        return self.subgraphs_root / "subgraph_edges.parquet"

    @property
    def subgraph_candidates_path(self) -> Path:
        return self.subgraphs_root / "subgraph_candidates.parquet"


def subgraph_part_path(
    config: PatientSubgraphBuildConfig,
    *,
    table_name: str,
    batch_index: int,
) -> Path:
    """Return one internal hash-batch part path for a subgraph table."""

    return (
        config.subgraphs_root
        / "_subgraph_parts"
        / table_name
        / f"{table_name}_part_{batch_index:04d}.parquet"
    )


def subgraph_join_part_paths(
    config: PatientSubgraphBuildConfig,
    *,
    table_name: str,
    node_batch_index: int,
) -> tuple[Path, ...]:
    """Return all join-shard parts belonging to one node batch."""

    shard_count = subgraph_join_shard_count(config)
    first_part_index = node_batch_index * shard_count
    return tuple(
        subgraph_part_path(
            config,
            table_name=table_name,
            batch_index=first_part_index + shard_index,
        )
        for shard_index in range(shard_count)
    )


def hash_batch_filter_sql(
    *,
    expression: str,
    batch_index: int,
    batch_count: int,
) -> str:
    """Return a deterministic DuckDB hash-batch predicate."""

    return (
        f"(HASH(COALESCE(CAST({expression} AS VARCHAR), '')) "
        f"% {int(batch_count)}) = {int(batch_index)}"
    )


def subgraph_batch_count(config: PatientSubgraphBuildConfig) -> int:
    """Return a positive patient-subgraph batch count."""

    return max(1, int(config.subgraph_batch_count))


def subgraph_join_shard_count(config: PatientSubgraphBuildConfig) -> int:
    """Return a positive shard count for node-membership joins."""

    return max(1, int(config.subgraph_join_shards))


def edge_duckdb_thread_count(config: PatientSubgraphBuildConfig) -> int:
    """Return a positive edge-assembly thread cap."""

    return max(1, int(config.edge_duckdb_threads))


def stay_hash_batch_filter_sql(
    *,
    table_alias: str,
    batch_index: int,
    batch_count: int,
) -> str:
    """Partition source-qualified stays without splitting ranking groups."""

    return hash_batch_filter_sql(
        expression=(
            f"COALESCE({table_alias}.source, '') || '|' || "
            f"COALESCE(CAST({table_alias}.stay_uid AS VARCHAR), '')"
        ),
        batch_index=batch_index,
        batch_count=batch_count,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable aggregate JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def configure_connection(
    config: PatientSubgraphBuildConfig,
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Apply shared memory-safe DuckDB settings."""

    configure_duckdb_connection(
        connection,
        temp_directory=config.duckdb_temp_directory,
        memory_limit=config.duckdb_memory_limit,
        threads=config.duckdb_threads,
    )


def missing_input_tables(
    config: PatientSubgraphBuildConfig,
) -> list[dict[str, str]]:
    """Return missing model-ready and train-fit graph inputs."""

    required = (
        ("event_sequences", config.event_sequences_path),
        (
            "patient_condition_medication",
            config.patient_condition_medication_path,
        ),
        ("graph_edges", config.graph_edges_path),
    )
    return [
        {"table_name": table_name, "path": str(path)}
        for table_name, path in required
        if not path.exists()
    ]


def resolve_versions(
    config: PatientSubgraphBuildConfig,
) -> PatientSubgraphBuildConfig:
    """Return config stamped from the actual input artifacts."""

    feature_version = infer_consistent_version(
        (
            config.event_sequences_path,
            config.patient_condition_medication_path,
            config.graph_edges_path,
        ),
        column_name="feature_version",
        declared_version=config.feature_version,
        fallback_version=FEATURE_VERSION,
    )
    graph_version = infer_consistent_version(
        (config.graph_edges_path,),
        column_name="graph_version",
        declared_version=config.graph_version,
        fallback_version=GRAPH_VERSION,
    )
    return replace(
        config,
        feature_version=feature_version,
        graph_version=graph_version,
    )


def graph_fit_scope_query(config: PatientSubgraphBuildConfig) -> str:
    """Return aggregate graph fit-scope counts."""

    return f"""
SELECT
    fit_source,
    fit_split,
    COUNT(*) AS edge_count
FROM {parquet_scan(config.graph_edges_path)}
GROUP BY fit_source, fit_split
ORDER BY fit_source, fit_split
"""


def graph_fit_scope_is_valid(rows: Sequence[dict[str, Any]]) -> bool:
    """Return whether every input edge was fit on MIMIC train only."""

    return all(
        row["fit_source"] == "mimiciv" and row["fit_split"] == "train" for row in rows
    )


def subgraph_nodes_query(
    config: PatientSubgraphBuildConfig,
    *,
    generated_at: str,
    batch_index: int | None = None,
    batch_count: int = 1,
) -> str:
    """Build query, candidate, and observed connected context nodes."""

    pcm = config.patient_condition_medication_path
    events = config.event_sequences_path
    edges = config.graph_edges_path
    event_types = ", ".join(sql_string(value) for value in CONTEXT_EVENT_TYPES)
    groups_batch_filter = (
        "TRUE"
        if batch_index is None
        else stay_hash_batch_filter_sql(
            table_alias="pcm",
            batch_index=batch_index,
            batch_count=batch_count,
        )
    )
    candidates_batch_filter = groups_batch_filter
    events_batch_filter = (
        "TRUE"
        if batch_index is None
        else stay_hash_batch_filter_sql(
            table_alias="events",
            batch_index=batch_index,
            batch_count=batch_count,
        )
    )
    return f"""
WITH groups AS (
    SELECT DISTINCT
        source,
        split,
        stay_uid,
        ranking_group_id,
        index_condition_token
    FROM {parquet_scan(pcm)} AS pcm
    WHERE pcm.ranking_group_id IS NOT NULL
        AND pcm.index_condition_token IS NOT NULL
        AND {groups_batch_filter}
),
graph_nodes AS (
    SELECT src_id AS node_id FROM {parquet_scan(edges)}
    UNION
    SELECT dst_id AS node_id FROM {parquet_scan(edges)}
),
condition_nodes AS (
    SELECT
        source,
        split,
        stay_uid,
        ranking_group_id AS subgraph_id,
        'condition|' || index_condition_token AS node_id,
        'condition' AS node_type,
        'query_condition' AS node_role,
        FALSE AS observed_predecision
    FROM groups
),
candidate_nodes AS (
    SELECT DISTINCT
        pcm.source,
        pcm.split,
        pcm.stay_uid,
        pcm.ranking_group_id AS subgraph_id,
        'medication|' || pcm.candidate_medication_token AS node_id,
        'medication' AS node_type,
        'candidate_medication' AS node_role,
        FALSE AS observed_predecision
    FROM {parquet_scan(pcm)} AS pcm
    WHERE pcm.ranking_group_id IS NOT NULL
        AND pcm.candidate_medication_token IS NOT NULL
        AND {candidates_batch_filter}
),
context_nodes AS (
    SELECT DISTINCT
        groups.source,
        groups.split,
        groups.stay_uid,
        groups.ranking_group_id AS subgraph_id,
        events.event_type || '|' || events.event_token AS node_id,
        events.event_type AS node_type,
        'observed_context' AS node_role,
        TRUE AS observed_predecision
    FROM groups
    INNER JOIN {parquet_scan(events)} AS events
        ON groups.source = events.source
        AND groups.stay_uid = events.stay_uid
    INNER JOIN {parquet_scan(edges)} AS graph_edges
        ON graph_edges.src_id = 'condition|' || groups.index_condition_token
        AND graph_edges.dst_id = events.event_type || '|' || events.event_token
        AND graph_edges.fit_source = 'mimiciv'
        AND graph_edges.fit_split = 'train'
    WHERE events.event_type IN ({event_types})
        AND events.event_token IS NOT NULL
        AND events.event_time_hours_from_admit >= 0
        AND events.event_time_hours_from_admit <= {config.prediction_offset_hours}
        AND {events_batch_filter}
),
all_nodes AS (
    SELECT * FROM condition_nodes
    UNION ALL
    SELECT * FROM candidate_nodes
    UNION ALL
    SELECT * FROM context_nodes
),
numbered AS (
    SELECT
        all_nodes.*,
        graph_nodes.node_id IS NOT NULL AS in_train_graph,
        ROW_NUMBER() OVER (
            PARTITION BY subgraph_id
            ORDER BY
                CASE node_role
                    WHEN 'query_condition' THEN 0
                    WHEN 'candidate_medication' THEN 1
                    ELSE 2
                END,
                node_type,
                node_id
        ) - 1 AS node_index
    FROM all_nodes
    LEFT JOIN graph_nodes USING (node_id)
)
SELECT
    source,
    split,
    stay_uid,
    subgraph_id,
    CAST(node_index AS BIGINT) AS node_index,
    node_id,
    node_type,
    node_role,
    observed_predecision,
    in_train_graph,
    NOT in_train_graph AS cold_start,
    {sql_string(config.feature_version or FEATURE_VERSION)} AS feature_version,
    {sql_string(config.graph_version or GRAPH_VERSION)} AS graph_version,
    {sql_string(config.split_version)} AS split_version,
    {sql_string(generated_at)} AS generated_at
FROM numbered
"""


def subgraph_edges_query(
    config: PatientSubgraphBuildConfig,
    *,
    generated_at: str,
    nodes_path: Path | None = None,
    join_shard_index: int | None = None,
    join_shard_count: int = 1,
) -> str:
    """Attach train-fit edges through narrow integer node memberships."""

    nodes = nodes_path or config.subgraph_nodes_path
    shard_filter = (
        "TRUE"
        if join_shard_index is None
        else hash_batch_filter_sql(
            expression="nodes.subgraph_id",
            batch_index=join_shard_index,
            batch_count=join_shard_count,
        )
    )
    return f"""
WITH filtered_nodes AS MATERIALIZED (
    SELECT
        source,
        split,
        stay_uid,
        subgraph_id,
        node_index,
        node_id,
        node_role
    FROM {parquet_scan(nodes)} AS nodes
    WHERE {shard_filter}
),
graph_node_ids AS MATERIALIZED (
    SELECT
        node_id,
        ROW_NUMBER() OVER () - 1 AS graph_node_index
    FROM (
        SELECT src_id AS node_id
        FROM {parquet_scan(config.graph_edges_path)}
        WHERE fit_source = 'mimiciv' AND fit_split = 'train'
        UNION
        SELECT dst_id AS node_id
        FROM {parquet_scan(config.graph_edges_path)}
        WHERE fit_source = 'mimiciv' AND fit_split = 'train'
    )
),
subgraph_keys AS MATERIALIZED (
    SELECT
        source,
        split,
        stay_uid,
        subgraph_id,
        ROW_NUMBER() OVER () - 1 AS subgraph_key
    FROM filtered_nodes
    WHERE node_role = 'query_condition'
),
memberships AS MATERIALIZED (
    SELECT
        keys.subgraph_key,
        nodes.node_index,
        graph_nodes.graph_node_index,
        nodes.node_role
    FROM filtered_nodes AS nodes
    INNER JOIN subgraph_keys AS keys
        ON nodes.source = keys.source
        AND nodes.split = keys.split
        AND nodes.stay_uid = keys.stay_uid
        AND nodes.subgraph_id = keys.subgraph_id
    INNER JOIN graph_node_ids AS graph_nodes
        ON nodes.node_id = graph_nodes.node_id
),
encoded_graph_edges AS MATERIALIZED (
    SELECT
        src.graph_node_index AS src_graph_node_index,
        dst.graph_node_index AS dst_graph_node_index,
        graph_edges.src_id,
        graph_edges.dst_id,
        graph_edges.src_type,
        graph_edges.dst_type,
        graph_edges.relation_type,
        graph_edges.support_count,
        graph_edges.fit_source,
        graph_edges.fit_split
    FROM {parquet_scan(config.graph_edges_path)} AS graph_edges
    INNER JOIN graph_node_ids AS src
        ON graph_edges.src_id = src.node_id
    INNER JOIN graph_node_ids AS dst
        ON graph_edges.dst_id = dst.node_id
    WHERE graph_edges.fit_source = 'mimiciv'
        AND graph_edges.fit_split = 'train'
),
condition_edge_pairs AS (
    SELECT
        src.subgraph_key,
        src.node_index AS src_node_index,
        dst.node_index AS dst_node_index,
        graph_edges.src_id,
        graph_edges.dst_id,
        graph_edges.src_type,
        graph_edges.dst_type,
        graph_edges.relation_type,
        graph_edges.support_count,
        graph_edges.fit_source,
        graph_edges.fit_split
    FROM memberships AS src
    INNER JOIN encoded_graph_edges AS graph_edges
        ON src.graph_node_index = graph_edges.src_graph_node_index
        AND graph_edges.src_type = 'condition'
    INNER JOIN memberships AS dst
        ON src.subgraph_key = dst.subgraph_key
        AND graph_edges.dst_graph_node_index = dst.graph_node_index
    WHERE src.node_role = 'query_condition'
),
medication_edge_pairs AS (
    SELECT
        src.subgraph_key,
        src.node_index AS src_node_index,
        dst.node_index AS dst_node_index,
        graph_edges.src_id,
        graph_edges.dst_id,
        graph_edges.src_type,
        graph_edges.dst_type,
        graph_edges.relation_type,
        graph_edges.support_count,
        graph_edges.fit_source,
        graph_edges.fit_split
    FROM memberships AS src
    INNER JOIN encoded_graph_edges AS graph_edges
        ON src.graph_node_index = graph_edges.src_graph_node_index
        AND graph_edges.relation_type =
            'medication_medication_train_coprescribed'
    INNER JOIN memberships AS dst
        ON src.subgraph_key = dst.subgraph_key
        AND graph_edges.dst_graph_node_index = dst.graph_node_index
        AND dst.node_role = 'candidate_medication'
    WHERE src.node_role = 'candidate_medication'
),
edge_pairs AS (
    SELECT * FROM condition_edge_pairs
    UNION ALL
    SELECT * FROM medication_edge_pairs
)
SELECT
    keys.source,
    keys.split,
    keys.stay_uid,
    keys.subgraph_id,
    edge_pairs.src_node_index,
    edge_pairs.dst_node_index,
    edge_pairs.src_id,
    edge_pairs.dst_id,
    edge_pairs.src_type,
    edge_pairs.dst_type,
    edge_pairs.relation_type,
    edge_pairs.support_count,
    edge_pairs.fit_source,
    edge_pairs.fit_split,
    {sql_string(config.feature_version or FEATURE_VERSION)} AS feature_version,
    {sql_string(config.graph_version or GRAPH_VERSION)} AS graph_version,
    {sql_string(generated_at)} AS generated_at
FROM edge_pairs
INNER JOIN subgraph_keys AS keys USING (subgraph_key)
"""


def subgraph_candidates_query(
    config: PatientSubgraphBuildConfig,
    *,
    generated_at: str,
    nodes_path: Path | None = None,
    batch_index: int | None = None,
    batch_count: int = 1,
    join_shard_index: int | None = None,
    join_shard_count: int = 1,
) -> str:
    """Build one loader-ready candidate row per ranking candidate."""

    pcm = config.patient_condition_medication_path
    nodes = nodes_path or config.subgraph_nodes_path
    batch_filter = (
        "TRUE"
        if batch_index is None
        else stay_hash_batch_filter_sql(
            table_alias="pcm",
            batch_index=batch_index,
            batch_count=batch_count,
        )
    )
    pcm_shard_filter = (
        "TRUE"
        if join_shard_index is None
        else hash_batch_filter_sql(
            expression="pcm.ranking_group_id",
            batch_index=join_shard_index,
            batch_count=join_shard_count,
        )
    )
    nodes_shard_filter = (
        "TRUE"
        if join_shard_index is None
        else hash_batch_filter_sql(
            expression="nodes.subgraph_id",
            batch_index=join_shard_index,
            batch_count=join_shard_count,
        )
    )
    return f"""
WITH candidate_nodes AS MATERIALIZED (
    SELECT
        source,
        split,
        stay_uid,
        subgraph_id,
        node_id,
        node_index,
        in_train_graph,
        cold_start
    FROM {parquet_scan(nodes)} AS nodes
    WHERE nodes.node_role = 'candidate_medication'
        AND {nodes_shard_filter}
)
SELECT
    pcm.source,
    pcm.split,
    pcm.stay_uid,
    pcm.ranking_group_id AS subgraph_id,
    pcm.index_condition_token,
    pcm.candidate_medication_token,
    'medication|' || pcm.candidate_medication_token AS candidate_node_id,
    nodes.node_index AS candidate_node_index,
    pcm.candidate_rank,
    pcm.label_prescribed,
    nodes.in_train_graph,
    nodes.cold_start,
    {sql_string(config.feature_version or FEATURE_VERSION)} AS feature_version,
    {sql_string(config.graph_version or GRAPH_VERSION)} AS graph_version,
    {sql_string(config.label_version)} AS label_version,
    {sql_string(config.split_version)} AS split_version,
    {sql_string(generated_at)} AS generated_at
FROM {parquet_scan(pcm)} AS pcm
INNER JOIN candidate_nodes AS nodes
    ON pcm.source = nodes.source
    AND pcm.split = nodes.split
    AND pcm.stay_uid = nodes.stay_uid
    AND pcm.ranking_group_id = nodes.subgraph_id
    AND nodes.node_id = 'medication|' || pcm.candidate_medication_token
WHERE pcm.ranking_group_id IS NOT NULL
    AND pcm.candidate_medication_token IS NOT NULL
    AND {batch_filter}
    AND {pcm_shard_filter}
"""


def subgraph_index_query(
    config: PatientSubgraphBuildConfig,
    *,
    generated_at: str,
    nodes_path: Path | None = None,
    edges_paths: Sequence[Path] | None = None,
    candidates_paths: Sequence[Path] | None = None,
    batch_index: int | None = None,
    batch_count: int = 1,
) -> str:
    """Build one aggregate loader index row per ranking group."""

    pcm = config.patient_condition_medication_path
    nodes = nodes_path or config.subgraph_nodes_path
    edges_scan = (
        parquet_scan_paths(edges_paths)
        if edges_paths is not None
        else parquet_scan(config.subgraph_edges_path)
    )
    candidates_scan = (
        parquet_scan_paths(candidates_paths)
        if candidates_paths is not None
        else parquet_scan(config.subgraph_candidates_path)
    )
    batch_filter = (
        "TRUE"
        if batch_index is None
        else stay_hash_batch_filter_sql(
            table_alias="pcm",
            batch_index=batch_index,
            batch_count=batch_count,
        )
    )
    return f"""
WITH groups AS (
    SELECT DISTINCT
        source,
        split,
        stay_uid,
        ranking_group_id AS subgraph_id,
        index_condition_token
    FROM {parquet_scan(pcm)} AS pcm
    WHERE pcm.ranking_group_id IS NOT NULL
        AND pcm.index_condition_token IS NOT NULL
        AND {batch_filter}
),
node_counts AS (
    SELECT source, split, stay_uid, subgraph_id, COUNT(*) AS node_count
    FROM {parquet_scan(nodes)}
    GROUP BY source, split, stay_uid, subgraph_id
),
edge_counts AS (
    SELECT source, split, stay_uid, subgraph_id, COUNT(*) AS edge_count
    FROM {edges_scan}
    GROUP BY source, split, stay_uid, subgraph_id
),
candidate_counts AS (
    SELECT
        source,
        split,
        stay_uid,
        subgraph_id,
        COUNT(*) AS candidate_count,
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END) AS positive_count,
        SUM(CASE WHEN cold_start THEN 1 ELSE 0 END) AS cold_candidate_count
    FROM {candidates_scan}
    GROUP BY source, split, stay_uid, subgraph_id
)
SELECT
    groups.source,
    groups.split,
    groups.stay_uid,
    groups.subgraph_id,
    groups.index_condition_token,
    COALESCE(node_counts.node_count, 0) AS node_count,
    COALESCE(edge_counts.edge_count, 0) AS edge_count,
    COALESCE(candidate_counts.candidate_count, 0) AS candidate_count,
    COALESCE(candidate_counts.positive_count, 0) AS positive_count,
    COALESCE(candidate_counts.cold_candidate_count, 0) AS cold_candidate_count,
    {sql_string(config.feature_version or FEATURE_VERSION)} AS feature_version,
    {sql_string(config.graph_version or GRAPH_VERSION)} AS graph_version,
    {sql_string(config.label_version)} AS label_version,
    {sql_string(config.split_version)} AS split_version,
    {sql_string(generated_at)} AS generated_at
FROM groups
LEFT JOIN node_counts USING (source, split, stay_uid, subgraph_id)
LEFT JOIN edge_counts USING (source, split, stay_uid, subgraph_id)
LEFT JOIN candidate_counts USING (source, split, stay_uid, subgraph_id)
"""


def materialize_batched_subgraph_artifact(
    connection: duckdb.DuckDBPyConnection,
    config: PatientSubgraphBuildConfig,
    *,
    table_name: str,
    output_path: Path,
    query_for_batch: Callable[[int], str],
    part_count: int | None = None,
    build_strategy: str = "source_stay_hash_batches",
    build_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize one normalized table through bounded query parts."""

    batch_count = part_count or subgraph_batch_count(config)
    part_paths = [
        subgraph_part_path(
            config,
            table_name=table_name,
            batch_index=batch_index,
        )
        for batch_index in range(batch_count)
    ]
    batch_row_counts: list[int] = []
    active_batch_index: int | None = None
    try:
        for active_batch_index, part_path in enumerate(part_paths):
            batch_row_counts.append(
                copy_query_to_parquet(
                    connection,
                    query_for_batch(active_batch_index),
                    part_path,
                )
            )
        active_batch_index = None
        row_count = copy_query_to_parquet(
            connection,
            f"SELECT * FROM {parquet_scan_paths(part_paths)}",
            output_path,
        )
    except Exception as error:
        record = {
            "table_name": table_name,
            "output_path": str(output_path),
            "status": "failed",
            "row_count": None,
            "reason": safe_error_message(error),
            "build_strategy": build_strategy,
            "batch_count": batch_count,
            "completed_batch_count": len(batch_row_counts),
            "failed_batch_index": active_batch_index,
            "stale_output_exists": output_path.exists(),
        }
        if build_metadata:
            record.update(build_metadata)
            shard_count = build_metadata.get("join_shards_per_node_batch")
            if active_batch_index is not None and isinstance(shard_count, int):
                record["failed_node_batch_index"] = active_batch_index // shard_count
                record["failed_join_shard_index"] = active_batch_index % shard_count
        return record
    record = {
        "table_name": table_name,
        "output_path": str(output_path),
        "status": "completed",
        "row_count": row_count,
        "build_strategy": build_strategy,
        "batch_count": batch_count,
        "part_count": len(part_paths),
        "batch_row_counts": batch_row_counts,
    }
    if build_metadata:
        record.update(build_metadata)
    return record


def base_manifest(
    config: PatientSubgraphBuildConfig,
    *,
    status: str,
    generated_at: str,
) -> dict[str, Any]:
    """Return the aggregate-only patient-subgraph manifest shell."""

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "generated_at": generated_at,
        "parameters": {
            "prediction_offset_hours": config.prediction_offset_hours,
            "context_event_types": list(CONTEXT_EVENT_TYPES),
            "subgraph_unit": "ranking_group_id",
            "build_strategy": "source_stay_hash_batches",
            "batch_key": ["source", "stay_uid"],
            "subgraph_batches": subgraph_batch_count(config),
            "join_shards_per_batch": subgraph_join_shard_count(config),
            "edge_part_count": (
                subgraph_batch_count(config) * subgraph_join_shard_count(config)
            ),
            "edge_duckdb_threads": edge_duckdb_thread_count(config),
        },
        "versions": {
            "feature_version": config.feature_version or FEATURE_VERSION,
            "graph_version": config.graph_version or GRAPH_VERSION,
            "label_version": config.label_version,
            "split_version": config.split_version,
        },
        "leakage_policy": {
            "graph_fit_source": "mimiciv",
            "graph_fit_split": "train",
            "context_window": (
                f"0 <= event_time_hours_from_admit <= {config.prediction_offset_hours}"
            ),
            "validation_test_external_graph_use": "read_only_coverage",
        },
        "data_safety": {
            "manifest_contains_patient_rows": False,
            "manifest_contains_row_samples": False,
            "local_artifacts_contain_patient_level_rows": True,
            "artifact_storage": str(config.subgraphs_root),
        },
        "artifacts": {},
        "tables": [],
    }


def build_patient_subgraphs(
    config: PatientSubgraphBuildConfig = PatientSubgraphBuildConfig(),
) -> dict[str, Any]:
    """Materialize normalized patient query-subgraph artifacts."""

    generated_at = datetime.now(UTC).isoformat()
    missing = missing_input_tables(config)
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

    config.subgraphs_root.mkdir(parents=True, exist_ok=True)
    parts_root = config.subgraphs_root / "_subgraph_parts"
    shutil.rmtree(parts_root, ignore_errors=True)
    manifest = base_manifest(config, status="completed", generated_at=generated_at)
    try:
        with duckdb.connect(database=":memory:") as connection:
            configure_connection(config, connection)
            fit_scope = fetch_dict_rows(connection, graph_fit_scope_query(config))
            manifest["graph_fit_scope"] = fit_scope
            if not graph_fit_scope_is_valid(fit_scope):
                manifest["status"] = "failed_graph_fit_scope"
                write_json(config.manifest_path, manifest)
                return manifest

            batch_count = subgraph_batch_count(config)
            join_shards = subgraph_join_shard_count(config)
            join_part_count = batch_count * join_shards
            build_specs: tuple[
                tuple[
                    str,
                    Path,
                    Callable[[int], str],
                    int,
                    str,
                    dict[str, Any],
                ],
                ...,
            ] = (
                (
                    "subgraph_nodes",
                    config.subgraph_nodes_path,
                    lambda batch_index: subgraph_nodes_query(
                        config,
                        generated_at=generated_at,
                        batch_index=batch_index,
                        batch_count=batch_count,
                    ),
                    batch_count,
                    "source_stay_hash_batches",
                    {"node_batch_count": batch_count},
                ),
                (
                    "subgraph_edges",
                    config.subgraph_edges_path,
                    lambda part_index: subgraph_edges_query(
                        config,
                        generated_at=generated_at,
                        nodes_path=subgraph_part_path(
                            config,
                            table_name="subgraph_nodes",
                            batch_index=part_index // join_shards,
                        ),
                        join_shard_index=part_index % join_shards,
                        join_shard_count=join_shards,
                    ),
                    join_part_count,
                    "encoded_relation_specific_join_shards",
                    {
                        "node_batch_count": batch_count,
                        "join_shards_per_node_batch": join_shards,
                        "edge_duckdb_threads": edge_duckdb_thread_count(config),
                    },
                ),
                (
                    "subgraph_candidates",
                    config.subgraph_candidates_path,
                    lambda part_index: subgraph_candidates_query(
                        config,
                        generated_at=generated_at,
                        nodes_path=subgraph_part_path(
                            config,
                            table_name="subgraph_nodes",
                            batch_index=part_index // join_shards,
                        ),
                        batch_index=part_index // join_shards,
                        batch_count=batch_count,
                        join_shard_index=part_index % join_shards,
                        join_shard_count=join_shards,
                    ),
                    join_part_count,
                    "source_stay_hash_batches_with_join_shards",
                    {
                        "node_batch_count": batch_count,
                        "join_shards_per_node_batch": join_shards,
                    },
                ),
                (
                    "subgraph_index",
                    config.subgraph_index_path,
                    lambda batch_index: subgraph_index_query(
                        config,
                        generated_at=generated_at,
                        nodes_path=subgraph_part_path(
                            config,
                            table_name="subgraph_nodes",
                            batch_index=batch_index,
                        ),
                        edges_paths=subgraph_join_part_paths(
                            config,
                            table_name="subgraph_edges",
                            node_batch_index=batch_index,
                        ),
                        candidates_paths=subgraph_join_part_paths(
                            config,
                            table_name="subgraph_candidates",
                            node_batch_index=batch_index,
                        ),
                        batch_index=batch_index,
                        batch_count=batch_count,
                    ),
                    batch_count,
                    "source_stay_hash_batches",
                    {"node_batch_count": batch_count},
                ),
            )
            for (
                table_name,
                output_path,
                query_for_batch,
                part_count,
                build_strategy,
                build_metadata,
            ) in build_specs:
                if table_name == "subgraph_edges":
                    connection.execute(
                        f"SET threads = {edge_duckdb_thread_count(config)}"
                    )
                record = materialize_batched_subgraph_artifact(
                    connection,
                    config,
                    table_name=table_name,
                    output_path=output_path,
                    query_for_batch=query_for_batch,
                    part_count=part_count,
                    build_strategy=build_strategy,
                    build_metadata=build_metadata,
                )
                manifest["tables"].append(record)
                if record["status"] != "completed":
                    manifest["status"] = "failed"
                    manifest["reason"] = record["reason"]
                    break
                manifest["artifacts"][table_name] = str(output_path)

            if manifest["status"] == "completed":
                manifest["subgraphs_by_source_split"] = fetch_dict_rows(
                    connection,
                    f"""
SELECT
    source,
    split,
    COUNT(*) AS subgraph_count,
    SUM(node_count) AS node_count,
    SUM(edge_count) AS edge_count,
    SUM(candidate_count) AS candidate_count,
    SUM(positive_count) AS positive_count,
    SUM(cold_candidate_count) AS cold_candidate_count
FROM {parquet_scan(config.subgraph_index_path)}
GROUP BY source, split
ORDER BY source, split
""",
                )
                manifest["temporal_exclusions"] = fetch_dict_rows(
                    connection,
                    f"""
SELECT
    source,
    split,
    SUM(CASE WHEN event_time_hours_from_admit < 0 THEN 1 ELSE 0 END)
        AS pre_admission_event_rows_excluded,
    SUM(CASE
        WHEN event_time_hours_from_admit > {config.prediction_offset_hours}
        THEN 1 ELSE 0
    END) AS future_event_rows_excluded
FROM {parquet_scan(config.event_sequences_path)}
GROUP BY source, split
ORDER BY source, split
""",
                )
    finally:
        shutil.rmtree(parts_root, ignore_errors=True)

    write_json(config.manifest_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Build normalized leakage-safe patient query subgraphs.",
    )
    parser.add_argument("--features-root", type=Path, default=FEATURES_ROOT)
    parser.add_argument("--training-root", type=Path, default=TRAINING_ROOT)
    parser.add_argument("--graph-root", type=Path, default=MILESTONE8_GRAPH_ROOT)
    parser.add_argument(
        "--subgraphs-root",
        type=Path,
        default=DEFAULT_SUBGRAPHS_ROOT,
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--prediction-offset-hours",
        type=int,
        default=int(DEFAULT_MODELING_PARAMETERS["prediction_offset_hours"]),
    )
    parser.add_argument("--feature-version", default=None)
    parser.add_argument("--graph-version", default=None)
    parser.add_argument("--duckdb-temp-dir", type=Path, default=DUCKDB_TEMP_DIR)
    parser.add_argument("--duckdb-memory-limit", default=DUCKDB_MEMORY_LIMIT)
    parser.add_argument("--duckdb-threads", type=int, default=DUCKDB_THREADS)
    parser.add_argument(
        "--subgraph-batches",
        type=int,
        default=DEFAULT_SUBGRAPH_BATCH_COUNT,
    )
    parser.add_argument(
        "--subgraph-join-shards",
        type=int,
        default=DEFAULT_SUBGRAPH_JOIN_SHARDS,
    )
    parser.add_argument(
        "--edge-duckdb-threads",
        type=int,
        default=DEFAULT_SUBGRAPH_EDGE_THREADS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""

    args = parse_args(argv)
    manifest = build_patient_subgraphs(
        PatientSubgraphBuildConfig(
            features_root=args.features_root,
            training_root=args.training_root,
            graph_root=args.graph_root,
            subgraphs_root=args.subgraphs_root,
            manifest_path=args.manifest,
            prediction_offset_hours=args.prediction_offset_hours,
            feature_version=args.feature_version,
            graph_version=args.graph_version,
            duckdb_temp_directory=args.duckdb_temp_dir,
            duckdb_memory_limit=args.duckdb_memory_limit,
            duckdb_threads=args.duckdb_threads,
            subgraph_batch_count=args.subgraph_batches,
            subgraph_join_shards=args.subgraph_join_shards,
            edge_duckdb_threads=args.edge_duckdb_threads,
        )
    )
    print(
        "Wrote patient-subgraph manifest: "
        f"status={manifest['status']}, tables={len(manifest.get('tables', []))}"
    )
    if manifest["status"] == "failed_missing_inputs":
        return 2
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
