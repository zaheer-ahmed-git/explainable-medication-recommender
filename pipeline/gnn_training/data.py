"""Prepare compact, leakage-labelled patient-subgraph caches for the GNN.

The source patient-subgraph artifacts are deliberately normalized tables.  A
training loader must not repeatedly scan those large tables, especially the
edge table, so this module projects and hash-partitions each table once with
DuckDB.  Complete ranking groups share the same deterministic shard.

Only the unified concept vocabulary is learned.  It is fitted on MIMIC-train
nodes with ``PAD=0`` and ``UNK=1``; node types, node roles, and relations are
fixed schema mappings.  The prepared graph is the existing full-MIMIC-train
reference graph, so every output is explicitly marked
``scope=full_train_refit_only`` and ``selection_eligible=false``.  These caches
must never be represented as cross-fit or fold-ablation-safe inputs.

This module is PyTorch-free.  Frozen Transformer representation extraction is
a separate required component; until its cache manifest is complete,
``prepare_gnn_caches`` reports ``pending_required_component`` rather than
claiming that preparation is complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import duckdb

from pipeline.extract_utils import (
    configure_duckdb_connection,
    parquet_scan,
    safe_error_message,
    sql_string,
)
from pipeline.features import copy_query_to_parquet, fetch_dict_rows
from pipeline.gate_recovery import patient_fold_sql
from pipeline.gnn_training.config import (
    FEATURE_LAYOUT_VERSION,
    FULL_TRAIN_REFIT_SCOPE,
    PREPARE_PENDING_STATUS,
    PREPARE_SCHEMA_VERSION,
    RELATION_VOCABULARY_VERSION,
    GNNTrainingConfig,
)
from pipeline.gnn_training.graph_encode import (
    FORWARD_RELATION_TYPES,
    NODE_CONTINUOUS_FEATURES,
    NODE_ROLE_TO_INDEX,
    NODE_ROLE_VOCABULARY,
    NODE_TYPE_TO_INDEX,
    NODE_TYPE_VOCABULARY,
    PAD_INDEX,
    PAD_TOKEN,
    RELATION_TO_INDEX,
    RELATION_TYPES,
    RESERVED_TOKEN_COUNT,
    SELF_LOOP_RELATION,
    TIME_BIN_COUNT,
    UNK_INDEX,
    UNK_TOKEN,
)
from pipeline.training_contract import (
    approved_model_projection,
    schema_columns,
    sha256_file,
)

DEVELOPMENT_SOURCE = "mimiciv"
GRAPH_CACHE_SCHEMA_VERSION = "phase8-p1-gnn-graph-cache-v2"
GRAPH_CACHE_ARTIFACT_LOCK_VERSION = "phase8-p0-gnn-graph-cache-lock-v1"
FROZEN_TRANSFORMER_CACHE_SCHEMA_VERSION = "phase8-p0-gnn-frozen-transformer-cache-v1"
FROZEN_TRANSFORMER_CACHE_ARTIFACT_LOCK_VERSION = (
    "phase8-p0-gnn-frozen-transformer-cache-lock-v1"
)
TRANSFORMER_CONTEXT_TABLE = "contexts"
TRANSFORMER_LOGIT_TABLE = "candidate_logits"
FROZEN_TRANSFORMER_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    TRANSFORMER_CONTEXT_TABLE: (
        "source",
        "split",
        "ranking_group_id",
        "patient_fold_id",
        "transformer_context",
    ),
    TRANSFORMER_LOGIT_TABLE: (
        "source",
        "split",
        "ranking_group_id",
        "index_condition_token",
        "candidate_medication_token",
        "candidate_rank",
        "frozen_transformer_logit",
    ),
}

GROUP_TABLE = "groups"
NODE_TABLE = "nodes"
EDGE_TABLE = "edges"
CANDIDATE_TABLE = "candidates"
CACHE_TABLES = (GROUP_TABLE, NODE_TABLE, EDGE_TABLE, CANDIDATE_TABLE)


def artifact_tree_digest(hashes: dict[str, str]) -> str:
    """Hash a sorted relative-path/file-hash map into one stable lock."""

    digest = hashlib.sha256()
    for relative_path, file_hash in sorted(hashes.items()):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def graph_cache_artifact_hashes(root: Path) -> dict[str, str]:
    """Hash exactly the full-refit layout, vocabularies, and shard tables."""

    root = Path(root)
    paths: list[Path] = []
    layout = root / "feature_layout.json"
    if layout.is_file():
        paths.append(layout)
    for directory in (root / "vocab", root / "cache" / "shards"):
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    return {
        path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(paths)
    }


_SUBGRAPH_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "subgraph_index": (
        "source",
        "split",
        "subgraph_id",
        "index_condition_token",
        "node_count",
        "edge_count",
        "candidate_count",
        "positive_count",
    ),
    "subgraph_nodes": (
        "source",
        "split",
        "subgraph_id",
        "node_index",
        "node_id",
        "node_type",
        "node_role",
        "observed_predecision",
        "cold_start",
    ),
    "subgraph_edges": (
        "source",
        "split",
        "subgraph_id",
        "src_node_index",
        "dst_node_index",
        "relation_type",
        "support_count",
    ),
    "subgraph_candidates": (
        "source",
        "split",
        "subgraph_id",
        "index_condition_token",
        "candidate_medication_token",
        "candidate_node_index",
        "candidate_rank",
        "label_prescribed",
        "cold_start",
    ),
    "patient_condition_medication": (
        "source",
        "split",
        "patient_uid",
        "ranking_group_id",
        "index_condition_token",
        "candidate_medication_token",
        "candidate_rank",
        "label_prescribed",
    ),
}

# These are the fields that enter graph model tensors.  Local join keys and
# observed targets are intentionally handled separately.
_MODEL_PROJECTION_COLUMNS: dict[str, tuple[str, ...]] = {
    "subgraph_nodes": (
        "node_index",
        "node_id",
        "node_type",
        "node_role",
        "observed_predecision",
        "cold_start",
    ),
    "subgraph_edges": (
        "src_node_index",
        "dst_node_index",
        "relation_type",
        "support_count",
    ),
    "subgraph_candidates": (
        "index_condition_token",
        "candidate_medication_token",
        "candidate_node_index",
        "candidate_rank",
        "cold_start",
    ),
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write stable JSON without exposing partial manifests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Create a JSON marker exactly once, failing if it already exists.

    Final-score markers use this helper as an atomic claim before any test
    predictions are computed.  A process crash therefore fails closed instead
    of permitting a second test pass.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # The marker intentionally remains present on an interrupted or failed
        # final claim.  Removing it would make the one-shot gate race-prone.
        raise


def configure_connection(
    config: GNNTrainingConfig,
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Apply the repository's bounded DuckDB settings."""

    configure_duckdb_connection(
        connection,
        temp_directory=config.duckdb_temp_directory,
        memory_limit=config.duckdb_memory_limit,
        max_temp_directory_size=config.duckdb_max_temp_directory_size,
        threads=config.duckdb_threads,
    )


def _input_paths(config: GNNTrainingConfig) -> dict[str, Path]:
    return {
        "subgraph_index": config.subgraph_index_path,
        "subgraph_nodes": config.subgraph_nodes_path,
        "subgraph_edges": config.subgraph_edges_path,
        "subgraph_candidates": config.subgraph_candidates_path,
        "patient_condition_medication": (config.patient_condition_medication_path),
    }


def validate_input_projections(
    connection: duckdb.DuckDBPyConnection,
    config: GNNTrainingConfig,
) -> None:
    """Validate required schemas and the contract-approved model projection."""

    for artifact_name, path in _input_paths(config).items():
        columns = schema_columns(connection, path)
        available = {name for name, _dtype in columns}
        missing = sorted(set(_SUBGRAPH_REQUIRED_COLUMNS[artifact_name]) - available)
        if missing:
            raise ValueError(
                f"{artifact_name} is missing required columns: " + ", ".join(missing)
            )
        if artifact_name not in _MODEL_PROJECTION_COLUMNS:
            continue
        approved = set(approved_model_projection(artifact_name, columns))
        rejected = sorted(set(_MODEL_PROJECTION_COLUMNS[artifact_name]) - approved)
        if rejected:
            raise ValueError(
                f"{artifact_name} model projection is not contract-approved: "
                + ", ".join(rejected)
            )


def _split_values(config: GNNTrainingConfig) -> tuple[str, ...]:
    splits = tuple(dict.fromkeys(config.evaluation_splits()))
    if not splits:
        raise ValueError("at least one cache split is required")
    return splits


def _split_predicate(config: GNNTrainingConfig, *, alias: str) -> str:
    values = ", ".join(sql_string(value) for value in _split_values(config))
    return (
        f"{alias}.source = {sql_string(DEVELOPMENT_SOURCE)} "
        f"AND {alias}.split IN ({values})"
    )


def shard_expression(
    *,
    source_sql: str,
    ranking_group_sql: str,
    shard_count: int,
) -> str:
    """Return the deterministic complete-ranking-group shard expression."""

    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    key = (
        f"COALESCE(CAST({source_sql} AS VARCHAR), '') || '|' || "
        f"COALESCE(CAST({ranking_group_sql} AS VARCHAR), '')"
    )
    return f"CAST(HASH({key}) % {int(shard_count)} AS INTEGER)"


def node_vocabulary_query(config: GNNTrainingConfig) -> str:
    """Return a unified MIMIC-train concept vocabulary with reserved rows."""

    nodes = parquet_scan(config.subgraph_nodes_path)
    return f"""
WITH train_concepts AS (
    SELECT DISTINCT node_id
    FROM {nodes}
    WHERE source = {sql_string(DEVELOPMENT_SOURCE)}
        AND split = 'train'
        AND node_id IS NOT NULL
        AND node_id NOT IN ({sql_string(PAD_TOKEN)}, {sql_string(UNK_TOKEN)})
),
numbered AS (
    SELECT
        CAST(
            ROW_NUMBER() OVER (ORDER BY node_id) - 1
            + {RESERVED_TOKEN_COUNT} AS BIGINT
        ) AS concept_index,
        node_id
    FROM train_concepts
)
SELECT
    CAST({PAD_INDEX} AS BIGINT) AS concept_index,
    {sql_string(PAD_TOKEN)} AS node_id,
    {sql_string(DEVELOPMENT_SOURCE)} AS fit_source,
    'train' AS fit_split
UNION ALL
SELECT
    CAST({UNK_INDEX} AS BIGINT) AS concept_index,
    {sql_string(UNK_TOKEN)} AS node_id,
    {sql_string(DEVELOPMENT_SOURCE)} AS fit_source,
    'train' AS fit_split
UNION ALL
SELECT
    concept_index,
    node_id,
    {sql_string(DEVELOPMENT_SOURCE)} AS fit_source,
    'train' AS fit_split
FROM numbered
"""


def _mapping_payload(
    *,
    schema_version: str,
    vocabulary: Sequence[str],
    mapping: dict[str, int] | Any,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "vocabulary": list(vocabulary),
        "token_to_index": {str(key): int(value) for key, value in mapping.items()},
    }


def _write_vocabularies(
    connection: duckdb.DuckDBPyConnection,
    config: GNNTrainingConfig,
    *,
    vocab_root: Path,
) -> dict[str, int]:
    vocab_root.mkdir(parents=True, exist_ok=True)
    concept_path = vocab_root / config.graph_node_vocabulary_path.name
    concept_rows = copy_query_to_parquet(
        connection,
        node_vocabulary_query(config),
        concept_path,
    )
    write_json(
        vocab_root / config.node_type_vocabulary_path.name,
        _mapping_payload(
            schema_version=FEATURE_LAYOUT_VERSION,
            vocabulary=NODE_TYPE_VOCABULARY,
            mapping=NODE_TYPE_TO_INDEX,
        ),
    )
    write_json(
        vocab_root / config.node_role_vocabulary_path.name,
        _mapping_payload(
            schema_version=FEATURE_LAYOUT_VERSION,
            vocabulary=NODE_ROLE_VOCABULARY,
            mapping=NODE_ROLE_TO_INDEX,
        ),
    )
    write_json(
        vocab_root / config.relation_vocabulary_path.name,
        {
            "schema_version": RELATION_VOCABULARY_VERSION,
            "status": "completed",
            "relations": list(RELATION_TYPES),
            "relation_to_index": {
                name: int(index) for name, index in RELATION_TO_INDEX.items()
            },
        },
    )
    return {
        "concept_vocab_size": concept_rows,
        "node_type_vocab_size": len(NODE_TYPE_VOCABULARY),
        "node_role_vocab_size": len(NODE_ROLE_VOCABULARY),
        "relation_count": len(RELATION_TYPES),
    }


def _create_group_scope(
    connection: duckdb.DuckDBPyConnection,
    config: GNNTrainingConfig,
) -> None:
    """Materialize the small group/fold join once for all projected tables."""

    pcm = parquet_scan(config.patient_condition_medication_path)
    index = parquet_scan(config.subgraph_index_path)
    pcm_scope = _split_predicate(config, alias="pcm")
    index_scope = _split_predicate(config, alias="subgraphs")
    fold = patient_fold_sql(
        seed=config.seed,
        fold_count=config.fold_count,
        alias="pcm_groups",
    )
    shard = shard_expression(
        source_sql="subgraphs.source",
        ranking_group_sql="subgraphs.subgraph_id",
        shard_count=config.shard_count,
    )
    connection.execute(
        f"""
CREATE OR REPLACE TEMP TABLE gnn_pcm_groups AS
SELECT
    pcm.source,
    pcm.split,
    pcm.ranking_group_id,
    MIN(pcm.patient_uid) AS patient_uid,
    MIN(pcm.stay_uid) AS stay_uid,
    COUNT(DISTINCT pcm.patient_uid) AS patient_count,
    COUNT(DISTINCT pcm.stay_uid) AS stay_count,
    COUNT(*) AS candidate_count,
    COUNT(pcm.label_prescribed) AS labelled_candidate_count,
    SUM(CASE WHEN pcm.label_prescribed THEN 1 ELSE 0 END) AS positive_count
FROM {pcm} AS pcm
WHERE {pcm_scope}
    AND pcm.ranking_group_id IS NOT NULL
GROUP BY pcm.source, pcm.split, pcm.ranking_group_id
"""
    )
    invalid = connection.execute(
        """
SELECT COUNT(*)
FROM gnn_pcm_groups
WHERE patient_count <> 1
    OR stay_count <> 1
    OR candidate_count <> labelled_candidate_count
"""
    ).fetchone()
    if invalid is not None and int(invalid[0]) > 0:
        raise ValueError(
            "ranking groups must map to one patient with complete candidate labels"
        )

    connection.execute(
        f"""
CREATE OR REPLACE TEMP TABLE gnn_source_groups AS
SELECT
    subgraphs.source,
    subgraphs.split,
    subgraphs.subgraph_id,
    subgraphs.node_count,
    subgraphs.edge_count,
    subgraphs.candidate_count,
    subgraphs.positive_count
FROM {index} AS subgraphs
WHERE {index_scope}
    AND subgraphs.subgraph_id IS NOT NULL
"""
    )
    duplicate_groups = connection.execute(
        """
SELECT COUNT(*)
FROM (
    SELECT source, split, subgraph_id
    FROM gnn_source_groups
    GROUP BY source, split, subgraph_id
    HAVING COUNT(*) <> 1
) AS duplicate_groups
"""
    ).fetchone()
    if duplicate_groups is not None and int(duplicate_groups[0]) > 0:
        raise ValueError("subgraph index contains duplicate ranking groups")

    candidates = parquet_scan(config.subgraph_candidates_path)
    candidate_scope = _split_predicate(config, alias="candidates")
    connection.execute(
        f"""
CREATE OR REPLACE TEMP TABLE gnn_candidate_groups AS
SELECT
    candidates.source,
    candidates.split,
    candidates.subgraph_id,
    COUNT(*) AS candidate_count,
    COUNT(candidates.label_prescribed) AS labelled_candidate_count,
    SUM(CASE WHEN candidates.label_prescribed THEN 1 ELSE 0 END)
        AS positive_count
FROM {candidates} AS candidates
WHERE {candidate_scope}
    AND candidates.subgraph_id IS NOT NULL
GROUP BY candidates.source, candidates.split, candidates.subgraph_id
"""
    )
    candidate_identity_mismatches = connection.execute(
        f"""
WITH pcm_candidates AS (
    SELECT
        pcm.source,
        pcm.split,
        pcm.ranking_group_id,
        pcm.index_condition_token,
        pcm.candidate_medication_token,
        CAST(pcm.candidate_rank AS BIGINT) AS candidate_rank,
        CAST(pcm.label_prescribed AS BOOLEAN) AS label_prescribed
    FROM {pcm} AS pcm
    WHERE {pcm_scope}
        AND pcm.ranking_group_id IS NOT NULL
),
graph_candidates AS (
    SELECT
        candidates.source,
        candidates.split,
        candidates.subgraph_id AS ranking_group_id,
        candidates.index_condition_token,
        candidates.candidate_medication_token,
        CAST(candidates.candidate_rank AS BIGINT) AS candidate_rank,
        CAST(candidates.label_prescribed AS BOOLEAN) AS label_prescribed
    FROM {candidates} AS candidates
    WHERE {candidate_scope}
        AND candidates.subgraph_id IS NOT NULL
),
pcm_only AS (
    SELECT * FROM pcm_candidates
    EXCEPT ALL
    SELECT * FROM graph_candidates
),
graph_only AS (
    SELECT * FROM graph_candidates
    EXCEPT ALL
    SELECT * FROM pcm_candidates
)
SELECT
    (SELECT COUNT(*) FROM pcm_only)
    + (SELECT COUNT(*) FROM graph_only)
    + (
        SELECT COUNT(*)
        FROM pcm_candidates
        WHERE index_condition_token IS NULL
            OR candidate_medication_token IS NULL
            OR candidate_rank IS NULL
            OR label_prescribed IS NULL
    )
    + (
        SELECT COUNT(*)
        FROM graph_candidates
        WHERE index_condition_token IS NULL
            OR candidate_medication_token IS NULL
            OR candidate_rank IS NULL
            OR label_prescribed IS NULL
    )
"""
    ).fetchone()
    if (
        candidate_identity_mismatches is not None
        and int(candidate_identity_mismatches[0]) > 0
    ):
        raise ValueError(
            "subgraph candidate identities or observed labels are inconsistent"
        )

    connection.execute(
        """
CREATE OR REPLACE TEMP TABLE gnn_group_audit AS
SELECT
    subgraphs.source,
    subgraphs.split,
    subgraphs.subgraph_id,
    subgraphs.node_count,
    subgraphs.edge_count,
    subgraphs.candidate_count AS declared_candidate_count,
    subgraphs.positive_count AS declared_positive_count,
    pcm_groups.candidate_count AS pcm_candidate_count,
    pcm_groups.positive_count AS pcm_positive_count,
    candidate_groups.candidate_count AS cached_candidate_count,
    candidate_groups.positive_count AS cached_positive_count
FROM gnn_source_groups AS subgraphs
LEFT JOIN gnn_pcm_groups AS pcm_groups
    ON subgraphs.source = pcm_groups.source
    AND subgraphs.split = pcm_groups.split
    AND subgraphs.subgraph_id = pcm_groups.ranking_group_id
LEFT JOIN gnn_candidate_groups AS candidate_groups
    ON subgraphs.source = candidate_groups.source
    AND subgraphs.split = candidate_groups.split
    AND subgraphs.subgraph_id = candidate_groups.subgraph_id
"""
    )
    inconsistent_groups = connection.execute(
        """
SELECT
    (
        SELECT COUNT(*)
        FROM gnn_group_audit
        WHERE node_count <= 0
            OR edge_count < 0
            OR declared_candidate_count <= 0
            OR declared_positive_count < 0
            OR pcm_candidate_count IS NULL
            OR cached_candidate_count IS NULL
            OR declared_candidate_count <> pcm_candidate_count
            OR declared_candidate_count <> cached_candidate_count
            OR declared_positive_count <> pcm_positive_count
            OR declared_positive_count <> cached_positive_count
    )
    + (
        SELECT COUNT(*)
        FROM gnn_pcm_groups AS pcm_groups
        LEFT JOIN gnn_source_groups AS subgraphs
            ON subgraphs.source = pcm_groups.source
            AND subgraphs.split = pcm_groups.split
            AND subgraphs.subgraph_id = pcm_groups.ranking_group_id
        WHERE subgraphs.subgraph_id IS NULL
    )
    + (
        SELECT COUNT(*)
        FROM gnn_candidate_groups AS candidate_groups
        LEFT JOIN gnn_source_groups AS subgraphs
            ON subgraphs.source = candidate_groups.source
            AND subgraphs.split = candidate_groups.split
            AND subgraphs.subgraph_id = candidate_groups.subgraph_id
        WHERE subgraphs.subgraph_id IS NULL
            OR candidate_groups.candidate_count
                <> candidate_groups.labelled_candidate_count
    )
"""
    ).fetchone()
    if inconsistent_groups is not None and int(inconsistent_groups[0]) > 0:
        raise ValueError(
            "subgraph group counts or observed candidate labels are inconsistent"
        )

    connection.execute(
        f"""
CREATE OR REPLACE TEMP TABLE gnn_cache_groups AS
SELECT
    subgraphs.source,
    subgraphs.split,
    subgraphs.subgraph_id AS ranking_group_id,
    pcm_groups.stay_uid,
    CAST({fold} AS INTEGER) AS patient_fold_id,
    CAST(subgraphs.node_count AS BIGINT) AS node_count,
    CAST(subgraphs.edge_count AS BIGINT) AS forward_edge_count,
    CAST(2 * subgraphs.edge_count + subgraphs.node_count AS BIGINT)
        AS expanded_edge_count,
    CAST(candidate_groups.candidate_count AS BIGINT) AS candidate_count,
    CAST(candidate_groups.positive_count AS BIGINT) AS positive_count,
    {shard} AS shard_id
FROM gnn_source_groups AS subgraphs
INNER JOIN gnn_pcm_groups AS pcm_groups
    ON subgraphs.source = pcm_groups.source
    AND subgraphs.split = pcm_groups.split
    AND subgraphs.subgraph_id = pcm_groups.ranking_group_id
INNER JOIN gnn_candidate_groups AS candidate_groups
    ON subgraphs.source = candidate_groups.source
    AND subgraphs.split = candidate_groups.split
    AND subgraphs.subgraph_id = candidate_groups.subgraph_id
WHERE subgraphs.node_count > 0
    AND candidate_groups.candidate_count > 0
    AND (
        subgraphs.split <> 'train'
        OR COALESCE(candidate_groups.positive_count, 0) > 0
    )
"""
    )


def group_coverage(
    connection: duckdb.DuckDBPyConnection,
    config: GNNTrainingConfig,
) -> list[dict[str, Any]]:
    """Return aggregate kept and zero-positive group coverage by split."""

    return fetch_dict_rows(
        connection,
        """
WITH source_groups AS (
    SELECT
        source,
        split,
        subgraph_id,
        COALESCE(cached_positive_count, 0) AS positive_count
    FROM gnn_group_audit
)
SELECT
    source_groups.source,
    source_groups.split,
    COUNT(*) AS ranking_group_count,
    SUM(CASE WHEN positive_count > 0 THEN 1 ELSE 0 END)
        AS positive_group_count,
    SUM(CASE WHEN positive_count = 0 THEN 1 ELSE 0 END)
        AS zero_positive_group_count,
    SUM(
        CASE
            WHEN split = 'train' AND positive_count = 0 THEN 1
            ELSE 0
        END
    ) AS train_groups_excluded_from_fit_cache,
    SUM(
        CASE
            WHEN split <> 'train' OR positive_count > 0 THEN 1
            ELSE 0
        END
    ) AS cached_group_count
FROM source_groups
GROUP BY source_groups.source, source_groups.split
ORDER BY source_groups.source, source_groups.split
""",
    )


def _case_index(expression: str, mapping: dict[str, int] | Any, *, kind: str) -> str:
    branches = " ".join(
        f"WHEN {sql_string(str(name))} THEN {int(index)}"
        for name, index in mapping.items()
    )
    return (
        f"CASE {expression} {branches} "
        f"ELSE error('unsupported {kind} in patient subgraph') END"
    )


def groups_cache_query() -> str:
    """Return the compact group/fold cache query."""

    return """
SELECT
    source,
    split,
    ranking_group_id,
    patient_fold_id,
    node_count,
    expanded_edge_count,
    candidate_count,
    positive_count,
    shard_id
FROM gnn_cache_groups
"""


def nodes_cache_query(
    config: GNNTrainingConfig,
    *,
    vocabulary_path: Path,
) -> str:
    """Return encoded graph nodes without concept or patient identifiers."""

    nodes = parquet_scan(config.subgraph_nodes_path)
    vocabulary = parquet_scan(vocabulary_path)
    node_type = _case_index("nodes.node_type", NODE_TYPE_TO_INDEX, kind="node type")
    node_role = _case_index("nodes.node_role", NODE_ROLE_TO_INDEX, kind="node role")
    events = parquet_scan(config.event_sequences_path)
    finite_value = (
        "CASE WHEN events.value_numeric IS NOT NULL "
        "AND isfinite(CAST(events.value_numeric AS DOUBLE)) "
        "AND ABS(CAST(events.value_numeric AS DOUBLE)) <= 1e100 "
        "THEN CAST(events.value_numeric AS DOUBLE) ELSE NULL END"
    )
    return f"""
WITH fit_numeric_events AS MATERIALIZED (
    SELECT
        events.event_type,
        events.event_token,
        {finite_value} AS numeric_value
    FROM {events} AS events
    WHERE events.source = 'mimiciv'
        AND events.split = 'train'
        AND events.event_type IN ('lab', 'vital')
        AND events.event_token IS NOT NULL
        AND events.event_time_hours_from_admit >= 0
        AND events.event_time_hours_from_admit <= 24.0
),
fit_stats AS MATERIALIZED (
    SELECT
        event_type,
        event_token,
        AVG(numeric_value) AS value_mean,
        STDDEV_SAMP(numeric_value) AS value_std
    FROM fit_numeric_events
    WHERE numeric_value IS NOT NULL
    GROUP BY event_type, event_token
),
node_event_summary AS MATERIALIZED (
    SELECT
        events.source,
        events.split,
        events.stay_uid,
        events.event_type,
        events.event_token,
        MAX(events.event_time_hours_from_admit) AS last_time,
        ARG_MAX(
            {finite_value},
            events.event_time_hours_from_admit
        ) FILTER (WHERE {finite_value} IS NOT NULL) AS last_value,
        REGR_SLOPE(
            {finite_value},
            events.event_time_hours_from_admit
        ) FILTER (WHERE {finite_value} IS NOT NULL) AS value_slope
    FROM {events} AS events
    WHERE events.event_type IN ('lab', 'vital', 'intervention')
        AND events.event_token IS NOT NULL
        AND events.event_time_hours_from_admit >= 0
        AND events.event_time_hours_from_admit <= 24.0
    GROUP BY
        events.source,
        events.split,
        events.stay_uid,
        events.event_type,
        events.event_token
),
attributes AS (
    SELECT
        summary.*,
        stats.value_mean,
        stats.value_std,
        CASE
            WHEN summary.last_value IS NOT NULL
                AND stats.value_std IS NOT NULL
                AND stats.value_std > 1e-12
            THEN GREATEST(-10.0, LEAST(
                10.0,
                (summary.last_value - stats.value_mean) / stats.value_std
            ))
            ELSE 0.0
        END AS value_zscore,
        CASE WHEN summary.last_value IS NULL THEN 0.0 ELSE 1.0 END AS value_mask,
        CASE
            WHEN summary.last_value IS NULL
                OR stats.value_std IS NULL
                OR stats.value_std <= 1e-12 THEN 0.0
            WHEN (summary.last_value - stats.value_mean) / stats.value_std <= -2.0
                THEN -1.0
            WHEN (summary.last_value - stats.value_mean) / stats.value_std >= 2.0
                THEN 1.0
            ELSE 0.0
        END AS abnormal_direction,
        CASE
            WHEN summary.value_slope IS NOT NULL
                AND stats.value_std IS NOT NULL
                AND stats.value_std > 1e-12
            THEN GREATEST(-10.0, LEAST(
                10.0,
                summary.value_slope * 24.0 / stats.value_std
            ))
            ELSE 0.0
        END AS trend_zscore_per_window
    FROM node_event_summary AS summary
    LEFT JOIN fit_stats AS stats
        USING (event_type, event_token)
)
SELECT
    nodes.source,
    nodes.split,
    groups.ranking_group_id,
    groups.shard_id,
    CAST(nodes.node_index AS BIGINT) AS node_index,
    CAST(COALESCE(vocab.concept_index, {UNK_INDEX}) AS BIGINT)
        AS node_concept_index,
    CAST({node_type} AS INTEGER) AS node_type_index,
    CAST({node_role} AS INTEGER) AS node_role_index,
    CAST(nodes.observed_predecision AS BOOLEAN) AS observed_predecision,
    CAST(nodes.cold_start AS BOOLEAN) AS cold_start
    ,CAST(COALESCE(attributes.value_zscore, 0.0) AS REAL) AS value_zscore
    ,CAST(COALESCE(attributes.value_mask, 0.0) AS REAL) AS value_mask
    ,CAST(COALESCE(attributes.abnormal_direction, 0.0) AS REAL)
        AS abnormal_direction
    ,CAST(COALESCE(attributes.trend_zscore_per_window, 0.0) AS REAL)
        AS trend_zscore_per_window
    ,CAST(COALESCE(attributes.last_time / 24.0, 0.0) AS REAL)
        AS time_normalized
    ,CAST(
        CASE
            WHEN attributes.last_time IS NULL THEN 0
            ELSE LEAST(4, FLOOR(attributes.last_time / 6.0) + 1)
        END AS INTEGER
    ) AS time_bin_index
FROM {nodes} AS nodes
INNER JOIN gnn_cache_groups AS groups
    ON nodes.source = groups.source
    AND nodes.split = groups.split
    AND nodes.subgraph_id = groups.ranking_group_id
LEFT JOIN {vocabulary} AS vocab
    ON nodes.node_id = vocab.node_id
LEFT JOIN attributes
    ON nodes.source = attributes.source
    AND nodes.split = attributes.split
    AND groups.stay_uid = attributes.stay_uid
    AND nodes.node_id = attributes.event_type || '|' || attributes.event_token
"""


def candidates_cache_query(config: GNNTrainingConfig) -> str:
    """Return complete candidate sets plus restricted local score-join keys."""

    candidates = parquet_scan(config.subgraph_candidates_path)
    return f"""
SELECT
    candidates.source,
    candidates.split,
    groups.ranking_group_id,
    groups.shard_id,
    candidates.index_condition_token,
    candidates.candidate_medication_token,
    CAST(candidates.candidate_node_index AS BIGINT) AS candidate_node_index,
    CAST(candidates.candidate_rank AS BIGINT) AS candidate_rank,
    CAST(candidates.label_prescribed AS BOOLEAN) AS label_prescribed,
    CAST(candidates.cold_start AS BOOLEAN) AS cold_start
FROM {candidates} AS candidates
INNER JOIN gnn_cache_groups AS groups
    ON candidates.source = groups.source
    AND candidates.split = groups.split
    AND candidates.subgraph_id = groups.ranking_group_id
"""


def edges_cache_query(config: GNNTrainingConfig) -> str:
    """Return forward, reverse, and self-loop edges with normalized weights."""

    edges = parquet_scan(config.subgraph_edges_path)
    nodes = parquet_scan(config.subgraph_nodes_path)
    forward_relation = _case_index(
        "source_edges.relation_type",
        {name: RELATION_TO_INDEX[name] for name in FORWARD_RELATION_TYPES},
        kind="forward relation",
    )
    reverse_relation = _case_index(
        "source_edges.relation_type",
        {name: RELATION_TO_INDEX[f"reverse_{name}"] for name in FORWARD_RELATION_TYPES},
        kind="forward relation",
    )
    self_relation = RELATION_TO_INDEX[SELF_LOOP_RELATION]
    support = (
        "CASE "
        "WHEN source_edges.support_count IS NULL OR source_edges.support_count < 0 "
        "THEN error('edge support must be non-negative') "
        "ELSE LN(1.0 + CAST(source_edges.support_count AS DOUBLE)) END"
    )
    return f"""
WITH source_edges AS (
    SELECT
        edges.source,
        edges.split,
        groups.ranking_group_id,
        groups.shard_id,
        edges.src_node_index,
        edges.dst_node_index,
        edges.relation_type,
        edges.support_count
    FROM {edges} AS edges
    INNER JOIN gnn_cache_groups AS groups
        ON edges.source = groups.source
        AND edges.split = groups.split
        AND edges.subgraph_id = groups.ranking_group_id
),
directed_edges AS (
    SELECT
        source_edges.source,
        source_edges.split,
        source_edges.ranking_group_id,
        source_edges.shard_id,
        CAST(
            CASE
                WHEN directions.is_reverse THEN source_edges.dst_node_index
                ELSE source_edges.src_node_index
            END AS BIGINT
        ) AS src_node_index,
        CAST(
            CASE
                WHEN directions.is_reverse THEN source_edges.src_node_index
                ELSE source_edges.dst_node_index
            END AS BIGINT
        ) AS dst_node_index,
        CAST(
            CASE
                WHEN directions.is_reverse THEN {reverse_relation}
                ELSE {forward_relation}
            END AS INTEGER
        ) AS relation_index,
        {support} AS transformed_support
    FROM source_edges
    CROSS JOIN (VALUES (FALSE), (TRUE)) AS directions(is_reverse)
),
self_edges AS (
    SELECT
        nodes.source,
        nodes.split,
        groups.ranking_group_id,
        groups.shard_id,
        CAST(nodes.node_index AS BIGINT) AS src_node_index,
        CAST(nodes.node_index AS BIGINT) AS dst_node_index,
        CAST({self_relation} AS INTEGER) AS relation_index,
        LN(2.0) AS transformed_support
    FROM {nodes} AS nodes
    INNER JOIN gnn_cache_groups AS groups
        ON nodes.source = groups.source
        AND nodes.split = groups.split
        AND nodes.subgraph_id = groups.ranking_group_id
),
expanded AS (
    SELECT * FROM directed_edges
    UNION ALL
    SELECT * FROM self_edges
),
normalization AS (
    SELECT
        expanded.*,
        SUM(transformed_support) OVER (
            PARTITION BY
                source,
                split,
                ranking_group_id,
                relation_index,
                dst_node_index
        ) AS incoming_support,
        COUNT(*) OVER (
            PARTITION BY
                source,
                split,
                ranking_group_id,
                relation_index,
                dst_node_index
        ) AS incoming_count
    FROM expanded
)
SELECT
    source,
    split,
    ranking_group_id,
    shard_id,
    src_node_index,
    dst_node_index,
    relation_index,
    CAST(transformed_support AS DOUBLE) AS edge_log_support,
    CAST(
        CASE
            WHEN incoming_support > 0
            THEN transformed_support / incoming_support
            ELSE 1.0 / incoming_count
        END AS DOUBLE
    ) AS edge_weight
FROM normalization
"""


def _copy_partitioned(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output_root: Path,
) -> int:
    """Write one projected scan into Hive-style split/shard partitions."""

    output_root.mkdir(parents=True, exist_ok=True)
    row = connection.execute(
        f"""
COPY ({query})
TO {sql_string(output_root)}
(
    FORMAT PARQUET,
    COMPRESSION ZSTD,
    PARTITION_BY (split, shard_id),
    WRITE_PARTITION_COLUMNS TRUE,
    PER_THREAD_OUTPUT FALSE,
    FILENAME_PATTERN 'part_{{i}}',
    OVERWRITE_OR_IGNORE TRUE
)
"""
    ).fetchone()
    coerce_single_parquet_partitions(output_root)
    return int(row[0]) if row is not None and row[0] is not None else 0


def coerce_single_parquet_partitions(table_root: Path) -> None:
    """Collapse multi-fragment Hive partitions into one Parquet file each.

    DuckDB ``COPY ... PARTITION_BY`` can still emit several ``part_*.parquet``
    files per ``split=*/shard_id=*`` directory even with
    ``PER_THREAD_OUTPUT FALSE`` (row-group / size thresholds).  The GNN loader
    and P1 cache contract require exactly one fragment per logical shard.
    """

    if not table_root.is_dir():
        return
    for shard_directory in sorted(table_root.glob("split=*/shard_id=*")):
        files = sorted(shard_directory.glob("*.parquet"))
        if len(files) <= 1:
            continue
        merged = shard_directory / f".merged-{uuid.uuid4().hex}.parquet"
        final = shard_directory / "part_0.parquet"
        try:
            with duckdb.connect(database=":memory:") as connection:
                connection.execute(
                    f"""
COPY (
    SELECT * FROM read_parquet(
        {sql_string((shard_directory / "*.parquet").as_posix())},
        union_by_name = TRUE
    )
)
TO {sql_string(merged)}
(FORMAT PARQUET, COMPRESSION ZSTD)
"""
                )
            for path in files:
                path.unlink(missing_ok=True)
            os.replace(merged, final)
        finally:
            if merged.exists():
                merged.unlink(missing_ok=True)


def _validate_compact_partitions(table_roots: Iterable[Path]) -> None:
    """Require at most one Parquet fragment per table/split/shard partition."""

    for table_root in table_roots:
        for shard_directory in table_root.glob("split=*/shard_id=*"):
            if len(tuple(shard_directory.glob("*.parquet"))) > 1:
                raise RuntimeError(
                    "GNN cache compaction invariant failed: a table/split/shard "
                    "partition contains multiple Parquet fragments"
                )


def _validate_staged_group_rows(
    connection: duckdb.DuckDBPyConnection,
    *,
    staged_shards: Path,
    group_count: int,
) -> None:
    """Reconcile staged row counts and labels for every ranking group."""

    if group_count == 0:
        return
    scans = {
        table_name: (
            "read_parquet("
            f"{sql_string(staged_shards / table_name / '**' / '*.parquet')}, "
            "hive_partitioning = TRUE)"
        )
        for table_name in CACHE_TABLES
    }
    inconsistent = connection.execute(
        f"""
WITH node_counts AS (
    SELECT source, split, ranking_group_id, COUNT(*) AS row_count
    FROM {scans[NODE_TABLE]}
    GROUP BY source, split, ranking_group_id
),
edge_counts AS (
    SELECT source, split, ranking_group_id, COUNT(*) AS row_count
    FROM {scans[EDGE_TABLE]}
    GROUP BY source, split, ranking_group_id
),
candidate_counts AS (
    SELECT
        source,
        split,
        ranking_group_id,
        COUNT(*) AS row_count,
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END) AS positive_count
    FROM {scans[CANDIDATE_TABLE]}
    GROUP BY source, split, ranking_group_id
)
SELECT COUNT(*)
FROM gnn_cache_groups AS groups
LEFT JOIN node_counts
    USING (source, split, ranking_group_id)
LEFT JOIN edge_counts
    USING (source, split, ranking_group_id)
LEFT JOIN candidate_counts
    USING (source, split, ranking_group_id)
WHERE COALESCE(node_counts.row_count, -1) <> groups.node_count
    OR COALESCE(edge_counts.row_count, -1) <> groups.expanded_edge_count
    OR COALESCE(candidate_counts.row_count, -1) <> groups.candidate_count
    OR COALESCE(candidate_counts.positive_count, -1) <> groups.positive_count
"""
    ).fetchone()
    if inconsistent is not None and int(inconsistent[0]) > 0:
        raise ValueError(
            "staged graph rows or observed candidate labels are incomplete by group"
        )


def _stage_root(config: GNNTrainingConfig) -> Path:
    config.gnn_root.mkdir(parents=True, exist_ok=True)
    root = config.gnn_root / f".prepare-stage-{uuid.uuid4().hex}"
    _guard_under_gnn_root(config, root)
    root.mkdir(parents=False, exist_ok=False)
    return root


def _guard_under_gnn_root(config: GNNTrainingConfig, path: Path) -> None:
    """Reject cleanup/replacement outside the configured stage-owned root."""

    root = config.gnn_root.resolve()
    target = path.resolve()
    if target == root:
        raise ValueError("refusing to replace the GNN root itself")
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError("GNN cache output must remain under gnn_root") from error


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _write_prepare_report_if_safe(
    config: GNNTrainingConfig,
    report: dict[str, Any],
) -> None:
    """Never honor a configured aggregate-report path outside REPORTS_ROOT."""

    if _is_under(config.prepare_manifest_path, config.reports_root):
        write_json(config.prepare_manifest_path, report)


def _safe_prepare_reason(error: Exception) -> str:
    """Keep direct identifier column names out of the public failure report."""

    reason = safe_error_message(error)
    for column_name in (
        "patient_uid",
        "stay_uid",
        "encounter_uid",
    ):
        reason = reason.replace(column_name, "restricted_identifier")
    return reason


def _remove_owned_path(config: GNNTrainingConfig, path: Path) -> None:
    _guard_under_gnn_root(config, path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _promote_paths(
    config: GNNTrainingConfig,
    *,
    replacements: Sequence[tuple[Path, Path]],
) -> None:
    """Atomically promote all staged outputs, rolling back them as a unit."""

    transaction_id = uuid.uuid4().hex
    backups: dict[Path, Path] = {}
    promoted: list[Path] = []
    try:
        for staged, destination in replacements:
            _guard_under_gnn_root(config, staged)
            _guard_under_gnn_root(config, destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup = destination.with_name(
                f".{destination.name}.backup-{transaction_id}"
            )
            _guard_under_gnn_root(config, backup)
            if destination.exists():
                os.replace(destination, backup)
                backups[destination] = backup
        for staged, destination in replacements:
            os.replace(staged, destination)
            promoted.append(destination)
    except Exception:
        for destination in reversed(promoted):
            if destination.exists():
                _remove_owned_path(config, destination)
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    for backup in backups.values():
        if backup.exists():
            _remove_owned_path(config, backup)


def _frozen_transformer_hashes(config: GNNTrainingConfig) -> dict[str, str]:
    paths = {
        "checkpoint": config.neural_checkpoint_path,
        "feature_layout": config.neural_feature_layout_path,
        "calibration": config.neural_calibration_path,
    }
    return {name: sha256_file(path) for name, path in paths.items() if path.is_file()}


def _upstream_manifest_hashes(config: GNNTrainingConfig) -> dict[str, str | None]:
    """Bind a graph cache to the small immutable upstream control documents."""

    paths = {
        "training_contract_lock_sha256": config.contract_lock_path,
        "patient_subgraphs_manifest_sha256": config.subgraphs_manifest_path,
    }
    return {
        name: sha256_file(path) if path.is_file() else None
        for name, path in paths.items()
    }


def _representation_table_row_count(
    config: GNNTrainingConfig,
    paths: Sequence[Path],
    *,
    required_columns: Sequence[str],
    vector_column: str | None = None,
    vector_length: int | None = None,
    numeric_column: str | None = None,
    invalid_row_predicate: str | None = None,
) -> int | None:
    """Validate every representation shard schema and return its aggregate rows."""

    if not paths:
        return None
    forbidden_columns = {"patient_uid", "stay_uid", "encounter_uid"}
    try:
        with duckdb.connect(database=":memory:") as connection:
            configure_connection(config, connection)
            total = 0
            for path in paths:
                cursor = connection.execute(
                    "SELECT * FROM read_parquet(?) LIMIT 0",
                    [str(path)],
                )
                schema = {
                    str(description[0]): str(description[1]).upper()
                    for description in cursor.description
                }
                if not set(required_columns).issubset(
                    schema
                ) or forbidden_columns.intersection(schema):
                    return None
                if vector_column is not None and not any(
                    marker in schema[vector_column]
                    for marker in ("[]", "ARRAY", "LIST")
                ):
                    return None
                if vector_column is not None and not schema[vector_column].startswith(
                    ("FLOAT", "DOUBLE", "DECIMAL")
                ):
                    return None
                if numeric_column is not None and not schema[numeric_column].startswith(
                    ("FLOAT", "DOUBLE", "DECIMAL")
                ):
                    return None
                aggregate_columns = ["COUNT(*)"]
                if vector_column is not None:
                    aggregate_columns.extend(
                        (
                            f"MIN(array_length({vector_column}))",
                            f"MAX(array_length({vector_column}))",
                        )
                    )
                if invalid_row_predicate is not None:
                    aggregate_columns.append(
                        f"SUM(CASE WHEN {invalid_row_predicate} THEN 1 ELSE 0 END)"
                    )
                row = connection.execute(
                    "SELECT " + ", ".join(aggregate_columns) + " FROM read_parquet(?)",
                    [str(path)],
                ).fetchone()
                if row is None:
                    return None
                row_count = int(row[0])
                if (
                    vector_column is not None
                    and vector_length is not None
                    and row_count > 0
                    and (int(row[1]) != vector_length or int(row[2]) != vector_length)
                ):
                    return None
                if invalid_row_predicate is not None and int(row[-1] or 0) > 0:
                    return None
                total += row_count
    except (duckdb.Error, OSError, ValueError):
        return None
    return total


def _representation_group_keys_match(config: GNNTrainingConfig) -> bool:
    """Check exact graph/context/logit coverage without returning keys."""

    context_scan = (
        "read_parquet("
        f"{sql_string(config.frozen_transformer_cache_root / TRANSFORMER_CONTEXT_TABLE / '**' / '*.parquet')}, "
        "hive_partitioning = TRUE)"
    )
    logit_scan = (
        "read_parquet("
        f"{sql_string(config.frozen_transformer_cache_root / TRANSFORMER_LOGIT_TABLE / '**' / '*.parquet')}, "
        "hive_partitioning = TRUE)"
    )
    group_paths = sorted((config.shards_root / GROUP_TABLE).rglob("*.parquet"))
    candidate_paths = sorted((config.shards_root / CANDIDATE_TABLE).rglob("*.parquet"))
    if not group_paths or not candidate_paths:
        return False
    group_scan = (
        "read_parquet("
        f"{sql_string(config.shards_root / GROUP_TABLE / '**' / '*.parquet')}, "
        "hive_partitioning = TRUE)"
    )
    candidate_scan = (
        "read_parquet("
        f"{sql_string(config.shards_root / CANDIDATE_TABLE / '**' / '*.parquet')}, "
        "hive_partitioning = TRUE)"
    )
    try:
        with duckdb.connect(database=":memory:") as connection:
            configure_connection(config, connection)
            row = connection.execute(
                f"""
WITH context_groups AS (
    SELECT source, split, ranking_group_id, COUNT(*) AS context_count
    FROM {context_scan}
    GROUP BY source, split, ranking_group_id
),
logit_groups AS (
    SELECT
        source,
        split,
        ranking_group_id,
        COUNT(DISTINCT index_condition_token) AS condition_count
    FROM {logit_scan}
    GROUP BY source, split, ranking_group_id
),
graph_groups AS (
    SELECT
        source,
        split,
        ranking_group_id,
        patient_fold_id
    FROM {group_scan}
),
graph_candidates AS (
    SELECT
        source,
        split,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        candidate_rank
    FROM {candidate_scan}
),
transformer_candidates AS (
    SELECT
        source,
        split,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        candidate_rank
    FROM {logit_scan}
),
context_only AS (
    SELECT source, split, ranking_group_id
    FROM context_groups
    EXCEPT
    SELECT source, split, ranking_group_id
    FROM logit_groups
),
logit_only AS (
    SELECT source, split, ranking_group_id
    FROM logit_groups
    EXCEPT
    SELECT source, split, ranking_group_id
    FROM context_groups
),
graph_context_only AS (
    SELECT source, split, ranking_group_id, patient_fold_id
    FROM graph_groups
    EXCEPT ALL
    SELECT source, split, ranking_group_id, patient_fold_id
    FROM {context_scan}
),
transformer_context_only AS (
    SELECT source, split, ranking_group_id, patient_fold_id
    FROM {context_scan}
    EXCEPT ALL
    SELECT source, split, ranking_group_id, patient_fold_id
    FROM graph_groups
),
graph_logit_only AS (
    SELECT * FROM graph_candidates
    EXCEPT ALL
    SELECT * FROM transformer_candidates
),
transformer_logit_only AS (
    SELECT * FROM transformer_candidates
    EXCEPT ALL
    SELECT * FROM graph_candidates
),
duplicate_logits AS (
    SELECT
        source,
        split,
        ranking_group_id,
        candidate_medication_token
    FROM {logit_scan}
    GROUP BY
        source,
        split,
        ranking_group_id,
        candidate_medication_token
    HAVING COUNT(*) <> 1
),
duplicate_ranks AS (
    SELECT
        source,
        split,
        ranking_group_id,
        candidate_rank
    FROM {logit_scan}
    GROUP BY source, split, ranking_group_id, candidate_rank
    HAVING COUNT(*) <> 1
)
SELECT
    (SELECT COUNT(*) FROM context_only)
    + (SELECT COUNT(*) FROM logit_only)
    + (
        SELECT COUNT(*)
        FROM context_groups
        WHERE context_count <> 1
    )
    + (
        SELECT COUNT(*)
        FROM logit_groups
        WHERE condition_count <> 1
    )
    + (SELECT COUNT(*) FROM duplicate_logits)
    + (SELECT COUNT(*) FROM duplicate_ranks)
    + (SELECT COUNT(*) FROM graph_context_only)
    + (SELECT COUNT(*) FROM transformer_context_only)
    + (SELECT COUNT(*) FROM graph_logit_only)
    + (SELECT COUNT(*) FROM transformer_logit_only)
"""
            ).fetchone()
    except (duckdb.Error, OSError, ValueError):
        return False
    return row is not None and int(row[0]) == 0


def _transformer_cache_status(config: GNNTrainingConfig) -> str:
    path = config.transformer_cache_manifest_path
    required_frozen_artifacts = (
        config.neural_checkpoint_path,
        config.neural_feature_layout_path,
        config.neural_calibration_path,
    )
    if not path.is_file() or any(
        not artifact.is_file() for artifact in required_frozen_artifacts
    ):
        return "pending"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "pending"
    if not isinstance(payload, dict):
        return "pending"
    expected_hashes = _frozen_transformer_hashes(config)
    recorded_hashes = payload.get(
        "frozen_transformer_hashes",
        payload.get("artifact_hashes"),
    )
    raw_counts = payload.get("table_row_counts")
    counts_valid = (
        isinstance(raw_counts, dict)
        and set(raw_counts) == set(FROZEN_TRANSFORMER_TABLE_COLUMNS)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in raw_counts.values()
        )
    )
    table_paths = {
        table_name: sorted(
            (config.frozen_transformer_cache_root / table_name).rglob("*.parquet")
        )
        for table_name in FROZEN_TRANSFORMER_TABLE_COLUMNS
    }
    expected_paths = {path for paths in table_paths.values() for path in paths}
    all_paths = set(config.frozen_transformer_cache_root.rglob("*.parquet"))
    actual_artifact_hashes = {
        artifact.relative_to(config.frozen_transformer_cache_root).as_posix(): (
            sha256_file(artifact)
        )
        for artifact in sorted(all_paths)
    }
    recorded_artifact_hashes = payload.get("artifact_hashes")
    recorded_splits = payload.get("cached_splits")
    actual_splits = sorted(
        {
            part.removeprefix("split=")
            for artifact in all_paths
            for part in artifact.relative_to(config.frozen_transformer_cache_root).parts
            if part.startswith("split=")
        }
    )
    expected_splits = sorted(config.evaluation_splits())
    split_lock_valid = recorded_splits == actual_splits and (
        config.allow_ungated or actual_splits == expected_splits
    )
    context_rows = _representation_table_row_count(
        config,
        table_paths[TRANSFORMER_CONTEXT_TABLE],
        required_columns=FROZEN_TRANSFORMER_TABLE_COLUMNS[TRANSFORMER_CONTEXT_TABLE],
        vector_column="transformer_context",
        vector_length=config.architecture.transformer_context_dim,
        invalid_row_predicate=(
            "source IS NULL "
            "OR split IS NULL "
            "OR ranking_group_id IS NULL "
            "OR patient_fold_id IS NULL "
            "OR patient_fold_id < 0 "
            f"OR patient_fold_id >= {int(config.fold_count)} "
            "OR transformer_context IS NULL "
            "OR array_length(list_filter("
            "transformer_context, value -> value IS NULL OR NOT isfinite(value)"
            ")) > 0"
        ),
    )
    logit_rows = _representation_table_row_count(
        config,
        table_paths[TRANSFORMER_LOGIT_TABLE],
        required_columns=FROZEN_TRANSFORMER_TABLE_COLUMNS[TRANSFORMER_LOGIT_TABLE],
        numeric_column="frozen_transformer_logit",
        invalid_row_predicate=(
            "source IS NULL "
            "OR split IS NULL "
            "OR ranking_group_id IS NULL "
            "OR index_condition_token IS NULL "
            "OR candidate_medication_token IS NULL "
            "OR candidate_rank IS NULL "
            "OR candidate_rank < 0 "
            "OR frozen_transformer_logit IS NULL "
            "OR NOT isfinite(frozen_transformer_logit)"
        ),
    )
    physical_counts = {
        TRANSFORMER_CONTEXT_TABLE: context_rows,
        TRANSFORMER_LOGIT_TABLE: logit_rows,
    }
    complete = (
        payload.get("status") == "completed"
        and payload.get("schema_version") == FROZEN_TRANSFORMER_CACHE_SCHEMA_VERSION
        and payload.get("artifact_lock_version")
        == FROZEN_TRANSFORMER_CACHE_ARTIFACT_LOCK_VERSION
        and payload.get("scope") == FULL_TRAIN_REFIT_SCOPE
        and payload.get("selection_eligible") is False
        and payload.get("shard_count") == config.shard_count
        and payload.get("transformer_context_dim")
        == config.architecture.transformer_context_dim
        and len(expected_hashes) == len(required_frozen_artifacts)
        and recorded_hashes == expected_hashes
        and payload.get("upstream_provenance") == _upstream_manifest_hashes(config)
        and counts_valid
        and expected_paths == all_paths
        and recorded_artifact_hashes == actual_artifact_hashes
        and payload.get("artifact_tree_digest")
        == artifact_tree_digest(actual_artifact_hashes)
        and split_lock_valid
        and physical_counts == raw_counts
        and _representation_group_keys_match(config)
    )
    return "completed" if complete else "pending"


def _feature_layout(
    *,
    vocab_sizes: dict[str, int],
    shard_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": FEATURE_LAYOUT_VERSION,
        "cache_schema_version": GRAPH_CACHE_SCHEMA_VERSION,
        "scope": FULL_TRAIN_REFIT_SCOPE,
        "selection_eligible": False,
        "pad_index": PAD_INDEX,
        "unk_index": UNK_INDEX,
        "reserved_token_count": RESERVED_TOKEN_COUNT,
        **vocab_sizes,
        "node_type_vocabulary": list(NODE_TYPE_VOCABULARY),
        "node_type_to_index": {
            name: int(index) for name, index in NODE_TYPE_TO_INDEX.items()
        },
        "node_role_vocabulary": list(NODE_ROLE_VOCABULARY),
        "node_role_to_index": {
            name: int(index) for name, index in NODE_ROLE_TO_INDEX.items()
        },
        "relation_vocabulary": list(RELATION_TYPES),
        "relation_to_index": {
            name: int(index) for name, index in RELATION_TO_INDEX.items()
        },
        "node_continuous_features": list(NODE_CONTINUOUS_FEATURES),
        "time_bin_count": TIME_BIN_COUNT,
        "time_bin_policy": "missing_or_four_six_hour_predecision_bins",
        "numeric_attribute_fit_scope": "mimiciv_train",
        "shard_count": int(shard_count),
        "shard_key": ["source", "ranking_group_id"],
        "edge_support_transform": "log1p",
        "edge_weight_normalization": ("incoming_per_subgraph_relation_destination"),
    }


def _base_manifest(
    config: GNNTrainingConfig,
    *,
    generated_at: str,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": PREPARE_SCHEMA_VERSION,
        "status": status,
        "stage": "prepare",
        "mode": config.mode,
        "generated_at": generated_at,
        "scope": FULL_TRAIN_REFIT_SCOPE,
        "selection_eligible": False,
        "preparation_complete": False,
        "parameters": {
            "development_source": DEVELOPMENT_SOURCE,
            "shard_count": int(config.shard_count),
            "fold_count": int(config.fold_count),
            "seed": int(config.seed),
            "splits": list(_split_values(config)),
        },
        "leakage_policy": {
            "concept_vocabulary_fit_scope": "mimiciv_train",
            "graph_reference_scope": "mimiciv_train",
            "train_cache_excludes_zero_positive_groups": True,
            "evaluation_caches_keep_zero_positive_groups": True,
            "crossfit_claimed": False,
            "full_train_cache_selection_eligible": False,
        },
        "components": {
            "graph_cache": "pending",
            "crossfit_graph_caches": "pending",
            "frozen_transformer_cache": "pending",
        },
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "report_contains_identifier_values": False,
            "restricted_join_keys_in_local_cache_only": True,
        },
        "gate_policy": config.gate_policy(),
    }


def _prepare_graph_stage(
    connection: duckdb.DuckDBPyConnection,
    config: GNNTrainingConfig,
    *,
    stage_root: Path,
    transformer_status: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build all graph artifacts beneath ``stage_root`` and return aggregates."""

    upstream_hashes = _upstream_manifest_hashes(config)
    validate_input_projections(connection, config)
    staged_vocab = stage_root / "vocab"
    staged_cache = stage_root / "cache"
    staged_shards = staged_cache / "shards"
    staged_layout = stage_root / "feature_layout.json"
    staged_cache_manifest = staged_cache / "cache_manifest.json"

    vocab_sizes = _write_vocabularies(
        connection,
        config,
        vocab_root=staged_vocab,
    )
    _create_group_scope(connection, config)
    coverage = group_coverage(connection, config)

    table_queries = {
        GROUP_TABLE: groups_cache_query(),
        NODE_TABLE: nodes_cache_query(
            config,
            vocabulary_path=staged_vocab / config.graph_node_vocabulary_path.name,
        ),
        EDGE_TABLE: edges_cache_query(config),
        CANDIDATE_TABLE: candidates_cache_query(config),
    }
    table_counts = {
        table_name: _copy_partitioned(
            connection,
            query,
            staged_shards / table_name,
        )
        for table_name, query in table_queries.items()
    }
    _validate_compact_partitions(
        staged_shards / table_name for table_name in table_queries
    )

    group_rows = int(table_counts[GROUP_TABLE])
    _validate_staged_group_rows(
        connection,
        staged_shards=staged_shards,
        group_count=group_rows,
    )
    candidate_rows = int(table_counts[CANDIDATE_TABLE])
    declared_candidates_row = connection.execute(
        "SELECT COALESCE(SUM(candidate_count), 0) FROM gnn_cache_groups"
    ).fetchone()
    declared_candidates = (
        int(declared_candidates_row[0]) if declared_candidates_row is not None else 0
    )
    declared_graph_rows = connection.execute(
        """
SELECT
    COALESCE(SUM(node_count), 0) AS node_count,
    COALESCE(SUM(expanded_edge_count), 0) AS expanded_edge_count
FROM gnn_cache_groups
"""
    ).fetchone()
    declared_nodes = int(declared_graph_rows[0]) if declared_graph_rows else 0
    declared_edges = int(declared_graph_rows[1]) if declared_graph_rows else 0
    if int(table_counts[NODE_TABLE]) != declared_nodes:
        raise ValueError("cached node rows do not match complete declared node sets")
    if int(table_counts[EDGE_TABLE]) != declared_edges:
        raise ValueError(
            "cached expanded edge rows do not match forward, reverse, and self edges"
        )
    if candidate_rows != declared_candidates:
        raise ValueError(
            "cached candidate rows do not match complete declared candidate sets"
        )
    if group_rows != sum(int(row["cached_group_count"]) for row in coverage):
        raise ValueError("cached group count does not match split coverage")

    layout = _feature_layout(
        vocab_sizes=vocab_sizes,
        shard_count=config.shard_count,
    )
    write_json(staged_layout, layout)
    transformer_hashes = _frozen_transformer_hashes(config)
    if _upstream_manifest_hashes(config) != upstream_hashes:
        raise RuntimeError(
            "upstream control manifests changed during cache preparation"
        )
    artifact_hashes = graph_cache_artifact_hashes(stage_root)
    cache_manifest = {
        "schema_version": GRAPH_CACHE_SCHEMA_VERSION,
        "artifact_lock_version": GRAPH_CACHE_ARTIFACT_LOCK_VERSION,
        # This local manifest describes the completed graph-cache component.
        # Overall preparation remains pending in the public prepare report
        # until frozen Transformer extraction is present.
        "status": "completed",
        "scope": FULL_TRAIN_REFIT_SCOPE,
        "selection_eligible": False,
        "preparation_complete": transformer_status == "completed",
        "components": {
            "graph_cache": "completed",
            "crossfit_graph_caches": "pending",
            "frozen_transformer_cache": transformer_status,
        },
        "shard_count": int(config.shard_count),
        "table_row_counts": table_counts,
        "split_aggregates": coverage,
        "cached_splits": sorted(
            {
                str(row["split"])
                for row in coverage
                if int(row["cached_group_count"]) > 0
            }
        ),
        "vocab_sizes": vocab_sizes,
        "artifact_hashes": artifact_hashes,
        "artifact_tree_digest": artifact_tree_digest(artifact_hashes),
        "frozen_transformer_hashes": transformer_hashes,
        "upstream_provenance": upstream_hashes,
        "data_safety": {
            "manifest_contains_patient_rows": False,
            "local_cache_contains_patient_level_rows": True,
            "local_cache_is_restricted": True,
            "contains_row_samples": False,
            "restricted_join_keys_present": True,
            "direct_patient_identifiers_present": False,
        },
    }
    write_json(staged_cache_manifest, cache_manifest)

    _promote_paths(
        config,
        replacements=(
            (staged_shards, config.shards_root),
            (staged_vocab, config.vocab_root),
            (staged_layout, config.feature_layout_path),
            (staged_cache_manifest, config.cache_manifest_path),
        ),
    )
    return cache_manifest, layout


def prepare_gnn_caches(config: GNNTrainingConfig) -> dict[str, Any]:
    """Build bounded graph caches after the GNN contract preflight.

    The full-train graph component can complete independently.  Overall
    preparation completes only after physical fold-excluded selection caches
    and an exactly reconciled frozen-Transformer representation cache are also
    present.
    """

    from pipeline.gnn_training.contract import blocked_report, preflight_errors

    generated_at = datetime.now(UTC).isoformat()
    errors = preflight_errors(config, stage="prepare")
    if errors:
        report = blocked_report(
            config=config,
            stage="prepare",
            schema_version=PREPARE_SCHEMA_VERSION,
            generated_at=generated_at,
            errors=errors,
        )
        _write_prepare_report_if_safe(config, report)
        return report

    manifest = _base_manifest(
        config,
        generated_at=generated_at,
        status="running",
    )
    stage_root: Path | None = None
    try:
        transformer_status = _transformer_cache_status(config)
        stage_root = _stage_root(config)
        with duckdb.connect(database=":memory:") as connection:
            configure_connection(config, connection)
            cache_manifest, layout = _prepare_graph_stage(
                connection,
                config,
                stage_root=stage_root,
                transformer_status=transformer_status,
            )

        from pipeline.gnn_training.crossfit import prepare_crossfit_graph_caches
        from pipeline.gnn_training.transformer_cache import (
            prepare_frozen_transformer_cache,
        )

        crossfit_status = "pending"
        crossfit_report: dict[str, Any] | None = None
        crossfit_inputs = (
            config.subgraph_index_path,
            config.subgraph_nodes_path,
            config.subgraph_edges_path,
            config.subgraph_candidates_path,
            config.patient_condition_medication_path,
            config.event_sequences_path,
        )
        if all(path.is_file() for path in crossfit_inputs):
            crossfit_report = prepare_crossfit_graph_caches(config)
            crossfit_status = str(crossfit_report.get("status", "failed"))

        transformer_inputs = (
            config.neural_checkpoint_path,
            config.neural_feature_layout_path,
            config.neural_calibration_path,
        )
        neural_cache_ready = any(
            (config.neural_root / "cache" / "groups" / split).is_dir()
            for split in _split_values(config)
        )
        if all(path.is_file() for path in transformer_inputs) and neural_cache_ready:
            transformer_report = prepare_frozen_transformer_cache(config)
            transformer_status = str(transformer_report.get("status", "failed"))
        else:
            transformer_status = _transformer_cache_status(config)

        manifest["components"] = {
            "graph_cache": "completed",
            "crossfit_graph_caches": crossfit_status,
            "frozen_transformer_cache": transformer_status,
        }
        manifest["preparation_complete"] = (
            crossfit_status == "completed" and transformer_status == "completed"
        )
        component_failed = "failed" in {crossfit_status, transformer_status}
        if manifest["preparation_complete"]:
            manifest["status"] = "completed"
        elif component_failed:
            manifest["status"] = "failed"
            manifest["reason"] = (
                "one or more required preparation components failed; inspect "
                "the component aggregate manifest without exposing local rows"
            )
        else:
            manifest["status"] = PREPARE_PENDING_STATUS
        manifest["leakage_policy"]["crossfit_claimed"] = crossfit_status == "completed"
        manifest["table_row_counts"] = cache_manifest["table_row_counts"]
        manifest["split_aggregates"] = cache_manifest["split_aggregates"]
        manifest["vocab_sizes"] = {
            key: int(layout[key])
            for key in (
                "concept_vocab_size",
                "node_type_vocab_size",
                "node_role_vocab_size",
                "relation_count",
            )
        }
        manifest["frozen_transformer_hashes"] = cache_manifest[
            "frozen_transformer_hashes"
        ]
        manifest["upstream_provenance"] = cache_manifest["upstream_provenance"]
        if crossfit_report is not None:
            manifest["crossfit_summary"] = {
                key: crossfit_report.get(key)
                for key in (
                    "schema_version",
                    "fold_count",
                    "seed",
                    "patient_grouped",
                )
                if key in crossfit_report
            }

        cache_manifest["components"] = dict(manifest["components"])
        cache_manifest["preparation_complete"] = manifest["preparation_complete"]
        if config.crossfit_graph_manifest_path.is_file():
            cache_manifest["crossfit_graph_manifest_sha256"] = sha256_file(
                config.crossfit_graph_manifest_path
            )
        write_json(config.cache_manifest_path, cache_manifest)
    except Exception as error:  # noqa: BLE001 - aggregate fail-closed report
        manifest["status"] = "failed"
        manifest["reason"] = _safe_prepare_reason(error)
    finally:
        if stage_root is not None and stage_root.exists():
            _remove_owned_path(config, stage_root)

    _write_prepare_report_if_safe(config, manifest)
    return manifest
