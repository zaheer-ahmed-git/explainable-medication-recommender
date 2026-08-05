"""Leakage-safe patient-grouped cross-fit graph cache construction.

The full-train graph and caches produced by :mod:`pipeline.gnn_training.data`
are valid only for a final refit.  They are not valid model-selection inputs:
condition--medication, medication coprescription, and condition--event support
counts contain labels or observations from every MIMIC-IV train patient.

This module builds one independent selection cache per held-out patient fold.
For every fold it:

* derives the patient fold with :func:`pipeline.gate_recovery.patient_fold_sql`;
* recomputes all five forward graph relations without held-out patients;
* uses the locked patient-subgraph tables only as group/node/edge membership;
* refits the concept vocabulary without held-out patients;
* rebuilds node cold-start flags, candidate indexes, edge supports and weights;
* adds deterministic reverse relations and self loops; and
* hash-partitions complete ranking groups into bounded Parquet shards.

Patient and stay identifiers exist only in restricted DuckDB intermediates.
Local caches retain restricted ranking-group and candidate join keys required
by training/scoring, while public reports contain aggregate counts and hashes
only.  Promotion is transactional: a completed staged cross-fit tree replaces
the previous tree as a unit, and the aggregate manifest is written atomically
only after promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from pipeline.extract_utils import (
    configure_duckdb_connection,
    parquet_scan,
    safe_error_message,
    sql_string,
)
from pipeline.gate_recovery import patient_fold_sql
from pipeline.gnn_training.config import (
    CROSS_FIT_SCHEMA_VERSION,
    CROSS_FIT_SELECTION_SCOPE,
    FEATURE_LAYOUT_VERSION,
    RELATION_VOCABULARY_VERSION,
    GNNTrainingConfig,
)
from pipeline.gnn_training.data import (
    CACHE_TABLES,
    CANDIDATE_TABLE,
    EDGE_TABLE,
    GROUP_TABLE,
    NODE_TABLE,
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
from pipeline.training_contract import load_json, schema_columns, sha256_file

DEVELOPMENT_SOURCE = "mimiciv"
CROSS_FIT_CAPACITY_ENV = "GNN_CROSSFIT_MIN_FREE_GIB"
GIBIBYTE = 1024**3
CROSS_FIT_GRAPH_SCHEMA_VERSION = "phase8-p1-gnn-crossfit-graph-v2"
CROSS_FIT_CACHE_SCHEMA_VERSION = "phase8-p1-gnn-crossfit-cache-v2"
CROSS_FIT_ARTIFACT_LOCK_VERSION = "phase8-p0-gnn-crossfit-artifact-lock-v1"
PREFLIGHT_ATTESTATION_VERSION = "phase8-p0-crossfit-preflight-attestation-v1"
GRAPH_MANIFEST_NAME = "graph_manifest.json"

_INPUT_COLUMNS: dict[str, tuple[str, ...]] = {
    "subgraph_index": (
        "source",
        "split",
        "subgraph_id",
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
    ),
    "subgraph_edges": (
        "source",
        "split",
        "subgraph_id",
        "src_node_index",
        "dst_node_index",
        "src_id",
        "dst_id",
        "relation_type",
    ),
    "subgraph_candidates": (
        "source",
        "split",
        "subgraph_id",
        "index_condition_token",
        "candidate_medication_token",
        "candidate_rank",
        "label_prescribed",
    ),
    "patient_condition_medication": (
        "source",
        "split",
        "patient_uid",
        "stay_uid",
        "ranking_group_id",
        "index_condition_token",
        "candidate_medication_token",
        "candidate_rank",
        "label_prescribed",
    ),
    "event_sequences": (
        "source",
        "split",
        "stay_uid",
        "event_type",
        "event_token",
        "event_time_hours_from_admit",
        "value_numeric",
    ),
}

_CONDITION_EVENT_RELATIONS = (
    "condition_lab_predecision",
    "condition_vital_predecision",
    "condition_intervention_predecision",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _guard_restricted_output(config: GNNTrainingConfig, path: Path) -> None:
    if not _is_under(path, config.gnn_root) or _is_under(path, config.neural_root):
        raise ValueError("cross-fit artifacts must stay under the isolated GNN root")


def _remove_owned_path(config: GNNTrainingConfig, path: Path) -> None:
    _guard_restricted_output(config, path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _safe_reason(error: Exception) -> str:
    reason = safe_error_message(error)
    for restricted_name in (
        "patient_uid",
        "stay_uid",
        "encounter_uid",
        "subject_id",
        "hadm_id",
    ):
        reason = reason.replace(restricted_name, "restricted_identifier")
    return reason


def _write_public_report(
    config: GNNTrainingConfig,
    payload: Mapping[str, Any],
) -> None:
    if not _is_under(config.crossfit_graph_manifest_path, config.reports_root):
        return
    _write_json(config.crossfit_graph_manifest_path, payload)


def _input_paths(config: GNNTrainingConfig) -> dict[str, Path]:
    return {
        "subgraph_index": config.subgraph_index_path,
        "subgraph_nodes": config.subgraph_nodes_path,
        "subgraph_edges": config.subgraph_edges_path,
        "subgraph_candidates": config.subgraph_candidates_path,
        "patient_condition_medication": config.patient_condition_medication_path,
        "event_sequences": config.event_sequences_path,
    }


def _validate_input_schemas(
    connection: duckdb.DuckDBPyConnection,
    config: GNNTrainingConfig,
) -> None:
    """Validate metadata only; never fetch source rows."""

    for artifact_name, path in _input_paths(config).items():
        columns = {name for name, _dtype in schema_columns(connection, path)}
        missing = sorted(set(_INPUT_COLUMNS[artifact_name]) - columns)
        if missing:
            raise ValueError(
                f"{artifact_name} is missing required schema fields: "
                + ", ".join(missing)
            )


def _configure_connection(
    config: GNNTrainingConfig,
    connection: duckdb.DuckDBPyConnection,
) -> None:
    configure_duckdb_connection(
        connection,
        temp_directory=config.duckdb_temp_directory,
        memory_limit=config.duckdb_memory_limit,
        max_temp_directory_size=config.duckdb_max_temp_directory_size,
        threads=config.duckdb_threads,
    )


def _parquet_scan_any(path: Path) -> str:
    """Scan one Parquet file or every ``*.parquet`` file under a directory."""

    if path.is_dir():
        return f"read_parquet({sql_string(path / '*.parquet')}, union_by_name = TRUE)"
    return parquet_scan(path)


def _hash_path(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "kind": "file",
            "file_count": 1,
            "sha256": sha256_file(path),
        }
    if not path.is_dir():
        raise FileNotFoundError("required cross-fit input artifact is missing")
    hashes = _artifact_hashes(path)
    return {
        "kind": "tree",
        "file_count": len(hashes),
        "sha256": artifact_tree_digest(hashes),
    }


def _upstream_locks(config: GNNTrainingConfig) -> dict[str, dict[str, Any]]:
    """Return exact hashes without publishing source paths."""

    paths: dict[str, Path] = {
        **_input_paths(config),
        "training_contract_lock": config.contract_lock_path,
        "patient_subgraphs_manifest": config.subgraphs_manifest_path,
    }
    if config.allow_ungated:
        paths = {
            name: path
            for name, path in paths.items()
            if name not in {"training_contract_lock", "patient_subgraphs_manifest"}
            or path.exists()
        }
    return {name: _hash_path(path) for name, path in sorted(paths.items())}


def _artifact_hashes(
    root: Path,
    *,
    excluded_relative_paths: Sequence[str] = (),
    progress_interval_files: int | None = None,
    progress_label: str = "artifact_tree",
) -> dict[str, str]:
    """Hash an artifact tree once, with optional aggregate-only progress."""

    excluded = set(excluded_relative_paths)
    hashes: dict[str, str] = {}
    started = time.monotonic()
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        hashes[relative] = sha256_file(path)
        if (
            progress_interval_files is not None
            and len(hashes) % progress_interval_files == 0
        ):
            print(
                json.dumps(
                    {
                        "event": "artifact_integrity_heartbeat",
                        "scope": progress_label,
                        "file_count": len(hashes),
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return hashes


def artifact_tree_digest(hashes: Mapping[str, str]) -> str:
    """Hash a sorted relative-path/file-hash map into one reproducible lock."""

    digest = hashlib.sha256()
    for relative_path, file_hash in sorted(hashes.items()):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _artifact_stat_digest(root: Path) -> tuple[int, str]:
    """Return a cheap allocation-local fingerprint without rereading bodies."""

    digest = hashlib.sha256()
    count = 0
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        stat = path.stat()
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
        count += 1
    return count, digest.hexdigest()


def _preflight_attestation_path(config: GNNTrainingConfig) -> Path | None:
    """Resolve a per-OAR-allocation attestation below WORK_SCRATCH."""

    allocation_id = os.environ.get("OAR_JOB_ID")
    scratch = os.environ.get("WORK_SCRATCH")
    if not allocation_id or not scratch:
        return None
    scratch_root = Path(scratch).expanduser().resolve()
    configured = os.environ.get("GNN_PREFLIGHT_ATTESTATION_PATH")
    path = (
        Path(configured).expanduser().resolve()
        if configured
        else scratch_root / "gnn-preflight" / f"oar-{allocation_id}.json"
    )
    if not _is_under(path, scratch_root):
        raise ValueError("GNN preflight attestation must remain below WORK_SCRATCH")
    if _is_under(path, config.crossfit_root):
        raise ValueError("GNN preflight attestation must be outside immutable caches")
    return path


def _attestation_is_current(
    config: GNNTrainingConfig,
    *,
    payload: Mapping[str, Any],
    stat_file_count: int,
    stat_digest: str,
) -> bool:
    path = _preflight_attestation_path(config)
    if path is None or not path.is_file():
        return False
    try:
        attestation = load_json(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    return (
        attestation.get("schema_version") == PREFLIGHT_ATTESTATION_VERSION
        and attestation.get("allocation_id") == os.environ.get("OAR_JOB_ID")
        and attestation.get("crossfit_root") == str(config.crossfit_root.resolve())
        and attestation.get("manifest_sha256")
        == sha256_file(config.crossfit_graph_manifest_path)
        and attestation.get("crossfit_tree_digest")
        == payload.get("crossfit_tree_digest")
        and attestation.get("stat_file_count") == stat_file_count
        and attestation.get("stat_digest") == stat_digest
    )


def _write_preflight_attestation(
    config: GNNTrainingConfig,
    *,
    payload: Mapping[str, Any],
    stat_file_count: int,
    stat_digest: str,
) -> None:
    path = _preflight_attestation_path(config)
    if path is None:
        return
    _write_json(
        path,
        {
            "schema_version": PREFLIGHT_ATTESTATION_VERSION,
            "allocation_id": os.environ["OAR_JOB_ID"],
            "crossfit_root": str(config.crossfit_root.resolve()),
            "manifest_sha256": sha256_file(config.crossfit_graph_manifest_path),
            "crossfit_tree_digest": payload.get("crossfit_tree_digest"),
            "stat_file_count": stat_file_count,
            "stat_digest": stat_digest,
            "verified_at": _utc_now(),
        },
    )
    path.chmod(0o600)


def _contract_digest(config: GNNTrainingConfig) -> str | None:
    if not config.contract_lock_path.exists():
        return None
    payload = load_json(config.contract_lock_path)
    value = payload.get("contract_digest")
    return str(value) if isinstance(value, str) and value else None


def _create_private_group_scope(
    connection: duckdb.DuckDBPyConnection,
    config: GNNTrainingConfig,
) -> None:
    """Create the sole temporary relation that contains patient identifiers."""

    pcm = parquet_scan(config.patient_condition_medication_path)
    index = parquet_scan(config.subgraph_index_path)
    candidates = parquet_scan(config.subgraph_candidates_path)
    fold = patient_fold_sql(
        seed=config.seed,
        fold_count=config.fold_count,
        alias="grouped",
    )
    shard_key = (
        "COALESCE(CAST(grouped.source AS VARCHAR), '') || '|' || "
        "COALESCE(CAST(grouped.ranking_group_id AS VARCHAR), '')"
    )
    connection.execute(
        f"""
CREATE OR REPLACE TEMP TABLE crossfit_private_groups AS
WITH grouped AS (
    SELECT
        pcm.source,
        pcm.split,
        pcm.ranking_group_id,
        MIN(pcm.patient_uid) AS patient_uid,
        MIN(pcm.stay_uid) AS stay_uid,
        MIN(pcm.index_condition_token) AS index_condition_token,
        COUNT(DISTINCT pcm.patient_uid) AS patient_count,
        COUNT(DISTINCT pcm.stay_uid) AS stay_count,
        COUNT(DISTINCT pcm.index_condition_token) AS condition_count,
        COUNT(*) AS pcm_candidate_count,
        COUNT(pcm.label_prescribed) AS labelled_candidate_count,
        SUM(CASE WHEN pcm.label_prescribed THEN 1 ELSE 0 END) AS positive_count
    FROM {pcm} AS pcm
    WHERE pcm.source = {sql_string(DEVELOPMENT_SOURCE)}
        AND pcm.split = 'train'
        AND pcm.ranking_group_id IS NOT NULL
    GROUP BY pcm.source, pcm.split, pcm.ranking_group_id
),
candidate_counts AS (
    SELECT
        source,
        split,
        subgraph_id AS ranking_group_id,
        COUNT(*) AS candidate_count,
        COUNT(label_prescribed) AS labelled_candidate_count,
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END) AS positive_count
    FROM {candidates}
    WHERE source = {sql_string(DEVELOPMENT_SOURCE)}
        AND split = 'train'
        AND subgraph_id IS NOT NULL
    GROUP BY source, split, subgraph_id
),
index_rows AS (
    SELECT
        source,
        split,
        subgraph_id AS ranking_group_id,
        COUNT(*) AS index_row_count
    FROM {index}
    WHERE source = {sql_string(DEVELOPMENT_SOURCE)}
        AND split = 'train'
        AND subgraph_id IS NOT NULL
    GROUP BY source, split, subgraph_id
)
SELECT
    grouped.source,
    grouped.split,
    grouped.ranking_group_id,
    grouped.patient_uid,
    grouped.stay_uid,
    grouped.index_condition_token,
    CAST({fold} AS INTEGER) AS patient_fold_id,
    CAST(
        HASH({shard_key}) % {int(config.shard_count)}
        AS INTEGER
    ) AS shard_id,
    CAST(grouped.pcm_candidate_count AS BIGINT) AS candidate_count,
    CAST(grouped.positive_count AS BIGINT) AS positive_count,
    grouped.patient_count,
    grouped.stay_count,
    grouped.condition_count,
    grouped.labelled_candidate_count,
    candidate_counts.candidate_count AS locked_candidate_count,
    candidate_counts.labelled_candidate_count AS locked_labelled_candidate_count,
    candidate_counts.positive_count AS locked_positive_count,
    index_rows.index_row_count
FROM grouped
LEFT JOIN candidate_counts
    USING (source, split, ranking_group_id)
LEFT JOIN index_rows
    USING (source, split, ranking_group_id)
"""
    )
    invalid = connection.execute(
        """
SELECT COUNT(*)
FROM crossfit_private_groups
WHERE patient_count <> 1
    OR stay_count <> 1
    OR condition_count <> 1
    OR candidate_count <= 0
    OR labelled_candidate_count <> candidate_count
    OR locked_candidate_count IS NULL
    OR locked_candidate_count <> candidate_count
    OR locked_labelled_candidate_count <> candidate_count
    OR locked_positive_count <> positive_count
    OR index_row_count <> 1
"""
    ).fetchone()
    if invalid is not None and int(invalid[0]) > 0:
        raise ValueError(
            "locked ranking groups, candidate labels, or group ownership are "
            "inconsistent"
        )

    candidate_identity_mismatches = connection.execute(
        f"""
WITH pcm_candidates AS (
    SELECT
        source,
        split,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        CAST(candidate_rank AS BIGINT) AS candidate_rank,
        CAST(label_prescribed AS BOOLEAN) AS label_prescribed
    FROM {pcm}
    WHERE source = {sql_string(DEVELOPMENT_SOURCE)}
        AND split = 'train'
),
locked_candidates AS (
    SELECT
        source,
        split,
        subgraph_id AS ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        CAST(candidate_rank AS BIGINT) AS candidate_rank,
        CAST(label_prescribed AS BOOLEAN) AS label_prescribed
    FROM {candidates}
    WHERE source = {sql_string(DEVELOPMENT_SOURCE)}
        AND split = 'train'
),
pcm_only AS (
    SELECT * FROM pcm_candidates
    EXCEPT ALL
    SELECT * FROM locked_candidates
),
locked_only AS (
    SELECT * FROM locked_candidates
    EXCEPT ALL
    SELECT * FROM pcm_candidates
)
SELECT
    (SELECT COUNT(*) FROM pcm_only)
    + (SELECT COUNT(*) FROM locked_only)
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
        FROM locked_candidates
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
            "locked candidate identities or observed labels differ from the "
            "patient-condition-medication contract"
        )

    fold_rows = connection.execute(
        """
SELECT patient_fold_id, COUNT(DISTINCT patient_uid) AS patient_count
FROM crossfit_private_groups
GROUP BY patient_fold_id
ORDER BY patient_fold_id
"""
    ).fetchall()
    observed = {int(row[0]): int(row[1]) for row in fold_rows}
    if set(observed) != set(range(config.fold_count)) or any(
        count <= 0 for count in observed.values()
    ):
        raise ValueError("every configured patient fold must be non-empty")


def _fold_counts(
    connection: duckdb.DuckDBPyConnection,
    *,
    fold_index: int,
) -> dict[str, int]:
    row = connection.execute(
        f"""
WITH held_out_patients AS (
    SELECT DISTINCT patient_uid
    FROM crossfit_private_groups
    WHERE patient_fold_id = {int(fold_index)}
),
fit_patients AS (
    SELECT DISTINCT patient_uid
    FROM crossfit_private_groups
    WHERE patient_fold_id <> {int(fold_index)}
)
SELECT
    COUNT(DISTINCT CASE
        WHEN patient_fold_id = {int(fold_index)} THEN patient_uid
    END) AS held_out_patient_count,
    COUNT(DISTINCT CASE
        WHEN patient_fold_id <> {int(fold_index)} THEN patient_uid
    END) AS fit_patient_count,
    SUM(CASE WHEN patient_fold_id = {int(fold_index)} THEN 1 ELSE 0 END)
        AS held_out_group_count,
    SUM(CASE WHEN patient_fold_id <> {int(fold_index)} THEN 1 ELSE 0 END)
        AS fit_group_count,
    (
        SELECT COUNT(*)
        FROM held_out_patients
        INNER JOIN fit_patients USING (patient_uid)
    ) AS patient_overlap_count
FROM crossfit_private_groups
""",
    ).fetchone()
    if row is None:
        raise ValueError("patient-fold scope is empty")
    counts = {
        "held_out_patient_count": int(row[0]),
        "fit_patient_count": int(row[1]),
        "held_out_group_count": int(row[2]),
        "fit_group_count": int(row[3]),
        "patient_overlap_count": int(row[4]),
    }
    if (
        counts["held_out_patient_count"] <= 0
        or counts["fit_patient_count"] <= 0
        or counts["patient_overlap_count"] != 0
    ):
        raise ValueError("patient-fold scope is empty or overlapping")
    return counts


def _graph_edges_query(
    config: GNNTrainingConfig,
    *,
    fold_index: int,
    generated_at: str,
) -> str:
    """Match the canonical Milestone 8 support semantics on fit patients only."""

    pcm = parquet_scan(config.patient_condition_medication_path)
    events = parquet_scan(config.event_sequences_path)
    return f"""
WITH fit_groups AS MATERIALIZED (
    SELECT
        source,
        split,
        ranking_group_id,
        stay_uid,
        index_condition_token
    FROM crossfit_private_groups
    WHERE patient_fold_id <> {int(fold_index)}
),
fit_pcm AS MATERIALIZED (
    SELECT
        pcm.source,
        pcm.split,
        pcm.stay_uid,
        pcm.ranking_group_id,
        pcm.index_condition_token,
        pcm.candidate_medication_token,
        pcm.label_prescribed
    FROM {pcm} AS pcm
    INNER JOIN fit_groups AS groups
        ON pcm.source = groups.source
        AND pcm.split = groups.split
        AND pcm.ranking_group_id = groups.ranking_group_id
),
train_positive_rows AS MATERIALIZED (
    SELECT
        source,
        split,
        stay_uid,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token
    FROM fit_pcm
    WHERE label_prescribed
        AND index_condition_token IS NOT NULL
        AND candidate_medication_token IS NOT NULL
),
train_stay_conditions AS MATERIALIZED (
    SELECT DISTINCT
        source,
        split,
        stay_uid,
        index_condition_token
    FROM fit_groups
    WHERE index_condition_token IS NOT NULL
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
        first.candidate_medication_token AS src_medication_token,
        second.candidate_medication_token AS dst_medication_token,
        first.ranking_group_id
    FROM train_positive_rows AS first
    INNER JOIN train_positive_rows AS second
        ON first.ranking_group_id = second.ranking_group_id
        AND first.candidate_medication_token < second.candidate_medication_token
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
predecision_events AS MATERIALIZED (
    SELECT DISTINCT
        events.source,
        events.split,
        events.stay_uid,
        events.event_type,
        events.event_token
    FROM {events} AS events
    INNER JOIN (
        SELECT DISTINCT source, split, stay_uid
        FROM fit_groups
    ) AS fit_stays
        ON events.source = fit_stays.source
        AND events.split = fit_stays.split
        AND events.stay_uid = fit_stays.stay_uid
    WHERE events.event_type IN ('lab', 'vital', 'intervention')
        AND events.event_time_hours_from_admit >= 0
        AND events.event_time_hours_from_admit <= 24.0
        AND events.event_token IS NOT NULL
),
condition_event_edges AS (
    SELECT
        'condition|' || conditions.index_condition_token AS src_id,
        events.event_type || '|' || events.event_token AS dst_id,
        'condition' AS src_type,
        events.event_type AS dst_type,
        CASE events.event_type
            WHEN 'lab' THEN 'condition_lab_predecision'
            WHEN 'vital' THEN 'condition_vital_predecision'
            ELSE 'condition_intervention_predecision'
        END AS relation_type,
        COUNT(DISTINCT conditions.stay_uid) AS support_count
    FROM train_stay_conditions AS conditions
    INNER JOIN predecision_events AS events
        ON conditions.source = events.source
        AND conditions.split = events.split
        AND conditions.stay_uid = events.stay_uid
    GROUP BY
        conditions.index_condition_token,
        events.event_type,
        events.event_token
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
    {sql_string(DEVELOPMENT_SOURCE)} AS fit_source,
    'train' AS fit_split,
    CAST({int(fold_index)} AS INTEGER) AS held_out_fold_index,
    {sql_string(CROSS_FIT_GRAPH_SCHEMA_VERSION)} AS graph_version,
    {sql_string(generated_at)} AS generated_at
FROM all_edges
"""


def _copy_query(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = connection.execute(
        f"""
COPY ({query})
TO {sql_string(output_path)}
(FORMAT PARQUET, COMPRESSION ZSTD)
"""
    ).fetchone()
    return int(row[0]) if row is not None and row[0] is not None else 0


def _copy_partitioned(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    output_root: Path,
) -> int:
    from pipeline.gnn_training.data import coerce_single_parquet_partitions

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


def _validate_compact_partitions(table_roots: Iterable[Path]) -> None:
    """Require at most one Parquet fragment per table/split/shard partition."""

    for table_root in table_roots:
        for shard_directory in table_root.glob("split=*/shard_id=*"):
            if len(tuple(shard_directory.glob("*.parquet"))) > 1:
                raise RuntimeError(
                    "cross-fit cache compaction invariant failed: a "
                    "table/split/shard partition contains multiple fragments"
                )


def _case_index(expression: str, mapping: Mapping[str, int], *, kind: str) -> str:
    branches = " ".join(
        f"WHEN {sql_string(name)} THEN {int(index)}" for name, index in mapping.items()
    )
    return (
        f"CASE {expression} {branches} "
        f"ELSE error('unsupported {kind} in locked patient subgraph') END"
    )


def _fold_nodes_query(
    config: GNNTrainingConfig,
    *,
    fold_graph_path: Path,
) -> str:
    nodes = parquet_scan(config.subgraph_nodes_path)
    graph = parquet_scan(fold_graph_path)
    condition_relations = ", ".join(
        sql_string(relation) for relation in _CONDITION_EVENT_RELATIONS
    )
    return f"""
WITH scoped_nodes AS MATERIALIZED (
    SELECT
        nodes.source,
        nodes.split,
        groups.stay_uid,
        nodes.subgraph_id AS ranking_group_id,
        nodes.node_index AS old_node_index,
        nodes.node_id,
        nodes.node_type,
        nodes.node_role,
        nodes.observed_predecision,
        groups.patient_fold_id,
        groups.shard_id
    FROM {nodes} AS nodes
    INNER JOIN crossfit_private_groups AS groups
        ON nodes.source = groups.source
        AND nodes.split = groups.split
        AND nodes.subgraph_id = groups.ranking_group_id
),
query_nodes AS MATERIALIZED (
    SELECT source, split, ranking_group_id, node_id AS query_node_id
    FROM scoped_nodes
    WHERE node_role = 'query_condition'
),
graph_nodes AS MATERIALIZED (
    SELECT src_id AS node_id FROM {graph}
    UNION
    SELECT dst_id AS node_id FROM {graph}
),
eligible AS (
    SELECT
        nodes.*,
        graph_nodes.node_id IS NULL AS cold_start
    FROM scoped_nodes AS nodes
    INNER JOIN query_nodes
        USING (source, split, ranking_group_id)
    LEFT JOIN graph_nodes
        ON nodes.node_id = graph_nodes.node_id
    WHERE nodes.node_role <> 'observed_context'
        OR EXISTS (
            SELECT 1
            FROM {graph} AS graph_edges
            WHERE graph_edges.src_id = query_nodes.query_node_id
                AND graph_edges.dst_id = nodes.node_id
                AND graph_edges.relation_type IN ({condition_relations})
        )
),
numbered AS (
    SELECT
        eligible.*,
        ROW_NUMBER() OVER (
            PARTITION BY source, split, ranking_group_id
            ORDER BY
                CASE node_role
                    WHEN 'query_condition' THEN 0
                    WHEN 'candidate_medication' THEN 1
                    ELSE 2
                END,
                node_type,
                node_id,
                old_node_index
        ) - 1 AS new_node_index
    FROM eligible
)
SELECT
    source,
    split,
    stay_uid,
    ranking_group_id,
    patient_fold_id,
    shard_id,
    CAST(old_node_index AS BIGINT) AS old_node_index,
    CAST(new_node_index AS BIGINT) AS node_index,
    node_id,
    node_type,
    node_role,
    CAST(observed_predecision AS BOOLEAN) AS observed_predecision,
    CAST(cold_start AS BOOLEAN) AS cold_start
FROM numbered
"""


def _fold_vocabulary_query(
    *,
    raw_nodes_path: Path,
    fold_index: int,
) -> str:
    nodes = parquet_scan(raw_nodes_path)
    return f"""
WITH fit_concepts AS (
    SELECT DISTINCT node_id
    FROM {nodes}
    WHERE patient_fold_id <> {int(fold_index)}
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
    FROM fit_concepts
)
SELECT
    CAST({PAD_INDEX} AS BIGINT) AS concept_index,
    {sql_string(PAD_TOKEN)} AS node_id,
    {sql_string(DEVELOPMENT_SOURCE)} AS fit_source,
    'train' AS fit_split,
    CAST({int(fold_index)} AS INTEGER) AS held_out_fold_index
UNION ALL
SELECT
    CAST({UNK_INDEX} AS BIGINT) AS concept_index,
    {sql_string(UNK_TOKEN)} AS node_id,
    {sql_string(DEVELOPMENT_SOURCE)} AS fit_source,
    'train' AS fit_split,
    CAST({int(fold_index)} AS INTEGER) AS held_out_fold_index
UNION ALL
SELECT
    concept_index,
    node_id,
    {sql_string(DEVELOPMENT_SOURCE)} AS fit_source,
    'train' AS fit_split,
    CAST({int(fold_index)} AS INTEGER) AS held_out_fold_index
FROM numbered
"""


def _nodes_cache_query(
    config: GNNTrainingConfig,
    *,
    raw_nodes_path: Path,
    vocabulary_path: Path,
    fold_index: int,
) -> str:
    nodes = parquet_scan(raw_nodes_path)
    vocabulary = parquet_scan(vocabulary_path)
    node_type = _case_index(
        "nodes.node_type",
        NODE_TYPE_TO_INDEX,
        kind="node type",
    )
    node_role = _case_index(
        "nodes.node_role",
        NODE_ROLE_TO_INDEX,
        kind="node role",
    )
    events = parquet_scan(config.event_sequences_path)
    finite_value = (
        "CASE WHEN events.value_numeric IS NOT NULL "
        "AND isfinite(CAST(events.value_numeric AS DOUBLE)) "
        "AND ABS(CAST(events.value_numeric AS DOUBLE)) <= 1e100 "
        "THEN CAST(events.value_numeric AS DOUBLE) ELSE NULL END"
    )
    return f"""
WITH fit_stays AS MATERIALIZED (
    SELECT DISTINCT source, split, stay_uid
    FROM {nodes}
    WHERE patient_fold_id <> {int(fold_index)}
),
fit_numeric_events AS MATERIALIZED (
    SELECT
        events.event_type,
        events.event_token,
        {finite_value} AS numeric_value
    FROM {events} AS events
    INNER JOIN fit_stays
        USING (source, split, stay_uid)
    WHERE events.event_type IN ('lab', 'vital')
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
    INNER JOIN (
        SELECT DISTINCT source, split, stay_uid FROM {nodes}
    ) AS scoped_stays
        USING (source, split, stay_uid)
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
    nodes.ranking_group_id,
    nodes.shard_id,
    nodes.node_index,
    CAST(COALESCE(vocab.concept_index, {UNK_INDEX}) AS BIGINT)
        AS node_concept_index,
    CAST({node_type} AS INTEGER) AS node_type_index,
    CAST({node_role} AS INTEGER) AS node_role_index,
    nodes.observed_predecision,
    nodes.cold_start,
    CAST(COALESCE(attributes.value_zscore, 0.0) AS REAL) AS value_zscore,
    CAST(COALESCE(attributes.value_mask, 0.0) AS REAL) AS value_mask,
    CAST(COALESCE(attributes.abnormal_direction, 0.0) AS REAL)
        AS abnormal_direction,
    CAST(COALESCE(attributes.trend_zscore_per_window, 0.0) AS REAL)
        AS trend_zscore_per_window,
    CAST(COALESCE(attributes.last_time / 24.0, 0.0) AS REAL)
        AS time_normalized,
    CAST(
        CASE
            WHEN attributes.last_time IS NULL THEN 0
            ELSE LEAST(4, FLOOR(attributes.last_time / 6.0) + 1)
        END AS INTEGER
    ) AS time_bin_index
FROM {nodes} AS nodes
LEFT JOIN {vocabulary} AS vocab
    ON nodes.node_id = vocab.node_id
LEFT JOIN attributes
    ON nodes.source = attributes.source
    AND nodes.split = attributes.split
    AND nodes.stay_uid = attributes.stay_uid
    AND nodes.node_id = attributes.event_type || '|' || attributes.event_token
"""


def _candidates_cache_query(
    config: GNNTrainingConfig,
    *,
    raw_nodes_path: Path,
) -> str:
    candidates = parquet_scan(config.subgraph_candidates_path)
    nodes = parquet_scan(raw_nodes_path)
    return f"""
WITH candidate_nodes AS MATERIALIZED (
    SELECT
        source,
        split,
        ranking_group_id,
        node_id,
        node_index,
        cold_start
    FROM {nodes}
    WHERE node_role = 'candidate_medication'
)
SELECT
    candidates.source,
    candidates.split,
    groups.ranking_group_id,
    groups.shard_id,
    candidates.index_condition_token,
    candidates.candidate_medication_token,
    CAST(nodes.node_index AS BIGINT) AS candidate_node_index,
    CAST(candidates.candidate_rank AS BIGINT) AS candidate_rank,
    CAST(candidates.label_prescribed AS BOOLEAN) AS label_prescribed,
    CAST(nodes.cold_start AS BOOLEAN) AS cold_start
FROM {candidates} AS candidates
INNER JOIN crossfit_private_groups AS groups
    ON candidates.source = groups.source
    AND candidates.split = groups.split
    AND candidates.subgraph_id = groups.ranking_group_id
INNER JOIN candidate_nodes AS nodes
    ON candidates.source = nodes.source
    AND candidates.split = nodes.split
    AND candidates.subgraph_id = nodes.ranking_group_id
    AND nodes.node_id =
        'medication|' || candidates.candidate_medication_token
"""


def _forward_edges_query(
    config: GNNTrainingConfig,
    *,
    fold_graph_path: Path,
    raw_nodes_path: Path,
    shard_id: int | None = None,
) -> str:
    """Filter locked memberships, but replace every fitted support value.

    Large patient-subgraph edge tables must not be forced into a single
    ``MATERIALIZED`` CTE.  Callers should rebuild one ranking-group shard at a
    time so DuckDB can spill under a bounded memory ceiling.
    """

    locked_edges = parquet_scan(config.subgraph_edges_path)
    graph = parquet_scan(fold_graph_path)
    nodes = parquet_scan(raw_nodes_path)
    forward_relations = ", ".join(
        sql_string(relation) for relation in FORWARD_RELATION_TYPES
    )
    shard_filter = "" if shard_id is None else f"AND groups.shard_id = {int(shard_id)}"
    return f"""
WITH scoped_locked_edges AS (
    SELECT
        edges.source,
        edges.split,
        edges.subgraph_id AS ranking_group_id,
        edges.src_node_index AS old_src_node_index,
        edges.dst_node_index AS old_dst_node_index,
        edges.src_id,
        edges.dst_id,
        edges.relation_type,
        groups.shard_id
    FROM {locked_edges} AS edges
    INNER JOIN crossfit_private_groups AS groups
        ON edges.source = groups.source
        AND edges.split = groups.split
        AND edges.subgraph_id = groups.ranking_group_id
    WHERE edges.relation_type IN ({forward_relations})
      {shard_filter}
),
surviving AS (
    SELECT
        edges.*,
        graph_edges.support_count
    FROM scoped_locked_edges AS edges
    INNER JOIN {graph} AS graph_edges
        ON edges.src_id = graph_edges.src_id
        AND edges.dst_id = graph_edges.dst_id
        AND edges.relation_type = graph_edges.relation_type
),
node_map AS (
    SELECT
        source,
        split,
        ranking_group_id,
        old_node_index,
        node_index,
        shard_id
    FROM {nodes}
)
SELECT
    surviving.source,
    surviving.split,
    surviving.ranking_group_id,
    surviving.shard_id,
    CAST(src.node_index AS BIGINT) AS src_node_index,
    CAST(dst.node_index AS BIGINT) AS dst_node_index,
    surviving.relation_type,
    CAST(surviving.support_count AS BIGINT) AS support_count
FROM surviving
INNER JOIN node_map AS src
    ON surviving.source = src.source
    AND surviving.split = src.split
    AND surviving.ranking_group_id = src.ranking_group_id
    AND surviving.old_src_node_index = src.old_node_index
INNER JOIN node_map AS dst
    ON surviving.source = dst.source
    AND surviving.split = dst.split
    AND surviving.ranking_group_id = dst.ranking_group_id
    AND surviving.old_dst_node_index = dst.old_node_index
"""


def _groups_cache_query(
    *,
    raw_nodes_path: Path,
    forward_edges_path: Path,
    candidate_cache_root: Path,
) -> str:
    nodes = parquet_scan(raw_nodes_path)
    forward_edges = _parquet_scan_any(forward_edges_path)
    candidates = (
        "read_parquet("
        f"{sql_string(candidate_cache_root / '**' / '*.parquet')}, "
        "hive_partitioning = TRUE)"
    )
    return f"""
WITH node_counts AS (
    SELECT
        source,
        split,
        ranking_group_id,
        COUNT(*) AS node_count
    FROM {nodes}
    GROUP BY source, split, ranking_group_id
),
edge_counts AS (
    SELECT
        source,
        split,
        ranking_group_id,
        COUNT(*) AS forward_edge_count
    FROM {forward_edges}
    GROUP BY source, split, ranking_group_id
),
candidate_counts AS (
    SELECT
        source,
        split,
        ranking_group_id,
        COUNT(*) AS candidate_count,
        SUM(CASE WHEN label_prescribed THEN 1 ELSE 0 END) AS positive_count
    FROM {candidates}
    GROUP BY source, split, ranking_group_id
)
SELECT
    groups.source,
    groups.split,
    groups.ranking_group_id,
    groups.patient_fold_id,
    CAST(node_counts.node_count AS BIGINT) AS node_count,
    CAST(
        2 * COALESCE(edge_counts.forward_edge_count, 0)
        + node_counts.node_count AS BIGINT
    ) AS expanded_edge_count,
    CAST(candidate_counts.candidate_count AS BIGINT) AS candidate_count,
    CAST(candidate_counts.positive_count AS BIGINT) AS positive_count,
    groups.shard_id
FROM crossfit_private_groups AS groups
INNER JOIN node_counts
    USING (source, split, ranking_group_id)
LEFT JOIN edge_counts
    USING (source, split, ranking_group_id)
INNER JOIN candidate_counts
    USING (source, split, ranking_group_id)
"""


def _edges_cache_query(
    *,
    raw_nodes_path: Path,
    forward_edges_path: Path,
) -> str:
    nodes = parquet_scan(raw_nodes_path)
    forward_edges = _parquet_scan_any(forward_edges_path)
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
    return f"""
WITH source_edges AS (
    SELECT *
    FROM {forward_edges}
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
        LN(1.0 + CAST(source_edges.support_count AS DOUBLE))
            AS transformed_support
    FROM source_edges
    CROSS JOIN (VALUES (FALSE), (TRUE)) AS directions(is_reverse)
),
self_edges AS (
    SELECT
        nodes.source,
        nodes.split,
        nodes.ranking_group_id,
        nodes.shard_id,
        CAST(nodes.node_index AS BIGINT) AS src_node_index,
        CAST(nodes.node_index AS BIGINT) AS dst_node_index,
        CAST({self_relation} AS INTEGER) AS relation_index,
        LN(2.0) AS transformed_support
    FROM {nodes} AS nodes
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


def _scan_cache_table(cache_root: Path, table_name: str) -> str:
    return (
        "read_parquet("
        f"{sql_string(cache_root / table_name / '**' / '*.parquet')}, "
        "hive_partitioning = TRUE)"
    )


def _validate_complete_cache(
    connection: duckdb.DuckDBPyConnection,
    *,
    cache_root: Path,
) -> None:
    scans = {
        table_name: _scan_cache_table(cache_root, table_name)
        for table_name in CACHE_TABLES
    }
    mismatch = connection.execute(
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
FROM {scans[GROUP_TABLE]} AS groups
LEFT JOIN node_counts USING (source, split, ranking_group_id)
LEFT JOIN edge_counts USING (source, split, ranking_group_id)
LEFT JOIN candidate_counts USING (source, split, ranking_group_id)
WHERE node_counts.row_count IS NULL
    OR edge_counts.row_count IS NULL
    OR candidate_counts.row_count IS NULL
    OR node_counts.row_count <> groups.node_count
    OR edge_counts.row_count <> groups.expanded_edge_count
    OR candidate_counts.row_count <> groups.candidate_count
    OR candidate_counts.positive_count <> groups.positive_count
"""
    ).fetchone()
    if mismatch is not None and int(mismatch[0]) > 0:
        raise ValueError("cross-fit cache contains incomplete ranking groups")

    scope_mismatch = connection.execute(
        f"""
WITH cached_groups AS (
    SELECT
        source,
        split,
        ranking_group_id,
        patient_fold_id,
        candidate_count,
        positive_count
    FROM {scans[GROUP_TABLE]}
),
missing_or_changed AS (
    SELECT COUNT(*) AS mismatch_count
    FROM crossfit_private_groups AS expected
    LEFT JOIN cached_groups AS cached
        USING (source, split, ranking_group_id)
    WHERE cached.ranking_group_id IS NULL
        OR cached.patient_fold_id <> expected.patient_fold_id
        OR cached.candidate_count <> expected.candidate_count
        OR cached.positive_count <> expected.positive_count
),
unexpected AS (
    SELECT COUNT(*) AS mismatch_count
    FROM cached_groups AS cached
    LEFT JOIN crossfit_private_groups AS expected
        USING (source, split, ranking_group_id)
    WHERE expected.ranking_group_id IS NULL
)
SELECT
    (SELECT mismatch_count FROM missing_or_changed)
    + (SELECT mismatch_count FROM unexpected)
"""
    ).fetchone()
    if scope_mismatch is not None and int(scope_mismatch[0]) > 0:
        raise ValueError("cross-fit cache group scope differs from the locked inputs")

    totals = {
        table_name: int(
            connection.execute(f"SELECT COUNT(*) FROM {scan}").fetchone()[0]
        )
        for table_name, scan in scans.items()
    }
    if totals[GROUP_TABLE] <= 0:
        raise ValueError("cross-fit cache contains no ranking groups")

    shard_mismatch = connection.execute(
        f"""
WITH all_rows AS (
    SELECT
        source, split, ranking_group_id,
        MIN(shard_id) AS minimum_shard,
        MAX(shard_id) AS maximum_shard
    FROM (
        SELECT source, split, ranking_group_id, shard_id
        FROM {scans[GROUP_TABLE]}
        UNION ALL
        SELECT source, split, ranking_group_id, shard_id
        FROM {scans[NODE_TABLE]}
        UNION ALL
        SELECT source, split, ranking_group_id, shard_id
        FROM {scans[EDGE_TABLE]}
        UNION ALL
        SELECT source, split, ranking_group_id, shard_id
        FROM {scans[CANDIDATE_TABLE]}
    )
    GROUP BY source, split, ranking_group_id
)
SELECT COUNT(*)
FROM all_rows
WHERE minimum_shard <> maximum_shard
"""
    ).fetchone()
    if shard_mismatch is not None and int(shard_mismatch[0]) > 0:
        raise ValueError("cross-fit cache split a ranking group across shards")


def _relation_aggregates(
    connection: duckdb.DuckDBPyConnection,
    *,
    graph_path: Path,
) -> list[dict[str, Any]]:
    cursor = connection.execute(
        f"""
SELECT
    relation_type,
    COUNT(*) AS edge_count,
    COALESCE(SUM(support_count), 0) AS total_support_count,
    COALESCE(MIN(support_count), 0) AS minimum_support_count,
    COALESCE(MAX(support_count), 0) AS maximum_support_count
FROM {parquet_scan(graph_path)}
GROUP BY relation_type
ORDER BY relation_type
"""
    )
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _fixed_vocabulary_payloads() -> dict[str, dict[str, Any]]:
    return {
        "node_type_vocabulary.json": {
            "schema_version": FEATURE_LAYOUT_VERSION,
            "vocabulary": list(NODE_TYPE_VOCABULARY),
            "token_to_index": {
                name: int(index) for name, index in NODE_TYPE_TO_INDEX.items()
            },
        },
        "node_role_vocabulary.json": {
            "schema_version": FEATURE_LAYOUT_VERSION,
            "vocabulary": list(NODE_ROLE_VOCABULARY),
            "token_to_index": {
                name: int(index) for name, index in NODE_ROLE_TO_INDEX.items()
            },
        },
        "relation_vocabulary.json": {
            "schema_version": RELATION_VOCABULARY_VERSION,
            "status": "completed",
            "relations": list(RELATION_TYPES),
            "relation_to_index": {
                name: int(index) for name, index in RELATION_TO_INDEX.items()
            },
        },
    }


def _feature_layout(
    config: GNNTrainingConfig,
    *,
    fold_index: int,
    concept_vocab_size: int,
) -> dict[str, Any]:
    return {
        "schema_version": FEATURE_LAYOUT_VERSION,
        "cache_schema_version": CROSS_FIT_CACHE_SCHEMA_VERSION,
        "scope": CROSS_FIT_SELECTION_SCOPE,
        "selection_eligible": True,
        "held_out_fold_index": int(fold_index),
        "fit_fold_indices": sorted(set(range(config.fold_count)) - {fold_index}),
        "fit_source": DEVELOPMENT_SOURCE,
        "fit_split": "train",
        "pad_index": PAD_INDEX,
        "unk_index": UNK_INDEX,
        "reserved_token_count": RESERVED_TOKEN_COUNT,
        "concept_vocab_size": int(concept_vocab_size),
        "node_type_vocab_size": len(NODE_TYPE_VOCABULARY),
        "node_role_vocab_size": len(NODE_ROLE_VOCABULARY),
        "relation_count": len(RELATION_TYPES),
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
        "numeric_attribute_fit_scope": "mimiciv_train_excluding_held_out_fold",
        "shard_count": int(config.shard_count),
        "shard_key": ["source", "ranking_group_id"],
        "edge_support_transform": "log1p",
        "edge_weight_normalization": ("incoming_per_subgraph_relation_destination"),
        "fit_excludes_held_out_fold": {
            "graph": True,
            "vocabulary": True,
            "support": True,
        },
    }


def _table_counts(
    connection: duckdb.DuckDBPyConnection,
    *,
    cache_root: Path,
) -> dict[str, int]:
    return {
        table_name: int(
            connection.execute(
                f"SELECT COUNT(*) FROM {_scan_cache_table(cache_root, table_name)}"
            ).fetchone()[0]
        )
        for table_name in CACHE_TABLES
    }


def _fold_coverage(
    connection: duckdb.DuckDBPyConnection,
    *,
    cache_root: Path,
    fold_index: int,
) -> dict[str, int]:
    row = connection.execute(
        f"""
SELECT
    COUNT(*) AS ranking_group_count,
    SUM(CASE WHEN positive_count > 0 THEN 1 ELSE 0 END)
        AS positive_group_count,
    SUM(CASE WHEN positive_count = 0 THEN 1 ELSE 0 END)
        AS zero_positive_group_count,
    SUM(CASE WHEN patient_fold_id = {int(fold_index)} THEN 1 ELSE 0 END)
        AS held_out_group_count,
    SUM(CASE WHEN patient_fold_id <> {int(fold_index)} THEN 1 ELSE 0 END)
        AS fit_group_count
FROM {_scan_cache_table(cache_root, GROUP_TABLE)}
"""
    ).fetchone()
    if row is None:
        raise ValueError("cross-fit cache coverage is unavailable")
    return {
        "ranking_group_count": int(row[0]),
        "positive_group_count": int(row[1]),
        "zero_positive_group_count": int(row[2]),
        "held_out_group_count": int(row[3]),
        "fit_group_count": int(row[4]),
    }


def _build_fold(
    connection: duckdb.DuckDBPyConnection,
    config: GNNTrainingConfig,
    *,
    stage_crossfit_root: Path,
    work_root: Path,
    fold_index: int,
    generated_at: str,
) -> dict[str, Any]:
    fold_root = stage_crossfit_root / f"fold_{fold_index:02d}"
    fold_work_root = work_root / f"fold_{fold_index:02d}"
    fold_root.mkdir(parents=True, exist_ok=False)
    fold_work_root.mkdir(parents=True, exist_ok=False)

    graph_path = fold_root / config.fold_graph_edges_path(fold_index).name
    graph_row_count = _copy_query(
        connection,
        _graph_edges_query(
            config,
            fold_index=fold_index,
            generated_at=generated_at,
        ),
        graph_path,
    )
    relation_aggregates = _relation_aggregates(
        connection,
        graph_path=graph_path,
    )
    fold_counts = _fold_counts(connection, fold_index=fold_index)
    graph_manifest = {
        "schema_version": CROSS_FIT_GRAPH_SCHEMA_VERSION,
        "status": "completed",
        "scope": CROSS_FIT_SELECTION_SCOPE,
        "selection_eligible": True,
        "fit_source": DEVELOPMENT_SOURCE,
        "fit_split": "train",
        "held_out_fold_index": int(fold_index),
        "fit_fold_indices": sorted(set(range(config.fold_count)) - {fold_index}),
        "edge_count": int(graph_row_count),
        "relation_aggregates": relation_aggregates,
        "graph_edges_sha256": sha256_file(graph_path),
        "exclusion_proof": {
            "graph": True,
            "vocabulary": True,
            "support": True,
        },
        "data_safety": {
            "manifest_contains_patient_rows": False,
            "manifest_contains_identifier_values": False,
            "graph_is_concept_aggregate_only": True,
        },
    }
    graph_manifest_path = fold_root / GRAPH_MANIFEST_NAME
    _write_json(graph_manifest_path, graph_manifest)

    raw_nodes_path = fold_work_root / "nodes.parquet"
    _copy_query(
        connection,
        _fold_nodes_query(config, fold_graph_path=graph_path),
        raw_nodes_path,
    )

    vocab_root = fold_root / "vocab"
    vocab_root.mkdir(parents=True, exist_ok=False)
    concept_vocabulary_path = (
        vocab_root / config.fold_concept_vocabulary_path(fold_index).name
    )
    concept_vocab_size = _copy_query(
        connection,
        _fold_vocabulary_query(
            raw_nodes_path=raw_nodes_path,
            fold_index=fold_index,
        ),
        concept_vocabulary_path,
    )
    for name, payload in _fixed_vocabulary_payloads().items():
        _write_json(vocab_root / name, payload)

    layout = _feature_layout(
        config,
        fold_index=fold_index,
        concept_vocab_size=concept_vocab_size,
    )
    layout_path = fold_root / config.fold_feature_layout_path(fold_index).name
    _write_json(layout_path, layout)

    raw_forward_edges_root = fold_work_root / "forward_edges"
    raw_forward_edges_root.mkdir(parents=True, exist_ok=False)
    for shard_id in range(config.shard_count):
        _copy_query(
            connection,
            _forward_edges_query(
                config,
                fold_graph_path=graph_path,
                raw_nodes_path=raw_nodes_path,
                shard_id=shard_id,
            ),
            raw_forward_edges_root / f"shard_{shard_id:03d}.parquet",
        )

    cache_root = fold_root / "cache"
    shards_root = cache_root / "shards"
    table_counts: dict[str, int] = {}
    table_counts[NODE_TABLE] = _copy_partitioned(
        connection,
        _nodes_cache_query(
            config,
            raw_nodes_path=raw_nodes_path,
            vocabulary_path=concept_vocabulary_path,
            fold_index=fold_index,
        ),
        shards_root / NODE_TABLE,
    )
    table_counts[CANDIDATE_TABLE] = _copy_partitioned(
        connection,
        _candidates_cache_query(config, raw_nodes_path=raw_nodes_path),
        shards_root / CANDIDATE_TABLE,
    )
    table_counts[EDGE_TABLE] = _copy_partitioned(
        connection,
        _edges_cache_query(
            raw_nodes_path=raw_nodes_path,
            forward_edges_path=raw_forward_edges_root,
        ),
        shards_root / EDGE_TABLE,
    )
    table_counts[GROUP_TABLE] = _copy_partitioned(
        connection,
        _groups_cache_query(
            raw_nodes_path=raw_nodes_path,
            forward_edges_path=raw_forward_edges_root,
            candidate_cache_root=shards_root / CANDIDATE_TABLE,
        ),
        shards_root / GROUP_TABLE,
    )
    _validate_compact_partitions(
        shards_root / table_name
        for table_name in (GROUP_TABLE, NODE_TABLE, EDGE_TABLE, CANDIDATE_TABLE)
    )

    _validate_complete_cache(connection, cache_root=shards_root)
    physical_counts = _table_counts(connection, cache_root=shards_root)
    if physical_counts != table_counts:
        raise RuntimeError("cross-fit cache physical row counts changed during build")
    coverage = _fold_coverage(
        connection,
        cache_root=shards_root,
        fold_index=fold_index,
    )
    if (
        coverage["held_out_group_count"] != fold_counts["held_out_group_count"]
        or coverage["fit_group_count"] != fold_counts["fit_group_count"]
    ):
        raise ValueError("cross-fit cache fold coverage does not match patient scope")

    cache_manifest_relative = (
        config.fold_cache_manifest_path(fold_index)
        .relative_to(config.fold_graph_root(fold_index))
        .as_posix()
    )
    artifact_hashes = _artifact_hashes(
        fold_root,
        excluded_relative_paths=(cache_manifest_relative,),
    )
    tree_digest = artifact_tree_digest(artifact_hashes)
    cache_manifest = {
        "schema_version": CROSS_FIT_CACHE_SCHEMA_VERSION,
        "artifact_lock_version": CROSS_FIT_ARTIFACT_LOCK_VERSION,
        "status": "completed",
        "scope": CROSS_FIT_SELECTION_SCOPE,
        "selection_eligible": True,
        "fit_source": DEVELOPMENT_SOURCE,
        "fit_split": "train",
        "held_out_fold_index": int(fold_index),
        "fit_fold_indices": sorted(set(range(config.fold_count)) - {fold_index}),
        "seed": int(config.seed),
        "fold_count": int(config.fold_count),
        "shard_count": int(config.shard_count),
        **fold_counts,
        "table_row_counts": table_counts,
        "coverage": coverage,
        "vocab_sizes": {
            "concept_vocab_size": int(concept_vocab_size),
            "node_type_vocab_size": len(NODE_TYPE_VOCABULARY),
            "node_role_vocab_size": len(NODE_ROLE_VOCABULARY),
            "relation_count": len(RELATION_TYPES),
        },
        "artifact_hashes": artifact_hashes,
        "artifact_tree_digest": tree_digest,
        "exclusion_proof": {
            "graph": True,
            "vocabulary": True,
            "support": True,
        },
        "data_safety": {
            "manifest_contains_patient_rows": False,
            "manifest_contains_identifier_values": False,
            "contains_row_samples": False,
            "local_cache_contains_patient_level_rows": True,
            "direct_patient_identifiers_present": False,
            "restricted_join_keys_present": True,
        },
    }
    cache_manifest_path = fold_root / config.fold_cache_manifest_path(
        fold_index
    ).relative_to(config.fold_graph_root(fold_index))
    _write_json(cache_manifest_path, cache_manifest)

    return {
        "fold_index": int(fold_index),
        "held_out_fold_index": int(fold_index),
        "fit_fold_indices": sorted(set(range(config.fold_count)) - {fold_index}),
        "fit_source": DEVELOPMENT_SOURCE,
        "fit_split": "train",
        **fold_counts,
        "graph_fit_excludes_held_out_fold": True,
        "vocabulary_fit_excludes_held_out_fold": True,
        "support_fit_excludes_held_out_fold": True,
        "exclusion_proof": {
            "graph": True,
            "vocabulary": True,
            "support": True,
        },
        "scope": CROSS_FIT_SELECTION_SCOPE,
        "selection_eligible": True,
        "table_row_counts": table_counts,
        "coverage": coverage,
        "relation_aggregates": relation_aggregates,
        "artifact_locks": {
            "graph_edges": {
                "relative_path": graph_path.relative_to(fold_root).as_posix(),
                "sha256": sha256_file(graph_path),
            },
            "graph_manifest": {
                "relative_path": graph_manifest_path.relative_to(fold_root).as_posix(),
                "sha256": sha256_file(graph_manifest_path),
            },
            "feature_layout": {
                "relative_path": layout_path.relative_to(fold_root).as_posix(),
                "sha256": sha256_file(layout_path),
            },
            "cache_manifest": {
                "relative_path": cache_manifest_path.relative_to(fold_root).as_posix(),
                "sha256": sha256_file(cache_manifest_path),
            },
            "artifact_tree_digest": tree_digest,
        },
    }


def _promote_crossfit_tree(
    config: GNNTrainingConfig,
    *,
    staged_crossfit_root: Path,
) -> None:
    destination = config.crossfit_root
    _guard_restricted_output(config, staged_crossfit_root)
    _guard_restricted_output(config, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    transaction_id = uuid.uuid4().hex
    backup = destination.with_name(f".{destination.name}.backup-{transaction_id}")
    _guard_restricted_output(config, backup)
    promoted = False
    try:
        if destination.exists():
            os.replace(destination, backup)
        os.replace(staged_crossfit_root, destination)
        promoted = True
    except Exception:
        if promoted and destination.exists():
            _remove_owned_path(config, destination)
        if backup.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        _remove_owned_path(config, backup)


def _base_report(
    config: GNNTrainingConfig,
    *,
    generated_at: str,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": CROSS_FIT_SCHEMA_VERSION,
        "status": status,
        "stage": "prepare-crossfit-graphs",
        "mode": config.mode,
        "generated_at": generated_at,
        "contract_digest": _contract_digest(config),
        "seed": int(config.seed),
        "fold_count": int(config.fold_count),
        "patient_grouped": True,
        "fit_scope": {
            "source": DEVELOPMENT_SOURCE,
            "split": "train",
            "grouping_unit": "patient_uid",
        },
        "scope": CROSS_FIT_SELECTION_SCOPE,
        "selection_eligible": True,
        "folds": [],
        "gate_policy": config.gate_policy(),
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "report_contains_identifier_values": False,
            "restricted_artifacts_remain_under_gnn_root": True,
        },
    }


def _tree_size_bytes(root: Path) -> int:
    """Return aggregate physical file bytes without opening restricted rows."""

    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _full_graph_cache_size_bytes(config: GNNTrainingConfig) -> int:
    """Size graph cache inputs comparable to each cross-fit fold tree."""

    total = _tree_size_bytes(config.shards_root) + _tree_size_bytes(config.vocab_root)
    for path in (config.feature_layout_path, config.cache_manifest_path):
        if path.is_file():
            total += path.stat().st_size
    return total


def _capacity_review(config: GNNTrainingConfig) -> dict[str, Any]:
    """Fail closed unless protected cross-fit storage was capacity-reviewed."""

    full_cache_bytes = _full_graph_cache_size_bytes(config)
    estimated_stage_bytes = math.ceil(
        full_cache_bytes * max(1, config.fold_count - 1) * 1.25
    )
    if config.allow_ungated:
        return {
            "required_for_protected_run": False,
            "synthetic_bypass": True,
            "full_cache_bytes": full_cache_bytes,
            "estimated_stage_bytes": estimated_stage_bytes,
        }

    raw_minimum = os.environ.get(CROSS_FIT_CAPACITY_ENV)
    if raw_minimum is None:
        raise ValueError(
            f"{CROSS_FIT_CAPACITY_ENV} must record the capacity-reviewed "
            "minimum free space for protected cross-fit preparation"
        )
    try:
        reviewed_gib = float(raw_minimum)
    except ValueError as error:
        raise ValueError(f"{CROSS_FIT_CAPACITY_ENV} must be numeric") from error
    if not math.isfinite(reviewed_gib) or reviewed_gib <= 0:
        raise ValueError(f"{CROSS_FIT_CAPACITY_ENV} must be positive and finite")

    reviewed_bytes = math.ceil(reviewed_gib * GIBIBYTE)
    required_bytes = max(reviewed_bytes, estimated_stage_bytes)
    available_bytes = shutil.disk_usage(config.gnn_root).free
    if available_bytes < required_bytes:
        raise OSError(
            "cross-fit preparation lacks the capacity-reviewed free space: "
            f"required_bytes={required_bytes}, available_bytes={available_bytes}"
        )
    return {
        "required_for_protected_run": True,
        "synthetic_bypass": False,
        "reviewed_minimum_gib": reviewed_gib,
        "full_cache_bytes": full_cache_bytes,
        "estimated_stage_bytes": estimated_stage_bytes,
        "required_free_bytes": required_bytes,
        "available_free_bytes": available_bytes,
        "passed": True,
    }


def prepare_crossfit_graph_caches(config: GNNTrainingConfig) -> dict[str, Any]:
    """Build and atomically promote every fold-excluded selection cache.

    Protected runs enforce the same immutable upstream preflight as ``prepare``.
    ``allow_ungated`` remains available only for synthetic roots, as enforced by
    the package path-safety contract.
    """

    from pipeline.gnn_training.contract import path_safety_errors, preflight_errors

    generated_at = _utc_now()
    report = _base_report(config, generated_at=generated_at, status="running")
    errors = (
        path_safety_errors(config)
        if config.allow_ungated
        else preflight_errors(config, stage="prepare")
    )
    if errors:
        report["status"] = "blocked_preflight"
        report["errors"] = errors
        _write_public_report(config, report)
        return report

    stage_root: Path | None = None
    try:
        config.gnn_root.mkdir(parents=True, exist_ok=True)
        report["capacity_review"] = _capacity_review(config)
        stage_root = config.gnn_root / f".crossfit-stage-{uuid.uuid4().hex}"
        _guard_restricted_output(config, stage_root)
        stage_root.mkdir(parents=False, exist_ok=False)
        stage_crossfit_root = stage_root / "crossfit"
        work_root = stage_root / "work"
        stage_crossfit_root.mkdir(parents=False, exist_ok=False)
        work_root.mkdir(parents=False, exist_ok=False)

        upstream_locks = _upstream_locks(config)
        folds: list[dict[str, Any]] = []
        # One DuckDB connection per fold so hash-join buffers and temp files are
        # released before the next fold-excluded rebuild starts.
        for fold_index in range(config.fold_count):
            with duckdb.connect(database=":memory:") as connection:
                _configure_connection(config, connection)
                if fold_index == 0:
                    _validate_input_schemas(connection, config)
                _create_private_group_scope(connection, config)
                folds.append(
                    _build_fold(
                        connection,
                        config,
                        stage_crossfit_root=stage_crossfit_root,
                        work_root=work_root,
                        fold_index=fold_index,
                        generated_at=generated_at,
                    )
                )

        if _upstream_locks(config) != upstream_locks:
            raise RuntimeError("locked upstream artifacts changed during cross-fitting")

        complete_tree_hashes = _artifact_hashes(stage_crossfit_root)
        complete_tree_digest = artifact_tree_digest(complete_tree_hashes)
        _promote_crossfit_tree(
            config,
            staged_crossfit_root=stage_crossfit_root,
        )
        report.update(
            {
                "status": "completed",
                "folds": folds,
                "upstream_artifact_locks": upstream_locks,
                "artifact_lock_version": CROSS_FIT_ARTIFACT_LOCK_VERSION,
                "crossfit_tree_file_count": len(complete_tree_hashes),
                "crossfit_tree_digest": complete_tree_digest,
                "full_train_cache_reused_for_selection": False,
            }
        )
    except Exception as error:  # noqa: BLE001 - aggregate fail-closed report
        report["status"] = "failed"
        report["reason"] = _safe_reason(error)
    finally:
        if stage_root is not None and stage_root.exists():
            _remove_owned_path(config, stage_root)

    _write_public_report(config, report)
    return report


# Short compatibility alias for callers that prefer the stage noun.
prepare_crossfit_caches = prepare_crossfit_graph_caches


def _lock_error(
    code: str,
    detail: str,
    *,
    fold_index: int | None = None,
    artifact_name: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"code": code, "detail": detail}
    if fold_index is not None:
        row["fold_index"] = int(fold_index)
    if artifact_name is not None:
        row["artifact_name"] = artifact_name
    return row


def crossfit_artifact_lock_errors(
    config: GNNTrainingConfig,
) -> list[dict[str, Any]]:
    """Verify the public-to-local hash chain before model selection.

    The verifier resolves every fold path from configuration, never from an
    untrusted manifest path, and checks:

    * public hashes for graph, layout, graph manifest and cache manifest;
    * every exact file hash recorded by the local cache manifest;
    * each fold tree digest; and
    * the complete promoted cross-fit tree digest.
    """

    if not config.crossfit_graph_manifest_path.is_file():
        return [
            _lock_error(
                "missing_crossfit_artifact_lock",
                "completed cross-fit artifact manifest is missing",
            )
        ]
    try:
        payload = load_json(config.crossfit_graph_manifest_path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return [
            _lock_error(
                "invalid_crossfit_artifact_lock",
                "cross-fit artifact manifest is unreadable or malformed",
            )
        ]
    if payload.get("status") != "completed":
        return [
            _lock_error(
                "invalid_crossfit_artifact_lock",
                "cross-fit artifact manifest is not completed",
            )
        ]
    folds = payload.get("folds")
    if not isinstance(folds, list):
        return [
            _lock_error(
                "invalid_crossfit_artifact_lock",
                "cross-fit artifact manifest has no fold locks",
            )
        ]

    if config.crossfit_root.is_dir():
        stat_file_count, stat_digest = _artifact_stat_digest(config.crossfit_root)
        if _attestation_is_current(
            config,
            payload=payload,
            stat_file_count=stat_file_count,
            stat_digest=stat_digest,
        ):
            print(
                json.dumps(
                    {
                        "event": "crossfit_preflight_attestation_reused",
                        "file_count": stat_file_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return []
    else:
        stat_file_count, stat_digest = 0, ""

    errors: list[dict[str, Any]] = []
    if config.crossfit_root.is_dir():
        complete_tree_hashes = _artifact_hashes(
            config.crossfit_root,
            progress_interval_files=10_000,
            progress_label="crossfit_preflight",
        )
    else:
        complete_tree_hashes = {}
    by_index = {
        row.get("fold_index"): row
        for row in folds
        if isinstance(row, dict) and isinstance(row.get("fold_index"), int)
    }
    for fold_index in range(config.fold_count):
        fold = by_index.get(fold_index)
        if fold is None:
            errors.append(
                _lock_error(
                    "missing_crossfit_fold_lock",
                    "cross-fit artifact manifest does not lock this fold",
                    fold_index=fold_index,
                )
            )
            continue
        locks = fold.get("artifact_locks")
        if not isinstance(locks, dict):
            errors.append(
                _lock_error(
                    "invalid_crossfit_fold_lock",
                    "cross-fit fold artifact locks are malformed",
                    fold_index=fold_index,
                )
            )
            continue
        fold_root = config.fold_graph_root(fold_index)
        expected_paths = {
            "graph_edges": config.fold_graph_edges_path(fold_index),
            "graph_manifest": fold_root / GRAPH_MANIFEST_NAME,
            "feature_layout": config.fold_feature_layout_path(fold_index),
            "cache_manifest": config.fold_cache_manifest_path(fold_index),
        }
        for artifact_name, path in expected_paths.items():
            lock = locks.get(artifact_name)
            expected_relative = path.relative_to(fold_root).as_posix()
            tree_relative = path.relative_to(config.crossfit_root).as_posix()
            if (
                not isinstance(lock, dict)
                or lock.get("relative_path") != expected_relative
                or not isinstance(lock.get("sha256"), str)
                or complete_tree_hashes.get(tree_relative) != lock["sha256"]
            ):
                errors.append(
                    _lock_error(
                        "crossfit_artifact_hash_mismatch",
                        "cross-fit fold artifact differs from its public lock",
                        fold_index=fold_index,
                        artifact_name=artifact_name,
                    )
                )

        cache_manifest_path = expected_paths["cache_manifest"]
        if not cache_manifest_path.is_file():
            continue
        try:
            cache_manifest = load_json(cache_manifest_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            errors.append(
                _lock_error(
                    "invalid_crossfit_cache_manifest",
                    "cross-fit cache manifest is unreadable or malformed",
                    fold_index=fold_index,
                )
            )
            continue
        recorded_hashes = cache_manifest.get("artifact_hashes")
        if not isinstance(recorded_hashes, dict) or not all(
            isinstance(relative, str) and isinstance(file_hash, str)
            for relative, file_hash in recorded_hashes.items()
        ):
            errors.append(
                _lock_error(
                    "invalid_crossfit_cache_manifest",
                    "cross-fit cache exact artifact hashes are malformed",
                    fold_index=fold_index,
                )
            )
            continue
        for relative, expected_hash in recorded_hashes.items():
            path = fold_root / relative
            if not _is_under(path, fold_root):
                errors.append(
                    _lock_error(
                        "unsafe_crossfit_artifact_path",
                        "cross-fit cache manifest contains an unsafe relative path",
                        fold_index=fold_index,
                    )
                )
                continue
            tree_relative = path.relative_to(config.crossfit_root).as_posix()
            if complete_tree_hashes.get(tree_relative) != expected_hash:
                errors.append(
                    _lock_error(
                        "crossfit_artifact_hash_mismatch",
                        "cross-fit cache file differs from its exact local lock",
                        fold_index=fold_index,
                        artifact_name=relative,
                    )
                )
        fold_prefix = fold_root.relative_to(config.crossfit_root).as_posix() + "/"
        excluded_manifest = cache_manifest_path.relative_to(fold_root).as_posix()
        fold_hashes = {
            tree_relative.removeprefix(fold_prefix): file_hash
            for tree_relative, file_hash in complete_tree_hashes.items()
            if tree_relative.startswith(fold_prefix)
            and tree_relative.removeprefix(fold_prefix) != excluded_manifest
        }
        actual_tree = artifact_tree_digest(fold_hashes)
        if (
            cache_manifest.get("artifact_tree_digest") != actual_tree
            or locks.get("artifact_tree_digest") != actual_tree
        ):
            errors.append(
                _lock_error(
                    "crossfit_artifact_tree_mismatch",
                    "cross-fit fold artifact tree differs from its digest lock",
                    fold_index=fold_index,
                )
            )

    if config.crossfit_root.is_dir():
        actual_crossfit_tree = artifact_tree_digest(complete_tree_hashes)
        if payload.get("crossfit_tree_digest") != actual_crossfit_tree:
            errors.append(
                _lock_error(
                    "crossfit_artifact_tree_mismatch",
                    "complete cross-fit artifact tree differs from its public lock",
                )
            )
    else:
        errors.append(
            _lock_error(
                "missing_crossfit_artifact_tree",
                "promoted cross-fit artifact tree is missing",
            )
        )
    if not errors:
        _write_preflight_attestation(
            config,
            payload=payload,
            stat_file_count=stat_file_count,
            stat_digest=stat_digest,
        )
    return errors


def assert_crossfit_artifact_locks(config: GNNTrainingConfig) -> None:
    """Raise a row-safe error if any selection artifact lock has drifted."""

    errors = crossfit_artifact_lock_errors(config)
    if errors:
        codes = sorted({str(error["code"]) for error in errors})
        raise RuntimeError("cross-fit artifact locks failed: " + ", ".join(codes))
