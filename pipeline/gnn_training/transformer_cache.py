"""Extract immutable Transformer representations into bounded GNN shards."""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.extract_utils import parquet_scan, sql_string
from pipeline.gnn_training.config import (
    FULL_TRAIN_REFIT_SCOPE,
    GNNTrainingConfig,
)
from pipeline.gnn_training.data import (
    FROZEN_TRANSFORMER_CACHE_ARTIFACT_LOCK_VERSION,
    FROZEN_TRANSFORMER_CACHE_SCHEMA_VERSION,
    TRANSFORMER_CONTEXT_TABLE,
    TRANSFORMER_LOGIT_TABLE,
    _frozen_transformer_hashes,
    _promote_paths,
    _remove_owned_path,
    _upstream_manifest_hashes,
    artifact_tree_digest,
    configure_connection,
    write_json,
)
from pipeline.gnn_training.frozen_transformer import (
    assert_artifact_hashes_unchanged,
    extract_frozen_outputs,
    load_frozen_transformer,
)
from pipeline.neural_training.config import NeuralTrainingConfig
from pipeline.neural_training.dataset import iter_batches
from pipeline.training_contract import sha256_file


def _cache_artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(
            candidate for candidate in root.rglob("*.parquet") if candidate.is_file()
        )
    }


def _cached_splits(root: Path) -> list[str]:
    splits: set[str] = set()
    for path in root.rglob("*.parquet"):
        for part in path.relative_to(root).parts:
            if part.startswith("split="):
                splits.add(part.removeprefix("split="))
    return sorted(splits)


def _neural_shard_count(config: GNNTrainingConfig) -> int:
    """Infer the immutable neural cache shard count from physical filenames."""

    observed: set[int] = set()
    for split in config.evaluation_splits():
        directory = config.neural_root / "cache" / "groups" / split
        for path in directory.glob("shard_*.parquet"):
            try:
                observed.add(int(path.stem.removeprefix("shard_")))
            except ValueError:
                continue
    if not observed:
        raise FileNotFoundError(
            "frozen Transformer extraction requires prepared neural cache shards"
        )
    expected = set(range(max(observed) + 1))
    if observed != expected:
        raise ValueError("neural cache shard indexes are not contiguous from zero")
    return len(expected)


def _neural_config(
    config: GNNTrainingConfig,
    *,
    shard_count: int,
) -> NeuralTrainingConfig:
    defaults = NeuralTrainingConfig()
    return replace(
        defaults,
        features_root=config.features_root,
        training_root=config.training_root,
        graph_root=config.graph_root,
        neural_root=config.neural_root,
        mode=config.mode,
        seed=config.seed,
        shard_count=shard_count,
        device=config.device,
        duckdb_temp_directory=config.duckdb_temp_directory,
        duckdb_memory_limit=config.duckdb_memory_limit,
        duckdb_threads=config.duckdb_threads,
    )


def _context_schema(context_dim: int) -> pa.Schema:
    return pa.schema(
        [
            ("source", pa.string()),
            ("split", pa.string()),
            ("ranking_group_id", pa.string()),
            (
                "transformer_context",
                pa.list_(pa.float32(), list_size=context_dim),
            ),
        ]
    )


LOGIT_SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("split", pa.string()),
        ("ranking_group_id", pa.string()),
        ("index_condition_token", pa.string()),
        ("candidate_medication_token", pa.string()),
        ("candidate_rank", pa.int64()),
        ("frozen_transformer_logit", pa.float64()),
    ]
)


def _write_raw_outputs(
    config: GNNTrainingConfig,
    *,
    stage_root: Path,
) -> tuple[Path, Path, dict[str, str], int]:
    """Stream frozen forward outputs without retaining patient batches."""

    import torch

    shard_count = _neural_shard_count(config)
    neural_config = _neural_config(config, shard_count=shard_count)
    device = torch.device(
        config.device
        if config.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    bundle = load_frozen_transformer(
        checkpoint_path=config.neural_checkpoint_path,
        feature_layout_path=config.neural_feature_layout_path,
        calibration_path=config.neural_calibration_path,
        device=device,
        expected_context_dim=config.architecture.transformer_context_dim,
    )
    raw_root = stage_root / "_raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    contexts_path = raw_root / "contexts.parquet"
    logits_path = raw_root / "candidate_logits.parquet"
    context_schema = _context_schema(config.architecture.transformer_context_dim)
    context_writer = pq.ParquetWriter(contexts_path, context_schema)
    logit_writer = pq.ParquetWriter(logits_path, LOGIT_SCHEMA)
    try:
        for split in config.evaluation_splits():
            for batch in iter_batches(
                neural_config,
                bundle.feature_layout,
                split=split,
                batch_groups=config.optimization.batch_ranking_groups,
                shuffle=False,
                seed=config.seed,
            ):
                moved = batch.to(device)
                outputs = extract_frozen_outputs(bundle, moved)
                contexts = outputs.context.cpu().to(torch.float32).tolist()
                context_writer.write_table(
                    pa.Table.from_pydict(
                        {
                            "source": list(batch.sources),
                            "split": list(batch.splits),
                            "ranking_group_id": list(batch.ranking_group_ids),
                            "transformer_context": contexts,
                        },
                        schema=context_schema,
                    )
                )
                mask = batch.candidate_mask.cpu().numpy()
                logits = outputs.candidate_logits.cpu().numpy()
                payload: dict[str, list[Any]] = {
                    name: [] for name in LOGIT_SCHEMA.names
                }
                for row in range(batch.num_groups):
                    for position, token in enumerate(batch.candidate_tokens[row]):
                        if not bool(mask[row, position]):
                            continue
                        payload["source"].append(batch.sources[row])
                        payload["split"].append(batch.splits[row])
                        payload["ranking_group_id"].append(batch.ranking_group_ids[row])
                        payload["index_condition_token"].append(
                            batch.index_condition_tokens[row]
                        )
                        payload["candidate_medication_token"].append(token)
                        payload["candidate_rank"].append(
                            int(batch.candidate_rank[row, position])
                        )
                        payload["frozen_transformer_logit"].append(
                            float(logits[row, position])
                        )
                if payload["ranking_group_id"]:
                    logit_writer.write_table(
                        pa.Table.from_pydict(payload, schema=LOGIT_SCHEMA)
                    )
    finally:
        context_writer.close()
        logit_writer.close()
    assert_artifact_hashes_unchanged(
        bundle.artifact_hashes,
        checkpoint_path=config.neural_checkpoint_path,
        feature_layout_path=config.neural_feature_layout_path,
        calibration_path=config.neural_calibration_path,
    )
    return contexts_path, logits_path, bundle.artifact_hashes, shard_count


def _validate_exact_keys(
    connection: duckdb.DuckDBPyConnection,
    config: GNNTrainingConfig,
    *,
    contexts_path: Path,
    logits_path: Path,
) -> tuple[int, int]:
    """Prove exact graph-group/candidate coverage without returning keys."""

    groups = parquet_scan(config.shards_root / "groups" / "**" / "*.parquet")
    candidates = parquet_scan(config.shards_root / "candidates" / "**" / "*.parquet")
    contexts = parquet_scan(contexts_path)
    logits = parquet_scan(logits_path)
    row = connection.execute(
        f"""
WITH graph_groups AS (
    SELECT source, split, ranking_group_id, patient_fold_id, shard_id
    FROM {groups}
),
context_groups AS (
    SELECT source, split, ranking_group_id, COUNT(*) AS row_count
    FROM {contexts}
    GROUP BY source, split, ranking_group_id
),
graph_candidates AS (
    SELECT
        source,
        split,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        candidate_rank
    FROM {candidates}
),
transformer_candidates AS (
    SELECT
        source,
        split,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        candidate_rank
    FROM {logits}
),
graph_group_only AS (
    SELECT source, split, ranking_group_id FROM graph_groups
    EXCEPT ALL
    SELECT source, split, ranking_group_id FROM context_groups
),
context_group_only AS (
    SELECT source, split, ranking_group_id FROM context_groups
    EXCEPT ALL
    SELECT source, split, ranking_group_id FROM graph_groups
),
graph_candidate_only AS (
    SELECT * FROM graph_candidates
    EXCEPT ALL
    SELECT * FROM transformer_candidates
),
transformer_candidate_only AS (
    SELECT * FROM transformer_candidates
    EXCEPT ALL
    SELECT * FROM graph_candidates
)
SELECT
    (SELECT COUNT(*) FROM graph_groups) AS group_count,
    (SELECT COUNT(*) FROM graph_candidates) AS candidate_count,
    (SELECT COUNT(*) FROM graph_group_only)
        + (SELECT COUNT(*) FROM context_group_only)
        + (SELECT COUNT(*) FROM graph_candidate_only)
        + (SELECT COUNT(*) FROM transformer_candidate_only)
        + (
            SELECT COUNT(*) FROM context_groups WHERE row_count <> 1
        ) AS mismatch_count
"""
    ).fetchone()
    if row is None or int(row[2]) != 0:
        raise ValueError(
            "frozen Transformer outputs do not exactly match graph groups/candidates"
        )
    return int(row[0]), int(row[1])


def _copy_partitioned(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    path: Path,
) -> int:
    from pipeline.gnn_training.data import coerce_single_parquet_partitions

    path.mkdir(parents=True, exist_ok=True)
    row = connection.execute(
        f"""
COPY ({query})
TO {sql_string(path)}
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
    coerce_single_parquet_partitions(path)
    return int(row[0]) if row and row[0] is not None else 0


def prepare_frozen_transformer_cache(
    config: GNNTrainingConfig,
) -> dict[str, Any]:
    """Extract, reconcile, partition, hash-lock, and atomically promote cache."""

    generated_at = datetime.now(UTC).isoformat()
    stage_root = config.gnn_root / f".frozen-transformer-stage-{uuid.uuid4().hex}"
    stage_root.mkdir(parents=True, exist_ok=False)
    try:
        (
            contexts_path,
            logits_path,
            artifact_hashes,
            neural_shard_count,
        ) = _write_raw_outputs(config, stage_root=stage_root)
        output_root = stage_root / "frozen_transformer"
        with duckdb.connect(database=":memory:") as connection:
            configure_connection(config, connection)
            group_count, candidate_count = _validate_exact_keys(
                connection,
                config,
                contexts_path=contexts_path,
                logits_path=logits_path,
            )
            groups = parquet_scan(config.shards_root / "groups" / "**" / "*.parquet")
            context_rows = _copy_partitioned(
                connection,
                f"""
SELECT
    contexts.source,
    contexts.split,
    contexts.ranking_group_id,
    groups.patient_fold_id,
    contexts.transformer_context,
    groups.shard_id
FROM {parquet_scan(contexts_path)} AS contexts
INNER JOIN {groups} AS groups
    USING (source, split, ranking_group_id)
""",
                output_root / TRANSFORMER_CONTEXT_TABLE,
            )
            logit_rows = _copy_partitioned(
                connection,
                f"""
SELECT
    logits.source,
    logits.split,
    logits.ranking_group_id,
    logits.index_condition_token,
    logits.candidate_medication_token,
    logits.candidate_rank,
    logits.frozen_transformer_logit,
    groups.shard_id
FROM {parquet_scan(logits_path)} AS logits
INNER JOIN {groups} AS groups
    USING (source, split, ranking_group_id)
""",
                output_root / TRANSFORMER_LOGIT_TABLE,
            )
        if context_rows != group_count or logit_rows != candidate_count:
            raise ValueError("partitioned Transformer cache row counts drifted")
        assert_artifact_hashes_unchanged(
            artifact_hashes,
            checkpoint_path=config.neural_checkpoint_path,
            feature_layout_path=config.neural_feature_layout_path,
            calibration_path=config.neural_calibration_path,
        )
        cache_artifact_hashes = _cache_artifact_hashes(output_root)
        manifest = {
            "schema_version": FROZEN_TRANSFORMER_CACHE_SCHEMA_VERSION,
            "artifact_lock_version": (FROZEN_TRANSFORMER_CACHE_ARTIFACT_LOCK_VERSION),
            "status": "completed",
            "scope": FULL_TRAIN_REFIT_SCOPE,
            "selection_eligible": False,
            "generated_at": generated_at,
            "shard_count": config.shard_count,
            "neural_source_shard_count": neural_shard_count,
            "transformer_context_dim": config.architecture.transformer_context_dim,
            "table_row_counts": {
                TRANSFORMER_CONTEXT_TABLE: context_rows,
                TRANSFORMER_LOGIT_TABLE: logit_rows,
            },
            "cached_splits": _cached_splits(output_root),
            "artifact_hashes": cache_artifact_hashes,
            "artifact_tree_digest": artifact_tree_digest(cache_artifact_hashes),
            "frozen_transformer_hashes": _frozen_transformer_hashes(config),
            "upstream_provenance": _upstream_manifest_hashes(config),
            "extraction_policy": {
                "model_eval": True,
                "requires_grad": False,
                "torch_no_grad": True,
                "outputs_detached": True,
                "neural_artifact_writes": False,
                "exact_graph_candidate_reconciliation": True,
            },
            "data_safety": {
                "manifest_contains_patient_rows": False,
                "manifest_contains_row_samples": False,
                "manifest_contains_identifier_values": False,
                "local_cache_contains_restricted_group_keys": True,
                "direct_patient_identifiers_present": False,
            },
        }
        write_json(output_root / "cache_manifest.json", manifest)
        _promote_paths(
            config,
            replacements=((output_root, config.frozen_transformer_cache_root),),
        )
        return manifest
    finally:
        if stage_root.exists():
            _remove_owned_path(config, stage_root)
