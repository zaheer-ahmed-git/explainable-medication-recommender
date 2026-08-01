"""Canonical score materialization and aggregate metric helpers.

Compact model predictions contain restricted local join keys and remain below
the ignored GNN artifact root.  Canonical score tables are rebuilt by joining
those predictions to the locked ranking table so key and label types exactly
match the repository's authoritative DuckDB evaluation schema.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from pipeline.evaluate_baselines import (
    BaselineEvaluationConfig,
    append_metric_summaries,
)
from pipeline.extract_utils import parquet_scan, sql_string
from pipeline.features import copy_query_to_parquet
from pipeline.gnn_training.config import (
    SELECTION_K,
    GNNTrainingConfig,
)
from pipeline.gnn_training.data import DEVELOPMENT_SOURCE, configure_connection
from pipeline.gnn_training.dataset import GNNBatch

COMPACT_PREDICTION_SCHEMA = pa.schema(
    [
        ("ranking_group_id", pa.string()),
        ("candidate_medication_token", pa.string()),
        ("score", pa.float64()),
    ]
)

OOF_PREDICTION_SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("split", pa.string()),
        ("ranking_group_id", pa.string()),
        ("index_condition_token", pa.string()),
        ("candidate_medication_token", pa.string()),
        ("candidate_rank", pa.int64()),
        ("label_prescribed", pa.bool_()),
        ("patient_fold_id", pa.int32()),
        ("shard_id", pa.int32()),
        ("gnn_logit", pa.float64()),
    ]
)


class AtomicParquetWriter:
    """Streaming writer that promotes only a successfully closed Parquet file."""

    def __init__(self, path: Path, schema: pa.Schema):
        self.path = Path(path)
        self.schema = schema
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_name(
            f".{self.path.name}.tmp-{uuid.uuid4().hex}"
        )
        self.writer = pq.ParquetWriter(self.temporary, schema)
        self.row_count = 0
        self._committed = False

    def write(self, payload: dict[str, list[Any]]) -> None:
        count = len(next(iter(payload.values()), []))
        if count == 0:
            return
        if any(len(values) != count for values in payload.values()):
            raise ValueError("Parquet payload columns have inconsistent lengths")
        self.writer.write_table(pa.Table.from_pydict(payload, schema=self.schema))
        self.row_count += count

    def commit(self) -> int:
        if self._committed:
            return self.row_count
        self.writer.close()
        self.temporary.replace(self.path)
        self._committed = True
        return self.row_count

    def abort(self) -> None:
        try:
            self.writer.close()
        finally:
            if self.temporary.exists():
                self.temporary.unlink()

    def __enter__(self) -> AtomicParquetWriter:
        return self

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> None:
        del error, traceback
        if error_type is None and not self._committed:
            self.commit()
        elif error_type is not None:
            self.abort()


def write_compact_batch(
    writer: AtomicParquetWriter,
    batch: GNNBatch,
    scores: torch.Tensor,
) -> None:
    """Append valid candidate probabilities for a complete GNN batch."""

    if scores.shape != batch.candidate_mask.shape:
        raise ValueError("prediction scores must match the candidate mask")
    mask = batch.candidate_mask.cpu().numpy()
    values = scores.detach().cpu().numpy()
    group_ids: list[str] = []
    tokens: list[str] = []
    output_scores: list[float] = []
    for row in range(batch.num_groups):
        for position, token in enumerate(batch.candidate_tokens[row]):
            if not bool(mask[row, position]):
                continue
            group_ids.append(batch.ranking_group_ids[row])
            tokens.append(token)
            output_scores.append(float(values[row, position]))
    writer.write(
        {
            "ranking_group_id": group_ids,
            "candidate_medication_token": tokens,
            "score": output_scores,
        }
    )


def write_oof_batch(
    writer: AtomicParquetWriter,
    batch: GNNBatch,
    logits: torch.Tensor,
) -> None:
    """Append raw selected-variant OOF logits with restricted local keys."""

    if logits.shape != batch.candidate_mask.shape:
        raise ValueError("OOF logits must match the candidate mask")
    values = logits.detach().cpu().numpy()
    labels = batch.labels.cpu().numpy()
    ranks = batch.candidate_rank.cpu().numpy()
    payload: dict[str, list[Any]] = {name: [] for name in OOF_PREDICTION_SCHEMA.names}
    for row in range(batch.num_groups):
        for position, token in enumerate(batch.candidate_tokens[row]):
            payload["source"].append(batch.sources[row])
            payload["split"].append(batch.splits[row])
            payload["ranking_group_id"].append(batch.ranking_group_ids[row])
            payload["index_condition_token"].append(batch.index_condition_tokens[row])
            payload["candidate_medication_token"].append(token)
            payload["candidate_rank"].append(int(ranks[row, position]))
            payload["label_prescribed"].append(bool(labels[row, position] > 0.5))
            payload["patient_fold_id"].append(int(batch.patient_fold_ids[row]))
            payload["shard_id"].append(int(batch.shard_ids[row]))
            payload["gnn_logit"].append(float(values[row, position]))
    writer.write(payload)


def materialize_canonical_scores(
    connection: duckdb.DuckDBPyConnection,
    config: GNNTrainingConfig,
    *,
    predictions_path: Path,
    output_path: Path,
    split: str,
    baseline_name: str,
    baseline_version: str,
    evaluation_version: str,
    generated_at: str,
) -> int:
    """Rebuild the exact 13-column canonical score schema."""

    mismatch_row = connection.execute(
        f"""
WITH locked_candidates AS (
    SELECT
        CAST(ranking_group_id AS VARCHAR) AS ranking_group_id,
        CAST(candidate_medication_token AS VARCHAR) AS candidate_medication_token
    FROM {parquet_scan(config.patient_condition_medication_path)}
    WHERE source = {sql_string(DEVELOPMENT_SOURCE)}
        AND split = {sql_string(split)}
),
predictions AS (
    SELECT ranking_group_id, candidate_medication_token
    FROM {parquet_scan(predictions_path)}
),
locked_only AS (
    SELECT * FROM locked_candidates
    EXCEPT ALL
    SELECT * FROM predictions
),
prediction_only AS (
    SELECT * FROM predictions
    EXCEPT ALL
    SELECT * FROM locked_candidates
)
SELECT
    (SELECT COUNT(*) FROM locked_only)
        + (SELECT COUNT(*) FROM prediction_only) AS key_mismatch_count,
    (
        SELECT COUNT(*)
        FROM {parquet_scan(predictions_path)}
        WHERE ranking_group_id IS NULL
            OR candidate_medication_token IS NULL
            OR score IS NULL
            OR NOT isfinite(score)
    ) AS invalid_prediction_count,
    (SELECT COUNT(*) FROM locked_candidates) AS locked_candidate_count,
    (SELECT COUNT(*) FROM predictions) AS prediction_count
"""
    ).fetchone()
    if (
        mismatch_row is None
        or int(mismatch_row[0]) != 0
        or int(mismatch_row[1]) != 0
        or int(mismatch_row[2]) <= 0
        or int(mismatch_row[3]) <= 0
    ):
        raise ValueError(
            "model predictions do not exactly match the locked candidate set"
        )

    query = f"""
SELECT
    pcm.source,
    pcm.split,
    pcm.ranking_group_id,
    pcm.index_condition_token,
    pcm.candidate_medication_token,
    pcm.candidate_rank,
    pcm.label_prescribed,
    {sql_string(baseline_name)} AS baseline_name,
    predictions.score,
    {int(config.seed)} AS seed,
    {sql_string(baseline_version)} AS baseline_version,
    {sql_string(evaluation_version)} AS evaluation_version,
    {sql_string(generated_at)} AS generated_at
FROM {parquet_scan(config.patient_condition_medication_path)} AS pcm
INNER JOIN {parquet_scan(predictions_path)} AS predictions
    ON CAST(pcm.ranking_group_id AS VARCHAR) = predictions.ranking_group_id
    AND CAST(pcm.candidate_medication_token AS VARCHAR)
        = predictions.candidate_medication_token
WHERE pcm.source = {sql_string(DEVELOPMENT_SOURCE)}
    AND pcm.split = {sql_string(split)}
"""
    return copy_query_to_parquet(connection, query, output_path)


def combine_score_tables(
    connection: duckdb.DuckDBPyConnection,
    paths: Iterable[Path],
    *,
    output_path: Path,
) -> int:
    """Union canonical score files by name."""

    existing = [Path(path) for path in paths if Path(path).is_file()]
    if not existing:
        raise ValueError("no canonical score tables are available to combine")
    query = "\nUNION ALL BY NAME\n".join(
        f"SELECT * FROM {parquet_scan(path)}" for path in existing
    )
    return copy_query_to_parquet(connection, query, output_path)


def append_authoritative_metrics(
    connection: duckdb.DuckDBPyConnection,
    config: GNNTrainingConfig,
    report: dict[str, Any],
    *,
    evaluation_root: Path,
) -> None:
    """Populate the standard row/ranking/condition aggregate metric sections."""

    metric_config = BaselineEvaluationConfig(
        features_root=config.features_root,
        training_root=config.training_root,
        evaluation_root=evaluation_root,
        top_k=config.top_k,
        mode=config.mode,
        frozen_selection=config.frozen_selection,
        seed=config.seed,
        feature_version="temporal-features-v2",
        duckdb_temp_directory=config.duckdb_temp_directory,
        duckdb_memory_limit=config.duckdb_memory_limit,
        duckdb_threads=config.duckdb_threads,
    )
    append_metric_summaries(connection, metric_config, report)


def metric_row(
    report: dict[str, Any],
    *,
    baseline_name: str,
    split: str,
    k: int = SELECTION_K,
) -> dict[str, Any] | None:
    """Return one canonical ranking metric row."""

    for row in report.get("ranking_metrics", []):
        try:
            matches = (
                row.get("baseline_name") == baseline_name
                and row.get("source") == DEVELOPMENT_SOURCE
                and row.get("split") == split
                and int(row.get("k", -1)) == int(k)
            )
        except (TypeError, ValueError):
            matches = False
        if matches:
            return dict(row)
    return None


def qualification_decision(
    *,
    candidate: dict[str, Any],
    reference: dict[str, Any],
    minimum_ndcg_lift: float,
    maximum_secondary_drop: float,
) -> dict[str, Any]:
    """Apply the pre-registered NDCG lift and MRR/Hit non-inferiority gate."""

    candidate_ndcg = float(candidate["ndcg_at_k"])
    reference_ndcg = float(reference["ndcg_at_k"])
    ndcg_delta = candidate_ndcg - reference_ndcg
    mrr_delta = float(candidate["mrr_at_k"]) - float(reference["mrr_at_k"])
    hit_delta = float(candidate["hit_rate_at_k"]) - float(reference["hit_rate_at_k"])
    qualified = (
        ndcg_delta >= minimum_ndcg_lift
        and mrr_delta >= -maximum_secondary_drop
        and hit_delta >= -maximum_secondary_drop
    )
    return {
        "qualified": qualified,
        "required_ndcg_at_10": reference_ndcg + minimum_ndcg_lift,
        "ndcg_at_10_delta": ndcg_delta,
        "mrr_at_10_delta": mrr_delta,
        "hit_rate_at_10_delta": hit_delta,
        "minimum_ndcg_lift": minimum_ndcg_lift,
        "maximum_secondary_drop": maximum_secondary_drop,
    }


def base_score_report(
    config: GNNTrainingConfig,
    *,
    schema_version: str,
    stage: str,
    split: str,
    baseline_name: str,
    output_path: Path,
) -> dict[str, Any]:
    """Return the shared aggregate-only score report shell."""

    return {
        "schema_version": schema_version,
        "status": "completed",
        "stage": stage,
        "mode": config.mode,
        "generated_at": datetime.now(UTC).isoformat(),
        "scored_split": split,
        "baseline_name": baseline_name,
        "seed": config.seed,
        "selection_k": SELECTION_K,
        "artifacts": {"baseline_scores": str(output_path)},
        "clinical_claim_boundary": (
            "Offline research ranking of observed historical prescriptions; "
            "not validated clinical advice."
        ),
        "label_caveat": (
            "Observed prescriptions are historical positives; unobserved "
            "candidates are weak observational negatives."
        ),
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "report_contains_identifier_values": False,
            "local_scores_contain_restricted_group_keys": True,
            "local_scores_are_ignored_and_protected": True,
        },
    }


def configured_duckdb(config: GNNTrainingConfig) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with repository bounds."""

    connection = duckdb.connect(database=":memory:")
    configure_connection(config, connection)
    return connection
