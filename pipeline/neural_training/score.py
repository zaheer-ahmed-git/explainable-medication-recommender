"""Score the trained Transformer, compute metrics, and decide the neural gate.

Scoring runs the calibrated best checkpoint over the evaluation split, writes a
compact ``(ranking_group_id, candidate_medication_token, score)`` prediction
table, then rematerializes the canonical ``baseline_scores`` schema by joining
those predictions back to the ranking table in DuckDB. Joining through the
ranking table preserves the native key/label types so the neural scores union
cleanly with the Stage 1 structured recovery baseline and reuse the
authoritative DuckDB metric queries (:mod:`pipeline.evaluate_baselines`).

Development mode gates the neural model against the frozen Stage 1 recovery
winner (default ``xgboost_rank_ndcg_oof_late_fusion``) on MIMIC validation and
freezes the checkpoint for final scoring; final mode scores the held-out test
split behind the frozen selection. Reports are aggregate-only.

PyTorch is imported directly; heavy evaluation dependencies are imported lazily.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from pipeline.extract_utils import parquet_scan, safe_error_message, sql_string
from pipeline.features import copy_query_to_parquet
from pipeline.neural_training.config import (
    EXPERIMENT_VERSION,
    LEGACY_MILESTONE8B_XGBOOST_NDCG_AT_10,
    SELECTION_K,
    SELECTION_SCHEMA_VERSION,
    TRAINING_SCHEMA_VERSION,
    NeuralTrainingConfig,
)
from pipeline.neural_training.contract import (
    blocked_report,
    contract_digest_or_none,
    neural_gate_decision,
    preflight_errors,
    resolve_structured_reference,
)
from pipeline.neural_training.data import (
    DEVELOPMENT_SOURCE,
    configure_connection,
    write_json,
)
from pipeline.neural_training.dataset import FeatureLayoutSpec, iter_batches
from pipeline.neural_training.train import load_model_from_checkpoint, resolve_device
from pipeline.training_contract import load_json, sha256_file

NEURAL_BASELINE_NAME = "transformer_patient_context"

PREDICTION_SCHEMA = pa.schema(
    [
        ("ranking_group_id", pa.string()),
        ("candidate_medication_token", pa.string()),
        ("score", pa.float64()),
    ]
)


def _load_temperature(config: NeuralTrainingConfig) -> float:
    """Return the fitted temperature (defaults to 1.0 when absent)."""

    if not config.calibration_path.exists():
        return 1.0
    payload = load_json(config.calibration_path)
    temperature = float(payload.get("temperature", 1.0))
    return temperature if temperature > 0 else 1.0


@torch.no_grad()
def _write_predictions(
    model: Any,
    config: NeuralTrainingConfig,
    spec: FeatureLayoutSpec,
    *,
    device: torch.device,
    split: str,
    temperature: float,
    output_path: Path,
) -> int:
    """Stream calibrated per-candidate probabilities to a compact Parquet file."""

    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with pq.ParquetWriter(output_path, PREDICTION_SCHEMA) as writer:
        for batch in iter_batches(
            config,
            spec,
            split=split,
            batch_groups=config.optimization.batch_ranking_groups,
            shuffle=False,
            seed=config.seed,
        ):
            moved = batch.to(device)
            logits = model.forward_batch(moved)
            probabilities = torch.sigmoid(logits / temperature).cpu().numpy()
            mask = batch.candidate_mask.cpu().numpy()
            group_ids: list[str] = []
            tokens: list[str] = []
            scores: list[float] = []
            for row in range(batch.num_groups):
                valid = mask[row]
                candidate_tokens = batch.candidate_tokens[row]
                for position in range(len(candidate_tokens)):
                    if not valid[position]:
                        continue
                    group_ids.append(batch.ranking_group_ids[row])
                    tokens.append(candidate_tokens[position])
                    scores.append(float(probabilities[row][position]))
            if not group_ids:
                continue
            writer.write_table(
                pa.Table.from_pydict(
                    {
                        "ranking_group_id": group_ids,
                        "candidate_medication_token": tokens,
                        "score": scores,
                    },
                    schema=PREDICTION_SCHEMA,
                )
            )
            row_count += len(group_ids)
    return row_count


def _materialize_candidate_scores(
    connection: duckdb.DuckDBPyConnection,
    config: NeuralTrainingConfig,
    *,
    split: str,
    predictions_path: Path,
    output_path: Path,
    generated_at: str,
) -> int:
    """Join predictions back to the ranking table to rebuild canonical scores."""

    query = f"""
SELECT
    pcm.source,
    pcm.split,
    pcm.ranking_group_id,
    pcm.index_condition_token,
    pcm.candidate_medication_token,
    pcm.candidate_rank,
    pcm.label_prescribed,
    {sql_string(NEURAL_BASELINE_NAME)} AS baseline_name,
    predictions.score,
    {config.seed} AS seed,
    {sql_string(EXPERIMENT_VERSION)} AS baseline_version,
    {sql_string(TRAINING_SCHEMA_VERSION)} AS evaluation_version,
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


def _combine_with_reference(
    connection: duckdb.DuckDBPyConnection,
    config: NeuralTrainingConfig,
    *,
    candidate_path: Path,
    split: str,
    reference_baseline_name: str,
) -> tuple[int, bool]:
    """Append Stage 1 structured-baseline rows for ``split`` when present."""

    config.score_output_path.parent.mkdir(parents=True, exist_ok=True)
    if not config.reference_scores_path.exists():
        row_count = copy_query_to_parquet(
            connection,
            f"SELECT * FROM {parquet_scan(candidate_path)}",
            config.score_output_path,
        )
        return row_count, False
    query = f"""
SELECT * FROM {parquet_scan(candidate_path)}
UNION ALL BY NAME
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
    seed,
    baseline_version,
    evaluation_version,
    generated_at
FROM {parquet_scan(config.reference_scores_path)}
WHERE source = {sql_string(DEVELOPMENT_SOURCE)}
    AND split = {sql_string(split)}
    AND baseline_name = {sql_string(reference_baseline_name)}
"""
    row_count = copy_query_to_parquet(connection, query, config.score_output_path)
    return row_count, True


def _metric_row(
    manifest: dict[str, Any],
    *,
    baseline_name: str,
    split: str,
    k: int = SELECTION_K,
) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in manifest.get("ranking_metrics", [])
            if row.get("baseline_name") == baseline_name
            and row.get("source") == DEVELOPMENT_SOURCE
            and row.get("split") == split
            and int(row.get("k", -1)) == k
        ),
        None,
    )


def _reference_metrics_or_anchor(
    manifest: dict[str, Any],
    *,
    split: str,
    reference_available: bool,
    reference_baseline_name: str,
    anchor_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return computed reference metrics, or the Stage 1 selection anchors."""

    if reference_available:
        reference = _metric_row(
            manifest, baseline_name=reference_baseline_name, split=split
        )
        if reference is not None:
            return reference
    if anchor_metrics is not None:
        return dict(anchor_metrics)
    # Last-resort fallback for incomplete fixtures: NDCG-only legacy note.
    return {
        "ndcg_at_k": LEGACY_MILESTONE8B_XGBOOST_NDCG_AT_10,
        "mrr_at_k": 0.0,
        "hit_rate_at_k": 0.0,
        "baseline_name": reference_baseline_name,
        "source": "legacy_milestone8b_anchor_fallback",
    }


def _frozen_artifacts(config: NeuralTrainingConfig) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in (
            ("checkpoint", config.checkpoint_path),
            ("calibration", config.calibration_path),
            ("feature_layout", config.feature_layout_path),
        )
        if path.exists()
    }


def _append_metric_summaries(
    connection: duckdb.DuckDBPyConnection,
    config: NeuralTrainingConfig,
    manifest: dict[str, Any],
) -> None:
    """Reuse the authoritative DuckDB metric queries for the neural scores."""

    from pipeline.evaluate_baselines import (
        BaselineEvaluationConfig,
        append_metric_summaries,
    )

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


def _base_report(
    config: NeuralTrainingConfig,
    *,
    status: str,
    generated_at: str,
    split: str,
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "status": status,
        "stage": "score",
        "mode": config.mode,
        "generated_at": generated_at,
        "scored_split": split,
        "baseline_name": NEURAL_BASELINE_NAME,
        "seed": config.seed,
        "selection_k": SELECTION_K,
        "artifacts": {"baseline_scores": str(config.score_output_path)},
        "clinical_claim_boundary": (
            "Offline research scoring of observed prescriptions; not a validated "
            "medication recommender."
        ),
        "label_caveat": (
            "Labels are observed historical prescriptions in the label window. "
            "Unobserved catalog candidates are weak observational negatives."
        ),
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "local_scores_contain_restricted_group_keys": True,
            "local_scores_are_ignored_and_protected": True,
        },
    }


def _selection_report(
    config: NeuralTrainingConfig,
    manifest: dict[str, Any],
    *,
    generated_at: str,
    split: str,
    reference_available: bool,
    contract_digest: str | None,
    structured_reference: dict[str, Any],
) -> dict[str, Any]:
    """Build the neural freeze/gate selection report for development mode."""

    reference_baseline_name = str(structured_reference["baseline_name"])
    candidate = _metric_row(manifest, baseline_name=NEURAL_BASELINE_NAME, split=split)
    if candidate is None:
        return {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "status": "failed_missing_validation_metrics",
            "generated_at": generated_at,
            "model_frozen": False,
            "decision": "retain_structured_recovery_baseline",
            "reference_experiment": reference_baseline_name,
            "data_safety": {
                "report_contains_patient_rows": False,
                "report_contains_row_samples": False,
            },
        }
    reference = _reference_metrics_or_anchor(
        manifest,
        split=split,
        reference_available=reference_available,
        reference_baseline_name=reference_baseline_name,
        anchor_metrics=structured_reference.get("anchor_metrics"),
    )
    decision = neural_gate_decision(candidate=candidate, reference=reference)
    if reference_available:
        reference_source = "computed_from_stage1_scores"
    elif structured_reference.get("anchor_metrics") is not None:
        reference_source = str(
            structured_reference["anchor_metrics"].get(
                "source", "stage1_gate_recovery_selection"
            )
        )
    else:
        reference_source = "legacy_milestone8b_anchor_fallback"
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "frozen",
        "generated_at": generated_at,
        "selected_experiment": NEURAL_BASELINE_NAME,
        "reference_experiment": reference_baseline_name,
        "selection_basis": {
            "model_selection": "mimic_validation_early_stopping",
            "gate_split": f"{DEVELOPMENT_SOURCE}_{split}",
            "k": SELECTION_K,
            "candidate_metrics": candidate,
            "reference_metrics": reference,
            "reference_source": reference_source,
            "legacy_milestone8b_ndcg_at_10": structured_reference.get(
                "legacy_milestone8b_ndcg_at_10"
            ),
            "note": (
                "The Transformer must clear +0.005 NDCG@10 over the Stage 1 "
                "structured recovery winner, not the older Milestone 8B "
                "xgboost_frozen_reference anchor."
            ),
        },
        **decision,
        "contract_digest": contract_digest,
        "seed": config.seed,
        "architecture": config.architecture.__dict__,
        "frozen_artifacts": _frozen_artifacts(config),
        "calibration": {
            "method": "single_temperature_validation_bce",
            "temperature": _load_temperature(config),
        },
        "gate_caveat": (
            "The neural model is selected and gated on MIMIC validation; the "
            "held-out test split is scored only in final mode after this "
            "selection is frozen."
        ),
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
        },
    }


def score_transformer(config: NeuralTrainingConfig) -> dict[str, Any]:
    """Score the trained model, compute metrics, and record the gate decision."""

    generated_at = datetime.now(UTC).isoformat()
    split = "test" if config.mode == "final" else "validation"
    errors = preflight_errors(config, stage="score")
    if errors:
        report = blocked_report(
            schema_version=TRAINING_SCHEMA_VERSION,
            stage="score",
            mode=config.mode,
            generated_at=generated_at,
            errors=errors,
        )
        write_json(config.score_report_path, report)
        return report

    if not config.checkpoint_path.exists():
        report = _base_report(
            config, status="failed", generated_at=generated_at, split=split
        )
        report["reason"] = "no trained checkpoint is available; run `train` first"
        write_json(config.score_report_path, report)
        return report

    contract_digest = contract_digest_or_none(config)
    structured_reference = resolve_structured_reference(config)
    reference_baseline_name = str(structured_reference["baseline_name"])
    spec = FeatureLayoutSpec.from_json(config.feature_layout_path)
    device = resolve_device(config)
    temperature = _load_temperature(config)
    report = _base_report(
        config, status="completed", generated_at=generated_at, split=split
    )
    report["temperature"] = temperature
    report["device"] = str(device)
    report["structured_reference_baseline"] = reference_baseline_name
    report["structured_reference_scores"] = str(config.reference_scores_path)

    try:
        model = load_model_from_checkpoint(config, spec, device=device)
        predictions_path = config.score_root / "_neural_predictions.parquet"
        candidate_path = config.score_root / "_candidate_scores.parquet"
        prediction_rows = _write_predictions(
            model,
            config,
            spec,
            device=device,
            split=split,
            temperature=temperature,
            output_path=predictions_path,
        )
        with duckdb.connect(database=":memory:") as connection:
            configure_connection(config, connection)
            _materialize_candidate_scores(
                connection,
                config,
                split=split,
                predictions_path=predictions_path,
                output_path=candidate_path,
                generated_at=generated_at,
            )
            score_rows, reference_available = _combine_with_reference(
                connection,
                config,
                candidate_path=candidate_path,
                split=split,
                reference_baseline_name=reference_baseline_name,
            )
            report["candidate_prediction_row_count"] = prediction_rows
            report["score_row_count"] = score_rows
            report["reference_appended"] = reference_available
            _append_metric_summaries(connection, config, report)

        if config.mode == "development":
            selection = _selection_report(
                config,
                report,
                generated_at=generated_at,
                split=split,
                reference_available=reference_available,
                contract_digest=contract_digest,
                structured_reference=structured_reference,
            )
            write_json(config.selection_report_path, selection)
            report["selection_report"] = str(config.selection_report_path)
            report["model_frozen"] = selection.get("model_frozen", False)
            report["neural_gate_decision"] = selection.get("decision")
        else:
            report["frozen_selection_reference"] = str(config.selection_report_path)
    except Exception as error:  # noqa: BLE001 - surfaced as aggregate status
        report["status"] = "failed"
        report["reason"] = safe_error_message(error)

    write_json(config.score_report_path, report)
    return report
