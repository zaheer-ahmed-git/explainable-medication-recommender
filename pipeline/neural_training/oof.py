"""Fold-isolated Transformer OOF training for paired late fusion.

Each task rebuilds every train-derived Transformer input from four patient
folds, uses the GNN graph fitted without the held-out fold, trains for a fixed
pre-registered epoch count, and writes raw logits for the fifth fold.  The
held-out fold is never consulted for early stopping or calibration.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb
import pyarrow as pa
import torch

from pipeline.extract_utils import parquet_scan, safe_error_message
from pipeline.features import copy_query_to_parquet
from pipeline.gate_recovery import patient_fold_sql
from pipeline.gnn_training.scoring import AtomicParquetWriter
from pipeline.late_fusion_protocol import (
    PAIRED_OOF_PROTOCOL_VERSION,
    PAIRED_OOF_SCHEMA_VERSION,
    PRODUCTION_FOLD_COUNT,
    transformer_fold_root,
    transformer_oof_predictions_path,
)
from pipeline.neural_training.config import (
    EXPERIMENT_VERSION,
    TRAINING_SCHEMA_VERSION,
    NeuralTrainingConfig,
)
from pipeline.neural_training.contract import preflight_errors
from pipeline.neural_training.data import configure_connection, prepare_neural_caches
from pipeline.neural_training.dataset import FeatureLayoutSpec, iter_batches
from pipeline.neural_training.model import build_model
from pipeline.neural_training.train import (
    ModelEMA,
    _train_one_epoch,
    _use_amp,
    build_warmup_cosine_scheduler,
    count_train_batches,
    evaluate_split,
    resolve_device,
    save_checkpoint,
    set_global_seed,
)
from pipeline.training_contract import load_json, sha256_file, write_json

TRANSFORMER_OOF_SCHEMA = pa.schema(
    [
        ("source", pa.string()),
        ("split", pa.string()),
        ("ranking_group_id", pa.string()),
        ("index_condition_token", pa.string()),
        ("candidate_medication_token", pa.string()),
        ("candidate_rank", pa.int64()),
        ("label_prescribed", pa.bool_()),
        ("patient_fold_id", pa.int32()),
        ("transformer_logit", pa.float64()),
    ]
)


def _is_protected(path: Path) -> bool:
    try:
        return "protected" in path.expanduser().resolve().parts
    except (OSError, RuntimeError):
        return False


def _resolved_fixed_epochs(config: NeuralTrainingConfig, value: int | None) -> int:
    if value is not None:
        if value <= 0:
            raise ValueError("fixed-epochs must be positive")
        return int(value)
    if not config.training_state_path.is_file():
        raise ValueError(
            "fixed epochs were not supplied and the frozen Transformer training "
            "state is missing"
        )
    state = load_json(config.training_state_path)
    best_epoch = state.get("best_epoch")
    if not isinstance(best_epoch, int) or best_epoch < 0:
        raise ValueError("frozen Transformer training state has no valid best_epoch")
    return best_epoch + 1


def _fold_config(
    base: NeuralTrainingConfig,
    *,
    gnn_root: Path,
    held_out_fold: int,
    fixed_epochs: int,
) -> NeuralTrainingConfig:
    root = transformer_fold_root(base.neural_root, held_out_fold)
    report_root = base.prepare_manifest_path.parent
    prefix = f"phase8_p0_transformer_paired_oof_fold_{held_out_fold:02d}"
    return replace(
        base,
        features_root=root / "inputs" / "features",
        training_root=root / "inputs" / "training",
        graph_root=Path(gnn_root) / "crossfit" / f"fold_{held_out_fold:02d}",
        neural_root=root / "model",
        mode="development",
        frozen_selection=False,
        optimization=replace(base.optimization, max_epochs=fixed_epochs),
        prepare_manifest_path=report_root / f"{prefix}_prepare.json",
        training_report_path=report_root / f"{prefix}_training.json",
        score_report_path=report_root / f"{prefix}_score.json",
        selection_report_path=report_root / f"{prefix}_selection.json",
    )


def _materialize_fold_inputs(
    base: NeuralTrainingConfig,
    fold: NeuralTrainingConfig,
    *,
    held_out_fold: int,
    fold_count: int,
) -> dict[str, int]:
    """Write train-only, relabelled inputs for one held-out patient fold."""

    fold_sql = patient_fold_sql(
        seed=base.seed,
        fold_count=fold_count,
        alias="pcm",
    )
    pcm_scan = parquet_scan(base.patient_condition_medication_path)
    psf_scan = parquet_scan(base.patient_stay_features_path)
    event_scan = parquet_scan(base.event_sequences_path)
    catalog_scan = parquet_scan(base.candidate_catalog_path)
    fold_case = (
        f"CASE WHEN patient_fold_id = {held_out_fold} "
        "THEN 'validation' ELSE 'train' END"
    )
    with duckdb.connect(database=":memory:") as connection:
        configure_connection(fold, connection)
        ambiguous = connection.execute(
            f"""
SELECT COUNT(*)
FROM (
    SELECT source, stay_uid, COUNT(DISTINCT patient_uid) AS patient_count
    FROM {pcm_scan}
    WHERE source = 'mimiciv' AND split = 'train'
    GROUP BY source, stay_uid
    HAVING patient_count <> 1
) AS ambiguous_stays
"""
        ).fetchone()
        if ambiguous is None or int(ambiguous[0]) != 0:
            raise ValueError("train stay-to-patient mapping is not one-to-one")

        patient_folds = f"""
SELECT DISTINCT
    pcm.source,
    pcm.patient_uid,
    pcm.stay_uid,
    {fold_sql} AS patient_fold_id
FROM {pcm_scan} AS pcm
WHERE pcm.source = 'mimiciv' AND pcm.split = 'train'
"""
        paths_and_queries = (
            (
                fold.patient_stay_features_path,
                f"""
WITH patient_folds AS ({patient_folds})
SELECT psf.* REPLACE ({fold_case} AS split)
FROM {psf_scan} AS psf
INNER JOIN patient_folds AS folds
    ON psf.source = folds.source
    AND psf.patient_uid = folds.patient_uid
    AND psf.stay_uid = folds.stay_uid
WHERE psf.source = 'mimiciv' AND psf.split = 'train'
""",
            ),
            (
                fold.event_sequences_path,
                f"""
WITH patient_folds AS ({patient_folds})
SELECT events.* REPLACE ({fold_case} AS split)
FROM {event_scan} AS events
INNER JOIN patient_folds AS folds
    ON events.source = folds.source
    AND events.stay_uid = folds.stay_uid
WHERE events.source = 'mimiciv' AND events.split = 'train'
""",
            ),
            (
                fold.patient_condition_medication_path,
                f"""
SELECT pcm.* REPLACE (
    CASE WHEN {fold_sql} = {held_out_fold}
        THEN 'validation' ELSE 'train' END AS split
)
FROM {pcm_scan} AS pcm
WHERE pcm.source = 'mimiciv' AND pcm.split = 'train'
""",
            ),
            (
                fold.candidate_catalog_path,
                f"""
WITH fit_pairs AS (
    SELECT DISTINCT index_condition_token, candidate_medication_token
    FROM {pcm_scan} AS pcm
    WHERE pcm.source = 'mimiciv'
        AND pcm.split = 'train'
        AND {fold_sql} <> {held_out_fold}
)
SELECT catalog.*
FROM {catalog_scan} AS catalog
INNER JOIN fit_pairs
    USING (index_condition_token, candidate_medication_token)
""",
            ),
        )
        counts = {
            path.name: copy_query_to_parquet(connection, query, path)
            for path, query in paths_and_queries
        }
        coverage = connection.execute(
            f"""
SELECT split, COUNT(DISTINCT patient_uid)
FROM {parquet_scan(fold.patient_condition_medication_path)}
GROUP BY split
"""
        ).fetchall()
    patient_counts = {str(split): int(count) for split, count in coverage}
    if patient_counts.get("train", 0) <= 0 or patient_counts.get("validation", 0) <= 0:
        raise ValueError("patient fold produced an empty fit or held-out partition")
    counts["fit_patient_count"] = patient_counts["train"]
    counts["held_out_patient_count"] = patient_counts["validation"]
    return counts


def _write_oof_predictions(
    model: torch.nn.Module,
    config: NeuralTrainingConfig,
    spec: FeatureLayoutSpec,
    *,
    device: torch.device,
    held_out_fold: int,
    output_path: Path,
) -> int:
    model.eval()
    with AtomicParquetWriter(output_path, TRANSFORMER_OOF_SCHEMA) as writer:
        with torch.no_grad():
            for batch in iter_batches(
                config,
                spec,
                split="validation",
                batch_groups=config.optimization.batch_ranking_groups,
                shuffle=False,
                seed=config.seed,
            ):
                logits = model.forward_batch(batch.to(device)).detach().cpu().numpy()
                labels = batch.labels.cpu().numpy()
                payload: dict[str, list[Any]] = {
                    name: [] for name in TRANSFORMER_OOF_SCHEMA.names
                }
                for row in range(batch.num_groups):
                    for position, token in enumerate(batch.candidate_tokens[row]):
                        payload["source"].append(batch.sources[row])
                        # Restore the original scope for exact pairing with GNN OOF.
                        payload["split"].append("train")
                        payload["ranking_group_id"].append(batch.ranking_group_ids[row])
                        payload["index_condition_token"].append(
                            batch.index_condition_tokens[row]
                        )
                        payload["candidate_medication_token"].append(token)
                        payload["candidate_rank"].append(
                            int(batch.candidate_ranks[row][position])
                        )
                        payload["label_prescribed"].append(
                            bool(labels[row, position] > 0.5)
                        )
                        payload["patient_fold_id"].append(held_out_fold)
                        payload["transformer_logit"].append(
                            float(logits[row, position])
                        )
                writer.write(payload)
        return writer.commit()


def _fit_fixed_epoch_model(
    config: NeuralTrainingConfig,
    *,
    fixed_epochs: int,
    held_out_fold: int,
) -> tuple[torch.nn.Module, FeatureLayoutSpec, dict[str, Any]]:
    errors = preflight_errors(config, stage="train")
    if errors:
        raise ValueError(f"fold training preflight failed with {len(errors)} error(s)")
    set_global_seed(config.seed)
    device = resolve_device(config)
    spec = FeatureLayoutSpec.from_json(config.feature_layout_path)
    model = build_model(spec, config.architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    steps_per_epoch = count_train_batches(config, spec)
    if steps_per_epoch <= 0:
        raise ValueError("fold fit partition produced no Transformer batches")
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        steps_per_epoch=steps_per_epoch,
        max_epochs=fixed_epochs,
        warmup_epochs=config.optimization.warmup_epochs,
        min_lr_ratio=config.optimization.min_lr_ratio,
    )
    amp_enabled = _use_amp(config, device)
    scaler = torch.cuda.amp.GradScaler() if amp_enabled else None
    ema = (
        ModelEMA(model, decay=config.optimization.ema_decay)
        if config.optimization.ema_decay > 0.0
        else None
    )
    history: list[dict[str, Any]] = []
    for epoch in range(fixed_epochs):
        metrics = _train_one_epoch(
            model,
            config,
            spec,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
            epoch=epoch,
        )
        history.append({"epoch": epoch, **metrics})
    if ema is not None:
        ema.copy_to(model)
    # This single post-fit evaluation is evidence, never a selection input for
    # epochs or hyperparameters inside the fold job.
    held_out = evaluate_split(
        model,
        config,
        spec,
        device=device,
        split="validation",
    )
    save_checkpoint(
        model,
        config,
        spec,
        path=config.checkpoint_path,
        best_epoch=fixed_epochs - 1,
        best_validation=held_out,
        metadata={
            "protocol_version": PAIRED_OOF_PROTOCOL_VERSION,
            "selection_policy": "fixed_epoch_no_heldout_selection",
            "held_out_fold_index": held_out_fold,
        },
    )
    return (
        model,
        spec,
        {
            "device": str(device),
            "mixed_precision": amp_enabled,
            "steps_per_epoch": steps_per_epoch,
            "epoch_history": history,
            "held_out_metrics": held_out,
        },
    )


def run_transformer_oof_fold(
    config: NeuralTrainingConfig,
    *,
    gnn_root: Path | None,
    held_out_fold: int,
    fold_count: int = PRODUCTION_FOLD_COUNT,
    fixed_epochs: int | None = None,
) -> dict[str, Any]:
    """Run one independently queueable paired-OOF Transformer fold task."""

    generated_at = datetime.now(UTC).isoformat()
    started = perf_counter()
    if fold_count < 2:
        raise ValueError("fold-count must be at least two")
    if held_out_fold < 0 or held_out_fold >= fold_count:
        raise ValueError("held-out-fold is outside the configured fold range")
    if _is_protected(config.neural_root) and fold_count != PRODUCTION_FOLD_COUNT:
        raise ValueError("protected paired OOF runs require exactly five folds")
    epochs = _resolved_fixed_epochs(config, fixed_epochs)
    resolved_gnn_root = (
        Path(gnn_root) if gnn_root is not None else config.neural_root.parent / "gnn"
    )
    fold = _fold_config(
        config,
        gnn_root=resolved_gnn_root,
        held_out_fold=held_out_fold,
        fixed_epochs=epochs,
    )
    output_path = transformer_oof_predictions_path(
        config.neural_root,
        held_out_fold,
    )
    report: dict[str, Any] = {
        "schema_version": PAIRED_OOF_SCHEMA_VERSION,
        "protocol_version": PAIRED_OOF_PROTOCOL_VERSION,
        "status": "running",
        "stage": "transformer-oof-fold",
        "generated_at": generated_at,
        "held_out_fold_index": held_out_fold,
        "fold_count": fold_count,
        "fit_fold_indices": [
            index for index in range(fold_count) if index != held_out_fold
        ],
        "fixed_epochs": epochs,
        "selection_policy": "fixed_epoch_no_heldout_selection",
        "leakage_policy": {
            "patient_fold_alignment": "patient_fold_sql shared with GNN",
            "preprocessing_fit_scope": "other_patient_folds_only",
            "graph_fit_scope": "GNN crossfit graph excluding held-out fold",
            "held_out_used_for_early_stopping": False,
            "held_out_used_for_calibration": False,
        },
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "report_contains_identifier_values": False,
            "predictions_are_restricted_patient_level_artifacts": True,
        },
    }
    report_path = fold.training_report_path
    try:
        base_errors = preflight_errors(config, stage="prepare")
        if base_errors:
            raise ValueError(
                f"base Transformer preflight failed with {len(base_errors)} error(s)"
            )
        if config.mode != "development":
            raise ValueError("paired OOF generation must use development mode")
        if config.require_neural_gate and not fold.graph_edges_path.is_file():
            raise FileNotFoundError("fold-excluded GNN graph_edges.parquet is missing")
        input_counts = _materialize_fold_inputs(
            config,
            fold,
            held_out_fold=held_out_fold,
            fold_count=fold_count,
        )
        prepare_report = prepare_neural_caches(fold)
        if prepare_report.get("status") != "completed":
            raise ValueError("fold-isolated Transformer cache preparation failed")
        model, spec, training = _fit_fixed_epoch_model(
            fold,
            fixed_epochs=epochs,
            held_out_fold=held_out_fold,
        )
        row_count = _write_oof_predictions(
            model,
            fold,
            spec,
            device=resolve_device(fold),
            held_out_fold=held_out_fold,
            output_path=output_path,
        )
        if row_count <= 0:
            raise ValueError("Transformer held-out fold produced no OOF predictions")
        report.update(
            {
                "status": "completed",
                "input_aggregate_counts": input_counts,
                "training": training,
                "model": {
                    "architecture": asdict(fold.architecture),
                    "optimization": asdict(fold.optimization),
                    "experiment_version": EXPERIMENT_VERSION,
                    "training_schema_version": TRAINING_SCHEMA_VERSION,
                },
                "artifacts": {
                    "fold_root": str(
                        transformer_fold_root(config.neural_root, held_out_fold)
                    ),
                    "checkpoint": str(fold.checkpoint_path),
                    "oof_predictions": str(output_path),
                    "oof_predictions_sha256": sha256_file(output_path),
                },
                "oof_prediction_row_count": row_count,
            }
        )
    except Exception as error:  # noqa: BLE001 - aggregate fail-closed report
        report["status"] = "failed"
        report["reason"] = safe_error_message(error)
    report["wall_time_seconds"] = perf_counter() - started
    write_json(report_path, report)
    return report
