"""Late-only refit and one-shot scoring for the paired-OOF protocol."""

from __future__ import annotations

import re
import statistics
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from pipeline.extract_utils import safe_error_message
from pipeline.gnn_training.config import (
    LATE_FUSION_BASELINE_NAME,
    MAXIMUM_SECONDARY_DROP,
    MINIMUM_NDCG_LIFT,
    TRANSFORMER_BASELINE_NAME,
    GNNTrainingConfig,
)
from pipeline.gnn_training.contract import (
    blocked_report,
    contract_digest_or_none,
    preflight_errors,
)
from pipeline.gnn_training.data import write_json, write_json_exclusive
from pipeline.gnn_training.fusion import late_fusion_logits
from pipeline.gnn_training.runtime import (
    FrozenTransformerCache,
    iter_gnn_batches,
    load_feature_spec,
    load_gnn_checkpoint,
    refit_gnn,
    require_finite_tensor,
    resolve_device,
    save_gnn_checkpoint,
)
from pipeline.gnn_training.scoring import (
    COMPACT_PREDICTION_SCHEMA,
    AtomicParquetWriter,
    append_authoritative_metrics,
    base_score_report,
    combine_score_tables,
    configured_duckdb,
    materialize_canonical_scores,
    metric_row,
    qualification_decision,
    write_compact_batch,
)
from pipeline.late_fusion_protocol import (
    FROZEN_GATE_SCHEMA_VERSION,
    PAIRED_OOF_PROTOCOL_VERSION,
    PAIRED_OOF_SELECTION_SCHEMA_VERSION,
)
from pipeline.neural_training.config import (
    EXPERIMENT_VERSION as NEURAL_EXPERIMENT_VERSION,
)
from pipeline.training_contract import load_json, sha256_file

GATE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


def _selection_payload(config: GNNTrainingConfig) -> dict[str, Any]:
    if not config.paired_oof_selection_path.is_file():
        raise FileNotFoundError("paired OOF selection artifact is missing")
    selection = load_json(config.paired_oof_selection_path)
    if (
        selection.get("schema_version") != PAIRED_OOF_SELECTION_SCHEMA_VERSION
        or selection.get("protocol_version") != PAIRED_OOF_PROTOCOL_VERSION
        or selection.get("selection_frozen") is not True
        or selection.get("status") not in {"completed", "selected_pending_gnn_refit"}
    ):
        raise ValueError("paired OOF selection artifact is incompatible")
    return selection


def _selected_refit_epochs(config: GNNTrainingConfig, variant: str) -> int:
    epochs: list[int] = []
    for fold_index in range(config.fold_count):
        completion = load_json(
            config.fold_completion_manifest_path(fold_index, variant)
        )
        best_epoch = completion.get("best_epoch")
        if (
            completion.get("status") != "completed"
            or completion.get("ablation_variant") != variant
            or completion.get("held_out_fold_index") != fold_index
            or not isinstance(best_epoch, int)
            or best_epoch < 0
        ):
            raise ValueError("selected GNN fold completion metadata is incompatible")
        epochs.append(best_epoch + 1)
    return max(1, int(round(statistics.median(epochs))))


def _compatible_existing_refit(
    config: GNNTrainingConfig,
    *,
    variant: str,
    epochs: int,
    device: torch.device,
) -> bool:
    if not config.gnn_checkpoint_path.is_file():
        return False
    spec = load_feature_spec(config)
    try:
        _model, payload = load_gnn_checkpoint(
            config.gnn_checkpoint_path,
            spec,
            device=device,
            expected_seed=config.seed,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return (
        payload.get("ablation_variant") == variant and payload.get("epochs") == epochs
    )


def refit_paired_gnn(config: GNNTrainingConfig) -> dict[str, Any]:
    """Reuse an exact full refit or fit the jointly selected GNN once."""

    generated_at = datetime.now(UTC).isoformat()
    started = perf_counter()
    errors = preflight_errors(config, stage="refit-paired-gnn")
    if errors:
        report = blocked_report(
            config=config,
            schema_version=PAIRED_OOF_SELECTION_SCHEMA_VERSION,
            stage="refit-paired-gnn",
            generated_at=generated_at,
            errors=errors,
        )
        write_json(config.paired_gnn_refit_report_path, report)
        return report
    report: dict[str, Any] = {
        "schema_version": PAIRED_OOF_SELECTION_SCHEMA_VERSION,
        "protocol_version": PAIRED_OOF_PROTOCOL_VERSION,
        "status": "running",
        "stage": "refit-paired-gnn",
        "generated_at": generated_at,
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "report_contains_identifier_values": False,
        },
    }
    try:
        selection = _selection_payload(config)
        variant = str(selection["selected_gnn_variant"])
        epochs = _selected_refit_epochs(config, variant)
        device = resolve_device(config)
        reused = _compatible_existing_refit(
            config,
            variant=variant,
            epochs=epochs,
            device=device,
        )
        if reused:
            checkpoint_path = config.gnn_checkpoint_path
        else:
            spec = load_feature_spec(config)
            model = refit_gnn(
                config,
                spec,
                ablation_variant=variant,
                epochs=epochs,
                device=device,
            )
            save_gnn_checkpoint(
                model,
                config,
                spec,
                path=config.paired_gnn_checkpoint_path,
                ablation_variant=variant,
                epochs=epochs,
                metadata={
                    "selection_source": "paired_patient_grouped_oof",
                    "paired_oof_selection_sha256": sha256_file(
                        config.paired_oof_selection_path
                    ),
                    "protocol_version": PAIRED_OOF_PROTOCOL_VERSION,
                },
            )
            checkpoint_path = config.paired_gnn_checkpoint_path
        checkpoint_lock = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        }
        late_checkpoint = dict(selection)
        late_checkpoint.update(
            {
                "status": "completed",
                "inference_ready": True,
                "refit_required": False,
                "gnn_checkpoint": checkpoint_lock,
                "gnn_refit_epochs": epochs,
                "gnn_refit_reused_existing": reused,
                "paired_oof_selection_sha256": sha256_file(
                    config.paired_oof_selection_path
                ),
                "refit_completed_at": datetime.now(UTC).isoformat(),
            }
        )
        write_json(config.paired_late_checkpoint_path, late_checkpoint)
        report.update(
            {
                "status": "completed",
                "device": str(device),
                "selected_gnn_variant": variant,
                "refit_epochs": epochs,
                "reused_existing_full_refit": reused,
                "artifacts": {"gnn_checkpoint": checkpoint_lock},
                "contract_digest": contract_digest_or_none(config),
            }
        )
    except Exception as error:  # noqa: BLE001 - aggregate fail-closed report
        report["status"] = "failed"
        report["reason"] = safe_error_message(error)
    report["wall_time_seconds"] = perf_counter() - started
    write_json(config.paired_gnn_refit_report_path, report)
    return report


def _validate_frozen_gate(
    config: GNNTrainingConfig,
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    path = config.paired_frozen_gate_manifest_path
    try:
        path.resolve().relative_to(config.reports_root.resolve())
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("frozen gate manifest must stay under REPORTS_ROOT") from error
    if not path.is_file():
        raise FileNotFoundError(
            "new frozen gate manifest is missing; gate construction is a reviewed "
            "protocol decision and is never inferred by the scoring command"
        )
    gate = load_json(path)
    split = gate.get("split")
    gate_id = gate.get("gate_id")
    required_true = (
        "selection_completed_before_gate_opened",
        "one_shot_scoring_authorized",
    )
    required_false = (
        "used_for_model_selection",
        "used_for_gnn_variant_selection",
        "used_for_alpha_selection",
        "previously_scored_by_hybrid",
    )
    try:
        gate_frozen_at = datetime.fromisoformat(str(gate["frozen_at"]))
        selection_generated_at = datetime.fromisoformat(str(checkpoint["generated_at"]))
        ordered_timestamps = (
            gate_frozen_at.tzinfo is not None
            and selection_generated_at.tzinfo is not None
            and gate_frozen_at > selection_generated_at
        )
    except (KeyError, TypeError, ValueError):
        ordered_timestamps = False
    if (
        gate.get("schema_version") != FROZEN_GATE_SCHEMA_VERSION
        or gate.get("protocol_version") != PAIRED_OOF_PROTOCOL_VERSION
        or gate.get("status") != "frozen"
        or gate.get("source") != "mimiciv"
        or not isinstance(split, str)
        or split == "train"
        or not isinstance(gate_id, str)
        or GATE_ID_PATTERN.fullmatch(gate_id) is None
        or any(gate.get(name) is not True for name in required_true)
        or any(gate.get(name) is not False for name in required_false)
        or not ordered_timestamps
        or gate.get("patient_overlap_with_train_count") != 0
        or gate.get("paired_oof_selection_sha256")
        != sha256_file(config.paired_oof_selection_path)
        or gate.get("gnn_cache_manifest_sha256")
        != sha256_file(config.cache_manifest_path)
        or gate.get("transformer_cache_manifest_sha256")
        != sha256_file(config.transformer_cache_manifest_path)
        or checkpoint.get("paired_oof_selection_sha256")
        != sha256_file(config.paired_oof_selection_path)
    ):
        raise ValueError("frozen gate manifest violates the paired OOF protocol")
    return gate, split


def _load_late_components(
    config: GNNTrainingConfig,
) -> tuple[dict[str, Any], Any, torch.device]:
    checkpoint = load_json(config.paired_late_checkpoint_path)
    if (
        checkpoint.get("protocol_version") != PAIRED_OOF_PROTOCOL_VERSION
        or checkpoint.get("status") != "completed"
        or checkpoint.get("selection_frozen") is not True
        or checkpoint.get("inference_ready") is not True
    ):
        raise ValueError("late-only checkpoint is not inference-ready")
    gnn_lock = checkpoint.get("gnn_checkpoint")
    if not isinstance(gnn_lock, dict):
        raise ValueError("late-only checkpoint has no locked GNN refit")
    gnn_path = Path(str(gnn_lock.get("path", "")))
    if not gnn_path.is_file() or gnn_lock.get("sha256") != sha256_file(gnn_path):
        raise ValueError("late-only GNN refit is missing or changed")
    if checkpoint.get("transformer_checkpoint_sha256") != sha256_file(
        config.neural_checkpoint_path
    ):
        raise ValueError("frozen Transformer changed after paired OOF selection")
    device = resolve_device(config)
    spec = load_feature_spec(config)
    gnn, payload = load_gnn_checkpoint(
        gnn_path,
        spec,
        device=device,
        expected_seed=config.seed,
    )
    if payload.get("ablation_variant") != checkpoint.get("selected_gnn_variant"):
        raise ValueError("late-only GNN variant differs from paired OOF selection")
    return checkpoint, gnn, device


@torch.no_grad()
def _write_late_predictions(
    config: GNNTrainingConfig,
    *,
    split: str,
    hybrid_path: Path,
    transformer_path: Path,
) -> tuple[int, int, str, float]:
    checkpoint, gnn, device = _load_late_components(config)
    alpha = float(checkpoint["selected_alpha"])
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("late-only alpha is outside [0, 1]")
    spec = load_feature_spec(config)
    gnn.eval()
    frozen_cache = FrozenTransformerCache(config)
    with (
        AtomicParquetWriter(hybrid_path, COMPACT_PREDICTION_SCHEMA) as hybrid_writer,
        AtomicParquetWriter(
            transformer_path,
            COMPACT_PREDICTION_SCHEMA,
        ) as transformer_writer,
    ):
        for batch in iter_gnn_batches(
            config,
            spec,
            split=split,
            shards_root=config.shards_root,
            shuffle=False,
        ):
            moved = batch.to(device)
            frozen = frozen_cache.align(batch, device=device)
            gnn_logits = gnn.forward_batch(moved).logits
            require_finite_tensor(
                gnn_logits,
                name="paired late-only GNN logits",
                mask=moved.candidate_mask,
            )
            hybrid_logits = late_fusion_logits(
                frozen.logits,
                gnn_logits,
                gnn_weight=alpha,
                candidate_mask=moved.candidate_mask,
            )
            write_compact_batch(
                hybrid_writer,
                batch,
                torch.sigmoid(hybrid_logits),
            )
            write_compact_batch(
                transformer_writer,
                batch,
                torch.sigmoid(frozen.logits),
            )
        hybrid_rows = hybrid_writer.commit()
        transformer_rows = transformer_writer.commit()
    return hybrid_rows, transformer_rows, str(device), alpha


def score_paired_late(config: GNNTrainingConfig) -> dict[str, Any]:
    """Claim and evaluate the newly frozen gate exactly once."""

    generated_at = datetime.now(UTC).isoformat()
    errors = preflight_errors(config, stage="score-paired-late")
    if config.paired_late_score_completion_path.exists():
        errors.append(
            {
                "code": "paired_gate_already_attempted",
                "detail": "the one-shot paired late-fusion gate was already claimed",
            }
        )
    if errors:
        report = blocked_report(
            config=config,
            schema_version=FROZEN_GATE_SCHEMA_VERSION,
            stage="score-paired-late",
            generated_at=generated_at,
            errors=errors,
        )
        write_json(config.paired_late_score_report_path, report)
        return report

    try:
        checkpoint = load_json(config.paired_late_checkpoint_path)
        gate, split = _validate_frozen_gate(config, checkpoint)
    except Exception as error:  # noqa: BLE001 - aggregate fail-closed report
        report = blocked_report(
            config=config,
            schema_version=FROZEN_GATE_SCHEMA_VERSION,
            stage="score-paired-late",
            generated_at=generated_at,
            errors=[
                {
                    "code": "invalid_frozen_gate",
                    "detail": safe_error_message(error),
                }
            ],
        )
        write_json(config.paired_late_score_report_path, report)
        return report

    try:
        write_json_exclusive(
            config.paired_late_score_completion_path,
            {
                "schema_version": FROZEN_GATE_SCHEMA_VERSION,
                "protocol_version": PAIRED_OOF_PROTOCOL_VERSION,
                "status": "running",
                "gate_id": gate["gate_id"],
                "split": split,
                "started_at": generated_at,
                "one_shot_claim": True,
            },
        )
    except FileExistsError:
        return blocked_report(
            config=config,
            schema_version=FROZEN_GATE_SCHEMA_VERSION,
            stage="score-paired-late",
            generated_at=generated_at,
            errors=[
                {
                    "code": "paired_gate_claim_race",
                    "detail": "another process claimed the one-shot paired gate",
                }
            ],
        )

    report = base_score_report(
        config,
        schema_version=FROZEN_GATE_SCHEMA_VERSION,
        stage="score-paired-late",
        split=split,
        baseline_name=LATE_FUSION_BASELINE_NAME,
        output_path=config.paired_late_score_output_path,
    )
    try:
        root = config.paired_late_score_root
        hybrid_predictions = root / "_hybrid_predictions.parquet"
        transformer_predictions = root / "_transformer_predictions.parquet"
        hybrid_rows, transformer_rows, device, alpha = _write_late_predictions(
            config,
            split=split,
            hybrid_path=hybrid_predictions,
            transformer_path=transformer_predictions,
        )
        hybrid_scores = root / "_hybrid_candidate_scores.parquet"
        transformer_scores = root / "_transformer_candidate_scores.parquet"
        with configured_duckdb(config) as connection:
            materialize_canonical_scores(
                connection,
                config,
                predictions_path=hybrid_predictions,
                output_path=hybrid_scores,
                split=split,
                baseline_name=LATE_FUSION_BASELINE_NAME,
                baseline_version=PAIRED_OOF_PROTOCOL_VERSION,
                evaluation_version=FROZEN_GATE_SCHEMA_VERSION,
                generated_at=generated_at,
            )
            materialize_canonical_scores(
                connection,
                config,
                predictions_path=transformer_predictions,
                output_path=transformer_scores,
                split=split,
                baseline_name=TRANSFORMER_BASELINE_NAME,
                baseline_version=NEURAL_EXPERIMENT_VERSION,
                evaluation_version=FROZEN_GATE_SCHEMA_VERSION,
                generated_at=generated_at,
            )
            score_rows = combine_score_tables(
                connection,
                (hybrid_scores, transformer_scores),
                output_path=config.paired_late_score_output_path,
            )
            append_authoritative_metrics(
                connection,
                config,
                report,
                evaluation_root=root,
            )
        candidate = metric_row(
            report,
            baseline_name=LATE_FUSION_BASELINE_NAME,
            split=split,
        )
        reference = metric_row(
            report,
            baseline_name=TRANSFORMER_BASELINE_NAME,
            split=split,
        )
        if candidate is None or reference is None:
            raise ValueError("paired gate produced no comparable ranking metrics")
        decision = qualification_decision(
            candidate=candidate,
            reference=reference,
            minimum_ndcg_lift=MINIMUM_NDCG_LIFT,
            maximum_secondary_drop=MAXIMUM_SECONDARY_DROP,
        )
        report.update(
            {
                "gate_id": gate["gate_id"],
                "frozen_gate_manifest_sha256": sha256_file(
                    config.paired_frozen_gate_manifest_path
                ),
                "selected_gnn_variant": checkpoint["selected_gnn_variant"],
                "selected_alpha": alpha,
                "device": device,
                "hybrid_prediction_row_count": hybrid_rows,
                "transformer_prediction_row_count": transformer_rows,
                "score_row_count": score_rows,
                "qualified": decision["qualified"],
                "decision": (
                    "promote_hybrid"
                    if decision["qualified"]
                    else "retain_frozen_transformer"
                ),
                "final_scoring_authorized": bool(decision["qualified"]),
                "qualification_decision": decision,
                "temperature_policy": (
                    "ranking gate uses raw-logit monotonic scores; no temperature "
                    "was fit on or selected with the frozen gate"
                ),
            }
        )
    except Exception as error:  # noqa: BLE001 - one-shot failure stays claimed
        report["status"] = "failed"
        report["reason"] = safe_error_message(error)
    write_json(config.paired_late_score_report_path, report)
    write_json(
        config.paired_late_score_completion_path,
        {
            "schema_version": FROZEN_GATE_SCHEMA_VERSION,
            "protocol_version": PAIRED_OOF_PROTOCOL_VERSION,
            "status": report.get("status", "failed"),
            "gate_id": gate["gate_id"],
            "split": split,
            "started_at": generated_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "one_shot_claim": True,
            "score_report_sha256": sha256_file(config.paired_late_score_report_path),
        },
    )
    return report
