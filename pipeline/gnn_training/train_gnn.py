"""Patient-grouped cross-fit selection and full-train GNN refit."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import torch

from pipeline.extract_utils import safe_error_message
from pipeline.gnn_training.config import (
    GNN_EXPERIMENT_VERSION,
    GNN_TRAINING_SCHEMA_VERSION,
    GNNTrainingConfig,
)
from pipeline.gnn_training.contract import (
    blocked_report,
    contract_digest_or_none,
    preflight_errors,
)
from pipeline.gnn_training.data import write_json
from pipeline.gnn_training.model import ABLATION_VARIANTS, build_model
from pipeline.gnn_training.runtime import (
    evaluate_gnn,
    fit_crossfit_gnn,
    fit_oof_temperature,
    fold_shards_root,
    load_feature_spec,
    load_gnn_checkpoint,
    precision_mode,
    refit_gnn,
    resolve_device,
    save_gnn_checkpoint,
    use_amp,
)
from pipeline.gnn_training.scoring import (
    OOF_PREDICTION_SCHEMA,
    AtomicParquetWriter,
    write_oof_batch,
)
from pipeline.training_contract import sha256_file
from pipeline.training_contract import load_json


def _weighted_fold_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate fold metrics by positive ranking-group count."""

    total = sum(int(row["positive_ranking_group_count"]) for row in rows)
    if total <= 0:
        raise ValueError("cross-fit variant has no positive held-out groups")

    def mean(name: str) -> float:
        return (
            sum(
                float(row[name]) * int(row["positive_ranking_group_count"])
                for row in rows
            )
            / total
        )

    return {
        "positive_ranking_group_count": total,
        "ndcg_at_k": mean("ndcg_at_k"),
        "mrr_at_k": mean("mrr_at_k"),
        "hit_rate_at_k": mean("hit_rate_at_k"),
    }


def _select_variant(
    variant_rows: dict[str, list[dict[str, Any]]],
) -> tuple[str, dict[str, dict[str, Any]]]:
    summaries = {
        variant: _weighted_fold_metrics(rows) for variant, rows in variant_rows.items()
    }
    # A metric tie preserves the pre-registered order (full first), avoiding a
    # post-hoc complexity preference after seeing held-out results.
    selected = max(
        ABLATION_VARIANTS,
        key=lambda variant: (
            float(summaries[variant]["ndcg_at_k"]),
            float(summaries[variant]["mrr_at_k"]),
            float(summaries[variant]["hit_rate_at_k"]),
            -ABLATION_VARIANTS.index(variant),
        ),
    )
    return selected, summaries


def _write_selected_oof(
    config: GNNTrainingConfig,
    *,
    selected_variant: str,
    device: torch.device,
) -> int:
    """Regenerate exactly one OOF prediction per MIMIC-train candidate."""

    with AtomicParquetWriter(
        config.gnn_oof_predictions_path,
        OOF_PREDICTION_SCHEMA,
    ) as writer:
        for fold_index in range(config.fold_count):
            spec = load_feature_spec(config, fold_index=fold_index)
            checkpoint = config.fold_checkpoint_path(fold_index, selected_variant)
            model, payload = load_gnn_checkpoint(
                checkpoint,
                spec,
                device=device,
                expected_seed=config.seed,
            )
            if payload.get("ablation_variant") != selected_variant:
                raise ValueError("fold checkpoint ablation variant does not match")
            evaluate_gnn(
                model,
                config,
                spec,
                device=device,
                split="train",
                shards_root=fold_shards_root(config, fold_index),
                include_fold_ids=frozenset({fold_index}),
                batch_callback=lambda batch, logits, _output: write_oof_batch(
                    writer,
                    batch,
                    logits,
                ),
            )
        return writer.commit()


def _load_completed_fold(
    config: GNNTrainingConfig,
    *,
    spec: Any,
    fold_index: int,
    variant: str,
    device: torch.device,
    crossfit_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return a compatible completed fold checkpoint, otherwise retrain it."""

    path = config.fold_checkpoint_path(fold_index, variant)
    completion_path = config.fold_completion_manifest_path(fold_index, variant)
    if not path.is_file() or not completion_path.is_file():
        return None
    try:
        completion = load_json(completion_path)
        if (
            completion.get("status") != "completed"
            or completion.get("seed") != config.seed
            or completion.get("held_out_fold_index") != fold_index
            or completion.get("ablation_variant") != variant
            or completion.get("architecture") != asdict(config.architecture)
            or completion.get("optimization") != asdict(config.optimization)
            or completion.get("crossfit_manifest_sha256") != crossfit_manifest_sha256
            or completion.get("checkpoint_sha256") != sha256_file(path)
        ):
            return None
        _model, payload = load_gnn_checkpoint(
            path,
            spec,
            device=device,
            expected_seed=config.seed,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    metadata = payload.get("metadata")
    if (
        payload.get("ablation_variant") != variant
        or payload.get("architecture") != asdict(config.architecture)
        or payload.get("optimization") != asdict(config.optimization)
        or not isinstance(metadata, dict)
        or metadata.get("held_out_fold_index") != fold_index
        or metadata.get("crossfit_manifest_sha256") != crossfit_manifest_sha256
        or not isinstance(metadata.get("best_metrics"), dict)
        or not isinstance(metadata.get("epoch_history"), list)
    ):
        return None
    best_epoch = int(metadata.get("best_epoch", -1))
    if best_epoch < 0:
        return None
    fold_row = {
        "fold_index": fold_index,
        "best_epoch": best_epoch,
        **metadata["best_metrics"],
    }
    history_row = {
        "fold_index": fold_index,
        "epochs": metadata["epoch_history"],
    }
    print(
        f'{{"event":"gnn_fold_resumed","scope":"{variant}/fold-{fold_index}"}}',
        flush=True,
    )
    return fold_row, history_row


def _fit_fold_checkpoint(
    config: GNNTrainingConfig,
    *,
    fold_index: int,
    variant: str,
    device: torch.device,
    crossfit_manifest_sha256: str,
    allow_fit: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one compatible completed fold or fit and checkpoint it."""

    spec = load_feature_spec(config, fold_index=fold_index)
    completed = _load_completed_fold(
        config,
        spec=spec,
        fold_index=fold_index,
        variant=variant,
        device=device,
        crossfit_manifest_sha256=crossfit_manifest_sha256,
    )
    if completed is not None:
        return completed
    if not allow_fit:
        raise FileNotFoundError(
            f"compatible completed fold checkpoint is missing for {variant}, "
            f"fold {fold_index}"
        )
    result = fit_crossfit_gnn(
        config,
        spec,
        held_out_fold=fold_index,
        ablation_variant=variant,
        device=device,
        crossfit_manifest_sha256=crossfit_manifest_sha256,
    )
    fold_model = build_model(
        spec,
        config.architecture,
        ablation_variant=variant,
    ).to(device)
    fold_model.load_state_dict(result.state_dict, strict=True)
    save_gnn_checkpoint(
        fold_model,
        config,
        spec,
        path=config.fold_checkpoint_path(fold_index, variant),
        ablation_variant=variant,
        epochs=result.best_epoch + 1,
        metadata={
            "held_out_fold_index": fold_index,
            "fit_fold_indices": [
                value for value in range(config.fold_count) if value != fold_index
            ],
            "best_epoch": result.best_epoch,
            "best_metrics": result.best_metrics,
            "epoch_history": list(result.history),
            "crossfit_manifest_sha256": crossfit_manifest_sha256,
        },
    )
    write_json(
        config.fold_completion_manifest_path(fold_index, variant),
        {
            "schema_version": GNN_TRAINING_SCHEMA_VERSION,
            "status": "completed",
            "seed": config.seed,
            "held_out_fold_index": fold_index,
            "fit_fold_indices": [
                value for value in range(config.fold_count) if value != fold_index
            ],
            "ablation_variant": variant,
            "architecture": asdict(config.architecture),
            "optimization": asdict(config.optimization),
            "crossfit_manifest_sha256": crossfit_manifest_sha256,
            "checkpoint_sha256": sha256_file(
                config.fold_checkpoint_path(fold_index, variant)
            ),
            "best_epoch": result.best_epoch,
            "best_metrics": result.best_metrics,
        },
    )
    return (
        {
            "fold_index": fold_index,
            "best_epoch": result.best_epoch,
            **result.best_metrics,
        },
        {
            "fold_index": fold_index,
            "epochs": list(result.history),
        },
    )


def train_gnn_fold(
    config: GNNTrainingConfig,
    *,
    variant: str,
    fold_index: int,
) -> dict[str, Any]:
    """Fit exactly one array-addressable ``(variant, fold)`` task."""

    generated_at = datetime.now(UTC).isoformat()
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"unsupported GNN ablation variant: {variant}")
    if fold_index not in range(config.fold_count):
        raise ValueError("held-out fold is outside the configured fold range")
    errors = preflight_errors(config, stage="train-gnn-fold")
    if errors:
        return blocked_report(
            config=config,
            schema_version=GNN_TRAINING_SCHEMA_VERSION,
            stage="train-gnn-fold",
            generated_at=generated_at,
            errors=errors,
        )
    device = resolve_device(config)
    manifest_hash = sha256_file(config.crossfit_graph_manifest_path)
    fold_row, history_row = _fit_fold_checkpoint(
        config,
        fold_index=fold_index,
        variant=variant,
        device=device,
        crossfit_manifest_sha256=manifest_hash,
    )
    return {
        "schema_version": GNN_TRAINING_SCHEMA_VERSION,
        "status": "completed",
        "stage": "train-gnn-fold",
        "generated_at": generated_at,
        "ablation_variant": variant,
        "fold_result": fold_row,
        "epoch_history": history_row,
        "checkpoint_created": True,
        "completion_manifest_created": True,
        "report_contains_patient_rows": False,
    }


def train_gnn(
    config: GNNTrainingConfig,
    *,
    allow_fold_fitting: bool = True,
    finalize_refit: bool = True,
    stage: str = "train-gnn",
) -> dict[str, Any]:
    """Select the relation ablation on fold-safe graphs and refit on all train."""

    generated_at = datetime.now(UTC).isoformat()
    errors = preflight_errors(config, stage=stage)
    if errors:
        report = blocked_report(
            config=config,
            schema_version=GNN_TRAINING_SCHEMA_VERSION,
            stage=stage,
            generated_at=generated_at,
            errors=errors,
        )
        write_json(config.gnn_training_report_path, report)
        return report

    report: dict[str, Any] = {
        "schema_version": GNN_TRAINING_SCHEMA_VERSION,
        "status": "running",
        "stage": stage,
        "mode": config.mode,
        "generated_at": generated_at,
        "seed": config.seed,
        "experiment_version": GNN_EXPERIMENT_VERSION,
        "selection_protocol": (
            "patient_grouped_mimic_train_crossfit_with_fold_excluded_graphs"
        ),
        "selection_metric": "crossfit_ndcg_at_10",
        "pre_registered_variants": list(ABLATION_VARIANTS),
        "fold_count": config.fold_count,
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "report_contains_identifier_values": False,
            "restricted_oof_predictions_are_local_only": True,
        },
    }
    try:
        device = resolve_device(config)
        crossfit_manifest_sha256 = sha256_file(config.crossfit_graph_manifest_path)
        fold_results: dict[str, list[dict[str, Any]]] = {
            variant: [] for variant in ABLATION_VARIANTS
        }
        epoch_history: dict[str, list[dict[str, Any]]] = {
            variant: [] for variant in ABLATION_VARIANTS
        }
        for variant in ABLATION_VARIANTS:
            for fold_index in range(config.fold_count):
                fold_row, history_row = _fit_fold_checkpoint(
                    config,
                    fold_index=fold_index,
                    variant=variant,
                    device=device,
                    crossfit_manifest_sha256=crossfit_manifest_sha256,
                    allow_fit=allow_fold_fitting,
                )
                fold_results[variant].append(fold_row)
                epoch_history[variant].append(history_row)

        selected_variant, variant_summaries = _select_variant(fold_results)
        refit_epochs = max(
            1,
            int(
                round(
                    statistics.median(
                        row["best_epoch"] + 1 for row in fold_results[selected_variant]
                    )
                )
            ),
        )
        oof_rows = _write_selected_oof(
            config,
            selected_variant=selected_variant,
            device=device,
        )
        temperature = fit_oof_temperature(
            config.gnn_oof_predictions_path,
            device=device,
        )
        write_json(
            config.gnn_crossfit_selection_path,
            {
                "schema_version": GNN_TRAINING_SCHEMA_VERSION,
                "status": "completed",
                "selected_variant": selected_variant,
                "refit_epochs": refit_epochs,
                "temperature": temperature,
                "crossfit_manifest_sha256": crossfit_manifest_sha256,
                "oof_predictions_sha256": sha256_file(config.gnn_oof_predictions_path),
                "architecture": asdict(config.architecture),
                "optimization": asdict(config.optimization),
                "generated_at": generated_at,
            },
        )

        if not finalize_refit:
            report.update(
                {
                    "status": "completed",
                    "device": str(device),
                    "selected_variant": selected_variant,
                    "variant_crossfit_metrics": variant_summaries,
                    "fold_results": fold_results,
                    "epoch_history": epoch_history,
                    "refit_epochs": refit_epochs,
                    "temperature": temperature,
                    "oof_prediction_row_count": oof_rows,
                    "selection_artifact": str(config.gnn_crossfit_selection_path),
                }
            )
            write_json(config.gnn_training_report_path, report)
            return report

        full_spec = load_feature_spec(config)
        final_model = refit_gnn(
            config,
            full_spec,
            ablation_variant=selected_variant,
            epochs=refit_epochs,
            device=device,
        )
        save_gnn_checkpoint(
            final_model,
            config,
            full_spec,
            path=config.gnn_checkpoint_path,
            ablation_variant=selected_variant,
            epochs=refit_epochs,
            metadata={
                "selection_source": "mimiciv_train_crossfit",
                "selected_variant": selected_variant,
            },
        )
        write_json(
            config.gnn_calibration_path,
            {
                "schema_version": GNN_TRAINING_SCHEMA_VERSION,
                "method": "bounded_log_grid_single_temperature_bce",
                "temperature": temperature,
                "fit_split": "mimiciv_train_patient_grouped_oof",
                "fit_after_ranking_selection": True,
                "generated_at": generated_at,
            },
        )
        training_state = {
            "schema_version": GNN_TRAINING_SCHEMA_VERSION,
            "status": "completed",
            "selected_variant": selected_variant,
            "refit_epochs": refit_epochs,
            "temperature": temperature,
            "seed": config.seed,
            "contract_digest": contract_digest_or_none(config),
            "crossfit_manifest_sha256": crossfit_manifest_sha256,
            "generated_at": generated_at,
        }
        write_json(config.gnn_training_state_path, training_state)

        report.update(
            {
                "status": "completed",
                "device": str(device),
                "mixed_precision": use_amp(config, device),
                "precision": precision_mode(config, device),
                "selected_variant": selected_variant,
                "variant_crossfit_metrics": variant_summaries,
                "fold_results": fold_results,
                "epoch_history": epoch_history,
                "refit_epochs": refit_epochs,
                "temperature": temperature,
                "model": {
                    "parameter_count": final_model.parameter_count(),
                    "architecture": asdict(config.architecture),
                    "optimization": asdict(config.optimization),
                },
                "artifacts": {
                    "checkpoint": str(config.gnn_checkpoint_path),
                    "calibration": str(config.gnn_calibration_path),
                    "training_state": str(config.gnn_training_state_path),
                    "selected_oof_predictions": str(config.gnn_oof_predictions_path),
                },
                "oof_prediction_row_count": oof_rows,
                "contract_digest": contract_digest_or_none(config),
            }
        )
    except Exception as error:  # noqa: BLE001 - aggregate fail-closed report
        report["status"] = "failed"
        report["reason"] = safe_error_message(error)

    write_json(config.gnn_training_report_path, report)
    return report


def select_gnn(config: GNNTrainingConfig) -> dict[str, Any]:
    """Select from completed array fold tasks and materialize OOF calibration."""

    return train_gnn(
        config,
        allow_fold_fitting=False,
        finalize_refit=False,
        stage="select-gnn",
    )


def refit_selected_gnn(config: GNNTrainingConfig) -> dict[str, Any]:
    """Refit only after a compatible completed cross-fit selection stage."""

    generated_at = datetime.now(UTC).isoformat()
    errors = preflight_errors(config, stage="refit-gnn")
    if errors:
        report = blocked_report(
            config=config,
            schema_version=GNN_TRAINING_SCHEMA_VERSION,
            stage="refit-gnn",
            generated_at=generated_at,
            errors=errors,
        )
        write_json(config.gnn_training_report_path, report)
        return report
    report: dict[str, Any] = {
        "schema_version": GNN_TRAINING_SCHEMA_VERSION,
        "status": "running",
        "stage": "refit-gnn",
        "mode": config.mode,
        "generated_at": generated_at,
        "seed": config.seed,
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "report_contains_identifier_values": False,
        },
    }
    try:
        if not config.gnn_crossfit_selection_path.is_file():
            raise FileNotFoundError(
                "completed GNN cross-fit selection artifact is missing"
            )
        payload = load_json(config.gnn_crossfit_selection_path)
        crossfit_hash = sha256_file(config.crossfit_graph_manifest_path)
        if (
            payload.get("status") != "completed"
            or payload.get("architecture") != asdict(config.architecture)
            or payload.get("optimization") != asdict(config.optimization)
            or payload.get("crossfit_manifest_sha256") != crossfit_hash
            or payload.get("oof_predictions_sha256")
            != sha256_file(config.gnn_oof_predictions_path)
        ):
            raise ValueError("GNN cross-fit selection artifact is incompatible")
        selected_variant = str(payload.get("selected_variant"))
        if selected_variant not in ABLATION_VARIANTS:
            raise ValueError("GNN selection names an unsupported variant")
        refit_epochs = int(payload.get("refit_epochs", 0))
        temperature = float(payload.get("temperature", 0.0))
        if refit_epochs < 1 or not math.isfinite(temperature):
            raise ValueError("GNN selection refit parameters are invalid")
        if temperature <= 0:
            raise ValueError("GNN selection temperature must be positive")

        device = resolve_device(config)
        full_spec = load_feature_spec(config)
        final_model = refit_gnn(
            config,
            full_spec,
            ablation_variant=selected_variant,
            epochs=refit_epochs,
            device=device,
        )
        save_gnn_checkpoint(
            final_model,
            config,
            full_spec,
            path=config.gnn_checkpoint_path,
            ablation_variant=selected_variant,
            epochs=refit_epochs,
            metadata={
                "selection_source": "mimiciv_train_crossfit",
                "selected_variant": selected_variant,
                "selection_artifact_sha256": sha256_file(
                    config.gnn_crossfit_selection_path
                ),
            },
        )
        write_json(
            config.gnn_calibration_path,
            {
                "schema_version": GNN_TRAINING_SCHEMA_VERSION,
                "method": "bounded_log_grid_single_temperature_bce",
                "temperature": temperature,
                "fit_split": "mimiciv_train_patient_grouped_oof",
                "fit_after_ranking_selection": True,
                "generated_at": generated_at,
            },
        )
        write_json(
            config.gnn_training_state_path,
            {
                "schema_version": GNN_TRAINING_SCHEMA_VERSION,
                "status": "completed",
                "selected_variant": selected_variant,
                "refit_epochs": refit_epochs,
                "temperature": temperature,
                "seed": config.seed,
                "contract_digest": contract_digest_or_none(config),
                "crossfit_manifest_sha256": crossfit_hash,
                "generated_at": generated_at,
            },
        )
        report.update(
            {
                "status": "completed",
                "device": str(device),
                "mixed_precision": use_amp(config, device),
                "precision": precision_mode(config, device),
                "selected_variant": selected_variant,
                "refit_epochs": refit_epochs,
                "temperature": temperature,
                "model": {
                    "parameter_count": final_model.parameter_count(),
                    "architecture": asdict(config.architecture),
                    "optimization": asdict(config.optimization),
                },
                "artifacts": {
                    "checkpoint": str(config.gnn_checkpoint_path),
                    "calibration": str(config.gnn_calibration_path),
                    "training_state": str(config.gnn_training_state_path),
                    "selection": str(config.gnn_crossfit_selection_path),
                },
                "contract_digest": contract_digest_or_none(config),
            }
        )
    except Exception as error:  # noqa: BLE001 - aggregate fail-closed report
        report["status"] = "failed"
        report["reason"] = safe_error_message(error)
    write_json(config.gnn_training_report_path, report)
    return report
