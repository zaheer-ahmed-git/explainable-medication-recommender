"""Patient-grouped cross-fit selection and full-train GNN refit."""

from __future__ import annotations

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
    fit_gnn_temperature,
    fold_shards_root,
    load_feature_spec,
    load_gnn_checkpoint,
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


def train_gnn(config: GNNTrainingConfig) -> dict[str, Any]:
    """Select the relation ablation on fold-safe graphs and refit on all train."""

    generated_at = datetime.now(UTC).isoformat()
    errors = preflight_errors(config, stage="train-gnn")
    if errors:
        report = blocked_report(
            config=config,
            schema_version=GNN_TRAINING_SCHEMA_VERSION,
            stage="train-gnn",
            generated_at=generated_at,
            errors=errors,
        )
        write_json(config.gnn_training_report_path, report)
        return report

    report: dict[str, Any] = {
        "schema_version": GNN_TRAINING_SCHEMA_VERSION,
        "status": "running",
        "stage": "train-gnn",
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
        fold_results: dict[str, list[dict[str, Any]]] = {
            variant: [] for variant in ABLATION_VARIANTS
        }
        epoch_history: dict[str, list[dict[str, Any]]] = {
            variant: [] for variant in ABLATION_VARIANTS
        }
        for variant in ABLATION_VARIANTS:
            for fold_index in range(config.fold_count):
                spec = load_feature_spec(config, fold_index=fold_index)
                result = fit_crossfit_gnn(
                    config,
                    spec,
                    held_out_fold=fold_index,
                    ablation_variant=variant,
                    device=device,
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
                            value
                            for value in range(config.fold_count)
                            if value != fold_index
                        ],
                    },
                )
                fold_results[variant].append(
                    {
                        "fold_index": fold_index,
                        "best_epoch": result.best_epoch,
                        **result.best_metrics,
                    }
                )
                epoch_history[variant].append(
                    {
                        "fold_index": fold_index,
                        "epochs": list(result.history),
                    }
                )

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

        full_spec = load_feature_spec(config)
        final_model = refit_gnn(
            config,
            full_spec,
            ablation_variant=selected_variant,
            epochs=refit_epochs,
            device=device,
        )
        temperature = fit_gnn_temperature(
            final_model,
            config,
            full_spec,
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
                "fit_split": "mimiciv_validation",
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
            "crossfit_manifest_sha256": sha256_file(
                config.crossfit_graph_manifest_path
            ),
            "generated_at": generated_at,
        }
        write_json(config.gnn_training_state_path, training_state)

        report.update(
            {
                "status": "completed",
                "device": str(device),
                "mixed_precision": use_amp(config, device),
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
