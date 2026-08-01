"""Canonical hybrid scoring and dynamic frozen-Transformer promotion gate."""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from pipeline.extract_utils import safe_error_message
from pipeline.gnn_training.config import (
    FUSION_EXPERIMENT_VERSION,
    FUSION_SELECTION_SCHEMA_VERSION,
    FUSION_TRAINING_SCHEMA_VERSION,
    LATE_FUSION_BASELINE_NAME,
    MAXIMUM_SECONDARY_DROP,
    MINIMUM_NDCG_LIFT,
    RESIDUAL_FUSION_BASELINE_NAME,
    TRANSFORMER_BASELINE_NAME,
    GNNArchitecture,
    GNNTrainingConfig,
)
from pipeline.gnn_training.contract import (
    blocked_report,
    contract_digest_or_none,
    preflight_errors,
)
from pipeline.gnn_training.data import write_json, write_json_exclusive
from pipeline.gnn_training.fusion import ResidualFusionHead, late_fusion_logits
from pipeline.gnn_training.model import build_model
from pipeline.gnn_training.runtime import (
    FrozenTransformerCache,
    feature_layout_snapshot,
    frozen_artifact_locks,
    iter_gnn_batches,
    load_feature_spec,
    load_gnn_checkpoint,
    read_positive_temperature,
    require_finite_tensor,
    resolve_device,
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
from pipeline.neural_training.config import (
    EXPERIMENT_VERSION as NEURAL_EXPERIMENT_VERSION,
    TRAINING_SCHEMA_VERSION as NEURAL_TRAINING_SCHEMA_VERSION,
)
from pipeline.training_contract import load_json, sha256_file


def _checkpoint_architecture(payload: dict[str, Any]) -> GNNArchitecture:
    raw = payload.get("architecture")
    if not isinstance(raw, dict):
        raise ValueError("fusion checkpoint is missing architecture")
    allowed = {item.name for item in fields(GNNArchitecture)}
    if set(raw) - allowed:
        raise ValueError("fusion checkpoint has unknown architecture fields")
    return GNNArchitecture(**raw)


def _load_selected_models(
    config: GNNTrainingConfig,
    *,
    device: torch.device,
) -> tuple[dict[str, Any], Any, Any | None]:
    payload = torch.load(
        config.fusion_checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("fusion checkpoint payload is invalid")
    if (
        payload.get("schema_version") != FUSION_TRAINING_SCHEMA_VERSION
        or payload.get("experiment_version") != FUSION_EXPERIMENT_VERSION
        or payload.get("seed") != config.seed
    ):
        raise ValueError("fusion checkpoint does not match the configured run")
    state = load_json(config.fusion_training_state_path)
    if (
        state.get("schema_version") != FUSION_TRAINING_SCHEMA_VERSION
        or state.get("status") != "completed"
        or state.get("seed") != config.seed
        or state.get("contract_digest") != contract_digest_or_none(config)
        or state.get("selected_model") != payload.get("selected_model")
        or state.get("selected_baseline_name") != payload.get("selected_baseline_name")
        or state.get("selected_gnn_variant") != payload.get("selected_gnn_variant")
    ):
        raise ValueError("fusion checkpoint and training state are inconsistent")
    spec = load_feature_spec(config)
    if payload.get("feature_layout") != feature_layout_snapshot(spec):
        raise ValueError("fusion checkpoint feature layout does not match cache")
    architecture = _checkpoint_architecture(payload)
    selected_model = payload.get("selected_model")
    expected_baseline = (
        LATE_FUSION_BASELINE_NAME
        if selected_model == "late"
        else RESIDUAL_FUSION_BASELINE_NAME
    )
    if payload.get("selected_baseline_name") != expected_baseline:
        raise ValueError("fusion checkpoint baseline does not match selected model")
    try:
        late_weight = float(payload["late_gnn_weight"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("fusion checkpoint late weight is invalid") from error
    if not 0.0 <= late_weight <= 1.0:
        raise ValueError("fusion checkpoint late weight is outside [0, 1]")
    if selected_model == "late":
        gnn, _gnn_payload = load_gnn_checkpoint(
            config.gnn_checkpoint_path,
            spec,
            device=device,
            expected_seed=config.seed,
        )
        expected = payload.get("standalone_gnn_checkpoint_sha256")
        if expected != sha256_file(config.gnn_checkpoint_path):
            raise ValueError("standalone GNN changed after fusion selection")
        return payload, gnn, None
    if selected_model != "residual":
        raise ValueError("fusion checkpoint has an unknown selected model")
    gnn = build_model(
        spec,
        architecture,
        ablation_variant=str(payload["selected_gnn_variant"]),
    ).to(device)
    gnn.load_state_dict(payload["gnn_state_dict"], strict=True)
    head = ResidualFusionHead(
        transformer_context_dim=architecture.transformer_context_dim,
        gnn_candidate_dim=gnn.candidate_representation_dim,
        hidden_dim=architecture.fusion_hidden_dim,
        dropout=architecture.dropout,
    ).to(device)
    head.load_state_dict(payload["fusion_state_dict"], strict=True)
    for component_name, state_dict in (
        ("residual GNN", payload["gnn_state_dict"]),
        ("residual fusion head", payload["fusion_state_dict"]),
    ):
        for parameter_name, value in state_dict.items():
            require_finite_tensor(
                value,
                name=f"{component_name} parameter {parameter_name!r}",
            )
    return payload, gnn, head


@torch.no_grad()
def _write_predictions(
    config: GNNTrainingConfig,
    *,
    split: str,
    hybrid_path: Path,
    transformer_path: Path,
) -> tuple[int, int, str, str]:
    device = resolve_device(config)
    spec = load_feature_spec(config)
    payload, gnn, head = _load_selected_models(config, device=device)
    if payload.get("frozen_transformer_checkpoint_sha256") != sha256_file(
        config.neural_checkpoint_path
    ):
        raise ValueError("frozen Transformer changed after fusion selection")
    gnn.eval()
    if head is not None:
        head.eval()
    frozen_cache = FrozenTransformerCache(config)
    fusion_temperature = read_positive_temperature(
        config.fusion_calibration_path,
        expected_schema_version=FUSION_TRAINING_SCHEMA_VERSION,
        allowed_methods=frozenset({"bounded_log_grid_single_temperature_bce"}),
        training_state_path=config.fusion_training_state_path,
    )
    transformer_temperature = read_positive_temperature(
        config.neural_calibration_path,
        expected_schema_version=NEURAL_TRAINING_SCHEMA_VERSION,
        allowed_methods=frozenset({"single_temperature_validation_bce"}),
    )
    with (
        AtomicParquetWriter(
            hybrid_path,
            COMPACT_PREDICTION_SCHEMA,
        ) as hybrid_writer,
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
            output = gnn.forward_batch(moved)
            require_finite_tensor(
                output.logits,
                name="fusion GNN scoring logits",
                mask=moved.candidate_mask,
            )
            if payload["selected_model"] == "late":
                hybrid_logits = late_fusion_logits(
                    frozen.logits,
                    output.logits,
                    gnn_weight=float(payload["late_gnn_weight"]),
                    candidate_mask=moved.candidate_mask,
                )
            else:
                if head is None:  # pragma: no cover - guarded by loader
                    raise RuntimeError("residual fusion head is missing")
                hybrid_logits = head(
                    frozen_logits=frozen.logits,
                    transformer_context=frozen.context,
                    gnn_candidate_representations=(output.candidate_representations),
                    candidate_mask=moved.candidate_mask,
                )
            require_finite_tensor(
                hybrid_logits,
                name="hybrid scoring logits",
                mask=moved.candidate_mask,
            )
            write_compact_batch(
                hybrid_writer,
                batch,
                torch.sigmoid(hybrid_logits / fusion_temperature),
            )
            write_compact_batch(
                transformer_writer,
                batch,
                torch.sigmoid(frozen.logits / transformer_temperature),
            )
        hybrid_rows = hybrid_writer.commit()
        transformer_rows = transformer_writer.commit()
    return (
        hybrid_rows,
        transformer_rows,
        str(device),
        str(payload["selected_baseline_name"]),
    )


def _selection_report(
    config: GNNTrainingConfig,
    report: dict[str, Any],
    *,
    baseline_name: str,
    split: str,
) -> dict[str, Any]:
    candidate = metric_row(report, baseline_name=baseline_name, split=split)
    reference = metric_row(
        report,
        baseline_name=TRANSFORMER_BASELINE_NAME,
        split=split,
    )
    if candidate is None or reference is None:
        return {
            "schema_version": FUSION_SELECTION_SCHEMA_VERSION,
            "status": "failed_missing_validation_metrics",
            "model_frozen": False,
            "hybrid_qualified": False,
            "generated_at": report["generated_at"],
            "data_safety": {
                "report_contains_patient_rows": False,
                "report_contains_row_samples": False,
                "report_contains_identifier_values": False,
            },
        }
    decision = qualification_decision(
        candidate=candidate,
        reference=reference,
        minimum_ndcg_lift=MINIMUM_NDCG_LIFT,
        maximum_secondary_drop=MAXIMUM_SECONDARY_DROP,
    )
    qualified = bool(decision["qualified"])
    return {
        "schema_version": FUSION_SELECTION_SCHEMA_VERSION,
        "status": "frozen",
        "model_frozen": qualified,
        "hybrid_qualified": qualified,
        "final_scoring_authorized": qualified,
        "decision": ("promote_hybrid" if qualified else "retain_frozen_transformer"),
        "generated_at": report["generated_at"],
        "selected_experiment": baseline_name,
        "reference_experiment": TRANSFORMER_BASELINE_NAME,
        "selection_basis": {
            "candidate_choice": "validation_ndcg_mrr_hit_then_simpler",
            "gate_split": f"mimiciv_{split}",
            "k": 10,
            "candidate_metrics": candidate,
            "reference_metrics": reference,
        },
        **decision,
        "contract_digest": contract_digest_or_none(config),
        "seed": config.seed,
        "frozen_artifacts": frozen_artifact_locks(
            {
                "checkpoint": config.fusion_checkpoint_path,
                "calibration": config.fusion_calibration_path,
                "feature_layout": config.feature_layout_path,
                "training_state": config.fusion_training_state_path,
                "gnn_checkpoint": config.gnn_checkpoint_path,
                "gnn_selection": config.gnn_selection_report_path,
                "transformer_checkpoint": config.neural_checkpoint_path,
                "transformer_calibration": config.neural_calibration_path,
                "transformer_feature_layout": config.neural_feature_layout_path,
                "transformer_cache_manifest": (config.transformer_cache_manifest_path),
                "crossfit_manifest": config.crossfit_graph_manifest_path,
            }
        ),
        "test_disclosure": (
            "The frozen Transformer was already scored on MIMIC test earlier "
            "in the research program; a promoted hybrid final run is one-shot "
            "but the broader test split is not wholly unseen."
        ),
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "report_contains_identifier_values": False,
        },
    }


def _final_attempt_exists(config: GNNTrainingConfig) -> bool:
    return (
        config.mode == "final" and config.fusion_final_score_completion_path.is_file()
    )


def score_fusion(config: GNNTrainingConfig) -> dict[str, Any]:
    """Score the selected hybrid and apply the frozen-Transformer gate."""

    generated_at = datetime.now(UTC).isoformat()
    split = "test" if config.mode == "final" else "validation"
    errors = preflight_errors(config, stage="score-fusion")
    if _final_attempt_exists(config):
        errors.append(
            {
                "code": "final_score_already_attempted",
                "detail": (
                    "one-shot final hybrid scoring was already claimed; existing "
                    "artifacts were left unchanged"
                ),
            }
        )
    if errors:
        report = blocked_report(
            config=config,
            schema_version=FUSION_TRAINING_SCHEMA_VERSION,
            stage="score-fusion",
            generated_at=generated_at,
            errors=errors,
        )
        # Preserve the completed one-shot report rather than overwriting it.
        if not _final_attempt_exists(config):
            write_json(config.active_fusion_score_report_path, report)
        return report

    final_claimed = False
    if config.mode == "final":
        try:
            write_json_exclusive(
                config.fusion_final_score_completion_path,
                {
                    "schema_version": FUSION_TRAINING_SCHEMA_VERSION,
                    "status": "running",
                    "mode": "final",
                    "scored_split": "test",
                    "started_at": generated_at,
                    "one_shot_claim": True,
                    "data_safety": {
                        "contains_patient_rows": False,
                        "contains_identifier_values": False,
                    },
                },
            )
            final_claimed = True
        except FileExistsError:
            report = blocked_report(
                config=config,
                schema_version=FUSION_TRAINING_SCHEMA_VERSION,
                stage="score-fusion",
                generated_at=generated_at,
                errors=[
                    {
                        "code": "final_score_claim_race",
                        "detail": (
                            "another process claimed the one-shot final hybrid run"
                        ),
                    }
                ],
            )
            return report

    try:
        state = load_json(config.fusion_training_state_path)
        baseline_name = str(state["selected_baseline_name"])
    except Exception as error:  # noqa: BLE001 - aggregate fail-closed report
        report = blocked_report(
            config=config,
            schema_version=FUSION_TRAINING_SCHEMA_VERSION,
            stage="score-fusion",
            generated_at=generated_at,
            errors=[
                {
                    "code": "fusion_training_state_invalid",
                    "detail": safe_error_message(error),
                }
            ],
        )
        write_json(config.active_fusion_score_report_path, report)
        if final_claimed:
            write_json(
                config.fusion_final_score_completion_path,
                {
                    "schema_version": FUSION_TRAINING_SCHEMA_VERSION,
                    "status": report["status"],
                    "mode": "final",
                    "scored_split": "test",
                    "started_at": generated_at,
                    "completed_at": datetime.now(UTC).isoformat(),
                    "one_shot_claim": True,
                    "data_safety": {
                        "contains_patient_rows": False,
                        "contains_identifier_values": False,
                    },
                },
            )
        return report
    report = base_score_report(
        config,
        schema_version=FUSION_TRAINING_SCHEMA_VERSION,
        stage="score-fusion",
        split=split,
        baseline_name=baseline_name,
        output_path=config.fusion_score_output_path,
    )
    try:
        hybrid_predictions = config.fusion_score_root / "_hybrid_predictions.parquet"
        transformer_predictions = (
            config.fusion_score_root / "_transformer_predictions.parquet"
        )
        hybrid_rows, transformer_rows, device, checkpoint_baseline = _write_predictions(
            config,
            split=split,
            hybrid_path=hybrid_predictions,
            transformer_path=transformer_predictions,
        )
        if checkpoint_baseline != baseline_name:
            raise ValueError("fusion state and checkpoint baseline names differ")
        hybrid_candidate = config.fusion_score_root / "_hybrid_candidate_scores.parquet"
        transformer_candidate = (
            config.fusion_score_root / "_transformer_candidate_scores.parquet"
        )
        with configured_duckdb(config) as connection:
            hybrid_candidate_rows = materialize_canonical_scores(
                connection,
                config,
                predictions_path=hybrid_predictions,
                output_path=hybrid_candidate,
                split=split,
                baseline_name=baseline_name,
                baseline_version=FUSION_EXPERIMENT_VERSION,
                evaluation_version=FUSION_TRAINING_SCHEMA_VERSION,
                generated_at=report["generated_at"],
            )
            transformer_candidate_rows = materialize_canonical_scores(
                connection,
                config,
                predictions_path=transformer_predictions,
                output_path=transformer_candidate,
                split=split,
                baseline_name=TRANSFORMER_BASELINE_NAME,
                baseline_version=NEURAL_EXPERIMENT_VERSION,
                evaluation_version=FUSION_TRAINING_SCHEMA_VERSION,
                generated_at=report["generated_at"],
            )
            score_rows = combine_score_tables(
                connection,
                (hybrid_candidate, transformer_candidate),
                output_path=config.fusion_score_output_path,
            )
            append_authoritative_metrics(
                connection,
                config,
                report,
                evaluation_root=config.fusion_score_root,
            )
        if (
            hybrid_rows <= 0
            or transformer_rows <= 0
            or hybrid_candidate_rows <= 0
            or transformer_candidate_rows <= 0
            or metric_row(report, baseline_name=baseline_name, split=split) is None
            or metric_row(
                report,
                baseline_name=TRANSFORMER_BASELINE_NAME,
                split=split,
            )
            is None
        ):
            raise ValueError("hybrid scoring produced no authoritative metrics")
        report.update(
            {
                "device": device,
                "hybrid_prediction_row_count": hybrid_rows,
                "transformer_prediction_row_count": transformer_rows,
                "hybrid_candidate_score_row_count": hybrid_candidate_rows,
                "transformer_candidate_score_row_count": (transformer_candidate_rows),
                "score_row_count": score_rows,
                "reference_baseline": TRANSFORMER_BASELINE_NAME,
            }
        )
        if config.mode == "development":
            selection = _selection_report(
                config,
                report,
                baseline_name=baseline_name,
                split=split,
            )
            write_json(config.fusion_selection_report_path, selection)
            report["selection_report"] = str(config.fusion_selection_report_path)
            report["hybrid_qualified"] = selection.get("hybrid_qualified", False)
            report["decision"] = selection.get("decision")
            if selection.get("status") != "frozen":
                report["status"] = "failed"
                report["reason"] = (
                    "hybrid selection could not freeze comparable validation metrics"
                )
        else:
            report["frozen_selection_reference"] = str(
                config.fusion_selection_report_path
            )
            report["test_disclosure"] = (
                "The frozen Transformer had already been scored on MIMIC test; "
                "this was the one authorized hybrid final scoring pass."
            )
    except Exception as error:  # noqa: BLE001 - aggregate fail-closed report
        report["status"] = "failed"
        report["reason"] = safe_error_message(error)

    write_json(config.active_fusion_score_report_path, report)
    if final_claimed:
        marker: dict[str, Any] = {
            "schema_version": FUSION_TRAINING_SCHEMA_VERSION,
            "status": report.get("status", "failed"),
            "mode": "final",
            "scored_split": "test",
            "started_at": generated_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "one_shot_claim": True,
            "data_safety": {
                "contains_patient_rows": False,
                "contains_identifier_values": False,
            },
        }
        if report.get("status") == "completed":
            marker.update(
                {
                    "score_output_sha256": sha256_file(config.fusion_score_output_path),
                    "selection_report_sha256": sha256_file(
                        config.fusion_selection_report_path
                    ),
                }
            )
        write_json(config.fusion_final_score_completion_path, marker)
    return report
