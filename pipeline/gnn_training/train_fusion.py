"""Train OOF late fusion and a separate residual GNN/fusion copy."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn

from pipeline.extract_utils import parquet_scan, safe_error_message
from pipeline.gnn_training.config import (
    FUSION_EXPERIMENT_VERSION,
    FUSION_TRAINING_SCHEMA_VERSION,
    GNN_TRAINING_SCHEMA_VERSION,
    LATE_FUSION_BASELINE_NAME,
    RESIDUAL_FUSION_BASELINE_NAME,
    SELECTION_K,
    GNNTrainingConfig,
)
from pipeline.gnn_training.contract import (
    blocked_report,
    contract_digest_or_none,
    preflight_errors,
)
from pipeline.gnn_training.data import configure_connection, write_json
from pipeline.gnn_training.fusion import (
    DEFAULT_LATE_FUSION_WEIGHTS,
    ResidualFusionHead,
    late_fusion_logits,
)
from pipeline.gnn_training.model import GNNRecommender, build_model
from pipeline.gnn_training.runtime import (
    MAX_AMP_OVERFLOW_RETRIES_PER_BATCH,
    FrozenTransformerCache,
    TemperatureGrid,
    atomic_torch_save,
    backward_optimizer_step,
    feature_layout_snapshot,
    iter_gnn_batches,
    load_feature_spec,
    load_gnn_checkpoint,
    require_finite_loss,
    require_finite_tensor,
    resolve_device,
    set_global_seed,
    use_amp,
)
from pipeline.neural_training.losses import combined_loss
from pipeline.neural_training.metrics import RankingMetricAccumulator
from pipeline.training_contract import load_json, sha256_file


def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _update_metrics(
    accumulator: RankingMetricAccumulator,
    batch: Any,
    logits: torch.Tensor,
) -> None:
    require_finite_tensor(
        logits,
        name="fusion metric logits",
        mask=batch.candidate_mask.to(logits.device),
    )
    scores = logits.detach().cpu().numpy()
    labels = batch.labels.cpu().numpy()
    mask = batch.candidate_mask.cpu().numpy()
    ranks = batch.candidate_rank.cpu().numpy()
    for row in range(batch.num_groups):
        valid = mask[row]
        accumulator.update(
            labels=labels[row][valid],
            scores=scores[row][valid],
            tie_breaker=ranks[row][valid].astype(np.float64),
        )


def _zscore(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError("fusion meta-fit logits contain non-finite values")
    output = np.zeros_like(values, dtype=np.float64)
    selected = values[finite].astype(np.float64)
    scale = float(selected.std())
    if scale >= 1e-6:
        output[finite] = (selected - float(selected.mean())) / scale
    return output


def _select_late_weight(config: GNNTrainingConfig) -> tuple[float, dict[str, Any]]:
    """Select alpha from bounded selected-GNN OOF/frozen-Transformer joins."""

    if not config.gnn_oof_predictions_path.is_file():
        raise FileNotFoundError("selected GNN OOF predictions are missing")
    frozen_glob = (
        config.frozen_transformer_cache_root / "candidate_logits" / "**" / "*.parquet"
    )
    accumulators = {
        float(weight): RankingMetricAccumulator(k=SELECTION_K)
        for weight in DEFAULT_LATE_FUSION_WEIGHTS
    }
    joined_rows = 0

    def update_group(group: pd.DataFrame) -> None:
        nonlocal joined_rows
        labels = group["label_prescribed"].to_numpy(dtype=np.float32)
        ranks = group["candidate_rank"].to_numpy(dtype=np.float64)
        if len(group) <= 0 or len(set(ranks.tolist())) != len(group):
            raise ValueError("late-fusion train group has invalid candidate ranks")
        frozen_z = _zscore(group["frozen_transformer_logit"].to_numpy(dtype=np.float64))
        gnn_z = _zscore(group["gnn_logit"].to_numpy(dtype=np.float64))
        for weight, accumulator in accumulators.items():
            accumulator.update(
                labels=labels,
                scores=(1.0 - weight) * frozen_z + weight * gnn_z,
                tie_breaker=ranks,
            )
        joined_rows += len(group)

    with duckdb.connect(database=":memory:") as connection:
        configure_connection(config, connection)
        locked = parquet_scan(config.patient_condition_medication_path)
        oof = parquet_scan(config.gnn_oof_predictions_path)
        coverage = connection.execute(
            f"""
WITH locked_candidates AS (
    SELECT
        source,
        split,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        CAST(candidate_rank AS BIGINT) AS candidate_rank,
        CAST(label_prescribed AS BOOLEAN) AS label_prescribed
    FROM {locked}
    WHERE source = 'mimiciv' AND split = 'train'
),
oof_candidates AS (
    SELECT
        source,
        split,
        ranking_group_id,
        index_condition_token,
        candidate_medication_token,
        CAST(candidate_rank AS BIGINT) AS candidate_rank,
        CAST(label_prescribed AS BOOLEAN) AS label_prescribed
    FROM {oof}
),
locked_only AS (
    SELECT * FROM locked_candidates
    EXCEPT ALL
    SELECT * FROM oof_candidates
),
oof_only AS (
    SELECT * FROM oof_candidates
    EXCEPT ALL
    SELECT * FROM locked_candidates
)
SELECT
    (SELECT COUNT(*) FROM locked_candidates) AS locked_count,
    (SELECT COUNT(*) FROM oof_candidates) AS oof_count,
    (SELECT COUNT(*) FROM locked_only)
        + (SELECT COUNT(*) FROM oof_only) AS mismatch_count,
    (
        SELECT COUNT(*)
        FROM {oof}
        WHERE source <> 'mimiciv'
            OR split <> 'train'
            OR patient_fold_id < 0
            OR patient_fold_id >= {int(config.fold_count)}
            OR shard_id < 0
            OR shard_id >= {int(config.shard_count)}
            OR gnn_logit IS NULL
            OR NOT isfinite(gnn_logit)
    ) AS invalid_count
"""
        ).fetchone()
        if (
            coverage is None
            or int(coverage[0]) <= 0
            or int(coverage[0]) != int(coverage[1])
            or int(coverage[2]) != 0
            or int(coverage[3]) != 0
        ):
            raise ValueError(
                "selected GNN OOF predictions do not exactly match train candidates"
            )

        reader = connection.sql(
            f"""
SELECT
    oof.ranking_group_id,
    oof.candidate_rank,
    oof.label_prescribed,
    oof.gnn_logit,
    frozen.frozen_transformer_logit
FROM {oof} AS oof
INNER JOIN {parquet_scan(frozen_glob)} AS frozen
    ON oof.source = frozen.source
    AND oof.split = frozen.split
    AND oof.ranking_group_id = frozen.ranking_group_id
    AND oof.index_condition_token = frozen.index_condition_token
    AND oof.candidate_medication_token = frozen.candidate_medication_token
    AND oof.candidate_rank = frozen.candidate_rank
WHERE oof.source = 'mimiciv'
    AND oof.split = 'train'
ORDER BY oof.ranking_group_id, oof.candidate_rank
"""
        ).to_arrow_reader(batch_size=100_000)
        pending = pd.DataFrame()
        for record_batch in reader:
            frame = record_batch.to_pandas()
            if not pending.empty:
                frame = pd.concat((pending, frame), ignore_index=True)
                pending = pd.DataFrame()
            if frame.empty:
                continue
            last_group = frame["ranking_group_id"].iloc[-1]
            complete = frame[frame["ranking_group_id"] != last_group]
            pending = frame[frame["ranking_group_id"] == last_group]
            for _group_id, group in complete.groupby(
                "ranking_group_id",
                sort=False,
            ):
                update_group(group)
        if not pending.empty:
            update_group(pending)
    if joined_rows <= 0:
        raise ValueError("OOF GNN and frozen Transformer caches have no joined rows")
    if joined_rows != int(coverage[0]):
        raise ValueError(
            "frozen Transformer and GNN OOF candidate coverage is not exact"
        )
    summaries = {
        str(weight): accumulator.summary()
        for weight, accumulator in accumulators.items()
    }
    selected = max(
        sorted(accumulators),
        key=lambda weight: (
            float(summaries[str(weight)]["ndcg_at_k"]),
            float(summaries[str(weight)]["mrr_at_k"]),
            float(summaries[str(weight)]["hit_rate_at_k"]),
            -weight,
        ),
    )
    return selected, {
        "joined_candidate_row_count": joined_rows,
        "candidate_weights": summaries,
        "selected_gnn_weight": selected,
        "selection_scope": "mimiciv_train_meta_fit",
        "frozen_transformer_policy": (
            "The Transformer checkpoint was frozen before Phase D and is a "
            "fixed train-derived covariate; the GNN side is patient-fold OOF. "
            "These train metrics fit alpha and are not promotion evidence."
        ),
    }


@torch.no_grad()
def _evaluate_residual(
    gnn: GNNRecommender,
    head: ResidualFusionHead,
    config: GNNTrainingConfig,
    spec: Any,
    *,
    device: torch.device,
    shards_root: Path,
    split: str,
    include_fold_ids: frozenset[int] | None = None,
    temperature_grid: TemperatureGrid | None = None,
) -> dict[str, Any]:
    gnn.eval()
    head.eval()
    frozen_cache = FrozenTransformerCache(config)
    accumulator = RankingMetricAccumulator(k=SELECTION_K)
    group_count = 0
    zero_positive = 0
    for batch in iter_gnn_batches(
        config,
        spec,
        split=split,
        shards_root=shards_root,
        shuffle=False,
        include_fold_ids=include_fold_ids,
    ):
        moved = batch.to(device)
        frozen = frozen_cache.align(batch, device=device)
        output = gnn.forward_batch(moved)
        logits = head(
            frozen_logits=frozen.logits,
            transformer_context=frozen.context,
            gnn_candidate_representations=output.candidate_representations,
            candidate_mask=moved.candidate_mask,
        )
        _update_metrics(accumulator, batch, logits)
        if temperature_grid is not None:
            temperature_grid.update(logits, moved.labels, moved.candidate_mask)
        group_count += batch.num_groups
        zero_positive += int((batch.labels.sum(dim=1) <= 0).sum().item())
    summary = accumulator.summary()
    summary.update(
        {
            "ranking_group_count": group_count,
            "zero_positive_ranking_group_count": zero_positive,
        }
    )
    return summary


def _refit_residual(
    config: GNNTrainingConfig,
    *,
    selected_variant: str,
    epochs: int,
    device: torch.device,
) -> tuple[GNNRecommender, ResidualFusionHead]:
    spec = load_feature_spec(config)
    set_global_seed(config.seed + 1000)
    gnn = build_model(
        spec,
        config.architecture,
        ablation_variant=selected_variant,
    ).to(device)
    head = ResidualFusionHead(
        transformer_context_dim=config.architecture.transformer_context_dim,
        gnn_candidate_dim=gnn.candidate_representation_dim,
        hidden_dim=config.architecture.fusion_hidden_dim,
        dropout=config.architecture.dropout,
    ).to(device)
    trainable_model = nn.ModuleList((gnn, head))
    optimizer = torch.optim.AdamW(
        trainable_model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    amp_enabled = use_amp(config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    frozen_cache = FrozenTransformerCache(config)
    for epoch in range(epochs):
        gnn.train()
        head.train()
        batch_count = 0
        for batch in iter_gnn_batches(
            config,
            spec,
            split="train",
            shards_root=config.shards_root,
            shuffle=True,
            epoch=epoch,
            require_positive=True,
        ):
            moved = batch.to(device)
            frozen = frozen_cache.align(batch, device=device)
            batch_overflow_retries = 0
            while True:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    output = gnn.forward_batch(moved)
                    logits = head(
                        frozen_logits=frozen.logits,
                        transformer_context=frozen.context,
                        gnn_candidate_representations=(
                            output.candidate_representations
                        ),
                        candidate_mask=moved.candidate_mask,
                    )
                    loss = combined_loss(
                        logits,
                        moved.labels,
                        moved.candidate_mask,
                        auxiliary_weight=(config.optimization.auxiliary_bce_weight),
                        primary_positive_weight=(
                            config.optimization.primary_positive_weight
                        ),
                        candidate_ranks=moved.candidate_rank,
                    )
                require_finite_tensor(
                    logits,
                    name="residual-fusion training logits",
                    mask=moved.candidate_mask,
                )
                require_finite_loss(loss)
                step_result = backward_optimizer_step(
                    loss=loss.total,
                    model=trainable_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    gradient_clip_norm=config.optimization.gradient_clip_norm,
                )
                if step_result.optimizer_step_applied:
                    break
                batch_overflow_retries += 1
                if batch_overflow_retries >= MAX_AMP_OVERFLOW_RETRIES_PER_BATCH:
                    names = ", ".join(step_result.nonfinite_parameter_names[:8])
                    detail = f"; parameters={names}" if names else ""
                    raise FloatingPointError(
                        "residual-fusion mixed-precision gradients remained "
                        "non-finite after "
                        f"{MAX_AMP_OVERFLOW_RETRIES_PER_BATCH} loss-scale "
                        f"backoffs for one batch{detail}"
                    )
            batch_count += 1
        if batch_count == 0:
            raise ValueError("residual full refit has no positive train groups")
    return gnn, head


@torch.no_grad()
def _evaluate_late(
    gnn: GNNRecommender,
    config: GNNTrainingConfig,
    spec: Any,
    *,
    gnn_weight: float,
    device: torch.device,
    temperature_grid: TemperatureGrid | None = None,
) -> dict[str, Any]:
    gnn.eval()
    frozen_cache = FrozenTransformerCache(config)
    accumulator = RankingMetricAccumulator(k=SELECTION_K)
    for batch in iter_gnn_batches(
        config,
        spec,
        split="validation",
        shards_root=config.shards_root,
        shuffle=False,
    ):
        moved = batch.to(device)
        frozen = frozen_cache.align(batch, device=device)
        output = gnn.forward_batch(moved)
        logits = late_fusion_logits(
            frozen.logits,
            output.logits,
            gnn_weight=gnn_weight,
            candidate_mask=moved.candidate_mask,
        )
        _update_metrics(accumulator, batch, logits)
        if temperature_grid is not None:
            temperature_grid.update(logits, moved.labels, moved.candidate_mask)
    return accumulator.summary()


def _selected_variant(config: GNNTrainingConfig) -> tuple[str, int]:
    payload = load_json(config.gnn_training_state_path)
    if (
        payload.get("schema_version") != GNN_TRAINING_SCHEMA_VERSION
        or payload.get("status") != "completed"
        or payload.get("seed") != config.seed
        or payload.get("contract_digest") != contract_digest_or_none(config)
    ):
        raise ValueError("GNN training state does not match the configured run")
    variant = payload.get("selected_variant")
    if not isinstance(variant, str) or not variant:
        raise ValueError("GNN training state does not contain selected_variant")
    try:
        refit_epochs = int(payload["refit_epochs"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("GNN training state does not contain refit_epochs") from error
    if refit_epochs <= 0:
        raise ValueError("GNN refit epoch count must be positive")
    return variant, refit_epochs


def _candidate_key(metrics: dict[str, Any], *, prefer_simple: bool) -> tuple[Any, ...]:
    return (
        float(metrics["ndcg_at_k"]),
        float(metrics["mrr_at_k"]),
        float(metrics["hit_rate_at_k"]),
        1 if prefer_simple else 0,
    )


def train_fusion(config: GNNTrainingConfig) -> dict[str, Any]:
    """Select late/residual fusion and freeze a full-train refit checkpoint."""

    generated_at = datetime.now(UTC).isoformat()
    errors = preflight_errors(config, stage="train-fusion")
    if errors:
        report = blocked_report(
            config=config,
            schema_version=FUSION_TRAINING_SCHEMA_VERSION,
            stage="train-fusion",
            generated_at=generated_at,
            errors=errors,
        )
        write_json(config.fusion_training_report_path, report)
        return report

    report: dict[str, Any] = {
        "schema_version": FUSION_TRAINING_SCHEMA_VERSION,
        "status": "running",
        "stage": "train-fusion",
        "mode": config.mode,
        "generated_at": generated_at,
        "seed": config.seed,
        "experiment_version": FUSION_EXPERIMENT_VERSION,
        "transformer_gradient_policy": "frozen_detached_no_optimizer_ownership",
        "standalone_gnn_mutation_policy": "immutable_separate_residual_copy",
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "report_contains_identifier_values": False,
        },
    }
    try:
        device = resolve_device(config)
        selected_variant, residual_epochs = _selected_variant(config)
        spec = load_feature_spec(config)
        standalone_gnn, gnn_payload = load_gnn_checkpoint(
            config.gnn_checkpoint_path,
            spec,
            device=device,
            expected_seed=config.seed,
        )
        if gnn_payload.get("ablation_variant") != selected_variant:
            raise ValueError(
                "GNN checkpoint and training state selected variants differ"
            )
        late_weight, late_meta_fit = _select_late_weight(config)

        residual_gnn, residual_head = _refit_residual(
            config,
            selected_variant=selected_variant,
            epochs=residual_epochs,
            device=device,
        )
        late_metrics = _evaluate_late(
            standalone_gnn,
            config,
            spec,
            gnn_weight=late_weight,
            device=device,
        )
        residual_metrics = _evaluate_residual(
            residual_gnn,
            residual_head,
            config,
            spec,
            device=device,
            shards_root=config.shards_root,
            split="validation",
        )
        selected_model = (
            "late"
            if _candidate_key(late_metrics, prefer_simple=True)
            >= _candidate_key(residual_metrics, prefer_simple=False)
            else "residual"
        )

        grid = TemperatureGrid(device)
        if selected_model == "late":
            _evaluate_late(
                standalone_gnn,
                config,
                spec,
                gnn_weight=late_weight,
                device=device,
                temperature_grid=grid,
            )
        else:
            _evaluate_residual(
                residual_gnn,
                residual_head,
                config,
                spec,
                device=device,
                shards_root=config.shards_root,
                split="validation",
                temperature_grid=grid,
            )
        temperature = grid.best()
        checkpoint: dict[str, Any] = {
            "schema_version": FUSION_TRAINING_SCHEMA_VERSION,
            "selected_model": selected_model,
            "selected_baseline_name": (
                LATE_FUSION_BASELINE_NAME
                if selected_model == "late"
                else RESIDUAL_FUSION_BASELINE_NAME
            ),
            "late_gnn_weight": late_weight,
            "selected_gnn_variant": selected_variant,
            "architecture": asdict(config.architecture),
            "feature_layout": feature_layout_snapshot(spec),
            "residual_refit_epochs": residual_epochs,
            "standalone_gnn_checkpoint_sha256": sha256_file(config.gnn_checkpoint_path),
            "frozen_transformer_checkpoint_sha256": sha256_file(
                config.neural_checkpoint_path
            ),
            "experiment_version": FUSION_EXPERIMENT_VERSION,
            "seed": config.seed,
        }
        if selected_model == "residual":
            checkpoint["gnn_state_dict"] = _cpu_state(residual_gnn)
            checkpoint["fusion_state_dict"] = _cpu_state(residual_head)
        atomic_torch_save(checkpoint, config.fusion_checkpoint_path)
        write_json(
            config.fusion_calibration_path,
            {
                "schema_version": FUSION_TRAINING_SCHEMA_VERSION,
                "method": "bounded_log_grid_single_temperature_bce",
                "temperature": temperature,
                "fit_split": "mimiciv_validation",
                "fit_after_ranking_selection": True,
                "generated_at": generated_at,
            },
        )
        state = {
            "schema_version": FUSION_TRAINING_SCHEMA_VERSION,
            "status": "completed",
            "seed": config.seed,
            "selected_model": selected_model,
            "selected_baseline_name": checkpoint["selected_baseline_name"],
            "late_gnn_weight": late_weight,
            "selected_gnn_variant": selected_variant,
            "residual_refit_epochs": residual_epochs,
            "temperature": temperature,
            "contract_digest": contract_digest_or_none(config),
            "generated_at": generated_at,
        }
        write_json(config.fusion_training_state_path, state)
        report.update(
            {
                "status": "completed",
                "device": str(device),
                "mixed_precision": use_amp(config, device),
                "selected_gnn_variant": selected_variant,
                "late_fusion_train_meta_fit": late_meta_fit,
                "residual_training_protocol": (
                    "full_mimic_train_fixed_epoch_count_from_frozen_standalone_"
                    "gnn_selection"
                ),
                "residual_refit_epochs": residual_epochs,
                "validation_candidate_metrics": {
                    LATE_FUSION_BASELINE_NAME: late_metrics,
                    RESIDUAL_FUSION_BASELINE_NAME: residual_metrics,
                },
                "selected_model": selected_model,
                "selected_baseline_name": checkpoint["selected_baseline_name"],
                "temperature": temperature,
                "artifacts": {
                    "checkpoint": str(config.fusion_checkpoint_path),
                    "calibration": str(config.fusion_calibration_path),
                    "training_state": str(config.fusion_training_state_path),
                },
                "contract_digest": contract_digest_or_none(config),
            }
        )
    except Exception as error:  # noqa: BLE001 - aggregate fail-closed report
        report["status"] = "failed"
        report["reason"] = safe_error_message(error)

    write_json(config.fusion_training_report_path, report)
    return report
