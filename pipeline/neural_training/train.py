"""Reproducible training, validation, checkpointing, and calibration.

The loop optimizes the multi-positive listwise + auxiliary BCE objective on the
MIMIC-train ranking-group cache, selects the checkpoint with the best MIMIC
validation NDCG@10 (early stopping on the same metric), and fits a single
temperature on validation logits so the exported probabilities are calibrated
without changing the ranking. Every reported artifact is aggregate-only; the
checkpoint stores model weights and the resolved layout, never patient rows.

PyTorch is imported directly; this module runs only after the structured
recovery gate clears and ``uv sync --group neural`` installs the optional group.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from pipeline.neural_training.config import (
    EXPERIMENT_VERSION,
    SELECTION_K,
    TRAINING_SCHEMA_VERSION,
    NeuralTrainingConfig,
)
from pipeline.neural_training.contract import blocked_report, preflight_errors
from pipeline.neural_training.data import write_json
from pipeline.neural_training.dataset import (
    FeatureLayoutSpec,
    NeuralBatch,
    iter_batches,
    iter_shard_examples,
)
from pipeline.neural_training.losses import combined_loss
from pipeline.neural_training.metrics import RankingMetricAccumulator
from pipeline.neural_training.model import TransformerRecommender, build_model


class ModelEMA:
    """Exponential moving average of model parameters for selection/export."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if self.decay <= 0.0:
            return
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(
                parameter.detach(), alpha=1.0 - self.decay
            )

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                parameter.data.copy_(self.shadow[name])


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: NeuralTrainingConfig) -> torch.device:
    """Return the training device, honoring an explicit override."""

    if config.device:
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _use_amp(config: NeuralTrainingConfig, device: torch.device) -> bool:
    return bool(config.optimization.mixed_precision and device.type == "cuda")


def count_train_batches(config: NeuralTrainingConfig, spec: FeatureLayoutSpec) -> int:
    """Return the number of ranking-group batches in one train epoch."""

    batch_groups = max(1, int(config.optimization.batch_ranking_groups))
    batch_count = 0
    for shard_index in range(config.shard_count):
        examples = iter_shard_examples(
            config, spec, split="train", shard_index=shard_index
        )
        if not examples:
            continue
        batch_count += (len(examples) + batch_groups - 1) // batch_groups
    return batch_count


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    steps_per_epoch: int,
    max_epochs: int,
    warmup_epochs: float,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a per-step linear-warmup then cosine-decay learning-rate schedule."""

    total_steps = max(1, steps_per_epoch * max_epochs)
    warmup_steps = max(1, int(round(steps_per_epoch * warmup_epochs)))
    min_ratio = float(min_lr_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _train_one_epoch(
    model: TransformerRecommender,
    config: NeuralTrainingConfig,
    spec: FeatureLayoutSpec,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: "torch.cuda.amp.GradScaler | None",
    ema: ModelEMA | None,
    epoch: int,
) -> dict[str, float]:
    """Run one shuffled training pass and return mean loss components."""

    model.train()
    use_amp = scaler is not None
    totals = {
        "total": 0.0,
        "listwise": 0.0,
        "auxiliary": 0.0,
        "primary_positive": 0.0,
    }
    batch_count = 0
    for batch in iter_batches(
        config,
        spec,
        split="train",
        batch_groups=config.optimization.batch_ranking_groups,
        shuffle=True,
        seed=config.seed,
        epoch=epoch,
    ):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model.forward_batch(batch)
            loss = combined_loss(
                logits,
                batch.labels,
                batch.candidate_mask,
                auxiliary_weight=config.optimization.auxiliary_bce_weight,
                primary_positive_weight=config.optimization.primary_positive_weight,
                candidate_ranks=batch.candidate_rank,
            )
        if use_amp and scaler is not None:
            scaler.scale(loss.total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), config.optimization.gradient_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.total.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), config.optimization.gradient_clip_norm
            )
            optimizer.step()
        if ema is not None:
            ema.update(model)
        scheduler.step()
        totals["total"] += float(loss.total.detach())
        totals["listwise"] += float(loss.listwise.detach())
        totals["auxiliary"] += float(loss.auxiliary.detach())
        totals["primary_positive"] += float(loss.primary_positive.detach())
        batch_count += 1
    if batch_count == 0:
        raise ValueError("training split produced no batches; run `prepare` first")
    metrics = {name: value / batch_count for name, value in totals.items()}
    metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
    return metrics


@torch.no_grad()
def evaluate_split(
    model: TransformerRecommender,
    config: NeuralTrainingConfig,
    spec: FeatureLayoutSpec,
    *,
    device: torch.device,
    split: str,
    k: int = SELECTION_K,
) -> dict[str, Any]:
    """Score a split and return aggregate loss and ranking metrics."""

    model.eval()
    accumulator = RankingMetricAccumulator(k=k)
    loss_total = 0.0
    batch_count = 0
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
        loss = combined_loss(
            logits,
            moved.labels,
            moved.candidate_mask,
            auxiliary_weight=config.optimization.auxiliary_bce_weight,
            primary_positive_weight=config.optimization.primary_positive_weight,
            candidate_ranks=moved.candidate_rank,
        )
        loss_total += float(loss.total.detach())
        batch_count += 1
        _accumulate_group_metrics(accumulator, batch, logits.detach().cpu())
    summary = accumulator.summary()
    summary["loss"] = loss_total / batch_count if batch_count else 0.0
    summary["split"] = split
    summary["k"] = k
    return summary


def _accumulate_group_metrics(
    accumulator: RankingMetricAccumulator,
    batch: NeuralBatch,
    logits: "torch.Tensor",
) -> None:
    """Add each group's ranking metrics using CPU-side masked scores."""

    mask = batch.candidate_mask.cpu().numpy()
    labels = batch.labels.cpu().numpy()
    scores = logits.numpy()
    for row in range(batch.num_groups):
        valid = mask[row]
        accumulator.update(
            labels=labels[row][valid],
            scores=scores[row][valid],
            tie_breaker=batch.candidate_ranks[row].astype(np.float64),
        )


@torch.no_grad()
def fit_temperature(
    model: TransformerRecommender,
    config: NeuralTrainingConfig,
    spec: FeatureLayoutSpec,
    *,
    device: torch.device,
) -> float:
    """Fit a single positive temperature on validation logits (BCE-optimal).

    Temperature scaling is monotonic, so it leaves the ranking (and NDCG)
    unchanged while improving probability calibration for the exported scores.
    """

    logit_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    for batch in iter_batches(
        config,
        spec,
        split="validation",
        batch_groups=config.optimization.batch_ranking_groups,
        shuffle=False,
        seed=config.seed,
    ):
        moved = batch.to(device)
        logits = model.forward_batch(moved)
        mask = moved.candidate_mask
        logit_batches.append(logits[mask].detach().cpu())
        label_batches.append(moved.labels[mask].detach().cpu())
    if not logit_batches:
        return 1.0
    logits = torch.cat(logit_batches)
    labels = torch.cat(label_batches)

    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)  # type: ignore[arg-type]
    return float(torch.exp(log_temperature.detach()))


def save_checkpoint(
    model: TransformerRecommender,
    config: NeuralTrainingConfig,
    spec: FeatureLayoutSpec,
    *,
    path: Path,
    best_epoch: int,
    best_validation: dict[str, Any],
) -> None:
    """Persist model weights plus the layout and hyperparameters to reload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "state_dict": model.state_dict(),
            "architecture": asdict(config.architecture),
            "optimization": asdict(config.optimization),
            "feature_layout": {
                "numeric_columns": list(spec.numeric_columns),
                "categorical_columns": list(spec.categorical_columns),
                "max_sequence_length": spec.max_sequence_length,
                "event_vocab_size": spec.event_vocab_size,
                "condition_vocab_size": spec.condition_vocab_size,
                "candidate_vocab_size": spec.candidate_vocab_size,
                "categorical_vocab_sizes": list(spec.categorical_vocab_sizes),
                "candidate_side_features": list(spec.candidate_side_features),
            },
            "experiment_version": EXPERIMENT_VERSION,
            "seed": config.seed,
            "best_epoch": best_epoch,
            "best_validation": best_validation,
        },
        path,
    )


def load_model_from_checkpoint(
    config: NeuralTrainingConfig,
    spec: FeatureLayoutSpec,
    *,
    device: torch.device,
) -> TransformerRecommender:
    """Rebuild the model and load the best weights from ``checkpoint_path``."""

    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model = build_model(spec, config.architecture)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def train_transformer(config: NeuralTrainingConfig) -> dict[str, Any]:
    """Train the Transformer branch and return an aggregate training report."""

    generated_at = datetime.now(UTC).isoformat()
    errors = preflight_errors(config, stage="train")
    if errors:
        report = blocked_report(
            schema_version=TRAINING_SCHEMA_VERSION,
            stage="train",
            mode=config.mode,
            generated_at=generated_at,
            errors=errors,
        )
        write_json(config.training_report_path, report)
        return report

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
        report = {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "status": "failed",
            "stage": "train",
            "mode": config.mode,
            "generated_at": generated_at,
            "reason": "training split produced no batches; run `prepare` first",
            "data_safety": {
                "report_contains_patient_rows": False,
                "report_contains_row_samples": False,
            },
        }
        write_json(config.training_report_path, report)
        return report
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        steps_per_epoch=steps_per_epoch,
        max_epochs=config.optimization.max_epochs,
        warmup_epochs=config.optimization.warmup_epochs,
        min_lr_ratio=config.optimization.min_lr_ratio,
    )
    use_amp = _use_amp(config, device)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    ema: ModelEMA | None = None
    if config.optimization.ema_decay > 0.0:
        ema = ModelEMA(model, decay=config.optimization.ema_decay)
    # Shadow model used only for EMA validation / checkpoint export.
    ema_model = build_model(spec, config.architecture).to(device) if ema else None

    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    best_epoch = -1
    best_validation: dict[str, Any] = {}
    epochs_without_improvement = 0
    min_delta = float(config.optimization.early_stopping_min_delta)

    for epoch in range(config.optimization.max_epochs):
        train_metrics = _train_one_epoch(
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
        eval_model = model
        if ema is not None and ema_model is not None:
            ema.copy_to(ema_model)
            eval_model = ema_model
        validation_metrics = evaluate_split(
            eval_model, config, spec, device=device, split="validation"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["total"],
                "train_listwise_loss": train_metrics["listwise"],
                "train_auxiliary_loss": train_metrics["auxiliary"],
                "train_primary_positive_loss": train_metrics["primary_positive"],
                "learning_rate": train_metrics["learning_rate"],
                "validation_loss": validation_metrics["loss"],
                "validation_ndcg_at_10": validation_metrics["ndcg_at_k"],
                "validation_mrr_at_10": validation_metrics["mrr_at_k"],
                "validation_hit_rate_at_10": validation_metrics["hit_rate_at_k"],
                "evaluated_with_ema": ema is not None,
            }
        )
        current = float(validation_metrics["ndcg_at_k"])
        if current > best_metric + min_delta:
            best_metric = current
            best_epoch = epoch
            best_validation = validation_metrics
            epochs_without_improvement = 0
            save_checkpoint(
                eval_model,
                config,
                spec,
                path=config.checkpoint_path,
                best_epoch=best_epoch,
                best_validation=best_validation,
            )
        else:
            epochs_without_improvement += 1
            if (
                epochs_without_improvement
                >= config.optimization.early_stopping_patience
            ):
                break

    if best_epoch < 0:
        report = {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "status": "failed",
            "stage": "train",
            "mode": config.mode,
            "generated_at": generated_at,
            "reason": "no epoch produced a valid validation metric",
            "data_safety": {
                "report_contains_patient_rows": False,
                "report_contains_row_samples": False,
            },
        }
        write_json(config.training_report_path, report)
        return report

    best_model = load_model_from_checkpoint(config, spec, device=device)
    temperature = fit_temperature(best_model, config, spec, device=device)
    write_json(
        config.calibration_path,
        {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "method": "single_temperature_validation_bce",
            "temperature": temperature,
            "fit_split": "mimiciv_validation",
            "generated_at": generated_at,
        },
    )

    report = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "status": "completed",
        "stage": "train",
        "mode": config.mode,
        "generated_at": generated_at,
        "device": str(device),
        "mixed_precision": use_amp,
        "seed": config.seed,
        "experiment_version": EXPERIMENT_VERSION,
        "selection_metric": "mimic_validation_ndcg_at_10",
        "best_epoch": best_epoch,
        "best_validation_metrics": best_validation,
        "temperature": temperature,
        "schedule": {
            "steps_per_epoch": steps_per_epoch,
            "warmup_epochs": config.optimization.warmup_epochs,
            "min_lr_ratio": config.optimization.min_lr_ratio,
            "ema_decay": config.optimization.ema_decay,
            "early_stopping_min_delta": config.optimization.early_stopping_min_delta,
        },
        "model": {
            "parameter_count": best_model.parameter_count(),
            "architecture": asdict(config.architecture),
            "optimization": asdict(config.optimization),
        },
        "epoch_history": history,
        "artifacts": {
            "checkpoint": str(config.checkpoint_path),
            "calibration": str(config.calibration_path),
            "feature_layout": str(config.feature_layout_path),
        },
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "checkpoint_contains_patient_rows": False,
        },
    }
    write_json(config.training_report_path, report)
    write_json(
        config.training_state_path,
        {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "best_epoch": best_epoch,
            "best_validation_metrics": best_validation,
            "temperature": temperature,
            "seed": config.seed,
            "generated_at": generated_at,
        },
    )
    return report
