"""Shared bounded runtime helpers for GNN and fusion stages.

This module is imported only by train/score commands, so importing PyTorch here
does not weaken the package's lightweight CLI/config boundary.  All iterators
retain at most one prepared graph shard and one frozen-Transformer shard.
Restricted ranking-group keys remain in ignored local artifacts and are never
returned in aggregate reports.
"""

from __future__ import annotations

import copy
import json
import math
import random
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch import nn

from pipeline.gnn_training.config import (
    GNN_EXPERIMENT_VERSION,
    GNN_TRAINING_SCHEMA_VERSION,
    SELECTION_K,
    GNNArchitecture,
    GNNTrainingConfig,
)
from pipeline.gnn_training.dataset import (
    GNNBatch,
    GNNFeatureLayoutSpec,
    iter_batches,
)
from pipeline.gnn_training.model import GNNRecommender, build_model
from pipeline.neural_training.losses import combined_loss
from pipeline.neural_training.metrics import RankingMetricAccumulator
from pipeline.training_contract import sha256_file


@dataclass(frozen=True)
class FitResult:
    """Best fold fit, stored without patient-level examples."""

    state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_metrics: dict[str, Any]
    history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FrozenBatch:
    """Frozen Transformer tensors aligned to one GNN batch."""

    context: torch.Tensor
    logits: torch.Tensor


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch deterministically."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def require_finite_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    mask: torch.Tensor | None = None,
) -> None:
    """Reject non-finite model values before metrics, calibration, or updates."""

    selected = tensor if mask is None else tensor[mask]
    if selected.numel() > 0 and not bool(torch.isfinite(selected).all().item()):
        raise FloatingPointError(f"{name} contains non-finite values")


def require_finite_loss(loss: Any) -> None:
    """Reject any non-finite component returned by the ranking loss."""

    for name in ("total", "listwise", "auxiliary", "primary_positive"):
        value = getattr(loss, name)
        require_finite_tensor(value, name=f"{name} loss")


def resolve_device(config: GNNTrainingConfig) -> torch.device:
    """Resolve the configured device with a CUDA-first default."""

    if config.device:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def use_amp(config: GNNTrainingConfig, device: torch.device) -> bool:
    return bool(config.optimization.mixed_precision and device.type == "cuda")


def fold_shards_root(config: GNNTrainingConfig, fold_index: int) -> Path:
    return config.fold_graph_root(fold_index) / "cache" / "shards"


def load_feature_spec(
    config: GNNTrainingConfig,
    *,
    fold_index: int | None = None,
) -> GNNFeatureLayoutSpec:
    path = (
        config.feature_layout_path
        if fold_index is None
        else config.fold_feature_layout_path(fold_index)
    )
    return GNNFeatureLayoutSpec.from_json(path)


def _state_dict_on_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _architecture_from_payload(payload: dict[str, Any]) -> GNNArchitecture:
    raw = payload.get("architecture")
    if not isinstance(raw, dict):
        raise ValueError("GNN checkpoint is missing architecture metadata")
    allowed = {item.name for item in fields(GNNArchitecture)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            "GNN checkpoint has unknown architecture fields: " + ", ".join(unknown)
        )
    try:
        return GNNArchitecture(**raw)
    except (TypeError, ValueError) as error:
        raise ValueError("GNN checkpoint architecture is invalid") from error


def feature_layout_snapshot(spec: GNNFeatureLayoutSpec) -> dict[str, Any]:
    """Return the checkpoint-relevant graph layout without concept values."""

    return {
        "schema_version": spec.schema_version,
        "concept_vocab_size": spec.concept_vocab_size,
        "node_type_vocabulary": list(spec.node_type_vocabulary),
        "node_role_vocabulary": list(spec.node_role_vocabulary),
        "relation_vocabulary": list(spec.relation_vocabulary),
        "pad_index": spec.pad_index,
        "unk_index": spec.unk_index,
        "scope": spec.scope,
        "selection_eligible": spec.selection_eligible,
        "shard_count": spec.shard_count,
        "held_out_fold_index": spec.held_out_fold_index,
    }


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Atomically replace one owned checkpoint file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_gnn_checkpoint(
    model: GNNRecommender,
    config: GNNTrainingConfig,
    spec: GNNFeatureLayoutSpec,
    *,
    path: Path,
    ablation_variant: str,
    epochs: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist weights plus immutable reconstruction metadata."""

    payload: dict[str, Any] = {
        "schema_version": GNN_TRAINING_SCHEMA_VERSION,
        "state_dict": _state_dict_on_cpu(model),
        "architecture": asdict(config.architecture),
        "optimization": asdict(config.optimization),
        "feature_layout": feature_layout_snapshot(spec),
        "experiment_version": GNN_EXPERIMENT_VERSION,
        "seed": config.seed,
        "ablation_variant": ablation_variant,
        "epochs": int(epochs),
    }
    if metadata:
        payload["metadata"] = metadata
    atomic_torch_save(payload, path)


def load_gnn_checkpoint(
    path: Path,
    spec: GNNFeatureLayoutSpec,
    *,
    device: torch.device,
    expected_seed: int | None = None,
) -> tuple[GNNRecommender, dict[str, Any]]:
    """Rebuild a GNN from checkpoint metadata, rejecting layout drift."""

    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("GNN checkpoint payload is invalid")
    if payload.get("schema_version") != GNN_TRAINING_SCHEMA_VERSION:
        raise ValueError("GNN checkpoint schema version does not match contract")
    if payload.get("experiment_version") != GNN_EXPERIMENT_VERSION:
        raise ValueError("GNN checkpoint experiment version does not match contract")
    if expected_seed is not None and payload.get("seed") != expected_seed:
        raise ValueError("GNN checkpoint seed does not match configured run")
    for name, value in payload["state_dict"].items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"GNN checkpoint parameter {name!r} is not a tensor")
        require_finite_tensor(value, name=f"GNN checkpoint parameter {name!r}")
    expected = feature_layout_snapshot(spec)
    recorded = payload.get("feature_layout")
    if recorded != expected:
        raise ValueError("GNN checkpoint feature layout does not match cache")
    architecture = _architecture_from_payload(payload)
    variant = str(payload.get("ablation_variant", "full"))
    model = build_model(spec, architecture, ablation_variant=variant)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device), payload


def _accumulate_metrics(
    accumulator: RankingMetricAccumulator,
    batch: GNNBatch,
    logits: torch.Tensor,
) -> None:
    require_finite_tensor(
        logits,
        name="GNN metric logits",
        mask=batch.candidate_mask.to(logits.device),
    )
    mask = batch.candidate_mask.cpu().numpy()
    labels = batch.labels.cpu().numpy()
    scores = logits.detach().cpu().numpy()
    ranks = batch.candidate_rank.cpu().numpy()
    for row in range(batch.num_groups):
        valid = mask[row]
        accumulator.update(
            labels=labels[row][valid],
            scores=scores[row][valid],
            tie_breaker=ranks[row][valid].astype(np.float64),
        )


@torch.no_grad()
def evaluate_gnn(
    model: GNNRecommender,
    config: GNNTrainingConfig,
    spec: GNNFeatureLayoutSpec,
    *,
    device: torch.device,
    split: str,
    shards_root: Path,
    include_fold_ids: frozenset[int] | None = None,
    exclude_fold_ids: frozenset[int] | None = None,
    k: int = SELECTION_K,
    batch_callback: Callable[[GNNBatch, torch.Tensor, Any], None] | None = None,
) -> dict[str, Any]:
    """Evaluate one bounded cache scope and optionally stream predictions."""

    model.eval()
    accumulator = RankingMetricAccumulator(k=k)
    loss_totals = {
        "total": 0.0,
        "listwise": 0.0,
        "auxiliary": 0.0,
        "primary_positive": 0.0,
    }
    batch_count = 0
    group_count = 0
    zero_positive_groups = 0
    for batch in iter_batches(
        config,
        spec,
        split=split,
        batch_groups=config.optimization.batch_ranking_groups,
        shuffle=False,
        seed=config.seed,
        shards_root=shards_root,
        include_fold_ids=include_fold_ids,
        exclude_fold_ids=exclude_fold_ids,
    ):
        moved = batch.to(device)
        output = model.forward_batch(moved)
        require_finite_tensor(
            output.logits,
            name="GNN evaluation logits",
            mask=moved.candidate_mask,
        )
        loss = combined_loss(
            output.logits,
            moved.labels,
            moved.candidate_mask,
            auxiliary_weight=config.optimization.auxiliary_bce_weight,
            primary_positive_weight=config.optimization.primary_positive_weight,
            candidate_ranks=moved.candidate_rank,
        )
        require_finite_loss(loss)
        for name in loss_totals:
            loss_totals[name] += float(getattr(loss, name).detach())
        _accumulate_metrics(accumulator, batch, output.logits)
        if batch_callback is not None:
            batch_callback(batch, output.logits.detach().cpu(), output)
        batch_count += 1
        group_count += batch.num_groups
        zero_positive_groups += int((batch.labels.sum(dim=1) <= 0).sum().item())
    summary = accumulator.summary()
    summary.update(
        {
            "split": split,
            "k": k,
            "batch_count": batch_count,
            "ranking_group_count": group_count,
            "zero_positive_ranking_group_count": zero_positive_groups,
        }
    )
    for name, total in loss_totals.items():
        summary[f"{name}_loss"] = total / batch_count if batch_count else 0.0
    return summary


def _train_gnn_epoch(
    model: GNNRecommender,
    config: GNNTrainingConfig,
    spec: GNNFeatureLayoutSpec,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    shards_root: Path,
    exclude_fold_ids: frozenset[int] | None,
) -> dict[str, float]:
    model.train()
    totals = {
        "total": 0.0,
        "listwise": 0.0,
        "auxiliary": 0.0,
        "primary_positive": 0.0,
    }
    batch_count = 0
    amp_enabled = scaler is not None
    for batch in iter_batches(
        config,
        spec,
        split="train",
        batch_groups=config.optimization.batch_ranking_groups,
        shuffle=True,
        seed=config.seed,
        epoch=epoch,
        shards_root=shards_root,
        exclude_fold_ids=exclude_fold_ids,
        require_positive=True,
    ):
        moved = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model.forward_batch(moved)
            loss = combined_loss(
                output.logits,
                moved.labels,
                moved.candidate_mask,
                auxiliary_weight=config.optimization.auxiliary_bce_weight,
                primary_positive_weight=config.optimization.primary_positive_weight,
                candidate_ranks=moved.candidate_rank,
            )
        require_finite_tensor(
            output.logits,
            name="GNN training logits",
            mask=moved.candidate_mask,
        )
        require_finite_loss(loss)
        if scaler is not None:
            scaler.scale(loss.total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(),
                config.optimization.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.total.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                config.optimization.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
        for name in totals:
            totals[name] += float(getattr(loss, name).detach())
        batch_count += 1
    if batch_count == 0:
        raise ValueError("GNN fitting scope produced no positive ranking groups")
    return {name: value / batch_count for name, value in totals.items()}


def fit_crossfit_gnn(
    config: GNNTrainingConfig,
    spec: GNNFeatureLayoutSpec,
    *,
    held_out_fold: int,
    ablation_variant: str,
    device: torch.device,
) -> FitResult:
    """Fit one ablation against exactly one fold-excluded graph cache."""

    if spec.held_out_fold_index != held_out_fold or not spec.selection_eligible:
        raise ValueError("cross-fit feature layout does not match held-out fold")
    set_global_seed(config.seed + held_out_fold)
    model = build_model(
        spec,
        config.architecture,
        ablation_variant=ablation_variant,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    amp_enabled = use_amp(config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    shards = fold_shards_root(config, held_out_fold)
    best_metric = float("-inf")
    best_epoch = -1
    best_metrics: dict[str, Any] = {}
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    stale_epochs = 0

    for epoch in range(config.optimization.max_epochs):
        train_metrics = _train_gnn_epoch(
            model,
            config,
            spec,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            shards_root=shards,
            exclude_fold_ids=frozenset({held_out_fold}),
        )
        held_out = evaluate_gnn(
            model,
            config,
            spec,
            device=device,
            split="train",
            shards_root=shards,
            include_fold_ids=frozenset({held_out_fold}),
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}_loss": value for key, value in train_metrics.items()},
            "held_out_ndcg_at_10": held_out["ndcg_at_k"],
            "held_out_mrr_at_10": held_out["mrr_at_k"],
            "held_out_hit_rate_at_10": held_out["hit_rate_at_k"],
            "held_out_positive_ranking_group_count": held_out[
                "positive_ranking_group_count"
            ],
        }
        history.append(row)
        current = float(held_out["ndcg_at_k"])
        if held_out[
            "positive_ranking_group_count"
        ] > 0 and current > best_metric + float(
            config.optimization.early_stopping_min_delta
        ):
            best_metric = current
            best_epoch = epoch
            best_metrics = held_out
            best_state = _state_dict_on_cpu(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.optimization.early_stopping_patience:
                break
    if best_state is None or best_epoch < 0:
        raise ValueError("cross-fit GNN produced no valid held-out metric")
    return FitResult(
        state_dict=best_state,
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        history=tuple(history),
    )


def refit_gnn(
    config: GNNTrainingConfig,
    spec: GNNFeatureLayoutSpec,
    *,
    ablation_variant: str,
    epochs: int,
    device: torch.device,
) -> GNNRecommender:
    """Refit the selected GNN on all positive MIMIC-train groups."""

    if spec.selection_eligible:
        raise ValueError("full refit requires the non-selection full-train cache")
    if epochs <= 0:
        raise ValueError("refit epochs must be positive")
    set_global_seed(config.seed)
    model = build_model(
        spec,
        config.architecture,
        ablation_variant=ablation_variant,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    amp_enabled = use_amp(config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if amp_enabled else None
    for epoch in range(epochs):
        _train_gnn_epoch(
            model,
            config,
            spec,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            shards_root=config.shards_root,
            exclude_fold_ids=None,
        )
    return model


class TemperatureGrid:
    """One-pass, bounded scalar temperature selection."""

    def __init__(self, device: torch.device, *, points: int = 81):
        if points < 3:
            raise ValueError("temperature grid requires at least three points")
        self.temperatures = torch.logspace(
            math.log10(0.05),
            math.log10(10.0),
            points,
            device=device,
        )
        self.loss_sums = torch.zeros(points, dtype=torch.float64, device=device)
        self.example_count = 0

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        valid_logits = logits[mask]
        valid_labels = labels[mask]
        if valid_logits.numel() == 0:
            return
        require_finite_tensor(valid_logits, name="temperature-grid logits")
        require_finite_tensor(valid_labels, name="temperature-grid labels")
        scaled = valid_logits.unsqueeze(1) / self.temperatures.unsqueeze(0)
        targets = valid_labels.unsqueeze(1).expand_as(scaled)
        losses = functional.binary_cross_entropy_with_logits(
            scaled,
            targets,
            reduction="none",
        )
        require_finite_tensor(losses, name="temperature-grid losses")
        self.loss_sums += losses.to(torch.float64).sum(dim=0)
        require_finite_tensor(
            self.loss_sums,
            name="temperature-grid accumulated losses",
        )
        self.example_count += int(valid_logits.numel())

    def best(self) -> float:
        if self.example_count == 0:
            return 1.0
        require_finite_tensor(
            self.loss_sums,
            name="temperature-grid accumulated losses",
        )
        index = int(torch.argmin(self.loss_sums).item())
        return float(self.temperatures[index].detach().cpu())


@torch.no_grad()
def fit_gnn_temperature(
    model: GNNRecommender,
    config: GNNTrainingConfig,
    spec: GNNFeatureLayoutSpec,
    *,
    device: torch.device,
) -> float:
    """Fit a bounded validation temperature without retaining all logits."""

    model.eval()
    grid = TemperatureGrid(device)
    for batch in iter_batches(
        config,
        spec,
        split="validation",
        batch_groups=config.optimization.batch_ranking_groups,
        shuffle=False,
        seed=config.seed,
        shards_root=config.shards_root,
    ):
        moved = batch.to(device)
        output = model.forward_batch(moved)
        grid.update(output.logits, moved.labels, moved.candidate_mask)
    return grid.best()


class FrozenTransformerCache:
    """Read and align one frozen-Transformer split/shard partition at a time."""

    def __init__(self, config: GNNTrainingConfig):
        self.config = config
        self._key: tuple[str, int] | None = None
        self._contexts: dict[str, np.ndarray] = {}
        self._logits: dict[tuple[str, str], tuple[str, int, float]] = {}

    @staticmethod
    def _read_partition(root: Path, split: str, shard_id: int) -> pd.DataFrame:
        directory = root / f"split={split}" / f"shard_id={shard_id}"
        if not directory.exists():
            return pd.DataFrame()
        files = sorted(directory.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        frame = pd.concat(
            (pd.read_parquet(path) for path in files),
            ignore_index=True,
        )
        if "split" not in frame.columns:
            frame["split"] = split
        if "shard_id" not in frame.columns:
            frame["shard_id"] = shard_id
        return frame

    def _load(self, split: str, shard_id: int) -> None:
        key = (split, shard_id)
        if self._key == key:
            return
        root = self.config.frozen_transformer_cache_root
        contexts = self._read_partition(root / "contexts", split, shard_id)
        logits = self._read_partition(root / "candidate_logits", split, shard_id)
        if contexts.empty or logits.empty:
            raise ValueError("frozen Transformer cache partition is missing")
        forbidden = {"patient_uid", "stay_uid", "encounter_uid"}
        if forbidden.intersection(contexts.columns) or forbidden.intersection(
            logits.columns
        ):
            raise ValueError("frozen Transformer cache contains direct identifiers")
        if contexts["ranking_group_id"].duplicated().any():
            raise ValueError("frozen Transformer context cache has duplicate groups")
        if logits.duplicated(["ranking_group_id", "candidate_medication_token"]).any():
            raise ValueError("frozen Transformer logit cache has duplicate candidates")
        self._contexts = {
            str(row.ranking_group_id): np.asarray(
                row.transformer_context,
                dtype=np.float32,
            )
            for row in contexts.itertuples(index=False)
        }
        self._logits = {
            (str(row.ranking_group_id), str(row.candidate_medication_token)): (
                str(row.index_condition_token),
                int(row.candidate_rank),
                float(row.frozen_transformer_logit),
            )
            for row in logits.itertuples(index=False)
        }
        self._key = key

    def align(self, batch: GNNBatch, *, device: torch.device) -> FrozenBatch:
        shard_ids = set(batch.shard_ids)
        splits = set(batch.splits)
        if len(shard_ids) != 1 or len(splits) != 1:
            raise ValueError("a GNN batch must stay within one split/shard partition")
        shard_id = next(iter(shard_ids))
        split = next(iter(splits))
        self._load(split, shard_id)
        contexts: list[np.ndarray] = []
        logits = torch.full(
            batch.candidate_mask.shape,
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        for row, group_id in enumerate(batch.ranking_group_ids):
            context = self._contexts.get(group_id)
            if context is None:
                raise ValueError("frozen Transformer cache is missing a graph group")
            if context.shape != (self.config.architecture.transformer_context_dim,):
                raise ValueError("frozen Transformer context dimension mismatch")
            if not bool(np.isfinite(context).all()):
                raise ValueError(
                    "frozen Transformer context contains non-finite values"
                )
            contexts.append(context)
            for position, token in enumerate(batch.candidate_tokens[row]):
                frozen = self._logits.get((group_id, token))
                if frozen is None:
                    raise ValueError(
                        "frozen Transformer cache is missing a graph candidate"
                    )
                condition, rank, value = frozen
                if (
                    condition != batch.index_condition_tokens[row]
                    or rank != int(batch.candidate_rank[row, position])
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        "frozen Transformer candidate identity/rank is inconsistent"
                    )
                logits[row, position] = value
        context_tensor = torch.as_tensor(
            np.stack(contexts),
            dtype=torch.float32,
            device=device,
        )
        context_tensor.requires_grad_(False)
        logits.requires_grad_(False)
        return FrozenBatch(context=context_tensor, logits=logits)


def frozen_artifact_locks(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    """Return exact path/hash locks for existing immutable artifacts."""

    return {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
        if path.is_file()
    }


def read_positive_temperature(
    path: Path,
    *,
    expected_schema_version: str,
    allowed_methods: frozenset[str],
    training_state_path: Path | None = None,
) -> float:
    """Read and validate one immutable calibration artifact fail-closed."""

    if not path.is_file():
        raise FileNotFoundError("temperature calibration artifact is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("calibration payload must be an object")
        if "temperature" not in payload:
            raise KeyError("temperature")
        value = float(payload["temperature"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("temperature calibration artifact is malformed") from error
    if (
        payload.get("schema_version") != expected_schema_version
        or payload.get("method") not in allowed_methods
        or payload.get("fit_split") != "mimiciv_validation"
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("temperature calibration artifact violates its contract")
    if training_state_path is not None:
        try:
            state = json.loads(training_state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise TypeError("training state must be an object")
            state_temperature = float(state["temperature"])
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "training state cannot validate the calibration temperature"
            ) from error
        if not math.isfinite(state_temperature) or not math.isclose(
            value, state_temperature, rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("calibration and training-state temperatures differ")
    return value


def validate_gnn_training_state(
    config: GNNTrainingConfig,
    checkpoint_payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate the completed GNN state against its checkpoint and live locks."""

    from pipeline.gnn_training.contract import contract_digest_or_none

    try:
        state = json.loads(config.gnn_training_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError("GNN training state is malformed") from error
    if not isinstance(state, dict):
        raise ValueError("GNN training state must be a JSON object")
    if (
        state.get("schema_version") != GNN_TRAINING_SCHEMA_VERSION
        or state.get("status") != "completed"
        or state.get("seed") != config.seed
        or state.get("contract_digest") != contract_digest_or_none(config)
    ):
        raise ValueError("GNN training state does not match the configured run")
    variant = state.get("selected_variant")
    if (
        not isinstance(variant, str)
        or not variant
        or checkpoint_payload.get("ablation_variant") != variant
    ):
        raise ValueError("GNN training state and checkpoint variants differ")
    try:
        state_epochs = int(state["refit_epochs"])
        checkpoint_epochs = int(checkpoint_payload["epochs"])
        temperature = float(state["temperature"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("GNN training state has invalid fit metadata") from error
    if (
        state_epochs <= 0
        or checkpoint_epochs != state_epochs
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("GNN training state fit metadata violates its contract")
    if not config.crossfit_graph_manifest_path.is_file() or state.get(
        "crossfit_manifest_sha256"
    ) != sha256_file(config.crossfit_graph_manifest_path):
        raise ValueError("GNN training state cross-fit lock is stale")
    metadata = checkpoint_payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("selected_variant") != variant:
        raise ValueError("GNN checkpoint selection metadata is incomplete")
    return state


def clone_model(model: GNNRecommender, *, device: torch.device) -> GNNRecommender:
    """Return an independent model copy for residual fusion."""

    return copy.deepcopy(model).to(device)


def iter_gnn_batches(
    config: GNNTrainingConfig,
    spec: GNNFeatureLayoutSpec,
    *,
    split: str,
    shards_root: Path,
    shuffle: bool,
    epoch: int = 0,
    include_fold_ids: frozenset[int] | None = None,
    exclude_fold_ids: frozenset[int] | None = None,
    require_positive: bool = False,
) -> Iterator[GNNBatch]:
    """Named wrapper used by fusion code to keep the cache scope explicit."""

    yield from iter_batches(
        config,
        spec,
        split=split,
        batch_groups=config.optimization.batch_ranking_groups,
        shuffle=shuffle,
        seed=config.seed,
        epoch=epoch,
        shards_root=shards_root,
        include_fold_ids=include_fold_ids,
        exclude_fold_ids=exclude_fold_ids,
        require_positive=require_positive,
    )
