"""Frozen-Transformer late and residual fusion primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

DEFAULT_LATE_FUSION_WEIGHTS = tuple(index / 20.0 for index in range(21))


def within_group_finite_z_normalize(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Return finite population-z scores independently within each row.

    Non-finite or masked entries are excluded from the moments and returned as
    zero.  Constant and singleton groups also return zero rather than NaN.
    """

    if scores.ndim != 2:
        raise ValueError("scores must have shape (G, C)")
    if not scores.is_floating_point():
        scores = scores.to(dtype=torch.float32)
    if candidate_mask is None:
        candidate_mask = torch.ones_like(scores, dtype=torch.bool)
    if candidate_mask.shape != scores.shape:
        raise ValueError("candidate_mask must match scores")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    finite = candidate_mask.to(dtype=torch.bool) & torch.isfinite(scores)
    counts = finite.sum(dim=1, keepdim=True).clamp(min=1).to(scores.dtype)
    safe_scores = torch.where(finite, scores, torch.zeros_like(scores))
    means = safe_scores.sum(dim=1, keepdim=True) / counts
    centered = torch.where(finite, scores - means, torch.zeros_like(scores))
    variances = centered.square().sum(dim=1, keepdim=True) / counts
    scales = variances.sqrt().clamp_min(epsilon)
    normalized = centered / scales
    return torch.where(finite, normalized, torch.zeros_like(normalized))


# Concise alias for callers that already operate on two-dimensional groups.
finite_group_zscore = within_group_finite_z_normalize


def late_fusion_logits(
    frozen_logits: torch.Tensor,
    gnn_logits: torch.Tensor,
    *,
    gnn_weight: float,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse within-group z scores with ``gnn_weight`` constrained to ``[0, 1]``."""

    if frozen_logits.shape != gnn_logits.shape:
        raise ValueError("frozen and GNN logits must have the same shape")
    if not 0.0 <= float(gnn_weight) <= 1.0:
        raise ValueError("gnn_weight must be in [0, 1]")
    if candidate_mask is None:
        candidate_mask = torch.ones_like(frozen_logits, dtype=torch.bool)
    if candidate_mask.shape != frozen_logits.shape:
        raise ValueError("candidate_mask must match logits")
    valid = (
        candidate_mask.to(dtype=torch.bool)
        & torch.isfinite(frozen_logits)
        & torch.isfinite(gnn_logits)
    )
    frozen_z = within_group_finite_z_normalize(frozen_logits.detach(), valid)
    gnn_z = within_group_finite_z_normalize(gnn_logits, valid)
    fused = (1.0 - float(gnn_weight)) * frozen_z + float(gnn_weight) * gnn_z
    return fused.masked_fill(~valid, float("-inf"))


def select_late_fusion_weight(
    frozen_logits: torch.Tensor,
    gnn_logits: torch.Tensor,
    *,
    objective_fn: Callable[[torch.Tensor], float | torch.Tensor],
    candidate_mask: torch.Tensor | None = None,
    candidate_weights: Sequence[float] = DEFAULT_LATE_FUSION_WEIGHTS,
) -> float:
    """Select a constrained GNN weight by maximizing a caller-supplied objective.

    Candidate weights are evaluated in sorted order.  Exact objective ties keep
    the smaller GNN weight, deterministically favoring the frozen reference.
    """

    if not candidate_weights:
        raise ValueError("candidate_weights must not be empty")
    weights = sorted({float(weight) for weight in candidate_weights})
    if any(not 0.0 <= weight <= 1.0 for weight in weights):
        raise ValueError("candidate fusion weights must be in [0, 1]")

    best_weight: float | None = None
    best_value = float("-inf")
    with torch.no_grad():
        for weight in weights:
            fused = late_fusion_logits(
                frozen_logits,
                gnn_logits,
                gnn_weight=weight,
                candidate_mask=candidate_mask,
            )
            raw_value = objective_fn(fused)
            value = (
                float(raw_value.detach().cpu())
                if isinstance(raw_value, torch.Tensor)
                else float(raw_value)
            )
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError("fusion objective must return a finite scalar")
            if best_weight is None or value > best_value:
                best_weight = weight
                best_value = value
    if best_weight is None:  # pragma: no cover - guarded by non-empty weights
        raise RuntimeError("no fusion weight was evaluated")
    return best_weight


# Explicit constrained-selection alias used by higher-level training code.
select_constrained_fusion_weight = select_late_fusion_weight


def detach_tensors(value: Any) -> Any:
    """Recursively detach tensor outputs while preserving common containers."""

    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, Mapping):
        return type(value)((key, detach_tensors(item)) for key, item in value.items())
    if isinstance(value, tuple):
        detached = tuple(detach_tensors(item) for item in value)
        if hasattr(value, "_fields"):
            return type(value)(*detached)
        return detached
    if isinstance(value, list):
        return [detach_tensors(item) for item in value]
    return value


def freeze_module(module: nn.Module) -> nn.Module:
    """Put a module in eval mode, disable its gradients, and clear old grads."""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return module


def frozen_forward(
    module: nn.Module,
    *args: Any,
    method: str | None = None,
    **kwargs: Any,
) -> Any:
    """Evaluate a frozen module method and return recursively detached outputs."""

    freeze_module(module)
    callable_object = module if method is None else getattr(module, method)
    with torch.no_grad():
        output = callable_object(*args, **kwargs)
    return detach_tensors(output)


class FrozenModule(nn.Module):
    """Wrapper that keeps an owned module frozen even when a parent trains."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = freeze_module(module)
        super().train(False)

    def train(self, mode: bool = True) -> FrozenModule:
        """Ignore training-mode requests and keep the wrapped module in eval."""

        del mode
        super().train(False)
        self.module.eval()
        return self

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return frozen_forward(self.module, *args, **kwargs)


class ResidualFusionHead(nn.Module):
    """Add a learned GNN residual to immutable Transformer logits.

    The final layer is initialized to exact zeros, so every initial hybrid
    logit is bit-for-bit equal to its frozen Transformer input.
    """

    def __init__(
        self,
        *,
        transformer_context_dim: int,
        gnn_candidate_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        if transformer_context_dim <= 0 or gnn_candidate_dim <= 0:
            raise ValueError("fusion input dimensions must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.transformer_context_dim = transformer_context_dim
        self.gnn_candidate_dim = gnn_candidate_dim
        self.hidden = nn.Sequential(
            nn.LayerNorm(transformer_context_dim + gnn_candidate_dim),
            nn.Linear(transformer_context_dim + gnn_candidate_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def residual_logits(
        self,
        *,
        transformer_context: torch.Tensor,
        gnn_candidate_representations: torch.Tensor,
    ) -> torch.Tensor:
        """Return residual logits while detaching the frozen context tensor."""

        if transformer_context.ndim != 2:
            raise ValueError("transformer_context must have shape (G, T)")
        if gnn_candidate_representations.ndim != 3:
            raise ValueError("gnn_candidate_representations must have shape (G, C, R)")
        group_count, candidate_count, representation_dim = (
            gnn_candidate_representations.shape
        )
        if transformer_context.shape != (
            group_count,
            self.transformer_context_dim,
        ):
            raise ValueError("transformer_context shape does not match fusion head")
        if representation_dim != self.gnn_candidate_dim:
            raise ValueError(
                "GNN candidate representation size does not match fusion head"
            )
        frozen_context = transformer_context.detach()
        expanded_context = frozen_context.unsqueeze(1).expand(-1, candidate_count, -1)
        features = torch.cat(
            (expanded_context, gnn_candidate_representations),
            dim=-1,
        )
        return self.output_layer(self.hidden(features)).squeeze(-1)

    def forward(
        self,
        *,
        frozen_logits: torch.Tensor,
        transformer_context: torch.Tensor,
        gnn_candidate_representations: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return frozen logits plus the trainable residual."""

        expected = gnn_candidate_representations.shape[:2]
        if frozen_logits.shape != expected:
            raise ValueError("frozen_logits must match GNN group/candidate dimensions")
        if candidate_mask is None:
            candidate_mask = torch.ones_like(frozen_logits, dtype=torch.bool)
        if candidate_mask.shape != frozen_logits.shape:
            raise ValueError("candidate_mask must match frozen_logits")
        residual = self.residual_logits(
            transformer_context=transformer_context,
            gnn_candidate_representations=gnn_candidate_representations,
        )
        hybrid = frozen_logits.detach() + residual
        return hybrid.masked_fill(~candidate_mask, float("-inf"))


def _module_sequence(
    modules: nn.Module | Iterable[nn.Module],
) -> tuple[nn.Module, ...]:
    if isinstance(modules, nn.Module):
        return (modules,)
    return tuple(modules)


def assert_gradient_ownership(
    *,
    trainable_modules: nn.Module | Iterable[nn.Module],
    frozen_modules: nn.Module | Iterable[nn.Module] = (),
    frozen_tensors: Iterable[torch.Tensor] = (),
    require_trainable_gradient: bool = True,
) -> None:
    """Assert finite trainable grads and absence of frozen-module/tensor grads."""

    trainable = _module_sequence(trainable_modules)
    frozen = _module_sequence(frozen_modules)
    trainable_gradients: list[torch.Tensor] = []
    for module in trainable:
        for parameter in module.parameters():
            if parameter.requires_grad and parameter.grad is not None:
                trainable_gradients.append(parameter.grad)
    if require_trainable_gradient and not trainable_gradients:
        raise AssertionError("no trainable parameter received a gradient")
    if any(
        not bool(torch.isfinite(gradient).all()) for gradient in trainable_gradients
    ):
        raise AssertionError("a trainable parameter received a non-finite gradient")

    for module in frozen:
        if module.training:
            raise AssertionError("a frozen module is in training mode")
        for parameter in module.parameters():
            if parameter.requires_grad:
                raise AssertionError("a frozen parameter still requires gradients")
            if parameter.grad is not None:
                raise AssertionError("a frozen parameter received a gradient")
    for tensor in frozen_tensors:
        if tensor.grad is not None:
            raise AssertionError("a frozen input tensor received a gradient")
