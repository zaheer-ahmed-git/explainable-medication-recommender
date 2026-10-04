"""Finite-change accounting for fixed-slate, z-normalized late fusion.

Float64 reference arithmetic mirrors the production normalization equation.
It is not a replacement scorer or a certificate of input/clinical validity.
Objects suppress automatic reprs because even derived scores can be sensitive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DEFAULT_EPSILON = 1e-6


@dataclass(frozen=True, repr=False)
class _Branch:
    mean: float
    variance: float
    scale: float
    normalized: tuple[float, ...]


def _normalize(
    scores: tuple[float, ...], mask: tuple[bool, ...], eps: float
) -> _Branch:
    values = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    normalized = np.zeros_like(values)
    if not valid.any():
        return _Branch(0.0, 0.0, eps, tuple(normalized))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            selected = values[valid]
            mean = float(selected.mean())
            centered = selected - mean
            variance = float(np.mean(centered * centered))
            scale = max(math.sqrt(variance), eps)
            normalized[valid] = centered / scale
    except FloatingPointError:
        raise ValueError("score normalization exceeds the numerical range") from None
    return _Branch(mean, variance, scale, tuple(float(v) for v in normalized))


@dataclass(frozen=True, repr=False)
class ScoreSlate:
    """Immutable aligned branch scores for one context, without patient keys.

    Non-finite scores are permitted only in explicitly masked slots. A model
    adapter must supply the authoritative joint finite mask. It must not change
    that mask when an intervention produces invalid scores. No ranking/tie rule
    is inferred from candidate identity or array position.
    """

    candidate_ids: tuple[str, ...]
    transformer: tuple[float, ...]
    gnn: tuple[float, ...]
    gnn_weight: float
    candidate_mask: tuple[bool, ...] | None = None
    epsilon: float = DEFAULT_EPSILON
    _t: _Branch = field(init=False)
    _g: _Branch = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.candidate_ids, str):
            raise ValueError("candidate identifiers must be a sequence of strings")
        ids = tuple(self.candidate_ids)
        if not ids or any(not isinstance(v, str) or not v for v in ids):
            raise ValueError("candidate identifiers must be non-empty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("candidate identifiers must be unique")
        mask = (
            (True,) * len(ids)
            if self.candidate_mask is None
            else tuple(self.candidate_mask)
        )
        if len(mask) != len(ids) or any(type(v) is not bool for v in mask):
            raise ValueError("candidate mask must contain one Boolean per candidate")
        for name in ("gnn_weight", "epsilon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("fusion parameters must be finite numbers")
            if not math.isfinite(value):
                raise ValueError("fusion parameters must be finite numbers")
        if not 0 <= self.gnn_weight <= 1:
            raise ValueError("fusion weight must be in [0, 1]")
        if self.epsilon <= 0 or not math.isfinite(1.0 / self.epsilon):
            raise ValueError("epsilon must have a positive finite reciprocal")
        object.__setattr__(self, "candidate_ids", ids)
        object.__setattr__(self, "candidate_mask", mask)
        for name in ("transformer", "gnn"):
            if isinstance(getattr(self, name), (str, bytes)):
                raise ValueError("branch scores must be numeric vectors")
            try:
                values = tuple(float(v) for v in getattr(self, name))
            except (TypeError, ValueError, OverflowError):
                raise ValueError("branch scores must be numeric vectors") from None
            if len(values) != len(ids):
                raise ValueError("branch scores must align with the candidate slate")
            if any(keep and not math.isfinite(v) for v, keep in zip(values, mask)):
                raise ValueError(
                    "active candidate scores must be finite in both branches"
                )
            object.__setattr__(self, name, values)
        object.__setattr__(self, "_t", _normalize(self.transformer, mask, self.epsilon))
        object.__setattr__(self, "_g", _normalize(self.gnn, mask, self.epsilon))

    @property
    def fused_scores(self) -> tuple[float, ...]:
        """Return reference scores; explicitly masked candidates remain -inf."""
        return tuple(
            (1 - self.gnn_weight) * t + self.gnn_weight * g if keep else -math.inf
            for t, g, keep in zip(
                self._t.normalized, self._g.normalized, self.candidate_mask
            )
        )

    def pair_indexes(self, a: str, b: str) -> tuple[int, int]:
        try:
            i, j = self.candidate_ids.index(a), self.candidate_ids.index(b)
        except ValueError:
            raise ValueError("comparison candidates must belong to the slate") from None
        if i == j or not self.candidate_mask[i] or not self.candidate_mask[j]:
            raise ValueError("comparison requires two distinct active candidates")
        return i, j

    def margin(self, a: str, b: str) -> PairMargin:
        i, j = self.pair_indexes(a, b)
        transformer = (1 - self.gnn_weight) * (
            self._t.normalized[i] - self._t.normalized[j]
        )
        gnn = self.gnn_weight * (self._g.normalized[i] - self._g.normalized[j])
        return PairMargin(transformer, gnn)


@dataclass(frozen=True, repr=False)
class PairMargin:
    transformer: float
    gnn: float

    @property
    def total(self) -> float:
        return self.transformer + self.gnn


def require_same_slate(before: ScoreSlate, after: ScoreSlate) -> None:
    """Reject changed membership, alignment, masks or fusion parameters."""
    if (
        before.candidate_ids != after.candidate_ids
        or before.candidate_mask != after.candidate_mask
        or before.gnn_weight != after.gnn_weight
        or before.epsilon != after.epsilon
    ):
        raise ValueError("interventions must preserve the aligned slate and fusion")


@dataclass(frozen=True, repr=False)
class BranchTransition:
    """Signed route allocations aligned with the original candidate slots."""

    direct_by_candidate: tuple[float, ...]
    normalization_by_candidate: tuple[float, ...]
    margin_change: float

    @property
    def direct(self) -> float:
        return math.fsum(self.direct_by_candidate)

    @property
    def normalization(self) -> float:
        return math.fsum(self.normalization_by_candidate)

    @property
    def residual(self) -> float:
        return self.margin_change - math.fsum((self.direct, self.normalization))


@dataclass(frozen=True, repr=False)
class MarginTransition:
    transformer: BranchTransition
    gnn: BranchTransition

    @property
    def margin_change(self) -> float:
        return self.transformer.margin_change + self.gnn.margin_change

    @property
    def residual(self) -> float:
        return self.transformer.residual + self.gnn.residual


def _branch_transition(
    before: tuple[float, ...],
    after: tuple[float, ...],
    start: _Branch,
    end: _Branch,
    mask: tuple[bool, ...],
    i: int,
    j: int,
    weight: float,
    epsilon: float,
) -> BranchTransition:
    direct = np.zeros(len(before), dtype=np.float64)
    normalization = np.zeros_like(direct)
    change = weight * (
        (end.normalized[i] - end.normalized[j])
        - (start.normalized[i] - start.normalized[j])
    )
    if weight == 0:
        return BranchTransition(tuple(direct), tuple(normalization), change)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            valid = np.asarray(mask, dtype=bool)
            old, new = np.asarray(before)[valid], np.asarray(after)[valid]
            delta = new - old
            r_mean = 0.5 / start.scale + 0.5 / end.scale
            gap_mean = 0.5 * (before[i] - before[j]) + 0.5 * (after[i] - after[j])
            direct[i] = weight * r_mean * (after[i] - before[i])
            direct[j] = -weight * r_mean * (after[j] - before[j])

            if start.scale == epsilon and end.scale == epsilon:
                slope = 0.0
            elif start.scale > epsilon and end.scale > epsilon:
                # Stable divided difference of v**(-1/2), also for equal v.
                slope = -(1.0 / start.scale) / end.scale / (start.scale + end.scale)
            else:
                reciprocal_change = (start.scale - end.scale) / start.scale / end.scale
                slope = reciprocal_change / (end.variance - start.variance)
            variance_routes = (new - end.mean + old - start.mean) * delta / len(old)
            normalization[valid] = weight * gap_mean * slope * variance_routes
    except (FloatingPointError, OverflowError, ZeroDivisionError):
        raise ValueError("route allocation exceeds the numerical range") from None
    if not np.isfinite(direct).all() or not np.isfinite(normalization).all():
        raise ValueError("route allocation exceeds the numerical range")
    return BranchTransition(tuple(direct), tuple(normalization), change)


def decompose_transition(
    before: ScoreSlate, after: ScoreSlate, a: str, b: str
) -> MarginTransition:
    """Split a finite full-slate change; never renormalize away the residual.

    The symmetric product allocation is a convention. Candidate routes are
    computational allocations, not independent interventions or causal effects.
    """
    require_same_slate(before, after)
    i, j = before.pair_indexes(a, b)
    return MarginTransition(
        _branch_transition(
            before.transformer,
            after.transformer,
            before._t,
            after._t,
            before.candidate_mask,
            i,
            j,
            1 - before.gnn_weight,
            before.epsilon,
        ),
        _branch_transition(
            before.gnn,
            after.gnn,
            before._g,
            after._g,
            before.candidate_mask,
            i,
            j,
            before.gnn_weight,
            before.epsilon,
        ),
    )
