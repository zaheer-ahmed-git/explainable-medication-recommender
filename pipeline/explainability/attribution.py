"""Bounded, shared-coalition attribution of fused ranking margins.

The callback owns intervention semantics. This module neither constructs
clinical records nor establishes their validity. It retains no source records,
uses one reference, and performs no logging or persistence.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from pipeline.explainability.margins import (
    PairMargin,
    ScoreSlate,
    decompose_transition,
    require_same_slate,
)

MAX_EXACT_GROUPS = 6


class EvaluationBudgetExceeded(RuntimeError):
    """No partial attribution is returned when the scoring budget is exhausted."""


@dataclass(frozen=True, repr=False)
class GroupContribution:
    group_id: str
    transformer_direct: float
    transformer_normalization: float
    gnn_direct: float
    gnn_normalization: float
    # Average marginal effect measured directly from full-slate branch margins.
    value: float
    # Permutation sampling only; None for exact enumeration or a single draw.
    standard_error: float | None

    @property
    def route_residual(self) -> float:
        return self.value - math.fsum(
            (
                self.transformer_direct,
                self.transformer_normalization,
                self.gnn_direct,
                self.gnn_normalization,
            )
        )


@dataclass(frozen=True, repr=False)
class ComparisonAttribution:
    candidates: tuple[str, str]
    baseline: PairMargin
    observed: PairMargin
    contributions: tuple[GroupContribution, ...]

    @property
    def completeness_residual(self) -> float:
        return self.observed.total - math.fsum(
            (
                self.baseline.total,
                *(v.value for v in self.contributions),
            )
        )


@dataclass(frozen=True, repr=False)
class AttributionResult:
    comparisons: tuple[ComparisonAttribution, ...]
    method: str
    permutations: int
    seed: int | None
    evaluations: int
    cache_hits: int


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def attribute_groups(
    scorer: Callable[[frozenset[str]], ScoreSlate],
    group_ids: Sequence[str],
    comparisons: Sequence[tuple[str, str]],
    *,
    n_permutations: int | None = 32,
    seed: int = 20260917,
    max_evaluations: int = 1024,
    cache_size: int = 128,
) -> AttributionResult:
    """Attribute multiple pairwise margins with shared full-slate evaluations.

    ``scorer(S)`` must deterministically return the entire fixed slate after
    retaining groups S under a single fixed reference policy. All groups must
    use the same intervention class. Group identifiers specify the RNG's initial
    order; references and scorer/model provenance remain the caller's job.

    ``n_permutations=None`` enumerates every order (at most six groups).
    Otherwise uniform permutations are sampled with replacement. Empty/full
    states are pinned; other states use an LRU cache bounded by ``cache_size``.
    The budget counts actual callback invocations, including both endpoints.
    Callback exceptions are sanitized; a callback's own logs are its concern.

    Memory is O(groups * comparisons + cache_size * candidates). Candidate-level
    routes are available in decompose_transition; only four branch/route totals
    per group/comparison are retained here. Numerical completeness, sampling
    precision, and the validity of the intervention are separate questions.
    """
    groups = tuple(group_ids)
    if (
        isinstance(group_ids, str)
        or not groups
        or any(not isinstance(v, str) or not v for v in groups)
    ):
        raise ValueError("groups must be non-empty string identifiers")
    if len(set(groups)) != len(groups):
        raise ValueError("group identifiers must be unique")
    if isinstance(comparisons, (str, bytes)) or any(
        isinstance(pair, (str, bytes)) for pair in comparisons
    ):
        raise ValueError("comparisons must contain candidate identifier pairs")
    pairs = tuple(tuple(pair) for pair in comparisons)
    if not pairs or any(
        len(pair) != 2 or any(not isinstance(v, str) for v in pair) for pair in pairs
    ):
        raise ValueError("comparisons must contain candidate identifier pairs")
    if len(set(pairs)) != len(pairs):
        raise ValueError("comparison pairs must be unique")
    for value, name in (
        (max_evaluations, "max_evaluations"),
        (cache_size, "cache_size"),
    ):
        _positive_integer(value, name)
    if max_evaluations < 2:
        raise ValueError("budget must include both endpoint evaluations")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if n_permutations is None:
        if len(groups) > MAX_EXACT_GROUPS:
            raise ValueError("exact enumeration supports at most six groups")
        orders = itertools.permutations(range(len(groups)))
    else:
        _positive_integer(n_permutations, "n_permutations")
        rng = random.Random(seed)
        orders = (
            tuple(rng.sample(range(len(groups)), len(groups)))
            for _ in range(n_permutations)
        )

    evaluations = 0
    cache_hits = 0
    cache: OrderedDict[frozenset[str], ScoreSlate] = OrderedDict()

    def evaluate(retained: frozenset[str]) -> ScoreSlate:
        nonlocal evaluations
        if evaluations >= max_evaluations:
            raise EvaluationBudgetExceeded("full-slate evaluation budget exhausted")
        evaluations += 1
        try:
            result = scorer(retained)
        except Exception:
            raise ValueError("scorer failed; no explanation returned") from None
        if not isinstance(result, ScoreSlate):
            raise ValueError("scorer must return a ScoreSlate")
        return result

    empty = frozenset()
    full = frozenset(groups)
    baseline = evaluate(empty)
    # Validate comparators before spending the remaining query budget.
    for a, b in pairs:
        baseline.pair_indexes(a, b)
    observed = evaluate(full)
    require_same_slate(baseline, observed)

    def get(retained: frozenset[str]) -> ScoreSlate:
        nonlocal cache_hits
        if retained == empty:
            cache_hits += 1
            return baseline
        if retained == full:
            cache_hits += 1
            return observed
        if retained in cache:
            cache_hits += 1
            cache.move_to_end(retained)
            return cache[retained]
        result = evaluate(retained)
        require_same_slate(baseline, result)
        cache[retained] = result
        if len(cache) > cache_size:
            cache.popitem(last=False)
        return result

    # Welford updates avoid retaining permutation samples. Column 4 stores the
    # independently measured effect, not a forced sum of allocated routes.
    means = np.zeros((len(groups), len(pairs), 5), dtype=np.float64)
    sum_squares = np.zeros((len(groups), len(pairs)), dtype=np.float64)
    count = 0
    for count, order in enumerate(orders, start=1):
        retained = empty
        before = baseline
        for group_index in order:
            retained = retained | {groups[group_index]}
            after = get(retained)
            for pair_index, (a, b) in enumerate(pairs):
                transition = decompose_transition(before, after, a, b)
                sample = np.array(
                    (
                        transition.transformer.direct,
                        transition.transformer.normalization,
                        transition.gnn.direct,
                        transition.gnn.normalization,
                        transition.margin_change,
                    )
                )
                delta = sample - means[group_index, pair_index]
                means[group_index, pair_index] += delta / count
                sum_squares[group_index, pair_index] += delta[4] * (
                    sample[4] - means[group_index, pair_index, 4]
                )
            before = after

    if not np.isfinite(means).all() or not np.isfinite(sum_squares).all():
        raise ValueError("attribution accumulation exceeds the numerical range")
    results = []
    for pair_index, (a, b) in enumerate(pairs):
        contributions = []
        for group_index, group in enumerate(groups):
            values = means[group_index, pair_index]
            se = None
            if n_permutations is not None and count > 1:
                se = math.sqrt(
                    max(0.0, float(sum_squares[group_index, pair_index]))
                    / (count - 1)
                    / count
                )
            contributions.append(
                GroupContribution(
                    group,
                    *(float(v) for v in values),
                    standard_error=se,
                )
            )
        results.append(
            ComparisonAttribution(
                (a, b),
                baseline.margin(a, b),
                observed.margin(a, b),
                tuple(contributions),
            )
        )
    return AttributionResult(
        tuple(results),
        "exact" if n_permutations is None else "permutation",
        count,
        None if n_permutations is None else seed,
        evaluations,
        cache_hits,
    )
