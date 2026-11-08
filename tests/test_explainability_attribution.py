"""Small exact-game oracles for shared ranking attribution."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest

from pipeline.explainability.attribution import (
    EvaluationBudgetExceeded,
    attribute_groups,
)
from pipeline.explainability.margins import ScoreSlate

GROUPS = ("event", "relation", "third_candidate")
PAIRS = (("A", "B"), ("B", "C"), ("A", "C"), ("B", "A"))


def raw_scores(retained):
    x, y, z = (int(g in retained) for g in GROUPS)
    return (
        (1 + x + 2 * x * y, 0.2 + 0.3 * y, -0.4 + 3 * z),
        (0.3 + y, 1 + 0.5 * x, 0.1 + x * z),
    )


def scorer(retained):
    t, g = raw_scores(retained)
    return ScoreSlate(("A", "B", "C"), t, g, 0.25)


def oracle_margin(retained, a=0, b=1):
    # Independent score-only oracle, without the proposed route operator.
    normalized = []
    for scores in raw_scores(retained):
        values = np.array(scores, dtype=float)
        normalized.append((values - values.mean()) / max(values.std(), 1e-6))
    u = 0.75 * normalized[0] + 0.25 * normalized[1]
    return u[a] - u[b]


def test_exact_attribution_matches_subset_shapley_oracle_and_reuses_calls():
    result = attribute_groups(scorer, GROUPS, PAIRS, n_permutations=None)
    assert result.method == "exact"
    assert result.permutations == 6
    assert result.evaluations == 8
    assert result.seed is None
    for j, group in enumerate(GROUPS):
        others = [g for g in GROUPS if g != group]
        expected = 0
        for size in range(3):
            weight = math.factorial(size) * math.factorial(2 - size) / math.factorial(3)
            for subset in itertools.combinations(others, size):
                expected += weight * (
                    oracle_margin(set(subset) | {group}) - oracle_margin(set(subset))
                )
        contribution = result.comparisons[0].contributions[j]
        assert contribution.value == pytest.approx(expected, abs=1e-12)
        assert abs(contribution.route_residual) < 1e-12
        assert contribution.standard_error is None
    for comparison in result.comparisons:
        assert abs(comparison.completeness_residual) < 1e-12
        assert (
            comparison.baseline.total != 0
        )  # Baseline is retained, not forced to zero.


def test_shared_sampling_preserves_comparator_consistency_and_reproducibility():
    first = attribute_groups(scorer, GROUPS, PAIRS, n_permutations=19, seed=872)
    second = attribute_groups(scorer, GROUPS, PAIRS, n_permutations=19, seed=872)
    assert first == second
    ab, bc, ac, ba = first.comparisons
    for j in range(3):
        for channel in (
            "value",
            "transformer_direct",
            "transformer_normalization",
            "gnn_direct",
            "gnn_normalization",
        ):
            a, b, c, d = [
                getattr(x.contributions[j], channel) for x in (ab, bc, ac, ba)
            ]
            assert a + b == pytest.approx(c, abs=1e-12)
            assert a == pytest.approx(-d, abs=1e-12)
        assert ab.contributions[j].standard_error >= 0
    assert abs(ab.completeness_residual) < 1e-12


def test_sampling_standard_error_matches_independent_permutation_samples():
    import random

    count, seed = 17, 223
    result = attribute_groups(
        scorer, GROUPS, PAIRS[:1], n_permutations=count, seed=seed
    )
    rng = random.Random(seed)
    samples = [[] for _ in GROUPS]
    for _ in range(count):
        retained = set()
        for index in rng.sample(range(3), 3):
            before = oracle_margin(retained)
            retained.add(GROUPS[index])
            samples[index].append(oracle_margin(retained) - before)
    for item, values in zip(result.comparisons[0].contributions, samples):
        assert item.standard_error == pytest.approx(
            np.std(values, ddof=1) / math.sqrt(count)
        )


def test_dummy_group_has_zero_effect_and_one_draw_has_no_standard_error():
    result = attribute_groups(scorer, GROUPS + ("unused",), PAIRS[:1], n_permutations=1)
    dummy = result.comparisons[0].contributions[-1]
    assert dummy.value == 0
    assert dummy.standard_error is None
    assert dummy.transformer_direct == dummy.transformer_normalization == 0
    assert dummy.gnn_direct == dummy.gnn_normalization == 0


def test_budget_is_hard_and_no_partial_explanation_is_returned():
    calls = []

    def recording_scorer(retained):
        calls.append(retained)
        return scorer(retained)

    with pytest.raises(EvaluationBudgetExceeded):
        attribute_groups(recording_scorer, GROUPS, PAIRS, max_evaluations=3)
    assert len(calls) == 3


def test_small_cache_changes_cost_but_not_exact_attribution():
    cached = attribute_groups(scorer, GROUPS, PAIRS[:1], n_permutations=None)
    bounded = attribute_groups(
        scorer, GROUPS, PAIRS[:1], n_permutations=None, cache_size=1
    )
    assert bounded.evaluations > cached.evaluations
    assert bounded.comparisons == cached.comparisons


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_ids", ("B", "A", "C")),
        ("candidate_mask", (True, True, False)),
        ("gnn_weight", 0.5),
        ("epsilon", 1e-5),
    ],
)
def test_scorer_cannot_change_comparison_context(field, value):
    def inconsistent(retained):
        return (
            replace(scorer(retained), **{field: value})
            if len(retained) == 1
            else scorer(retained)
        )

    with pytest.raises(ValueError, match="preserve"):
        attribute_groups(inconsistent, GROUPS, PAIRS[:1])


def test_callback_errors_are_sanitized_and_nonfinite_interventions_fail():
    def broken(_):
        raise RuntimeError("private patient record")

    with pytest.raises(ValueError, match="scorer failed") as error:
        attribute_groups(broken, GROUPS, PAIRS[:1])
    assert "private patient record" not in str(error.value)
    assert error.value.__suppress_context__

    def invalid(retained):
        return replace(scorer(retained), transformer=(math.nan, 1, 0))

    with pytest.raises(ValueError, match="scorer failed"):
        attribute_groups(invalid, GROUPS, PAIRS[:1])


@pytest.mark.parametrize("pairs", ["AB", ("AB",)])
def test_strings_are_not_interpreted_as_comparison_vectors(pairs):
    with pytest.raises(ValueError):
        attribute_groups(scorer, GROUPS, pairs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_ids": ()},
        {"group_ids": ("repeat", "repeat")},
        {"group_ids": ("",)},
        {"comparisons": ()},
        {"comparisons": (("A", "A"),)},
        {"comparisons": (("private-reference", "B"),)},
        {"n_permutations": 0},
        {"n_permutations": True},
        {"seed": 1.5},
        {"max_evaluations": 1},
        {"cache_size": 0},
        {"group_ids": tuple(str(i) for i in range(7)), "n_permutations": None},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    args = {"group_ids": GROUPS, "comparisons": PAIRS[:1]} | kwargs
    with pytest.raises(ValueError):
        attribute_groups(scorer, **args)
