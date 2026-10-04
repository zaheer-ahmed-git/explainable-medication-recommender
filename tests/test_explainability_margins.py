"""Synthetic numerical oracles; no patient data, checkpoints or GPU use."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from pipeline.explainability.margins import ScoreSlate, decompose_transition


def slate(t=(2.0, 1.0, 0.0), g=(0.0, 2.0, 1.0), **kwargs):
    return ScoreSlate(("A", "B", "C"), t, g, kwargs.pop("gnn_weight", 0.25), **kwargs)


def test_branch_accounting_and_third_candidate_reversal():
    before = slate()
    after = replace(before, transformer=(2.0, 1.0, 100.0))
    margin = before.margin("A", "B")
    assert margin.transformer == pytest.approx(0.9185586535436917)
    assert margin.gnn == pytest.approx(-0.6123724356957945)
    assert margin.total == pytest.approx(0.30618621784789724)
    assert after.margin("A", "B").total == pytest.approx(-0.5962208745978942)
    transition = decompose_transition(before, after, "A", "B")
    assert transition.transformer.direct == 0
    assert transition.gnn.margin_change == 0
    assert transition.transformer.normalization_by_candidate[:2] == (0, 0)
    assert transition.transformer.normalization_by_candidate[2] < 0
    assert transition.transformer.normalization == pytest.approx(
        after.margin("A", "B").total - margin.total
    )
    assert abs(transition.residual) < 1e-12


@pytest.mark.parametrize("weight", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("mask", [None, (True, True, False)])
def test_fusion_matches_independent_population_zscore(weight, mask):
    scores = slate(gnn_weight=weight, candidate_mask=mask)
    valid = np.array(scores.candidate_mask)
    expected = np.zeros(3)
    for raw, w in ((scores.transformer, 1 - weight), (scores.gnn, weight)):
        x = np.array(raw)[valid]
        expected[valid] += w * (x - x.mean()) / max(x.std(ddof=0), 1e-6)
    expected[~valid] = -np.inf
    np.testing.assert_allclose(scores.fused_scores, expected, atol=1e-14)
    assert scores.margin("A", "B").total == pytest.approx(expected[0] - expected[1])


def test_masked_nonfinite_slots_never_enter_moments_or_routes():
    before = slate(
        (2, 1, math.nan), (0, 2, math.inf), candidate_mask=(True, True, False)
    )
    after = replace(before, transformer=(3, 1, -math.inf), gnn=(2, 1, math.nan))
    assert before.fused_scores == pytest.approx((0.5, -0.5, -math.inf))
    result = decompose_transition(before, after, "A", "B")
    for branch in (result.transformer, result.gnn):
        assert branch.direct_by_candidate[2] == 0
        assert branch.normalization_by_candidate[2] == 0
    assert abs(result.residual) < 1e-12
    with pytest.raises(ValueError, match="active"):
        before.margin("A", "C")


def test_constant_singleton_empty_and_tied_slates():
    constant = slate((3, 3, 3), (7, 7, 7))
    assert constant.fused_scores == (0, 0, 0)
    assert constant.margin("A", "B").total == 0
    singleton = replace(constant, candidate_mask=(True, False, False))
    assert singleton.fused_scores == (0, -math.inf, -math.inf)
    empty = replace(constant, candidate_mask=(False,) * 3)
    assert empty.fused_scores == (-math.inf,) * 3
    with pytest.raises(ValueError):
        empty.margin("A", "B")


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ((0, 0, 0), (2, 1, 0)),
        ((0, 1e-8, -1e-8), (1e-8, 0, -1e-8)),
        ((0, 2e-6, -2e-6), (0, 1e-8, -1e-8)),
        ((2, 1, 0), (0, 1, 2)),
        ((0, 0, 0), (0, 0, 0)),
    ],
)
def test_floor_crossings_equal_variance_and_unchanged_states(start, end):
    before = slate(start)
    after = slate(end)
    result = decompose_transition(before, after, "A", "B")
    expected = after.margin("A", "B").total - before.margin("A", "B").total
    assert result.margin_change == pytest.approx(expected)
    assert (
        result.transformer.direct + result.transformer.normalization
        == pytest.approx(expected, abs=1e-8)
    )
    assert abs(result.residual) < 1e-8


def test_routes_conserve_and_are_consistent_across_comparisons():
    rng = np.random.default_rng(742)
    for _ in range(50):
        before, after = (
            slate(tuple(rng.normal(size=3)), tuple(rng.normal(size=3)))
            for _ in range(2)
        )
        ab = decompose_transition(before, after, "A", "B")
        bc = decompose_transition(before, after, "B", "C")
        ac = decompose_transition(before, after, "A", "C")
        ba = decompose_transition(before, after, "B", "A")
        reverse = decompose_transition(after, before, "A", "B")
        assert abs(ab.residual) < 1e-12
        for branch in ("transformer", "gnn"):
            for route in ("direct_by_candidate", "normalization_by_candidate"):
                values = [
                    np.array(getattr(getattr(x, branch), route))
                    for x in (ab, bc, ac, ba, reverse)
                ]
                np.testing.assert_allclose(values[0] + values[1], values[2], atol=1e-12)
                np.testing.assert_allclose(values[0], -values[3], atol=1e-12)
                np.testing.assert_allclose(values[0], -values[4], atol=1e-12)


def test_ill_conditioned_routes_expose_error_instead_of_forcing_conservation():
    before = slate((0, 0, 0))
    after = slate((2e12, 1e12, 0))
    branch = decompose_transition(before, after, "A", "B").transformer
    assert branch.margin_change == pytest.approx(0.9185586535436917)
    # A floor-to-huge-scale transition has opposing ~1e17 route terms. Their
    # float64 cancellation loses precision; the measured margin remains intact.
    assert abs(branch.residual) > 0.1
    assert branch.residual == branch.margin_change - math.fsum(
        (branch.direct, branch.normalization)
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"candidate_ids": ("B", "A", "C")},
        {"candidate_mask": (True, True, False)},
        {"gnn_weight": 0.5},
        {"epsilon": 1e-5},
    ],
)
def test_transition_rejects_a_different_estimand(updates):
    before = slate()
    with pytest.raises(ValueError, match="preserve"):
        decompose_transition(before, replace(before, **updates), "A", "B")


@pytest.mark.parametrize(
    "updates",
    [
        {"gnn_weight": math.nan},
        {"gnn_weight": math.inf},
        {"gnn_weight": -0.1},
        {"gnn_weight": True},
        {"epsilon": 0},
        {"epsilon": math.inf},
        {"epsilon": math.nan},
        {"epsilon": 1e-320},
        {"transformer": (1, math.nan, 3)},
        {"gnn": (1, math.inf, 3)},
        {"transformer": (1, 2)},
        {"transformer": ("sensitive-score", 1, 2)},
        {"candidate_mask": (1, 1, 1)},
        {"candidate_mask": (True,)},
        {"candidate_ids": ("A", "A", "C")},
        {"candidate_ids": ("A", "", "C")},
        {"transformer": (1e308, 1e308, -1e308)},
    ],
)
def test_invalid_input_fails_without_echoing_values(updates):
    with pytest.raises(ValueError) as error:
        replace(slate(), **updates)
    assert "sensitive-score" not in str(error.value)


def test_input_copy_and_repr_do_not_expose_or_mutate_scores():
    raw = [2, 1, 0]
    scores = slate(raw)
    raw[0] = 123
    assert scores.transformer == (2, 1, 0)
    assert "transformer=" not in repr(scores)
    with pytest.raises(FrozenInstanceError):
        scores.gnn_weight = 0.5
    with pytest.raises(ValueError) as error:
        scores.margin("private-reference", "A")
    assert "private-reference" not in str(error.value)


@pytest.mark.parametrize(
    "updates",
    [
        {"candidate_ids": "ABC"},
        {"transformer": "123"},
        {"gnn": b"123"},
    ],
)
def test_string_vectors_are_rejected(updates):
    with pytest.raises(ValueError):
        replace(slate(), **updates)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_optional_production_fusion_parity(dtype_name):
    torch = pytest.importorskip("torch")
    from pipeline.gnn_training.fusion import late_fusion_logits

    cases = [
        slate(),
        slate((0, 0, 0), (1, 1, 1)),
        slate((0, 1e-8, 2e-8)),
        slate((1, 2, math.nan), candidate_mask=(True, True, False)),
        slate(candidate_mask=(False,) * 3),
    ]
    dtype = getattr(torch, dtype_name)
    for case in cases:
        actual = late_fusion_logits(
            torch.tensor([case.transformer], dtype=dtype),
            torch.tensor([case.gnn], dtype=dtype),
            gnn_weight=case.gnn_weight,
            candidate_mask=torch.tensor([case.candidate_mask]),
        ).numpy()[0]
        tolerance = 1e-6 if dtype_name == "float32" else 1e-12
        np.testing.assert_allclose(
            actual, case.fused_scores, atol=tolerance, rtol=tolerance
        )
