"""Tests for the torch-free validation ranking-metric accumulator."""

from __future__ import annotations

import math

import numpy as np

from pipeline.neural_training.metrics import RankingMetricAccumulator


def test_perfect_ranking_scores_one() -> None:
    accumulator = RankingMetricAccumulator(k=10)
    accumulator.update(
        labels=np.array([1.0, 0.0, 0.0]),
        scores=np.array([0.9, 0.5, 0.1]),
        tie_breaker=np.array([1.0, 2.0, 3.0]),
    )
    summary = accumulator.summary()
    assert summary["positive_ranking_group_count"] == 1
    assert math.isclose(summary["ndcg_at_k"], 1.0)
    assert math.isclose(summary["mrr_at_k"], 1.0)
    assert math.isclose(summary["hit_rate_at_k"], 1.0)


def test_zero_positive_group_is_ignored() -> None:
    accumulator = RankingMetricAccumulator(k=10)
    accumulator.update(
        labels=np.array([0.0, 0.0]),
        scores=np.array([0.9, 0.1]),
        tie_breaker=np.array([1.0, 2.0]),
    )
    summary = accumulator.summary()
    assert summary["positive_ranking_group_count"] == 0
    assert summary["ndcg_at_k"] == 0.0


def test_ndcg_penalizes_lower_rank() -> None:
    accumulator = RankingMetricAccumulator(k=10)
    # Single positive placed at rank 2 (score below the negative).
    accumulator.update(
        labels=np.array([1.0, 0.0]),
        scores=np.array([0.1, 0.9]),
        tie_breaker=np.array([1.0, 2.0]),
    )
    summary = accumulator.summary()
    expected_ndcg = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert math.isclose(summary["ndcg_at_k"], expected_ndcg)
    assert math.isclose(summary["mrr_at_k"], 0.5)


def test_tie_breaker_orders_equal_scores() -> None:
    accumulator = RankingMetricAccumulator(k=1)
    # Equal scores; the positive has the worse (higher) tie-breaker, so at k=1
    # the negative wins and the hit is missed.
    accumulator.update(
        labels=np.array([1.0, 0.0]),
        scores=np.array([0.5, 0.5]),
        tie_breaker=np.array([2.0, 1.0]),
    )
    summary = accumulator.summary()
    assert summary["hit_rate_at_k"] == 0.0
