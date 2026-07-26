"""In-memory ranking metrics for validation-time model selection.

These pure-NumPy helpers compute per-group NDCG@K, MRR@K, and hit-rate@K using
the same descending-score / ``candidate_rank`` tie-break the authoritative
DuckDB metric queries use (:mod:`pipeline.evaluate_baselines`). The DuckDB path
remains the source of truth for reported metrics; these helpers only drive early
stopping and the neural validation gate without materializing scores to Parquet
each epoch.

This module never imports PyTorch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RankingMetricAccumulator:
    """Accumulate per-group ranking metrics over positive groups only."""

    k: int
    positive_group_count: int = 0
    ndcg_sum: float = 0.0
    mrr_sum: float = 0.0
    hit_sum: float = 0.0

    def update(
        self,
        *,
        labels: np.ndarray,
        scores: np.ndarray,
        tie_breaker: np.ndarray,
    ) -> None:
        """Add one ranking group's contribution (ignored if it has no positive)."""

        positives = int(labels.sum())
        if positives <= 0:
            return
        order = np.lexsort((tie_breaker, -scores))
        ranked_labels = labels[order]
        window = ranked_labels[: self.k]
        hits = float(window.sum())

        discounts = 1.0 / np.log2(np.arange(2, window.shape[0] + 2))
        dcg = float((window * discounts).sum())
        ideal_hits = min(positives, self.k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

        positive_positions = np.nonzero(window > 0)[0]
        reciprocal_rank = (
            1.0 / (int(positive_positions[0]) + 1) if positive_positions.size else 0.0
        )

        self.positive_group_count += 1
        self.ndcg_sum += dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        self.mrr_sum += reciprocal_rank
        self.hit_sum += 1.0 if hits > 0 else 0.0

    def summary(self) -> dict[str, float | int]:
        count = self.positive_group_count
        if count == 0:
            return {
                "positive_ranking_group_count": 0,
                "ndcg_at_k": 0.0,
                "mrr_at_k": 0.0,
                "hit_rate_at_k": 0.0,
            }
        return {
            "positive_ranking_group_count": count,
            "ndcg_at_k": self.ndcg_sum / count,
            "mrr_at_k": self.mrr_sum / count,
            "hit_rate_at_k": self.hit_sum / count,
        }
