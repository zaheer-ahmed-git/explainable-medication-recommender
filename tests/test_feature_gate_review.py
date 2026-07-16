from __future__ import annotations

import json
from pathlib import Path

from pipeline.feature_gate_review import (
    FeatureGateReviewConfig,
    build_feature_gate_review,
)


def _write_report(path: Path, *, ndcg: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "ranking_metrics": [
                    {
                        "baseline_name": "xgboost_graph_augmented",
                        "source": "mimiciv",
                        "split": "validation",
                        "k": 10,
                        "ndcg_at_k": ndcg,
                        "mrr_at_k": 0.48,
                        "hit_rate_at_k": 0.84,
                    }
                ],
                "row_level_metrics": [
                    {
                        "baseline_name": "xgboost_graph_augmented",
                        "source": "mimiciv",
                        "split": "validation",
                        "average_precision": 0.2,
                        "roc_auc": 0.7,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_feature_gate_review_retains_canonical_on_subthreshold_lift(
    tmp_path: Path,
) -> None:
    phase8_report = tmp_path / "reports" / "phase8_p0_milestone8b.json"
    canonical_report = tmp_path / "reports" / "milestone8b.json"
    output_path = tmp_path / "reports" / "phase8_p0_feature_gate_review.json"
    _write_report(canonical_report, ndcg=0.3732)
    _write_report(phase8_report, ndcg=0.3770)

    review = build_feature_gate_review(
        FeatureGateReviewConfig(
            phase8_evaluation_report_path=phase8_report,
            canonical_frozen_selection_path=tmp_path / "reports" / "missing.json",
            canonical_evaluation_report_path=canonical_report,
            output_path=output_path,
        )
    )

    assert review["status"] == "completed"
    assert review["decision"] == "reject_inconclusive"
    assert review["promotion_ready"] is False
    assert review["comparison"]["primary"]["delta"] < 0.005
    assert review["data_safety"]["report_contains_patient_rows"] is False
    assert json.loads(output_path.read_text(encoding="utf-8"))["decision"] == (
        "reject_inconclusive"
    )
