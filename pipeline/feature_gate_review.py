"""Write aggregate Phase 8 P0 feature-promotion gate reviews."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from pipeline.config import REPORTS_ROOT


SCHEMA_VERSION = "phase8-p0-feature-gate-review-v1"
DEFAULT_PHASE8_EVALUATION_REPORT = (
    REPORTS_ROOT / "phase8_p0_milestone8b_ablation_evaluation.json"
)
DEFAULT_CANONICAL_FROZEN_SELECTION = REPORTS_ROOT / "milestone8b_frozen_selection.json"
DEFAULT_CANONICAL_EVALUATION_REPORT = (
    REPORTS_ROOT / "milestone8b_ablation_evaluation.json"
)
DEFAULT_OUTPUT_PATH = REPORTS_ROOT / "phase8_p0_feature_gate_review.json"
DEFAULT_CANDIDATE_NAME = "xgboost_graph_augmented"
DEFAULT_REFERENCE_NAME = "xgboost_graph_augmented"
DEFAULT_SELECTION_K = 10
MINIMUM_NDCG_LIFT = 0.005
MAXIMUM_SECONDARY_DROP = 0.01
SECONDARY_METRICS = ("mrr_at_k", "hit_rate_at_k", "average_precision", "roc_auc")


@dataclass(frozen=True)
class FeatureGateReviewConfig:
    """Configuration for a Phase 8 P0 promotion-gate review."""

    phase8_evaluation_report_path: Path = DEFAULT_PHASE8_EVALUATION_REPORT
    canonical_frozen_selection_path: Path = DEFAULT_CANONICAL_FROZEN_SELECTION
    canonical_evaluation_report_path: Path = DEFAULT_CANONICAL_EVALUATION_REPORT
    output_path: Path = DEFAULT_OUTPUT_PATH
    candidate_name: str = DEFAULT_CANDIDATE_NAME
    reference_name: str = DEFAULT_REFERENCE_NAME
    selection_k: int = DEFAULT_SELECTION_K
    minimum_ndcg_lift: float = MINIMUM_NDCG_LIFT
    maximum_secondary_drop: float = MAXIMUM_SECONDARY_DROP


def load_json_if_present(path: Path) -> dict[str, Any] | None:
    """Load a JSON object if it exists."""

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metric_value(row: dict[str, Any] | None, metric: str) -> float | None:
    if row is None:
        return None
    value = row.get(metric)
    return float(value) if value is not None else None


def _find_ranking_metrics(
    report: dict[str, Any] | None,
    *,
    baseline_name: str,
    selection_k: int,
) -> dict[str, Any] | None:
    if report is None:
        return None
    for row in report.get("ranking_metrics", []):
        if (
            row.get("baseline_name") == baseline_name
            and row.get("source") == "mimiciv"
            and row.get("split") == "validation"
            and int(row.get("k", -1)) == selection_k
        ):
            return row
    return None


def _find_row_metrics(
    report: dict[str, Any] | None,
    *,
    baseline_name: str,
) -> dict[str, Any] | None:
    if report is None:
        return None
    for row in report.get("row_level_metrics", []):
        if (
            row.get("baseline_name") == baseline_name
            and row.get("source") == "mimiciv"
            and row.get("split") == "validation"
        ):
            return row
    return None


def _candidate_metrics(
    report: dict[str, Any] | None,
    *,
    baseline_name: str,
    selection_k: int,
) -> dict[str, Any] | None:
    ranking = _find_ranking_metrics(
        report,
        baseline_name=baseline_name,
        selection_k=selection_k,
    )
    if ranking is None:
        return None
    row_metrics = _find_row_metrics(report, baseline_name=baseline_name)
    return {
        "baseline_name": baseline_name,
        "source": "mimiciv",
        "split": "validation",
        "k": selection_k,
        "ndcg_at_k": _metric_value(ranking, "ndcg_at_k"),
        "mrr_at_k": _metric_value(ranking, "mrr_at_k"),
        "hit_rate_at_k": _metric_value(ranking, "hit_rate_at_k"),
        "average_precision": _metric_value(row_metrics, "average_precision"),
        "roc_auc": _metric_value(row_metrics, "roc_auc"),
    }


def _metrics_from_frozen_selection(
    frozen_selection: dict[str, Any] | None,
    *,
    baseline_name: str,
    selection_k: int,
) -> dict[str, Any] | None:
    if frozen_selection is None:
        return None
    basis = frozen_selection.get("selection_basis", {})
    ranking = basis.get("selected_metrics")
    if not isinstance(ranking, dict) or ranking.get("baseline_name") != baseline_name:
        return None
    if int(ranking.get("k", selection_k)) != selection_k:
        return None
    row_metrics = basis.get("selected_row_level_metrics")
    return {
        "baseline_name": baseline_name,
        "source": "mimiciv",
        "split": "validation",
        "k": selection_k,
        "ndcg_at_k": _metric_value(ranking, "ndcg_at_k"),
        "mrr_at_k": _metric_value(ranking, "mrr_at_k"),
        "hit_rate_at_k": _metric_value(ranking, "hit_rate_at_k"),
        "average_precision": _metric_value(row_metrics, "average_precision"),
        "roc_auc": _metric_value(row_metrics, "roc_auc"),
    }


def _reference_metrics(
    *,
    frozen_selection: dict[str, Any] | None,
    evaluation_report: dict[str, Any] | None,
    baseline_name: str,
    selection_k: int,
) -> dict[str, Any] | None:
    return _metrics_from_frozen_selection(
        frozen_selection,
        baseline_name=baseline_name,
        selection_k=selection_k,
    ) or _candidate_metrics(
        evaluation_report,
        baseline_name=baseline_name,
        selection_k=selection_k,
    )


def _passes_secondary_metric(
    *,
    candidate_value: float | None,
    reference_value: float | None,
    maximum_drop: float,
) -> bool:
    if reference_value is None:
        return True
    if candidate_value is None:
        return False
    return candidate_value >= reference_value - maximum_drop


def metric_comparison(
    *,
    candidate: dict[str, Any],
    reference: dict[str, Any],
    minimum_ndcg_lift: float,
    maximum_secondary_drop: float,
) -> dict[str, Any]:
    """Return metric deltas and pass/fail details."""

    primary_candidate = _metric_value(candidate, "ndcg_at_k")
    primary_reference = _metric_value(reference, "ndcg_at_k")
    primary_delta = (
        None
        if primary_candidate is None or primary_reference is None
        else primary_candidate - primary_reference
    )
    secondary = {}
    for metric in SECONDARY_METRICS:
        candidate_value = _metric_value(candidate, metric)
        reference_value = _metric_value(reference, metric)
        delta = (
            None
            if candidate_value is None or reference_value is None
            else candidate_value - reference_value
        )
        secondary[metric] = {
            "candidate": candidate_value,
            "reference": reference_value,
            "delta": delta,
            "passes": _passes_secondary_metric(
                candidate_value=candidate_value,
                reference_value=reference_value,
                maximum_drop=maximum_secondary_drop,
            ),
        }
    return {
        "primary_metric": "ndcg_at_k",
        "primary": {
            "candidate": primary_candidate,
            "reference": primary_reference,
            "delta": primary_delta,
            "minimum_lift": minimum_ndcg_lift,
            "passes": (
                primary_delta is not None and primary_delta >= minimum_ndcg_lift
            ),
        },
        "secondary": secondary,
    }


def build_feature_gate_review(
    config: FeatureGateReviewConfig = FeatureGateReviewConfig(),
) -> dict[str, Any]:
    """Build and write an aggregate Phase 8 P0 promotion-gate review."""

    generated_at = datetime.now(UTC).isoformat()
    phase8_report = load_json_if_present(config.phase8_evaluation_report_path)
    frozen_selection = load_json_if_present(config.canonical_frozen_selection_path)
    canonical_report = load_json_if_present(config.canonical_evaluation_report_path)
    missing_inputs = [
        {
            "name": "phase8_evaluation_report",
            "path": str(config.phase8_evaluation_report_path),
        }
        for report in (phase8_report,)
        if report is None
    ]
    reference = _reference_metrics(
        frozen_selection=frozen_selection,
        evaluation_report=canonical_report,
        baseline_name=config.reference_name,
        selection_k=config.selection_k,
    )
    candidate = _candidate_metrics(
        phase8_report,
        baseline_name=config.candidate_name,
        selection_k=config.selection_k,
    )
    if reference is None:
        missing_inputs.append(
            {
                "name": "canonical_reference_metrics",
                "path": str(config.canonical_frozen_selection_path),
                "fallback_path": str(config.canonical_evaluation_report_path),
            }
        )
    if candidate is None:
        missing_inputs.append(
            {
                "name": "phase8_candidate_metrics",
                "path": str(config.phase8_evaluation_report_path),
            }
        )
    if missing_inputs:
        review = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked_missing_inputs",
            "generated_at": generated_at,
            "decision": "reject_inconclusive",
            "missing_inputs": missing_inputs,
            "data_safety": {
                "report_contains_patient_rows": False,
                "report_contains_row_samples": False,
                "metrics_are_aggregate_only": True,
            },
        }
        write_json(config.output_path, review)
        return review

    assert candidate is not None
    assert reference is not None
    comparison = metric_comparison(
        candidate=candidate,
        reference=reference,
        minimum_ndcg_lift=config.minimum_ndcg_lift,
        maximum_secondary_drop=config.maximum_secondary_drop,
    )
    primary_pass = bool(comparison["primary"]["passes"])
    secondary_pass = all(
        bool(row["passes"]) for row in comparison["secondary"].values()
    )
    promote = primary_pass and secondary_pass
    review = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "generated_at": generated_at,
        "decision": "promote" if promote else "reject_inconclusive",
        "promotion_ready": promote,
        "reason": (
            "Phase 8 P0 cleared the validation NDCG@10 lift gate without "
            "secondary metric drops over threshold."
            if promote
            else "Phase 8 P0 did not clear the validation lift and secondary-metric gate."
        ),
        "candidate": candidate,
        "canonical_reference": reference,
        "comparison": comparison,
        "gate_rules": {
            "primary_metric": "mimic_validation_ndcg_at_10",
            "minimum_ndcg_lift": config.minimum_ndcg_lift,
            "secondary_metrics": list(SECONDARY_METRICS),
            "maximum_secondary_drop": config.maximum_secondary_drop,
            "promote_action": (
                "Promote isolated phase8_p0 roots only after explicit review; "
                "otherwise keep default canonical artifacts unchanged."
            ),
        },
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
            "metrics_are_aggregate_only": True,
        },
    }
    write_json(config.output_path, review)
    return review


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a Phase 8 P0 aggregate feature-promotion gate review.",
    )
    parser.add_argument(
        "--phase8-evaluation-report",
        type=Path,
        default=DEFAULT_PHASE8_EVALUATION_REPORT,
        help="Phase 8 P0 Milestone 8B development evaluation report.",
    )
    parser.add_argument(
        "--canonical-frozen-selection",
        type=Path,
        default=DEFAULT_CANONICAL_FROZEN_SELECTION,
        help="Current canonical Milestone 8B frozen-selection report.",
    )
    parser.add_argument(
        "--canonical-evaluation-report",
        type=Path,
        default=DEFAULT_CANONICAL_EVALUATION_REPORT,
        help="Fallback current canonical Milestone 8B evaluation report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for the aggregate feature-gate review.",
    )
    parser.add_argument(
        "--candidate-name",
        default=DEFAULT_CANDIDATE_NAME,
        help="Phase 8 candidate baseline name to compare.",
    )
    parser.add_argument(
        "--reference-name",
        default=DEFAULT_REFERENCE_NAME,
        help="Canonical reference baseline name to compare.",
    )
    parser.add_argument(
        "--selection-k",
        type=int,
        default=DEFAULT_SELECTION_K,
        help="Ranking cutoff used for the primary NDCG gate.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    review = build_feature_gate_review(
        FeatureGateReviewConfig(
            phase8_evaluation_report_path=args.phase8_evaluation_report,
            canonical_frozen_selection_path=args.canonical_frozen_selection,
            canonical_evaluation_report_path=args.canonical_evaluation_report,
            output_path=args.output,
            candidate_name=args.candidate_name,
            reference_name=args.reference_name,
            selection_k=args.selection_k,
        )
    )
    print(
        "Wrote Phase 8 P0 feature-gate review: "
        f"status={review['status']}, decision={review['decision']}"
    )
    return 0 if review["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
