import json
from pathlib import Path

from visualization.phase4_to_9 import generate_visualizations, phase_statuses


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_synthetic_reports(root: Path) -> None:
    write_json(
        root / "harmonization_coverage.json",
        {
            "coverage": [
                {
                    "domain": "labs",
                    "source": "mimiciv",
                    "row_count": 1000,
                    "mapping_status": "mapped",
                },
                {
                    "domain": "labs",
                    "source": "eicu_crd",
                    "row_count": 800,
                    "mapping_status": "mapped",
                },
            ]
        },
    )
    feature_manifest = {
        "status": "completed",
        "feature_rows_by_source": [
            {"source": "mimiciv", "split": "train", "row_count": 80},
            {"source": "mimiciv", "split": "validation", "row_count": 10},
            {"source": "eicu_crd", "split": "external", "row_count": 60},
        ],
        "split_counts": [
            {
                "source": "mimiciv",
                "split": "train",
                "stay_count": 80,
                "patient_count": 70,
            },
            {
                "source": "mimiciv",
                "split": "validation",
                "stay_count": 10,
                "patient_count": 9,
            },
            {
                "source": "eicu_crd",
                "split": "external",
                "stay_count": 60,
                "patient_count": 55,
            },
        ],
    }
    write_json(root / "milestone6_feature_manifest.json", feature_manifest)
    write_json(
        root / "phase8_p0_milestone6_feature_manifest.json",
        feature_manifest
        | {
            "feature_column_counts_by_family": {
                "demographic_context_columns": 5,
                "lab_columns": 7,
                "vital_columns": 5,
                "condition_columns": 4,
                "missingness_columns": 3,
                "total_columns": 24,
            }
        },
    )
    write_json(
        root / "training_table_manifest.json",
        {
            "status": "completed",
            "candidate_catalog_counts": {
                "candidate_count": 20,
                "condition_count": 4,
                "max_candidate_rank": 5,
            },
            "split_integrity": {
                "patient_count": 134,
                "patients_with_multiple_splits": 0,
            },
            "training_rows_by_source_split": [
                {
                    "source": "mimiciv",
                    "split": "train",
                    "row_count": 100,
                    "positive_row_count": 20,
                    "ranking_group_count": 25,
                },
                {
                    "source": "eicu_crd",
                    "split": "external",
                    "row_count": 40,
                    "positive_row_count": 0,
                    "ranking_group_count": 8,
                },
            ],
        },
    )
    write_json(
        root / "preprocessing_manifest.json",
        {
            "status": "completed",
            "fit_scope": {"source": "mimiciv", "split": "train"},
            "stay_numeric_columns": ["age_years"],
            "stay_categorical_columns": ["source"],
            "row_numeric_columns": ["candidate_rank"],
            "row_categorical_columns": ["index_condition_token"],
        },
    )
    write_json(
        root / "milestone7_coverage_report.json",
        {
            "status": "completed",
            "source_split_coverage": [
                {
                    "source": "eicu_crd",
                    "split": "external",
                    "performance_status": "coverage_only_no_in_catalog_positive_groups",
                }
            ],
        },
    )
    ranking_rows = []
    for baseline, ndcg in {
        "random": 0.1,
        "global_popularity": 0.2,
        "condition_popularity": 0.24,
        "linear": 0.25,
        "xgboost": 0.31,
    }.items():
        for split in ("validation", "test"):
            ranking_rows.append(
                {
                    "baseline_name": baseline,
                    "source": "mimiciv",
                    "split": split,
                    "k": 10,
                    "ndcg_at_k": ndcg,
                }
            )
    write_json(
        root / "milestone7_baseline_evaluation.json",
        {"status": "completed", "ranking_metrics": ranking_rows},
    )
    write_json(
        root / "milestone7_frozen_selection.json",
        {"status": "frozen", "selected_headline_baseline": "xgboost"},
    )
    write_json(
        root / "milestone8_graph_suitability.json",
        {
            "status": "completed",
            "node_counts": [
                {"node_type": "condition", "node_count": 4},
                {"node_type": "medication", "node_count": 7},
            ],
            "edge_counts_by_relation": [
                {
                    "relation_type": "condition_medication_train_positive",
                    "edge_count": 15,
                }
            ],
            "gate_review": {"result": "pass_for_graph_ablation"},
            "leakage_audit": {"status": "pass", "train_only_graph_fit": True},
        },
    )
    write_json(
        root / "milestone8b_graph_feature_manifest.json", {"status": "completed"}
    )
    ablation_rows = []
    for baseline, ndcg in {
        "xgboost_frozen_reference": 0.31,
        "graph_only_xgboost": 0.23,
        "xgboost_graph_augmented": 0.33,
        "late_fusion_validation_weighted": 0.30,
        "simple_ensemble_mean": 0.28,
    }.items():
        for split in ("validation", "test"):
            ablation_rows.append(
                {
                    "baseline_name": baseline,
                    "source": "mimiciv",
                    "split": split,
                    "k": 10,
                    "ndcg_at_k": ndcg,
                }
            )
    write_json(
        root / "milestone8b_ablation_evaluation.json",
        {"status": "completed", "ranking_metrics": ablation_rows},
    )
    write_json(
        root / "milestone8b_frozen_selection.json",
        {
            "status": "frozen",
            "selected_experiment": "xgboost_graph_augmented",
            "fusion_weight": {
                "selected_graph_weight": 0.0,
                "candidates": [
                    {"graph_weight": 0.0, "ndcg_at_k": 0.31, "mrr_at_k": 0.4},
                    {"graph_weight": 0.5, "ndcg_at_k": 0.27, "mrr_at_k": 0.35},
                    {"graph_weight": 1.0, "ndcg_at_k": 0.22, "mrr_at_k": 0.3},
                ],
            },
        },
    )


def test_phase_statuses_mark_done_and_planned() -> None:
    statuses = phase_statuses(
        {
            "milestone6_features": {"status": "completed"},
            "training_table": {
                "status": "completed",
                "split_integrity": {"patients_with_multiple_splits": 0},
            },
            "preprocessing": {"status": "completed"},
            "milestone7_evaluation": {"status": "completed"},
            "milestone7_frozen": {"status": "frozen"},
            "milestone8_suitability": {
                "gate_review": {"result": "pass_for_graph_ablation"},
                "leakage_audit": {"status": "pass"},
            },
            "milestone8b_frozen": {"status": "frozen"},
        }
    )

    assert statuses[0]["status"] == "completed"
    assert statuses[-1]["status"] == "planned"


def test_generate_visualizations_writes_aggregate_pack(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    figures_root = tmp_path / "visualization" / "figures"
    markdown_path = tmp_path / "visualization" / "meeting_figure_pack.md"
    summary_path = tmp_path / "visualization" / "meeting_figure_pack.json"
    write_synthetic_reports(reports_root)

    summary = generate_visualizations(
        reports_root=reports_root,
        figures_root=figures_root,
        markdown_path=markdown_path,
        summary_path=summary_path,
    )

    assert summary["data_safety"]["contains_patient_rows"] is False
    assert len(summary["figures"]) == 10
    assert markdown_path.exists()
    assert summary_path.exists()
    assert all(
        (tmp_path / "visualization" / figure["relative_path"]).exists()
        for figure in summary["figures"]
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Observed prescriptions are historical labels" in markdown
    assert "Phase 9 remains planned" in markdown
