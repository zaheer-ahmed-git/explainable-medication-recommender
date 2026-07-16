from __future__ import annotations

import json
from pathlib import Path

from pipeline.graph_ablation import GraphAblationConfig, build_graph_ablation
from tests.milestone6_helpers import read_parquet_rows, write_parquet_rows


def _candidate_rows() -> tuple[tuple[object, ...], ...]:
    return (
        (
            "mimiciv",
            "train",
            "stay-secret-train-1",
            "rg-secret-train-1",
            "condition:a",
            "rxnorm:a",
            1,
            True,
        ),
        (
            "mimiciv",
            "train",
            "stay-secret-train-1",
            "rg-secret-train-1",
            "condition:a",
            "rxnorm:b",
            2,
            False,
        ),
        (
            "mimiciv",
            "train",
            "stay-secret-train-1",
            "rg-secret-train-1",
            "condition:a",
            "rxnorm:c",
            3,
            False,
        ),
        (
            "mimiciv",
            "train",
            "stay-secret-train-2",
            "rg-secret-train-2",
            "condition:a",
            "rxnorm:a",
            1,
            False,
        ),
        (
            "mimiciv",
            "train",
            "stay-secret-train-2",
            "rg-secret-train-2",
            "condition:a",
            "rxnorm:b",
            2,
            True,
        ),
        (
            "mimiciv",
            "train",
            "stay-secret-train-2",
            "rg-secret-train-2",
            "condition:a",
            "rxnorm:c",
            3,
            False,
        ),
        (
            "mimiciv",
            "validation",
            "stay-secret-val-1",
            "rg-secret-val-1",
            "condition:a",
            "rxnorm:a",
            1,
            True,
        ),
        (
            "mimiciv",
            "validation",
            "stay-secret-val-1",
            "rg-secret-val-1",
            "condition:a",
            "rxnorm:b",
            2,
            False,
        ),
        (
            "mimiciv",
            "validation",
            "stay-secret-val-1",
            "rg-secret-val-1",
            "condition:a",
            "rxnorm:c",
            3,
            False,
        ),
        (
            "mimiciv",
            "test",
            "stay-secret-test-1",
            "rg-secret-test-1",
            "condition:a",
            "rxnorm:a",
            1,
            True,
        ),
        (
            "mimiciv",
            "test",
            "stay-secret-test-1",
            "rg-secret-test-1",
            "condition:a",
            "rxnorm:b",
            2,
            False,
        ),
        (
            "eicu_crd",
            "external",
            "stay-secret-external-1",
            "rg-secret-external-1",
            "condition:external",
            "rxnorm:z",
            1,
            False,
        ),
    )


def _write_graph_ablation_fixture(tmp_path: Path) -> dict[str, Path]:
    features_root = tmp_path / "Dataset" / "processed" / "features"
    training_root = tmp_path / "Dataset" / "processed" / "training"
    graph_root = tmp_path / "Dataset" / "processed" / "graph" / "milestone8"
    milestone7_root = tmp_path / "Dataset" / "processed" / "evaluation" / "milestone7"
    evaluation_root = tmp_path / "Dataset" / "processed" / "evaluation" / "milestone8b"
    reports_root = tmp_path / "reports"

    write_parquet_rows(
        features_root / "patient_stay_features.parquet",
        (
            "source",
            "split",
            "stay_uid",
            "age_years",
            "sex",
            "lab_event_count_24h",
            "lab_lactate_slope_24h",
            "condition_ccsr_INF002_present_24h",
            "vital_event_count_24h",
            "allergy_event_count_24h",
            "predecision_intervention_count_24h",
        ),
        (
            ("mimiciv", "train", "stay-secret-train-1", 64.0, "F", 3, 0.1, 1, 1, 0, 1),
            ("mimiciv", "train", "stay-secret-train-2", 71.0, "M", 1, None, 1, 0, 1, 0),
            (
                "mimiciv",
                "validation",
                "stay-secret-val-1",
                58.0,
                "F",
                2,
                0.2,
                0,
                2,
                0,
                0,
            ),
            ("mimiciv", "test", "stay-secret-test-1", 75.0, "F", 1, 0.05, 1, 0, 0, 0),
            (
                "eicu_crd",
                "external",
                "stay-secret-external-1",
                69.0,
                "M",
                2,
                0.03,
                0,
                1,
                1,
                0,
            ),
        ),
    )
    write_parquet_rows(
        training_root / "candidate_catalog.parquet",
        (
            "index_condition_token",
            "candidate_medication_token",
            "candidate_rank",
            "positive_train_stay_count",
        ),
        (
            ("condition:a", "rxnorm:a", 1, 1),
            ("condition:a", "rxnorm:b", 2, 1),
            ("condition:a", "rxnorm:c", 3, 0),
        ),
    )
    write_parquet_rows(
        training_root / "patient_condition_medication.parquet",
        (
            "source",
            "split",
            "stay_uid",
            "ranking_group_id",
            "index_condition_token",
            "candidate_medication_token",
            "candidate_rank",
            "label_prescribed",
        ),
        _candidate_rows(),
    )
    write_parquet_rows(
        graph_root / "graph_edges.parquet",
        (
            "src_id",
            "dst_id",
            "src_type",
            "dst_type",
            "relation_type",
            "support_count",
            "fit_source",
            "fit_split",
            "graph_version",
            "feature_version",
            "label_version",
            "split_version",
            "generated_at",
        ),
        (
            (
                "condition|condition:a",
                "medication|rxnorm:a",
                "condition",
                "medication",
                "condition_medication_train_positive",
                3,
                "mimiciv",
                "train",
                "graph-suitability-v1",
                "temporal-features-v1",
                "observed-medication-label-v1",
                "patient-split-v1",
                "2026-07-10T00:00:00+00:00",
            ),
            (
                "condition|condition:a",
                "medication|rxnorm:b",
                "condition",
                "medication",
                "condition_medication_train_positive",
                1,
                "mimiciv",
                "train",
                "graph-suitability-v1",
                "temporal-features-v1",
                "observed-medication-label-v1",
                "patient-split-v1",
                "2026-07-10T00:00:00+00:00",
            ),
            (
                "condition|condition:a",
                "lab|lactate",
                "condition",
                "lab",
                "condition_lab_predecision",
                4,
                "mimiciv",
                "train",
                "graph-suitability-v1",
                "temporal-features-v1",
                "observed-medication-label-v1",
                "patient-split-v1",
                "2026-07-10T00:00:00+00:00",
            ),
            (
                "condition|condition:a",
                "vital|heart_rate",
                "condition",
                "vital",
                "condition_vital_predecision",
                4,
                "mimiciv",
                "train",
                "graph-suitability-v1",
                "temporal-features-v1",
                "observed-medication-label-v1",
                "patient-split-v1",
                "2026-07-10T00:00:00+00:00",
            ),
            (
                "medication|rxnorm:a",
                "medication|rxnorm:b",
                "medication",
                "medication",
                "medication_medication_train_coprescribed",
                2,
                "mimiciv",
                "train",
                "graph-suitability-v1",
                "temporal-features-v1",
                "observed-medication-label-v1",
                "patient-split-v1",
                "2026-07-10T00:00:00+00:00",
            ),
        ),
    )

    baseline_rows = []
    baseline_scores = {
        "rxnorm:a": 0.9,
        "rxnorm:b": 0.6,
        "rxnorm:c": 0.2,
        "rxnorm:z": 0.1,
    }
    for row in _candidate_rows():
        (
            source,
            split,
            _stay_uid,
            ranking_group_id,
            condition_token,
            medication_token,
            candidate_rank,
            label_prescribed,
        ) = row
        baseline_rows.append(
            (
                source,
                split,
                ranking_group_id,
                condition_token,
                medication_token,
                candidate_rank,
                label_prescribed,
                "xgboost",
                baseline_scores[str(medication_token)],
                20260617,
                "baseline-ranking-v1",
                "milestone7-evaluation-v1",
                "2026-07-10T00:00:00+00:00",
            )
        )
    write_parquet_rows(
        milestone7_root / "baseline_scores.parquet",
        (
            "source",
            "split",
            "ranking_group_id",
            "index_condition_token",
            "candidate_medication_token",
            "candidate_rank",
            "label_prescribed",
            "baseline_name",
            "score",
            "seed",
            "baseline_version",
            "evaluation_version",
            "generated_at",
        ),
        tuple(baseline_rows),
    )

    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "milestone6_feature_manifest.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )
    (reports_root / "training_table_manifest.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )
    (reports_root / "milestone7_baseline_evaluation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "parameters": {"mode": "final", "frozen_selection": True},
            }
        ),
        encoding="utf-8",
    )
    (reports_root / "milestone8_graph_suitability.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "gate_review": {"result": "pass_for_graph_ablation"},
                "leakage_audit": {"status": "pass"},
            }
        ),
        encoding="utf-8",
    )

    return {
        "features_root": features_root,
        "training_root": training_root,
        "graph_root": graph_root,
        "milestone7_root": milestone7_root,
        "evaluation_root": evaluation_root,
        "reports_root": reports_root,
    }


def _config(
    paths: dict[str, Path], *, mode: str = "development"
) -> GraphAblationConfig:
    reports_root = paths["reports_root"]
    return GraphAblationConfig(
        features_root=paths["features_root"],
        training_root=paths["training_root"],
        graph_root=paths["graph_root"],
        milestone7_evaluation_root=paths["milestone7_root"],
        evaluation_root=paths["evaluation_root"],
        feature_manifest_path=reports_root / "milestone8b_graph_feature_manifest.json",
        evaluation_report_path=reports_root / "milestone8b_ablation_evaluation.json",
        frozen_selection_path=reports_root / "milestone8b_frozen_selection.json",
        milestone6_feature_manifest_path=reports_root
        / "milestone6_feature_manifest.json",
        training_manifest_path=reports_root / "training_table_manifest.json",
        milestone7_evaluation_report_path=reports_root
        / "milestone7_baseline_evaluation.json",
        milestone8_suitability_report_path=reports_root
        / "milestone8_graph_suitability.json",
        top_k=(1, 3, 10),
        mode=mode,
        frozen_selection=(mode == "final"),
        min_subgroup_positive_groups=1,
    )


def test_graph_ablation_builds_graph_features_scores_and_safe_reports(
    tmp_path: Path,
) -> None:
    paths = _write_graph_ablation_fixture(tmp_path)
    config = _config(paths)

    manifest = build_graph_ablation(config)

    assert manifest["status"] == "completed"
    assert manifest["parameters"]["mode"] == "development"
    assert manifest["frozen_selection"]["status"] == "frozen"
    assert manifest["frozen_selection"]["selection_basis"]["split"] == (
        "mimiciv_validation"
    )

    feature_rows = read_parquet_rows(config.graph_feature_matrix_path)
    direct_feature = next(
        row
        for row in feature_rows
        if row["source"] == "mimiciv"
        and row["split"] == "validation"
        and row["candidate_medication_token"] == "rxnorm:a"
    )
    assert direct_feature["graph_condition_medication_support_count"] == 3
    assert direct_feature["graph_direct_edge_present"] == 1

    external_feature = next(row for row in feature_rows if row["source"] == "eicu_crd")
    assert external_feature["graph_condition_in_graph"] == 0
    assert external_feature["graph_candidate_in_graph"] == 0

    score_rows = read_parquet_rows(config.score_output_path)
    assert {row["split"] for row in score_rows} == {
        "external",
        "train",
        "validation",
    }
    assert {row["baseline_name"] for row in score_rows} == {
        "xgboost_frozen_reference",
        "graph_only_xgboost",
        "xgboost_graph_augmented",
        "late_fusion_validation_weighted",
        "simple_ensemble_mean",
    }
    assert all(0.0 <= float(row["score"]) <= 1.0 for row in score_rows)
    graph_augmented = next(
        experiment
        for experiment in manifest["graph_ablation_models"]["experiments"]
        if experiment["experiment_name"] == "xgboost_graph_augmented"
    )
    assert "condition_ccsr_INF002_present_24h" in graph_augmented["feature_columns"]
    assert "lab_lactate_slope_24h" in graph_augmented["feature_columns"]
    assert "stay_uid" not in graph_augmented["feature_columns"]

    external_metrics = [
        row
        for row in manifest["row_level_metrics"]
        if row["source"] == "eicu_crd" and row["split"] == "external"
    ]
    assert external_metrics
    assert all(row["average_precision"] is None for row in external_metrics)

    for report_path in (
        config.feature_manifest_path,
        config.evaluation_report_path,
        config.frozen_selection_path,
    ):
        text = report_path.read_text(encoding="utf-8")
        assert "stay-secret" not in text
        assert "rg-secret" not in text
        parsed = json.loads(text)
        if "data_safety" in parsed:
            assert parsed["data_safety"]["report_contains_patient_rows"] is False


def test_graph_ablation_blocks_final_without_frozen_selection(
    tmp_path: Path,
) -> None:
    paths = _write_graph_ablation_fixture(tmp_path)
    config = _config(paths, mode="final")

    manifest = build_graph_ablation(config)

    assert manifest["status"] == "blocked_final_requires_frozen_selection"
    assert "row_level_metrics" not in manifest
    assert "requires --frozen-selection" in manifest["reason"]


def test_graph_ablation_final_mode_uses_frozen_selection_and_scores_test(
    tmp_path: Path,
) -> None:
    paths = _write_graph_ablation_fixture(tmp_path)
    development_config = _config(paths)
    development_manifest = build_graph_ablation(development_config)
    assert development_manifest["status"] == "completed"

    final_config = _config(paths, mode="final")
    final_manifest = build_graph_ablation(final_config)

    assert final_manifest["status"] == "completed"
    assert final_manifest["parameters"]["mode"] == "final"
    assert final_manifest["frozen_selection"]["status"] == "frozen"
    score_rows = read_parquet_rows(final_config.score_output_path)
    assert "test" in {row["split"] for row in score_rows}
    test_metrics = [
        row
        for row in final_manifest["ranking_metrics"]
        if row["source"] == "mimiciv" and row["split"] == "test"
    ]
    assert test_metrics


def test_graph_ablation_blocks_when_graph_gate_has_not_passed(tmp_path: Path) -> None:
    paths = _write_graph_ablation_fixture(tmp_path)
    reports_root = paths["reports_root"]
    (reports_root / "milestone8_graph_suitability.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "gate_review": {"result": "blocked_graph_not_ready"},
                "leakage_audit": {"status": "pass"},
            }
        ),
        encoding="utf-8",
    )
    config = _config(paths)

    manifest = build_graph_ablation(config)

    assert manifest["status"] == "blocked_graph_gate_not_passed"
    assert "graph gate did not pass" in manifest["reason"]
