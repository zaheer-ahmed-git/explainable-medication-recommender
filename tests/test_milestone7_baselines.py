from __future__ import annotations

import json
from pathlib import Path

import duckdb

from pipeline.build_training_table import (
    TrainingTableBuildConfig,
    build_training_artifacts,
)
from pipeline.evaluate_baselines import (
    BaselineEvaluationConfig,
    build_baseline_evaluation,
    group_ranking_metrics,
    row_level_classification_metrics,
)
from pipeline.features import FeatureBuildConfig, build_feature_artifacts
from pipeline.learned_baselines import (
    resolve_feature_spec,
    training_sample_query,
    write_training_sample,
)
from tests.milestone6_helpers import (
    read_parquet_rows,
    write_milestone6_harmonized_fixture,
    write_parquet_rows,
)


def write_milestone7_fixture(tmp_path: Path) -> dict[str, Path]:
    features_root = tmp_path / "Dataset" / "processed" / "features"
    training_root = tmp_path / "Dataset" / "processed" / "training"
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
            (
                "mimiciv",
                "validation",
                "stay-secret-val-2",
                66.0,
                "M",
                0,
                None,
                0,
                1,
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
            ("project:sepsis", "rxnorm:a", 1, 1),
            ("project:sepsis", "rxnorm:b", 2, 1),
            ("project:sepsis", "rxnorm:c", 3, 0),
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
        (
            (
                "mimiciv",
                "train",
                "stay-secret-train-1",
                "stay-secret-train-1|project:sepsis",
                "project:sepsis",
                "rxnorm:a",
                1,
                True,
            ),
            (
                "mimiciv",
                "train",
                "stay-secret-train-1",
                "stay-secret-train-1|project:sepsis",
                "project:sepsis",
                "rxnorm:b",
                2,
                False,
            ),
            (
                "mimiciv",
                "train",
                "stay-secret-train-1",
                "stay-secret-train-1|project:sepsis",
                "project:sepsis",
                "rxnorm:c",
                3,
                False,
            ),
            (
                "mimiciv",
                "train",
                "stay-secret-train-2",
                "stay-secret-train-2|project:sepsis",
                "project:sepsis",
                "rxnorm:a",
                1,
                False,
            ),
            (
                "mimiciv",
                "train",
                "stay-secret-train-2",
                "stay-secret-train-2|project:sepsis",
                "project:sepsis",
                "rxnorm:b",
                2,
                True,
            ),
            (
                "mimiciv",
                "train",
                "stay-secret-train-2",
                "stay-secret-train-2|project:sepsis",
                "project:sepsis",
                "rxnorm:c",
                3,
                False,
            ),
            (
                "mimiciv",
                "validation",
                "stay-secret-val-1",
                "stay-secret-val-1|project:sepsis",
                "project:sepsis",
                "rxnorm:a",
                1,
                False,
            ),
            (
                "mimiciv",
                "validation",
                "stay-secret-val-1",
                "stay-secret-val-1|project:sepsis",
                "project:sepsis",
                "rxnorm:b",
                2,
                False,
            ),
            (
                "mimiciv",
                "validation",
                "stay-secret-val-1",
                "stay-secret-val-1|project:sepsis",
                "project:sepsis",
                "rxnorm:c",
                3,
                True,
            ),
            (
                "mimiciv",
                "validation",
                "stay-secret-val-2",
                "stay-secret-val-2|project:sepsis",
                "project:sepsis",
                "rxnorm:a",
                1,
                False,
            ),
            (
                "mimiciv",
                "validation",
                "stay-secret-val-2",
                "stay-secret-val-2|project:sepsis",
                "project:sepsis",
                "rxnorm:b",
                2,
                False,
            ),
            (
                "mimiciv",
                "validation",
                "stay-secret-val-2",
                "stay-secret-val-2|project:sepsis",
                "project:sepsis",
                "rxnorm:c",
                3,
                False,
            ),
            (
                "mimiciv",
                "test",
                "stay-secret-test-1",
                "stay-secret-test-1|project:sepsis",
                "project:sepsis",
                "rxnorm:a",
                1,
                True,
            ),
            (
                "mimiciv",
                "test",
                "stay-secret-test-1",
                "stay-secret-test-1|project:sepsis",
                "project:sepsis",
                "rxnorm:b",
                2,
                False,
            ),
            (
                "eicu_crd",
                "external",
                "stay-secret-external-1",
                "stay-secret-external-1|project:sepsis",
                "project:sepsis",
                "rxnorm:a",
                1,
                False,
            ),
            (
                "eicu_crd",
                "external",
                "stay-secret-external-1",
                "stay-secret-external-1|project:sepsis",
                "project:sepsis",
                "rxnorm:b",
                2,
                False,
            ),
        ),
    )
    training_manifest = {
        "out_of_catalog_positives": [
            {
                "source": "eicu_crd",
                "split": "external",
                "positive_condition_medication_stay_count": 4,
                "out_of_catalog_positive_stay_count": 4,
            }
        ],
        "training_rows_by_source_split": [
            {
                "source": "mimiciv",
                "split": "validation",
                "row_count": 6,
                "positive_row_count": 1,
                "ranking_group_count": 2,
            }
        ],
    }
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / "training_table_manifest.json").write_text(
        json.dumps(training_manifest),
        encoding="utf-8",
    )
    return {
        "features_root": features_root,
        "training_root": training_root,
        "reports_root": reports_root,
    }


def milestone7_config(
    tmp_path: Path, *, baselines: tuple[str, ...] | None = None
) -> BaselineEvaluationConfig:
    paths = write_milestone7_fixture(tmp_path)
    return BaselineEvaluationConfig(
        features_root=paths["features_root"],
        training_root=paths["training_root"],
        evaluation_root=tmp_path / "Dataset" / "processed" / "evaluation",
        coverage_report_path=paths["reports_root"] / "milestone7_coverage_report.json",
        evaluation_report_path=paths["reports_root"]
        / "milestone7_baseline_evaluation.json",
        training_manifest_path=paths["reports_root"] / "training_table_manifest.json",
        top_k=(1, 3, 10),
        min_subgroup_positive_groups=2,
        baselines=baselines or BaselineEvaluationConfig().baselines,
    )


def test_group_ranking_metrics_handles_large_k_and_no_positive_groups() -> None:
    metrics = group_ranking_metrics([False, True, True], [0.9, 0.8, 0.1], top_k=(1, 5))
    assert metrics[1]["precision_at_k"] == 0.0
    assert metrics[5]["precision_at_k"] == 2 / 3
    assert metrics[5]["recall_at_k"] == 1.0
    assert metrics[5]["hit_rate_at_k"] == 1.0

    no_positive = group_ranking_metrics([False, False], [0.7, 0.2], top_k=(3,))
    assert no_positive[3]["precision_at_k"] == 0.0
    assert no_positive[3]["recall_at_k"] == 0.0
    assert no_positive[3]["ndcg_at_k"] == 0.0


def test_row_level_classification_metrics_return_null_for_single_class() -> None:
    all_negative = row_level_classification_metrics(
        [False, False, False], [0.2, 0.5, 0.8]
    )
    assert all_negative["average_precision"] is None
    assert all_negative["roc_auc"] is None
    assert all_negative["brier_score"] is None

    mixed = row_level_classification_metrics(
        [False, True, True, False], [0.1, 0.9, 0.8, 0.2]
    )
    assert mixed["average_precision"] is not None
    assert mixed["roc_auc"] is not None
    assert mixed["brier_score"] is not None


def test_development_evaluation_scores_nonlearned_baselines_safely(
    tmp_path: Path,
) -> None:
    config = milestone7_config(tmp_path)

    manifest = build_baseline_evaluation(config)

    assert manifest["status"] == "completed"
    assert manifest["learned_baselines"]["status"] == "not_requested"
    score_rows = read_parquet_rows(config.score_output_path)
    assert {row["split"] for row in score_rows} == {"train", "validation"}
    assert {row["baseline_name"] for row in score_rows} == {
        "random",
        "global_popularity",
        "condition_popularity",
    }

    condition_scores = [
        row
        for row in score_rows
        if row["baseline_name"] == "condition_popularity"
        and row["split"] == "validation"
        and row["candidate_medication_token"] == "rxnorm:c"
    ]
    assert condition_scores
    assert all(row["score"] == 0.0 for row in condition_scores)

    random_scores_by_key = {
        (
            row["ranking_group_id"],
            row["candidate_medication_token"],
        ): row["score"]
        for row in score_rows
        if row["baseline_name"] == "random"
    }
    second_config = BaselineEvaluationConfig(
        features_root=config.features_root,
        training_root=config.training_root,
        evaluation_root=tmp_path / "second_eval",
        coverage_report_path=tmp_path / "second_reports" / "coverage.json",
        evaluation_report_path=tmp_path / "second_reports" / "evaluation.json",
        training_manifest_path=config.training_manifest_path,
        top_k=(1, 3, 10),
        min_subgroup_positive_groups=2,
    )
    build_baseline_evaluation(second_config)
    second_random_scores = {
        (
            row["ranking_group_id"],
            row["candidate_medication_token"],
        ): row["score"]
        for row in read_parquet_rows(second_config.score_output_path)
        if row["baseline_name"] == "random"
    }
    assert random_scores_by_key == second_random_scores

    coverage = json.loads(config.coverage_report_path.read_text(encoding="utf-8"))
    external = next(
        row for row in coverage["source_split_coverage"] if row["source"] == "eicu_crd"
    )
    assert (
        external["performance_status"] == "coverage_only_no_in_catalog_positive_groups"
    )
    assert coverage["data_safety"]["report_contains_patient_rows"] is False

    report_text = config.evaluation_report_path.read_text(encoding="utf-8")
    assert "stay-secret" not in report_text
    parsed = json.loads(report_text)
    validation_metrics = [
        row
        for row in parsed["row_level_metrics"]
        if row["source"] == "mimiciv" and row["split"] == "validation"
    ]
    assert validation_metrics
    assert all(row["roc_auc"] is not None for row in validation_metrics)


def test_metric_slicing_matches_whole_table_metrics(tmp_path: Path) -> None:
    from pipeline.evaluate_baselines import (
        configure_connection,
        metric_slice_predicate,
        metric_slices,
        row_level_metric_query,
    )
    from pipeline.features import fetch_dict_rows

    config = milestone7_config(tmp_path)
    manifest = build_baseline_evaluation(config)
    assert manifest["status"] == "completed"

    def metric_key(row: dict) -> tuple[str, str, str]:
        return (row["baseline_name"], row["source"], row["split"])

    def rounded(row: dict) -> dict:
        return {
            name: (round(value, 9) if isinstance(value, float) else value)
            for name, value in row.items()
        }

    per_slice = {metric_key(row): rounded(row) for row in manifest["row_level_metrics"]}

    with duckdb.connect(database=":memory:") as connection:
        configure_connection(config, connection)
        whole_table = {
            metric_key(row): rounded(row)
            for row in fetch_dict_rows(connection, row_level_metric_query(config))
        }
        slices = metric_slices(connection, config)

    assert per_slice
    assert set(per_slice) == set(whole_table)
    assert per_slice == whole_table
    assert {metric_key(slice_row) for slice_row in slices} == set(per_slice)

    sample_slice = slices[0]
    predicate = metric_slice_predicate(sample_slice)
    assert f"baseline_name = '{sample_slice['baseline_name']}'" in predicate
    assert predicate in row_level_metric_query(config, slice_predicate=predicate)


def test_learned_baselines_train_score_and_handle_unknown_categories(
    tmp_path: Path,
) -> None:
    config = milestone7_config(
        tmp_path,
        baselines=("linear", "xgboost"),
    )

    manifest = build_baseline_evaluation(config)

    assert manifest["status"] == "completed"
    assert manifest["learned_baselines"]["status"] == "completed"
    assert manifest["learned_baselines"]["training_sample"]["positive_row_count"] == 2
    score_rows = read_parquet_rows(config.score_output_path)
    assert {row["baseline_name"] for row in score_rows} == {"linear", "xgboost"}
    assert all(0.0 <= float(row["score"]) <= 1.0 for row in score_rows)
    assert (config.evaluation_root / "models" / "learned_preprocessor.joblib").exists()
    assert (config.evaluation_root / "models" / "linear_sgd_model.joblib").exists()
    assert (config.evaluation_root / "models" / "xgboost_model.json").exists()

    report_text = config.evaluation_report_path.read_text(encoding="utf-8")
    assert "stay-secret" not in report_text


def test_learned_training_sample_is_hash_sampled_before_feature_join(
    tmp_path: Path,
) -> None:
    config = milestone7_config(tmp_path, baselines=("linear",))
    sample_path = config.evaluation_root / "sample.parquet"

    with duckdb.connect(database=":memory:") as connection:
        feature_spec = resolve_feature_spec(
            connection, config.patient_stay_features_path
        )
        query = training_sample_query(
            patient_condition_medication_path=config.patient_condition_medication_path,
            patient_stay_features_path=config.patient_stay_features_path,
            feature_spec=feature_spec,
            condition_filter_sql="TRUE",
            seed=config.seed,
        )
        assert "ROW_NUMBER" not in query.upper()
        assert "selected_rows AS" in query

        row_count = write_training_sample(
            connection,
            query=query,
            output_path=sample_path,
        )

    sample_rows = read_parquet_rows(sample_path)
    assert row_count == len(sample_rows)
    assert row_count == 6
    assert sum(bool(row["label_prescribed"]) for row in sample_rows) == 2
    assert all("age_years" in row for row in sample_rows)


def test_phase8_p0_columns_are_resolved_without_identifiers(tmp_path: Path) -> None:
    config = milestone7_config(tmp_path, baselines=("linear",))

    with duckdb.connect(database=":memory:") as connection:
        feature_spec = resolve_feature_spec(
            connection,
            config.patient_stay_features_path,
        )

    assert "condition_ccsr_INF002_present_24h" in feature_spec.stay_numeric
    assert "lab_lactate_slope_24h" in feature_spec.stay_numeric
    assert "stay_uid" not in feature_spec.model_columns
    assert "patient_uid" not in feature_spec.model_columns


def test_milestone6_fixture_supports_learned_baseline_evaluation(
    tmp_path: Path,
) -> None:
    fixture = write_milestone6_harmonized_fixture(tmp_path)
    features_root = tmp_path / "Dataset" / "processed" / "features"
    training_root = tmp_path / "Dataset" / "processed" / "training"
    reports_root = tmp_path / "reports"

    feature_result = build_feature_artifacts(
        FeatureBuildConfig(
            harmonized_root=Path(fixture["harmonized_root"]),
            features_root=features_root,
            manifest_path=reports_root / "milestone6_feature_manifest.json",
        )
    )
    assert feature_result["status"] == "completed"

    training_result = build_training_artifacts(
        TrainingTableBuildConfig(
            harmonized_root=Path(fixture["harmonized_root"]),
            features_root=features_root,
            training_root=training_root,
            manifest_path=reports_root / "training_table_manifest.json",
        )
    )
    assert training_result["status"] == "completed"

    manifest = build_baseline_evaluation(
        BaselineEvaluationConfig(
            features_root=features_root,
            training_root=training_root,
            evaluation_root=tmp_path / "Dataset" / "processed" / "evaluation",
            coverage_report_path=reports_root / "milestone7_coverage_report.json",
            evaluation_report_path=reports_root / "milestone7_baseline_evaluation.json",
            training_manifest_path=reports_root / "training_table_manifest.json",
            baselines=("linear",),
            top_k=(1, 3),
        )
    )

    assert manifest["status"] == "completed"
    assert manifest["learned_baselines"]["status"] == "completed"
    score_rows = read_parquet_rows(
        tmp_path / "Dataset" / "processed" / "evaluation" / "baseline_scores.parquet"
    )
    assert {row["baseline_name"] for row in score_rows} == {"linear"}


def test_final_mode_blocks_test_metrics_without_frozen_selection(
    tmp_path: Path,
) -> None:
    config = milestone7_config(tmp_path)
    blocked_config = BaselineEvaluationConfig(
        features_root=config.features_root,
        training_root=config.training_root,
        evaluation_root=config.evaluation_root,
        coverage_report_path=config.coverage_report_path,
        evaluation_report_path=config.evaluation_report_path,
        training_manifest_path=config.training_manifest_path,
        mode="final",
        frozen_selection=False,
    )

    manifest = build_baseline_evaluation(blocked_config)

    assert manifest["status"] == "blocked_final_requires_frozen_selection"
    assert "row_level_metrics" not in manifest
    assert "Final/test evaluation requires" in manifest["blocker"]
