from __future__ import annotations

import json
from pathlib import Path

from pipeline.features import FeatureBuildConfig, build_feature_artifacts
from tests.milestone6_helpers import (
    read_parquet_rows,
    write_milestone6_harmonized_fixture,
    write_parquet_rows,
)


def test_feature_builder_applies_temporal_boundaries_and_safe_manifest(
    tmp_path: Path,
) -> None:
    fixture = write_milestone6_harmonized_fixture(tmp_path)
    features_root = tmp_path / "Dataset" / "processed" / "features"
    manifest_path = tmp_path / "reports" / "milestone6_feature_manifest.json"

    manifest = build_feature_artifacts(
        FeatureBuildConfig(
            harmonized_root=Path(fixture["harmonized_root"]),
            features_root=features_root,
            manifest_path=manifest_path,
            stay_feature_batches=2,
            event_sequence_batches=2,
        )
    )

    assert manifest["status"] == "completed"
    assert {
        "cohort_decision_times",
        "patient_stay_features",
        "event_sequences",
    } == set(manifest["artifacts"])

    decision_rows = read_parquet_rows(features_root / "cohort_decision_times.parquet")
    status_by_stay = {
        str(row["stay_uid"]): row["eligibility_status"] for row in decision_rows
    }
    assert status_by_stay[str(fixture["train_stay"])] == "eligible_primary"
    assert (
        status_by_stay[str(fixture["censored_stay"])] == "censored_before_label_window"
    )
    assert status_by_stay[str(fixture["external_stay"])] == "eligible_primary"

    feature_rows = read_parquet_rows(features_root / "patient_stay_features.parquet")
    feature_by_stay = {str(row["stay_uid"]): row for row in feature_rows}
    train_features = feature_by_stay[str(fixture["train_stay"])]
    assert train_features["lab_event_count_24h"] == 2
    assert train_features["lab_lactate_count_24h"] == 2
    assert train_features["lab_lactate_observed_24h"] == 1
    assert train_features["lab_lactate_max_24h"] == 2.0
    assert train_features["vital_heart_rate_observed_24h"] == 1
    assert train_features["predecision_intervention_count_24h"] == 1

    event_rows = read_parquet_rows(features_root / "event_sequences.parquet")
    event_ids = {str(row["source_event_id"]) for row in event_rows}
    assert "lab-boundary" in event_ids
    assert "lab-early" in event_ids
    assert "vital-map" in event_ids
    assert "lab-future" not in event_ids
    assert "train-med-pre" not in event_ids
    assert "condition-1" not in event_ids
    assert all(row["event_time_hours_from_admit"] <= 24.0 for row in event_rows)
    train_events = sorted(
        (
            row
            for row in event_rows
            if str(row["stay_uid"]) == str(fixture["train_stay"])
        ),
        key=lambda row: row["event_sequence_position"],
    )
    assert [row["source_event_id"] for row in train_events] == [
        "lab-early",
        "lab-boundary",
    ]
    assert [row["event_sequence_position"] for row in train_events] == [1, 2]

    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert str(fixture["train_patient_uid"]) not in manifest_text
    assert str(fixture["train_stay"]) not in manifest_text
    parsed_manifest = json.loads(manifest_text)
    assert parsed_manifest["data_safety"]["manifest_contains_patient_rows"] is False
    assert parsed_manifest["parameters"]["stay_feature_batches"] == 2
    assert parsed_manifest["parameters"]["event_sequence_batches"] == 2
    stay_feature_record = next(
        table
        for table in parsed_manifest["tables"]
        if table["table_name"] == "patient_stay_features"
    )
    assert stay_feature_record["build_strategy"] == "stay_hash_batches"
    assert stay_feature_record["batch_count"] == 2
    assert (
        sum(stay_feature_record["batch_row_counts"]) == stay_feature_record["row_count"]
    )
    event_record = next(
        table
        for table in parsed_manifest["tables"]
        if table["table_name"] == "event_sequences"
    )
    assert event_record["build_strategy"] == "staged_hash_batches"
    assert event_record["batch_count"] == 2
    assert event_record["staged_row_count"] == event_record["row_count"]
    assert sum(event_record["batch_row_counts"]) == event_record["row_count"]
    assert parsed_manifest["temporal_event_exclusions"]


def test_phase8_p0_adds_train_only_conditions_trends_and_safe_manifest(
    tmp_path: Path,
) -> None:
    fixture = write_milestone6_harmonized_fixture(tmp_path)
    harmonized_root = Path(fixture["harmonized_root"])
    features_root = tmp_path / "Dataset" / "processed" / "phase8_p0" / "features"
    manifest_path = tmp_path / "reports" / "phase8_p0_milestone6_feature_manifest.json"
    condition_columns = (
        "source",
        "source_version",
        "patient_uid",
        "encounter_uid",
        "stay_uid",
        "project_condition_token",
        "project_condition_group",
        "normalized_condition_token",
        "normalized_condition_name",
        "condition_text",
        "condition_rollup_level",
        "mapping_status",
    )
    write_parquet_rows(
        harmonized_root / "conditions.parquet",
        condition_columns,
        (
            (
                "mimiciv",
                "3.1",
                fixture["train_patient_uid"],
                "mimiciv:enc-train",
                fixture["train_stay"],
                "project:sepsis",
                "sepsis",
                "ccsr:INF002",
                "Sepsis",
                None,
                "project",
                "mapped_ccsr",
            ),
            (
                "mimiciv",
                "3.1",
                fixture["heldout_patient_uid"],
                "mimiciv:enc-heldout",
                fixture["heldout_stay"],
                "project:heldout_only",
                "heldout",
                "ccsr:HELDOUT",
                "Heldout only",
                None,
                "project",
                "mapped_ccsr",
            ),
            (
                "eicu_crd",
                "2.0",
                fixture["external_patient_uid"],
                "eicu_crd:enc-external",
                fixture["external_stay"],
                "project:external_only",
                "external",
                None,
                None,
                "synthetic external condition",
                "project",
                "mapped_text_to_condition",
            ),
        ),
    )

    manifest = build_feature_artifacts(
        FeatureBuildConfig(
            harmonized_root=harmonized_root,
            features_root=features_root,
            manifest_path=manifest_path,
            feature_set="phase8_p0",
            condition_feature_top_n=1,
            trend_min_events=2,
            stay_feature_batches=2,
            event_sequence_batches=2,
        )
    )

    assert manifest["status"] == "completed"
    assert manifest["feature_set"] == "phase8_p0"
    assert manifest["feature_version"] == "temporal-features-v2"
    assert manifest["condition_vocabulary_size"] == 1
    assert manifest["condition_columns_added"] == ["condition_ccsr_INF002_present_24h"]
    assert "lab_lactate_slope_24h" in manifest["trend_columns_added"]
    assert "vital_heart_rate_missing_24h" in manifest["missingness_columns_added"]
    assert manifest["feature_column_counts_by_family"]["condition_columns"] == 1

    feature_rows = read_parquet_rows(features_root / "patient_stay_features.parquet")
    feature_by_stay = {str(row["stay_uid"]): row for row in feature_rows}
    train_features = feature_by_stay[str(fixture["train_stay"])]
    heldout_features = feature_by_stay[str(fixture["heldout_stay"])]
    external_features = feature_by_stay[str(fixture["external_stay"])]

    assert train_features["condition_ccsr_INF002_present_24h"] == 1
    assert heldout_features["condition_ccsr_INF002_present_24h"] == 0
    assert external_features["condition_ccsr_INF002_present_24h"] == 0
    assert "condition_ccsr_HELDOUT_present_24h" not in train_features
    assert "condition_project_external_only_present_24h" not in train_features

    assert train_features["lab_lactate_first_24h"] == 1.5
    assert train_features["lab_lactate_last_24h"] == 2.0
    assert train_features["lab_lactate_delta_24h"] == 0.5
    assert round(float(train_features["lab_lactate_slope_24h"]), 6) == round(
        0.5 / 19.0,
        6,
    )
    assert train_features["lab_lactate_hours_since_last_24h"] == 0.0
    assert train_features["lab_lactate_missing_24h"] == 0
    assert train_features["lab_glucose_missing_24h"] == 1
    assert train_features["vital_heart_rate_slope_24h"] is None
    assert heldout_features["lab_lactate_missing_24h"] == 1
    assert external_features["lab_creatinine_missing_24h"] == 0

    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert "ccsr:HELDOUT" not in manifest_text
    assert "project:external_only" not in manifest_text
    parsed_manifest = json.loads(manifest_text)
    assert parsed_manifest["data_safety"]["manifest_contains_patient_rows"] is False
    assert parsed_manifest["condition_token_precedence"] == [
        "normalized_condition_token",
        "project_condition_token",
    ]
    assert any(
        row["oov_condition_token_row_count"] > 0
        for row in parsed_manifest["condition_oov_counts"]
    )
