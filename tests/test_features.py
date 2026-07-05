from __future__ import annotations

import json
from pathlib import Path

from pipeline.features import FeatureBuildConfig, build_feature_artifacts
from tests.milestone6_helpers import (
    read_parquet_rows,
    write_milestone6_harmonized_fixture,
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
    assert "vital-map" in event_ids
    assert "lab-future" not in event_ids
    assert "train-med-pre" not in event_ids
    assert "condition-1" not in event_ids
    assert all(row["event_time_hours_from_admit"] <= 24.0 for row in event_rows)

    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert str(fixture["train_patient_uid"]) not in manifest_text
    assert str(fixture["train_stay"]) not in manifest_text
    parsed_manifest = json.loads(manifest_text)
    assert parsed_manifest["data_safety"]["manifest_contains_patient_rows"] is False
    assert parsed_manifest["temporal_event_exclusions"]
