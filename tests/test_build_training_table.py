from __future__ import annotations

import json
from pathlib import Path

from pipeline.build_training_table import (
    TrainingTableBuildConfig,
    build_training_artifacts,
)
from pipeline.features import FeatureBuildConfig, build_feature_artifacts
from tests.milestone6_helpers import (
    read_parquet_rows,
    write_milestone6_harmonized_fixture,
)


def test_training_builder_uses_train_only_candidates_and_label_window(
    tmp_path: Path,
) -> None:
    fixture = write_milestone6_harmonized_fixture(tmp_path)
    features_root = tmp_path / "Dataset" / "processed" / "features"
    training_root = tmp_path / "Dataset" / "processed" / "training"
    feature_manifest = tmp_path / "reports" / "milestone6_feature_manifest.json"
    training_manifest = tmp_path / "reports" / "training_table_manifest.json"

    feature_result = build_feature_artifacts(
        FeatureBuildConfig(
            harmonized_root=Path(fixture["harmonized_root"]),
            features_root=features_root,
            manifest_path=feature_manifest,
        )
    )
    assert feature_result["status"] == "completed"

    manifest = build_training_artifacts(
        TrainingTableBuildConfig(
            harmonized_root=Path(fixture["harmonized_root"]),
            features_root=features_root,
            training_root=training_root,
            manifest_path=training_manifest,
        )
    )

    assert manifest["status"] == "completed"
    assert {
        "split_manifest",
        "candidate_catalog",
        "patient_condition_medication",
    } == set(manifest["artifacts"])

    split_rows = read_parquet_rows(training_root / "split_manifest.parquet")
    patient_splits: dict[str, set[str]] = {}
    for row in split_rows:
        patient_splits.setdefault(str(row["patient_uid"]), set()).add(str(row["split"]))
    assert all(len(splits) == 1 for splits in patient_splits.values())
    assert patient_splits[str(fixture["external_patient_uid"])] == {"external"}

    catalog_rows = read_parquet_rows(training_root / "candidate_catalog.parquet")
    catalog_tokens = {str(row["candidate_medication_token"]) for row in catalog_rows}
    assert "rxnorm:111" in catalog_tokens
    assert "rxnorm:333" not in catalog_tokens

    table_rows = read_parquet_rows(
        training_root / "patient_condition_medication.parquet"
    )
    assert {str(row["stay_uid"]) for row in table_rows}.isdisjoint(
        {str(fixture["censored_stay"])}
    )
    by_stay_candidate = {
        (str(row["stay_uid"]), str(row["candidate_medication_token"])): row
        for row in table_rows
    }
    train_positive = by_stay_candidate[(str(fixture["train_stay"]), "rxnorm:111")]
    assert train_positive["label_prescribed"] is True
    assert train_positive["label_first_observed_hours_from_admit"] == 25.0
    assert train_positive["label_event_count"] == 2

    heldout_positive = by_stay_candidate[(str(fixture["heldout_stay"]), "rxnorm:111")]
    assert heldout_positive["label_prescribed"] is True
    assert heldout_positive["label_first_observed_hours_from_admit"] == 48.0

    external_positive = by_stay_candidate[(str(fixture["external_stay"]), "rxnorm:111")]
    assert external_positive["label_prescribed"] is True
    assert external_positive["split"] == "external"

    manifest_text = training_manifest.read_text(encoding="utf-8")
    assert str(fixture["train_patient_uid"]) not in manifest_text
    assert str(fixture["train_stay"]) not in manifest_text
    parsed_manifest = json.loads(manifest_text)
    assert parsed_manifest["split_integrity"]["patients_with_multiple_splits"] == 0
    assert parsed_manifest["out_of_catalog_positives"]
    label_loss_rows = parsed_manifest["medication_label_loss"]
    assert any(
        row["missing_medication_start_time_events"] > 0 for row in label_loss_rows
    )
    assert parsed_manifest["data_safety"]["manifest_contains_patient_rows"] is False
