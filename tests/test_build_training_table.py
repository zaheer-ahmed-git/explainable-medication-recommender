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
    write_parquet_rows,
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
        "cohort_stays",
        "split_manifest",
        "candidate_catalog",
        "patient_condition_medication",
    } == set(manifest["artifacts"])

    cohort_rows = read_parquet_rows(training_root / "cohort_stays.parquet")
    cohort_by_stay = {str(row["stay_uid"]): row for row in cohort_rows}
    train_cohort = cohort_by_stay[str(fixture["train_stay"])]
    assert train_cohort["split"] == "train"
    assert train_cohort["t0_hours_from_admit"] == 0.0
    assert train_cohort["prediction_time_hours_from_admit"] == 24.0
    assert train_cohort["label_window_end_hours_from_admit"] == 48.0
    assert train_cohort["eligibility_status"] == "eligible_primary"
    assert train_cohort["age_years"] == 66.0

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


def test_phase8_training_infers_v2_feature_provenance(tmp_path: Path) -> None:
    fixture = write_milestone6_harmonized_fixture(tmp_path)
    features_root = tmp_path / "Dataset" / "processed" / "phase8_p0" / "features"
    training_root = tmp_path / "Dataset" / "processed" / "phase8_p0" / "training"

    feature_result = build_feature_artifacts(
        FeatureBuildConfig(
            harmonized_root=Path(fixture["harmonized_root"]),
            features_root=features_root,
            manifest_path=tmp_path / "reports" / "phase8_features.json",
            feature_set="phase8_p0",
        )
    )
    assert feature_result["status"] == "completed"

    manifest = build_training_artifacts(
        TrainingTableBuildConfig(
            harmonized_root=Path(fixture["harmonized_root"]),
            features_root=features_root,
            training_root=training_root,
            manifest_path=tmp_path / "reports" / "phase8_training.json",
        )
    )

    assert manifest["status"] == "completed"
    assert manifest["versions"]["feature_version"] == "temporal-features-v2"
    cohort_rows = read_parquet_rows(training_root / "cohort_stays.parquet")
    assert {row["feature_version"] for row in cohort_rows} == {"temporal-features-v2"}
    split_rows = read_parquet_rows(training_root / "split_manifest.parquet")
    catalog_rows = read_parquet_rows(training_root / "candidate_catalog.parquet")
    assert {row["feature_version"] for row in split_rows} == {"temporal-features-v2"}
    assert {row["feature_version"] for row in catalog_rows} == {"temporal-features-v2"}


def test_training_builder_can_use_atc_class_candidate_tokens(tmp_path: Path) -> None:
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
            candidate_token_strategy="atc3_or_rxnorm",
        )
    )

    assert manifest["status"] == "completed"
    assert manifest["parameters"]["candidate_token_strategy"] == "atc3_or_rxnorm"

    catalog_rows = read_parquet_rows(training_root / "candidate_catalog.parquet")
    catalog_tokens = {str(row["candidate_medication_token"]) for row in catalog_rows}
    assert "atc:J01X" in catalog_tokens
    assert "rxnorm:111" not in catalog_tokens

    table_rows = read_parquet_rows(
        training_root / "patient_condition_medication.parquet"
    )
    by_stay_candidate = {
        (str(row["stay_uid"]), str(row["candidate_medication_token"])): row
        for row in table_rows
    }
    heldout_class_positive = by_stay_candidate[
        (str(fixture["heldout_stay"]), "atc:J01X")
    ]
    assert heldout_class_positive["label_prescribed"] is True
    assert heldout_class_positive["label_event_count"] == 2

    external_class_positive = by_stay_candidate[
        (str(fixture["external_stay"]), "atc:J01X")
    ]
    assert external_class_positive["label_prescribed"] is True
    assert external_class_positive["split"] == "external"


def test_training_builder_fails_on_harmonized_rows_without_decision_join(
    tmp_path: Path,
) -> None:
    fixture = write_milestone6_harmonized_fixture(tmp_path)
    harmonized_root = Path(fixture["harmonized_root"])
    medications_path = harmonized_root / "medications.parquet"
    features_root = tmp_path / "Dataset" / "processed" / "features"
    training_root = tmp_path / "Dataset" / "processed" / "training"
    feature_manifest = tmp_path / "reports" / "milestone6_feature_manifest.json"
    training_manifest = tmp_path / "reports" / "training_table_manifest.json"

    medications = read_parquet_rows(medications_path)
    columns = tuple(medications[0])
    orphan = dict(medications[0])
    orphan["stay_uid"] = "mimiciv:orphan-stay"
    orphan["source_event_id"] = "orphan-medication"
    write_parquet_rows(
        medications_path,
        columns,
        tuple(
            tuple(row[column] for column in columns) for row in (*medications, orphan)
        ),
    )

    feature_result = build_feature_artifacts(
        FeatureBuildConfig(
            harmonized_root=harmonized_root,
            features_root=features_root,
            manifest_path=feature_manifest,
        )
    )
    assert feature_result["status"] == "completed"

    manifest = build_training_artifacts(
        TrainingTableBuildConfig(
            harmonized_root=harmonized_root,
            features_root=features_root,
            training_root=training_root,
            manifest_path=training_manifest,
        )
    )

    medication_integrity = [
        row for row in manifest["join_integrity"] if row["table_name"] == "medications"
    ][0]
    assert manifest["status"] == "failed_join_integrity"
    assert medication_integrity["orphan_row_count"] == 1
    assert manifest["tables"] == []
    assert not (training_root / "candidate_catalog.parquet").exists()
