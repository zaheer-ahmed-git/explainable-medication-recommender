from __future__ import annotations

import json
from pathlib import Path

from pipeline.build_training_table import (
    TrainingTableBuildConfig,
    build_training_artifacts,
)
from pipeline.features import FeatureBuildConfig, build_feature_artifacts
from pipeline.preprocessing import (
    PreprocessingBuildConfig,
    build_preprocessing_artifacts,
)
from tests.milestone6_helpers import (
    read_parquet_rows,
    write_milestone6_harmonized_fixture,
)


def test_preprocessing_artifacts_fit_on_mimic_train_only(tmp_path: Path) -> None:
    fixture = write_milestone6_harmonized_fixture(tmp_path)
    features_root = tmp_path / "Dataset" / "processed" / "features"
    training_root = tmp_path / "Dataset" / "processed" / "training"
    preprocessing_root = training_root / "preprocessing"
    feature_manifest = tmp_path / "reports" / "milestone6_feature_manifest.json"
    training_manifest = tmp_path / "reports" / "training_table_manifest.json"
    preprocessing_manifest = tmp_path / "reports" / "preprocessing_manifest.json"

    feature_result = build_feature_artifacts(
        FeatureBuildConfig(
            harmonized_root=Path(fixture["harmonized_root"]),
            features_root=features_root,
            manifest_path=feature_manifest,
        )
    )
    assert feature_result["status"] == "completed"
    training_result = build_training_artifacts(
        TrainingTableBuildConfig(
            harmonized_root=Path(fixture["harmonized_root"]),
            features_root=features_root,
            training_root=training_root,
            manifest_path=training_manifest,
        )
    )
    assert training_result["status"] == "completed"

    manifest = build_preprocessing_artifacts(
        PreprocessingBuildConfig(
            features_root=features_root,
            training_root=training_root,
            preprocessing_root=preprocessing_root,
            manifest_path=preprocessing_manifest,
        )
    )

    sample_rows = read_parquet_rows(
        preprocessing_root / "train_preprocessing_sample.parquet"
    )
    manifest_text = preprocessing_manifest.read_text(encoding="utf-8")
    parsed_manifest = json.loads(manifest_text)

    assert manifest["status"] == "completed"
    assert (preprocessing_root / "train_fitted_preprocessor.joblib").exists()
    assert sample_rows
    assert {row["source"] for row in sample_rows} == {"mimiciv"}
    assert {row["split"] for row in sample_rows} == {"train"}
    assert parsed_manifest["fit_scope"]["source"] == "mimiciv"
    assert parsed_manifest["fit_scope"]["split"] == "train"
    assert parsed_manifest["categorical_vocabulary_summary"]
    assert str(fixture["train_patient_uid"]) not in manifest_text
    assert str(fixture["train_stay"]) not in manifest_text
    assert parsed_manifest["data_safety"]["manifest_contains_patient_rows"] is False
