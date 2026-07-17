from __future__ import annotations

import json
from pathlib import Path

from pipeline.build_training_table import (
    TrainingTableBuildConfig,
    build_training_artifacts,
)
from pipeline.features import FeatureBuildConfig, build_feature_artifacts
from pipeline.graph_suitability import GraphSuitabilityConfig, build_graph_suitability
from pipeline.model_ready_package import (
    ModelReadyPackageConfig,
    build_model_ready_package,
    external_validation_readiness,
)
from pipeline.patient_subgraphs import (
    PatientSubgraphBuildConfig,
    build_patient_subgraphs,
)
from pipeline.preprocessing import (
    PreprocessingBuildConfig,
    build_preprocessing_artifacts,
)
from tests.milestone6_helpers import (
    read_parquet_rows,
    write_milestone6_harmonized_fixture,
)


def test_model_ready_package_builds_vocabularies_and_schema_only_dictionary(
    tmp_path: Path,
) -> None:
    fixture = write_milestone6_harmonized_fixture(tmp_path)
    harmonized_root = Path(fixture["harmonized_root"])
    phase8_root = tmp_path / "Dataset" / "processed" / "phase8_p0"
    features_root = phase8_root / "features"
    training_root = phase8_root / "training"
    sensitivity_root = phase8_root / "sensitivity" / "atc3_or_rxnorm" / "training"
    graph_root = phase8_root / "graph" / "milestone8"
    subgraphs_root = graph_root / "patient_subgraphs"
    reports_root = tmp_path / "reports"
    feature_manifest = reports_root / "phase8_features.json"
    training_manifest = reports_root / "phase8_training.json"
    sensitivity_manifest = reports_root / "phase8_atc3_training.json"
    preprocessing_manifest = reports_root / "phase8_preprocessing.json"

    assert (
        build_feature_artifacts(
            FeatureBuildConfig(
                harmonized_root=harmonized_root,
                features_root=features_root,
                manifest_path=feature_manifest,
                feature_set="phase8_p0",
            )
        )["status"]
        == "completed"
    )
    assert (
        build_training_artifacts(
            TrainingTableBuildConfig(
                harmonized_root=harmonized_root,
                features_root=features_root,
                training_root=training_root,
                manifest_path=training_manifest,
            )
        )["status"]
        == "completed"
    )
    assert (
        build_training_artifacts(
            TrainingTableBuildConfig(
                harmonized_root=harmonized_root,
                features_root=features_root,
                training_root=sensitivity_root,
                manifest_path=sensitivity_manifest,
                candidate_token_strategy="atc3_or_rxnorm",
            )
        )["status"]
        == "completed"
    )
    preprocessing_result = build_preprocessing_artifacts(
        PreprocessingBuildConfig(
            features_root=features_root,
            training_root=training_root,
            preprocessing_root=training_root / "preprocessing",
            manifest_path=preprocessing_manifest,
        )
    )
    assert preprocessing_result["status"] == "completed"
    assert preprocessing_result["versions"]["feature_version"] == (
        "temporal-features-v2"
    )
    graph_result = build_graph_suitability(
        GraphSuitabilityConfig(
            features_root=features_root,
            training_root=training_root,
            graph_root=graph_root,
            schema_report_path=reports_root / "graph_schema.json",
            suitability_report_path=reports_root / "graph_suitability.json",
            ablation_plan_path=reports_root / "graph_ablation_plan.json",
        )
    )
    assert graph_result["status"] == "completed"
    assert graph_result["versions"]["feature_version"] == "temporal-features-v2"
    assert (
        build_patient_subgraphs(
            PatientSubgraphBuildConfig(
                features_root=features_root,
                training_root=training_root,
                graph_root=graph_root,
                subgraphs_root=subgraphs_root,
                manifest_path=reports_root / "patient_subgraphs.json",
            )
        )["status"]
        == "completed"
    )

    data_dictionary_path = reports_root / "model_ready_data_dictionary.json"
    package_manifest_path = reports_root / "model_ready_manifest.json"
    config = ModelReadyPackageConfig(
        features_root=features_root,
        training_root=training_root,
        graph_root=graph_root,
        subgraphs_root=subgraphs_root,
        preprocessing_root=training_root / "preprocessing",
        package_root=phase8_root / "model_ready",
        data_dictionary_path=data_dictionary_path,
        manifest_path=package_manifest_path,
        primary_training_manifest_path=training_manifest,
        sensitivity_training_manifest_path=sensitivity_manifest,
        preprocessing_manifest_path=preprocessing_manifest,
        subgraphs_manifest_path=reports_root / "patient_subgraphs.json",
    )

    manifest = build_model_ready_package(config)

    assert manifest["status"] == "completed"
    assert manifest["versions"]["feature_version"] == "temporal-features-v2"
    expected = {
        "cohort_stays",
        "patient_stay_features",
        "patient_condition_medication",
        "event_sequences",
        "split_manifest",
        "candidate_catalog",
        "graph_edges",
        "subgraph_index",
        "subgraph_nodes",
        "subgraph_edges",
        "subgraph_candidates",
        "condition_vocabulary",
        "candidate_medication_vocabulary",
        "event_vocabulary",
        "graph_node_vocabulary",
        "train_fitted_preprocessor",
        "preprocessing_manifest",
        "patient_subgraphs_manifest",
        "data_dictionary",
    }
    assert expected.issubset(manifest["artifacts"])
    assert manifest["external_validation_readiness"]["status"] == (
        "externally_evaluable"
    )
    assert manifest["train_fit_scope"]
    assert all(
        row["fit_source"] == "mimiciv" and row["fit_split"] == "train"
        for row in manifest["train_fit_scope"]
    )

    condition_vocab = read_parquet_rows(config.condition_vocabulary_path)
    assert condition_vocab
    assert {row["fit_split"] for row in condition_vocab} == {"train"}

    dictionary_text = data_dictionary_path.read_text(encoding="utf-8")
    assert str(fixture["train_patient_uid"]) not in dictionary_text
    assert str(fixture["train_stay"]) not in dictionary_text
    dictionary = json.loads(dictionary_text)
    assert dictionary["data_safety"]["contains_row_samples"] is False
    cohort_columns = {
        row["column_name"] for row in dictionary["artifacts"]["cohort_stays"]["columns"]
    }
    assert {"split", "prediction_timestamp", "eligibility_status"}.issubset(
        cohort_columns
    )


def test_external_readiness_blocks_performance_when_both_strategies_have_no_groups(
    tmp_path: Path,
) -> None:
    manifests: list[Path] = []
    for strategy in ("rxnorm_or_atc", "atc3_or_rxnorm"):
        path = tmp_path / f"{strategy}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "parameters": {"candidate_token_strategy": strategy},
                    "external_validation": {
                        "status": "coverage_only_no_in_catalog_positive_groups",
                        "positive_ranking_group_count": 0,
                        "performance_claims_allowed": False,
                    },
                }
            ),
            encoding="utf-8",
        )
        manifests.append(path)

    readiness = external_validation_readiness(*manifests)

    assert readiness["status"] == "coverage_only_no_in_catalog_positive_groups"
    assert readiness["performance_claims_allowed"] is False
    assert readiness["evaluable_candidate_token_strategies"] == []
