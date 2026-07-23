from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from pipeline.training_contract import (
    TrainingContractConfig,
    approved_model_projection,
    build_training_contract_lock,
    validate_requested_columns,
)
from tests.milestone6_helpers import write_parquet_rows


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_contract_fixture(
    tmp_path: Path,
    *,
    medication_event: bool = False,
) -> TrainingContractConfig:
    dataset_root = tmp_path / "Dataset"
    artifact_root = dataset_root / "processed" / "phase8_p0"
    reports_root = tmp_path / "reports"
    feature_version = "temporal-features-v2"
    graph_version = "graph-suitability-v1"
    label_version = "observed-medication-label-v1"
    split_version = "patient-split-v1"

    paths = {
        "cohort_stays": artifact_root / "training" / "cohort_stays.parquet",
        "cohort_decision_times": artifact_root
        / "features"
        / "cohort_decision_times.parquet",
        "patient_stay_features": artifact_root
        / "features"
        / "patient_stay_features.parquet",
        "patient_condition_medication": artifact_root
        / "training"
        / "patient_condition_medication.parquet",
        "event_sequences": artifact_root / "features" / "event_sequences.parquet",
        "split_manifest": artifact_root / "training" / "split_manifest.parquet",
        "candidate_catalog": artifact_root / "training" / "candidate_catalog.parquet",
        "graph_edges": artifact_root / "graph" / "milestone8" / "graph_edges.parquet",
        "subgraph_index": artifact_root
        / "graph"
        / "milestone8"
        / "patient_subgraphs"
        / "subgraph_index.parquet",
        "subgraph_nodes": artifact_root
        / "graph"
        / "milestone8"
        / "patient_subgraphs"
        / "subgraph_nodes.parquet",
        "subgraph_edges": artifact_root
        / "graph"
        / "milestone8"
        / "patient_subgraphs"
        / "subgraph_edges.parquet",
        "subgraph_candidates": artifact_root
        / "graph"
        / "milestone8"
        / "patient_subgraphs"
        / "subgraph_candidates.parquet",
        "condition_vocabulary": artifact_root
        / "model_ready"
        / "condition_vocabulary.parquet",
        "candidate_medication_vocabulary": artifact_root
        / "model_ready"
        / "candidate_medication_vocabulary.parquet",
        "event_vocabulary": artifact_root / "model_ready" / "event_vocabulary.parquet",
        "graph_node_vocabulary": artifact_root
        / "model_ready"
        / "graph_node_vocabulary.parquet",
    }

    write_parquet_rows(
        paths["cohort_stays"],
        ("source", "patient_uid", "stay_uid"),
        (("mimiciv", "patient-secret", "stay-secret"),),
    )
    write_parquet_rows(
        paths["cohort_decision_times"],
        ("source", "patient_uid", "stay_uid", "prediction_time_hours_from_admit"),
        (("mimiciv", "patient-secret", "stay-secret", 24.0),),
    )
    write_parquet_rows(
        paths["patient_stay_features"],
        (
            "source",
            "patient_uid",
            "stay_uid",
            "split",
            "prediction_time_hours_from_admit",
            "label_window_end_hours_from_admit",
            "age_years",
            "hospital_id",
            "feature_version",
            "split_version",
        ),
        (
            (
                "mimiciv",
                "patient-secret",
                "stay-secret",
                "train",
                24.0,
                48.0,
                63.0,
                "hospital-secret",
                feature_version,
                split_version,
            ),
        ),
    )
    write_parquet_rows(
        paths["event_sequences"],
        (
            "source",
            "patient_uid",
            "stay_uid",
            "split",
            "event_sequence_position",
            "event_type",
            "event_time_hours_from_admit",
            "event_token",
            "value_numeric",
            "normalized_unit",
            "source_event_id",
            "source_text",
            "value_text",
            "source_code",
            "feature_version",
        ),
        (
            (
                "mimiciv",
                "patient-secret",
                "stay-secret",
                "train",
                1,
                "medication" if medication_event else "lab",
                12.0,
                "lab:lactate",
                2.0,
                "mmol/L",
                "event-secret",
                "restricted text",
                "restricted value",
                "restricted code",
                feature_version,
            ),
        ),
    )
    write_parquet_rows(
        paths["patient_condition_medication"],
        (
            "source",
            "patient_uid",
            "stay_uid",
            "split",
            "ranking_group_id",
            "index_condition_token",
            "candidate_medication_token",
            "candidate_rank",
            "label_prescribed",
            "label_first_observed_hours_from_admit",
            "prediction_time_hours_from_admit",
            "label_window_end_hours_from_admit",
            "feature_version",
            "label_version",
            "split_version",
        ),
        (
            (
                "mimiciv",
                "patient-secret",
                "stay-secret",
                "train",
                "group-secret",
                "condition:a",
                "rxnorm:a",
                1,
                True,
                30.0,
                24.0,
                48.0,
                feature_version,
                label_version,
                split_version,
            ),
        ),
    )
    write_parquet_rows(
        paths["split_manifest"],
        ("source", "patient_uid", "split", "split_version"),
        (("mimiciv", "patient-secret", "train", split_version),),
    )
    write_parquet_rows(
        paths["candidate_catalog"],
        (
            "index_condition_token",
            "candidate_medication_token",
            "candidate_rank",
            "feature_version",
            "label_version",
            "split_version",
        ),
        (
            (
                "condition:a",
                "rxnorm:a",
                1,
                feature_version,
                label_version,
                split_version,
            ),
        ),
    )
    write_parquet_rows(
        paths["graph_edges"],
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
        ),
        (
            (
                "condition|condition:a",
                "medication|rxnorm:a",
                "condition",
                "medication",
                "condition_medication_train_positive",
                2,
                "mimiciv",
                "train",
                graph_version,
                feature_version,
                label_version,
                split_version,
            ),
        ),
    )
    write_parquet_rows(
        paths["subgraph_index"],
        (
            "source",
            "split",
            "stay_uid",
            "subgraph_id",
            "index_condition_token",
            "node_count",
            "edge_count",
            "candidate_count",
            "positive_count",
            "feature_version",
            "graph_version",
            "label_version",
            "split_version",
        ),
        (
            (
                "mimiciv",
                "train",
                "stay-secret",
                "subgraph-secret",
                "condition:a",
                2,
                1,
                1,
                1,
                feature_version,
                graph_version,
                label_version,
                split_version,
            ),
        ),
    )
    write_parquet_rows(
        paths["subgraph_nodes"],
        (
            "source",
            "split",
            "stay_uid",
            "subgraph_id",
            "node_index",
            "node_id",
            "node_type",
            "node_role",
            "observed_predecision",
            "in_train_graph",
            "cold_start",
            "feature_version",
            "graph_version",
            "split_version",
        ),
        (
            (
                "mimiciv",
                "train",
                "stay-secret",
                "subgraph-secret",
                0,
                "condition|condition:a",
                "condition",
                "index",
                True,
                True,
                False,
                feature_version,
                graph_version,
                split_version,
            ),
        ),
    )
    write_parquet_rows(
        paths["subgraph_edges"],
        (
            "source",
            "split",
            "stay_uid",
            "subgraph_id",
            "src_node_index",
            "dst_node_index",
            "src_id",
            "dst_id",
            "src_type",
            "dst_type",
            "relation_type",
            "support_count",
            "fit_source",
            "fit_split",
            "feature_version",
            "graph_version",
        ),
        (
            (
                "mimiciv",
                "train",
                "stay-secret",
                "subgraph-secret",
                0,
                1,
                "condition|condition:a",
                "medication|rxnorm:a",
                "condition",
                "medication",
                "condition_medication_train_positive",
                2,
                "mimiciv",
                "train",
                feature_version,
                graph_version,
            ),
        ),
    )
    write_parquet_rows(
        paths["subgraph_candidates"],
        (
            "source",
            "split",
            "stay_uid",
            "subgraph_id",
            "index_condition_token",
            "candidate_medication_token",
            "candidate_node_index",
            "candidate_rank",
            "label_prescribed",
            "in_train_graph",
            "cold_start",
            "feature_version",
            "graph_version",
            "label_version",
            "split_version",
        ),
        (
            (
                "mimiciv",
                "train",
                "stay-secret",
                "subgraph-secret",
                "condition:a",
                "rxnorm:a",
                1,
                1,
                True,
                True,
                False,
                feature_version,
                graph_version,
                label_version,
                split_version,
            ),
        ),
    )

    vocabulary_rows = {
        "condition_vocabulary": ("condition_token", "condition:a"),
        "candidate_medication_vocabulary": ("candidate_medication_token", "rxnorm:a"),
        "event_vocabulary": ("event_token", "lab:lactate"),
        "graph_node_vocabulary": ("node_id", "condition|condition:a"),
    }
    for artifact_name, (token_column, token_value) in vocabulary_rows.items():
        write_parquet_rows(
            paths[artifact_name],
            ("token_index", token_column, "fit_source", "fit_split"),
            ((0, token_value, "mimiciv", "train"),),
        )

    preprocessing_manifest = reports_root / "preprocessing.json"
    subgraphs_manifest = reports_root / "subgraphs.json"
    data_dictionary = reports_root / "dictionary.json"
    preprocessor = artifact_root / "training" / "preprocessing" / "preprocessor.joblib"
    preprocessor.parent.mkdir(parents=True, exist_ok=True)
    preprocessor.write_bytes(b"synthetic-preprocessor")
    _write_json(
        preprocessing_manifest,
        {"status": "completed", "fit_scope": {"source": "mimiciv", "split": "train"}},
    )
    _write_json(
        subgraphs_manifest,
        {
            "status": "completed",
            "graph_fit_scope": [
                {"fit_source": "mimiciv", "fit_split": "train", "edge_count": 1}
            ],
        },
    )
    _write_json(data_dictionary, {"status": "completed", "artifacts": {}})

    package_artifacts = {name: str(path) for name, path in paths.items()}
    package_artifacts.update(
        {
            "train_fitted_preprocessor": str(preprocessor),
            "preprocessing_manifest": str(preprocessing_manifest),
            "patient_subgraphs_manifest": str(subgraphs_manifest),
            "data_dictionary": str(data_dictionary),
        }
    )
    package_manifest = reports_root / "package.json"
    _write_json(
        package_manifest,
        {
            "status": "completed",
            "versions": {
                "feature_version": feature_version,
                "graph_version": graph_version,
                "label_version": label_version,
                "split_version": split_version,
            },
            "artifacts": package_artifacts,
            "tables": [
                {"table_name": name, "row_count": 1}
                for name, path in paths.items()
                if path.suffix == ".parquet"
            ],
            "train_fit_scope": [
                {"fit_source": "mimiciv", "fit_split": "train", "edge_count": 1}
            ],
        },
    )
    primary_manifest = reports_root / "primary.json"
    sensitivity_manifest = reports_root / "sensitivity.json"
    feature_manifest = reports_root / "features.json"
    graph_report = reports_root / "graph.json"
    _write_json(
        primary_manifest,
        {
            "status": "completed",
            "parameters": {
                "development_source": "mimiciv",
                "candidate_token_strategy": "rxnorm_or_atc",
                "prediction_offset_hours": 24,
                "label_window_hours": 24,
            },
            "split_integrity": {"patients_with_multiple_splits": 0},
            "external_validation": {
                "status": "coverage_only_no_in_catalog_positive_groups",
                "positive_ranking_group_count": 0,
                "performance_claims_allowed": False,
            },
        },
    )
    _write_json(
        sensitivity_manifest,
        {
            "status": "completed",
            "parameters": {"candidate_token_strategy": "atc3_or_rxnorm"},
            "external_validation": {
                "status": "externally_evaluable",
                "positive_ranking_group_count": 1,
                "performance_claims_allowed": True,
            },
        },
    )
    _write_json(
        feature_manifest,
        {
            "status": "completed",
            "parameters": {
                "include_predecision_medications": False,
                "prediction_offset_hours": 24,
            },
        },
    )
    _write_json(
        graph_report,
        {
            "status": "completed",
            "leakage_audit": {"status": "pass"},
            "gate_review": {"result": "pass_for_graph_ablation"},
        },
    )
    return TrainingContractConfig(
        package_manifest_path=package_manifest,
        data_dictionary_path=data_dictionary,
        primary_training_manifest_path=primary_manifest,
        sensitivity_training_manifest_path=sensitivity_manifest,
        feature_manifest_path=feature_manifest,
        preprocessing_manifest_path=preprocessing_manifest,
        subgraphs_manifest_path=subgraphs_manifest,
        graph_suitability_report_path=graph_report,
        output_path=reports_root / "contract_lock.json",
        dataset_root=dataset_root,
        require_protected_storage=False,
        duckdb_temp_directory=tmp_path / "duckdb",
        duckdb_memory_limit="1GB",
        duckdb_threads=1,
    )


def test_training_contract_writes_safe_completed_lock(tmp_path: Path) -> None:
    config = _write_contract_fixture(tmp_path)

    report = build_training_contract_lock(config)

    assert report["status"] == "completed"
    assert len(report["contract_digest"]) == 64
    assert (
        report["contract"]["split_integrity"][0]["patients_with_multiple_splits"] == 0
    )
    assert (
        report["contract"]["temporal_integrity"]["event_sequences"][
            "medication_row_count"
        ]
        == 0
    )
    serialized = config.output_path.read_text(encoding="utf-8")
    assert "patient-secret" not in serialized
    assert "restricted text" not in serialized


def test_training_contract_rejects_medication_event_leakage(tmp_path: Path) -> None:
    config = _write_contract_fixture(tmp_path, medication_event=True)

    report = build_training_contract_lock(config)

    assert report["status"] == "failed_contract_audit"
    assert "event_temporal_leakage" in {row["code"] for row in report["errors"]}


def test_training_contract_rejects_changed_locked_artifact(tmp_path: Path) -> None:
    config = _write_contract_fixture(tmp_path)
    initial = build_training_contract_lock(config)
    expected_lock = tmp_path / "expected_contract_lock.json"
    expected_lock.write_text(
        config.output_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    preprocessor_path = Path(
        initial["contract"]["artifacts"]["train_fitted_preprocessor"]["path"]
    )
    preprocessor_path.write_bytes(b"changed-synthetic-preprocessor")

    report = build_training_contract_lock(
        replace(
            config,
            output_path=tmp_path / "latest_audit.json",
            expected_lock_path=expected_lock,
        )
    )

    assert report["status"] == "failed_contract_audit"
    assert "contract_digest_mismatch" in {row["code"] for row in report["errors"]}


def test_approved_projection_rejects_identifiers_and_raw_text() -> None:
    columns = (
        ("event_sequence_position", "BIGINT"),
        ("event_type", "VARCHAR"),
        ("event_time_hours_from_admit", "DOUBLE"),
        ("event_token", "VARCHAR"),
        ("value_numeric", "DOUBLE"),
        ("normalized_unit", "VARCHAR"),
        ("patient_uid", "VARCHAR"),
        ("source_text", "VARCHAR"),
    )

    projection = approved_model_projection("event_sequences", columns)

    assert "patient_uid" not in projection
    assert "source_text" not in projection
    try:
        validate_requested_columns("event_sequences", ("source_text",), columns)
    except ValueError as error:
        assert "unsafe or unavailable" in str(error)
    else:
        raise AssertionError("unsafe model projection was accepted")


def test_stay_projection_rejects_outcomes_and_raw_codes() -> None:
    columns = (
        ("age_years", "DOUBLE"),
        ("condition_count_24h", "BIGINT"),
        ("hospital_mortality", "BOOLEAN"),
        ("diagnosis_code", "BIGINT"),
        ("ward_id", "BIGINT"),
    )

    projection = approved_model_projection("patient_stay_features", columns)

    assert projection == ("age_years", "condition_count_24h")
