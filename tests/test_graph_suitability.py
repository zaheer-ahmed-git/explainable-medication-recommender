from __future__ import annotations

import json
from pathlib import Path

from pipeline.graph_suitability import (
    GraphSuitabilityConfig,
    build_graph_suitability,
)
from tests.milestone6_helpers import read_parquet_rows, write_parquet_rows


def _write_minimal_graph_inputs(
    tmp_path: Path, *, train_edges: bool = True
) -> dict[str, Path]:
    features_root = tmp_path / "Dataset" / "processed" / "features"
    training_root = tmp_path / "Dataset" / "processed" / "training"
    graph_root = tmp_path / "Dataset" / "processed" / "graph" / "milestone8"
    reports_root = tmp_path / "reports"

    event_columns = (
        "source",
        "split",
        "stay_uid",
        "event_type",
        "event_token",
        "event_time_hours_from_admit",
    )
    if train_edges:
        event_rows = (
            ("mimiciv", "train", "stay-train-1", "lab", "lactate", 5.0),
            ("mimiciv", "train", "stay-train-1", "lab", "future_marker", 25.0),
            ("mimiciv", "train", "stay-train-1", "vital", "heart_rate", 6.0),
            (
                "mimiciv",
                "train",
                "stay-train-1",
                "intervention",
                "ventilation",
                7.0,
            ),
            ("mimiciv", "validation", "stay-val-1", "lab", "validation_only", 5.0),
            ("eicu_crd", "external", "stay-ext-1", "lab", "external_only", 5.0),
        )
    else:
        event_rows = (
            ("mimiciv", "validation", "stay-val-1", "lab", "validation_only", 5.0),
        )
    write_parquet_rows(
        features_root / "event_sequences.parquet", event_columns, event_rows
    )

    pcm_columns = (
        "source",
        "split",
        "stay_uid",
        "ranking_group_id",
        "index_condition_token",
        "candidate_medication_token",
        "label_prescribed",
    )
    if train_edges:
        pcm_rows = (
            (
                "mimiciv",
                "train",
                "stay-train-1",
                "rg-train-1",
                "condition:a",
                "rxnorm:1",
                True,
            ),
            (
                "mimiciv",
                "train",
                "stay-train-1",
                "rg-train-1",
                "condition:a",
                "rxnorm:2",
                True,
            ),
            (
                "mimiciv",
                "train",
                "stay-train-1",
                "rg-train-1",
                "condition:a",
                "rxnorm:3",
                False,
            ),
            (
                "mimiciv",
                "train",
                "stay-train-2",
                "rg-train-2",
                "condition:a",
                "rxnorm:1",
                True,
            ),
            (
                "mimiciv",
                "validation",
                "stay-val-1",
                "rg-val-1",
                "condition:a",
                "rxnorm:1",
                True,
            ),
            (
                "mimiciv",
                "validation",
                "stay-val-1",
                "rg-val-1",
                "condition:a",
                "rxnorm:3",
                True,
            ),
            (
                "eicu_crd",
                "external",
                "stay-ext-1",
                "rg-ext-1",
                "condition:b",
                "rxnorm:1",
                True,
            ),
        )
    else:
        pcm_rows = (
            (
                "mimiciv",
                "validation",
                "stay-val-1",
                "rg-val-1",
                "condition:a",
                "rxnorm:1",
                True,
            ),
        )
    write_parquet_rows(
        training_root / "patient_condition_medication.parquet",
        pcm_columns,
        pcm_rows,
    )

    catalog_columns = (
        "index_condition_token",
        "candidate_medication_token",
        "candidate_rank",
    )
    write_parquet_rows(
        training_root / "candidate_catalog.parquet",
        catalog_columns,
        (
            ("condition:a", "rxnorm:1", 1),
            ("condition:a", "rxnorm:2", 2),
            ("condition:a", "rxnorm:3", 3),
        ),
    )
    return {
        "features_root": features_root,
        "training_root": training_root,
        "graph_root": graph_root,
        "reports_root": reports_root,
    }


def _config(paths: dict[str, Path]) -> GraphSuitabilityConfig:
    reports_root = paths["reports_root"]
    return GraphSuitabilityConfig(
        features_root=paths["features_root"],
        training_root=paths["training_root"],
        graph_root=paths["graph_root"],
        schema_report_path=reports_root / "milestone8_graph_schema.json",
        suitability_report_path=reports_root / "milestone8_graph_suitability.json",
        ablation_plan_path=reports_root / "milestone8_ablation_plan.json",
    )


def test_graph_suitability_builds_train_only_edges_and_safe_reports(
    tmp_path: Path,
) -> None:
    paths = _write_minimal_graph_inputs(tmp_path)
    config = _config(paths)

    report = build_graph_suitability(config)

    assert report["status"] == "completed"
    assert report["gate_review"]["result"] == "pass_for_graph_ablation"
    assert report["leakage_audit"]["status"] == "pass"
    assert report["leakage_audit"]["fit_source_split_counts"] == [
        {"edge_count": 6, "fit_source": "mimiciv", "fit_split": "train"}
    ]

    edges = read_parquet_rows(config.graph_edges_path)
    edge_keys = {
        (row["src_id"], row["dst_id"], row["relation_type"]): row for row in edges
    }
    assert (
        "condition|condition:a",
        "medication|rxnorm:1",
        "condition_medication_train_positive",
    ) in edge_keys
    assert (
        edge_keys[
            (
                "condition|condition:a",
                "medication|rxnorm:1",
                "condition_medication_train_positive",
            )
        ]["support_count"]
        == 2
    )
    assert (
        "medication|rxnorm:1",
        "medication|rxnorm:2",
        "medication_medication_train_coprescribed",
    ) in edge_keys
    assert (
        "condition|condition:a",
        "lab|lactate",
        "condition_lab_predecision",
    ) in edge_keys
    assert all("future_marker" not in row["dst_id"] for row in edges)
    assert all("validation_only" not in row["dst_id"] for row in edges)
    assert all("external_only" not in row["dst_id"] for row in edges)
    assert {row["fit_split"] for row in edges} == {"train"}

    coverage = {
        (row["source"], row["split"]): row
        for row in report["split_coverage_and_cold_start"]
    }
    assert coverage[("mimiciv", "validation")]["unseen_candidate_medication_count"] == 1
    assert coverage[("eicu_crd", "external")]["unseen_condition_count"] == 1
    assert coverage[("eicu_crd", "external")]["positive_graph_coverage_rate"] == 0.0

    for report_name in (
        "milestone8_graph_schema.json",
        "milestone8_graph_suitability.json",
        "milestone8_ablation_plan.json",
    ):
        text = (paths["reports_root"] / report_name).read_text(encoding="utf-8")
        assert "stay-train-1" not in text
        assert "validation_only" not in text
        assert "future_marker" not in text
        parsed = json.loads(text)
        assert parsed["data_safety"]["report_contains_patient_rows"] is False


def test_graph_suitability_handles_sparse_empty_graph(tmp_path: Path) -> None:
    paths = _write_minimal_graph_inputs(tmp_path, train_edges=False)
    config = _config(paths)

    report = build_graph_suitability(config)

    assert report["status"] == "completed"
    assert report["tables"] == [
        {"row_count": 0, "status": "completed", "table_name": "graph_edges"}
    ]
    assert report["gate_review"]["result"] == "blocked_graph_not_ready"
    assert report["connected_components"] == {
        "component_count": 0,
        "connected_node_count": 0,
        "largest_component_node_count": 0,
        "singleton_component_count": 0,
    }


def test_graph_suitability_writes_failed_reports_for_missing_inputs(
    tmp_path: Path,
) -> None:
    paths = {
        "features_root": tmp_path / "missing" / "features",
        "training_root": tmp_path / "missing" / "training",
        "graph_root": tmp_path / "Dataset" / "processed" / "graph" / "milestone8",
        "reports_root": tmp_path / "reports",
    }
    config = _config(paths)

    report = build_graph_suitability(config)

    assert report["status"] == "failed_missing_inputs"
    assert {row["table_name"] for row in report["missing_inputs"]} == {
        "event_sequences",
        "patient_condition_medication",
        "candidate_catalog",
    }
    schema = json.loads(config.schema_report_path.read_text(encoding="utf-8"))
    assert schema["status"] == "failed_missing_inputs"
    ablation = json.loads(config.ablation_plan_path.read_text(encoding="utf-8"))
    assert ablation["status"] == "blocked_missing_inputs"
