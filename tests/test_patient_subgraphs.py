from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

import pipeline.patient_subgraphs as patient_subgraphs_module
from pipeline.patient_subgraphs import (
    PatientSubgraphBuildConfig,
    build_patient_subgraphs,
)
from tests.milestone6_helpers import read_parquet_rows, write_parquet_rows


def write_subgraph_inputs(tmp_path: Path) -> PatientSubgraphBuildConfig:
    features_root = tmp_path / "Dataset" / "processed" / "features"
    training_root = tmp_path / "Dataset" / "processed" / "training"
    graph_root = tmp_path / "Dataset" / "processed" / "graph" / "milestone8"
    subgraphs_root = graph_root / "patient_subgraphs"

    write_parquet_rows(
        features_root / "event_sequences.parquet",
        (
            "source",
            "split",
            "stay_uid",
            "event_type",
            "event_token",
            "event_time_hours_from_admit",
            "feature_version",
        ),
        (
            (
                "mimiciv",
                "train",
                "stay-train",
                "lab",
                "lactate",
                5.0,
                "temporal-features-v2",
            ),
            (
                "mimiciv",
                "validation",
                "stay-val",
                "lab",
                "lactate",
                8.0,
                "temporal-features-v2",
            ),
            (
                "mimiciv",
                "validation",
                "stay-val",
                "lab",
                "future",
                25.0,
                "temporal-features-v2",
            ),
            (
                "mimiciv",
                "validation",
                "stay-val",
                "lab",
                "validation_only",
                7.0,
                "temporal-features-v2",
            ),
            (
                "eicu_crd",
                "external",
                "stay-ext",
                "lab",
                "lactate",
                4.0,
                "temporal-features-v2",
            ),
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
            "feature_version",
        ),
        (
            (
                "mimiciv",
                "train",
                "stay-train",
                "rg-train",
                "condition:a",
                "rxnorm:1",
                1,
                True,
                "temporal-features-v2",
            ),
            (
                "mimiciv",
                "train",
                "stay-train",
                "rg-train",
                "condition:a",
                "rxnorm:2",
                2,
                True,
                "temporal-features-v2",
            ),
            (
                "mimiciv",
                "train",
                "stay-train",
                "rg-train",
                "condition:a",
                "rxnorm:3",
                3,
                False,
                "temporal-features-v2",
            ),
            (
                "mimiciv",
                "validation",
                "stay-val",
                "rg-val",
                "condition:a",
                "rxnorm:1",
                1,
                True,
                "temporal-features-v2",
            ),
            (
                "mimiciv",
                "validation",
                "stay-val",
                "rg-val",
                "condition:a",
                "rxnorm:3",
                3,
                False,
                "temporal-features-v2",
            ),
            (
                "eicu_crd",
                "external",
                "stay-ext",
                "rg-ext",
                "condition:b",
                "rxnorm:1",
                1,
                False,
                "temporal-features-v2",
            ),
        ),
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
        ),
        (
            (
                "condition|condition:a",
                "medication|rxnorm:1",
                "condition",
                "medication",
                "condition_medication_train_positive",
                3,
                "mimiciv",
                "train",
                "graph-suitability-v1",
                "temporal-features-v2",
            ),
            (
                "condition|condition:a",
                "lab|lactate",
                "condition",
                "lab",
                "condition_lab_predecision",
                2,
                "mimiciv",
                "train",
                "graph-suitability-v1",
                "temporal-features-v2",
            ),
            (
                "medication|rxnorm:1",
                "medication|rxnorm:2",
                "medication",
                "medication",
                "medication_medication_train_coprescribed",
                1,
                "mimiciv",
                "train",
                "graph-suitability-v1",
                "temporal-features-v2",
            ),
        ),
    )
    return PatientSubgraphBuildConfig(
        features_root=features_root,
        training_root=training_root,
        graph_root=graph_root,
        subgraphs_root=subgraphs_root,
        manifest_path=tmp_path / "reports" / "patient_subgraphs_manifest.json",
        subgraph_batch_count=2,
        subgraph_join_shards=2,
        edge_duckdb_threads=1,
    )


def test_patient_subgraphs_use_train_fit_edges_and_predecision_context(
    tmp_path: Path,
) -> None:
    config = write_subgraph_inputs(tmp_path)

    manifest = build_patient_subgraphs(config)

    assert manifest["status"] == "completed"
    assert manifest["versions"]["feature_version"] == "temporal-features-v2"
    assert manifest["parameters"]["build_strategy"] == "source_stay_hash_batches"
    assert manifest["parameters"]["batch_key"] == ["source", "stay_uid"]
    assert manifest["parameters"]["subgraph_batches"] == 2
    assert manifest["parameters"]["join_shards_per_batch"] == 2
    assert manifest["parameters"]["edge_part_count"] == 4
    assert manifest["parameters"]["edge_duckdb_threads"] == 1
    assert manifest["graph_fit_scope"] == [
        {"edge_count": 3, "fit_source": "mimiciv", "fit_split": "train"}
    ]
    assert {record["table_name"] for record in manifest["tables"]} == {
        "subgraph_nodes",
        "subgraph_edges",
        "subgraph_candidates",
        "subgraph_index",
    }
    table_records = {record["table_name"]: record for record in manifest["tables"]}
    assert table_records["subgraph_nodes"]["part_count"] == 2
    assert table_records["subgraph_edges"]["part_count"] == 4
    assert table_records["subgraph_candidates"]["part_count"] == 4
    assert table_records["subgraph_index"]["part_count"] == 2
    assert table_records["subgraph_edges"]["build_strategy"] == (
        "encoded_relation_specific_join_shards"
    )
    assert table_records["subgraph_edges"]["join_shards_per_node_batch"] == 2
    assert all(
        sum(record["batch_row_counts"]) == record["row_count"]
        for record in manifest["tables"]
    )
    assert not (config.subgraphs_root / "_subgraph_parts").exists()

    index_rows = read_parquet_rows(config.subgraph_index_path)
    assert {row["subgraph_id"] for row in index_rows} == {
        "rg-train",
        "rg-val",
        "rg-ext",
    }
    nodes = read_parquet_rows(config.subgraph_nodes_path)
    val_nodes = {row["node_id"]: row for row in nodes if row["subgraph_id"] == "rg-val"}
    assert "condition|condition:a" in val_nodes
    assert "medication|rxnorm:3" in val_nodes
    assert val_nodes["medication|rxnorm:3"]["cold_start"] is True
    assert "lab|lactate" in val_nodes
    assert "lab|future" not in val_nodes
    assert "lab|validation_only" not in val_nodes

    ext_nodes = {row["node_id"] for row in nodes if row["subgraph_id"] == "rg-ext"}
    assert ext_nodes == {"condition|condition:b", "medication|rxnorm:1"}

    edges = read_parquet_rows(config.subgraph_edges_path)
    assert {(row["subgraph_id"], row["src_id"], row["dst_id"]) for row in edges} == {
        ("rg-train", "condition|condition:a", "medication|rxnorm:1"),
        ("rg-train", "condition|condition:a", "lab|lactate"),
        ("rg-train", "medication|rxnorm:1", "medication|rxnorm:2"),
        ("rg-val", "condition|condition:a", "medication|rxnorm:1"),
        ("rg-val", "condition|condition:a", "lab|lactate"),
    }
    assert {row["fit_source"] for row in edges} == {"mimiciv"}
    assert {row["fit_split"] for row in edges} == {"train"}
    assert all("future" not in row["dst_id"] for row in edges)

    candidates = read_parquet_rows(config.subgraph_candidates_path)
    assert len(candidates) == 6
    assert all(row["candidate_node_index"] is not None for row in candidates)

    manifest_text = config.manifest_path.read_text(encoding="utf-8")
    assert "stay-val" not in manifest_text
    assert "rg-val" not in manifest_text
    assert (
        json.loads(manifest_text)["data_safety"]["manifest_contains_patient_rows"]
        is False
    )


def test_patient_subgraphs_fail_closed_on_non_train_graph_edges(tmp_path: Path) -> None:
    config = write_subgraph_inputs(tmp_path)
    edge_path = config.graph_edges_path
    edges = read_parquet_rows(edge_path)
    columns = tuple(edges[0])
    invalid = dict(edges[0])
    invalid["fit_split"] = "validation"
    write_parquet_rows(
        edge_path,
        columns,
        tuple(tuple(row[column] for column in columns) for row in (*edges, invalid)),
    )

    manifest = build_patient_subgraphs(config)

    assert manifest["status"] == "failed_graph_fit_scope"
    assert manifest["tables"] == []


def test_patient_subgraph_join_failure_reports_exact_shard_and_cleans_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = write_subgraph_inputs(tmp_path)
    original_copy = patient_subgraphs_module.copy_query_to_parquet

    def fail_second_edge_part(
        connection: duckdb.DuckDBPyConnection,
        query: str,
        output_path: Path,
    ) -> int:
        if output_path.name == "subgraph_edges_part_0001.parquet":
            raise RuntimeError("Out of Memory Error: synthetic spill limit")
        return original_copy(connection, query, output_path)

    monkeypatch.setattr(
        patient_subgraphs_module,
        "copy_query_to_parquet",
        fail_second_edge_part,
    )

    manifest = build_patient_subgraphs(config)

    assert manifest["status"] == "failed"
    edge_record = manifest["tables"][-1]
    assert edge_record["table_name"] == "subgraph_edges"
    assert edge_record["completed_batch_count"] == 1
    assert edge_record["failed_batch_index"] == 1
    assert edge_record["failed_node_batch_index"] == 0
    assert edge_record["failed_join_shard_index"] == 1
    assert not (config.subgraphs_root / "_subgraph_parts").exists()
