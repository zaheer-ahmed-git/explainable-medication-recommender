"""Synthetic leakage and artifact-lock tests for GNN cross-fit caches."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest

import pipeline.gnn_training.crossfit as crossfit_module
from pipeline.gate_recovery import patient_fold_sql
from pipeline.gnn_training.config import (
    CROSS_FIT_SELECTION_SCOPE,
    GNNTrainingConfig,
)
from pipeline.gnn_training.crossfit import (
    CROSS_FIT_CAPACITY_ENV,
    _capacity_review,
    crossfit_artifact_lock_errors,
    prepare_crossfit_graph_caches,
)
from pipeline.gnn_training.dataset import (
    GNNFeatureLayoutSpec,
    iter_shard_examples,
)
from pipeline.gnn_training.graph_encode import (
    RELATION_TO_INDEX,
    SELF_LOOP_RELATION,
    UNK_INDEX,
)
from pipeline.training_contract import sha256_file
from tests.milestone6_helpers import sql_string, write_parquet_rows


def _config(tmp_path: Path) -> GNNTrainingConfig:
    dataset_root = tmp_path / "synthetic_dataset"
    phase_root = dataset_root / "processed" / "phase8_p0"
    graph_root = phase_root / "graph" / "milestone8"
    reports_root = tmp_path / "reports"
    return GNNTrainingConfig(
        dataset_root=dataset_root,
        reports_root=reports_root,
        graph_root=graph_root,
        subgraphs_root=graph_root / "patient_subgraphs",
        features_root=phase_root / "features",
        training_root=phase_root / "training",
        neural_root=phase_root / "neural",
        gnn_root=phase_root / "gnn",
        graph_reference_scores_path=(
            phase_root / "evaluation" / "milestone8b" / "graph_ablation_scores.parquet"
        ),
        graph_reference_report_path=reports_root / "graph_reference.json",
        contract_lock_path=reports_root / "contract.json",
        subgraphs_manifest_path=reports_root / "subgraphs.json",
        neural_selection_path=reports_root / "neural_selection.json",
        prepare_manifest_path=reports_root / "gnn_prepare.json",
        crossfit_graph_manifest_path=reports_root / "crossfit.json",
        gnn_training_report_path=reports_root / "gnn_training.json",
        gnn_score_report_path=reports_root / "gnn_score.json",
        gnn_selection_report_path=reports_root / "gnn_selection.json",
        fusion_training_report_path=reports_root / "fusion_training.json",
        fusion_score_report_path=reports_root / "fusion_score.json",
        fusion_selection_report_path=reports_root / "fusion_selection.json",
        mode="development",
        allow_ungated=True,
        seed=20260617,
        fold_count=2,
        shard_count=2,
    )


def _patient_for_each_fold(config: GNNTrainingConfig) -> dict[int, str]:
    by_fold: dict[int, str] = {}
    with duckdb.connect(database=":memory:") as connection:
        for index in range(100):
            patient = f"invented-patient-{index}"
            row = connection.execute(
                f"""
SELECT {
                    patient_fold_sql(
                        seed=config.seed,
                        fold_count=config.fold_count,
                        alias="candidate",
                    )
                }
FROM (SELECT {sql_string(patient)} AS patient_uid) AS candidate
"""
            ).fetchone()
            assert row is not None
            by_fold.setdefault(int(row[0]), patient)
            if len(by_fold) == config.fold_count:
                break
    assert set(by_fold) == set(range(config.fold_count))
    return by_fold


def _write_inputs(config: GNNTrainingConfig) -> dict[int, str]:
    patients = _patient_for_each_fold(config)
    group_specs = (
        (
            0,
            patients[0],
            "invented-stay-a",
            "group-a",
            "condition-a",
            "medication-a",
            "lab-a",
        ),
        (
            1,
            patients[1],
            "invented-stay-b",
            "group-b",
            "condition-b",
            "medication-b",
            "lab-b",
        ),
    )

    write_parquet_rows(
        config.subgraph_index_path,
        (
            "source",
            "split",
            "subgraph_id",
            "index_condition_token",
            "node_count",
            "edge_count",
            "candidate_count",
            "positive_count",
        ),
        tuple(
            (
                "mimiciv",
                "train",
                group_id,
                condition,
                5,
                5,
                2,
                2,
            )
            for (
                _fold,
                _patient,
                _stay,
                group_id,
                condition,
                _medication,
                _lab,
            ) in group_specs
        ),
    )

    node_rows: list[tuple[object, ...]] = []
    candidate_rows: list[tuple[object, ...]] = []
    pcm_rows: list[tuple[object, ...]] = []
    event_rows: list[tuple[object, ...]] = []
    edge_rows: list[tuple[object, ...]] = []
    for (
        _fold,
        patient,
        stay,
        group_id,
        condition,
        medication,
        lab,
    ) in group_specs:
        node_rows.extend(
            (
                (
                    "mimiciv",
                    "train",
                    group_id,
                    10,
                    f"condition|{condition}",
                    "condition",
                    "query_condition",
                    False,
                    False,
                ),
                (
                    "mimiciv",
                    "train",
                    group_id,
                    20,
                    f"medication|{medication}",
                    "medication",
                    "candidate_medication",
                    False,
                    False,
                ),
                (
                    "mimiciv",
                    "train",
                    group_id,
                    30,
                    "medication|shared-medication",
                    "medication",
                    "candidate_medication",
                    False,
                    False,
                ),
                (
                    "mimiciv",
                    "train",
                    group_id,
                    40,
                    f"lab|{lab}",
                    "lab",
                    "observed_context",
                    True,
                    False,
                ),
                (
                    "mimiciv",
                    "train",
                    group_id,
                    50,
                    f"lab|{lab}-future",
                    "lab",
                    "observed_context",
                    True,
                    False,
                ),
            )
        )
        candidate_rows.extend(
            (
                (
                    "mimiciv",
                    "train",
                    group_id,
                    condition,
                    medication,
                    20,
                    1,
                    True,
                    False,
                ),
                (
                    "mimiciv",
                    "train",
                    group_id,
                    condition,
                    "shared-medication",
                    30,
                    2,
                    True,
                    False,
                ),
            )
        )
        pcm_rows.extend(
            (
                (
                    "mimiciv",
                    "train",
                    patient,
                    stay,
                    group_id,
                    condition,
                    medication,
                    1,
                    True,
                ),
                (
                    "mimiciv",
                    "train",
                    patient,
                    stay,
                    group_id,
                    condition,
                    "shared-medication",
                    2,
                    True,
                ),
            )
        )
        event_rows.append(("mimiciv", "train", stay, "lab", lab, 12.0, 2.0))
        event_rows.append(("mimiciv", "train", stay, "lab", f"{lab}-future", 36.0, 3.0))
        # Deliberately stale support values prove the cross-fit cache replaces,
        # rather than reuses, full-train supports.
        edge_rows.extend(
            (
                (
                    "mimiciv",
                    "train",
                    group_id,
                    10,
                    20,
                    f"condition|{condition}",
                    f"medication|{medication}",
                    "condition",
                    "medication",
                    "condition_medication_train_positive",
                    99,
                ),
                (
                    "mimiciv",
                    "train",
                    group_id,
                    10,
                    30,
                    f"condition|{condition}",
                    "medication|shared-medication",
                    "condition",
                    "medication",
                    "condition_medication_train_positive",
                    99,
                ),
                (
                    "mimiciv",
                    "train",
                    group_id,
                    10,
                    40,
                    f"condition|{condition}",
                    f"lab|{lab}",
                    "condition",
                    "lab",
                    "condition_lab_predecision",
                    99,
                ),
                (
                    "mimiciv",
                    "train",
                    group_id,
                    20,
                    30,
                    f"medication|{medication}",
                    "medication|shared-medication",
                    "medication",
                    "medication",
                    "medication_medication_train_coprescribed",
                    99,
                ),
                (
                    "mimiciv",
                    "train",
                    group_id,
                    10,
                    50,
                    f"condition|{condition}",
                    f"lab|{lab}-future",
                    "condition",
                    "lab",
                    "condition_lab_predecision",
                    99,
                ),
            )
        )

    write_parquet_rows(
        config.subgraph_nodes_path,
        (
            "source",
            "split",
            "subgraph_id",
            "node_index",
            "node_id",
            "node_type",
            "node_role",
            "observed_predecision",
            "cold_start",
        ),
        tuple(node_rows),
    )
    write_parquet_rows(
        config.subgraph_candidates_path,
        (
            "source",
            "split",
            "subgraph_id",
            "index_condition_token",
            "candidate_medication_token",
            "candidate_node_index",
            "candidate_rank",
            "label_prescribed",
            "cold_start",
        ),
        tuple(candidate_rows),
    )
    write_parquet_rows(
        config.patient_condition_medication_path,
        (
            "source",
            "split",
            "patient_uid",
            "stay_uid",
            "ranking_group_id",
            "index_condition_token",
            "candidate_medication_token",
            "candidate_rank",
            "label_prescribed",
        ),
        tuple(pcm_rows),
    )
    write_parquet_rows(
        config.event_sequences_path,
        (
            "source",
            "split",
            "stay_uid",
            "event_type",
            "event_token",
            "event_time_hours_from_admit",
            "value_numeric",
        ),
        tuple(event_rows),
    )
    write_parquet_rows(
        config.subgraph_edges_path,
        (
            "source",
            "split",
            "subgraph_id",
            "src_node_index",
            "dst_node_index",
            "src_id",
            "dst_id",
            "src_type",
            "dst_type",
            "relation_type",
            "support_count",
        ),
        tuple(edge_rows),
    )
    return patients


def _read_parquet_tree(path: Path) -> list[dict[str, object]]:
    scan_path = path if path.is_file() else path / "**" / "*.parquet"
    hive_partitioning = "FALSE" if path.is_file() else "TRUE"
    with duckdb.connect(database=":memory:") as connection:
        cursor = connection.execute(
            f"""
SELECT *
FROM read_parquet(
    {sql_string(scan_path)},
    hive_partitioning = {hive_partitioning}
)
"""
        )
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def test_crossfit_excludes_heldout_support_and_vocab_and_locks_artifacts(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    patients = _write_inputs(config)

    manifest = prepare_crossfit_graph_caches(config)

    assert manifest["status"] == "completed"
    assert manifest["fit_scope"] == {
        "source": "mimiciv",
        "split": "train",
        "grouping_unit": "patient_uid",
    }
    assert manifest["scope"] == CROSS_FIT_SELECTION_SCOPE
    assert manifest["selection_eligible"] is True
    assert manifest["full_train_cache_reused_for_selection"] is False
    assert len(manifest["folds"]) == config.fold_count
    assert crossfit_artifact_lock_errors(config) == []

    layout_spec = GNNFeatureLayoutSpec.from_json(config.fold_feature_layout_path(0))
    examples = [
        example
        for shard_index in range(config.shard_count)
        for example in iter_shard_examples(
            config,
            layout_spec,
            split="train",
            shard_index=shard_index,
            shards_root=config.fold_graph_root(0) / "cache" / "shards",
        )
    ]
    assert {example.ranking_group_id for example in examples} == {
        "group-a",
        "group-b",
    }

    fold_zero_graph = _read_parquet_tree(config.fold_graph_edges_path(0))
    graph_tokens = {
        str(row[field])
        for row in fold_zero_graph
        if "relation_type" in row
        for field in ("src_id", "dst_id")
    }
    assert "condition|condition-a" not in graph_tokens
    assert "medication|medication-a" not in graph_tokens
    assert "lab|lab-a" not in graph_tokens
    assert "condition|condition-b" in graph_tokens
    assert "lab|lab-b-future" not in graph_tokens

    vocab_rows = _read_parquet_tree(config.fold_concept_vocabulary_path(0).parent)
    vocabulary = {str(row["node_id"]) for row in vocab_rows}
    assert "condition|condition-a" not in vocabulary
    assert "medication|medication-a" not in vocabulary
    assert "lab|lab-a" not in vocabulary
    assert "condition|condition-b" in vocabulary
    assert "lab|lab-b-future" not in vocabulary

    node_rows = _read_parquet_tree(
        config.fold_graph_root(0) / "cache" / "shards" / "nodes"
    )
    held_out_medication = [
        row
        for row in node_rows
        if row["ranking_group_id"] == "group-a"
        and int(row["node_type_index"]) == 1
        and int(row["node_concept_index"]) == UNK_INDEX
    ]
    assert held_out_medication
    assert not any(
        row["ranking_group_id"] == "group-a" and int(row["node_type_index"]) == 2
        for row in node_rows
    )

    candidate_rows = _read_parquet_tree(
        config.fold_graph_root(0) / "cache" / "shards" / "candidates"
    )
    assert sum(row["ranking_group_id"] == "group-a" for row in candidate_rows) == 2

    edge_rows = _read_parquet_tree(
        config.fold_graph_root(0) / "cache" / "shards" / "edges"
    )
    non_self_support = [
        float(row["edge_log_support"])
        for row in edge_rows
        if int(row["relation_index"]) != RELATION_TO_INDEX[SELF_LOOP_RELATION]
    ]
    # Every recomputed support is one in this fixture, so even non-self edges
    # transform to log(2); stale support=99 would transform to log(100).
    assert non_self_support
    assert all(math.isclose(value, math.log(2.0)) for value in non_self_support)
    assert not any(
        math.isclose(float(row["edge_log_support"]), math.log(100.0))
        for row in edge_rows
    )

    public_text = config.crossfit_graph_manifest_path.read_text(encoding="utf-8")
    assert all(value not in public_text for value in patients.values())
    for fold in manifest["folds"]:
        locks = fold["artifact_locks"]
        fold_index = int(fold["fold_index"])
        assert locks["artifact_tree_digest"]
        assert locks["cache_manifest"]["sha256"] == sha256_file(
            config.fold_cache_manifest_path(fold_index)
        )
        local_manifest = json.loads(
            config.fold_cache_manifest_path(fold_index).read_text(encoding="utf-8")
        )
        assert local_manifest["artifact_hashes"]
        assert local_manifest["artifact_tree_digest"] == locks["artifact_tree_digest"]


def test_crossfit_hash_verifier_detects_artifact_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_inputs(config)
    assert prepare_crossfit_graph_caches(config)["status"] == "completed"
    assert crossfit_artifact_lock_errors(config) == []

    layout_path = config.fold_feature_layout_path(0)
    layout_path.write_text(
        layout_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    errors = crossfit_artifact_lock_errors(config)
    codes = {error["code"] for error in errors}
    assert "crossfit_artifact_hash_mismatch" in codes
    assert "crossfit_artifact_tree_mismatch" in codes


def test_crossfit_hash_verifier_reads_each_artifact_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _write_inputs(config)
    assert prepare_crossfit_graph_caches(config)["status"] == "completed"
    original = crossfit_module.sha256_file
    hashed_paths: list[Path] = []

    def tracked_hash(path: Path) -> str:
        hashed_paths.append(Path(path))
        return original(path)

    monkeypatch.setattr(crossfit_module, "sha256_file", tracked_hash)

    assert crossfit_artifact_lock_errors(config) == []
    assert hashed_paths
    assert len(hashed_paths) == len(set(hashed_paths))


def test_crossfit_preflight_reuses_allocation_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    _write_inputs(config)
    assert prepare_crossfit_graph_caches(config)["status"] == "completed"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("OAR_JOB_ID", "synthetic-123")
    monkeypatch.setenv("WORK_SCRATCH", str(scratch))

    assert crossfit_artifact_lock_errors(config) == []
    attestation = scratch / "gnn-preflight" / "oar-synthetic-123.json"
    assert attestation.is_file()

    def unexpected_rehash(*args: object, **kwargs: object) -> dict[str, str]:
        del args, kwargs
        raise AssertionError("allocation attestation should bypass content rehashing")

    monkeypatch.setattr(crossfit_module, "_artifact_hashes", unexpected_rehash)
    assert crossfit_artifact_lock_errors(config) == []


def test_protected_crossfit_requires_explicit_capacity_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_config(tmp_path), allow_ungated=False)
    config.cache_root.mkdir(parents=True)
    monkeypatch.delenv(CROSS_FIT_CAPACITY_ENV, raising=False)

    with pytest.raises(ValueError, match=CROSS_FIT_CAPACITY_ENV):
        _capacity_review(config)
