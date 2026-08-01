"""Synthetic tests for bounded, leakage-labelled GNN cache preparation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import duckdb
import pytest

from pipeline.gnn_training.config import (
    FULL_TRAIN_REFIT_SCOPE,
    PREPARE_PENDING_STATUS,
    RELATION_TO_INDEX,
    GNNTrainingConfig,
)
from pipeline.gnn_training.data import (
    FROZEN_TRANSFORMER_CACHE_ARTIFACT_LOCK_VERSION,
    FROZEN_TRANSFORMER_CACHE_SCHEMA_VERSION,
    TRANSFORMER_CONTEXT_TABLE,
    TRANSFORMER_LOGIT_TABLE,
    _promote_paths,
    _transformer_cache_status,
    artifact_tree_digest,
    prepare_gnn_caches,
)
from pipeline.gnn_training.dataset import (
    GNNFeatureLayoutSpec,
    iter_shard_examples,
)
from pipeline.gnn_training.graph_encode import (
    FORWARD_RELATION_TYPES,
    PAD_INDEX,
    SELF_LOOP_RELATION,
    UNK_INDEX,
)
from pipeline.gate_recovery import patient_fold_sql
from pipeline.training_contract import sha256_file
from tests.milestone6_helpers import sql_string, write_parquet_rows


def _config(tmp_path: Path, *, shard_count: int = 3) -> GNNTrainingConfig:
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
        mode="final",
        allow_ungated=True,
        shard_count=shard_count,
        fold_count=3,
    )


def _write_inputs(config: GNNTrainingConfig) -> None:
    index_columns = (
        "source",
        "split",
        "subgraph_id",
        "index_condition_token",
        "node_count",
        "edge_count",
        "candidate_count",
        "positive_count",
    )
    write_parquet_rows(
        config.subgraph_index_path,
        index_columns,
        (
            ("mimiciv", "train", "group-train", "condition:train", 7, 6, 3, 1),
            (
                "mimiciv",
                "train",
                "group-train-zero",
                "condition:zero",
                2,
                0,
                1,
                0,
            ),
            (
                "mimiciv",
                "validation",
                "group-validation",
                "condition:validation-only",
                2,
                0,
                1,
                0,
            ),
            (
                "mimiciv",
                "test",
                "group-test",
                "condition:train",
                2,
                0,
                1,
                1,
            ),
        ),
    )

    node_columns = (
        "source",
        "split",
        "subgraph_id",
        "node_index",
        "node_id",
        "node_type",
        "node_role",
        "observed_predecision",
        "cold_start",
    )
    train_nodes = (
        (10, "condition|condition:train", "condition", "query_condition", False),
        (20, "medication|rxnorm:1", "medication", "candidate_medication", False),
        (30, "medication|rxnorm:2", "medication", "candidate_medication", False),
        (40, "medication|rxnorm:3", "medication", "candidate_medication", False),
        (50, "lab|train-lab", "lab", "observed_context", True),
        (60, "vital|train-vital", "vital", "observed_context", True),
        (
            70,
            "intervention|train-intervention",
            "intervention",
            "observed_context",
            True,
        ),
    )
    node_rows: list[tuple[object, ...]] = [
        (
            "mimiciv",
            "train",
            "group-train",
            index,
            node_id,
            node_type,
            role,
            observed,
            False,
        )
        for index, node_id, node_type, role, observed in train_nodes
    ]
    node_rows.extend(
        (
            (
                "mimiciv",
                "train",
                "group-train-zero",
                3,
                "condition|condition:zero",
                "condition",
                "query_condition",
                False,
                False,
            ),
            (
                "mimiciv",
                "train",
                "group-train-zero",
                9,
                "medication|rxnorm:zero",
                "medication",
                "candidate_medication",
                False,
                False,
            ),
            (
                "mimiciv",
                "validation",
                "group-validation",
                101,
                "condition|condition:validation-only",
                "condition",
                "query_condition",
                False,
                True,
            ),
            (
                "mimiciv",
                "validation",
                "group-validation",
                303,
                "medication|rxnorm:validation-only",
                "medication",
                "candidate_medication",
                False,
                True,
            ),
            (
                "mimiciv",
                "test",
                "group-test",
                5,
                "condition|condition:train",
                "condition",
                "query_condition",
                False,
                False,
            ),
            (
                "mimiciv",
                "test",
                "group-test",
                8,
                "medication|rxnorm:1",
                "medication",
                "candidate_medication",
                False,
                False,
            ),
        )
    )
    write_parquet_rows(config.subgraph_nodes_path, node_columns, tuple(node_rows))

    edge_columns = (
        "source",
        "split",
        "subgraph_id",
        "src_node_index",
        "dst_node_index",
        "relation_type",
        "support_count",
    )
    write_parquet_rows(
        config.subgraph_edges_path,
        edge_columns,
        (
            (
                "mimiciv",
                "train",
                "group-train",
                10,
                20,
                FORWARD_RELATION_TYPES[0],
                3,
            ),
            (
                "mimiciv",
                "train",
                "group-train",
                10,
                50,
                FORWARD_RELATION_TYPES[1],
                1,
            ),
            (
                "mimiciv",
                "train",
                "group-train",
                10,
                60,
                FORWARD_RELATION_TYPES[2],
                2,
            ),
            (
                "mimiciv",
                "train",
                "group-train",
                10,
                70,
                FORWARD_RELATION_TYPES[3],
                4,
            ),
            (
                "mimiciv",
                "train",
                "group-train",
                20,
                30,
                FORWARD_RELATION_TYPES[4],
                1,
            ),
            (
                "mimiciv",
                "train",
                "group-train",
                40,
                30,
                FORWARD_RELATION_TYPES[4],
                3,
            ),
        ),
    )

    candidate_columns = (
        "source",
        "split",
        "subgraph_id",
        "index_condition_token",
        "candidate_medication_token",
        "candidate_node_index",
        "candidate_rank",
        "label_prescribed",
        "cold_start",
    )
    write_parquet_rows(
        config.subgraph_candidates_path,
        candidate_columns,
        (
            (
                "mimiciv",
                "train",
                "group-train",
                "condition:train",
                "rxnorm:1",
                20,
                1,
                True,
                False,
            ),
            (
                "mimiciv",
                "train",
                "group-train",
                "condition:train",
                "rxnorm:2",
                30,
                2,
                False,
                False,
            ),
            (
                "mimiciv",
                "train",
                "group-train",
                "condition:train",
                "rxnorm:3",
                40,
                3,
                False,
                False,
            ),
            (
                "mimiciv",
                "train",
                "group-train-zero",
                "condition:zero",
                "rxnorm:zero",
                9,
                1,
                False,
                False,
            ),
            (
                "mimiciv",
                "validation",
                "group-validation",
                "condition:validation-only",
                "rxnorm:validation-only",
                303,
                1,
                False,
                True,
            ),
            (
                "mimiciv",
                "test",
                "group-test",
                "condition:train",
                "rxnorm:1",
                8,
                1,
                True,
                False,
            ),
        ),
    )

    write_parquet_rows(
        config.patient_condition_medication_path,
        (
            "source",
            "split",
            "patient_uid",
            "ranking_group_id",
            "index_condition_token",
            "candidate_medication_token",
            "candidate_rank",
            "label_prescribed",
        ),
        (
            (
                "mimiciv",
                "train",
                "synthetic-patient-a",
                "group-train",
                "condition:train",
                "rxnorm:1",
                1,
                True,
            ),
            (
                "mimiciv",
                "train",
                "synthetic-patient-a",
                "group-train",
                "condition:train",
                "rxnorm:2",
                2,
                False,
            ),
            (
                "mimiciv",
                "train",
                "synthetic-patient-a",
                "group-train",
                "condition:train",
                "rxnorm:3",
                3,
                False,
            ),
            (
                "mimiciv",
                "train",
                "synthetic-patient-b",
                "group-train-zero",
                "condition:zero",
                "rxnorm:zero",
                1,
                False,
            ),
            (
                "mimiciv",
                "validation",
                "synthetic-patient-c",
                "group-validation",
                "condition:validation-only",
                "rxnorm:validation-only",
                1,
                False,
            ),
            (
                "mimiciv",
                "test",
                "synthetic-patient-d",
                "group-test",
                "condition:train",
                "rxnorm:1",
                1,
                True,
            ),
        ),
    )


def _read_tree(path: Path) -> list[dict[str, object]]:
    with duckdb.connect(database=":memory:") as connection:
        cursor = connection.execute(
            f"""
SELECT *
FROM read_parquet(
    {sql_string(path / "**" / "*.parquet")},
    hive_partitioning = TRUE
)
"""
        )
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def test_prepare_builds_train_vocab_complete_shards_and_safe_manifests(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_inputs(config)

    manifest = prepare_gnn_caches(config)

    assert manifest["status"] == PREPARE_PENDING_STATUS
    assert manifest["preparation_complete"] is False
    assert manifest["components"] == {
        "graph_cache": "completed",
        "crossfit_graph_caches": "pending",
        "frozen_transformer_cache": "pending",
    }
    assert manifest["scope"] == FULL_TRAIN_REFIT_SCOPE
    assert manifest["selection_eligible"] is False
    assert manifest["leakage_policy"]["crossfit_claimed"] is False
    cache_manifest = json.loads(config.cache_manifest_path.read_text(encoding="utf-8"))
    assert cache_manifest["data_safety"] == {
        "contains_row_samples": False,
        "direct_patient_identifiers_present": False,
        "local_cache_contains_patient_level_rows": True,
        "local_cache_is_restricted": True,
        "manifest_contains_patient_rows": False,
        "restricted_join_keys_present": True,
    }

    layout = json.loads(config.feature_layout_path.read_text(encoding="utf-8"))
    assert layout["scope"] == FULL_TRAIN_REFIT_SCOPE
    assert layout["selection_eligible"] is False
    assert layout["pad_index"] == PAD_INDEX
    assert layout["unk_index"] == UNK_INDEX

    vocab_rows = _read_tree(config.graph_node_vocabulary_path.parent)
    vocab = {str(row["node_id"]): int(row["concept_index"]) for row in vocab_rows}
    assert vocab["<PAD>"] == PAD_INDEX
    assert vocab["<UNK>"] == UNK_INDEX
    assert "condition|condition:validation-only" not in vocab

    node_rows = _read_tree(config.shards_root / "nodes")
    validation = [
        row for row in node_rows if row["ranking_group_id"] == "group-validation"
    ]
    assert validation
    assert {int(row["node_concept_index"]) for row in validation} == {UNK_INDEX}

    group_rows = _read_tree(config.shards_root / "groups")
    cached_groups = {str(row["ranking_group_id"]) for row in group_rows}
    assert cached_groups == {"group-train", "group-validation", "group-test"}
    assert "group-train-zero" not in cached_groups
    train_group = next(
        row for row in group_rows if row["ranking_group_id"] == "group-train"
    )
    with duckdb.connect(database=":memory:") as connection:
        expected_fold = connection.execute(
            f"""
SELECT {patient_fold_sql(seed=config.seed, fold_count=config.fold_count, alias="pcm")}
FROM (
    SELECT 'synthetic-patient-a' AS patient_uid
) AS pcm
"""
        ).fetchone()
    assert expected_fold is not None
    assert int(train_group["patient_fold_id"]) == int(expected_fold[0])

    # Every table uses the same deterministic partition for a complete group.
    for group_id in cached_groups:
        shard_sets = []
        for table_name in ("groups", "nodes", "edges", "candidates"):
            rows = [
                row
                for row in _read_tree(config.shards_root / table_name)
                if row["ranking_group_id"] == group_id
            ]
            assert rows
            shard_sets.append({int(row["shard_id"]) for row in rows})
        assert len({next(iter(values)) for values in shard_sets}) == 1

    # Public and local JSON metadata are aggregate/schema-only and never carry
    # the direct patient identifier column or a row value.
    for path in (
        config.prepare_manifest_path,
        config.cache_manifest_path,
        config.feature_layout_path,
    ):
        text = path.read_text(encoding="utf-8")
        assert "patient_uid" not in text
        assert "synthetic-patient" not in text
    for table_name in ("groups", "nodes", "edges", "candidates"):
        rows = _read_tree(config.shards_root / table_name)
        assert all("patient_uid" not in row for row in rows)

    spec = GNNFeatureLayoutSpec.from_json(config.feature_layout_path)
    examples_by_split = {
        split: [
            example
            for shard_index in range(config.shard_count)
            for example in iter_shard_examples(
                config,
                spec,
                split=split,
                shard_index=shard_index,
            )
        ]
        for split in ("train", "validation", "test")
    }
    assert [example.ranking_group_id for example in examples_by_split["train"]] == [
        "group-train"
    ]
    assert examples_by_split["validation"][0].node_concept_index.tolist() == [
        UNK_INDEX,
        UNK_INDEX,
    ]
    assert examples_by_split["validation"][0].num_edges == 2  # self loops only


def test_prepare_expands_relations_and_normalizes_log_support(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_inputs(config)
    assert prepare_gnn_caches(config)["components"]["graph_cache"] == "completed"

    edges = [
        row
        for row in _read_tree(config.shards_root / "edges")
        if row["ranking_group_id"] == "group-train"
    ]
    # Six forward edges + six deterministic reverse edges + seven self loops.
    assert len(edges) == 19
    relation_indexes = {int(row["relation_index"]) for row in edges}
    assert set(range(len(RELATION_TO_INDEX))) <= relation_indexes
    assert RELATION_TO_INDEX[SELF_LOOP_RELATION] in relation_indexes

    med_relation = RELATION_TO_INDEX[FORWARD_RELATION_TYPES[4]]
    incoming = [
        row
        for row in edges
        if int(row["relation_index"]) == med_relation
        and int(row["dst_node_index"]) == 30
    ]
    assert len(incoming) == 2
    supports = sorted(float(row["edge_log_support"]) for row in incoming)
    assert supports == [math.log1p(1), math.log1p(3)]
    assert sum(float(row["edge_weight"]) for row in incoming) == pytest.approx(1.0)

    sums: dict[tuple[int, int], float] = {}
    for row in edges:
        key = (int(row["relation_index"]), int(row["dst_node_index"]))
        sums[key] = sums.get(key, 0.0) + float(row["edge_weight"])
    assert all(value == pytest.approx(1.0) for value in sums.values())


def test_stage_owned_promotion_rolls_back_as_one_transaction(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.gnn_root.mkdir(parents=True)
    first = config.gnn_root / "first.json"
    second = config.gnn_root / "second.json"
    first.write_text("old-first", encoding="utf-8")
    second.write_text("old-second", encoding="utf-8")
    staged_first = config.gnn_root / ".stage-first.json"
    staged_first.write_text("new-first", encoding="utf-8")
    missing_staged_second = config.gnn_root / ".missing-stage-second.json"

    with pytest.raises(FileNotFoundError):
        _promote_paths(
            config,
            replacements=(
                (staged_first, first),
                (missing_staged_second, second),
            ),
        )

    assert first.read_text(encoding="utf-8") == "old-first"
    assert second.read_text(encoding="utf-8") == "old-second"


def test_prepare_rejects_candidate_label_drift_with_matching_group_totals(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_inputs(config)
    config.patient_condition_medication_path.unlink()
    write_parquet_rows(
        config.patient_condition_medication_path,
        (
            "source",
            "split",
            "patient_uid",
            "ranking_group_id",
            "index_condition_token",
            "candidate_medication_token",
            "candidate_rank",
            "label_prescribed",
        ),
        (
            (
                "mimiciv",
                "train",
                "synthetic-a",
                "group-train",
                "condition:train",
                "rxnorm:1",
                1,
                False,
            ),
            (
                "mimiciv",
                "train",
                "synthetic-a",
                "group-train",
                "condition:train",
                "rxnorm:2",
                2,
                True,
            ),
            (
                "mimiciv",
                "train",
                "synthetic-a",
                "group-train",
                "condition:train",
                "rxnorm:3",
                3,
                False,
            ),
            (
                "mimiciv",
                "train",
                "synthetic-b",
                "group-train-zero",
                "condition:zero",
                "rxnorm:zero",
                1,
                False,
            ),
            (
                "mimiciv",
                "validation",
                "synthetic-c",
                "group-validation",
                "condition:validation-only",
                "rxnorm:validation-only",
                1,
                False,
            ),
            (
                "mimiciv",
                "test",
                "synthetic-d",
                "group-test",
                "condition:train",
                "rxnorm:1",
                1,
                True,
            ),
        ),
    )

    manifest = prepare_gnn_caches(config)

    assert manifest["status"] == "failed"
    assert (
        "candidate identities or observed labels are inconsistent" in manifest["reason"]
    )
    assert not config.cache_manifest_path.exists()


def test_graph_cache_manifest_binds_upstream_control_hashes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_inputs(config)
    config.contract_lock_path.parent.mkdir(parents=True, exist_ok=True)
    config.contract_lock_path.write_text(
        '{"synthetic": "contract"}\n', encoding="utf-8"
    )
    config.subgraphs_manifest_path.write_text(
        '{"synthetic": "subgraphs"}\n',
        encoding="utf-8",
    )

    assert prepare_gnn_caches(config)["components"]["graph_cache"] == "completed"

    cache_manifest = json.loads(config.cache_manifest_path.read_text(encoding="utf-8"))
    assert cache_manifest["upstream_provenance"] == {
        "patient_subgraphs_manifest_sha256": sha256_file(
            config.subgraphs_manifest_path
        ),
        "training_contract_lock_sha256": sha256_file(config.contract_lock_path),
    }


def test_transformer_cache_status_rejects_stub_and_validates_rows_and_hashes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    for path in (
        config.neural_checkpoint_path,
        config.neural_feature_layout_path,
        config.neural_calibration_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic-{path.name}\n", encoding="utf-8")
    config.transformer_cache_manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    config.transformer_cache_manifest_path.write_text(
        '{"status": "completed"}\n',
        encoding="utf-8",
    )
    assert _transformer_cache_status(config) == "pending"

    context_path = (
        config.frozen_transformer_cache_root
        / TRANSFORMER_CONTEXT_TABLE
        / "part_0.parquet"
    )
    context_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(database=":memory:") as connection:
        connection.execute(
            f"""
COPY (
    SELECT
        'mimiciv' AS source,
        'train' AS split,
        'synthetic-group' AS ranking_group_id,
        0 AS patient_fold_id,
        list_transform(
            range({config.architecture.transformer_context_dim}),
            value -> CAST(value AS FLOAT)
        ) AS transformer_context
)
TO {sql_string(context_path)}
(FORMAT PARQUET)
"""
        )
    logit_path = (
        config.frozen_transformer_cache_root
        / TRANSFORMER_LOGIT_TABLE
        / "part_0.parquet"
    )
    write_parquet_rows(
        logit_path,
        (
            "source",
            "split",
            "ranking_group_id",
            "index_condition_token",
            "candidate_medication_token",
            "candidate_rank",
            "frozen_transformer_logit",
        ),
        (
            (
                "mimiciv",
                "train",
                "synthetic-group",
                "condition:synthetic",
                "rxnorm:synthetic",
                1,
                0.5,
            ),
        ),
    )
    write_parquet_rows(
        (
            config.shards_root
            / "groups"
            / "split=train"
            / "shard_id=0"
            / "part_0.parquet"
        ),
        (
            "source",
            "split",
            "ranking_group_id",
            "patient_fold_id",
            "shard_id",
        ),
        (("mimiciv", "train", "synthetic-group", 0, 0),),
    )
    write_parquet_rows(
        (
            config.shards_root
            / "candidates"
            / "split=train"
            / "shard_id=0"
            / "part_0.parquet"
        ),
        (
            "source",
            "split",
            "ranking_group_id",
            "index_condition_token",
            "candidate_medication_token",
            "candidate_rank",
            "shard_id",
        ),
        (
            (
                "mimiciv",
                "train",
                "synthetic-group",
                "condition:synthetic",
                "rxnorm:synthetic",
                1,
                0,
            ),
        ),
    )
    hashes = {
        "checkpoint": sha256_file(config.neural_checkpoint_path),
        "feature_layout": sha256_file(config.neural_feature_layout_path),
        "calibration": sha256_file(config.neural_calibration_path),
    }
    cache_hashes = {
        path.relative_to(config.frozen_transformer_cache_root).as_posix(): (
            sha256_file(path)
        )
        for path in sorted((context_path, logit_path))
    }
    config.transformer_cache_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": FROZEN_TRANSFORMER_CACHE_SCHEMA_VERSION,
                "artifact_lock_version": (
                    FROZEN_TRANSFORMER_CACHE_ARTIFACT_LOCK_VERSION
                ),
                "status": "completed",
                "scope": FULL_TRAIN_REFIT_SCOPE,
                "selection_eligible": False,
                "shard_count": config.shard_count,
                "transformer_context_dim": (
                    config.architecture.transformer_context_dim
                ),
                "table_row_counts": {
                    TRANSFORMER_CONTEXT_TABLE: 1,
                    TRANSFORMER_LOGIT_TABLE: 1,
                },
                "cached_splits": [],
                "artifact_hashes": cache_hashes,
                "artifact_tree_digest": artifact_tree_digest(cache_hashes),
                "frozen_transformer_hashes": hashes,
                "upstream_provenance": {
                    "patient_subgraphs_manifest_sha256": None,
                    "training_contract_lock_sha256": None,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _transformer_cache_status(config) == "completed"

    context_path.unlink()
    with duckdb.connect(database=":memory:") as connection:
        connection.execute(
            f"""
COPY (
    SELECT
        'mimiciv' AS source,
        'train' AS split,
        'synthetic-group' AS ranking_group_id,
        0 AS patient_fold_id,
        list_transform(
            range({config.architecture.transformer_context_dim}),
            value -> CASE
                WHEN value = 0 THEN NULL::FLOAT
                ELSE CAST(value AS FLOAT)
            END
        ) AS transformer_context
)
TO {sql_string(context_path)}
(FORMAT PARQUET)
"""
        )
    assert _transformer_cache_status(config) == "pending"

    context_path.unlink()
    with duckdb.connect(database=":memory:") as connection:
        connection.execute(
            f"""
COPY (
    SELECT
        'mimiciv' AS source,
        'train' AS split,
        'synthetic-group' AS ranking_group_id,
        0 AS patient_fold_id,
        list_transform(
            range({config.architecture.transformer_context_dim}),
            value -> CAST(value AS FLOAT)
        ) AS transformer_context
)
TO {sql_string(context_path)}
(FORMAT PARQUET)
"""
        )
    assert _transformer_cache_status(config) == "completed"

    logit_path.unlink()
    write_parquet_rows(
        logit_path,
        (
            "source",
            "split",
            "ranking_group_id",
            "index_condition_token",
            "candidate_medication_token",
            "candidate_rank",
            "frozen_transformer_logit",
        ),
        (
            (
                "mimiciv",
                "train",
                "unrelated-group",
                "condition:synthetic",
                "rxnorm:synthetic",
                1,
                0.5,
            ),
        ),
    )
    assert _transformer_cache_status(config) == "pending"

    logit_path.unlink()
    write_parquet_rows(
        logit_path,
        (
            "source",
            "split",
            "ranking_group_id",
            "index_condition_token",
            "candidate_medication_token",
            "candidate_rank",
            "frozen_transformer_logit",
        ),
        (
            (
                "mimiciv",
                "train",
                "synthetic-group",
                "condition:synthetic",
                "rxnorm:synthetic",
                1,
                math.nan,
            ),
        ),
    )
    assert _transformer_cache_status(config) == "pending"
