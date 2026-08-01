"""Synthetic tests for graph-shard validation and disjoint collation."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import pipeline.gnn_training.dataset as dataset_module
from pipeline.gnn_training.config import GNNArchitecture, GNNTrainingConfig
from pipeline.gnn_training.dataset import (
    GNNFeatureLayoutSpec,
    build_shard_examples,
    collate_examples,
    iter_batches,
    table_shard_directory,
)
from pipeline.gnn_training.graph_encode import (
    FORWARD_RELATION_TYPES,
    NODE_ROLE_TO_INDEX,
    NODE_ROLE_VOCABULARY,
    NODE_TYPE_TO_INDEX,
    NODE_TYPE_VOCABULARY,
    RELATION_TO_INDEX,
    RELATION_TYPES,
    SELF_LOOP_RELATION,
    UNK_INDEX,
)
from tests.milestone6_helpers import write_parquet_rows


def _spec() -> GNNFeatureLayoutSpec:
    return GNNFeatureLayoutSpec(
        schema_version="synthetic-gnn-layout",
        concept_vocab_size=32,
        node_type_vocabulary=tuple(NODE_TYPE_VOCABULARY),
        node_role_vocabulary=tuple(NODE_ROLE_VOCABULARY),
        relation_vocabulary=tuple(RELATION_TYPES),
        shard_count=2,
    )


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = pd.DataFrame(
        (
            {
                "source": "mimiciv",
                "split": "train",
                "ranking_group_id": "group-a",
                "patient_fold_id": 0,
                "node_count": 3,
                "candidate_count": 1,
            },
            {
                "source": "mimiciv",
                "split": "train",
                "ranking_group_id": "group-b",
                "patient_fold_id": 2,
                "node_count": 3,
                "candidate_count": 2,
            },
        )
    )
    nodes = pd.DataFrame(
        (
            # Arbitrary, non-contiguous input indexes for group A.
            {
                "source": "mimiciv",
                "split": "train",
                "ranking_group_id": "group-a",
                "node_index": 10,
                "node_concept_index": 2,
                "node_type_index": NODE_TYPE_TO_INDEX["condition"],
                "node_role_index": NODE_ROLE_TO_INDEX["query_condition"],
                "observed_predecision": False,
                "cold_start": False,
            },
            {
                "source": "mimiciv",
                "split": "train",
                "ranking_group_id": "group-a",
                "node_index": 30,
                "node_concept_index": 3,
                "node_type_index": NODE_TYPE_TO_INDEX["medication"],
                "node_role_index": NODE_ROLE_TO_INDEX["candidate_medication"],
                "observed_predecision": False,
                "cold_start": False,
            },
            {
                "source": "mimiciv",
                "split": "train",
                "ranking_group_id": "group-a",
                "node_index": 90,
                "node_concept_index": 4,
                "node_type_index": NODE_TYPE_TO_INDEX["lab"],
                "node_role_index": NODE_ROLE_TO_INDEX["observed_context"],
                "observed_predecision": True,
                "cold_start": False,
            },
            # Group B has no observed context and one cold/OOV candidate.
            {
                "source": "mimiciv",
                "split": "train",
                "ranking_group_id": "group-b",
                "node_index": 7,
                "node_concept_index": 5,
                "node_type_index": NODE_TYPE_TO_INDEX["condition"],
                "node_role_index": NODE_ROLE_TO_INDEX["query_condition"],
                "observed_predecision": False,
                "cold_start": False,
            },
            {
                "source": "mimiciv",
                "split": "train",
                "ranking_group_id": "group-b",
                "node_index": 11,
                "node_concept_index": 6,
                "node_type_index": NODE_TYPE_TO_INDEX["medication"],
                "node_role_index": NODE_ROLE_TO_INDEX["candidate_medication"],
                "observed_predecision": False,
                "cold_start": False,
            },
            {
                "source": "mimiciv",
                "split": "train",
                "ranking_group_id": "group-b",
                "node_index": 99,
                "node_concept_index": UNK_INDEX,
                "node_type_index": NODE_TYPE_TO_INDEX["medication"],
                "node_role_index": NODE_ROLE_TO_INDEX["candidate_medication"],
                "observed_predecision": False,
                "cold_start": True,
            },
        )
    )
    edges = pd.DataFrame(
        (
            {
                "source": "mimiciv",
                "split": "train",
                "ranking_group_id": "group-a",
                "src_node_index": 10,
                "dst_node_index": 30,
                "relation_index": 0,
                "edge_weight": 1.0,
            },
        )
    )
    candidates = pd.DataFrame(
        (
            {
                "source": "mimiciv",
                "split": "train",
                "ranking_group_id": "group-a",
                "index_condition_token": "condition:a",
                "candidate_medication_token": "rxnorm:1",
                "candidate_node_index": 30,
                "candidate_rank": 1,
                "label_prescribed": True,
            },
            {
                "source": "mimiciv",
                "split": "train",
                "ranking_group_id": "group-b",
                "index_condition_token": "condition:b",
                "candidate_medication_token": "rxnorm:2",
                "candidate_node_index": 11,
                "candidate_rank": 1,
                "label_prescribed": False,
            },
            {
                "source": "mimiciv",
                "split": "train",
                "ranking_group_id": "group-b",
                "index_condition_token": "condition:b",
                "candidate_medication_token": "rxnorm:oov",
                "candidate_node_index": 99,
                "candidate_rank": 2,
                "label_prescribed": False,
            },
        )
    )
    return groups, nodes, edges, candidates


def test_build_shard_remaps_local_offsets_and_handles_empty_context_edges() -> None:
    groups, nodes, edges, candidates = _frames()

    examples = build_shard_examples(
        _spec(),
        groups=groups,
        nodes=nodes,
        edges=edges,
        candidates=candidates,
    )

    first, second = examples
    assert first.query_node_index == 0
    assert first.candidate_node_index.tolist() == [1]
    assert first.context_node_index.tolist() == [2]
    assert first.edge_index.tolist() == [[0], [1]]

    assert second.query_node_index == 0
    assert second.candidate_node_index.tolist() == [1, 2]
    assert second.context_node_index.shape == (0,)
    assert second.edge_index.shape == (2, 0)
    assert second.node_concept_index[-1] == UNK_INDEX
    assert second.cold_start_mask[-1]
    assert not second.has_positive


def test_collate_offsets_disjoint_graphs_and_pads_candidates() -> None:
    torch = pytest.importorskip("torch")
    groups, nodes, edges, candidates = _frames()
    examples = build_shard_examples(
        _spec(),
        groups=groups,
        nodes=nodes,
        edges=edges,
        candidates=candidates,
    )

    batch = collate_examples(examples)

    assert batch.num_graphs == 2
    assert batch.num_nodes == 6
    assert batch.num_edges == 1
    assert batch.graph_index.tolist() == [0, 0, 0, 1, 1, 1]
    assert batch.query_node_index.tolist() == [0, 3]
    assert batch.candidate_node_index.tolist() == [[1, 0], [4, 5]]
    assert batch.candidate_mask.tolist() == [[True, False], [True, True]]
    assert batch.context_node_index.tolist() == [[2], [0]]
    assert batch.context_mask.tolist() == [[True], [False]]
    assert batch.candidate_rank.tolist() == [[1, 0], [1, 2]]
    assert batch.labels.tolist() == [[1.0, 0.0], [0.0, 0.0]]
    assert batch.patient_fold_ids == (0, 2)
    assert batch.ranking_group_ids == ("group-a", "group-b")
    assert isinstance(batch.candidate_tokens, tuple)
    assert not isinstance(batch.ranking_group_ids, torch.Tensor)
    assert not isinstance(batch.patient_fold_ids, torch.Tensor)

    moved = batch.to("cpu")
    assert moved.query_node_index.device.type == "cpu"
    assert moved.ranking_group_ids == batch.ranking_group_ids

    from pipeline.gnn_training.model import build_model

    model = build_model(
        _spec(),
        GNNArchitecture(
            concept_embedding_dim=8,
            node_type_embedding_dim=4,
            node_role_embedding_dim=4,
            hidden_dim=8,
            relation_layers=1,
            dropout=0.0,
            scorer_hidden_dim=8,
        ),
    )
    output = model.forward_batch(batch)
    assert output.logits.shape == (2, 2)
    assert torch.isneginf(output.logits[0, 1])


@pytest.mark.parametrize("dangling_table", ("edges", "candidates"))
def test_build_shard_rejects_dangling_references(dangling_table: str) -> None:
    groups, nodes, edges, candidates = _frames()
    if dangling_table == "edges":
        edges.loc[0, "dst_node_index"] = 999
    else:
        candidates.loc[0, "candidate_node_index"] = 999

    with pytest.raises(ValueError, match="dangling"):
        build_shard_examples(
            _spec(),
            groups=groups.iloc[[0]],
            nodes=nodes[nodes["ranking_group_id"] == "group-a"],
            edges=edges[edges["ranking_group_id"] == "group-a"],
            candidates=candidates[candidates["ranking_group_id"] == "group-a"],
        )


def test_build_shard_requires_one_query_and_complete_candidates() -> None:
    groups, nodes, edges, candidates = _frames()
    group_a_nodes = nodes[nodes["ranking_group_id"] == "group-a"].copy()
    group_a_nodes.loc[group_a_nodes["node_index"] == 90, "node_role_index"] = (
        NODE_ROLE_TO_INDEX["query_condition"]
    )

    with pytest.raises(ValueError, match="exactly one query"):
        build_shard_examples(
            _spec(),
            groups=groups.iloc[[0]],
            nodes=group_a_nodes,
            edges=edges,
            candidates=candidates[candidates["ranking_group_id"] == "group-a"],
        )

    with pytest.raises(ValueError, match="candidate set is incomplete"):
        build_shard_examples(
            _spec(),
            groups=groups.iloc[[1]],
            nodes=nodes[nodes["ranking_group_id"] == "group-b"],
            edges=edges.iloc[0:0],
            candidates=candidates[candidates["ranking_group_id"] == "group-b"].iloc[
                [0]
            ],
        )


def test_build_shard_rejects_compensating_per_group_edge_count_mismatch() -> None:
    groups, nodes, _edges, candidates = _frames()
    groups = groups.assign(expanded_edge_count=(5, 3))
    log_support = math.log(2.0)

    def edge(
        group_id: str,
        src: int,
        dst: int,
        relation_index: int,
    ) -> dict[str, object]:
        return {
            "source": "mimiciv",
            "split": "train",
            "ranking_group_id": group_id,
            "src_node_index": src,
            "dst_node_index": dst,
            "relation_index": relation_index,
            "edge_log_support": log_support,
            "edge_weight": 1.0,
        }

    forward = RELATION_TO_INDEX[FORWARD_RELATION_TYPES[0]]
    reverse = RELATION_TO_INDEX[f"reverse_{FORWARD_RELATION_TYPES[0]}"]
    self_loop = RELATION_TO_INDEX[SELF_LOOP_RELATION]
    # Group A is missing its reverse edge (4 instead of 5). Group B has an
    # extra self-loop (4 instead of 3), so the shard-global total still agrees.
    edges = pd.DataFrame(
        (
            edge("group-a", 10, 30, forward),
            edge("group-a", 10, 10, self_loop),
            edge("group-a", 30, 30, self_loop),
            edge("group-a", 90, 90, self_loop),
            edge("group-b", 7, 7, self_loop),
            edge("group-b", 11, 11, self_loop),
            edge("group-b", 99, 99, self_loop),
            edge("group-b", 99, 99, self_loop),
        )
    )
    assert len(edges) == int(groups["expanded_edge_count"].sum())
    assert reverse not in set(edges["relation_index"])

    with pytest.raises(ValueError, match="expanded edge set is incomplete"):
        build_shard_examples(
            _spec(),
            groups=groups,
            nodes=nodes,
            edges=edges,
            candidates=candidates,
        )


def test_build_shard_rejects_redistributed_incoming_edge_weights() -> None:
    groups, nodes, _edges, candidates = _frames()
    group = groups.iloc[[0]].assign(expanded_edge_count=7, positive_count=1)
    group_nodes = nodes[nodes["ranking_group_id"] == "group-a"]
    group_candidates = candidates[candidates["ranking_group_id"] == "group-a"]
    forward = RELATION_TO_INDEX[FORWARD_RELATION_TYPES[0]]
    reverse = RELATION_TO_INDEX[f"reverse_{FORWARD_RELATION_TYPES[0]}"]
    self_loop = RELATION_TO_INDEX[SELF_LOOP_RELATION]
    log_one = math.log1p(1)
    log_three = math.log1p(3)

    def edge(
        src: int,
        dst: int,
        relation_index: int,
        support: float,
        weight: float,
    ) -> dict[str, object]:
        return {
            "source": "mimiciv",
            "split": "train",
            "ranking_group_id": "group-a",
            "src_node_index": src,
            "dst_node_index": dst,
            "relation_index": relation_index,
            "edge_log_support": support,
            "edge_weight": weight,
        }

    edges = pd.DataFrame(
        (
            # These two incoming forward weights sum to one but should be 1/3
            # and 2/3 after normalizing their log-support values.
            edge(10, 30, forward, log_one, 0.5),
            edge(90, 30, forward, log_three, 0.5),
            edge(30, 10, reverse, log_one, 1.0),
            edge(30, 90, reverse, log_three, 1.0),
            edge(10, 10, self_loop, log_one, 1.0),
            edge(30, 30, self_loop, log_one, 1.0),
            edge(90, 90, self_loop, log_one, 1.0),
        )
    )

    with pytest.raises(ValueError, match="incoming log-support normalization"):
        build_shard_examples(
            _spec(),
            groups=group,
            nodes=group_nodes,
            edges=edges,
            candidates=group_candidates,
        )


def test_build_shard_retains_zero_positive_train_group_for_evaluation() -> None:
    groups, nodes, edges, candidates = _frames()
    groups = groups.assign(positive_count=(1, 0))

    examples = build_shard_examples(
        _spec(),
        groups=groups,
        nodes=nodes,
        edges=edges,
        candidates=candidates,
    )

    assert [example.has_positive for example in examples] == [True, False]


def test_build_shard_order_is_independent_of_physical_row_order() -> None:
    groups, nodes, edges, candidates = _frames()
    extra_edge = edges.iloc[[0]].copy()
    extra_edge["src_node_index"] = 90
    extra_edge["relation_index"] = 1
    edges = pd.concat((edges, extra_edge), ignore_index=True)

    baseline = build_shard_examples(
        _spec(),
        groups=groups,
        nodes=nodes,
        edges=edges,
        candidates=candidates,
    )
    permuted = build_shard_examples(
        _spec(),
        groups=groups.iloc[::-1],
        nodes=nodes.iloc[::-1],
        edges=edges.iloc[::-1],
        candidates=candidates.iloc[::-1],
    )

    assert [example.ranking_group_id for example in baseline] == [
        example.ranking_group_id for example in permuted
    ]
    for expected, actual in zip(baseline, permuted, strict=True):
        assert expected.edge_index.tolist() == actual.edge_index.tolist()
        assert expected.edge_type.tolist() == actual.edge_type.tolist()
        assert expected.candidate_tokens == actual.candidate_tokens


def _write_loader_shard(
    config: GNNTrainingConfig,
    *,
    shard_index: int,
    group_id: str,
) -> None:
    rows = {
        "groups": (
            (
                "source",
                "split",
                "ranking_group_id",
                "patient_fold_id",
                "node_count",
                "candidate_count",
            ),
            (("mimiciv", "train", group_id, shard_index, 2, 1),),
        ),
        "nodes": (
            (
                "source",
                "split",
                "ranking_group_id",
                "node_index",
                "node_concept_index",
                "node_type_index",
                "node_role_index",
                "observed_predecision",
                "cold_start",
            ),
            (
                (
                    "mimiciv",
                    "train",
                    group_id,
                    4,
                    2,
                    NODE_TYPE_TO_INDEX["condition"],
                    NODE_ROLE_TO_INDEX["query_condition"],
                    False,
                    False,
                ),
                (
                    "mimiciv",
                    "train",
                    group_id,
                    8,
                    3,
                    NODE_TYPE_TO_INDEX["medication"],
                    NODE_ROLE_TO_INDEX["candidate_medication"],
                    False,
                    False,
                ),
            ),
        ),
        "edges": (
            (
                "source",
                "split",
                "ranking_group_id",
                "src_node_index",
                "dst_node_index",
                "relation_index",
                "edge_weight",
            ),
            (),
        ),
        "candidates": (
            (
                "source",
                "split",
                "ranking_group_id",
                "index_condition_token",
                "candidate_medication_token",
                "candidate_node_index",
                "candidate_rank",
                "label_prescribed",
            ),
            (
                (
                    "mimiciv",
                    "train",
                    group_id,
                    "condition:synthetic",
                    "rxnorm:synthetic",
                    8,
                    1,
                    True,
                ),
            ),
        ),
    }
    for table_name, (columns, table_rows) in rows.items():
        directory = table_shard_directory(
            config,
            table_name=table_name,
            split="train",
            shard_index=shard_index,
        )
        write_parquet_rows(
            directory / "part_0.parquet",
            columns,
            table_rows,
        )


def test_batch_iterator_reads_only_one_physical_shard_before_yield(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        GNNTrainingConfig(),
        gnn_root=tmp_path / "gnn",
        shard_count=2,
    )
    _write_loader_shard(config, shard_index=0, group_id="group-0")
    _write_loader_shard(config, shard_index=1, group_id="group-1")
    original = dataset_module.pd.read_parquet
    reads: list[Path] = []

    def tracked_read(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        reads.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(dataset_module.pd, "read_parquet", tracked_read)
    batches = iter_batches(
        config,
        _spec(),
        split="train",
        batch_groups=1,
        shuffle=False,
        seed=7,
    )

    first = next(batches)

    assert first.ranking_group_ids == ("group-0",)
    assert len(reads) == 4
    assert all("shard_id=0" in str(path) for path in reads)
    assert not any("shard_id=1" in str(path) for path in reads)


def test_table_shard_reader_combines_multiple_physical_files(
    tmp_path: Path,
) -> None:
    config = replace(
        GNNTrainingConfig(),
        gnn_root=tmp_path / "gnn",
        shard_count=2,
    )
    directory = table_shard_directory(
        config,
        table_name="edges",
        split="train",
        shard_index=0,
    )
    columns = (
        "source",
        "ranking_group_id",
        "src_node_index",
        "dst_node_index",
        "relation_index",
        "edge_weight",
    )
    write_parquet_rows(
        directory / "part_1.parquet",
        columns,
        (("mimiciv", "group-b", 7, 11, 1, 0.5),),
    )
    write_parquet_rows(
        directory / "part_0.parquet",
        columns,
        (("mimiciv", "group-a", 10, 30, 0, 1.0),),
    )

    frame = dataset_module._read_table_shard(
        config,
        table_name="edges",
        split="train",
        shard_index=0,
    )

    assert frame["ranking_group_id"].tolist() == ["group-a", "group-b"]
    assert frame["split"].tolist() == ["train", "train"]
    assert frame["shard_id"].tolist() == [0, 0]
