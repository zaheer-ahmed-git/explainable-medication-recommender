"""Reassemble one compact graph shard at a time and collate packed batches.

The prepared cache contains four partitioned tables (groups, nodes, expanded
edges, and candidates).  A loader visits one logical split/shard partition at a
time, validates every complete ranking group, remaps arbitrary source-local
``node_index`` values to contiguous offsets, and releases that shard before
opening the next one.

PyTorch is imported only by :func:`collate_examples`.  Layout parsing and graph
validation therefore remain available for lightweight login-node tests without
the optional neural dependency group.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pipeline.gnn_training.config import (
    CROSS_FIT_SELECTION_SCOPE,
    FULL_TRAIN_REFIT_SCOPE,
    GNNTrainingConfig,
)
from pipeline.gnn_training.data import (
    CACHE_TABLES,
    CANDIDATE_TABLE,
    EDGE_TABLE,
    GROUP_TABLE,
    NODE_TABLE,
)
from pipeline.gnn_training.graph_encode import (
    FORWARD_RELATION_TYPES,
    NODE_CONTINUOUS_FEATURES,
    NODE_ROLE_TO_INDEX,
    NODE_ROLE_VOCABULARY,
    NODE_TYPE_TO_INDEX,
    NODE_TYPE_VOCABULARY,
    PAD_INDEX,
    RELATION_TO_INDEX,
    RELATION_TYPES,
    SELF_LOOP_RELATION,
    TIME_BIN_COUNT,
    UNK_INDEX,
)

if TYPE_CHECKING:  # pragma: no cover - keeps import torch lazy
    import torch


@dataclass(frozen=True)
class GNNFeatureLayoutSpec:
    """In-memory view of the versioned graph feature-layout contract."""

    schema_version: str
    concept_vocab_size: int
    node_type_vocabulary: tuple[str, ...]
    node_role_vocabulary: tuple[str, ...]
    relation_vocabulary: tuple[str, ...]
    node_continuous_features: tuple[str, ...] = NODE_CONTINUOUS_FEATURES
    time_bin_count: int = TIME_BIN_COUNT
    pad_index: int = PAD_INDEX
    unk_index: int = UNK_INDEX
    scope: str = "full_train_refit_only"
    selection_eligible: bool = False
    shard_count: int = 1
    held_out_fold_index: int | None = None

    @property
    def node_type_vocab_size(self) -> int:
        return len(self.node_type_vocabulary)

    @property
    def node_role_vocab_size(self) -> int:
        return len(self.node_role_vocabulary)

    @property
    def relation_count(self) -> int:
        return len(self.relation_vocabulary)

    @property
    def query_role_index(self) -> int:
        return self.node_role_vocabulary.index("stay_query")

    @property
    def condition_role_index(self) -> int:
        return self.node_role_vocabulary.index("query_condition")

    @property
    def candidate_role_index(self) -> int:
        return self.node_role_vocabulary.index("candidate_medication")

    @property
    def observed_role_index(self) -> int:
        return self.node_role_vocabulary.index("observed_context")

    @classmethod
    def from_json(cls, path: Path) -> GNNFeatureLayoutSpec:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        expected_mappings = {
            "node_type_to_index": dict(NODE_TYPE_TO_INDEX),
            "node_role_to_index": dict(NODE_ROLE_TO_INDEX),
            "relation_to_index": dict(RELATION_TO_INDEX),
        }
        for name, expected in expected_mappings.items():
            raw = payload.get(name)
            if raw is not None and raw != expected:
                raise ValueError(f"graph layout {name} does not match contract")
        node_types = tuple(payload.get("node_type_vocabulary", NODE_TYPE_VOCABULARY))
        node_roles = tuple(payload.get("node_role_vocabulary", NODE_ROLE_VOCABULARY))
        relations = tuple(payload.get("relation_vocabulary", RELATION_TYPES))
        spec = cls(
            schema_version=str(payload["schema_version"]),
            concept_vocab_size=int(payload["concept_vocab_size"]),
            node_type_vocabulary=node_types,
            node_role_vocabulary=node_roles,
            relation_vocabulary=relations,
            node_continuous_features=tuple(payload.get("node_continuous_features", ())),
            time_bin_count=int(payload.get("time_bin_count", 0)),
            pad_index=int(payload.get("pad_index", PAD_INDEX)),
            unk_index=int(payload.get("unk_index", UNK_INDEX)),
            scope=str(payload.get("scope", "unknown")),
            selection_eligible=bool(payload.get("selection_eligible", False)),
            shard_count=int(payload.get("shard_count", 1)),
            held_out_fold_index=(
                int(payload["held_out_fold_index"])
                if payload.get("held_out_fold_index") is not None
                else None
            ),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.concept_vocab_size <= self.unk_index:
            raise ValueError("concept vocabulary must include PAD and UNK")
        if self.pad_index != PAD_INDEX or self.unk_index != UNK_INDEX:
            raise ValueError("graph layout must reserve PAD=0 and UNK=1")
        full_refit = (
            self.scope == FULL_TRAIN_REFIT_SCOPE
            and not self.selection_eligible
            and self.held_out_fold_index is None
        )
        crossfit = (
            self.scope == CROSS_FIT_SELECTION_SCOPE
            and self.selection_eligible
            and self.held_out_fold_index is not None
            and self.held_out_fold_index >= 0
        )
        if not (full_refit or crossfit):
            raise ValueError(
                "graph layout must be either full-train-refit-only or a "
                "held-out-fold-specific selection cache"
            )
        if self.node_type_vocabulary != tuple(NODE_TYPE_VOCABULARY):
            raise ValueError(
                "graph layout node-type vocabulary does not match contract"
            )
        if self.node_role_vocabulary != tuple(NODE_ROLE_VOCABULARY):
            raise ValueError(
                "graph layout node-role vocabulary does not match contract"
            )
        if self.relation_vocabulary != tuple(RELATION_TYPES):
            raise ValueError("graph layout relation vocabulary does not match contract")
        if self.node_continuous_features != tuple(NODE_CONTINUOUS_FEATURES):
            raise ValueError(
                "graph layout continuous node features do not match contract"
            )
        if self.time_bin_count != TIME_BIN_COUNT:
            raise ValueError("graph layout time-bin count does not match contract")


# Compatibility aliases for model/test callers that use a shorter name.
GraphFeatureLayoutSpec = GNNFeatureLayoutSpec
FeatureLayoutSpec = GNNFeatureLayoutSpec


@dataclass
class GNNExample:
    """One validated ranking-group graph using contiguous local node offsets."""

    node_concept_index: np.ndarray  # (N,) int64
    node_type_index: np.ndarray  # (N,) int64
    node_role_index: np.ndarray  # (N,) int64
    observed_mask: np.ndarray  # (N,) bool
    cold_start_mask: np.ndarray  # (N,) bool
    node_continuous: np.ndarray  # (N, F) float32
    node_time_bin_index: np.ndarray  # (N,) int64
    edge_index: np.ndarray  # (2, E) int64
    edge_type: np.ndarray  # (E,) int64
    edge_weight: np.ndarray  # (E,) float32
    query_node_index: int
    candidate_node_index: np.ndarray  # (C,) int64
    context_node_index: np.ndarray  # (K,) int64
    candidate_rank: np.ndarray  # (C,) int64
    labels: np.ndarray  # (C,) float32
    patient_fold_id: int
    source: str
    split: str
    ranking_group_id: str
    index_condition_token: str
    candidate_tokens: tuple[str, ...]
    shard_id: int = 0

    @property
    def num_nodes(self) -> int:
        return int(self.node_concept_index.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_type.shape[0])

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_node_index.shape[0])

    @property
    def num_context_nodes(self) -> int:
        return int(self.context_node_index.shape[0])

    @property
    def has_positive(self) -> bool:
        return bool(self.labels.sum() > 0)


GraphExample = GNNExample


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    *,
    table_name: str,
) -> None:
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(
            f"{table_name} shard is missing columns: " + ", ".join(missing)
        )


def _group_frames(
    frame: pd.DataFrame,
) -> dict[tuple[Any, Any, Any], pd.DataFrame]:
    if frame.empty:
        return {}
    return {
        key: group
        for key, group in frame.groupby(
            ["source", "split", "ranking_group_id"],
            sort=False,
            dropna=False,
        )
    }


def _empty_edges() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((2, 0), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.float32),
    )


def _validate_prepared_edges(
    spec: GNNFeatureLayoutSpec,
    group_row: pd.Series,
    *,
    source_node_indexes: np.ndarray,
    edges: pd.DataFrame,
) -> None:
    """Validate expanded-edge completeness and normalization per prepared group."""

    if "expanded_edge_count" not in group_row.index or pd.isna(
        group_row["expanded_edge_count"]
    ):
        return
    expected_count = int(group_row["expanded_edge_count"])
    if expected_count < 0 or len(edges) != expected_count:
        raise ValueError("ranking group expanded edge set is incomplete")
    if "edge_log_support" not in edges.columns:
        raise ValueError("prepared edge shard is missing transformed support")

    support = edges["edge_log_support"].to_numpy(dtype=np.float64)
    if not np.isfinite(support).all() or (support < 0).any():
        raise ValueError("ranking group has invalid transformed edge support")
    source = edges["src_node_index"].to_numpy(dtype=np.int64)
    destination = edges["dst_node_index"].to_numpy(dtype=np.int64)
    relation = edges["relation_index"].to_numpy(dtype=np.int64)
    weights = edges["edge_weight"].to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("ranking group has invalid edge weights")

    self_relation = spec.relation_vocabulary.index(SELF_LOOP_RELATION)
    self_mask = relation == self_relation
    if int(self_mask.sum()) != len(source_node_indexes):
        raise ValueError("ranking group self-loop set is incomplete")
    if not np.array_equal(source[self_mask], destination[self_mask]):
        raise ValueError("ranking group has invalid self-loop endpoints")
    if Counter(int(value) for value in source[self_mask]) != Counter(
        int(value) for value in source_node_indexes
    ):
        raise ValueError("ranking group self-loop coverage is invalid")
    if not np.allclose(support[self_mask], math.log(2.0), rtol=1e-9, atol=1e-12):
        raise ValueError("ranking group self-loop support is invalid")

    for forward_name in FORWARD_RELATION_TYPES:
        forward_index = spec.relation_vocabulary.index(forward_name)
        reverse_index = spec.relation_vocabulary.index(f"reverse_{forward_name}")
        forward = Counter(
            (int(src), int(dst), float(value))
            for src, dst, value in zip(
                source[relation == forward_index],
                destination[relation == forward_index],
                support[relation == forward_index],
                strict=True,
            )
        )
        reverse = Counter(
            (int(dst), int(src), float(value))
            for src, dst, value in zip(
                source[relation == reverse_index],
                destination[relation == reverse_index],
                support[relation == reverse_index],
                strict=True,
            )
        )
        if forward != reverse:
            raise ValueError("ranking group forward/reverse edge pairs are incomplete")

    incoming_groups: dict[tuple[int, int], list[int]] = {}
    for edge_offset, (relation_index, destination_index) in enumerate(
        zip(relation, destination, strict=True)
    ):
        key = (int(relation_index), int(destination_index))
        incoming_groups.setdefault(key, []).append(edge_offset)
    for offsets in incoming_groups.values():
        incoming_support = float(support[offsets].sum())
        if incoming_support > 0:
            expected_weights = support[offsets] / incoming_support
        else:
            expected_weights = np.full(
                len(offsets),
                1.0 / len(offsets),
                dtype=np.float64,
            )
        if not np.allclose(
            weights[offsets],
            expected_weights,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(
                "ranking group edge weights do not match incoming log-support "
                "normalization"
            )


def _build_example(
    spec: GNNFeatureLayoutSpec,
    group_row: pd.Series,
    *,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    candidates: pd.DataFrame,
) -> GNNExample:
    group_id = str(group_row["ranking_group_id"])
    if nodes.empty:
        raise ValueError("ranking group has no nodes")
    if candidates.empty:
        raise ValueError("ranking group has no candidates")

    ordered_nodes = nodes.sort_values("node_index", kind="mergesort")
    if (
        ordered_nodes[
            [
                "node_index",
                "node_concept_index",
                "node_type_index",
                "node_role_index",
                "observed_predecision",
                "cold_start",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("ranking group has null node fields")
    source_indexes = ordered_nodes["node_index"].to_numpy(dtype=np.int64)
    if len(np.unique(source_indexes)) != len(source_indexes):
        raise ValueError("ranking group has duplicate node indexes")
    remap = {int(value): offset for offset, value in enumerate(source_indexes)}

    concept = ordered_nodes["node_concept_index"].to_numpy(dtype=np.int64)
    node_type = ordered_nodes["node_type_index"].to_numpy(dtype=np.int64)
    node_role = ordered_nodes["node_role_index"].to_numpy(dtype=np.int64)
    observed = ordered_nodes["observed_predecision"].to_numpy(dtype=bool)
    cold = ordered_nodes["cold_start"].to_numpy(dtype=bool)
    if (concept < 0).any() or (concept >= spec.concept_vocab_size).any():
        raise ValueError("ranking group has invalid concept indexes")
    if (node_type < 0).any() or (node_type >= spec.node_type_vocab_size).any():
        raise ValueError("ranking group has invalid node type indexes")
    if (node_role < 0).any() or (node_role >= spec.node_role_vocab_size).any():
        raise ValueError("ranking group has invalid node role indexes")

    condition_offsets = np.flatnonzero(node_role == spec.condition_role_index)
    if condition_offsets.shape[0] != 1:
        raise ValueError("ranking group must contain exactly one condition node")
    condition_index = int(condition_offsets[0])

    # Context pooling uses the explicit temporal-observation flag.  A graph
    # without context remains valid and receives an empty index vector.
    context_index = np.flatnonzero(observed).astype(np.int64, copy=False)

    if all(name in ordered_nodes.columns for name in NODE_CONTINUOUS_FEATURES):
        node_continuous = ordered_nodes[list(NODE_CONTINUOUS_FEATURES)].to_numpy(
            dtype=np.float32
        )
    else:
        node_continuous = np.zeros(
            (len(ordered_nodes), len(NODE_CONTINUOUS_FEATURES)),
            dtype=np.float32,
        )
    if not np.isfinite(node_continuous).all():
        raise ValueError("ranking group has non-finite continuous node features")
    if "time_bin_index" in ordered_nodes.columns:
        node_time_bin = ordered_nodes["time_bin_index"].to_numpy(dtype=np.int64)
    else:
        node_time_bin = np.zeros(len(ordered_nodes), dtype=np.int64)
    if (node_time_bin < 0).any() or (node_time_bin >= TIME_BIN_COUNT).any():
        raise ValueError("ranking group has invalid node time bins")

    ordered_candidates = candidates.sort_values(
        ["candidate_rank", "candidate_medication_token"],
        kind="mergesort",
    )
    declared_candidates = int(group_row["candidate_count"])
    if len(ordered_candidates) != declared_candidates:
        raise ValueError("ranking group candidate set is incomplete")
    if int((node_role == spec.candidate_role_index).sum()) != declared_candidates:
        raise ValueError("ranking group candidate nodes are incomplete")
    if (
        ordered_candidates[
            [
                "index_condition_token",
                "candidate_medication_token",
                "candidate_node_index",
                "candidate_rank",
                "label_prescribed",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("ranking group has null candidate fields")
    if ordered_candidates["index_condition_token"].nunique(dropna=False) != 1:
        raise ValueError("ranking group has inconsistent index conditions")
    if ordered_candidates["candidate_medication_token"].duplicated().any():
        raise ValueError("ranking group has duplicate candidate medications")
    if (
        ordered_candidates["candidate_rank"].isna().any()
        or (ordered_candidates["candidate_rank"] < 0).any()
        or ordered_candidates["candidate_rank"].duplicated().any()
    ):
        raise ValueError("ranking group has invalid candidate ranks")
    source_candidate_indexes = ordered_candidates["candidate_node_index"].to_numpy(
        dtype=np.int64
    )
    dangling_candidates = sorted(
        {int(value) for value in source_candidate_indexes if int(value) not in remap}
    )
    if dangling_candidates:
        raise ValueError("ranking group has dangling candidate node indexes")
    candidate_index = np.asarray(
        [remap[int(value)] for value in source_candidate_indexes],
        dtype=np.int64,
    )
    if len(np.unique(candidate_index)) != len(candidate_index):
        raise ValueError("ranking group has duplicate candidate nodes")
    if not np.all(node_role[candidate_index] == spec.candidate_role_index):
        raise ValueError("ranking group candidates do not reference candidate nodes")

    edge_sort_columns = (
        [
            "relation_index",
            "dst_node_index",
            "src_node_index",
            "edge_log_support",
        ]
        if "edge_log_support" in edges.columns
        else ["relation_index", "dst_node_index", "src_node_index"]
    )
    ordered_edges = edges.sort_values(edge_sort_columns, kind="mergesort")
    _validate_prepared_edges(
        spec,
        group_row,
        source_node_indexes=source_indexes,
        edges=ordered_edges,
    )
    if ordered_edges.empty:
        edge_index, edge_type, edge_weight = _empty_edges()
    else:
        source_edge_src = ordered_edges["src_node_index"].to_numpy(dtype=np.int64)
        source_edge_dst = ordered_edges["dst_node_index"].to_numpy(dtype=np.int64)
        if any(
            int(value) not in remap
            for value in np.concatenate((source_edge_src, source_edge_dst))
        ):
            raise ValueError("ranking group has dangling edge endpoints")
        edge_index = np.asarray(
            (
                [remap[int(value)] for value in source_edge_src],
                [remap[int(value)] for value in source_edge_dst],
            ),
            dtype=np.int64,
        )
        edge_type = ordered_edges["relation_index"].to_numpy(dtype=np.int64)
        edge_weight = ordered_edges["edge_weight"].to_numpy(dtype=np.float32)
        if (edge_type < 0).any() or (edge_type >= spec.relation_count).any():
            raise ValueError("ranking group has invalid relation indexes")
        if not np.isfinite(edge_weight).all() or (edge_weight < 0).any():
            raise ValueError("ranking group has invalid edge weights")

    labels = (
        ordered_candidates["label_prescribed"]
        .astype("float32")
        .to_numpy(dtype=np.float32)
    )
    if not np.isfinite(labels).all() or not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("ranking group has invalid labels")
    if "positive_count" in group_row.index and not pd.isna(group_row["positive_count"]):
        declared_positive_count = int(group_row["positive_count"])
        actual_positive_count = int(labels.sum())
        if declared_positive_count != actual_positive_count:
            raise ValueError("ranking group observed positive count is inconsistent")
    fold_id = int(group_row["patient_fold_id"])
    if fold_id < 0:
        raise ValueError("ranking group has an invalid fold")

    # P1 adds one patient-specific stay/query node at load time. This preserves
    # the immutable concept-cache rows while avoiding a duplicated stay token
    # in the train-fitted concept vocabulary. The stay is connected to the
    # index condition and every observed pre-decision context node.
    query_index = len(concept)
    concept = np.append(concept, UNK_INDEX).astype(np.int64, copy=False)
    node_type = np.append(node_type, NODE_TYPE_TO_INDEX["stay"]).astype(
        np.int64, copy=False
    )
    node_role = np.append(node_role, spec.query_role_index).astype(np.int64, copy=False)
    observed = np.append(observed, False)
    cold = np.append(cold, False)
    node_continuous = np.vstack(
        (
            node_continuous,
            np.zeros((1, len(NODE_CONTINUOUS_FEATURES)), dtype=np.float32),
        )
    )
    node_time_bin = np.append(node_time_bin, 0).astype(np.int64, copy=False)

    dynamic_sources = [query_index, condition_index, query_index]
    dynamic_destinations = [condition_index, query_index, query_index]
    dynamic_types = [
        RELATION_TO_INDEX["stay_index_condition"],
        RELATION_TO_INDEX["reverse_stay_index_condition"],
        RELATION_TO_INDEX[SELF_LOOP_RELATION],
    ]
    dynamic_weights = [1.0, 1.0, 1.0]
    if context_index.size:
        reverse_weight = 1.0 / float(context_index.size)
        for context_node in context_index.tolist():
            dynamic_sources.extend((query_index, int(context_node)))
            dynamic_destinations.extend((int(context_node), query_index))
            dynamic_types.extend(
                (
                    RELATION_TO_INDEX["stay_context_observed"],
                    RELATION_TO_INDEX["reverse_stay_context_observed"],
                )
            )
            dynamic_weights.extend((1.0, reverse_weight))
    edge_index = np.concatenate(
        (
            edge_index,
            np.asarray((dynamic_sources, dynamic_destinations), dtype=np.int64),
        ),
        axis=1,
    )
    edge_type = np.concatenate((edge_type, np.asarray(dynamic_types, dtype=np.int64)))
    edge_weight = np.concatenate(
        (edge_weight, np.asarray(dynamic_weights, dtype=np.float32))
    )

    return GNNExample(
        node_concept_index=concept,
        node_type_index=node_type,
        node_role_index=node_role,
        observed_mask=observed,
        cold_start_mask=cold,
        node_continuous=node_continuous,
        node_time_bin_index=node_time_bin,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        query_node_index=query_index,
        candidate_node_index=candidate_index,
        context_node_index=context_index,
        candidate_rank=ordered_candidates["candidate_rank"].to_numpy(dtype=np.int64),
        labels=labels,
        patient_fold_id=fold_id,
        source=str(group_row["source"]),
        split=str(group_row["split"]),
        ranking_group_id=group_id,
        index_condition_token=str(ordered_candidates["index_condition_token"].iloc[0]),
        candidate_tokens=tuple(
            str(value)
            for value in ordered_candidates["candidate_medication_token"].tolist()
        ),
        shard_id=int(group_row.get("shard_id", 0)),
    )


def build_shard_examples(
    spec: GNNFeatureLayoutSpec,
    *,
    groups: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    candidates: pd.DataFrame,
) -> list[GNNExample]:
    """Validate and rebuild all complete ranking-group graphs in one shard."""

    spec.validate()
    if groups.empty:
        if any(not frame.empty for frame in (nodes, edges, candidates)):
            raise ValueError("graph shard has rows without a group index")
        return []

    _require_columns(
        groups,
        (
            "source",
            "split",
            "ranking_group_id",
            "patient_fold_id",
            "node_count",
            "candidate_count",
        ),
        table_name=GROUP_TABLE,
    )
    _require_columns(
        nodes,
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
        table_name=NODE_TABLE,
    )
    if edges.empty and not len(edges.columns):
        edges = pd.DataFrame(
            columns=(
                "source",
                "split",
                "ranking_group_id",
                "src_node_index",
                "dst_node_index",
                "relation_index",
                "edge_weight",
            )
        )
    if "expanded_edge_count" in groups.columns:
        _require_columns(
            edges,
            ("edge_log_support",),
            table_name=EDGE_TABLE,
        )
    _require_columns(
        edges,
        (
            "source",
            "split",
            "ranking_group_id",
            "src_node_index",
            "dst_node_index",
            "relation_index",
            "edge_weight",
        ),
        table_name=EDGE_TABLE,
    )
    _require_columns(
        candidates,
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
        table_name=CANDIDATE_TABLE,
    )
    if groups.duplicated(["source", "split", "ranking_group_id"]).any():
        raise ValueError("graph shard contains duplicate group-index rows")

    node_groups = _group_frames(nodes)
    edge_groups = _group_frames(edges)
    candidate_groups = _group_frames(candidates)
    expected_keys = {
        (row.source, row.split, row.ranking_group_id)
        for row in groups.itertuples(index=False)
    }
    for table_name, lookup in (
        (NODE_TABLE, node_groups),
        (EDGE_TABLE, edge_groups),
        (CANDIDATE_TABLE, candidate_groups),
    ):
        orphans = set(lookup) - expected_keys
        if orphans:
            raise ValueError(f"{table_name} shard contains rows for unknown groups")

    ordered_groups = groups.sort_values(
        ["source", "split", "ranking_group_id"],
        kind="mergesort",
    )
    examples: list[GNNExample] = []
    for _index, row in ordered_groups.iterrows():
        key = (row["source"], row["split"], row["ranking_group_id"])
        example = _build_example(
            spec,
            row,
            nodes=node_groups.get(key, pd.DataFrame(columns=nodes.columns)),
            edges=edge_groups.get(key, pd.DataFrame(columns=edges.columns)),
            candidates=candidate_groups.get(
                key, pd.DataFrame(columns=candidates.columns)
            ),
        )
        if example.num_nodes != int(row["node_count"]) + 1:
            raise ValueError("ranking group node set is incomplete")
        examples.append(example)
    return examples


@dataclass
class GNNBatch:
    """Disjoint packed graphs, padded candidates, and restricted metadata."""

    node_concept_index: "torch.Tensor"
    node_type_index: "torch.Tensor"
    node_role_index: "torch.Tensor"
    observed_mask: "torch.Tensor"
    cold_start_mask: "torch.Tensor"
    node_continuous: "torch.Tensor"
    node_time_bin_index: "torch.Tensor"
    edge_index: "torch.Tensor"
    edge_type: "torch.Tensor"
    edge_weight: "torch.Tensor"
    graph_index: "torch.Tensor"
    query_node_index: "torch.Tensor"
    candidate_node_index: "torch.Tensor"
    candidate_mask: "torch.Tensor"
    context_node_index: "torch.Tensor"
    context_mask: "torch.Tensor"
    candidate_rank: "torch.Tensor"
    labels: "torch.Tensor"
    patient_fold_ids: tuple[int, ...]
    sources: tuple[str, ...]
    splits: tuple[str, ...]
    ranking_group_ids: tuple[str, ...]
    index_condition_tokens: tuple[str, ...]
    candidate_tokens: tuple[tuple[str, ...], ...]
    shard_ids: tuple[int, ...]

    @property
    def num_groups(self) -> int:
        return int(self.query_node_index.shape[0])

    @property
    def num_graphs(self) -> int:
        return self.num_groups

    @property
    def num_nodes(self) -> int:
        return int(self.node_concept_index.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_type.shape[0])

    @property
    def max_candidates(self) -> int:
        return int(self.candidate_mask.shape[1])

    @property
    def patient_fold_id(self) -> tuple[int, ...]:
        """Singular-name compatibility alias for tuple fold metadata."""

        return self.patient_fold_ids

    def to(self, device: "torch.device | str") -> GNNBatch:
        """Return a copy with tensor fields moved and metadata unchanged."""

        return GNNBatch(
            node_concept_index=self.node_concept_index.to(device),
            node_type_index=self.node_type_index.to(device),
            node_role_index=self.node_role_index.to(device),
            observed_mask=self.observed_mask.to(device),
            cold_start_mask=self.cold_start_mask.to(device),
            node_continuous=self.node_continuous.to(device),
            node_time_bin_index=self.node_time_bin_index.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            edge_weight=self.edge_weight.to(device),
            graph_index=self.graph_index.to(device),
            query_node_index=self.query_node_index.to(device),
            candidate_node_index=self.candidate_node_index.to(device),
            candidate_mask=self.candidate_mask.to(device),
            context_node_index=self.context_node_index.to(device),
            context_mask=self.context_mask.to(device),
            candidate_rank=self.candidate_rank.to(device),
            labels=self.labels.to(device),
            patient_fold_ids=self.patient_fold_ids,
            sources=self.sources,
            splits=self.splits,
            ranking_group_ids=self.ranking_group_ids,
            index_condition_tokens=self.index_condition_tokens,
            candidate_tokens=self.candidate_tokens,
            shard_ids=self.shard_ids,
        )


def collate_examples(
    examples: Sequence[GNNExample],
    spec: GNNFeatureLayoutSpec | None = None,
) -> GNNBatch:
    """Pack variable-size graphs and pad only candidates/context indexes."""

    import torch

    if not examples:
        raise ValueError("cannot collate an empty graph batch")
    if spec is not None:
        spec.validate()

    group_count = len(examples)
    max_candidates = max(example.num_candidates for example in examples)
    max_context = max(1, max(example.num_context_nodes for example in examples))
    total_nodes = sum(example.num_nodes for example in examples)
    total_edges = sum(example.num_edges for example in examples)

    node_concept = torch.empty((total_nodes,), dtype=torch.long)
    node_type = torch.empty((total_nodes,), dtype=torch.long)
    node_role = torch.empty((total_nodes,), dtype=torch.long)
    observed = torch.empty((total_nodes,), dtype=torch.bool)
    cold = torch.empty((total_nodes,), dtype=torch.bool)
    node_continuous = torch.empty(
        (total_nodes, len(NODE_CONTINUOUS_FEATURES)), dtype=torch.float32
    )
    node_time_bin = torch.empty((total_nodes,), dtype=torch.long)
    graph_index = torch.empty((total_nodes,), dtype=torch.long)
    edge_index = torch.empty((2, total_edges), dtype=torch.long)
    edge_type = torch.empty((total_edges,), dtype=torch.long)
    edge_weight = torch.empty((total_edges,), dtype=torch.float32)
    query_index = torch.empty((group_count,), dtype=torch.long)
    candidate_index = torch.zeros((group_count, max_candidates), dtype=torch.long)
    candidate_mask = torch.zeros((group_count, max_candidates), dtype=torch.bool)
    context_index = torch.zeros((group_count, max_context), dtype=torch.long)
    context_mask = torch.zeros((group_count, max_context), dtype=torch.bool)
    candidate_rank = torch.zeros((group_count, max_candidates), dtype=torch.long)
    labels = torch.zeros((group_count, max_candidates), dtype=torch.float32)

    node_offset = 0
    edge_offset = 0
    for graph, example in enumerate(examples):
        node_end = node_offset + example.num_nodes
        edge_end = edge_offset + example.num_edges
        node_concept[node_offset:node_end] = torch.tensor(
            example.node_concept_index, dtype=torch.long
        )
        node_type[node_offset:node_end] = torch.tensor(
            example.node_type_index, dtype=torch.long
        )
        node_role[node_offset:node_end] = torch.tensor(
            example.node_role_index, dtype=torch.long
        )
        observed[node_offset:node_end] = torch.tensor(
            example.observed_mask, dtype=torch.bool
        )
        cold[node_offset:node_end] = torch.tensor(
            example.cold_start_mask, dtype=torch.bool
        )
        node_continuous[node_offset:node_end] = torch.tensor(
            example.node_continuous, dtype=torch.float32
        )
        node_time_bin[node_offset:node_end] = torch.tensor(
            example.node_time_bin_index, dtype=torch.long
        )
        graph_index[node_offset:node_end] = graph
        if example.num_edges:
            edge_index[:, edge_offset:edge_end] = (
                torch.tensor(example.edge_index, dtype=torch.long) + node_offset
            )
            edge_type[edge_offset:edge_end] = torch.tensor(
                example.edge_type, dtype=torch.long
            )
            edge_weight[edge_offset:edge_end] = torch.tensor(
                example.edge_weight, dtype=torch.float32
            )
        query_index[graph] = node_offset + example.query_node_index
        candidate_count = example.num_candidates
        candidate_index[graph, :candidate_count] = (
            torch.tensor(example.candidate_node_index, dtype=torch.long) + node_offset
        )
        candidate_mask[graph, :candidate_count] = True
        candidate_rank[graph, :candidate_count] = torch.tensor(
            example.candidate_rank, dtype=torch.long
        )
        labels[graph, :candidate_count] = torch.tensor(
            example.labels, dtype=torch.float32
        )
        context_count = example.num_context_nodes
        if context_count:
            context_index[graph, :context_count] = (
                torch.tensor(example.context_node_index, dtype=torch.long) + node_offset
            )
            context_mask[graph, :context_count] = True
        node_offset = node_end
        edge_offset = edge_end

    return GNNBatch(
        node_concept_index=node_concept,
        node_type_index=node_type,
        node_role_index=node_role,
        observed_mask=observed,
        cold_start_mask=cold,
        node_continuous=node_continuous,
        node_time_bin_index=node_time_bin,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=edge_weight,
        graph_index=graph_index,
        query_node_index=query_index,
        candidate_node_index=candidate_index,
        candidate_mask=candidate_mask,
        context_node_index=context_index,
        context_mask=context_mask,
        candidate_rank=candidate_rank,
        labels=labels,
        patient_fold_ids=tuple(int(example.patient_fold_id) for example in examples),
        sources=tuple(example.source for example in examples),
        splits=tuple(example.split for example in examples),
        ranking_group_ids=tuple(example.ranking_group_id for example in examples),
        index_condition_tokens=tuple(
            example.index_condition_token for example in examples
        ),
        candidate_tokens=tuple(example.candidate_tokens for example in examples),
        shard_ids=tuple(int(example.shard_id) for example in examples),
    )


def table_shard_directory(
    config: GNNTrainingConfig,
    *,
    table_name: str,
    split: str,
    shard_index: int,
    shards_root: Path | None = None,
) -> Path:
    """Return one canonical physical split/shard partition directory."""

    if table_name not in CACHE_TABLES:
        raise ValueError(f"unknown graph cache table: {table_name}")
    if shard_index < 0 or shard_index >= config.shard_count:
        raise ValueError("shard_index is outside configured range")
    root = config.shards_root if shards_root is None else Path(shards_root)
    return root / table_name / f"split={split}" / f"shard_id={int(shard_index)}"


def _read_table_shard(
    config: GNNTrainingConfig,
    *,
    table_name: str,
    split: str,
    shard_index: int,
    shards_root: Path | None = None,
) -> pd.DataFrame:
    """Read every physical Parquet fragment in one logical table partition.

    DuckDB may emit multiple ``part_*.parquet`` files for a single Hive
    split/shard partition.  A logical shard remains the loader's memory
    boundary; sorting fragment paths makes assembly deterministic before the
    existing shard-level schema and graph-integrity validation runs.
    """

    directory = table_shard_directory(
        config,
        table_name=table_name,
        split=split,
        shard_index=shard_index,
        shards_root=shards_root,
    )
    if not directory.exists():
        return pd.DataFrame()
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    if len(files) == 1:
        frame = pd.read_parquet(files[0])
    else:
        frame = pd.concat(
            (pd.read_parquet(path) for path in files),
            ignore_index=True,
        )
    # Partition columns are normally persisted by DuckDB.  Re-add them for
    # compatibility with fixture writers that omit Hive partition columns.
    if "split" not in frame.columns:
        frame["split"] = split
    if "shard_id" not in frame.columns:
        frame["shard_id"] = shard_index
    return frame


def iter_shard_examples(
    config: GNNTrainingConfig,
    spec: GNNFeatureLayoutSpec,
    *,
    split: str,
    shard_index: int,
    shards_root: Path | None = None,
    include_fold_ids: frozenset[int] | None = None,
    exclude_fold_ids: frozenset[int] | None = None,
    require_positive: bool = False,
) -> list[GNNExample]:
    """Load and reassemble exactly one logical graph shard."""

    frames = {
        table_name: _read_table_shard(
            config,
            table_name=table_name,
            split=split,
            shard_index=shard_index,
            shards_root=shards_root,
        )
        for table_name in CACHE_TABLES
    }
    examples = build_shard_examples(
        spec,
        groups=frames[GROUP_TABLE],
        nodes=frames[NODE_TABLE],
        edges=frames[EDGE_TABLE],
        candidates=frames[CANDIDATE_TABLE],
    )
    if any(example.patient_fold_id >= config.fold_count for example in examples):
        raise ValueError("ranking group fold is outside the configured range")
    if include_fold_ids is not None:
        examples = [
            example
            for example in examples
            if example.patient_fold_id in include_fold_ids
        ]
    if exclude_fold_ids is not None:
        examples = [
            example
            for example in examples
            if example.patient_fold_id not in exclude_fold_ids
        ]
    if require_positive:
        examples = [example for example in examples if example.has_positive]
    return examples


def iter_batches(
    config: GNNTrainingConfig,
    spec: GNNFeatureLayoutSpec,
    *,
    split: str,
    batch_groups: int,
    shuffle: bool,
    seed: int,
    epoch: int = 0,
    shards_root: Path | None = None,
    include_fold_ids: frozenset[int] | None = None,
    exclude_fold_ids: frozenset[int] | None = None,
    require_positive: bool = False,
    max_edges: int | None = None,
    max_nodes: int | None = None,
) -> Iterator[GNNBatch]:
    """Yield graph-size-bounded batches from no more than one loaded shard."""

    if batch_groups < 1:
        raise ValueError("batch_groups must be positive")
    edge_limit = (
        config.optimization.max_edges_per_batch if max_edges is None else max_edges
    )
    node_limit = (
        config.optimization.max_nodes_per_batch if max_nodes is None else max_nodes
    )
    if edge_limit is not None and edge_limit < 1:
        raise ValueError("max_edges must be positive when provided")
    if node_limit is not None and node_limit < 1:
        raise ValueError("max_nodes must be positive when provided")
    shard_order = list(range(config.shard_count))
    if shuffle:
        shard_rng = np.random.default_rng(seed + epoch)
        shard_rng.shuffle(shard_order)

    for shard_index in shard_order:
        examples = iter_shard_examples(
            config,
            spec,
            split=split,
            shard_index=shard_index,
            shards_root=shards_root,
            include_fold_ids=include_fold_ids,
            exclude_fold_ids=exclude_fold_ids,
            require_positive=require_positive,
        )
        if not examples:
            continue
        if shuffle:
            group_rng = np.random.default_rng(
                seed + epoch * config.shard_count + shard_index
            )
            order = group_rng.permutation(len(examples))
            examples = [examples[index] for index in order]
        for example_batch in iter_example_batches(
            examples,
            max_groups=batch_groups,
            max_edges=edge_limit,
            max_nodes=node_limit,
        ):
            yield collate_examples(example_batch, spec)


def iter_example_batches(
    examples: Sequence[GNNExample],
    *,
    max_groups: int,
    max_edges: int | None,
    max_nodes: int | None,
) -> Iterator[list[GNNExample]]:
    """Greedily group examples without crossing configured graph-size limits.

    An individual graph that exceeds a limit fails closed without exposing its
    identifier. Such a graph requires an explicit representation or capacity
    review rather than silently defeating the configured memory boundary.
    """

    if max_groups < 1:
        raise ValueError("max_groups must be positive")
    if max_edges is not None and max_edges < 1:
        raise ValueError("max_edges must be positive when provided")
    if max_nodes is not None and max_nodes < 1:
        raise ValueError("max_nodes must be positive when provided")

    pending: list[GNNExample] = []
    pending_edges = 0
    pending_nodes = 0
    for example in examples:
        if max_edges is not None and example.num_edges > max_edges:
            raise ValueError("single graph exceeds max_edges batch ceiling")
        if max_nodes is not None and example.num_nodes > max_nodes:
            raise ValueError("single graph exceeds max_nodes batch ceiling")
        crosses_limit = bool(
            pending
            and (
                len(pending) >= max_groups
                or (
                    max_edges is not None
                    and pending_edges + example.num_edges > max_edges
                )
                or (
                    max_nodes is not None
                    and pending_nodes + example.num_nodes > max_nodes
                )
            )
        )
        if crosses_limit:
            yield pending
            pending = []
            pending_edges = 0
            pending_nodes = 0
        pending.append(example)
        pending_edges += example.num_edges
        pending_nodes += example.num_nodes
    if pending:
        yield pending
