"""Torch-free vocabularies and local graph-edge encoding primitives.

The GNN preparation path uses one concept vocabulary fitted on MIMIC-train
nodes.  Node types, node roles, and relation indexes are fixed contracts rather
than learned vocabularies.  Patient-specific edges are expanded with reverse
relations and self loops, then normalized over incoming messages for each
``(relation, destination)`` pair.

This module intentionally has no PyTorch dependency so preparation and contract
checks can run without the optional neural dependency group.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_INDEX = 0
UNK_INDEX = 1
RESERVED_TOKEN_COUNT = 2

NODE_TYPES = ("condition", "medication", "lab", "vital", "intervention")
NODE_ROLES = ("query_condition", "candidate_medication", "observed_context")

NODE_TYPE_VOCABULARY = NODE_TYPES
NODE_ROLE_VOCABULARY = NODE_ROLES
NODE_TYPE_TO_INDEX: Mapping[str, int] = MappingProxyType(
    {token: index for index, token in enumerate(NODE_TYPE_VOCABULARY)}
)
NODE_ROLE_TO_INDEX: Mapping[str, int] = MappingProxyType(
    {token: index for index, token in enumerate(NODE_ROLE_VOCABULARY)}
)

FORWARD_RELATION_TYPES = (
    "condition_medication_train_positive",
    "condition_lab_predecision",
    "condition_vital_predecision",
    "condition_intervention_predecision",
    "medication_medication_train_coprescribed",
)


def reverse_relation_name(relation: str) -> str:
    """Return the stable reverse name for a forward graph relation."""

    if relation not in FORWARD_RELATION_TYPES:
        raise ValueError(f"unknown forward relation: {relation!r}")
    return f"reverse_{relation}"


REVERSE_RELATION_TYPES = tuple(
    f"reverse_{relation}" for relation in FORWARD_RELATION_TYPES
)
SELF_LOOP_RELATION = "self_loop"
RELATION_TYPES = (
    *FORWARD_RELATION_TYPES,
    *REVERSE_RELATION_TYPES,
    SELF_LOOP_RELATION,
)
RELATION_TO_INDEX: Mapping[str, int] = MappingProxyType(
    {relation: index for index, relation in enumerate(RELATION_TYPES)}
)

# Vocabulary aliases make the fixed nature of these mappings explicit at call
# sites while retaining concise names for cache preparation.
NODE_TYPE_VOCAB = NODE_TYPE_TO_INDEX
NODE_ROLE_VOCAB = NODE_ROLE_TO_INDEX
RELATION_VOCAB = RELATION_TO_INDEX


@dataclass(frozen=True)
class ConceptVocabulary:
    """Immutable, deterministic train-derived concept vocabulary.

    ``tokens`` always begins with ``<PAD>`` and ``<UNK>``.  All remaining
    concepts are exact node identifiers sorted lexicographically, so fitting is
    invariant to input order.
    """

    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.tokens) < RESERVED_TOKEN_COUNT:
            raise ValueError("concept vocabulary must contain PAD and UNK")
        if self.tokens[:RESERVED_TOKEN_COUNT] != (PAD_TOKEN, UNK_TOKEN):
            raise ValueError("concept vocabulary must reserve PAD=0 and UNK=1")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("concept vocabulary tokens must be unique")
        if any(not isinstance(token, str) or not token for token in self.tokens):
            raise ValueError("concept vocabulary tokens must be non-empty strings")

    @classmethod
    def fit(cls, train_concepts: Iterable[str]) -> ConceptVocabulary:
        """Fit a unified vocabulary from training concepts only."""

        concepts: set[str] = set()
        for concept in train_concepts:
            if not isinstance(concept, str):
                raise TypeError("training concepts must be strings")
            if not concept:
                raise ValueError("training concepts must be non-empty")
            if concept not in {PAD_TOKEN, UNK_TOKEN}:
                concepts.add(concept)
        return cls((PAD_TOKEN, UNK_TOKEN, *sorted(concepts)))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, int]) -> ConceptVocabulary:
        """Construct a vocabulary from a contiguous token-to-index mapping."""

        if set(mapping.values()) != set(range(len(mapping))):
            raise ValueError("concept vocabulary indexes must be contiguous from zero")
        ordered = tuple(
            token for token, _ in sorted(mapping.items(), key=lambda x: x[1])
        )
        return cls(ordered)

    @property
    def token_to_index(self) -> dict[str, int]:
        """Return a plain token-to-index dictionary suitable for serialization."""

        return {token: index for index, token in enumerate(self.tokens)}

    @property
    def size(self) -> int:
        return len(self.tokens)

    def encode(self, concept: str | None) -> int:
        """Encode one concept, mapping missing or unseen values to ``UNK``."""

        if concept is None:
            return UNK_INDEX
        return self.token_to_index.get(concept, UNK_INDEX)

    def encode_many(self, concepts: Iterable[str | None]) -> NDArray[np.int64]:
        """Encode concepts as a one-dimensional ``int64`` array."""

        mapping = self.token_to_index
        return np.asarray(
            [mapping.get(concept, UNK_INDEX) for concept in concepts],
            dtype=np.int64,
        )

    def to_dict(self) -> dict[str, int]:
        """Return a serializable token-to-index mapping."""

        return self.token_to_index


def fit_concept_vocabulary(train_concepts: Iterable[str]) -> ConceptVocabulary:
    """Return a deterministic unified vocabulary fitted from training concepts."""

    return ConceptVocabulary.fit(train_concepts)


def encode_concepts(
    concepts: Iterable[str | None],
    vocabulary: ConceptVocabulary | Mapping[str, int],
) -> NDArray[np.int64]:
    """Encode concepts with OOV values mapped to ``UNK_INDEX``."""

    mapping = (
        vocabulary.token_to_index
        if isinstance(vocabulary, ConceptVocabulary)
        else vocabulary
    )
    if mapping.get(PAD_TOKEN) != PAD_INDEX or mapping.get(UNK_TOKEN) != UNK_INDEX:
        raise ValueError("concept vocabulary must reserve PAD=0 and UNK=1")
    return np.asarray(
        [mapping.get(concept, UNK_INDEX) for concept in concepts],
        dtype=np.int64,
    )


def encode_node_types(node_types: Iterable[str | None]) -> NDArray[np.int64]:
    """Encode stable node types, rejecting values outside the graph schema."""

    encoded: list[int] = []
    for value in node_types:
        if value not in NODE_TYPE_TO_INDEX:
            raise ValueError(f"unknown node type: {value!r}")
        encoded.append(NODE_TYPE_TO_INDEX[value])
    return np.asarray(encoded, dtype=np.int64)


def encode_node_roles(node_roles: Iterable[str | None]) -> NDArray[np.int64]:
    """Encode stable node roles, rejecting values outside the graph schema."""

    encoded: list[int] = []
    for value in node_roles:
        if value not in NODE_ROLE_TO_INDEX:
            raise ValueError(f"unknown node role: {value!r}")
        encoded.append(NODE_ROLE_TO_INDEX[value])
    return np.asarray(encoded, dtype=np.int64)


@dataclass(frozen=True)
class LocalEdge:
    """One forward edge using node indexes local to a patient subgraph."""

    src_index: int
    dst_index: int
    relation: str
    support_count: float


@dataclass(frozen=True)
class EncodedGraphEdges:
    """Expanded, integer-encoded edge tensors represented as NumPy arrays."""

    edge_index: NDArray[np.int64]
    edge_type: NDArray[np.int64]
    edge_weight: NDArray[np.float32]
    transformed_support: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, edge_count)")
        edge_count = self.edge_index.shape[1]
        if self.edge_type.shape != (edge_count,):
            raise ValueError("edge_type must have shape (edge_count,)")
        if self.edge_weight.shape != (edge_count,):
            raise ValueError("edge_weight must have shape (edge_count,)")
        if self.transformed_support.shape != (edge_count,):
            raise ValueError("transformed_support must have shape (edge_count,)")

    @property
    def relation_index(self) -> NDArray[np.int64]:
        """Alias retained for callers that name the encoded field by relation."""

        return self.edge_type

    @property
    def edge_count(self) -> int:
        return int(self.edge_type.shape[0])


def transform_support_counts(
    support_counts: Sequence[float],
    *,
    log_support: bool = True,
) -> NDArray[np.float64]:
    """Validate and optionally apply ``log1p`` to non-negative supports."""

    transformed: list[float] = []
    for support in support_counts:
        if isinstance(support, bool):
            raise TypeError("edge support counts must be real numbers")
        try:
            value = float(support)
        except (TypeError, ValueError) as exc:
            raise TypeError("edge support counts must be real numbers") from exc
        if not math.isfinite(value):
            raise ValueError("edge support counts must be finite")
        if value < 0.0:
            raise ValueError("edge support counts must be non-negative")
        transformed.append(math.log1p(value) if log_support else value)
    return np.asarray(transformed, dtype=np.float64)


def normalize_incoming_support(
    *,
    dst_indices: Sequence[int] | NDArray[np.int64],
    relation_indices: Sequence[int] | NDArray[np.int64],
    transformed_support: Sequence[float] | NDArray[np.float64],
) -> NDArray[np.float32]:
    """Normalize weights within every ``(relation, destination)`` group.

    A zero-support group receives uniform weights.  This preserves finite
    messages and keeps every non-empty incoming group summing to one.
    """

    destinations = np.asarray(dst_indices, dtype=np.int64)
    relations = np.asarray(relation_indices, dtype=np.int64)
    support = np.asarray(transformed_support, dtype=np.float64)
    if destinations.ndim != 1 or relations.ndim != 1 or support.ndim != 1:
        raise ValueError("edge normalization inputs must be one-dimensional")
    if not (len(destinations) == len(relations) == len(support)):
        raise ValueError("edge normalization inputs must have equal lengths")
    if not np.isfinite(support).all():
        raise ValueError("transformed edge supports must be finite")
    if (support < 0.0).any():
        raise ValueError("transformed edge supports must be non-negative")

    weights = np.zeros(len(support), dtype=np.float64)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(zip(relations.tolist(), destinations.tolist())):
        groups.setdefault(key, []).append(index)
    for indexes in groups.values():
        total = float(support[indexes].sum())
        if total > 0.0:
            weights[indexes] = support[indexes] / total
        else:
            weights[indexes] = 1.0 / len(indexes)
    return weights.astype(np.float32)


def _validate_num_nodes(num_nodes: int) -> int:
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, Integral):
        raise TypeError("num_nodes must be an integer")
    value = int(num_nodes)
    if value < 0:
        raise ValueError("num_nodes must be non-negative")
    return value


def _validate_local_index(index: int, *, num_nodes: int, field: str) -> int:
    if isinstance(index, bool) or not isinstance(index, Integral):
        raise TypeError(f"{field} must be an integer")
    value = int(index)
    if value < 0 or value >= num_nodes:
        raise ValueError(
            f"{field}={value} is outside local node range [0, {num_nodes})"
        )
    return value


def encode_graph_edges(
    num_nodes: int,
    edges: Iterable[LocalEdge],
    *,
    add_reverse: bool = True,
    add_self_loops: bool = True,
    log_support: bool = True,
) -> EncodedGraphEdges:
    """Validate, expand, transform, and normalize forward local graph edges.

    Input relations must be one of the five train-fit forward relations.
    Original edges retain input order, followed by reverse edges in the same
    order and then self loops in ascending node-index order.
    """

    node_count = _validate_num_nodes(num_nodes)
    forward_edges = tuple(edges)
    source: list[int] = []
    destination: list[int] = []
    relation_names: list[str] = []
    supports: list[float] = []

    for edge in forward_edges:
        if not isinstance(edge, LocalEdge):
            raise TypeError("edges must contain LocalEdge instances")
        src = _validate_local_index(
            edge.src_index, num_nodes=node_count, field="src_index"
        )
        dst = _validate_local_index(
            edge.dst_index, num_nodes=node_count, field="dst_index"
        )
        if edge.relation not in FORWARD_RELATION_TYPES:
            raise ValueError(f"unknown forward relation: {edge.relation!r}")
        # Validate here so reverse expansion cannot duplicate an invalid value.
        transform_support_counts((edge.support_count,), log_support=False)
        source.append(src)
        destination.append(dst)
        relation_names.append(edge.relation)
        supports.append(float(edge.support_count))

    if add_reverse:
        for edge in forward_edges:
            source.append(int(edge.dst_index))
            destination.append(int(edge.src_index))
            relation_names.append(reverse_relation_name(edge.relation))
            supports.append(float(edge.support_count))

    if add_self_loops:
        for node_index in range(node_count):
            source.append(node_index)
            destination.append(node_index)
            relation_names.append(SELF_LOOP_RELATION)
            supports.append(1.0)

    edge_index = np.asarray((source, destination), dtype=np.int64)
    if edge_index.size == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
    edge_type = np.asarray(
        [RELATION_TO_INDEX[name] for name in relation_names],
        dtype=np.int64,
    )
    transformed = transform_support_counts(supports, log_support=log_support)
    weights = normalize_incoming_support(
        dst_indices=edge_index[1],
        relation_indices=edge_type,
        transformed_support=transformed,
    )
    return EncodedGraphEdges(
        edge_index=edge_index,
        edge_type=edge_type,
        edge_weight=weights,
        transformed_support=transformed,
    )


def expand_and_normalize_edges(
    *,
    num_nodes: int,
    src_indices: Sequence[int],
    dst_indices: Sequence[int],
    relations: Sequence[str],
    support_counts: Sequence[float],
    add_reverse: bool = True,
    add_self_loops: bool = True,
    log_support: bool = True,
) -> EncodedGraphEdges:
    """Column-oriented wrapper around :func:`encode_graph_edges`."""

    lengths = {
        len(src_indices),
        len(dst_indices),
        len(relations),
        len(support_counts),
    }
    if len(lengths) != 1:
        raise ValueError("edge columns must have equal lengths")
    edges = (
        LocalEdge(src, dst, relation, support)
        for src, dst, relation, support in zip(
            src_indices,
            dst_indices,
            relations,
            support_counts,
            strict=True,
        )
    )
    return encode_graph_edges(
        num_nodes,
        edges,
        add_reverse=add_reverse,
        add_self_loops=add_self_loops,
        log_support=log_support,
    )
