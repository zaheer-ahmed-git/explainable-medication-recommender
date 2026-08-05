"""Native-PyTorch relation-aware GNN recommendation primitives.

The model consumes packed patient query graphs through an attribute protocol;
it does not import the cache/dataset module at runtime.  Two R-GCN-style
message-passing layers update node representations, and a candidate-specific
attention pool summarizes observed pre-decision context.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Protocol

import torch
from torch import nn

from pipeline.gnn_training.config import GNNArchitecture
from pipeline.gnn_training.graph_encode import (
    FORWARD_RELATION_TYPES,
    NODE_CONTINUOUS_FEATURES,
    NODE_ROLE_TO_INDEX,
    NODE_TYPE_TO_INDEX,
    PAD_INDEX,
    RELATION_TO_INDEX,
    RELATION_TYPES,
)

ABLATION_VARIANTS = (
    "full",
    "rank_only",
    "no_message_passing",
    "no_condition_medication",
    "no_dense_lab_vital",
    "no_lab_vital_intervention",
)


class GraphBatchProtocol(Protocol):
    """Tensor attributes required by :class:`GNNRecommender`."""

    node_concept_index: torch.Tensor  # (N,) long
    node_type_index: torch.Tensor  # (N,) long
    node_role_index: torch.Tensor  # (N,) long
    observed_mask: torch.Tensor  # (N,) bool or float
    cold_start_mask: torch.Tensor  # (N,) bool or float
    node_continuous: torch.Tensor  # (N, F) float
    node_time_bin_index: torch.Tensor  # (N,) long
    edge_index: torch.Tensor  # (2, E) long
    edge_type: torch.Tensor  # (E,) long
    edge_weight: torch.Tensor  # (E,) float
    graph_index: torch.Tensor  # (N,) long in [0, G)
    query_node_index: torch.Tensor  # (G,) packed/global node indexes
    candidate_node_index: torch.Tensor  # (G, C) packed/global node indexes
    candidate_mask: torch.Tensor  # (G, C) bool
    candidate_rank: torch.Tensor  # (G, C) non-negative rank


class GNNFeatureSpecProtocol(Protocol):
    """Vocabulary sizes required to construct the embedding tables."""

    concept_vocab_size: int
    node_type_vocab_size: int
    node_role_vocab_size: int
    relation_count: int
    node_continuous_features: tuple[str, ...]
    time_bin_count: int


class GNNOutput(NamedTuple):
    """Candidate logits and the graph representation used by fusion."""

    logits: torch.Tensor
    candidate_representations: torch.Tensor


def _validate_ablation_variant(variant: str) -> str:
    if variant not in ABLATION_VARIANTS:
        allowed = ", ".join(ABLATION_VARIANTS)
        raise ValueError(
            f"unknown GNN ablation variant {variant!r}; expected {allowed}"
        )
    return variant


def _excluded_relation_indexes(variant: str) -> frozenset[int]:
    """Return relation indexes removed by a pre-registered ablation."""

    _validate_ablation_variant(variant)
    if variant in {"rank_only", "full", "no_message_passing"}:
        return frozenset()
    if variant == "no_condition_medication":
        names = {
            FORWARD_RELATION_TYPES[0],
            f"reverse_{FORWARD_RELATION_TYPES[0]}",
        }
    elif variant == "no_dense_lab_vital":
        forward = FORWARD_RELATION_TYPES[1:3]
        names = {*forward, *(f"reverse_{name}" for name in forward)}
    else:
        forward = FORWARD_RELATION_TYPES[1:4]
        names = {*forward, *(f"reverse_{name}" for name in forward)}
    return frozenset(RELATION_TO_INDEX[name] for name in names)


def relation_keep_mask(edge_type: torch.Tensor, variant: str) -> torch.Tensor:
    """Return the edge mask for one of the six registered variants."""

    excluded = _excluded_relation_indexes(variant)
    if not excluded:
        return torch.ones_like(edge_type, dtype=torch.bool)
    keep = torch.ones_like(edge_type, dtype=torch.bool)
    for relation_index in excluded:
        keep &= edge_type != relation_index
    return keep


class RelationMessagePassingLayer(nn.Module):
    """One relation-specific incoming-message aggregation layer."""

    def __init__(
        self,
        hidden_dim: int,
        relation_count: int,
        dropout: float,
        relation_dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if relation_count <= 0:
            raise ValueError("relation_count must be positive")
        if not 0.0 <= relation_dropout < 1.0:
            raise ValueError("relation_dropout must be in [0, 1)")
        self.hidden_dim = hidden_dim
        self.relation_count = relation_count
        self.relation_weight = nn.Parameter(
            torch.empty(relation_count, hidden_dim, hidden_dim)
        )
        # Start mostly open so the gated P1 model is close to the ungated
        # registered operator while still allowing evidence-driven shrinkage.
        self.relation_gate_logits = nn.Parameter(torch.full((relation_count,), 2.0))
        self.relation_dropout = float(relation_dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for relation_weight in self.relation_weight:
            nn.init.xavier_uniform_(relation_weight)
        self.norm.reset_parameters()

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        if node_states.ndim != 2 or node_states.shape[1] != self.hidden_dim:
            raise ValueError(f"node_states must have shape (N, {self.hidden_dim})")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        edge_count = edge_index.shape[1]
        if edge_type.shape != (edge_count,) or edge_weight.shape != (edge_count,):
            raise ValueError("edge_type and edge_weight must have shape (E,)")

        # Aggregate source states before applying a relation transform.  The
        # mathematically equivalent per-edge formulation,
        #
        #   (source_states @ relation_weight[edge_type]) * edge_weight
        #
        # materializes an ``(E, H, H)`` tensor.  At protected scale that was a
        # multi-GiB allocation even though there are only eleven relation
        # matrices.  Linearity lets us first accumulate one weighted ``(N, H)``
        # buffer per relation and apply that relation's matrix once.
        #
        # Keep aggregation and relation transforms in FP32 even when the outer
        # training context uses BF16/FP16.  Repeated incoming-message sums are
        # the numerically sensitive part of this branch, and returning FP32
        # states also keeps residual addition and LayerNorm stable.
        stable_states = node_states.to(dtype=torch.float32)
        update = torch.zeros_like(stable_states)
        with torch.autocast(device_type=node_states.device.type, enabled=False):
            if not edge_count:
                return self.norm(stable_states)

            if bool(((edge_type < 0) | (edge_type >= self.relation_count)).any()):
                raise ValueError("edge_type contains an unknown relation index")
            if not bool(torch.isfinite(edge_weight).all()):
                raise ValueError("edge_weight must be finite")
            source_index, destination_index = edge_index
            node_count = node_states.shape[0]
            invalid_node = (
                (source_index < 0)
                | (source_index >= node_count)
                | (destination_index < 0)
                | (destination_index >= node_count)
            )
            if bool(invalid_node.any()):
                raise ValueError("edge_index contains a non-local node index")
            stable_edge_weight = edge_weight.to(dtype=torch.float32)
            for relation_index in range(self.relation_count):
                relation_mask = edge_type == relation_index
                if not bool(relation_mask.any()):
                    continue
                relation_source = source_index[relation_mask]
                relation_destination = destination_index[relation_mask]
                weighted_sources = stable_states[relation_source] * (
                    stable_edge_weight[relation_mask].unsqueeze(-1)
                )
                aggregated = torch.zeros_like(stable_states)
                aggregated.index_add_(
                    0,
                    relation_destination,
                    weighted_sources,
                )
                relation_gate = torch.sigmoid(self.relation_gate_logits[relation_index])
                if self.training and self.relation_dropout > 0.0:
                    keep_relation = torch.rand((), device=node_states.device) >= (
                        self.relation_dropout
                    )
                    relation_gate = (
                        relation_gate
                        * keep_relation.to(dtype=relation_gate.dtype)
                        / (1.0 - self.relation_dropout)
                    )
                update.add_(
                    (aggregated @ self.relation_weight[relation_index]) * relation_gate
                )

            return self.norm(stable_states + self.dropout(self.activation(update)))


# Shorter alias used in architecture descriptions.
RelationAwareLayer = RelationMessagePassingLayer


class GNNRecommender(nn.Module):
    """Two-layer relation-aware graph branch with candidate scoring."""

    def __init__(
        self,
        *,
        concept_vocab_size: int,
        architecture: GNNArchitecture | None = None,
        node_type_vocab_size: int = len(NODE_TYPE_TO_INDEX),
        node_role_vocab_size: int = len(NODE_ROLE_TO_INDEX),
        ablation_variant: str = "full",
    ):
        super().__init__()
        architecture = architecture or GNNArchitecture()
        self.architecture = architecture
        self.ablation_variant = _validate_ablation_variant(ablation_variant)
        if concept_vocab_size <= 1:
            raise ValueError("concept_vocab_size must include PAD and UNK")
        if node_type_vocab_size <= 0 or node_role_vocab_size <= 0:
            raise ValueError("node type and role vocabularies must be non-empty")
        if architecture.relation_count != len(RELATION_TYPES):
            raise ValueError(
                f"relation_count must be the fixed {len(RELATION_TYPES)} relations"
            )
        if architecture.relation_layers <= 0:
            raise ValueError("relation_layers must be positive")

        self.concept_embedding = nn.Embedding(
            concept_vocab_size,
            architecture.concept_embedding_dim,
            padding_idx=PAD_INDEX,
        )
        self.node_type_embedding = nn.Embedding(
            node_type_vocab_size,
            architecture.node_type_embedding_dim,
        )
        self.node_role_embedding = nn.Embedding(
            node_role_vocab_size,
            architecture.node_role_embedding_dim,
        )
        self.time_bin_embedding = nn.Embedding(
            architecture.time_bin_count,
            architecture.time_bin_embedding_dim,
            padding_idx=0,
        )
        if architecture.node_continuous_dim != len(NODE_CONTINUOUS_FEATURES):
            raise ValueError("node_continuous_dim does not match the P1 contract")
        input_dim = (
            architecture.concept_embedding_dim
            + architecture.node_type_embedding_dim
            + architecture.node_role_embedding_dim
            + architecture.time_bin_embedding_dim
            + architecture.node_continuous_dim
            + 2
        )
        self.node_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, architecture.hidden_dim),
            nn.GELU(),
            nn.Dropout(architecture.dropout),
        )
        self.message_layers = nn.ModuleList(
            RelationMessagePassingLayer(
                architecture.hidden_dim,
                architecture.relation_count,
                architecture.dropout,
                architecture.relation_dropout,
            )
            for _ in range(architecture.relation_layers)
        )

        hidden = architecture.hidden_dim
        self.attention_query = nn.Linear(hidden * 2, hidden, bias=False)
        self.attention_key = nn.Linear(hidden, hidden, bias=False)
        self.attention_scale = 1.0 / math.sqrt(hidden)
        self.candidate_representation_dim = hidden * 4
        self.scorer = nn.Sequential(
            nn.LayerNorm(self.candidate_representation_dim + 1),
            nn.Linear(
                self.candidate_representation_dim + 1,
                architecture.scorer_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(architecture.dropout),
            nn.Linear(architecture.scorer_hidden_dim, 1),
        )
        self.rank_scale_raw = nn.Parameter(torch.zeros(()))
        self.rank_bias = nn.Parameter(torch.zeros(()))

    def encode_nodes(
        self,
        *,
        node_concept_index: torch.Tensor,
        node_type_index: torch.Tensor,
        node_role_index: torch.Tensor,
        observed_mask: torch.Tensor,
        cold_start_mask: torch.Tensor,
        node_continuous: torch.Tensor,
        node_time_bin_index: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Return packed node states after the selected message-passing variant."""

        node_count = node_concept_index.shape[0]
        expected = (node_count,)
        for name, value in (
            ("node_type_index", node_type_index),
            ("node_role_index", node_role_index),
            ("observed_mask", observed_mask),
            ("cold_start_mask", cold_start_mask),
            ("node_time_bin_index", node_time_bin_index),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape (N,)")
        if node_continuous.shape != (
            node_count,
            self.architecture.node_continuous_dim,
        ):
            raise ValueError("node_continuous must have shape (N, F)")
        if not bool(torch.isfinite(node_continuous).all()):
            raise ValueError("node_continuous must be finite")
        if bool(
            (
                (node_time_bin_index < 0)
                | (node_time_bin_index >= self.architecture.time_bin_count)
            ).any()
        ):
            raise ValueError("node_time_bin_index contains an unknown bin")

        concept = self.concept_embedding(node_concept_index)
        node_type = self.node_type_embedding(node_type_index)
        node_role = self.node_role_embedding(node_role_index)
        time_bin = self.time_bin_embedding(node_time_bin_index)
        flags = torch.stack(
            (
                observed_mask.to(dtype=concept.dtype),
                cold_start_mask.to(dtype=concept.dtype),
            ),
            dim=-1,
        )
        states = self.node_projection(
            torch.cat(
                (
                    concept,
                    node_type,
                    node_role,
                    time_bin,
                    node_continuous.to(dtype=concept.dtype),
                    flags,
                ),
                dim=-1,
            )
        )
        if self.ablation_variant in {"rank_only", "no_message_passing"}:
            return states

        keep = relation_keep_mask(edge_type, self.ablation_variant)
        selected_edge_index = edge_index[:, keep]
        selected_edge_type = edge_type[keep]
        selected_edge_weight = edge_weight[keep]
        for layer in self.message_layers:
            states = layer(
                states,
                selected_edge_index,
                selected_edge_type,
                selected_edge_weight,
            )
        return states

    def _attention_pool_observed(
        self,
        *,
        node_states: torch.Tensor,
        graph_index: torch.Tensor,
        observed_mask: torch.Tensor,
        query_states: torch.Tensor,
        candidate_states: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return candidate-conditioned observed-context pools."""

        group_count, candidate_count, hidden = candidate_states.shape
        pooled = node_states.new_zeros((group_count, candidate_count, hidden))
        observed = observed_mask.to(dtype=torch.bool)
        for group in range(group_count):
            context_indexes = torch.nonzero(
                observed & (graph_index == group), as_tuple=False
            ).flatten()
            if context_indexes.numel() == 0:
                continue
            context = node_states[context_indexes]
            valid_candidates = candidate_mask[group]
            if not bool(valid_candidates.any()):
                continue
            query = query_states[group].expand(candidate_count, -1)
            attention_query = self.attention_query(
                torch.cat((query, candidate_states[group]), dim=-1)
            )
            attention_key = self.attention_key(context)
            attention_logits = (
                attention_query @ attention_key.transpose(0, 1)
            ) * self.attention_scale
            attention = torch.softmax(attention_logits, dim=-1)
            group_pool = attention @ context
            pooled[group] = torch.where(
                valid_candidates.unsqueeze(-1),
                group_pool,
                torch.zeros_like(group_pool),
            )
        return pooled

    def score_candidates(
        self,
        *,
        node_states: torch.Tensor,
        graph_index: torch.Tensor,
        observed_mask: torch.Tensor,
        query_node_index: torch.Tensor,
        candidate_node_index: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_rank: torch.Tensor,
    ) -> GNNOutput:
        """Build candidate representations and return masked logits."""

        if query_node_index.ndim != 1:
            raise ValueError("query_node_index must have shape (G,)")
        group_count = query_node_index.shape[0]
        if candidate_node_index.ndim != 2:
            raise ValueError("candidate_node_index must have shape (G, C)")
        if candidate_node_index.shape[0] != group_count:
            raise ValueError("candidate_node_index group count does not match queries")
        if candidate_mask.shape != candidate_node_index.shape:
            raise ValueError("candidate_mask must match candidate_node_index")
        if candidate_rank.shape != candidate_node_index.shape:
            raise ValueError("candidate_rank must match candidate_node_index")
        if graph_index.shape != (node_states.shape[0],):
            raise ValueError("graph_index must have shape (N,)")
        if observed_mask.shape != (node_states.shape[0],):
            raise ValueError("observed_mask must have shape (N,)")

        node_count = node_states.shape[0]
        if bool(((query_node_index < 0) | (query_node_index >= node_count)).any()):
            raise ValueError("query_node_index contains a non-local node index")
        valid_candidate_indexes = candidate_node_index[candidate_mask]
        if bool(
            (
                (valid_candidate_indexes < 0) | (valid_candidate_indexes >= node_count)
            ).any()
        ):
            raise ValueError("candidate_node_index contains a non-local node index")

        safe_candidate_index = torch.where(
            candidate_mask,
            candidate_node_index,
            torch.zeros_like(candidate_node_index),
        )
        query_states = node_states[query_node_index]
        candidate_states = node_states[safe_candidate_index]
        query_expanded = query_states.unsqueeze(1).expand_as(candidate_states)
        pooled = self._attention_pool_observed(
            node_states=node_states,
            graph_index=graph_index,
            observed_mask=observed_mask,
            query_states=query_states,
            candidate_states=candidate_states,
            candidate_mask=candidate_mask,
        )
        representations = torch.cat(
            (
                query_expanded,
                candidate_states,
                query_expanded * candidate_states,
                pooled,
            ),
            dim=-1,
        )
        representations = torch.where(
            candidate_mask.unsqueeze(-1),
            representations,
            torch.zeros_like(representations),
        )

        ranks = candidate_rank.to(dtype=node_states.dtype)
        valid_ranks = ranks[candidate_mask]
        if not bool(torch.isfinite(valid_ranks).all()) or bool((valid_ranks < 0).any()):
            raise ValueError("valid candidate ranks must be finite and non-negative")
        log_rank = torch.log1p(ranks.clamp_min(0.0)).unsqueeze(-1)
        if self.ablation_variant == "rank_only":
            rank_logits = self.rank_bias - torch.nn.functional.softplus(
                self.rank_scale_raw
            ) * log_rank.squeeze(-1)
            rank_logits = rank_logits.masked_fill(~candidate_mask, float("-inf"))
            return GNNOutput(rank_logits, torch.zeros_like(representations))
        scorer_input = torch.cat((representations, log_rank), dim=-1)
        logits = self.scorer(scorer_input).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask, float("-inf"))
        return GNNOutput(logits, representations)

    def forward(
        self,
        *,
        node_concept_index: torch.Tensor,
        node_type_index: torch.Tensor,
        node_role_index: torch.Tensor,
        observed_mask: torch.Tensor,
        cold_start_mask: torch.Tensor,
        node_continuous: torch.Tensor,
        node_time_bin_index: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_weight: torch.Tensor,
        graph_index: torch.Tensor,
        query_node_index: torch.Tensor,
        candidate_node_index: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_rank: torch.Tensor,
    ) -> GNNOutput:
        node_states = self.encode_nodes(
            node_concept_index=node_concept_index,
            node_type_index=node_type_index,
            node_role_index=node_role_index,
            observed_mask=observed_mask,
            cold_start_mask=cold_start_mask,
            node_continuous=node_continuous,
            node_time_bin_index=node_time_bin_index,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_weight=edge_weight,
        )
        return self.score_candidates(
            node_states=node_states,
            graph_index=graph_index,
            observed_mask=observed_mask,
            query_node_index=query_node_index,
            candidate_node_index=candidate_node_index,
            candidate_mask=candidate_mask,
            candidate_rank=candidate_rank,
        )

    def forward_batch(self, batch: GraphBatchProtocol) -> GNNOutput:
        """Score any dataclass/object implementing :class:`GraphBatchProtocol`."""

        return self.forward(
            node_concept_index=batch.node_concept_index,
            node_type_index=batch.node_type_index,
            node_role_index=batch.node_role_index,
            observed_mask=batch.observed_mask,
            cold_start_mask=batch.cold_start_mask,
            node_continuous=batch.node_continuous,
            node_time_bin_index=batch.node_time_bin_index,
            edge_index=batch.edge_index,
            edge_type=batch.edge_type,
            edge_weight=batch.edge_weight,
            graph_index=batch.graph_index,
            query_node_index=batch.query_node_index,
            candidate_node_index=batch.candidate_node_index,
            candidate_mask=batch.candidate_mask,
            candidate_rank=batch.candidate_rank,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(
    spec: GNNFeatureSpecProtocol,
    architecture: GNNArchitecture | None = None,
    *,
    ablation_variant: str = "full",
) -> GNNRecommender:
    """Build a GNN from a dataset-compatible feature-layout object."""

    if spec.relation_count != len(RELATION_TYPES):
        raise ValueError(
            f"feature layout must use the fixed {len(RELATION_TYPES)} relations"
        )
    architecture = architecture or GNNArchitecture()
    if architecture.node_continuous_dim != len(spec.node_continuous_features):
        raise ValueError("architecture and layout continuous features differ")
    if architecture.time_bin_count != spec.time_bin_count:
        raise ValueError("architecture and layout time-bin counts differ")
    return GNNRecommender(
        concept_vocab_size=spec.concept_vocab_size,
        node_type_vocab_size=spec.node_type_vocab_size,
        node_role_vocab_size=spec.node_role_vocab_size,
        architecture=architecture,
        ablation_variant=ablation_variant,
    )
