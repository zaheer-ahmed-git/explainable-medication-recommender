"""Transformer patient/context recommender (Stage 2 neural branch, v3).

The model implements the Transformer-only branch of the target two-branch
architecture (``Documentation/TrainingplanDetailed.md``). A Transformer encoder
with learned positional encodings summarizes the in-window clinical event
sequence; numeric stay features pass through a residual MLP with feature
dropout and are fused with categorical embeddings and the sequence summary into
a patient-context vector. A dual-path candidate scorer (MLP + scaled
dot-product) ranks every candidate medication using projected train-fit
candidate-side features (rank, priors, and Stage-1-matched graph tabular
summaries).

The GNN relation branch and the joint fusion head remain documented extension
points: :class:`TransformerRecommender` exposes its context vector so a later
fusion head can concatenate GNN relation embeddings without reworking this
module.

This module imports PyTorch directly and is only loaded when the neural branch
runs (after the structured recovery gate clears and ``uv sync --group neural``).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from pipeline.neural_training.config import (
    CANDIDATE_SIDE_FEATURE_COUNT,
    PAD_INDEX,
    NeuralArchitecture,
)
from pipeline.neural_training.dataset import FeatureLayoutSpec, NeuralBatch

# Per-event continuous features fused with the event token embedding:
# normalized elapsed time, normalized value, and a value-present mask.
EVENT_CONTINUOUS_FEATURES = 3


class EventSequenceEncoder(nn.Module):
    """Encode an in-window event token sequence into a fixed summary vector.

    A learned summary (``CLS``) token is prepended and never masked, so fully
    padded sequences (stays with no in-window events) are handled without the
    all-masked-row NaN that a masked-mean pool would produce. Learned absolute
    positional encodings complement the continuous time feature.
    """

    def __init__(self, spec: FeatureLayoutSpec, architecture: NeuralArchitecture):
        super().__init__()
        dim = architecture.event_embedding_dim
        self.max_sequence_length = spec.max_sequence_length
        self.token_embedding = nn.Embedding(
            spec.event_vocab_size, dim, padding_idx=PAD_INDEX
        )
        self.feature_projection = nn.Linear(EVENT_CONTINUOUS_FEATURES, dim)
        # Position 0 is reserved for the CLS/summary token; event positions are
        # 1..max_sequence_length (oldest-first after dataset reordering).
        self.positional_embedding = nn.Embedding(spec.max_sequence_length + 1, dim)
        self.summary_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.summary_token, std=0.02)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(architecture.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=architecture.attention_heads,
            dim_feedforward=architecture.feedforward_dim,
            dropout=architecture.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=architecture.encoder_layers,
            enable_nested_tensor=False,
        )
        self.output_dim = dim

    def forward(
        self,
        *,
        event_index: torch.Tensor,
        event_time: torch.Tensor,
        event_value: torch.Tensor,
        event_value_mask: torch.Tensor,
        event_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        group_count, length = event_index.shape
        token = self.token_embedding(event_index)
        continuous = torch.stack((event_time, event_value, event_value_mask), dim=-1)
        embedded = token + self.feature_projection(continuous)
        if length:
            # Oldest-first event rows receive positions 1..L (CLS uses 0).
            positions = torch.arange(
                1, length + 1, device=event_index.device, dtype=torch.long
            )
            positions = positions.unsqueeze(0).expand(group_count, -1)
            positions = positions.clamp(max=self.max_sequence_length)
            embedded = embedded + self.positional_embedding(positions)
        embedded = self.dropout(self.input_norm(embedded))

        summary = self.summary_token.expand(group_count, -1, -1)
        summary = summary + self.positional_embedding.weight[0].view(1, 1, -1)
        sequence = torch.cat((summary, embedded), dim=1)
        summary_mask = torch.zeros(
            (group_count, 1), dtype=torch.bool, device=event_index.device
        )
        padding_mask = torch.cat((summary_mask, event_pad_mask), dim=1)

        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)
        return encoded[:, 0]


class NumericEncoder(nn.Module):
    """Project high-dimensional stay numerics through a residual MLP.

    Feature dropout (training only) randomly zeros input columns so the network
    cannot rely on any single tabular cue — a lightweight stand-in for the
    column subsampling trees use, without adding FT-Transformer capacity that
    would worsen the observed epoch-1 overfit.
    """

    def __init__(self, numeric_dim: int, architecture: NeuralArchitecture):
        super().__init__()
        hidden = architecture.context_hidden_dim
        self.feature_dropout = float(architecture.feature_dropout)
        if numeric_dim <= 0:
            self.network: nn.Module | None = None
            self.output_dim = 0
            return
        self.input_norm = nn.LayerNorm(numeric_dim)
        self.network = nn.Sequential(
            nn.Linear(numeric_dim, hidden),
            nn.GELU(),
            nn.Dropout(architecture.dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(architecture.dropout),
            nn.Linear(hidden, hidden),
        )
        self.skip = (
            nn.Identity()
            if numeric_dim == hidden
            else nn.Linear(numeric_dim, hidden, bias=False)
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.output_dim = hidden

    def forward(self, numeric: torch.Tensor) -> torch.Tensor:
        if self.network is None:
            return numeric.new_zeros((numeric.shape[0], 0))
        features = self.input_norm(numeric)
        if self.training and self.feature_dropout > 0.0 and features.shape[-1] > 0:
            keep = torch.rand(
                features.shape[-1], device=features.device, dtype=features.dtype
            )
            keep = (keep >= self.feature_dropout).to(features.dtype)
            # Keep expected scale when columns are dropped.
            keep = keep / max(1e-6, 1.0 - self.feature_dropout)
            features = features * keep.unsqueeze(0)
        return self.output_norm(self.network(features) + self.skip(features))


class StaticContextEncoder(nn.Module):
    """Embed numeric stay features (via residual MLP) and low-cardinality categoricals."""

    def __init__(self, spec: FeatureLayoutSpec, architecture: NeuralArchitecture):
        super().__init__()
        self.numeric_encoder = NumericEncoder(spec.numeric_dim, architecture)
        self.categorical_embeddings = nn.ModuleList(
            nn.Embedding(
                size, architecture.categorical_embedding_dim, padding_idx=PAD_INDEX
            )
            for size in spec.categorical_vocab_sizes
        )
        self.output_dim = self.numeric_encoder.output_dim + (
            len(spec.categorical_vocab_sizes) * architecture.categorical_embedding_dim
        )

    def forward(
        self,
        *,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = [self.numeric_encoder(numeric)]
        for position, embedding in enumerate(self.categorical_embeddings):
            parts.append(embedding(categorical[:, position]))
        parts = [part for part in parts if part.shape[-1] > 0]
        if not parts:
            return numeric.new_zeros((numeric.shape[0], 0))
        return torch.cat(parts, dim=-1)


class CandidateSideEncoder(nn.Module):
    """Project train-fit candidate-side features (rank, priors, graph) to a vector."""

    def __init__(self, candidate_side_dim: int, architecture: NeuralArchitecture):
        super().__init__()
        hidden = architecture.candidate_side_hidden_dim
        self.candidate_side_dim = candidate_side_dim
        if candidate_side_dim <= 0:
            self.network: nn.Module | None = None
            self.output_dim = 0
            return
        self.network = nn.Sequential(
            nn.LayerNorm(candidate_side_dim),
            nn.Linear(candidate_side_dim, hidden),
            nn.GELU(),
            nn.Dropout(architecture.dropout),
            nn.Linear(hidden, hidden),
        )
        self.output_dim = hidden

    def forward(self, candidate_side_features: torch.Tensor) -> torch.Tensor:
        if self.network is None:
            return candidate_side_features.new_zeros(
                (*candidate_side_features.shape[:-1], 0)
            )
        return self.network(candidate_side_features)


class DualPathCandidateScorer(nn.Module):
    """MLP path plus scaled context–candidate dot product.

    Condition and candidate embeddings are owned by
    :class:`TransformerRecommender`; this module scores already-embedded
    tensors after candidate-side features have been projected.
    """

    def __init__(
        self,
        *,
        context_dim: int,
        architecture: NeuralArchitecture,
        candidate_side_encoded_dim: int,
    ):
        super().__init__()
        self.candidate_side_encoded_dim = candidate_side_encoded_dim
        scorer_input = (
            context_dim
            + architecture.condition_embedding_dim
            + architecture.candidate_embedding_dim
            + candidate_side_encoded_dim
        )
        self.mlp = nn.Sequential(
            nn.Linear(scorer_input, architecture.scorer_hidden_dim),
            nn.GELU(),
            nn.Dropout(architecture.dropout),
            nn.Linear(architecture.scorer_hidden_dim, architecture.scorer_hidden_dim),
            nn.GELU(),
            nn.Dropout(architecture.dropout),
            nn.Linear(architecture.scorer_hidden_dim, 1),
        )
        interaction_dim = min(context_dim, architecture.candidate_embedding_dim)
        self.query_proj = nn.Linear(context_dim, interaction_dim, bias=False)
        self.key_proj = nn.Linear(
            architecture.candidate_embedding_dim, interaction_dim, bias=False
        )
        # Condition-aware bilinear path: context ⊙ (condition ⊗ candidate).
        self.condition_gate = nn.Linear(
            architecture.condition_embedding_dim, interaction_dim, bias=False
        )
        self.scale = 1.0 / math.sqrt(interaction_dim)

    def forward(
        self,
        *,
        context: torch.Tensor,
        condition: torch.Tensor,
        candidate: torch.Tensor,
        candidate_side_encoded: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        candidate_count = candidate.shape[1]
        context_expanded = context.unsqueeze(1).expand(-1, candidate_count, -1)
        condition_expanded = condition.unsqueeze(1).expand(-1, candidate_count, -1)
        scorer_input = torch.cat(
            (
                context_expanded,
                condition_expanded,
                candidate,
                candidate_side_encoded,
            ),
            dim=-1,
        )
        mlp_logits = self.mlp(scorer_input).squeeze(-1)
        query = self.query_proj(context).unsqueeze(1)
        key = self.key_proj(candidate)
        gate = torch.sigmoid(self.condition_gate(condition)).unsqueeze(1)
        dot_logits = (query * key * gate).sum(dim=-1) * self.scale
        logits = mlp_logits + dot_logits
        return logits.masked_fill(~candidate_mask, float("-inf"))


class TransformerRecommender(nn.Module):
    """Full Transformer-branch recommender: context encoder + candidate scorer."""

    def __init__(self, spec: FeatureLayoutSpec, architecture: NeuralArchitecture):
        super().__init__()
        self.spec = spec
        self.architecture = architecture

        self.event_encoder = EventSequenceEncoder(spec, architecture)
        self.static_encoder = StaticContextEncoder(spec, architecture)

        fusion_input = self.event_encoder.output_dim + self.static_encoder.output_dim
        hidden = architecture.context_hidden_dim
        self.fusion_norm = nn.LayerNorm(fusion_input) if fusion_input else nn.Identity()
        self.context_mlp = nn.Sequential(
            nn.Linear(fusion_input, hidden),
            nn.GELU(),
            nn.Dropout(architecture.dropout),
            nn.Linear(hidden, hidden),
        )
        self.context_dim = hidden

        self.condition_embedding = nn.Embedding(
            spec.condition_vocab_size,
            architecture.condition_embedding_dim,
            padding_idx=PAD_INDEX,
        )
        self.candidate_embedding = nn.Embedding(
            spec.candidate_vocab_size,
            architecture.candidate_embedding_dim,
            padding_idx=PAD_INDEX,
        )
        self.candidate_side_dim = (
            spec.candidate_side_dim
            if spec.candidate_side_dim
            else CANDIDATE_SIDE_FEATURE_COUNT
        )
        self.candidate_side_encoder = CandidateSideEncoder(
            self.candidate_side_dim, architecture
        )
        self.scorer = DualPathCandidateScorer(
            context_dim=hidden,
            architecture=architecture,
            candidate_side_encoded_dim=self.candidate_side_encoder.output_dim,
        )

    def encode_context(
        self,
        *,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        event_index: torch.Tensor,
        event_time: torch.Tensor,
        event_value: torch.Tensor,
        event_value_mask: torch.Tensor,
        event_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the fused patient-context vector ``(G, context_dim)``.

        Exposed separately so a future fusion head can concatenate GNN relation
        embeddings before scoring.
        """

        event_summary = self.event_encoder(
            event_index=event_index,
            event_time=event_time,
            event_value=event_value,
            event_value_mask=event_value_mask,
            event_pad_mask=event_pad_mask,
        )
        static_summary = self.static_encoder(numeric=numeric, categorical=categorical)
        fused = torch.cat((event_summary, static_summary), dim=-1)
        return self.context_mlp(self.fusion_norm(fused))

    def score_candidates(
        self,
        *,
        context: torch.Tensor,
        condition_index: torch.Tensor,
        candidate_index: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_side_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw candidate logits ``(G, C)`` (padded slots set to ``-inf``)."""

        condition = self.condition_embedding(condition_index)
        candidate = self.candidate_embedding(candidate_index)
        side_encoded = self.candidate_side_encoder(candidate_side_features)
        return self.scorer(
            context=context,
            condition=condition,
            candidate=candidate,
            candidate_side_encoded=side_encoded,
            candidate_mask=candidate_mask,
        )

    def forward(
        self,
        *,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        event_index: torch.Tensor,
        event_time: torch.Tensor,
        event_value: torch.Tensor,
        event_value_mask: torch.Tensor,
        event_pad_mask: torch.Tensor,
        condition_index: torch.Tensor,
        candidate_index: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_side_features: torch.Tensor,
    ) -> torch.Tensor:
        context = self.encode_context(
            numeric=numeric,
            categorical=categorical,
            event_index=event_index,
            event_time=event_time,
            event_value=event_value,
            event_value_mask=event_value_mask,
            event_pad_mask=event_pad_mask,
        )
        return self.score_candidates(
            context=context,
            condition_index=condition_index,
            candidate_index=candidate_index,
            candidate_mask=candidate_mask,
            candidate_side_features=candidate_side_features,
        )

    def forward_batch(self, batch: NeuralBatch) -> torch.Tensor:
        """Convenience wrapper that scores a collated :class:`NeuralBatch`."""

        return self.forward(
            numeric=batch.numeric,
            categorical=batch.categorical,
            event_index=batch.event_index,
            event_time=batch.event_time,
            event_value=batch.event_value,
            event_value_mask=batch.event_value_mask,
            event_pad_mask=batch.event_pad_mask,
            condition_index=batch.condition_index,
            candidate_index=batch.candidate_index,
            candidate_mask=batch.candidate_mask,
            candidate_side_features=batch.candidate_side_features,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(
    spec: FeatureLayoutSpec,
    architecture: NeuralArchitecture,
) -> TransformerRecommender:
    """Construct a :class:`TransformerRecommender` from a layout and architecture."""

    return TransformerRecommender(spec, architecture)
