"""Reassemble sharded caches into stay-grouped tensor batches.

The prepared caches (see :mod:`pipeline.neural_training.data`) hold three flat,
hash-sharded Parquet tables per split: stay context features, truncated event
token rows, and per-candidate ranking rows. This module reads one shard at a
time with pandas, rebuilds one example per ranking group, and collates a list of
examples into padded PyTorch tensors.

Batching is by ranking group so the multi-positive listwise loss and the ranking
metrics operate on complete candidate sets. Streaming one shard at a time keeps
peak memory bounded to a single shard while still permitting shard-order and
within-shard shuffling for reproducible stochastic training.

PyTorch is imported lazily inside :func:`collate_examples`; the rest of the
module (layout loading, shard assembly) works without the optional ``neural``
dependency group so it stays unit-testable on the login node.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pipeline.neural_training.config import (
    CANDIDATE_SIDE_FEATURE_COUNT,
    PAD_INDEX,
    NeuralTrainingConfig,
)

CANDIDATE_SIDE_FEATURE_COLUMNS = (
    "candidate_rank_feat",
    "global_prior",
    "condition_candidate_prior",
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids importing torch eagerly
    import torch


@dataclass(frozen=True)
class FeatureLayoutSpec:
    """In-memory view of ``feature_layout.json`` used by the dataset and model."""

    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    max_sequence_length: int
    pad_index: int
    unk_index: int
    event_vocab_size: int
    condition_vocab_size: int
    candidate_vocab_size: int
    categorical_vocab_sizes: tuple[int, ...]
    feature_version: str | None
    candidate_side_features: tuple[str, ...] = CANDIDATE_SIDE_FEATURE_COLUMNS

    @property
    def numeric_dim(self) -> int:
        return len(self.numeric_columns)

    @property
    def candidate_side_dim(self) -> int:
        return len(self.candidate_side_features) or CANDIDATE_SIDE_FEATURE_COUNT

    @property
    def categorical_index_columns(self) -> tuple[str, ...]:
        return tuple(f"{name}_index" for name in self.categorical_columns)

    @classmethod
    def from_json(cls, path: Path) -> FeatureLayoutSpec:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab_sizes = payload["vocab_sizes"]
        categorical = tuple(payload["categorical_columns"])
        categorical_sizes = tuple(
            int(vocab_sizes["categorical"].get(name, cls_reserved()))
            for name in categorical
        )
        side_features = tuple(
            payload.get("candidate_side_features", CANDIDATE_SIDE_FEATURE_COLUMNS)
        )
        return cls(
            numeric_columns=tuple(payload["numeric_columns"]),
            categorical_columns=categorical,
            max_sequence_length=int(payload["max_sequence_length"]),
            pad_index=int(payload.get("pad_index", PAD_INDEX)),
            unk_index=int(payload.get("unk_index", 1)),
            event_vocab_size=int(vocab_sizes["event"]),
            condition_vocab_size=int(vocab_sizes["condition"]),
            candidate_vocab_size=int(vocab_sizes["candidate"]),
            categorical_vocab_sizes=categorical_sizes,
            feature_version=payload.get("feature_version"),
            candidate_side_features=side_features or CANDIDATE_SIDE_FEATURE_COLUMNS,
        )


def cls_reserved() -> int:
    """Return the reserved token count used as a categorical embedding floor."""

    from pipeline.neural_training.config import RESERVED_TOKEN_COUNT

    return RESERVED_TOKEN_COUNT


@dataclass
class RankingGroupExample:
    """One ranking group: shared stay context plus its labeled candidates."""

    numeric: np.ndarray  # (numeric_dim,) float32
    categorical: np.ndarray  # (num_categorical,) int64
    event_index: np.ndarray  # (length,) int64
    event_time: np.ndarray  # (length,) float32
    event_value: np.ndarray  # (length,) float32
    event_value_mask: np.ndarray  # (length,) float32
    condition_index: int
    candidate_index: np.ndarray  # (num_candidates,) int64
    labels: np.ndarray  # (num_candidates,) float32
    candidate_side_features: np.ndarray  # (num_candidates, F) float32
    source: str
    split: str
    stay_uid: str
    ranking_group_id: str
    index_condition_token: str
    candidate_tokens: tuple[str, ...]
    candidate_rank: np.ndarray  # (num_candidates,) int64

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_index.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.event_index.shape[0])

    @property
    def has_positive(self) -> bool:
        return bool(self.labels.sum() > 0)


def _empty_float(dim: int) -> np.ndarray:
    return np.zeros((dim,), dtype=np.float32)


def build_shard_examples(
    spec: FeatureLayoutSpec,
    *,
    context_features: pd.DataFrame,
    context_events: pd.DataFrame,
    groups: pd.DataFrame,
) -> list[RankingGroupExample]:
    """Reassemble one shard of caches into ranking-group examples."""

    if groups.empty:
        return []

    context_by_stay = _index_context(spec, context_features)
    events_by_stay = _index_events(spec, context_events)

    examples: list[RankingGroupExample] = []
    numeric_dim = spec.numeric_dim
    categorical_dim = len(spec.categorical_columns)
    empty_events = (
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )
    for (source, stay_uid, ranking_group_id), frame in groups.groupby(
        ["source", "stay_uid", "ranking_group_id"], sort=False
    ):
        stay_key = (source, stay_uid)
        numeric, categorical = context_by_stay.get(
            stay_key,
            (_empty_float(numeric_dim), np.zeros((categorical_dim,), dtype=np.int64)),
        )
        event_index, event_time, event_value, event_value_mask = events_by_stay.get(
            stay_key, empty_events
        )
        ordered = frame.sort_values("candidate_rank", kind="stable")
        side = _candidate_side_matrix(spec, ordered)
        examples.append(
            RankingGroupExample(
                numeric=numeric,
                categorical=categorical,
                event_index=event_index,
                event_time=event_time,
                event_value=event_value,
                event_value_mask=event_value_mask,
                condition_index=int(ordered["condition_index"].iloc[0]),
                candidate_index=ordered["candidate_index"].to_numpy(dtype=np.int64),
                labels=ordered["label_prescribed"]
                .astype("float32")
                .to_numpy(dtype=np.float32),
                candidate_side_features=side,
                source=str(source),
                split=str(ordered["split"].iloc[0]),
                stay_uid=str(stay_uid),
                ranking_group_id=str(ranking_group_id),
                index_condition_token=str(ordered["index_condition_token"].iloc[0]),
                candidate_tokens=tuple(
                    str(token)
                    for token in ordered["candidate_medication_token"].tolist()
                ),
                candidate_rank=ordered["candidate_rank"].to_numpy(dtype=np.int64),
            )
        )
    return examples


def _candidate_side_matrix(
    spec: FeatureLayoutSpec, ordered: pd.DataFrame
) -> np.ndarray:
    """Stack train-fit candidate-side features for one ranking group."""

    columns = list(spec.candidate_side_features)
    if not columns:
        return np.zeros((len(ordered), 0), dtype=np.float32)
    missing = [name for name in columns if name not in ordered.columns]
    if missing:
        # Older caches without priors: degrade to zeros rather than failing
        # mid-batch so synthetic fixtures can still exercise the model path.
        matrix = np.zeros((len(ordered), len(columns)), dtype=np.float32)
        for index, name in enumerate(columns):
            if name in ordered.columns:
                matrix[:, index] = ordered[name].to_numpy(dtype=np.float32)
        return matrix
    return ordered[columns].to_numpy(dtype=np.float32)


def _index_context(
    spec: FeatureLayoutSpec,
    context_features: pd.DataFrame,
) -> dict[tuple[Any, Any], tuple[np.ndarray, np.ndarray]]:
    """Return a ``(source, stay_uid) -> (numeric, categorical)`` lookup."""

    if context_features.empty:
        return {}
    numeric_cols = list(spec.numeric_columns)
    categorical_cols = list(spec.categorical_index_columns)
    numeric_matrix = (
        context_features[numeric_cols].to_numpy(dtype=np.float32)
        if numeric_cols
        else np.zeros((len(context_features), 0), dtype=np.float32)
    )
    categorical_matrix = (
        context_features[categorical_cols].to_numpy(dtype=np.int64)
        if categorical_cols
        else np.zeros((len(context_features), 0), dtype=np.int64)
    )
    sources = context_features["source"].tolist()
    stays = context_features["stay_uid"].tolist()
    return {
        (sources[i], stays[i]): (numeric_matrix[i], categorical_matrix[i])
        for i in range(len(context_features))
    }


def _index_events(
    spec: FeatureLayoutSpec,
    context_events: pd.DataFrame,
) -> dict[tuple[Any, Any], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return a ``(source, stay_uid) -> (index, time, value, mask)`` lookup.

    Events are ordered oldest-first (descending ``recency_rank``) for stable,
    reproducible sequences. Order is otherwise immaterial: the encoder is
    permutation-tolerant and carries elapsed time as an explicit feature.
    """

    if context_events.empty:
        return {}
    ordered = context_events.sort_values(
        ["source", "stay_uid", "recency_rank"], ascending=[True, True, False]
    )
    lookup: dict[
        tuple[Any, Any], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for (source, stay_uid), frame in ordered.groupby(
        ["source", "stay_uid"], sort=False
    ):
        head = frame.head(spec.max_sequence_length)
        lookup[(source, stay_uid)] = (
            head["event_index"].to_numpy(dtype=np.int64),
            head["event_time_norm"].to_numpy(dtype=np.float32),
            head["event_value_norm"].to_numpy(dtype=np.float32),
            head["event_value_mask"].to_numpy(dtype=np.float32),
        )
    return lookup


@dataclass
class NeuralBatch:
    """A collated ranking-group batch of padded tensors plus scoring metadata."""

    numeric: "torch.Tensor"  # (G, numeric_dim)
    categorical: "torch.Tensor"  # (G, num_categorical)
    event_index: "torch.Tensor"  # (G, L)
    event_time: "torch.Tensor"  # (G, L)
    event_value: "torch.Tensor"  # (G, L)
    event_value_mask: "torch.Tensor"  # (G, L)
    event_pad_mask: "torch.Tensor"  # (G, L) bool, True where padded
    condition_index: "torch.Tensor"  # (G,)
    candidate_index: "torch.Tensor"  # (G, C)
    candidate_mask: "torch.Tensor"  # (G, C) bool, True where valid
    labels: "torch.Tensor"  # (G, C)
    candidate_side_features: "torch.Tensor"  # (G, C, F)
    sources: tuple[str, ...]
    splits: tuple[str, ...]
    stay_uids: tuple[str, ...]
    ranking_group_ids: tuple[str, ...]
    index_condition_tokens: tuple[str, ...]
    candidate_tokens: tuple[tuple[str, ...], ...]
    candidate_ranks: tuple[np.ndarray, ...]

    @property
    def num_groups(self) -> int:
        return int(self.numeric.shape[0])

    def to(self, device: "torch.device | str") -> NeuralBatch:
        """Return a copy with all tensor fields moved to ``device``."""

        return NeuralBatch(
            numeric=self.numeric.to(device),
            categorical=self.categorical.to(device),
            event_index=self.event_index.to(device),
            event_time=self.event_time.to(device),
            event_value=self.event_value.to(device),
            event_value_mask=self.event_value_mask.to(device),
            event_pad_mask=self.event_pad_mask.to(device),
            condition_index=self.condition_index.to(device),
            candidate_index=self.candidate_index.to(device),
            candidate_mask=self.candidate_mask.to(device),
            labels=self.labels.to(device),
            candidate_side_features=self.candidate_side_features.to(device),
            sources=self.sources,
            splits=self.splits,
            stay_uids=self.stay_uids,
            ranking_group_ids=self.ranking_group_ids,
            index_condition_tokens=self.index_condition_tokens,
            candidate_tokens=self.candidate_tokens,
            candidate_ranks=self.candidate_ranks,
        )


def collate_examples(
    examples: Sequence[RankingGroupExample],
    spec: FeatureLayoutSpec,
) -> NeuralBatch:
    """Collate ranking-group examples into padded CPU tensors."""

    import torch

    if not examples:
        raise ValueError("cannot collate an empty batch")

    group_count = len(examples)
    numeric_dim = spec.numeric_dim
    categorical_dim = len(spec.categorical_columns)
    max_length = max(1, max(example.sequence_length for example in examples))
    max_candidates = max(example.num_candidates for example in examples)

    numeric = torch.zeros((group_count, numeric_dim), dtype=torch.float32)
    categorical = torch.zeros((group_count, categorical_dim), dtype=torch.long)
    event_index = torch.full(
        (group_count, max_length), spec.pad_index, dtype=torch.long
    )
    event_time = torch.zeros((group_count, max_length), dtype=torch.float32)
    event_value = torch.zeros((group_count, max_length), dtype=torch.float32)
    event_value_mask = torch.zeros((group_count, max_length), dtype=torch.float32)
    event_pad_mask = torch.ones((group_count, max_length), dtype=torch.bool)
    condition_index = torch.zeros((group_count,), dtype=torch.long)
    candidate_index = torch.full(
        (group_count, max_candidates), spec.pad_index, dtype=torch.long
    )
    candidate_mask = torch.zeros((group_count, max_candidates), dtype=torch.bool)
    labels = torch.zeros((group_count, max_candidates), dtype=torch.float32)
    side_dim = spec.candidate_side_dim
    candidate_side = torch.zeros(
        (group_count, max_candidates, side_dim), dtype=torch.float32
    )

    for row, example in enumerate(examples):
        if numeric_dim:
            numeric[row] = torch.tensor(example.numeric, dtype=torch.float32)
        if categorical_dim:
            categorical[row] = torch.tensor(example.categorical, dtype=torch.long)
        length = example.sequence_length
        if length:
            event_index[row, :length] = torch.tensor(
                example.event_index, dtype=torch.long
            )
            event_time[row, :length] = torch.tensor(
                example.event_time, dtype=torch.float32
            )
            event_value[row, :length] = torch.tensor(
                example.event_value, dtype=torch.float32
            )
            event_value_mask[row, :length] = torch.tensor(
                example.event_value_mask, dtype=torch.float32
            )
            event_pad_mask[row, :length] = False
        condition_index[row] = int(example.condition_index)
        count = example.num_candidates
        candidate_index[row, :count] = torch.tensor(
            example.candidate_index, dtype=torch.long
        )
        candidate_mask[row, :count] = True
        labels[row, :count] = torch.tensor(example.labels, dtype=torch.float32)
        if side_dim and count:
            candidate_side[row, :count] = torch.tensor(
                example.candidate_side_features, dtype=torch.float32
            )

    return NeuralBatch(
        numeric=numeric,
        categorical=categorical,
        event_index=event_index,
        event_time=event_time,
        event_value=event_value,
        event_value_mask=event_value_mask,
        event_pad_mask=event_pad_mask,
        condition_index=condition_index,
        candidate_index=candidate_index,
        candidate_mask=candidate_mask,
        labels=labels,
        candidate_side_features=candidate_side,
        sources=tuple(example.source for example in examples),
        splits=tuple(example.split for example in examples),
        stay_uids=tuple(example.stay_uid for example in examples),
        ranking_group_ids=tuple(example.ranking_group_id for example in examples),
        index_condition_tokens=tuple(
            example.index_condition_token for example in examples
        ),
        candidate_tokens=tuple(example.candidate_tokens for example in examples),
        candidate_ranks=tuple(example.candidate_rank for example in examples),
    )


def _read_shard(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def iter_shard_examples(
    config: NeuralTrainingConfig,
    spec: FeatureLayoutSpec,
    *,
    split: str,
    shard_index: int,
) -> list[RankingGroupExample]:
    """Load and reassemble the examples in a single shard for ``split``."""

    name = f"shard_{shard_index:04d}.parquet"
    return build_shard_examples(
        spec,
        context_features=_read_shard(config.context_features_dir(split) / name),
        context_events=_read_shard(config.context_events_dir(split) / name),
        groups=_read_shard(config.groups_dir(split) / name),
    )


def iter_batches(
    config: NeuralTrainingConfig,
    spec: FeatureLayoutSpec,
    *,
    split: str,
    batch_groups: int,
    shuffle: bool,
    seed: int,
    epoch: int = 0,
) -> Iterator[NeuralBatch]:
    """Yield collated ranking-group batches, streaming one shard at a time.

    When ``shuffle`` is set, shard visitation order and within-shard example
    order are permuted with a per-epoch-seeded RNG for reproducible stochastic
    training. Evaluation passes leave ordering deterministic.
    """

    shard_order = list(range(config.shard_count))
    if shuffle:
        rng = np.random.default_rng(seed + epoch)
        rng.shuffle(shard_order)

    for shard_index in shard_order:
        examples = iter_shard_examples(
            config, spec, split=split, shard_index=shard_index
        )
        if not examples:
            continue
        if shuffle:
            rng = np.random.default_rng(seed + epoch * 1000 + shard_index)
            order = rng.permutation(len(examples))
            examples = [examples[i] for i in order]
        for start in range(0, len(examples), batch_groups):
            chunk = examples[start : start + batch_groups]
            yield collate_examples(chunk, spec)
