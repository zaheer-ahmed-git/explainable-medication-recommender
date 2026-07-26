"""Tests for reassembling sharded caches into ranking-group examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.neural_training.data import prepare_neural_caches
from pipeline.neural_training.dataset import (
    FeatureLayoutSpec,
    collate_examples,
    iter_shard_examples,
)
from tests.neural_training_helpers import CANDIDATE_TOKENS, write_neural_fixture


def _all_train_examples(config, spec):
    examples = []
    for shard_index in range(config.shard_count):
        examples.extend(
            iter_shard_examples(config, spec, split="train", shard_index=shard_index)
        )
    return examples


def test_shard_assembly_rebuilds_groups(tmp_path: Path) -> None:
    config = write_neural_fixture(tmp_path)
    assert prepare_neural_caches(config)["status"] == "completed"
    spec = FeatureLayoutSpec.from_json(config.feature_layout_path)

    examples = _all_train_examples(config, spec)
    assert len(examples) == 4  # four positive train groups

    example = examples[0]
    assert example.numeric.shape == (spec.numeric_dim,)
    assert example.categorical.shape == (len(spec.categorical_columns),)
    assert example.num_candidates == len(CANDIDATE_TOKENS)
    assert example.candidate_tokens == CANDIDATE_TOKENS
    assert example.candidate_side_features.shape == (
        len(CANDIDATE_TOKENS),
        spec.candidate_side_dim,
    )
    # rank-1 candidate is the only positive; its log1p(rank) is ln(2).
    assert example.labels[0] == 1.0
    assert example.labels[1:].sum() == 0.0
    assert example.candidate_side_features[0, 0] == pytest.approx(np.log1p(1.0))
    assert example.condition_index >= spec.unk_index
    assert (example.candidate_index >= spec.unk_index).all()


def test_event_sequence_window_and_order(tmp_path: Path) -> None:
    config = write_neural_fixture(tmp_path)
    prepare_neural_caches(config)
    spec = FeatureLayoutSpec.from_json(config.feature_layout_path)

    example = _all_train_examples(config, spec)[0]
    # lactate@2h, lactate@6h, heart_rate@10h are kept; medication and the 30h
    # creatinine event are excluded.
    assert example.sequence_length == 3
    # Oldest-first ordering => non-decreasing normalized event time.
    assert np.all(np.diff(example.event_time) >= 0)
    assert example.event_index.min() >= spec.unk_index


def test_sequence_truncation_respects_limit(tmp_path: Path) -> None:
    config = write_neural_fixture(tmp_path, max_sequence_length=2)
    prepare_neural_caches(config)
    spec = FeatureLayoutSpec.from_json(config.feature_layout_path)

    example = _all_train_examples(config, spec)[0]
    assert example.sequence_length == 2


def test_collate_pads_events_and_candidates(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    config = write_neural_fixture(tmp_path)
    prepare_neural_caches(config)
    spec = FeatureLayoutSpec.from_json(config.feature_layout_path)

    examples = _all_train_examples(config, spec)[:2]
    batch = collate_examples(examples, spec)

    assert batch.num_groups == 2
    assert batch.numeric.shape == (2, spec.numeric_dim)
    assert batch.categorical.shape == (2, len(spec.categorical_columns))
    assert batch.candidate_index.shape == (2, len(CANDIDATE_TOKENS))
    assert batch.candidate_mask.all()
    assert batch.labels.shape == (2, len(CANDIDATE_TOKENS))
    assert batch.candidate_side_features.shape == (
        2,
        len(CANDIDATE_TOKENS),
        spec.candidate_side_dim,
    )
    # Non-padded event positions must be marked valid in the pad mask.
    assert (~batch.event_pad_mask).sum().item() == sum(
        example.sequence_length for example in examples
    )
