"""Torch-guarded tests for the Transformer recommender and ranking losses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.neural_training.data import prepare_neural_caches
from pipeline.neural_training.dataset import (
    FeatureLayoutSpec,
    RankingGroupExample,
    collate_examples,
    iter_shard_examples,
)
from tests.neural_training_helpers import write_neural_fixture

torch = pytest.importorskip("torch")

from pipeline.neural_training.config import NeuralArchitecture  # noqa: E402
from pipeline.neural_training.losses import (  # noqa: E402
    auxiliary_bce_loss,
    combined_loss,
    listwise_softmax_loss,
)
from pipeline.neural_training.model import build_model  # noqa: E402


def _prepared_spec(tmp_path: Path):
    config = write_neural_fixture(tmp_path)
    prepare_neural_caches(config)
    spec = FeatureLayoutSpec.from_json(config.feature_layout_path)
    examples: list[RankingGroupExample] = []
    for shard_index in range(config.shard_count):
        examples.extend(
            iter_shard_examples(config, spec, split="train", shard_index=shard_index)
        )
    return spec, examples


def test_forward_produces_group_candidate_logits(tmp_path: Path) -> None:
    spec, examples = _prepared_spec(tmp_path)
    batch = collate_examples(examples, spec)
    model = build_model(spec, NeuralArchitecture())

    logits = model.forward_batch(batch)

    assert logits.shape == (batch.num_groups, batch.candidate_index.shape[1])
    assert torch.isfinite(logits[batch.candidate_mask]).all()


def test_empty_event_sequence_does_not_nan(tmp_path: Path) -> None:
    spec, examples = _prepared_spec(tmp_path)
    empty = examples[0]
    empty = RankingGroupExample(
        numeric=empty.numeric,
        categorical=empty.categorical,
        event_index=np.zeros((0,), dtype=np.int64),
        event_time=np.zeros((0,), dtype=np.float32),
        event_value=np.zeros((0,), dtype=np.float32),
        event_value_mask=np.zeros((0,), dtype=np.float32),
        condition_index=empty.condition_index,
        candidate_index=empty.candidate_index,
        labels=empty.labels,
        candidate_side_features=empty.candidate_side_features,
        source=empty.source,
        split=empty.split,
        stay_uid=empty.stay_uid,
        ranking_group_id=empty.ranking_group_id,
        index_condition_token=empty.index_condition_token,
        candidate_tokens=empty.candidate_tokens,
        candidate_rank=empty.candidate_rank,
    )
    batch = collate_examples([empty], spec)
    model = build_model(spec, NeuralArchitecture())

    logits = model.forward_batch(batch)

    assert torch.isfinite(logits[batch.candidate_mask]).all()


def test_listwise_loss_prefers_correct_ranking() -> None:
    mask = torch.tensor([[True, True, True]])
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    good = torch.tensor([[5.0, 0.0, 0.0]])
    bad = torch.tensor([[0.0, 0.0, 5.0]])

    good_loss = listwise_softmax_loss(good, labels, mask)
    bad_loss = listwise_softmax_loss(bad, labels, mask)

    assert good_loss < bad_loss
    assert good_loss >= 0.0


def test_listwise_loss_ignores_padded_candidates() -> None:
    mask = torch.tensor([[True, True, False]])
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    logits = torch.tensor([[3.0, 0.0, float("-inf")]])

    loss = listwise_softmax_loss(logits, labels, mask)

    assert torch.isfinite(loss)


def test_combined_loss_components_are_finite() -> None:
    mask = torch.tensor([[True, True, True]])
    labels = torch.tensor([[1.0, 0.0, 1.0]])
    logits = torch.tensor([[2.0, -1.0, 1.5]])

    outputs = combined_loss(logits, labels, mask, auxiliary_weight=0.1)

    assert torch.isfinite(outputs.total)
    assert torch.isfinite(outputs.listwise)
    assert torch.isfinite(outputs.auxiliary)
    assert torch.isclose(
        outputs.total,
        outputs.listwise + 0.1 * auxiliary_bce_loss(logits, labels, mask),
    )


def test_auxiliary_bce_upweights_sparse_positives() -> None:
    mask = torch.tensor([[True, True, True, True]])
    labels = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    # High logit on the sole positive should be rewarded more under pos_weight.
    good = torch.tensor([[4.0, -1.0, -1.0, -1.0]])
    bad = torch.tensor([[-1.0, 4.0, 4.0, 4.0]])

    assert auxiliary_bce_loss(good, labels, mask) < auxiliary_bce_loss(
        bad, labels, mask
    )


def test_dual_path_scorer_uses_candidate_side_features(tmp_path: Path) -> None:
    spec, examples = _prepared_spec(tmp_path)
    batch = collate_examples(examples[:1], spec)
    model = build_model(spec, NeuralArchitecture())

    assert batch.candidate_side_features.shape[-1] == spec.candidate_side_dim
    logits = model.forward_batch(batch)
    assert torch.isfinite(logits[batch.candidate_mask]).all()
