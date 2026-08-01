from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from pipeline.gnn_training.frozen_transformer import (  # noqa: E402
    assert_artifact_hashes_unchanged,
    extract_frozen_outputs,
    load_frozen_transformer,
    module_is_frozen,
)
from pipeline.neural_training.config import (  # noqa: E402
    CANDIDATE_SIDE_FEATURES,
    NeuralArchitecture,
)
from pipeline.neural_training.dataset import FeatureLayoutSpec  # noqa: E402
from pipeline.neural_training.model import build_model  # noqa: E402


def _layout_payload(spec: FeatureLayoutSpec) -> dict[str, object]:
    return {
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
        "max_sequence_length": spec.max_sequence_length,
        "pad_index": spec.pad_index,
        "unk_index": spec.unk_index,
        "feature_version": spec.feature_version,
        "candidate_side_features": list(spec.candidate_side_features),
        "vocab_sizes": {
            "event": spec.event_vocab_size,
            "condition": spec.condition_vocab_size,
            "candidate": spec.candidate_vocab_size,
            "categorical": {
                name: size
                for name, size in zip(
                    spec.categorical_columns,
                    spec.categorical_vocab_sizes,
                    strict=True,
                )
            },
        },
    }


def _checkpoint_layout(spec: FeatureLayoutSpec) -> dict[str, object]:
    return {
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
        "max_sequence_length": spec.max_sequence_length,
        "event_vocab_size": spec.event_vocab_size,
        "condition_vocab_size": spec.condition_vocab_size,
        "candidate_vocab_size": spec.candidate_vocab_size,
        "categorical_vocab_sizes": list(spec.categorical_vocab_sizes),
        "candidate_side_features": list(spec.candidate_side_features),
    }


def _write_bundle(tmp_path: Path) -> tuple[Path, Path, Path, FeatureLayoutSpec]:
    spec = FeatureLayoutSpec(
        numeric_columns=("age",),
        categorical_columns=("sex",),
        max_sequence_length=2,
        pad_index=0,
        unk_index=1,
        event_vocab_size=8,
        condition_vocab_size=6,
        candidate_vocab_size=7,
        categorical_vocab_sizes=(4,),
        feature_version="temporal-features-v2",
        candidate_side_features=CANDIDATE_SIDE_FEATURES,
    )
    architecture = NeuralArchitecture(
        event_embedding_dim=8,
        encoder_layers=1,
        attention_heads=2,
        feedforward_dim=16,
        dropout=0.0,
        feature_dropout=0.0,
        categorical_embedding_dim=4,
        condition_embedding_dim=4,
        candidate_embedding_dim=8,
        context_hidden_dim=256,
        scorer_hidden_dim=16,
        candidate_side_hidden_dim=8,
    )
    model = build_model(spec, architecture)
    checkpoint_path = tmp_path / "transformer.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": asdict(architecture),
            "feature_layout": _checkpoint_layout(spec),
            "experiment_version": "synthetic-frozen-transformer",
        },
        checkpoint_path,
    )
    layout_path = tmp_path / "feature_layout.json"
    layout_path.write_text(
        json.dumps(_layout_payload(spec), sort_keys=True),
        encoding="utf-8",
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text('{"temperature": 1.0}\n', encoding="utf-8")
    return checkpoint_path, layout_path, calibration_path, spec


def _batch(spec: FeatureLayoutSpec) -> SimpleNamespace:
    candidate_count = 2
    return SimpleNamespace(
        numeric=torch.tensor([[0.25]], dtype=torch.float32),
        categorical=torch.tensor([[2]], dtype=torch.long),
        event_index=torch.tensor([[2, 3]], dtype=torch.long),
        event_time=torch.tensor([[0.1, 0.5]], dtype=torch.float32),
        event_value=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        event_value_mask=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        event_pad_mask=torch.tensor([[False, False]], dtype=torch.bool),
        condition_index=torch.tensor([2], dtype=torch.long),
        candidate_index=torch.tensor([[2, 3]], dtype=torch.long),
        candidate_mask=torch.tensor([[True, True]], dtype=torch.bool),
        candidate_side_features=torch.zeros(
            (1, candidate_count, spec.candidate_side_dim),
            dtype=torch.float32,
        ),
    )


def test_frozen_transformer_outputs_are_detached_and_artifacts_unchanged(
    tmp_path: Path,
) -> None:
    checkpoint, layout, calibration, spec = _write_bundle(tmp_path)
    bundle = load_frozen_transformer(
        checkpoint_path=checkpoint,
        feature_layout_path=layout,
        calibration_path=calibration,
    )

    assert module_is_frozen(bundle.model)
    outputs = extract_frozen_outputs(bundle, _batch(spec))

    assert outputs.context.shape == (1, 256)
    assert outputs.candidate_logits.shape == (1, 2)
    assert not outputs.context.requires_grad
    assert not outputs.candidate_logits.requires_grad
    assert all(parameter.grad is None for parameter in bundle.model.parameters())
    assert_artifact_hashes_unchanged(
        bundle.artifact_hashes,
        checkpoint_path=checkpoint,
        feature_layout_path=layout,
        calibration_path=calibration,
    )


def test_frozen_transformer_rejects_layout_drift(tmp_path: Path) -> None:
    checkpoint, layout, calibration, _spec = _write_bundle(tmp_path)
    payload = json.loads(layout.read_text(encoding="utf-8"))
    payload["numeric_columns"] = ["different"]
    layout.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint/layout mismatch"):
        load_frozen_transformer(
            checkpoint_path=checkpoint,
            feature_layout_path=layout,
            calibration_path=calibration,
        )


def test_frozen_transformer_detects_artifact_overwrite(tmp_path: Path) -> None:
    checkpoint, layout, calibration, _spec = _write_bundle(tmp_path)
    bundle = load_frozen_transformer(
        checkpoint_path=checkpoint,
        feature_layout_path=layout,
        calibration_path=calibration,
    )
    calibration.write_text('{"temperature": 2.0}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="calibration"):
        assert_artifact_hashes_unchanged(
            bundle.artifact_hashes,
            checkpoint_path=checkpoint,
            feature_layout_path=layout,
            calibration_path=calibration,
        )
