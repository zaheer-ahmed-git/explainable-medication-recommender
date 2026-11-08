"""Immutable adapter for the frozen Stage 2 Transformer.

The GNN pipeline consumes the Transformer's patient-context vectors and raw
candidate logits, but it must never update or rewrite the selected Transformer
artifacts.  This module reconstructs the model from the architecture stored in
its checkpoint (rather than from mutable current defaults), verifies the
checkpoint/layout contract, freezes every parameter, and exposes detached
forward outputs for cache preparation and fusion.

PyTorch is imported lazily so importing :mod:`pipeline.gnn_training` and running
the graph-only preparation checks does not require the optional ``neural``
dependency group.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from pipeline.neural_training.config import NeuralArchitecture
from pipeline.neural_training.dataset import FeatureLayoutSpec
from pipeline.training_contract import sha256_file


@dataclass(frozen=True)
class FrozenTransformerBundle:
    """Loaded immutable Transformer plus its verified reconstruction metadata."""

    model: Any
    feature_layout: FeatureLayoutSpec
    architecture: NeuralArchitecture
    checkpoint_metadata: dict[str, Any]
    artifact_hashes: dict[str, str]


@dataclass(frozen=True)
class FrozenTransformerOutputs:
    """Detached context and candidate-logit tensors returned by one forward pass."""

    context: Any
    candidate_logits: Any


def _architecture_from_checkpoint(payload: dict[str, Any]) -> NeuralArchitecture:
    raw = payload.get("architecture")
    if not isinstance(raw, dict):
        raise ValueError("frozen Transformer checkpoint is missing architecture")
    allowed = {item.name for item in fields(NeuralArchitecture)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            "frozen Transformer checkpoint has unknown architecture fields: "
            + ", ".join(unknown)
        )
    try:
        return NeuralArchitecture(**raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "frozen Transformer checkpoint architecture is invalid"
        ) from error


def _validate_checkpoint_layout(
    payload: dict[str, Any],
    spec: FeatureLayoutSpec,
) -> None:
    """Reject a checkpoint reconstructed against a different feature layout."""

    raw = payload.get("feature_layout")
    if not isinstance(raw, dict):
        raise ValueError("frozen Transformer checkpoint is missing feature layout")
    expected: dict[str, Any] = {
        "numeric_columns": list(spec.numeric_columns),
        "categorical_columns": list(spec.categorical_columns),
        "max_sequence_length": spec.max_sequence_length,
        "event_vocab_size": spec.event_vocab_size,
        "condition_vocab_size": spec.condition_vocab_size,
        "candidate_vocab_size": spec.candidate_vocab_size,
        "categorical_vocab_sizes": list(spec.categorical_vocab_sizes),
        "candidate_side_features": list(spec.candidate_side_features),
    }
    mismatches = sorted(
        name for name, value in expected.items() if raw.get(name) != value
    )
    if mismatches:
        raise ValueError(
            "frozen Transformer checkpoint/layout mismatch: " + ", ".join(mismatches)
        )


def freeze_module(module: Any) -> Any:
    """Put a torch module in evaluation mode and disable every parameter gradient."""

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return module


def module_is_frozen(module: Any) -> bool:
    """Return whether ``module`` is in eval mode with no trainable parameters."""

    return not bool(module.training) and all(
        not parameter.requires_grad for parameter in module.parameters()
    )


def load_frozen_transformer(
    *,
    checkpoint_path: Path,
    feature_layout_path: Path,
    calibration_path: Path | None = None,
    device: str | Any = "cpu",
    expected_context_dim: int | None = 256,
) -> FrozenTransformerBundle:
    """Load and freeze the selected Transformer without relying on live defaults.

    ``torch.load(..., weights_only=True)`` keeps this adapter limited to tensor
    weights and primitive checkpoint metadata.  The caller is still responsible
    for running the GNN contract preflight, which verifies the frozen selection's
    recorded paths and hashes before this function is reached.
    """

    import torch

    from pipeline.neural_training.model import build_model

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"frozen Transformer checkpoint missing: {checkpoint_path}"
        )
    if not feature_layout_path.is_file():
        raise FileNotFoundError(
            f"frozen Transformer feature layout missing: {feature_layout_path}"
        )
    if calibration_path is not None and not calibration_path.is_file():
        raise FileNotFoundError(
            f"frozen Transformer calibration missing: {calibration_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device(device),
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise ValueError("frozen Transformer checkpoint payload is invalid")

    spec = FeatureLayoutSpec.from_json(feature_layout_path)
    architecture = _architecture_from_checkpoint(checkpoint)
    _validate_checkpoint_layout(checkpoint, spec)
    model = build_model(spec, architecture)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(torch.device(device))
    freeze_module(model)

    context_dim = int(getattr(model, "context_dim", -1))
    if expected_context_dim is not None and context_dim != expected_context_dim:
        raise ValueError(
            "frozen Transformer context dimension mismatch: "
            f"expected {expected_context_dim}, found {context_dim}"
        )

    paths = {
        "checkpoint": checkpoint_path,
        "feature_layout": feature_layout_path,
    }
    if calibration_path is not None:
        paths["calibration"] = calibration_path
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    return FrozenTransformerBundle(
        model=model,
        feature_layout=spec,
        architecture=architecture,
        checkpoint_metadata=metadata,
        artifact_hashes=hashes,
    )


def extract_frozen_outputs(
    bundle: FrozenTransformerBundle,
    batch: Any,
) -> FrozenTransformerOutputs:
    """Return detached context vectors and raw logits for a neural cache batch."""

    import torch

    if not module_is_frozen(bundle.model):
        raise RuntimeError("Transformer must remain frozen during GNN preparation")
    with torch.no_grad():
        context = bundle.model.encode_context(
            numeric=batch.numeric,
            categorical=batch.categorical,
            event_index=batch.event_index,
            event_time=batch.event_time,
            event_value=batch.event_value,
            event_value_mask=batch.event_value_mask,
            event_pad_mask=batch.event_pad_mask,
        )
        logits = bundle.model.score_candidates(
            context=context,
            condition_index=batch.condition_index,
            candidate_index=batch.candidate_index,
            candidate_mask=batch.candidate_mask,
            candidate_side_features=batch.candidate_side_features,
        )
    return FrozenTransformerOutputs(
        context=context.detach(),
        candidate_logits=logits.detach(),
    )


def assert_artifact_hashes_unchanged(
    expected: dict[str, str],
    *,
    checkpoint_path: Path,
    feature_layout_path: Path,
    calibration_path: Path | None = None,
) -> None:
    """Raise if immutable Transformer artifacts changed during a GNN stage."""

    paths = {
        "checkpoint": checkpoint_path,
        "feature_layout": feature_layout_path,
    }
    if calibration_path is not None:
        paths["calibration"] = calibration_path
    changed = sorted(
        name
        for name, path in paths.items()
        if name not in expected
        or not path.is_file()
        or sha256_file(path) != expected[name]
    )
    if changed:
        raise RuntimeError(
            "frozen Transformer artifact changed during GNN work: " + ", ".join(changed)
        )
