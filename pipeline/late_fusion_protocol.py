"""Versioned constants and paths for paired-OOF late-fusion selection.

This module is deliberately model-framework-free.  Transformer and GNN jobs
import the same protocol identifiers so their protected prediction artifacts
cannot be combined across incompatible experiments by accident.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

PAIRED_OOF_PROTOCOL_VERSION = "phase8-p0-paired-oof-late-fusion-v2"
PAIRED_OOF_SCHEMA_VERSION = "phase8-p0-paired-oof-predictions-v1"
PAIRED_OOF_SELECTION_SCHEMA_VERSION = "phase8-p0-paired-oof-selection-v1"
FROZEN_GATE_SCHEMA_VERSION = "phase8-p0-frozen-late-fusion-gate-v1"

PRODUCTION_FOLD_COUNT = 5
ALPHA_MIN = Decimal("0.000")
ALPHA_MAX = Decimal("0.250")
ALPHA_STEP = Decimal("0.005")


def alpha_grid() -> tuple[float, ...]:
    """Return the pre-registered inclusive fine grid without float drift."""

    count = int((ALPHA_MAX - ALPHA_MIN) / ALPHA_STEP)
    return tuple(float(ALPHA_MIN + index * ALPHA_STEP) for index in range(count + 1))


def protocol_root(phase8_root: Path) -> Path:
    """Return the restricted artifact root for this protocol version."""

    return Path(phase8_root) / "paired_oof_late_fusion" / PAIRED_OOF_PROTOCOL_VERSION


def transformer_fold_root(neural_root: Path, fold_index: int) -> Path:
    """Return one isolated Transformer fold root outside frozen neural files."""

    return (
        protocol_root(Path(neural_root).parent)
        / "transformer"
        / f"fold_{fold_index:02d}"
    )


def transformer_oof_predictions_path(neural_root: Path, fold_index: int) -> Path:
    return transformer_fold_root(neural_root, fold_index) / "oof_predictions.parquet"


def gnn_variant_oof_predictions_path(gnn_root: Path, variant: str) -> Path:
    return protocol_root(Path(gnn_root).parent) / "gnn" / f"{variant}.parquet"


def paired_selection_artifact_path(gnn_root: Path) -> Path:
    return protocol_root(Path(gnn_root).parent) / "selection" / "late_fusion.json"


def paired_late_checkpoint_path(gnn_root: Path) -> Path:
    return protocol_root(Path(gnn_root).parent) / "checkpoints" / "late_fusion.json"
