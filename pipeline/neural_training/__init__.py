"""Stage 2 conditional neural Transformer training pipeline (Phase 8 P0).

This package implements the gate-first neural branch described in
``Documentation/TrainingPlan.md`` and ``Documentation/TrainingplanDetailed.md``.
It exposes a single ``prepare`` / ``train`` / ``score`` interface with
``development`` and ``final`` modes and reuses the existing training-contract
lock, ranking-metric machinery, canonical score schema, and neural-readiness
gate decision.

Only leakage-reviewed, train-fit inputs are consumed and every public report is
aggregate-only. Real MIMIC training and MIMIC test scoring remain fail-closed
behind the structured recovery gate: neural work is authorized only once
``phase8_p0_gate_recovery_selection.json`` records
``neural_training_authorized = true``.

PyTorch is imported lazily by the model, dataset, and training modules so the
DuckDB cache-preparation step, the preflight gate, the scoring/report helpers,
and the rest of the repository remain importable without the optional ``neural``
dependency group installed.
"""

from __future__ import annotations

from pipeline.neural_training.config import (
    NeuralArchitecture,
    NeuralOptimization,
    NeuralTrainingConfig,
)

__all__ = [
    "NeuralArchitecture",
    "NeuralOptimization",
    "NeuralTrainingConfig",
]
