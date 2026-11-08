"""Fail-closed preflight and neural-readiness gate checks.

Stage 2 neural work is authorized only after the Stage 1 structured recovery
gate clears. These checks mirror ``pipeline.gate_recovery.preflight_errors`` so
neural preparation, training, and scoring cannot run against protected data
before the recorded gates pass. This module is PyTorch-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.neural_training.config import (
    DEFAULT_STRUCTURED_REFERENCE_BASELINE,
    LEGACY_MILESTONE8B_XGBOOST_NDCG_AT_10,
    MAXIMUM_SECONDARY_DROP,
    MINIMUM_NDCG_LIFT,
    NeuralTrainingConfig,
)
from pipeline.training_contract import load_json, sha256_file

STAGES = ("prepare", "train", "score")


def _error(code: str, detail: str, **extra: str) -> dict[str, str]:
    row = {"code": code, "detail": detail}
    row.update(extra)
    return row


def resolve_structured_reference(
    config: NeuralTrainingConfig,
) -> dict[str, Any]:
    """Resolve the Stage 1 structured baseline the Transformer must beat.

    Prefer the frozen gate-recovery selection (selected experiment name plus
    locked validation metrics). Fall back to the documented late-fusion name
    when the selection file is incomplete (synthetic fixtures).
    """

    baseline_name = config.reference_baseline_name
    anchor: dict[str, Any] | None = None
    selection: dict[str, Any] = {}
    if config.gate_selection_path.exists():
        try:
            selection = load_json(config.gate_selection_path)
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            selection = {}
    if baseline_name is None:
        selected = selection.get("selected_experiment")
        if isinstance(selected, str) and selected:
            baseline_name = selected
        else:
            baseline_name = DEFAULT_STRUCTURED_REFERENCE_BASELINE

    basis = selection.get("selection_basis", {})
    candidate_metrics = basis.get("candidate_metrics")
    if isinstance(candidate_metrics, dict) and "ndcg_at_k" in candidate_metrics:
        anchor = {
            "ndcg_at_k": float(candidate_metrics["ndcg_at_k"]),
            "mrr_at_k": float(candidate_metrics.get("mrr_at_k", 0.0)),
            "hit_rate_at_k": float(candidate_metrics.get("hit_rate_at_k", 0.0)),
            "baseline_name": baseline_name,
            "source": "stage1_gate_recovery_selection",
        }
    elif selection.get("candidate_ndcg_at_10") is not None:
        anchor = {
            "ndcg_at_k": float(selection["candidate_ndcg_at_10"]),
            "mrr_at_k": 0.0,
            "hit_rate_at_k": 0.0,
            "baseline_name": baseline_name,
            "source": "stage1_gate_recovery_selection_ndcg_only",
        }

    return {
        "baseline_name": baseline_name,
        "anchor_metrics": anchor,
        "scores_path": config.reference_scores_path,
        "legacy_milestone8b_ndcg_at_10": LEGACY_MILESTONE8B_XGBOOST_NDCG_AT_10,
    }


def neural_gate_decision(
    *,
    candidate: dict[str, Any],
    reference: dict[str, Any],
    minimum_ndcg_lift: float = MINIMUM_NDCG_LIFT,
    maximum_secondary_drop: float = MAXIMUM_SECONDARY_DROP,
) -> dict[str, Any]:
    """Apply the neural validation gate against the Stage 1 recovery winner."""

    ndcg_delta = float(candidate["ndcg_at_k"]) - float(reference["ndcg_at_k"])
    mrr_delta = float(candidate["mrr_at_k"]) - float(reference["mrr_at_k"])
    hit_delta = float(candidate["hit_rate_at_k"]) - float(reference["hit_rate_at_k"])
    passed = (
        ndcg_delta >= minimum_ndcg_lift
        and mrr_delta >= -maximum_secondary_drop
        and hit_delta >= -maximum_secondary_drop
    )
    return {
        "decision": "freeze_neural_model"
        if passed
        else "retain_structured_recovery_baseline",
        "model_frozen": passed,
        "primary_metric": "mimic_validation_ndcg_at_10",
        "reference_ndcg_at_10": float(reference["ndcg_at_k"]),
        "required_candidate_ndcg_at_10": float(reference["ndcg_at_k"])
        + minimum_ndcg_lift,
        "candidate_ndcg_at_10": float(candidate["ndcg_at_k"]),
        "ndcg_delta": ndcg_delta,
        "mrr_delta": mrr_delta,
        "hit_rate_delta": hit_delta,
        "minimum_ndcg_lift": minimum_ndcg_lift,
        "maximum_secondary_drop": maximum_secondary_drop,
    }


def required_inputs(config: NeuralTrainingConfig) -> dict[str, Path]:
    """Return the model-ready and reference inputs every stage depends on."""

    return {
        "training_contract_lock": config.contract_lock_path,
        "patient_stay_features": config.patient_stay_features_path,
        "event_sequences": config.event_sequences_path,
        "patient_condition_medication": config.patient_condition_medication_path,
        "candidate_catalog": config.candidate_catalog_path,
    }


def _contract_digest(
    config: NeuralTrainingConfig,
) -> tuple[str | None, list[dict[str, str]]]:
    """Return the locked contract digest and any lock-validation errors."""

    try:
        contract = load_json(config.contract_lock_path)
    except (OSError, json.JSONDecodeError):
        return None, [
            _error(
                "invalid_contract_lock",
                "training contract lock is not valid JSON",
            )
        ]
    if contract.get("status") != "completed" or not contract.get("contract_digest"):
        return None, [
            _error(
                "invalid_contract_lock",
                "training contract lock must be completed and contain a digest",
            )
        ]
    return str(contract["contract_digest"]), []


def neural_gate_errors(
    config: NeuralTrainingConfig,
    *,
    contract_digest: str | None,
) -> list[dict[str, str]]:
    """Return errors unless the structured recovery gate authorizes neural work."""

    if not config.require_neural_gate:
        return []
    if not config.gate_selection_path.exists():
        return [
            _error(
                "missing_gate_selection",
                "structured recovery selection report is missing; the neural "
                "gate has not cleared",
                path=str(config.gate_selection_path),
            )
        ]
    try:
        selection = load_json(config.gate_selection_path)
    except (OSError, json.JSONDecodeError):
        return [
            _error(
                "invalid_gate_selection",
                "structured recovery selection report is not valid JSON",
            )
        ]
    errors: list[dict[str, str]] = []
    if selection.get("status") != "frozen" or not selection.get(
        "neural_training_authorized", False
    ):
        errors.append(
            _error(
                "neural_gate_not_passed",
                "neural training is blocked until the structured recovery gate "
                "records neural_training_authorized=true",
            )
        )
    if (
        contract_digest is not None
        and selection.get("contract_digest") != contract_digest
    ):
        errors.append(
            _error(
                "gate_contract_mismatch",
                "recovery selection does not match the current training contract lock",
            )
        )
    return errors


def _frozen_neural_selection_errors(
    config: NeuralTrainingConfig,
    *,
    contract_digest: str | None,
) -> list[dict[str, str]]:
    """Validate a frozen neural development selection before final scoring."""

    errors: list[dict[str, str]] = []
    if not config.frozen_selection:
        errors.append(
            _error(
                "final_requires_frozen_selection",
                "final mode requires --frozen-selection",
            )
        )
    if not config.selection_report_path.exists():
        errors.append(
            _error(
                "missing_neural_selection",
                "development neural selection report is missing",
                path=str(config.selection_report_path),
            )
        )
        return errors
    try:
        selection = load_json(config.selection_report_path)
    except (OSError, json.JSONDecodeError):
        return [
            _error(
                "invalid_neural_selection",
                "development neural selection is not valid JSON",
            )
        ]
    if selection.get("status") != "frozen" or not selection.get("model_frozen", False):
        errors.append(
            _error(
                "neural_selection_not_frozen",
                "final scoring is blocked until the neural validation gate passes "
                "and the model is frozen",
            )
        )
    if (
        contract_digest is not None
        and selection.get("contract_digest") != contract_digest
    ):
        errors.append(
            _error(
                "neural_selection_contract_mismatch",
                "neural development selection does not match the contract lock",
            )
        )
    frozen = selection.get("frozen_artifacts", {})
    for name, lock in frozen.items():
        try:
            path = Path(lock["path"])
            matches = path.exists() and sha256_file(path) == lock["sha256"]
        except (KeyError, OSError, TypeError, ValueError):
            matches = False
        if not matches:
            errors.append(
                _error(
                    "frozen_neural_artifact_changed",
                    "a frozen neural artifact is missing or changed",
                    artifact_name=name,
                )
            )
    if not frozen:
        errors.append(
            _error(
                "missing_frozen_neural_artifacts",
                "neural development selection does not lock model artifacts",
            )
        )
    return errors


def cache_artifacts(config: NeuralTrainingConfig) -> dict[str, Path]:
    """Return the prepared-cache artifacts that train and score require."""

    return {
        "feature_layout": config.feature_layout_path,
        "event_vocabulary": config.event_vocabulary_path,
        "condition_vocabulary": config.condition_vocabulary_path,
        "candidate_medication_vocabulary": config.candidate_vocabulary_path,
        "categorical_vocabulary": config.categorical_vocabulary_path,
    }


def preflight_errors(
    config: NeuralTrainingConfig,
    *,
    stage: str,
) -> list[dict[str, str]]:
    """Return fail-closed aggregate preflight errors for one stage."""

    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage!r}")

    errors = [
        _error("missing_input", f"{name} does not exist", path=str(path))
        for name, path in required_inputs(config).items()
        if not path.exists()
    ]
    if errors:
        return errors

    contract_digest, lock_errors = _contract_digest(config)
    errors.extend(lock_errors)
    errors.extend(neural_gate_errors(config, contract_digest=contract_digest))

    if stage in {"train", "score"}:
        errors.extend(
            _error(
                "missing_cache_artifact",
                f"{name} is missing; run `prepare` first",
                path=str(path),
            )
            for name, path in cache_artifacts(config).items()
            if not path.exists()
        )

    if stage == "score" and config.mode == "final":
        errors.extend(
            _frozen_neural_selection_errors(config, contract_digest=contract_digest)
        )

    return errors


def contract_digest_or_none(config: NeuralTrainingConfig) -> str | None:
    """Return the locked contract digest, or ``None`` when unavailable."""

    digest, _errors = _contract_digest(config)
    return digest


def blocked_report(
    *,
    schema_version: str,
    stage: str,
    mode: str,
    generated_at: str,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a standard aggregate ``blocked_preflight`` report body."""

    return {
        "schema_version": schema_version,
        "status": "blocked_preflight",
        "stage": stage,
        "mode": mode,
        "generated_at": generated_at,
        "errors": errors,
        "data_safety": {
            "report_contains_patient_rows": False,
            "report_contains_row_samples": False,
        },
    }
