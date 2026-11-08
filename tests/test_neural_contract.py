"""Tests for the fail-closed neural preflight and gate checks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.neural_training.contract import preflight_errors
from tests.neural_training_helpers import write_neural_fixture


def test_missing_inputs_are_reported(tmp_path: Path) -> None:
    config = write_neural_fixture(tmp_path)
    config = replace(config, features_root=tmp_path / "absent")

    errors = preflight_errors(config, stage="prepare")

    codes = {error["code"] for error in errors}
    assert "missing_input" in codes


def test_train_requires_prepared_cache(tmp_path: Path) -> None:
    config = write_neural_fixture(tmp_path)

    errors = preflight_errors(config, stage="train")

    codes = {error["code"] for error in errors}
    assert "missing_cache_artifact" in codes


def test_gate_contract_mismatch_is_detected(tmp_path: Path) -> None:
    config = write_neural_fixture(tmp_path)
    # Rewrite the gate selection with a mismatched contract digest.
    config.gate_selection_path.write_text(
        '{"status": "frozen", "neural_training_authorized": true,'
        ' "contract_digest": "does-not-match"}',
        encoding="utf-8",
    )

    errors = preflight_errors(config, stage="prepare")

    codes = {error["code"] for error in errors}
    assert "gate_contract_mismatch" in codes


def test_final_mode_requires_frozen_selection(tmp_path: Path) -> None:
    config = write_neural_fixture(tmp_path, mode="final")

    errors = preflight_errors(config, stage="score")

    codes = {error["code"] for error in errors}
    # Final scoring is blocked until a frozen neural development selection and
    # prepared cache exist.
    assert "final_requires_frozen_selection" in codes
    assert "missing_neural_selection" in codes


def test_structured_reference_resolves_stage1_winner(tmp_path: Path) -> None:
    config = write_neural_fixture(tmp_path)
    from pipeline.neural_training.contract import (
        neural_gate_decision,
        resolve_structured_reference,
    )

    resolved = resolve_structured_reference(config)
    assert resolved["baseline_name"] == "xgboost_rank_ndcg_oof_late_fusion"
    assert resolved["anchor_metrics"] is not None
    assert resolved["anchor_metrics"]["ndcg_at_k"] == pytest.approx(0.3946069881534658)

    # Clearing +0.005 over the Stage 1 winner freezes the neural model.
    decision = neural_gate_decision(
        candidate={
            "ndcg_at_k": 0.4000,
            "mrr_at_k": 0.52,
            "hit_rate_at_k": 0.87,
        },
        reference=resolved["anchor_metrics"],
    )
    assert decision["model_frozen"] is True
    assert decision["required_candidate_ndcg_at_10"] == pytest.approx(
        0.3946069881534658 + 0.005
    )

    # Matching the old Milestone 8B bar (0.3799) is no longer enough.
    weak = neural_gate_decision(
        candidate={
            "ndcg_at_k": 0.3800,
            "mrr_at_k": 0.52,
            "hit_rate_at_k": 0.87,
        },
        reference=resolved["anchor_metrics"],
    )
    assert weak["model_frozen"] is False
    assert weak["decision"] == "retain_structured_recovery_baseline"
