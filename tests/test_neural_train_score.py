"""Torch-guarded end-to-end smoke test: prepare -> train -> score."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.neural_training.config import NeuralOptimization
from pipeline.neural_training.data import prepare_neural_caches
from tests.neural_training_helpers import write_neural_fixture

pytest.importorskip("torch")

from pipeline.neural_training.score import score_transformer  # noqa: E402
from pipeline.neural_training.train import train_transformer  # noqa: E402


def _fast_config(tmp_path: Path):
    config = write_neural_fixture(
        tmp_path,
        train_stays=6,
        validation_stays=4,
        require_neural_gate=False,
        gate_authorized=False,
    )
    return replace(
        config,
        device="cpu",
        optimization=replace(
            NeuralOptimization(),
            max_epochs=2,
            early_stopping_patience=2,
            batch_ranking_groups=4,
            mixed_precision=False,
        ),
    )


def test_train_then_score_smoke(tmp_path: Path) -> None:
    config = _fast_config(tmp_path)
    assert prepare_neural_caches(config)["status"] == "completed"

    training = train_transformer(config)
    assert training["status"] == "completed"
    assert config.checkpoint_path.exists()
    assert config.calibration_path.exists()
    assert training["epoch_history"]
    assert "validation_ndcg_at_10" in training["epoch_history"][0]

    scoring = score_transformer(config)
    assert scoring["status"] == "completed"
    assert config.score_output_path.exists()
    assert config.selection_report_path.exists()
    assert "ranking_metrics" in scoring
    assert isinstance(scoring.get("model_frozen"), bool)


def test_train_is_blocked_without_prepared_cache(tmp_path: Path) -> None:
    config = _fast_config(tmp_path)
    # No prepare() call: the cache artifacts are missing.
    report = train_transformer(config)
    assert report["status"] == "blocked_preflight"
    codes = {error["code"] for error in report["errors"]}
    assert "missing_cache_artifact" in codes
