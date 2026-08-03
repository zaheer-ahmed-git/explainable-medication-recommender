"""Synthetic fail-closed tests for canonical GNN score materialization."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
import torch

from pipeline.gnn_training.config import GNNTrainingConfig
from pipeline.gnn_training.data import write_json_exclusive
from pipeline.gnn_training.runtime import (
    TemperatureGrid,
    backward_optimizer_step,
    read_positive_temperature,
)
from pipeline.gnn_training.scoring import materialize_canonical_scores
from tests.milestone6_helpers import write_parquet_rows


def _config(tmp_path: Path) -> GNNTrainingConfig:
    return replace(
        GNNTrainingConfig(),
        training_root=tmp_path / "training",
        duckdb_temp_directory=None,
        duckdb_memory_limit=None,
        duckdb_threads=None,
    )


def _write_candidates(config: GNNTrainingConfig) -> None:
    write_parquet_rows(
        config.patient_condition_medication_path,
        (
            "source",
            "split",
            "ranking_group_id",
            "index_condition_token",
            "candidate_medication_token",
            "candidate_rank",
            "label_prescribed",
        ),
        (
            ("mimiciv", "validation", "group-a", "condition:a", "rxnorm:1", 1, True),
            (
                "mimiciv",
                "validation",
                "group-a",
                "condition:a",
                "rxnorm:2",
                2,
                False,
            ),
        ),
    )


def test_canonical_scores_reject_a_strict_prediction_subset(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_candidates(config)
    predictions = tmp_path / "predictions.parquet"
    write_parquet_rows(
        predictions,
        ("ranking_group_id", "candidate_medication_token", "score"),
        (("group-a", "rxnorm:1", 0.9),),
    )

    with (
        duckdb.connect(database=":memory:") as connection,
        pytest.raises(ValueError, match="exactly match"),
    ):
        materialize_canonical_scores(
            connection,
            config,
            predictions_path=predictions,
            output_path=tmp_path / "scores.parquet",
            split="validation",
            baseline_name="synthetic_gnn",
            baseline_version="synthetic-v1",
            evaluation_version="synthetic-eval-v1",
            generated_at="2026-01-01T00:00:00+00:00",
        )


def test_temperature_artifact_rejects_non_object_json(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text("[1.0]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed"):
        read_positive_temperature(
            calibration,
            expected_schema_version="synthetic-schema",
            allowed_methods=frozenset({"synthetic-method"}),
        )


def test_temperature_artifact_rejects_missing_temperature(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        '{"schema_version":"synthetic-schema",'
        '"method":"synthetic-method",'
        '"fit_split":"mimiciv_validation"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="malformed"):
        read_positive_temperature(
            calibration,
            expected_schema_version="synthetic-schema",
            allowed_methods=frozenset({"synthetic-method"}),
        )


def test_temperature_grid_rejects_nonfinite_logits() -> None:
    grid = TemperatureGrid(torch.device("cpu"), points=3)

    with pytest.raises(FloatingPointError, match="non-finite"):
        grid.update(
            torch.tensor([[float("nan")]], dtype=torch.float32),
            torch.tensor([[1.0]], dtype=torch.float32),
            torch.tensor([[True]]),
        )


class _FiniteForwardInfiniteBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
        del ctx
        return value * 0.0 + 1.0

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> torch.Tensor:
        del ctx
        return torch.full_like(gradient, float("inf"))


class _OverflowAboveScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
        del ctx
        return value * 0.0 + 1.0

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> torch.Tensor:
        del ctx
        if float(gradient.abs().max()) > 32_768.0:
            return torch.full_like(gradient, float("inf"))
        return gradient


def test_amp_optimizer_step_backs_off_and_skips_nonfinite_gradient() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=65_536.0)
    before = model.weight.detach().clone()
    loss = _FiniteForwardInfiniteBackward.apply(model.weight).sum()

    result = backward_optimizer_step(
        loss=loss,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        gradient_clip_norm=1.0,
    )

    assert not result.optimizer_step_applied
    assert result.gradient_norm is None
    assert result.nonfinite_parameter_names == ("weight",)
    assert result.loss_scale_before == 65_536.0
    assert result.loss_scale_after == 32_768.0
    assert torch.equal(model.weight.detach(), before)
    assert model.weight.grad is None


def test_amp_optimizer_step_succeeds_when_retried_at_lower_scale() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=65_536.0)
    before = model.weight.detach().clone()

    first = backward_optimizer_step(
        loss=_OverflowAboveScale.apply(model.weight).sum(),
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        gradient_clip_norm=1.0,
    )
    second = backward_optimizer_step(
        loss=_OverflowAboveScale.apply(model.weight).sum(),
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        gradient_clip_norm=1.0,
    )

    assert not first.optimizer_step_applied
    assert first.loss_scale_after == 32_768.0
    assert second.optimizer_step_applied
    assert second.loss_scale_before == 32_768.0
    assert not torch.equal(model.weight.detach(), before)


def test_full_precision_optimizer_step_rejects_nonfinite_gradient() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = _FiniteForwardInfiniteBackward.apply(model.weight).sum()

    with pytest.raises(FloatingPointError, match="weight"):
        backward_optimizer_step(
            loss=loss,
            model=model,
            optimizer=optimizer,
            scaler=None,
            gradient_clip_norm=1.0,
        )


def test_optimizer_step_clips_and_applies_finite_gradient() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    loss = model.weight.square().sum()

    result = backward_optimizer_step(
        loss=loss,
        model=model,
        optimizer=optimizer,
        scaler=None,
        gradient_clip_norm=1.0,
    )

    assert result.optimizer_step_applied
    assert result.gradient_norm is not None
    assert math.isfinite(result.gradient_norm)
    assert not torch.equal(model.weight.detach(), before)


def test_final_score_claim_is_exclusive(tmp_path: Path) -> None:
    marker = tmp_path / "final-score.json"
    write_json_exclusive(marker, {"status": "running"})

    with pytest.raises(FileExistsError):
        write_json_exclusive(marker, {"status": "running"})
