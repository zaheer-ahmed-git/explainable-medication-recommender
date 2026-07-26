"""Torch-guarded tests for the warmup + cosine learning-rate schedule."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pipeline.neural_training.train import build_warmup_cosine_scheduler  # noqa: E402


def test_warmup_cosine_schedule_rises_then_decays() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        steps_per_epoch=10,
        max_epochs=5,
        warmup_epochs=1.0,
        min_lr_ratio=0.1,
    )

    rates = []
    for _ in range(50):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    # Warmup ends near the peak LR after the first epoch of steps.
    assert rates[0] < rates[9]
    assert rates[9] == pytest.approx(1e-4, rel=1e-3)
    # Cosine decay finishes near the configured floor.
    assert rates[-1] == pytest.approx(1e-5, rel=5e-2)
    assert rates[-1] < rates[20]
