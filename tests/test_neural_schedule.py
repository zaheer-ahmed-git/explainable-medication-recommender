"""Torch-guarded tests for the warmup + cosine learning-rate schedule and EMA."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pipeline.neural_training.train import (  # noqa: E402
    ModelEMA,
    build_warmup_cosine_scheduler,
)


def test_warmup_cosine_schedule_rises_then_decays() -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5)
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
    assert rates[9] == pytest.approx(5e-5, rel=1e-3)
    # Cosine decay finishes near the configured floor.
    assert rates[-1] == pytest.approx(5e-6, rel=5e-2)
    assert rates[-1] < rates[20]


def test_model_ema_tracks_parameter_average() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)
    ema = ModelEMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)
    # shadow = 0.5 * 0 + 0.5 * 2 = 1
    assert float(ema.shadow["weight"].reshape(()).item()) == pytest.approx(1.0)
    target = torch.nn.Linear(1, 1, bias=False)
    ema.copy_to(target)
    assert float(target.weight.reshape(()).item()) == pytest.approx(1.0)
