from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from autops.wm.jepa import LeWMConfig
from autops.wm.schema import EVENTSAT_OBSERVATIONS
from autops.wm.training import TrainingConfig, train_lewm


def train_tiny_lewm(trace: Any) -> Any:
    """Train a one-step LeWM small enough for CPU tests."""

    return train_lewm(
        trace,
        model_config=LeWMConfig(
            obs_dim=len(EVENTSAT_OBSERVATIONS),
            action_dim=7,
            embed_dim=8,
            encoder_hidden_dim=8,
            predictor_depth=1,
            predictor_heads=1,
            predictor_head_dim=8,
            predictor_mlp_dim=16,
            projector_hidden_dim=16,
            dropout=0.0,
            sigreg_knots=3,
            sigreg_projections=4,
        ),
        training_config=TrainingConfig(
            max_steps=1,
            warmup_steps=0,
            batch_size=2,
            train_fraction=0.5,
            seed=17,
            validation_interval=1,
            validation_sample_size=4,
            train_loss_window=1,
        ),
    )


@pytest.fixture(scope="session")
def tiny_lewm() -> Callable[[Any], Any]:
    pytest.importorskip("torch")
    return train_tiny_lewm
