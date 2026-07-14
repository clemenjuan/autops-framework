from __future__ import annotations

import numpy as np
import pytest

from autops.wm.artifact import ProbeContract
from autops.wm.jepa import LeWMConfig
from autops.wm.schema import (
    EVENTSAT_OBSERVATIONS,
    EVENTSAT_STATES,
    TraceDataset,
    TraceMetadata,
    TraceSource,
)


def _source() -> TraceSource:
    return TraceSource(
        coordinate="eventsat/sas/ao/symb",
        config_sha256="0" * 64,
        source_revision="0" * 40,
        source_kind="git",
        source_dirty=False,
        orbital_backend="simplified",
        episode_count=1,
        seeds=(1,),
    )


def test_probe_contract_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError, match="non-empty matrix"):
        ProbeContract(
            W=(),
            b=(0.0,),
            attribute_names=("battery_margin",),
            target_mean=(0.0,),
            target_std=(1.0,),
        )


def test_sigreg_requires_two_integration_knots() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        LeWMConfig(sigreg_knots=1)


def test_trace_rejects_fractional_one_hot_action() -> None:
    metadata = TraceMetadata.for_mission("eventsat", timestep_s=60.0, sources=(_source(),))
    action = np.zeros((1, 1, 7), dtype=np.float32)
    action[0, 0, :2] = 0.5
    with pytest.raises(ValueError, match="discrete one-hot"):
        TraceDataset(
            metadata=metadata,
            obs=np.zeros((1, 1, len(EVENTSAT_OBSERVATIONS)), dtype=np.float32),
            action=action,
            state=np.zeros((1, 1, len(EVENTSAT_STATES)), dtype=np.float32),
            reward=np.zeros((1, 1), dtype=np.float32),
            mode=np.zeros((1, 1), dtype=np.int64),
            resolved_mode=np.zeros((1, 1), dtype=np.int64),
            forced_mode=np.zeros((1, 1), dtype=np.float32),
            episode_seed=np.asarray([1], dtype=np.int64),
            episode_id=np.asarray([0], dtype=np.int64),
        )
