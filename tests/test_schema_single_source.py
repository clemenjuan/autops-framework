from __future__ import annotations

import pytest

from autops.missions.eventsat.physics import MODES
from autops.wm.schema import (
    EVENTSAT_ACTIONS,
    EVENTSAT_OBSERVATIONS,
    EVENTSAT_STATES,
    SSA_ACTIONS,
    SSA_OBSERVATIONS,
    SSA_STATES,
    TraceMetadata,
    TraceSource,
)


def _source(mission: str = "eventsat") -> TraceSource:
    return TraceSource(
        coordinate=f"{mission}/sas/ao/symb",
        config_sha256="0" * 64,
        source_revision="0" * 40,
        source_kind="git",
        source_dirty=False,
        orbital_backend="not-applicable" if mission == "ssa" else "simplified",
        episode_count=1,
        seeds=(1,),
    )


def test_eventsat_vocabulary_has_one_canonical_definition() -> None:
    assert MODES is EVENTSAT_ACTIONS
    metadata = TraceMetadata.for_mission("eventsat", timestep_s=60.0, sources=(_source(),))
    assert metadata.action_names == EVENTSAT_ACTIONS
    assert metadata.observation_names == EVENTSAT_OBSERVATIONS
    assert metadata.state_names == EVENTSAT_STATES
    assert metadata.state_names[22] == "remaining_achievable_downlink_mb"


def test_ssa_vocabulary_is_canonical_and_mission_specific() -> None:
    metadata = TraceMetadata.for_mission(
        "ssa", timestep_s=60.0, sources=(_source("ssa"),), satellite_ids=("sat_0",)
    )
    assert metadata.observation_names == SSA_OBSERVATIONS
    assert metadata.state_names == SSA_STATES
    with pytest.raises(ValueError, match="observation names"):
        TraceMetadata(
            mission="ssa",
            observation_names=EVENTSAT_OBSERVATIONS,
            state_names=EVENTSAT_STATES,
            action_names=SSA_ACTIONS,
            timestep_s=60.0,
            sources=(_source("ssa"),),
            satellite_ids=("sat_0",),
        )
