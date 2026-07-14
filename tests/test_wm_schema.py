from __future__ import annotations

import numpy as np
import pytest

from autops.wm.schema import (
    EVENTSAT_ACTIONS,
    EVENTSAT_OBSERVATIONS,
    EVENTSAT_STATES,
    SSA_ACTIONS,
    TraceDataset,
    TraceMetadata,
    TraceSource,
    load_trace,
    write_trace,
)


def _source(seeds: tuple[int, ...] = (42, 43), mission: str = "eventsat") -> TraceSource:
    return TraceSource(
        coordinate=f"{mission}/sas/ao/symb",
        config_sha256="0" * 64,
        source_revision="0" * 40,
        source_kind="git",
        source_dirty=False,
        orbital_backend="not-applicable" if mission == "ssa" else "simplified",
        episode_count=len(seeds),
        seeds=seeds,
    )


def _trace(mission: str) -> TraceDataset:
    metadata = TraceMetadata.for_mission(
        mission,
        timestep_s=60.0,
        sources=(_source(mission=mission),),
        satellite_ids=("sat_0", "sat_1") if mission == "ssa" else (),
    )
    episodes, steps = 2, 4
    prefix = (episodes, steps, 2) if mission == "ssa" else (episodes, steps)
    action = np.zeros((*prefix, len(metadata.action_names)), dtype=np.float32)
    action[..., 0] = 1.0
    collective = {
        name: np.zeros((episodes, steps), dtype=np.float32) for name in metadata.collective_names
    }
    return TraceDataset(
        metadata=metadata,
        obs=np.zeros((*prefix, len(metadata.observation_names)), dtype=np.float32),
        action=action,
        state=np.zeros((*prefix, len(metadata.state_names)), dtype=np.float32),
        reward=np.zeros(prefix, dtype=np.float32),
        mode=np.zeros(prefix, dtype=np.int64),
        resolved_mode=np.zeros(prefix, dtype=np.int64),
        forced_mode=np.zeros(prefix, dtype=np.float32),
        episode_seed=np.array([42, 43], dtype=np.int64),
        episode_id=np.array([0, 1], dtype=np.int64),
        collective=collective,
    )


@pytest.mark.parametrize(
    ("mission", "actions"), (("eventsat", EVENTSAT_ACTIONS), ("ssa", SSA_ACTIONS))
)
def test_trace_roundtrip_derives_action_vocabulary_from_metadata(tmp_path, mission, actions):
    path = write_trace(tmp_path / f"{mission}.npz", _trace(mission))
    loaded = load_trace(path)

    assert loaded.metadata.action_names == actions
    assert loaded.action.shape[-1] == len(actions)
    assert loaded.n_episodes == 2
    assert loaded.n_steps == 4
    assert set(loaded.collective) == set(loaded.metadata.collective_names)


def test_canonical_event_dimensions_are_stable():
    assert len(EVENTSAT_OBSERVATIONS) == 25
    assert len(EVENTSAT_STATES) == 25
    assert len(EVENTSAT_ACTIONS) == 7
    assert len(SSA_ACTIONS) == 6


def test_trace_rejects_stale_ssa_action_contract():
    stale = (
        "charging",
        "payload_observe",
        "payload_compress",
        "payload_detect",
        "payload_send",
        "communication",
        "isl_share",
        "safe",
    )
    with pytest.raises(ValueError, match="action order"):
        TraceMetadata(
            mission="ssa",
            observation_names=EVENTSAT_OBSERVATIONS,
            state_names=EVENTSAT_STATES,
            action_names=stale,
            timestep_s=60.0,
            sources=(_source((42,), mission="ssa"),),
            satellite_ids=("sat_0",),
        )


def test_trace_rejects_non_one_hot_actions():
    trace = _trace("eventsat")
    trace.action[0, 0] = 0.0
    with pytest.raises(ValueError, match="one-hot"):
        trace.validate()


def test_trace_rejects_empty_axes() -> None:
    trace = _trace("eventsat")
    trace.obs = trace.obs[:, :0]
    with pytest.raises(ValueError, match="non-empty"):
        trace.validate()


def test_trace_rejects_noncanonical_episode_ids() -> None:
    trace = _trace("eventsat")
    trace.episode_id[:] = 0
    with pytest.raises(ValueError, match="canonical contiguous"):
        trace.validate()


def test_trace_rejects_action_mode_disagreement() -> None:
    trace = _trace("eventsat")
    trace.mode[0, 0] = 1
    with pytest.raises(ValueError, match="requested one-hot"):
        trace.validate()


def test_trace_rejects_fractional_forced_flags() -> None:
    trace = _trace("eventsat")
    trace.forced_mode[0, 0] = 0.3
    with pytest.raises(ValueError, match="binary"):
        trace.validate()


def test_trace_rejects_fractional_integer_arrays_before_coercion() -> None:
    trace = _trace("eventsat")
    with pytest.raises(ValueError, match="mode must use an integer dtype"):
        TraceDataset(
            metadata=trace.metadata,
            obs=trace.obs,
            action=trace.action,
            state=trace.state,
            reward=trace.reward,
            mode=trace.mode.astype(np.float32) + 0.25,
            resolved_mode=trace.resolved_mode,
            forced_mode=trace.forced_mode,
            episode_seed=trace.episode_seed,
            episode_id=trace.episode_id,
            collective=trace.collective,
        )


def test_trace_rejects_integer_dtype_mutation_during_revalidation() -> None:
    trace = _trace("eventsat")
    trace.mode = trace.mode.astype(np.float32)
    with pytest.raises(ValueError, match="mode must use an integer dtype"):
        trace.validate()
