"""Deterministic replay of logged EventSat trace episodes.

A trace stores encoded vectors, not the raw decision records and simulator
state that planners and counterfactual checks consume. Replaying an episode's
logged requested commands from its launch seed rebuilds them. Each replayed
record must re-encode to the logged observation exactly, which rejects traces
exported under other settings, another orbital backend, or with planning events
whose energy the trace does not record.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

import numpy as np

from autops.config import expand_coordinate
from autops.core.runner import eventsat_environment
from autops.missions.eventsat.env import EventSatEnvironment
from autops.missions.eventsat.observation import encode_vectors
from autops.wm.schema import EVENTSAT_ACTIONS, TraceDataset


def replay_environments(
    trace: TraceDataset,
    episode: int,
    steps: Iterable[int],
    *,
    planning_horizon: int = 0,
) -> dict[int, EventSatEnvironment]:
    """Return an independent copy of the environment at each requested step, before it acts."""

    if trace.metadata.mission != "eventsat":
        raise ValueError("replay supports EventSat traces")
    wanted = sorted({int(step) for step in steps})
    if not wanted or wanted[0] < 0 or wanted[-1] >= trace.n_steps:
        raise ValueError("replay steps must lie inside the trace episode")
    first = 0
    for source in trace.metadata.sources:
        if episode < first + source.episode_count:
            break
        first += source.episode_count
    else:
        raise ValueError(f"trace has no episode {episode}")
    seed = int(trace.episode_seed[episode])
    spec = expand_coordinate(source.coordinate, steps=trace.n_steps, seeds=[seed])
    env = eventsat_environment(
        spec,
        planning_horizon=planning_horizon,
        prefer_orekit=source.orbital_backend == "orekit",
    )
    observation = env.reset(seed)
    if env.orbit is None or env.orbit.backend != source.orbital_backend:
        raise ValueError(f"replay needs the {source.orbital_backend} orbital backend")
    snapshots: dict[int, EventSatEnvironment] = {}
    for step in range(wanted[-1] + 1):
        if not np.array_equal(encode_vectors(observation)[0], trace.obs[episode, step]):
            raise ValueError(f"replay of episode {episode} diverged from the trace at step {step}")
        if step in wanted:
            snapshots[step] = deepcopy(env)
        if step < wanted[-1]:
            mode = EVENTSAT_ACTIONS[int(trace.mode[episode, step])]
            observation = env.step({"eventsat_0": {"mode": mode}}).observation
    return snapshots


__all__ = ["replay_environments"]
