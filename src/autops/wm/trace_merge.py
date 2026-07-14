"""Compatibility-checked concatenation for policy-diverse trace corpora."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autops.wm.schema import TraceDataset, TraceMetadata


def concatenate_traces(traces: Sequence[TraceDataset]) -> TraceDataset:
    """Concatenate compatible traces and rebuild the canonical episode axis."""

    if not traces:
        raise ValueError("at least one trace is required")
    first = traces[0]
    first.validate()
    semantic_fields = (
        "mission",
        "observation_names",
        "state_names",
        "action_names",
        "satellite_ids",
        "collective_names",
        "timestep_s",
        "schema_version",
    )
    for index, trace in enumerate(traces[1:], start=1):
        trace.validate()
        if trace.n_steps != first.n_steps:
            raise ValueError(
                f"trace source {index} has {trace.n_steps} steps; expected {first.n_steps}"
            )
        for field_name in semantic_fields:
            if getattr(trace.metadata, field_name) != getattr(first.metadata, field_name):
                raise ValueError(f"trace source {index} has incompatible {field_name}")

    metadata = TraceMetadata(
        mission=first.metadata.mission,
        observation_names=first.metadata.observation_names,
        state_names=first.metadata.state_names,
        action_names=first.metadata.action_names,
        timestep_s=first.metadata.timestep_s,
        sources=tuple(source for trace in traces for source in trace.metadata.sources),
        satellite_ids=first.metadata.satellite_ids,
        collective_names=first.metadata.collective_names,
        schema_version=first.metadata.schema_version,
    )
    episode_count = sum(trace.n_episodes for trace in traces)
    return TraceDataset(
        metadata=metadata,
        obs=np.concatenate([trace.obs for trace in traces], axis=0),
        action=np.concatenate([trace.action for trace in traces], axis=0),
        state=np.concatenate([trace.state for trace in traces], axis=0),
        reward=np.concatenate([trace.reward for trace in traces], axis=0),
        mode=np.concatenate([trace.mode for trace in traces], axis=0),
        resolved_mode=np.concatenate([trace.resolved_mode for trace in traces], axis=0),
        forced_mode=np.concatenate([trace.forced_mode for trace in traces], axis=0),
        episode_seed=np.concatenate([trace.episode_seed for trace in traces], axis=0),
        episode_id=np.arange(episode_count, dtype=np.int64),
        collective={
            name: np.concatenate([trace.collective[name] for trace in traces], axis=0)
            for name in metadata.collective_names
        },
    )


__all__ = ["concatenate_traces"]
