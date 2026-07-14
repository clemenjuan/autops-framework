"""One mission-parameterised exporter for the shared world-model trace schema."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from autops.config import ExperimentSpec, asset_root, runtime_root
from autops.core.provenance import collect_provenance
from autops.core.runner import ExperimentRunner
from autops.core.ssa_runner import _episode_config, _organisation_config
from autops.missions.eventsat.env import EventSatEnvironment
from autops.missions.eventsat.physics import MODES as EVENTSAT_MODES
from autops.missions.eventsat.physics import encode_vectors
from autops.missions.ssa.env import SSAEnvironment
from autops.missions.ssa.policy import SSA_MODES
from autops.organisations.ssa import create_organisation
from autops.wm.schema import (
    SSA_OBSERVATIONS,
    SSA_STATES,
    TraceDataset,
    TraceMetadata,
    TraceSource,
    write_trace,
)
from autops.wm.trace_merge import concatenate_traces


def _trace_source(spec: ExperimentSpec, *, orbital_backend: str) -> TraceSource:
    provenance = collect_provenance(spec.model_dump(mode="json"), asset_root())
    revision = provenance.get("source_revision")
    kind = provenance.get("source_kind")
    dirty = provenance.get("git_dirty")
    if not isinstance(revision, str) or not isinstance(kind, str) or not isinstance(dirty, bool):
        raise RuntimeError("trace export requires immutable source provenance")
    return TraceSource(
        coordinate=spec.coordinate,
        config_sha256=str(provenance["config_sha256"]),
        source_revision=revision,
        source_kind=kind,
        source_dirty=dirty,
        orbital_backend=orbital_backend,
        episode_count=spec.episodes,
        seeds=tuple(spec.seeds),
    )


def _one_hot(mode: str, modes: tuple[str, ...]) -> np.ndarray:
    values = np.zeros(len(modes), dtype=np.float32)
    values[modes.index(mode) if mode in modes else 0] = 1.0
    return values


def _eventsat_trace(spec: ExperimentSpec, prefer_orekit: bool) -> TraceDataset:
    episode_rows: list[dict[str, list[Any]]] = []
    orbital_backends: set[str] = set()
    for seed in spec.seeds:
        env = EventSatEnvironment(
            spec.mission_config,
            max_steps=spec.steps,
            onboard_compute_active=spec.onboard_uses_jetson,
            anomaly_requires_ground_pass=spec.paradigm in {"ag", "conventional"},
            prefer_orekit=prefer_orekit,
        )
        paradigm = ExperimentRunner(spec, save=False, prefer_orekit=prefer_orekit)._build_paradigm()
        observation = env.reset(seed)
        backend = env.episode_provenance().get("orbital_backend")
        if not isinstance(backend, str):
            raise RuntimeError("EventSat trace export requires orbital backend provenance")
        orbital_backends.add(backend)
        paradigm.reset(seed, observation)
        rows: dict[str, list[Any]] = {
            name: [] for name in ("obs", "action", "state", "reward", "mode", "resolved", "forced")
        }
        while int(observation["step"]) < spec.steps:
            obs_vector, state_vector, _ = encode_vectors(observation)
            decision = paradigm.act(observation, physical_contact=env.physical_contact_active())
            requested = str(decision.actions.get("eventsat_0", {}).get("mode", "charging"))
            transition = env.step(decision.actions)
            resolved = str(transition.info.get("resolved_mode", "charging"))
            rows["obs"].append(obs_vector)
            rows["state"].append(state_vector)
            rows["action"].append(_one_hot(requested, EVENTSAT_MODES))
            rows["reward"].append(transition.reward)
            rows["mode"].append(
                EVENTSAT_MODES.index(requested) if requested in EVENTSAT_MODES else 0
            )
            rows["resolved"].append(
                EVENTSAT_MODES.index(resolved) if resolved in EVENTSAT_MODES else 0
            )
            rows["forced"].append(float(bool(transition.info.get("forced", False))))
            paradigm.after_step(transition.info, transition.observation)
            observation = transition.observation
        episode_rows.append(rows)
    if len(orbital_backends) != 1:
        raise ValueError("EventSat trace episodes must use one orbital backend")
    metadata = TraceMetadata.for_mission(
        "eventsat",
        timestep_s=spec.timestep_s,
        sources=(_trace_source(spec, orbital_backend=next(iter(orbital_backends))),),
    )
    return _assemble(metadata, episode_rows, spec.seeds)


def _ssa_vectors(
    observation: dict[str, Any],
    satellite_id: str,
    custody_tau_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    satellite = observation["satellites"][satellite_id]
    global_state = observation["global"]
    target_count = max(1, int(global_state.get("ssa_catalog_size", 0)))
    max_steps = max(1, int(global_state.get("max_steps", 1)))
    mode = str(satellite.get("mode", "charging"))
    observation_vector = np.zeros(len(SSA_OBSERVATIONS), dtype=np.float32)
    observation_vector[:12] = (
        float(satellite.get("battery_soc", 0.0)),
        float(satellite.get("storage_used_fraction", 0.0)),
        float(bool(satellite.get("ground_pass_active", False))),
        min(1.0, float(satellite.get("contact_seconds", 0.0)) / 60.0),
        float(bool(satellite.get("in_sunlight", False))),
        float(satellite.get("health", "nominal") == "nominal"),
        min(1.0, float(satellite.get("unprocessed_batches", 0)) / 10.0),
        min(1.0, float(satellite.get("undelivered_records", 0)) / target_count),
        min(
            1.0,
            float(satellite.get("undelivered_record_age_steps", 0)) / max(1, custody_tau_steps),
        ),
        min(1.0, len(satellite.get("known_objects", [])) / target_count),
        min(1.0, len(satellite.get("ground_view", {})) / target_count),
        min(1.0, len(satellite.get("predicted_in_fov", [])) / target_count),
    )
    observation_vector[12 + (SSA_MODES.index(mode) if mode in SSA_MODES else 0)] = 1.0
    state_vector = np.asarray(
        [
            satellite.get("battery_soc", 0.0),
            SSA_MODES.index(mode) if mode in SSA_MODES else 0,
            float(bool(satellite.get("ground_pass_active", False))),
            satellite.get("contact_seconds", 0.0),
            float(bool(satellite.get("in_sunlight", False))),
            float(satellite.get("health", "nominal") == "nominal"),
            satellite.get("jetson_raw_mb", 0.0),
            satellite.get("jetson_capacity_mb", 0.0),
            satellite.get("unprocessed_batches", 0),
            satellite.get("undelivered_records", 0),
            satellite.get("undelivered_record_age_steps", 0),
            len(satellite.get("known_objects", [])),
            len(satellite.get("ground_view", {})),
            len(satellite.get("predicted_in_fov", [])),
            sum(int(value) for value in satellite.get("detection_row", [])),
            target_count,
            float(observation.get("step", 0)) / max_steps,
            custody_tau_steps,
        ],
        dtype=np.float32,
    )
    return observation_vector, state_vector


def _ssa_trace(spec: ExperimentSpec) -> TraceDataset:
    episode_rows: list[dict[str, list[Any]]] = []
    satellite_ids = tuple(f"sat_{index}" for index in range(spec.constellation_size))
    custody_tau = int(spec.mission_config.get("ssa", {}).get("custody_tau_steps", 4320))
    for seed in spec.seeds:
        env = SSAEnvironment(_episode_config(spec))
        controller = create_organisation(spec.organisation, _organisation_config(spec))
        observation = env.reset(seed)
        controller.reset(seed, observation)
        rows: dict[str, list[Any]] = {
            name: []
            for name in (
                "obs",
                "action",
                "state",
                "reward",
                "mode",
                "resolved",
                "forced",
                "delivered_coverage",
                "onboard_coverage",
                "archive_records",
            )
        }
        while int(observation["step"]) < spec.steps:
            vectors = [
                _ssa_vectors(observation, satellite_id, custody_tau)
                for satellite_id in satellite_ids
            ]
            actions = controller.act(observation)
            transition = env.step(actions)
            requested = transition.info["requested_modes"]
            resolved = transition.info["resolved_modes"]
            requested_indices = [
                SSA_MODES.index(requested[satellite_id]) for satellite_id in satellite_ids
            ]
            resolved_indices = [
                SSA_MODES.index(resolved[satellite_id]) for satellite_id in satellite_ids
            ]
            rows["obs"].append(np.stack([item[0] for item in vectors]))
            rows["state"].append(np.stack([item[1] for item in vectors]))
            rows["action"].append(
                np.stack(
                    [_one_hot(requested[satellite_id], SSA_MODES) for satellite_id in satellite_ids]
                )
            )
            rows["reward"].append([transition.reward] * len(satellite_ids))
            rows["mode"].append(requested_indices)
            rows["resolved"].append(resolved_indices)
            rows["forced"].append(
                [
                    float(left != right)
                    for left, right in zip(requested_indices, resolved_indices, strict=True)
                ]
            )
            rows["delivered_coverage"].append(
                float(observation["global"].get("ssa_delivered_coverage", 0.0))
            )
            rows["onboard_coverage"].append(
                float(observation["global"].get("ssa_onboard_coverage", 0.0))
            )
            rows["archive_records"].append(
                float(observation["global"].get("ground_archive_records", 0.0))
            )
            controller.after_step(transition.info, transition.observation)
            observation = transition.observation
        episode_rows.append(rows)
    metadata = TraceMetadata.for_mission(
        "ssa",
        timestep_s=spec.timestep_s,
        sources=(_trace_source(spec, orbital_backend="not-applicable"),),
        satellite_ids=satellite_ids,
    )
    return _assemble(metadata, episode_rows, spec.seeds)


def _assemble(
    metadata: TraceMetadata,
    episodes: list[dict[str, list[Any]]],
    seeds: list[int],
) -> TraceDataset:
    collective = {
        name: np.asarray([episode[name] for episode in episodes], dtype=np.float32)
        for name in metadata.collective_names
    }
    return TraceDataset(
        metadata=metadata,
        obs=np.asarray([episode["obs"] for episode in episodes], dtype=np.float32),
        action=np.asarray([episode["action"] for episode in episodes], dtype=np.float32),
        state=np.asarray([episode["state"] for episode in episodes], dtype=np.float32),
        reward=np.asarray([episode["reward"] for episode in episodes], dtype=np.float32),
        mode=np.asarray([episode["mode"] for episode in episodes], dtype=np.int64),
        resolved_mode=np.asarray([episode["resolved"] for episode in episodes], dtype=np.int64),
        forced_mode=np.asarray([episode["forced"] for episode in episodes], dtype=np.float32),
        episode_seed=np.asarray(seeds, dtype=np.int64),
        episode_id=np.arange(len(episodes), dtype=np.int64),
        collective=collective,
    )


def export_traces(
    specs: Sequence[ExperimentSpec],
    output: str | Path | None = None,
    *,
    prefer_orekit: bool = True,
) -> Path:
    """Run one or more compatible coordinates into one canonical trace."""

    selected = tuple(specs)
    if not selected:
        raise ValueError("at least one coordinate is required for trace export")
    first = selected[0]
    for index, spec in enumerate(selected[1:], start=1):
        if spec.mission != first.mission:
            raise ValueError(f"coordinate {index} has incompatible mission")
        if spec.steps != first.steps:
            raise ValueError(f"coordinate {index} has incompatible episode steps")
        if spec.timestep_s != first.timestep_s:
            raise ValueError(f"coordinate {index} has incompatible timestep_s")
        if first.mission == "ssa" and spec.constellation_size != first.constellation_size:
            raise ValueError(f"coordinate {index} has incompatible constellation size")
    traces = [
        _eventsat_trace(spec, prefer_orekit) if spec.mission == "eventsat" else _ssa_trace(spec)
        for spec in selected
    ]
    trace = concatenate_traces(traces)
    filename = (
        f"{first.name}.npz" if len(selected) == 1 else f"{first.mission}_mixed_{len(selected)}.npz"
    )
    destination = (
        Path(output) if output is not None else runtime_root() / "data" / "traces" / filename
    )
    return write_trace(destination, trace)


def export_trace(
    spec: ExperimentSpec,
    output: str | Path | None = None,
    *,
    prefer_orekit: bool = True,
) -> Path:
    """Backward-compatible one-coordinate trace export."""

    return export_traces((spec,), output, prefer_orekit=prefer_orekit)


__all__ = ["SSA_OBSERVATIONS", "SSA_STATES", "export_trace", "export_traces"]
