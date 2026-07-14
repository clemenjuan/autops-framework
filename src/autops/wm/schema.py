"""Versioned, mission-parameterized world-model trace contract.

Rows store the pre-transition observation/state and the requested action.  The
reward, resolved mode, and forced-mode flag describe the transition from that
row; the next row contains the resulting observation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from autops.wm._contract_io import canonical_names as _names
from autops.wm._contract_io import integer_array as _integer_array
from autops.wm.trace_source import TraceSource

TRACE_SCHEMA_VERSION = "autops.world_model.trace/v1"

EVENTSAT_ACTIONS = (
    "charging",
    "communication",
    "payload_observe",
    "payload_compress",
    "payload_detect",
    "payload_send",
    "safe",
)

SSA_ACTIONS = (
    "charging",
    "communication",
    "payload_observe",
    "payload_detect",
    "isl_share",
    "safe",
)

SSA_OBSERVATIONS = (
    "battery_soc",
    "storage_used_fraction",
    "ground_pass_active",
    "contact_fraction",
    "in_sunlight",
    "health_nominal",
    "unprocessed_batches_norm",
    "undelivered_records_norm",
    "undelivered_record_age_norm",
    "known_objects_fraction",
    "ground_view_fraction",
    "predicted_in_fov_fraction",
    *(f"current_mode_{mode}" for mode in SSA_ACTIONS),
)

SSA_STATES = (
    "battery_soc",
    "current_mode_idx",
    "ground_pass_active",
    "contact_seconds",
    "in_sunlight",
    "health_nominal",
    "jetson_raw_mb",
    "jetson_capacity_mb",
    "unprocessed_batches",
    "undelivered_records",
    "undelivered_record_age_steps",
    "known_objects",
    "ground_view_objects",
    "predicted_in_fov_objects",
    "detected_objects",
    "target_count",
    "episode_progress",
    "custody_tau_steps",
)

EVENTSAT_OBSERVATIONS = (
    "battery_soc",
    "obc_fill",
    "jetson_raw_fill",
    "jetson_compressed_fill",
    "orbital_phase_sin",
    "orbital_phase_cos",
    "time_to_next_eclipse_norm",
    "time_to_next_pass_norm",
    "remaining_pass_duration_norm",
    "episode_progress",
    "in_sunlight",
    "ground_pass_active",
    "health_nominal",
    "uncompressed_observations_norm",
    "compression_progress_norm",
    "undetected_observations_norm",
    "detection_progress_norm",
    "downlink_utilization",
    *(f"current_mode_{mode}" for mode in EVENTSAT_ACTIONS),
)

EVENTSAT_STATES = (
    "battery_soc",
    "current_mode_idx",
    "in_sunlight",
    "ground_pass_active",
    "orbital_phase",
    "time_to_next_eclipse",
    "time_to_next_pass",
    "remaining_pass_duration",
    "following_gap_steps",
    "data_stored_mb",
    "obc_data_mb",
    "jetson_raw_mb",
    "jetson_compressed_mb",
    "data_downlinked_mb",
    "uncompressed_observations",
    "compression_progress",
    "undetected_observations",
    "detection_progress",
    "total_observation_s",
    "total_detections",
    "storage_capacity_mb",
    "jetson_capacity_mb",
    "remaining_achievable_downlink_mb",
    "achievable_downlink_mb",
    "health_nominal",
)

SSA_COLLECTIVE_FIELDS = (
    "delivered_coverage",
    "onboard_coverage",
    "archive_records",
)

_MISSION_ACTIONS = {"eventsat": EVENTSAT_ACTIONS, "ssa": SSA_ACTIONS}
_MISSION_OBSERVATIONS = {"eventsat": EVENTSAT_OBSERVATIONS, "ssa": SSA_OBSERVATIONS}
_MISSION_STATES = {"eventsat": EVENTSAT_STATES, "ssa": SSA_STATES}


@dataclass(frozen=True)
class TraceMetadata:
    """Names and axes needed to interpret a trace without importing a mission."""

    mission: str
    observation_names: tuple[str, ...]
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    timestep_s: float
    sources: tuple[TraceSource, ...]
    satellite_ids: tuple[str, ...] = ()
    collective_names: tuple[str, ...] = ()
    schema_version: str = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError(f"unsupported trace schema {self.schema_version!r}")
        if self.mission not in _MISSION_ACTIONS:
            raise ValueError(f"unsupported world-model mission {self.mission!r}")
        if (
            isinstance(self.timestep_s, bool)
            or not isinstance(self.timestep_s, (int, float))
            or not math.isfinite(self.timestep_s)
            or self.timestep_s <= 0.0
        ):
            raise ValueError("trace timestep_s must be finite and positive")
        object.__setattr__(self, "timestep_s", float(self.timestep_s))
        sources = tuple(self.sources)
        if not sources or any(not isinstance(source, TraceSource) for source in sources):
            raise ValueError("trace metadata requires at least one TraceSource")
        if any(source.coordinate.split("/", 1)[0] != self.mission for source in sources):
            raise ValueError("trace source coordinates must match the metadata mission")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(
            self, "observation_names", _names(self.observation_names, "observation_names")
        )
        object.__setattr__(self, "state_names", _names(self.state_names, "state_names"))
        object.__setattr__(self, "action_names", _names(self.action_names, "action_names"))
        object.__setattr__(self, "satellite_ids", tuple(str(v) for v in self.satellite_ids))
        object.__setattr__(self, "collective_names", tuple(str(v) for v in self.collective_names))
        if self.action_names != _MISSION_ACTIONS[self.mission]:
            raise ValueError(
                f"{self.mission} action order must be {_MISSION_ACTIONS[self.mission]}, "
                f"got {self.action_names}"
            )
        if self.observation_names != _MISSION_OBSERVATIONS[self.mission]:
            raise ValueError(f"{self.mission} observation names are not canonical")
        if self.state_names != _MISSION_STATES[self.mission]:
            raise ValueError(f"{self.mission} state names are not canonical")
        if self.mission == "eventsat" and self.satellite_ids:
            raise ValueError("EventSat traces do not carry a satellite axis")
        if self.mission == "ssa" and not self.satellite_ids:
            raise ValueError("SSA traces require satellite_ids")
        if len(set(self.satellite_ids)) != len(self.satellite_ids):
            raise ValueError("satellite_ids must not contain duplicates")
        if len(set(self.collective_names)) != len(self.collective_names):
            raise ValueError("collective_names must not contain duplicates")

    @classmethod
    def for_mission(
        cls,
        mission: str,
        *,
        timestep_s: float,
        sources: tuple[TraceSource, ...],
        satellite_ids: tuple[str, ...] = (),
        observation_names: tuple[str, ...] | None = None,
        state_names: tuple[str, ...] | None = None,
    ) -> TraceMetadata:
        """Build metadata from the canonical vocabulary of a supported mission."""

        if mission not in _MISSION_ACTIONS:
            raise ValueError(f"unsupported world-model mission {mission!r}")
        observations = (
            _MISSION_OBSERVATIONS[mission] if observation_names is None else observation_names
        )
        states = _MISSION_STATES[mission] if state_names is None else state_names
        collective = SSA_COLLECTIVE_FIELDS if mission == "ssa" else ()
        return cls(
            mission=mission,
            observation_names=observations,
            state_names=states,
            action_names=_MISSION_ACTIONS[mission],
            timestep_s=timestep_s,
            sources=sources,
            satellite_ids=satellite_ids,
            collective_names=collective,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mission": self.mission,
            "observation_names": list(self.observation_names),
            "state_names": list(self.state_names),
            "action_names": list(self.action_names),
            "timestep_s": self.timestep_s,
            "sources": [source.to_dict() for source in self.sources],
            "satellite_ids": list(self.satellite_ids),
            "collective_names": list(self.collective_names),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TraceMetadata:
        if not isinstance(payload, Mapping):
            raise ValueError("trace metadata must be a mapping")
        allowed = {
            "schema_version",
            "mission",
            "observation_names",
            "state_names",
            "action_names",
            "timestep_s",
            "sources",
            "satellite_ids",
            "collective_names",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown trace metadata fields: {sorted(unknown)}")
        missing = allowed - set(payload)
        if missing:
            raise ValueError(f"missing trace metadata fields: {sorted(missing)}")
        if not isinstance(payload["sources"], list):
            raise ValueError("trace metadata sources must be an array")
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            mission=str(payload.get("mission", "")),
            observation_names=tuple(payload.get("observation_names", ())),
            state_names=tuple(payload.get("state_names", ())),
            action_names=tuple(payload.get("action_names", ())),
            timestep_s=payload["timestep_s"],
            sources=tuple(TraceSource.from_dict(source) for source in payload["sources"]),
            satellite_ids=tuple(payload.get("satellite_ids", ())),
            collective_names=tuple(payload.get("collective_names", ())),
        )


@dataclass
class TraceDataset:
    """A validated collection of fixed-length mission episodes."""

    metadata: TraceMetadata
    obs: np.ndarray
    action: np.ndarray
    state: np.ndarray
    reward: np.ndarray
    mode: np.ndarray
    resolved_mode: np.ndarray
    forced_mode: np.ndarray
    episode_seed: np.ndarray
    episode_id: np.ndarray
    collective: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.obs = np.asarray(self.obs, dtype=np.float32)
        self.action = np.asarray(self.action, dtype=np.float32)
        self.state = np.asarray(self.state, dtype=np.float32)
        self.reward = np.asarray(self.reward, dtype=np.float32)
        self.mode = _integer_array(self.mode, "mode")
        self.resolved_mode = _integer_array(self.resolved_mode, "resolved_mode")
        self.forced_mode = np.asarray(self.forced_mode, dtype=np.float32)
        self.episode_seed = _integer_array(self.episode_seed, "episode_seed")
        self.episode_id = _integer_array(self.episode_id, "episode_id")
        self.collective = {str(k): np.asarray(v) for k, v in self.collective.items()}
        self.validate()

    @property
    def n_episodes(self) -> int:
        return int(self.obs.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.obs.shape[1])

    @property
    def transition_prefix(self) -> tuple[int, ...]:
        base = (self.n_episodes, self.n_steps)
        if self.metadata.mission == "ssa":
            return (*base, len(self.metadata.satellite_ids))
        return base

    def validate(self) -> None:
        for name in ("mode", "resolved_mode", "episode_seed", "episode_id"):
            if not np.issubdtype(np.asarray(getattr(self, name)).dtype, np.integer):
                raise ValueError(f"{name} must use an integer dtype")
        expected_rank = 4 if self.metadata.mission == "ssa" else 3
        if self.obs.ndim != expected_rank or any(size < 1 for size in self.obs.shape[:2]):
            raise ValueError("trace requires non-empty episode and time axes")
        prefix = self.transition_prefix
        expected = {
            "obs": (*prefix, len(self.metadata.observation_names)),
            "action": (*prefix, len(self.metadata.action_names)),
            "state": (*prefix, len(self.metadata.state_names)),
            "reward": prefix,
            "mode": prefix,
            "resolved_mode": prefix,
            "forced_mode": prefix,
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        episode_shape = (self.n_episodes,)
        if self.episode_seed.shape != episode_shape or self.episode_id.shape != episode_shape:
            raise ValueError("episode_seed and episode_id must have one value per episode")
        if sum(source.episode_count for source in self.metadata.sources) != self.n_episodes:
            raise ValueError("trace source episode counts must cover every episode exactly once")
        source_seeds = tuple(seed for source in self.metadata.sources for seed in source.seeds)
        if source_seeds != tuple(int(seed) for seed in self.episode_seed):
            raise ValueError("trace source seeds must match episode_seed in source order")
        if not np.array_equal(self.episode_id, np.arange(self.n_episodes, dtype=np.int64)):
            raise ValueError("episode_id must be the canonical contiguous range")
        if np.any(self.episode_seed < 0):
            raise ValueError("episode_seed values must be non-negative")
        if not np.allclose(self.action.sum(axis=-1), 1.0, atol=1e-5):
            raise ValueError("action must be one-hot over metadata.action_names")
        if not np.allclose(self.action, np.round(self.action), atol=1e-5):
            raise ValueError("action entries must be discrete one-hot values")
        if np.any((self.action < 0.0) | (self.action > 1.0)):
            raise ValueError("one-hot action entries must lie in [0, 1]")
        if not np.array_equal(self.mode, np.argmax(self.action, axis=-1)):
            raise ValueError("mode must equal the requested one-hot action index")
        action_dim = len(self.metadata.action_names)
        for name in ("mode", "resolved_mode"):
            value = getattr(self, name)
            if np.any((value < 0) | (value >= action_dim)):
                raise ValueError(f"{name} contains an invalid action index")
        if not np.array_equal(self.forced_mode, np.round(self.forced_mode)) or np.any(
            (self.forced_mode < 0.0) | (self.forced_mode > 1.0)
        ):
            raise ValueError("forced_mode must be binary")
        expected_collective = set(self.metadata.collective_names)
        if set(self.collective) != expected_collective:
            raise ValueError(
                "collective fields must match metadata: "
                f"expected {sorted(expected_collective)}, got {sorted(self.collective)}"
            )
        for name, value in self.collective.items():
            if value.shape != (self.n_episodes, self.n_steps):
                raise ValueError(f"collective field {name} must have shape [episode, time]")
            if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                raise ValueError(f"collective field {name} must be finite and numeric")


_ARRAY_KEYS = {
    "obs",
    "action",
    "state",
    "reward",
    "mode",
    "resolved_mode",
    "forced_mode",
    "episode_seed",
    "episode_id",
}
_METADATA_KEY = "__metadata_json__"
_COLLECTIVE_PREFIX = "collective__"


def _hash_array(hasher: Any, name: str, values: np.ndarray) -> None:
    array = np.asarray(values)
    dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
    header = json.dumps(
        {"name": name, "dtype": dtype.str, "shape": list(canonical.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    hasher.update(len(header).to_bytes(8, "big"))
    hasher.update(header)
    payload = canonical.tobytes(order="C")
    hasher.update(len(payload).to_bytes(8, "big"))
    hasher.update(payload)


def trace_sha256(trace: TraceDataset) -> str:
    """Hash every trace value and semantic field in a canonical byte order."""

    trace.validate()
    hasher = hashlib.sha256()
    metadata = json.dumps(trace.metadata.to_dict(), separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    hasher.update(len(metadata).to_bytes(8, "big"))
    hasher.update(metadata)
    for name in sorted(_ARRAY_KEYS):
        _hash_array(hasher, name, getattr(trace, name))
    for name in sorted(trace.collective):
        _hash_array(hasher, _COLLECTIVE_PREFIX + name, trace.collective[name])
    return hasher.hexdigest()


def write_trace(path_like: str | Path, trace: TraceDataset) -> Path:
    """Write a trace without pickle; all interpretation data travels in metadata."""

    trace.validate()
    path = Path(path_like)
    if path.suffix != ".npz":
        raise ValueError("world-model traces must use the .npz suffix")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {name: getattr(trace, name) for name in sorted(_ARRAY_KEYS)}
    arrays[_METADATA_KEY] = np.asarray(json.dumps(trace.metadata.to_dict(), sort_keys=True))
    arrays.update({_COLLECTIVE_PREFIX + name: value for name, value in trace.collective.items()})
    np.savez_compressed(path, **arrays)
    return path


def load_trace(path_like: str | Path) -> TraceDataset:
    """Load and strictly validate a versioned EventSat or SSA trace."""

    path = Path(path_like)
    with np.load(path, allow_pickle=False) as blob:
        missing = (_ARRAY_KEYS | {_METADATA_KEY}) - set(blob.files)
        if missing:
            raise ValueError(f"trace {path} is missing arrays: {sorted(missing)}")
        metadata = TraceMetadata.from_dict(json.loads(str(blob[_METADATA_KEY].item())))
        collective_keys = {
            key.removeprefix(_COLLECTIVE_PREFIX): key
            for key in blob.files
            if key.startswith(_COLLECTIVE_PREFIX)
        }
        known = _ARRAY_KEYS | {_METADATA_KEY} | set(collective_keys.values())
        unknown = set(blob.files) - known
        if unknown:
            raise ValueError(f"trace {path} contains unknown arrays: {sorted(unknown)}")
        values = {name: blob[name].copy() for name in _ARRAY_KEYS}
        collective = {name: blob[key].copy() for name, key in collective_keys.items()}
    return TraceDataset(metadata=metadata, collective=collective, **values)


__all__ = [
    "EVENTSAT_ACTIONS",
    "EVENTSAT_OBSERVATIONS",
    "EVENTSAT_STATES",
    "SSA_ACTIONS",
    "SSA_COLLECTIVE_FIELDS",
    "SSA_OBSERVATIONS",
    "SSA_STATES",
    "TRACE_SCHEMA_VERSION",
    "TraceDataset",
    "TraceMetadata",
    "TraceSource",
    "load_trace",
    "trace_sha256",
    "write_trace",
]
