"""SSA per-satellite encodings for the world-model trace and the RL policy.

Both vectors share one definition of every common feature. The RL vector keeps the
agentic framework's ``ssa_local_compact`` order for the features the lean SSA
environment has, appends the world model's record age and ground-view fraction, and
normalises by constants the satellite record declares (catalog size, custody tau,
orbital period), so an organisation's scoped view carries everything it needs. Pass
and eclipse countdowns are RL-only and present only when the environment publishes them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from autops.missions.ssa.policy import SSA_MODES
from autops.wm.schema import SSA_OBSERVATIONS

SSA_RL_OBSERVATIONS = (
    "battery_soc",
    "storage_used_fraction",
    "ground_pass_active",
    "health_nominal",
    "in_sunlight",
    "known_objects_fraction",
    "undelivered_records_norm",
    "predicted_in_fov_fraction",
    "mean_knowledge_age_norm",
    "remaining_pass_norm",
    "time_to_next_pass_norm",
    "time_to_next_eclipse_norm",
    "unprocessed_batches_norm",
    "detection_progress",
    "has_isl_peer",
    "undelivered_record_age_norm",
    "ground_view_fraction",
    *(f"current_mode_{mode}" for mode in SSA_MODES),
)


def _fraction(value: float, scale: float) -> float:
    return min(1.0, max(0.0, float(value)) / max(float(scale), 1e-12))


def ssa_features(
    satellite: Mapping[str, Any], *, catalog_size: float, custody_tau_steps: float
) -> dict[str, float]:
    """The world model's per-satellite inputs, shared with the RL vector."""

    mode = str(satellite.get("mode", "charging"))
    return {
        "battery_soc": float(satellite.get("battery_soc", 0.0)),
        "storage_used_fraction": float(satellite.get("storage_used_fraction", 0.0)),
        "ground_pass_active": float(bool(satellite.get("ground_pass_active", False))),
        "contact_fraction": _fraction(satellite.get("contact_seconds", 0.0), 60.0),
        "in_sunlight": float(bool(satellite.get("in_sunlight", False))),
        "health_nominal": float(satellite.get("health", "nominal") == "nominal"),
        "unprocessed_batches_norm": _fraction(satellite.get("unprocessed_batches", 0), 10.0),
        "undelivered_records_norm": _fraction(
            satellite.get("undelivered_records", 0), catalog_size
        ),
        "undelivered_record_age_norm": _fraction(
            satellite.get("undelivered_record_age_steps", 0), max(1.0, custody_tau_steps)
        ),
        "known_objects_fraction": _fraction(len(satellite.get("known_objects", [])), catalog_size),
        "ground_view_fraction": _fraction(len(satellite.get("ground_view", {})), catalog_size),
        "predicted_in_fov_fraction": _fraction(
            len(satellite.get("predicted_in_fov", [])), catalog_size
        ),
        **{f"current_mode_{name}": float(name == mode) for name in SSA_MODES},
    }


def encode_ssa_vectors(
    observation: Mapping[str, Any], satellite_id: str, custody_tau_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """World-model observation and privileged state label of one satellite."""

    satellite = observation["satellites"][satellite_id]
    global_state = observation["global"]
    target_count = max(1, int(global_state.get("ssa_catalog_size", 0)))
    max_steps = max(1, int(global_state.get("max_steps", 1)))
    mode = str(satellite.get("mode", "charging"))
    features = ssa_features(
        satellite, catalog_size=target_count, custody_tau_steps=custody_tau_steps
    )
    observation_vector = np.asarray([features[name] for name in SSA_OBSERVATIONS], np.float32)
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


def ssa_rl_vector(record: Mapping[str, Any], satellite_id: str) -> np.ndarray:
    """RL inputs of one satellite in a scoped view; an unseen satellite encodes as zeros."""

    satellite = record.get("satellites", {}).get(satellite_id)
    if satellite is None:
        return np.zeros(len(SSA_RL_OBSERVATIONS), np.float32)
    catalog = max(1.0, float(satellite.get("catalog_size", 1)))
    period = max(1.0, float(satellite.get("orbital_period_steps", 1)))
    max_steps = max(1.0, float(record.get("global", {}).get("max_steps", 1)))
    ages = list(satellite.get("known_object_ages", {}).values())
    values = {
        **ssa_features(
            satellite,
            catalog_size=catalog,
            custody_tau_steps=float(satellite.get("custody_tau_steps", 1)),
        ),
        "mean_knowledge_age_norm": _fraction(sum(ages) / len(ages) if ages else 0.0, max_steps),
        "remaining_pass_norm": _fraction(satellite.get("remaining_pass_duration", 0), period),
        "time_to_next_pass_norm": _fraction(satellite.get("time_to_next_pass", period), period),
        "time_to_next_eclipse_norm": _fraction(
            satellite.get("time_to_next_eclipse", period), period
        ),
        "detection_progress": _fraction(satellite.get("detection_progress", 0.0), 1.0),
        "has_isl_peer": float(bool(satellite.get("has_isl_peer", False))),
    }
    return np.asarray([values[name] for name in SSA_RL_OBSERVATIONS], np.float32)


__all__ = ["SSA_RL_OBSERVATIONS", "encode_ssa_vectors", "ssa_features", "ssa_rl_vector"]
