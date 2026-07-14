"""Detect-gated raw-batch and custody-record pipeline."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from autops.missions.ssa.model import DetectionBatch, record_step
from autops.missions.ssa.targets import OpticalAccess, detection_draw

if TYPE_CHECKING:
    from autops.missions.ssa.env import SSAEnvironment


def queue_observations(
    env: SSAEnvironment,
    modes: dict[str, str],
    target_positions: dict[str, tuple[float, float, float]],
    action_step: int,
    per_satellite: dict[str, dict[str, Any]],
) -> None:
    pending: list[tuple[str, DetectionBatch]] = []
    detection_counts: Counter[str] = Counter()
    observation_mb = float(env.config["storage"]["observation_size_mb"])
    capacity_mb = float(env.config["storage"]["jetson_capacity_mb"])
    for satellite_id, mode in modes.items():
        info = per_satellite[satellite_id]
        info["observation_accepted"] = False
        if mode != "payload_observe" or info["in_transition"]:
            continue
        runtime = env.satellites[satellite_id]
        if runtime.jetson_raw_mb + observation_mb > capacity_mb + 1e-12:
            info["failure_reason"] = "storage_full"
            continue
        known = set(runtime.estimates)
        outcomes: list[dict[str, Any]] = []
        for access in env._accesses(
            satellite_id,
            action_step * env.timestep_s,
            target_positions,
        ):
            draw = detection_draw(env.seed, access.object_id, satellite_id, action_step)
            if draw >= access.probability:
                continue
            ground_step = runtime.ground_catalog_steps.get(access.object_id)
            ground_age = action_step - ground_step if ground_step is not None else None
            outcomes.append(
                {
                    "access": access,
                    "was_cued": access.object_id in known,
                    "ground_fresh": (
                        ground_age is not None and ground_age < env.custody_tau_steps / 4.0
                    ),
                    "simultaneous_duplicate": False,
                }
            )
            detection_counts[access.object_id] += 1
        pending.append(
            (
                satellite_id,
                DetectionBatch(
                    observation_step=action_step,
                    raw_mb=observation_mb,
                    detections=outcomes,
                ),
            )
        )
        info["observation_accepted"] = True

    for satellite_id, batch in pending:
        for outcome in batch.detections:
            access: OpticalAccess = outcome["access"]
            outcome["simultaneous_duplicate"] = detection_counts[access.object_id] > 1
        runtime = env.satellites[satellite_id]
        runtime.pending_batches.append(batch)
        runtime.jetson_raw_mb += batch.raw_mb


def process_detection_batches(
    env: SSAEnvironment,
    modes: dict[str, str],
    action_step: int,
    per_satellite: dict[str, dict[str, Any]],
) -> None:
    required_s = float(env.config["payload"]["detection_time_s"])
    for satellite_id, mode in modes.items():
        runtime = env.satellites[satellite_id]
        info = per_satellite[satellite_id]
        info["detection_completed"] = False
        if mode != "payload_detect" or info["in_transition"]:
            continue
        if not runtime.pending_batches:
            runtime.detection_progress_s = 0.0
            info["failure_reason"] = "no_pending_batch"
            continue
        runtime.detection_progress_s += env.timestep_s
        if runtime.detection_progress_s + 1e-12 < required_s:
            continue
        runtime.detection_progress_s = 0.0
        batch = runtime.pending_batches.pop(0)
        runtime.jetson_raw_mb = max(0.0, runtime.jetson_raw_mb - batch.raw_mb)
        for outcome in batch.detections:
            record_detection(
                env,
                satellite_id,
                outcome["access"],
                observation_step=batch.observation_step,
                known_step=action_step,
                duplicate=bool(outcome["simultaneous_duplicate"] or outcome["ground_fresh"]),
                cued=bool(outcome["was_cued"]),
            )
        info["detection_completed"] = True
        info["detections_revealed"] = [outcome["access"].object_id for outcome in batch.detections]


def record_detection(
    env: SSAEnvironment,
    satellite_id: str,
    access: OpticalAccess,
    *,
    observation_step: int,
    known_step: int,
    duplicate: bool,
    cued: bool,
) -> None:
    target_index = env.target_index.get(access.object_id)
    if target_index is None:
        return
    runtime = env.satellites[satellite_id]
    runtime.detection_row[target_index] = 1
    runtime.first_known_steps.setdefault(access.object_id, known_step)
    env.stats.record_detection(
        access.object_id,
        observation_step,
        duplicate=duplicate,
        cued=cued,
    )
    record = {
        "object_id": access.object_id,
        "satellite_id": satellite_id,
        "position_km": list(access.position_km),
        "obs_step": observation_step,
        "known_step": known_step,
        "last_refresh_step": observation_step,
        "magnitude": access.magnitude,
        "detection_probability": access.probability,
        "quality": access.quality,
        "relay_hops": 0,
    }
    current = runtime.estimates.get(access.object_id)
    if current is None or record["quality"] > float(current.get("quality", 0.0)):
        runtime.estimates[access.object_id] = deepcopy(record)
    else:
        current["last_refresh_step"] = max(
            int(current.get("last_refresh_step", 0)),
            observation_step,
        )
    held = runtime.undelivered.get(access.object_id)
    held_step = record_step(held) if held is not None else -1
    if (
        held is None
        or observation_step > held_step
        or (observation_step == held_step and record["quality"] >= float(held.get("quality", 0.0)))
    ):
        runtime.undelivered[access.object_id] = deepcopy(record)
