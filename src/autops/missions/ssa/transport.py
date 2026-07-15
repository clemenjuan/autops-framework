"""Physical ISL knowledge/record relay and ground delivery."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from autops.missions.ssa.geometry import ground_contact_seconds, link_capacity_bytes
from autops.missions.ssa.model import record_step

if TYPE_CHECKING:
    from autops.missions.ssa.env import SSAEnvironment


def contact_seconds(env: SSAEnvironment, satellite_id: str, start_s: float) -> float:
    ground = env.config["ground_station"]
    return ground_contact_seconds(
        env.satellite_position,
        satellite_id,
        start_s,
        start_s + env.timestep_s,
        ground,
        resolution_s=float(ground["substep_resolution_s"]),
    )


def published_isl_pairs(env: SSAEnvironment) -> list[list[str]]:
    if bool(env.config["constellation"].get("share_plane", False)):
        capacities = env._episode_isl_capacities()
        return [
            [left, right]
            for index, left in enumerate(env.satellite_ids)
            for right in env.satellite_ids[index + 1 :]
            if capacities[(min(left, right), max(left, right))] > 0.0
        ]

    start_s = env.current_step * env.timestep_s
    end_s = start_s + env.timestep_s
    resolution = float(env.config["isl"]["substep_resolution_s"])
    cache: dict[tuple[str, float], tuple[float, float, float]] = {}
    pairs: list[list[str]] = []
    for index, left in enumerate(env.satellite_ids):
        for right in env.satellite_ids[index + 1 :]:
            capacity = link_capacity_bytes(
                env.satellite_position,
                left,
                right,
                start_s,
                end_s,
                env.link_budget,
                resolution_s=resolution,
                cache=cache,
            )
            if capacity > 0.0:
                pairs.append([left, right])
    return pairs


def ground_pass_windows(env: SSAEnvironment) -> list[dict[str, Any]]:
    if bool(env.config["ground_station"]["always_visible"]):
        return [
            {"satellite_id": satellite_id, "start_step": step, "end_step": step}
            for step in range(env.max_steps)
            for satellite_id in env.satellite_ids
        ]
    windows: list[dict[str, Any]] = []
    for satellite_id in env.satellite_ids:
        start: int | None = None
        for step in range(env.max_steps + 1):
            active = (
                step < env.max_steps
                and contact_seconds(env, satellite_id, step * env.timestep_s) > 0.0
            )
            if active and start is None:
                start = step
            elif not active and start is not None:
                windows.append(
                    {
                        "satellite_id": satellite_id,
                        "start_step": start,
                        "end_step": step - 1,
                    }
                )
                start = None
    return windows


def apply_isl(
    env: SSAEnvironment,
    modes: dict[str, str],
    epoch_s: float,
    per_satellite: dict[str, dict[str, Any]],
) -> None:
    sharers = [satellite_id for satellite_id, mode in modes.items() if mode == "isl_share"]
    if not sharers:
        return
    capacities = (
        env._episode_isl_capacities()
        if bool(env.config["constellation"].get("share_plane", False))
        else None
    )
    end_s = epoch_s + env.timestep_s
    resolution = float(env.config["isl"]["substep_resolution_s"])
    cache: dict[tuple[str, float], tuple[float, float, float]] = {}
    for source in sharers:
        feasible: dict[str, float] = {}
        for destination in env.satellite_ids:
            if destination == source:
                continue
            env.stats.isl_attempts += 1
            if modes[destination] not in {"charging", "safe", "isl_share"}:
                continue
            if capacities is not None:
                capacity = capacities[(min(source, destination), max(source, destination))]
            else:
                capacity = link_capacity_bytes(
                    env.satellite_position,
                    source,
                    destination,
                    epoch_s,
                    end_s,
                    env.link_budget,
                    resolution_s=resolution,
                    cache=cache,
                )
            if capacity <= 0.0:
                continue
            env.stats.isl_successes += 1
            feasible[destination] = capacity
            merge_knowledge(env, source, destination)
        per_satellite[source]["isl_feasible_receivers"] = sorted(feasible)
        if not feasible or not bool(env.config["ssa"]["isl_relay"]):
            continue
        destinations = (
            [max(feasible, key=feasible.get)]
            if bool(env.config["isl"]["unicast"])
            else list(feasible)
        )
        for destination in destinations:
            relay_records(env, source, destination, feasible[destination])


def merge_knowledge(env: SSAEnvironment, source: str, destination: str) -> None:
    source_state = env.satellites[source]
    destination_state = env.satellites[destination]
    for index, value in enumerate(source_state.detection_row):
        destination_state.detection_row[index] = max(destination_state.detection_row[index], value)
    for object_id, source_record in source_state.estimates.items():
        destination_state.first_known_steps.setdefault(object_id, env.current_step)
        destination_record = destination_state.estimates.get(object_id)
        if destination_record is None or float(source_record.get("quality", 0.0)) > float(
            destination_record.get("quality", 0.0)
        ):
            merged = deepcopy(source_record)
            merged["last_refresh_step"] = env.current_step
            destination_state.estimates[object_id] = merged
        else:
            destination_record["last_refresh_step"] = env.current_step


def relay_records(
    env: SSAEnvironment,
    source: str,
    destination: str,
    capacity_bytes: float,
) -> None:
    source_buffer = env.satellites[source].undelivered
    destination_buffer = env.satellites[destination].undelivered
    budget = capacity_bytes
    for object_id in sorted(source_buffer, key=lambda key: record_step(source_buffer[key])):
        if budget < env.record_size_bytes:
            break
        record = deepcopy(source_buffer.pop(object_id))
        budget -= env.record_size_bytes
        record["relay_hops"] = int(record.get("relay_hops", 0)) + 1
        env.stats.isl_records_relayed += 1
        env.stats.isl_bytes_transferred += env.record_size_bytes
        held = destination_buffer.get(object_id)
        held_step = record_step(held) if held is not None else -1
        candidate_step = record_step(record)
        if (
            held is None
            or candidate_step > held_step
            or (
                candidate_step == held_step
                and float(record.get("quality", 0.0)) > float(held.get("quality", 0.0))
            )
        ):
            destination_buffer[object_id] = record


def apply_ground_downlinks(
    env: SSAEnvironment,
    modes: dict[str, str],
    action_step: int,
    per_satellite: dict[str, dict[str, Any]],
) -> None:
    contacts: list[str] = []
    for satellite_id, mode in modes.items():
        info = per_satellite[satellite_id]
        if (
            mode != "communication"
            or info["in_transition"]
            or float(info["contact_seconds"]) <= 0.0
        ):
            continue
        runtime = env.satellites[satellite_id]
        for object_id, record in runtime.undelivered.items():
            if not env.ground_archive[object_id]:
                env.stats.record_first_delivery(
                    object_id,
                    action_step,
                    int(record.get("relay_hops", 0)),
                )
            env.ground_archive[object_id].append(deepcopy(record))
        info["downlinked_records"] = len(runtime.undelivered)
        runtime.undelivered = {}
        contacts.append(satellite_id)

    freshest = env._freshest_ground_steps()
    for satellite_id in contacts:
        runtime = env.satellites[satellite_id]
        runtime.ground_catalog_steps = dict(freshest)
        for object_id in freshest:
            runtime.first_known_steps.setdefault(object_id, action_step)
