"""Physical ISL knowledge/record relay and ground delivery."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from autops.missions.ssa.geometry import ground_contact_seconds, link_capacity_bytes
from autops.missions.ssa.model import (
    custody_record_is_better,
    estimate_is_better,
    record_step,
)

if TYPE_CHECKING:
    from autops.missions.ssa.env import SSAEnvironment


def contact_seconds(env: SSAEnvironment, satellite_id: str, start_s: float) -> float:
    ground = env.config["communications"]["ground_station"]
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
    if bool(env.config["communications"]["ground_station"]["always_visible"]):
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
    """Share knowledge and relay custody records, one hop per step.

    Every sharer's knowledge and custody buffer is snapshotted before any
    transfer, relays are planned from the snapshots, and both are committed
    afterwards, so nothing received this step is forwarded this step
    (agentic framework d31385d). Receivers must be authorised, idle, and
    physically reachable.
    """

    sharers = [
        satellite_id for satellite_id in env.satellite_ids if modes[satellite_id] == "isl_share"
    ]
    if not sharers:
        return
    knowledge = {
        source: (
            list(env.satellites[source].detection_row),
            deepcopy(env.satellites[source].estimates),
        )
        for source in sharers
    }
    custody = {source: deepcopy(env.satellites[source].undelivered) for source in sharers}
    feasible = {source: _feasible_receivers(env, source, modes, epoch_s) for source in sharers}
    relays: list[tuple[str, str, str, dict[str, Any]]] = []
    for source in sharers:
        per_satellite[source]["isl_feasible_receivers"] = sorted(feasible[source])
        if feasible[source] and bool(env.config["ssa"]["isl_relay"]):
            relays.extend(_plan_relays(env, source, custody[source], feasible[source]))
    for source in sharers:
        for destination in sorted(feasible[source]):
            merge_knowledge(env, knowledge[source], destination)
    _commit_relays(env, relays)


def _feasible_receivers(
    env: SSAEnvironment, source: str, modes: dict[str, str], epoch_s: float
) -> dict[str, float]:
    capacities = (
        env._episode_isl_capacities()
        if bool(env.config["constellation"].get("share_plane", False))
        else None
    )
    resolution = float(env.config["isl"]["substep_resolution_s"])
    cache: dict[tuple[str, float], tuple[float, float, float]] = {}
    feasible: dict[str, float] = {}
    for destination in env.authorized_isl_destinations(source):
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
                epoch_s + env.timestep_s,
                env.link_budget,
                resolution_s=resolution,
                cache=cache,
            )
        if capacity > 0.0:
            env.stats.isl_successes += 1
            feasible[destination] = capacity
    return feasible


def merge_knowledge(
    env: SSAEnvironment,
    snapshot: tuple[list[int], dict[str, dict[str, Any]]],
    destination: str,
) -> None:
    """Merge a sender's snapshot; received estimates keep their own acquisition age."""

    row, estimates = snapshot
    destination_state = env.satellites[destination]
    for index, value in enumerate(row):
        destination_state.detection_row[index] = max(destination_state.detection_row[index], value)
    for object_id, estimate in estimates.items():
        destination_state.first_known_steps.setdefault(object_id, env.current_step)
        current = destination_state.estimates.get(object_id)
        if current is None or estimate_is_better(estimate, current):
            destination_state.estimates[object_id] = deepcopy(estimate)


def _plan_relays(
    env: SSAEnvironment,
    source: str,
    buffer: dict[str, dict[str, Any]],
    feasible: dict[str, float],
) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Allocate snapshot records to receivers within each link's byte budget."""

    if bool(env.config["isl"]["unicast"]):
        best = min(feasible.items(), key=lambda item: (-item[1], item[0]))
        targets = [best]
    else:
        targets = sorted(feasible.items())
    available = sorted(buffer, key=lambda object_id: (record_step(buffer[object_id]), object_id))
    plans: list[tuple[str, str, str, dict[str, Any]]] = []
    for destination, capacity in targets:
        budget = capacity
        while available and budget >= env.record_size_bytes:
            object_id = available.pop(0)
            plans.append((source, destination, object_id, buffer[object_id]))
            budget -= env.record_size_bytes
    return plans


def _commit_relays(env: SSAEnvironment, plans: list[tuple[str, str, str, dict[str, Any]]]) -> None:
    for source, _, object_id, _ in plans:
        env.satellites[source].undelivered.pop(object_id, None)
    for _, destination, object_id, snapshot in plans:
        record = deepcopy(snapshot)
        record["relay_hops"] = int(record.get("relay_hops", 0)) + 1
        env.stats.isl_records_relayed += 1
        env.stats.isl_bytes_transferred += env.record_size_bytes
        buffer = env.satellites[destination].undelivered
        held = buffer.get(object_id)
        if held is None or custody_record_is_better(record, held):
            buffer[object_id] = record


def apply_ground_downlinks(
    env: SSAEnvironment,
    modes: dict[str, str],
    action_step: int,
    per_satellite: dict[str, dict[str, Any]],
) -> None:
    contacts: list[str] = []
    for satellite_id, mode in modes.items():
        info = per_satellite[satellite_id]
        if mode != "communication" or info["in_transition"]:
            continue
        if float(info["contact_seconds"]) <= 0.0:
            # A radio attempt without contact pays power and delivers nothing.
            info["failure_reason"] = "no_contact"
            continue
        runtime = env.satellites[satellite_id]
        budget = (
            float(env.config["communications"]["xband"]["downlink_rate_kbps"])
            * 1000.0
            / 8.0
            * float(info["contact_seconds"])
        )
        delivered = 0
        for object_id in sorted(
            runtime.undelivered, key=lambda key: record_step(runtime.undelivered[key])
        ):
            if budget < env.record_size_bytes:
                break
            record = runtime.undelivered.pop(object_id)
            budget -= env.record_size_bytes
            delivered += 1
            if not env.ground_archive[object_id]:
                env.stats.record_first_delivery(
                    object_id,
                    action_step,
                    int(record.get("relay_hops", 0)),
                )
            env.ground_archive[object_id].append(deepcopy(record))
        info["downlinked_records"] = delivered
        if not delivered:
            info["failure_reason"] = "no_source_data"
        contacts.append(satellite_id)

    freshest = env._freshest_ground_steps()
    for satellite_id in contacts:
        runtime = env.satellites[satellite_id]
        runtime.ground_catalog_steps = dict(freshest)
        for object_id in freshest:
            runtime.first_known_steps.setdefault(object_id, action_step)
