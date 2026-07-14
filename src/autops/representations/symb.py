"""Symbolic EventSat controller and ground schedule producer.

The policy is intentionally transparent: its ordered rules expose why each
mode was requested. Ground scheduling follows established pass-based planning
practice (Sellmaier et al. 2022, doi:10.1007/978-3-030-88593-9).
"""

from __future__ import annotations

import math
import random
from typing import Any

from autops.core.plugin import Representation, register
from autops.core.types import DecisionContext, SpaceSpec
from autops.missions.eventsat.physics import MODES, encode_vectors


def _action(mode: str, *, schedule: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"eventsat_0": {"mode": mode}}
    if schedule is not None:
        result["schedule"] = schedule
    return result


@register("symb", mission="eventsat", role="onboard")
class EventSatSymbolic(Representation):
    observation_space = SpaceSpec((25,), "float32", 0.0, 1.0)
    action_space = SpaceSpec((7,), "int64", 0, 1, MODES)

    def encode_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        _, _, raw = encode_vectors(observation)
        return raw

    def select_action(self, context: DecisionContext) -> dict[str, Any]:
        state = context.state
        soc = float(state.get("battery_soc", 0.5))
        if state.get("health_status", "nominal") != "nominal":
            return self._choose("safe", "R1 anomaly active: protective safe mode")
        if soc < 0.35:
            return self._choose("charging", f"R2 battery critical ({soc:.3f} < 0.35)")
        if state.get("contact_window_active") and float(state.get("obc_data_mb", 0.0)) > 0:
            return self._choose("communication", "R3 predicted contact with OBC backlog")
        if int(state.get("uncompressed_observations", 0)) > 0:
            return self._choose("payload_compress", "R5 raw product awaiting compression")
        if int(state.get("undetected_observations", 0)) > 0:
            return self._choose("payload_detect", "R5c compressed product awaiting detection")
        if float(state.get("jetson_compressed_mb", 0.0)) > 1e-12:
            return self._choose("payload_send", "R5d compressed bytes awaiting CAN transfer")
        projected = (
            float(state.get("obc_data_mb", 0.0))
            + float(state.get("jetson_compressed_mb", 0.0))
            + float(state.get("observation_size_mb", 9.41))
            / max(float(state.get("compression_ratio", 5.11)), 1e-12)
        )
        remaining = float(state.get("remaining_achievable_downlink_mb", math.inf))
        if projected > remaining + 1e-12:
            return self._choose("charging", "R5b new product exceeds remaining link capacity")
        capacity = float(state.get("storage_capacity_mb", 4096.0))
        if soc > 0.6 and float(state.get("data_stored_mb", 0.0)) < 0.8 * capacity:
            return self._choose("payload_observe", "R6 resources permit a science product")
        return self._choose("charging", "R-default no productive mode is currently admissible")

    def _choose(self, mode: str, rationale: str) -> dict[str, Any]:
        self._last_rationale = rationale
        return _action(mode)


@register("symb", mission="eventsat", role="ground")
class EventSatSymbolicScheduler(Representation):
    """Greedy whole-gap planner with explicit link-owned upload action."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._rng = random.Random(0)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._rng.seed(seed)

    def encode_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        _, _, raw = encode_vectors(observation)
        return raw

    def select_action(self, context: DecisionContext) -> dict[str, Any]:
        schedule = self._build_schedule(context.state)
        self._last_rationale = (
            f"Ground pass: selected communication and prepared {sum(x['steps'] for x in schedule)} "
            "steps of resource-aware commands."
        )
        return _action("communication", schedule=schedule)

    def _build_schedule(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        conventional = bool(self.config.get("conventional", False))
        gap_key = "following_gap_steps" if conventional else "planning_gap_steps"
        gap = max(1, int(state.get(gap_key, state.get("following_gap_steps", 92))))
        usable = max(1, int(gap * (0.85 if conventional else 1.0)))
        reserve = max(5, math.ceil(usable * 0.12))
        if conventional:
            reserve = math.ceil(reserve * 1.3)
        settle = max(1, int(state.get("settling_time_steps", 2)))
        comm_steps = min(usable, settle + 1)
        work_budget = max(0, usable - reserve - comm_steps)
        capacity = float(
            state.get("future_pass_capacity_mb", state.get("achievable_downlink_mb", 0.0))
        )
        compressed_mb = float(state.get("observation_size_mb", 9.41)) / max(
            float(state.get("compression_ratio", 5.11)), 1e-12
        )
        staged = float(state.get("obc_data_mb", 0.0)) + float(
            state.get("jetson_compressed_mb", 0.0)
        )
        products = max(0, int(max(0.0, capacity - staged) / max(compressed_mb, 1e-12)))
        if capacity > staged and products == 0:
            products = 1
        if conventional:
            products = min(products, 2)
            if self._rng.random() < 0.10:
                products = max(0, products - 1)
        blocks: list[tuple[str, int]] = []
        raw_backlog = int(state.get("uncompressed_observations", 0))
        detect_backlog = int(state.get("undetected_observations", 0))
        if raw_backlog:
            blocks.append(("payload_compress", min(work_budget, 2 * raw_backlog)))
        used = sum(length for _, length in blocks)
        if detect_backlog and used < work_budget:
            blocks.append(("payload_detect", min(work_budget - used, 5 * detect_backlog)))
        used = sum(length for _, length in blocks)
        if float(state.get("jetson_compressed_mb", 0.0)) > 0 and used < work_budget:
            blocks.append(("payload_send", 1))
        used = sum(length for _, length in blocks)
        cycle_steps = settle + 1 + settle + 2 + 5 + 1
        cycle = [
            ("payload_observe", settle + 1),
            ("payload_compress", settle + 2),
            ("payload_detect", 5),
            ("payload_send", 1),
        ]
        for _ in range(products):
            if used + cycle_steps > work_budget:
                break
            blocks.extend(cycle)
            used += cycle_steps
        charge = usable - used - comm_steps
        if charge > 0:
            blocks.append(("charging", charge))
        if comm_steps > 0:
            blocks.append(("communication", comm_steps))
        if gap > usable:
            blocks.append(("charging", gap - usable))
        return _merge_blocks(blocks)


def _merge_blocks(blocks: list[tuple[str, int]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for mode, steps in blocks:
        if steps <= 0:
            continue
        if merged and merged[-1]["mode"] == mode:
            merged[-1]["steps"] += int(steps)
        else:
            merged.append({"mode": mode, "steps": int(steps)})
    return merged or [{"mode": "charging", "steps": 1}]
