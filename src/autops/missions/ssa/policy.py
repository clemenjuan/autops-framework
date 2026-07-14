"""Local-knowledge symbolic custody policy for SSA."""

from __future__ import annotations

from typing import Any

from autops.core import DecisionContext, Representation, register
from autops.core.types import SpaceSpec

SSA_MODES = (
    "charging",
    "communication",
    "payload_observe",
    "payload_detect",
    "isl_share",
    "safe",
)

SSA_ACTION_SPACE = SpaceSpec(
    shape=(1,),
    dtype="int64",
    low=0,
    high=len(SSA_MODES) - 1,
    labels=SSA_MODES,
)


@register("symb", mission="ssa", role="onboard")
class RuleBasedSSA(Representation):
    """Resource-aware policy using only the satellites present in its scoped view."""

    action_space = SSA_ACTION_SPACE

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.satellite_id = self.config.get("satellite_id")
        self.low_soc = float(self.config.get("battery_threshold_low", 0.3))
        self.high_soc = float(self.config.get("battery_threshold_high", 0.8))
        self.observe_soc = float(self.config.get("observe_soc", 0.6))
        self.detect_soc = float(self.config.get("detect_soc", 0.45))
        self.storage_high = float(self.config.get("storage_threshold_high", 0.7))
        self.backlog_threshold = max(1, int(self.config.get("detect_backlog_threshold", 2)))
        self.custody_tau_steps = max(0, int(self.config.get("custody_tau_steps", 4320)))
        self.isl_aoi_threshold_steps = max(
            0,
            int(
                self.config.get(
                    "isl_aoi_threshold_steps",
                    self.custody_tau_steps // 8,
                )
            ),
        )

    def encode_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        satellites = observation.get("satellites", {}) if isinstance(observation, dict) else {}
        encoded: dict[str, dict[str, Any]] = {}
        for satellite_id, raw in satellites.items():
            encoded[satellite_id] = {
                key: raw.get(key, default)
                for key, default in (
                    ("battery_soc", 0.5),
                    ("health", "nominal"),
                    ("ground_pass_active", False),
                    ("storage_used_fraction", 0.0),
                    ("jetson_raw_mb", 0.0),
                    ("jetson_capacity_mb", 249036.8),
                    ("observation_size_mb", 2016.0),
                    ("unprocessed_batches", 0),
                    ("undelivered_records", 0),
                    ("undelivered_record_age_steps", 0),
                    ("predicted_in_fov", []),
                    ("ground_view", {}),
                )
            }
        return {"satellites": encoded}

    def select_action(self, context: DecisionContext) -> dict[str, Any]:
        satellites = context.state.get("satellites", {})
        if not satellites:
            self._last_rationale = "No SSA state is available; charging."
            return {}
        satellite_ids = sorted(satellites)
        if self.satellite_id in satellites:
            satellite_ids = [str(self.satellite_id)]
        coordinated = len(satellite_ids) > 1
        claimed: set[str] = set()
        actions: dict[str, dict[str, str]] = {}
        rationales: list[str] = []
        for satellite_id in satellite_ids:
            mode, objects, rationale = self.mode_for_satellite(
                satellite_id,
                satellites[satellite_id],
                claimed,
                coordinated=coordinated,
            )
            actions[satellite_id] = {"mode": mode}
            if mode == "payload_observe":
                claimed.update(objects)
            rationales.append(rationale)
        self._last_rationale = " ".join(rationales)
        return actions

    def mode_for_satellite(
        self,
        satellite_id: str,
        satellite: dict[str, Any],
        claimed: set[str],
        *,
        coordinated: bool,
    ) -> tuple[str, list[str], str]:
        soc = float(satellite.get("battery_soc", 0.5))
        health = str(satellite.get("health", "nominal"))
        pass_active = bool(satellite.get("ground_pass_active", False))
        backlog = int(satellite.get("unprocessed_batches", 0))
        undelivered = int(satellite.get("undelivered_records", 0))
        record_age = int(satellite.get("undelivered_record_age_steps", 0))
        storage_allows = self._storage_allows_observation(satellite)
        predicted = [str(value) for value in satellite.get("predicted_in_fov", [])]
        ground_view = {
            str(object_id): int(age) for object_id, age in satellite.get("ground_view", {}).items()
        }
        stale = [
            object_id
            for object_id in predicted
            if object_id not in ground_view or ground_view[object_id] > self.custody_tau_steps / 2.0
        ]
        candidates = [obj for obj in stale if not coordinated or obj not in claimed]

        if health != "nominal":
            return "safe", [], f"{satellite_id}: non-nominal health; safe."
        if soc < self.low_soc:
            return "charging", [], f"{satellite_id}: low battery; charging."
        if pass_active and undelivered:
            return "communication", [], f"{satellite_id}: downlinking fresh custody records."
        urgent_relay = (
            coordinated
            and undelivered > 0
            and not pass_active
            and soc > self.observe_soc
            and record_age >= self.isl_aoi_threshold_steps
        )
        if urgent_relay:
            return "isl_share", [], f"{satellite_id}: stale custody record; relaying."
        if backlog >= self.backlog_threshold and soc > self.detect_soc:
            return "payload_detect", [], f"{satellite_id}: processing detection backlog."
        if soc > self.observe_soc and storage_allows:
            if candidates:
                return "payload_observe", candidates, f"{satellite_id}: refreshing stale cues."
            if backlog == 0 and soc > self.high_soc:
                return "payload_observe", [], f"{satellite_id}: blind survey."
        if backlog and soc > self.detect_soc:
            return "payload_detect", [], f"{satellite_id}: processing oldest batch."
        if coordinated and undelivered and not pass_active and soc > self.observe_soc:
            return "isl_share", [], f"{satellite_id}: opportunistic record relay."
        return "charging", [], f"{satellite_id}: no productive custody action."

    def _storage_allows_observation(self, satellite: dict[str, Any]) -> bool:
        used_fraction = float(satellite.get("storage_used_fraction", 0.0))
        raw_mb = float(satellite.get("jetson_raw_mb", 0.0))
        capacity_mb = float(satellite.get("jetson_capacity_mb", 249036.8))
        observation_mb = float(satellite.get("observation_size_mb", 2016.0))
        return (
            used_fraction < self.storage_high
            and capacity_mb > 0.0
            and raw_mb + observation_mb <= capacity_mb + 1e-12
        )
