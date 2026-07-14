"""Single authoritative EventSat truth environment.

Orekit's Eckstein-Hechler propagation is preferred; a seeded fallback preserves
portable experiments. Operations logic sees the deterministic contact plan,
while data transfer remains gated by physical overlap. Spacecraft-operations
context follows Sellmaier et al. (2022), doi:10.1007/978-3-030-88593-9.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any

from autops.core.types import EnvironmentStep
from autops.missions.eventsat.physics import MODES, EventSatState, power_step
from autops.missions.eventsat.transitions import (
    PipelineParameters,
    apply_can_transfer,
    apply_compress,
    apply_detect,
    apply_downlink,
    apply_observe,
)
from autops.orbital import (
    GroundStation,
    OrbitElements,
    SimplifiedModel,
    apply_launch_lottery,
    build_orbital_context,
)


class EventSatEnvironment:
    satellite_id = "eventsat_0"

    def __init__(
        self,
        config: dict[str, Any],
        *,
        max_steps: int | None = None,
        onboard_compute_active: bool = False,
        anomaly_requires_ground_pass: bool = False,
        prefer_orekit: bool = True,
    ) -> None:
        self.config = config
        self.timestep_s = float(config["simulation"]["timestep_s"])
        self.max_steps = int(max_steps or config["simulation"]["max_steps"])
        self.onboard_compute_active = onboard_compute_active
        self.anomaly_requires_ground_pass = anomaly_requires_ground_pass
        self.prefer_orekit = prefer_orekit
        self.orbital_period_steps = max(
            1, int(float(config["orbit"]["orbital_period_s"]) / self.timestep_s)
        )
        transition = config["modes"]["transition_overhead"]
        self.settling_steps = max(0, int(float(transition["settling_time_s"]) / self.timestep_s))
        self.maneuver_modes = set(transition["attitude_maneuver_modes"])
        self.detection_steps = max(
            1, int(float(config["payload"]["detection_time_s"]) / self.timestep_s)
        )
        self.compression_steps = max(1, int(float(config["payload"]["compression_time_factor"])))
        self.state = EventSatState()
        self.orbit = None
        self._arrival_rng = random.Random()
        self._duration_rng = random.Random()
        self._last_info: dict[str, Any] = {}
        self._seed = 0

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._seed = 0 if seed is None else int(seed)
        initial_soc = float(self.config["power"]["battery"]["initial_soc"])
        self.state = EventSatState(battery_soc=initial_soc)
        anomaly_seed = self._seed * 131 + 7919
        self._arrival_rng.seed(anomaly_seed)
        self._duration_rng.seed(anomaly_seed + 104729)
        elements = self._orbit_elements()
        if self.config["orbit"].get("launch_lottery", False):
            elements = apply_launch_lottery(elements, self._seed)
        self.state.orbit_elements = {
            "raan_deg": elements.raan_deg,
            "arg_perigee_deg": elements.arg_perigee_deg,
            "true_anomaly_deg": elements.true_anomaly_deg,
        }
        self.orbit = build_orbital_context(
            elements,
            self._fallback_model(),
            self._ground_station(),
            downlink_rate_kbps=self.downlink_rate_kbps,
            step_s=self.timestep_s,
            total_steps=self.max_steps,
            seed=self._seed,
            prefer_orekit=self.prefer_orekit,
        )
        self._last_info = {}
        return self.observe()

    @property
    def downlink_rate_kbps(self) -> float:
        return float(self.config["communications"]["sband"]["downlink_rate_kbps"])

    def physical_contact_active(self) -> bool:
        return bool(self.orbit and self.orbit.is_ground_pass_active(self.state.step))

    def step(self, actions: dict[str, Any]) -> EnvironmentStep:
        requested, planner_power_w = self._parse_action(actions)
        resolved = self._resolve_mode(requested)
        safety_safe = resolved == "safe"
        effective, in_transition = self._settle(resolved)
        sunlight = bool(self.orbit and self.orbit.is_in_sunlight(self.state.step))
        contact_s = float(self.orbit.contact_seconds(self.state.step)) if self.orbit else 0.0
        energy = power_step(
            self.state, self.config, effective, sunlight, planner_power_w=planner_power_w
        )
        if contact_s > 0:
            self.state.total_contact_s += contact_s
        effects = self._apply_mode(effective, contact_s)
        anomaly_event = self._update_anomaly()
        self.state.current_mode = effective
        self.state.step += 1
        info = self._info(
            requested,
            resolved,
            effective,
            safety_safe,
            in_transition,
            contact_s,
            anomaly_event,
            energy,
            effects,
        )
        self._last_info = info
        return EnvironmentStep(
            observation=self.observe(),
            reward=self._reward(effective, info),
            done=self.state.step >= self.max_steps,
            info=info,
        )

    def observe(self) -> dict[str, Any]:
        lookahead = self._lookahead()
        state = self.state
        storage = self.config["storage"]
        max_downlink = state.total_contact_s * self.downlink_rate_kbps / 8.0 / 1000.0
        metadata = {
            **lookahead,
            "in_sunlight": bool(self.orbit and self.orbit.is_in_sunlight(state.step)),
            "physical_ground_pass_active": self.physical_contact_active(),
            "contact_window_active": (
                lookahead["contact_window_seconds"] > 0
                or 0 < lookahead["time_to_next_pass"] <= self.settling_steps
            ),
            "health_status": "nominal" if state.active_anomaly is None else state.active_anomaly,
            "jetson_raw_mb": state.jetson_raw_mb,
            "jetson_compressed_mb": state.jetson_compressed_mb,
            "obc_data_mb": state.obc_data_mb,
            "uncompressed_observations": state.uncompressed_observations,
            "undetected_observations": state.undetected_observations,
            "compression_progress": state.compression_progress,
            "detection_progress": state.detection_progress,
            "total_observation_s": state.total_observation_s,
            "total_detections": state.total_detections,
            "total_raw_captured_mb": state.total_raw_captured_mb,
            "obc_raw_equivalent_mb": state.obc_raw_equivalent_mb,
            "downlink_raw_equivalent_mb": state.downlink_raw_equivalent_mb,
            "storage_capacity_mb": float(storage["obc_capacity_mb"]),
            "jetson_capacity_mb": float(storage["jetson_capacity_mb"]),
            "observation_size_mb": float(storage["observation_size_mb"]),
            "compression_ratio": float(storage["compression_ratio"]),
            "detection_metadata_mb": float(storage["detection_metadata_mb"]),
            "jetson_to_obc_rate_kbps": float(storage["jetson_to_obc_rate_kbps"]),
            "downlink_rate_kbps": self.downlink_rate_kbps,
            "orbital_period_steps": self.orbital_period_steps,
            "settling_time_steps": self.settling_steps,
            "transition_steps_remaining": state.transition_steps_remaining,
            "max_achievable_downlink_mb": max_downlink,
            "achievable_downlink_mb": lookahead["future_pass_capacity_mb"],
            "remaining_achievable_downlink_mb": self._remaining_downlink_mb(),
        }
        return {
            "step": state.step,
            "epoch_s": state.step * self.timestep_s,
            "satellites": {
                self.satellite_id: {
                    "status": state.current_mode,
                    "resources": {
                        "battery_soc": state.battery_soc,
                        "data_stored_mb": state.data_stored_mb,
                        "obc_data_mb": state.obc_data_mb,
                        "data_downlinked_mb": state.data_downlinked_mb,
                    },
                    "metadata": metadata,
                }
            },
            "global": {
                "max_steps": self.max_steps,
                "orbital_backend": self.orbit.backend if self.orbit else "uninitialized",
            },
            "tasks": self._tasks(metadata),
        }

    def episode_provenance(self) -> dict[str, Any]:
        return {
            "seed": self._seed,
            "orbit": dict(self.state.orbit_elements),
            "orbital_backend": self.orbit.backend if self.orbit else None,
            "ground_passes": [
                {"start_s": item.start_s, "end_s": item.end_s}
                for item in (self.orbit.ground_passes if self.orbit else ())
            ],
        }

    def _parse_action(self, actions: dict[str, Any]) -> tuple[str, float]:
        raw = actions.get(self.satellite_id, {}) if isinstance(actions, dict) else {}
        raw = raw if isinstance(raw, dict) else {}
        mode = str(raw.get("mode", "charging"))
        planned = bool(raw.get("jetson_planned", True))
        explicit = raw.get("planner_power_w")
        if explicit is not None:
            return mode, max(0.0, float(explicit)) if planned else 0.0
        default = float(self.config["power"].get("onboard_compute_w", 0.0))
        return mode, default if self.onboard_compute_active and planned else 0.0

    def _resolve_mode(self, requested: str) -> str:
        if requested not in MODES:
            return "charging"
        state = self.state
        battery = self.config["power"]["battery"]
        if state.active_anomaly is not None or state.battery_soc <= float(battery["min_soc"]):
            return "safe"
        constraint = self.config["modes"].get("constraints", {}).get(requested, {})
        if state.battery_soc < float(constraint.get("min_battery_soc", 0.0)):
            return "charging"
        return requested

    def _settle(self, resolved: str) -> tuple[str, bool]:
        state = self.state
        if state.transition_steps_remaining > 0:
            state.transition_steps_remaining -= 1
            if state.transition_steps_remaining == 0:
                state.previous_mode = resolved
            return "charging", True
        maneuver = state.previous_mode != resolved and (
            resolved in self.maneuver_modes or state.previous_mode in self.maneuver_modes
        )
        if maneuver and self.settling_steps > 0:
            state.transition_steps_remaining = self.settling_steps - 1
            if state.transition_steps_remaining == 0:
                state.previous_mode = resolved
            return "charging", True
        state.previous_mode = resolved
        return resolved, False

    def _apply_mode(self, mode: str, contact_s: float) -> dict[str, Any]:
        state = self.state
        p = self._pipeline_parameters()
        result: dict[str, Any] = {"step_downlinked_mb": 0.0, "contact_seconds": contact_s}
        if mode != "payload_compress" and state.current_mode == "payload_compress":
            state.compression_progress = 0
        if mode != "payload_detect" and state.current_mode == "payload_detect":
            state.detection_progress = 0
        outcome = None
        if mode == "payload_observe":
            outcome = apply_observe(state.pipeline(), p)
        elif mode == "payload_compress" and state.uncompressed_observations:
            state.compression_progress += 1
            if state.compression_progress >= self.compression_steps:
                outcome = apply_compress(state.pipeline(), p)
                if outcome.accepted:
                    state.compression_progress = 0
        elif mode == "payload_detect" and state.undetected_observations:
            state.detection_progress += 1
            if state.detection_progress >= self.detection_steps:
                outcome = apply_detect(state.pipeline(), p)
                if outcome.accepted:
                    state.detection_progress = 0
        elif mode == "payload_send":
            outcome = apply_can_transfer(state.pipeline(), p)
        elif mode == "communication" and contact_s > 0:
            outcome = apply_downlink(state.pipeline(), p, contact_seconds=contact_s)
        if outcome and outcome.accepted:
            state.accept_pipeline(outcome.state)
            result["step_downlinked_mb"] = (
                outcome.transferred_mb if mode == "communication" else 0.0
            )
        result["action_accepted"] = (
            bool(outcome.accepted) if outcome else mode in {"charging", "safe"}
        )
        return result

    def _update_anomaly(self) -> str | None:
        cfg = self.config["anomalies"]
        if self._arrival_rng.random() < float(cfg["probability_per_step"]):
            duration = self._duration_rng.randint(
                int(cfg["min_duration_steps"]), int(cfg["max_duration_steps"])
            )
            self.state.active_anomaly = str(cfg["kind"])
            self.state.forced_safe_steps = max(self.state.forced_safe_steps, duration)
            return self.state.active_anomaly
        if self.state.active_anomaly is not None:
            self.state.forced_safe_steps -= 1
            may_clear = self.state.forced_safe_steps <= 0 and (
                not self.anomaly_requires_ground_pass or self.physical_contact_active()
            )
            if may_clear:
                self.state.active_anomaly = None
        return None

    def _info(
        self,
        requested: str,
        resolved: str,
        effective: str,
        safety_safe: bool,
        in_transition: bool,
        contact_s: float,
        anomaly_event: str | None,
        energy: dict[str, float],
        effects: dict[str, Any],
    ) -> dict[str, Any]:
        state = self.state
        return {
            "requested_mode": requested,
            "resolved_mode": effective,
            "safety_resolved_mode": resolved,
            "forced": resolved != requested,
            "safety_safe": float(safety_safe),
            "in_transition": in_transition,
            "contact_seconds": contact_s,
            "anomaly_event": anomaly_event,
            "anomaly_active": state.active_anomaly is not None,
            "battery_soc": state.battery_soc,
            "data_stored_mb": state.data_stored_mb,
            "obc_data_mb": state.obc_data_mb,
            "data_downlinked_mb": state.data_downlinked_mb,
            "step_downlinked_mb": effects.get("step_downlinked_mb", 0.0),
            "observation_hours": state.total_observation_s / 3600.0,
            "total_raw_captured_mb": state.total_raw_captured_mb,
            "downlink_raw_equivalent_mb": state.downlink_raw_equivalent_mb,
            "max_achievable_downlink_mb": state.total_contact_s
            * self.downlink_rate_kbps
            / 8.0
            / 1000.0,
            **energy,
            **effects,
        }

    def _lookahead(self) -> dict[str, float]:
        step = self.state.step
        now = step * self.timestep_s
        period = self.orbital_period_steps
        eclipses = self.orbit.eclipses if self.orbit else ()
        passes = self.orbit.ground_passes if self.orbit else ()
        future_eclipses = [item for item in eclipses if item.start_s > now]
        future_passes = [item for item in passes if item.start_s > now]
        current = self.orbit.get_current_pass(step) if self.orbit else None
        time_eclipse = (
            int((future_eclipses[0].start_s - now) / self.timestep_s) if future_eclipses else period
        )
        time_pass = (
            int((future_passes[0].start_s - now) / self.timestep_s) if future_passes else period
        )
        remaining_s = max(0.0, current.end_s - now) if current else 0.0
        current_contact_s = self.orbit.contact_seconds(step) if self.orbit else 0.0
        next_gap = period
        following_gap = period
        if future_passes:
            reference_end = current.end_s if current else now
            next_gap = max(1, int((future_passes[0].start_s - reference_end) / self.timestep_s))
        if len(future_passes) >= 2:
            following_gap = max(
                1, int((future_passes[1].start_s - future_passes[0].end_s) / self.timestep_s)
            )
        capacity_s = self.orbit.future_pass_contact_s(step, 1) if self.orbit else 0.0
        return {
            "orbital_phase": (step % period) / period,
            "time_to_next_eclipse": float(time_eclipse),
            "time_to_next_pass": float(time_pass),
            "remaining_pass_duration": remaining_s / self.timestep_s,
            "remaining_pass_duration_s": remaining_s,
            "contact_window_seconds": current_contact_s,
            "next_gap_steps": float(next_gap),
            "following_gap_steps": float(following_gap),
            "planning_gap_steps": float(next_gap),
            "future_pass_capacity_mb": capacity_s * self.downlink_rate_kbps / 8.0 / 1000.0,
        }

    def _remaining_downlink_mb(self) -> float:
        seconds = self.orbit.remaining_contact_s(self.state.step) if self.orbit else 0.0
        return seconds * self.downlink_rate_kbps / 8.0 / 1000.0

    def _pipeline_parameters(self) -> PipelineParameters:
        storage = self.config["storage"]
        return PipelineParameters(
            observation_size_mb=float(storage["observation_size_mb"]),
            compression_ratio=float(storage["compression_ratio"]),
            jetson_capacity_mb=float(storage["jetson_capacity_mb"]),
            obc_capacity_mb=float(storage["obc_capacity_mb"]),
            detection_metadata_mb=float(storage["detection_metadata_mb"]),
            jetson_to_obc_rate_kbps=float(storage["jetson_to_obc_rate_kbps"]),
            downlink_rate_kbps=self.downlink_rate_kbps,
            step_duration_s=self.timestep_s,
        )

    def _orbit_elements(self) -> OrbitElements:
        orbit = self.config["orbit"]
        epoch = datetime.fromisoformat(self.config["simulation"]["epoch"].replace("Z", "+00:00"))
        return OrbitElements(
            altitude_km=float(orbit["altitude_km"]),
            eccentricity=float(orbit["eccentricity"]),
            inclination_deg=float(orbit["inclination_deg"]),
            raan_deg=float(orbit["raan_deg"]),
            arg_perigee_deg=float(orbit["arg_perigee_deg"]),
            true_anomaly_deg=float(orbit["true_anomaly_deg"]),
            epoch=epoch,
            propagator=str(orbit["propagator"]),
        )

    def _fallback_model(self) -> SimplifiedModel:
        orbit = self.config["orbit"]
        passes = self.config["communications"]["passes"]
        return SimplifiedModel(
            orbital_period_s=float(orbit["orbital_period_s"]),
            eclipse_fraction=float(orbit["eclipse_fraction"]),
            passes_min_per_day=int(passes["min_per_day"]),
            passes_max_per_day=int(passes["max_per_day"]),
            pass_min_duration_s=float(passes["min_duration_s"]),
            pass_max_duration_s=float(passes["max_duration_s"]),
        )

    def _ground_station(self) -> GroundStation:
        station = self.config["communications"]["ground_station"]
        return GroundStation(
            latitude_deg=float(station["latitude_deg"]),
            longitude_deg=float(station["longitude_deg"]),
            altitude_m=float(station.get("altitude_m", 0.0)),
            min_elevation_deg=float(station["min_elevation_deg"]),
        )

    def _reward(self, mode: str, info: dict[str, Any]) -> float:
        rewards = self.config["rewards"]
        value = float(rewards["comm_reward_factor"]) * float(info.get("step_downlinked_mb", 0.0))
        if mode == "safe":
            value -= float(rewards["safe_penalty"])
        if not info.get("action_accepted", True):
            value -= float(rewards["failed_action_penalty"])
        return float(rewards["reward_scale"]) * value

    def _tasks(self, metadata: dict[str, Any]) -> list[dict[str, str]]:
        tasks: list[dict[str, str]] = []
        if self.state.battery_soc < 0.4:
            tasks.append({"type": "manage_power", "priority": "high"})
        if metadata["contact_window_active"] and self.state.obc_data_mb > 0:
            tasks.append({"type": "schedule_downlink", "priority": "high"})
        if self.state.battery_soc > 0.6:
            tasks.append({"type": "schedule_observation", "priority": "normal"})
        return tasks
