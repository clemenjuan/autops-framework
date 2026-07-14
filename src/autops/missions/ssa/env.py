"""Lean closed-loop SSA custody environment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from autops.core.types import EnvironmentStep
from autops.missions.ssa.dynamics import (
    apply_power,
    collective_reward,
    decode_actions,
    resolve_actions,
)
from autops.missions.ssa.geometry import (
    LinkBudget,
    build_constellation_orbits,
    position_and_velocity_hat,
    satellite_sunlit,
    tangent_for_static,
)
from autops.missions.ssa.metrics import EpisodeStats, custody_ceiling, discovery_ceiling
from autops.missions.ssa.model import SatelliteRuntime, merge_config, record_step
from autops.missions.ssa.pipeline import process_detection_batches, queue_observations
from autops.missions.ssa.targets import (
    OpticalAccess,
    Target,
    generate_catalog,
    optical_accesses,
    propagate_target,
    sun_unit_eci,
)
from autops.missions.ssa.transport import (
    apply_ground_downlinks,
    apply_isl,
    contact_seconds,
    ground_pass_windows,
    published_isl_pairs,
)


class SSAEnvironment:
    """Constellation sensing, onboard detection, record relay, and ground custody."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = merge_config(config)
        simulation = self.config["simulation"]
        constellation = self.config["constellation"]
        self.timestep_s = float(simulation["timestep_s"])
        self.max_steps = int(simulation["max_steps"])
        self.constellation_size = int(constellation["size"])
        self.satellite_ids = [f"sat_{index}" for index in range(self.constellation_size)]
        self.custody_tau_steps = max(0, int(self.config["ssa"]["custody_tau_steps"]))
        self.record_size_bytes = float(self.config["ssa"]["record_size_kb"]) * 1024.0
        self.link_budget = LinkBudget.from_mapping(self.config["isl"])
        self._position_provider: Callable[[str, float], tuple[float, float, float]] | None = None
        self.current_step = 0
        self.seed = 0
        self.targets: list[Target] = []
        self.target_ids: list[str] = []
        self.target_index: dict[str, int] = {}
        self.satellites: dict[str, SatelliteRuntime] = {}
        self.ground_archive: dict[str, list[dict[str, Any]]] = {}
        self.visibility_timeline: list[dict[str, Any]] = []
        self.stats = EpisodeStats()
        self.support_cut_count = 0
        self.physical_utility_ceiling = 0.0
        self.ssa_discovery_ceiling = 0.0

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.seed = int(seed or 0)
        self.current_step = 0
        self.stats = EpisodeStats()
        self._build_episode_geometry()
        initial_soc = float(self.config["power"]["initial_soc"])
        self.satellites = {
            satellite_id: SatelliteRuntime(
                satellite_id=satellite_id,
                battery_soc=initial_soc,
                detection_row=[0] * len(self.target_ids),
            )
            for satellite_id in self.satellite_ids
        }
        self.ground_archive = {object_id: [] for object_id in self.target_ids}
        pass_windows = ground_pass_windows(self)
        self.ssa_discovery_ceiling = discovery_ceiling(
            pass_windows,
            self.visibility_timeline,
            len(self.target_ids),
        )
        self.physical_utility_ceiling = custody_ceiling(
            pass_windows,
            self.visibility_timeline,
            len(self.target_ids),
            self.custody_tau_steps,
            self.max_steps,
        )
        return self.observe()

    def observe(self) -> dict[str, Any]:
        epoch_s = self.current_step * self.timestep_s
        target_positions = self._target_positions(epoch_s)
        satellites: dict[str, dict[str, Any]] = {}
        ground_active: dict[str, bool] = {}
        for satellite_id, runtime in self.satellites.items():
            position = self.satellite_position(satellite_id, epoch_s)
            contact_s = contact_seconds(self, satellite_id, epoch_s)
            ground_active[satellite_id] = contact_s > 0.0
            known = set(runtime.estimates)
            predicted = [
                access.object_id
                for access in self._accesses(
                    satellite_id,
                    epoch_s,
                    target_positions,
                    restrict=known,
                )
            ]
            oldest = runtime.oldest_record_step
            capacity = float(self.config["storage"]["jetson_capacity_mb"])
            satellites[satellite_id] = {
                "satellite_id": satellite_id,
                "position_km": list(position),
                "battery_soc": runtime.battery_soc,
                "mode": runtime.mode,
                "health": runtime.health,
                "in_sunlight": satellite_sunlit(position, epoch_s),
                "ground_pass_active": contact_s > 0.0,
                "contact_seconds": contact_s,
                "jetson_raw_mb": runtime.jetson_raw_mb,
                "jetson_capacity_mb": capacity,
                "observation_size_mb": float(self.config["storage"]["observation_size_mb"]),
                "storage_used_fraction": runtime.jetson_raw_mb / capacity if capacity else 1.0,
                "detection_row": list(runtime.detection_row),
                "known_objects": sorted(known),
                "known_object_ages": {
                    object_id: max(
                        0,
                        self.current_step
                        - int(record.get("last_refresh_step", record_step(record))),
                    )
                    for object_id, record in sorted(runtime.estimates.items())
                },
                "unprocessed_batches": len(runtime.pending_batches),
                "undelivered_records": len(runtime.undelivered),
                "undelivered_record_age_steps": (
                    max(0, self.current_step - oldest) if oldest is not None else 0
                ),
                "predicted_in_fov": predicted,
                "ground_view": {
                    object_id: max(0, self.current_step - observed_step)
                    for object_id, observed_step in sorted(runtime.ground_catalog_steps.items())
                },
            }
        global_state = {
            **self.metrics(),
            "max_steps": self.max_steps,
            "detection_matrix": [
                list(self.satellites[satellite_id].detection_row)
                for satellite_id in self.satellite_ids
            ],
            "isl_feasible_pairs": published_isl_pairs(self),
            "ground_pass_active": ground_active,
            "ground_archive_records": sum(map(len, self.ground_archive.values())),
        }
        return {
            "step": self.current_step,
            "epoch_s": epoch_s,
            "satellites": satellites,
            "global": global_state,
            "tasks": [],
        }

    def step(self, actions: dict[str, Any]) -> EnvironmentStep:
        if self.current_step >= self.max_steps:
            raise RuntimeError("SSA episode is already complete")
        action_step = self.current_step
        epoch_s = action_step * self.timestep_s
        requested = decode_actions(self, actions)
        effective, per_satellite = resolve_actions(self, requested, epoch_s)
        apply_power(self, effective, epoch_s, per_satellite)
        target_positions = self._target_positions(epoch_s)
        queue_observations(self, effective, target_positions, action_step, per_satellite)
        process_detection_batches(self, effective, action_step, per_satellite)
        apply_isl(self, effective, epoch_s, per_satellite)
        apply_ground_downlinks(self, effective, action_step, per_satellite)
        self.current_step += 1
        delivered = self.delivered_object_ids
        custody = self.custody_object_ids
        self.stats.update_step(delivered, custody, len(self.target_ids))
        reward = collective_reward(self, effective, per_satellite)
        info = {
            "requested_modes": requested,
            "resolved_modes": effective,
            "per_satellite": per_satellite,
            **self.metrics(),
        }
        return EnvironmentStep(
            observation=self.observe(),
            reward=reward,
            done=self.current_step >= self.max_steps,
            info=info,
        )

    def metrics(self) -> dict[str, float]:
        freshest = self._freshest_ground_steps()
        known = {
            object_id for runtime in self.satellites.values() for object_id in runtime.estimates
        }
        metrics = self.stats.snapshot(
            current_step=self.current_step,
            target_count=len(self.target_ids),
            delivered=self.delivered_object_ids,
            custody=self.custody_object_ids,
            freshest_ground_steps=freshest,
            known_count=len(known),
            mean_knowledge_latency_steps=self._mean_knowledge_latency(),
        )
        metrics.update(
            {
                "ssa_support_cut_count": float(self.support_cut_count),
                "physical_utility_ceiling": self.physical_utility_ceiling,
                "ssa_discovery_ceiling": self.ssa_discovery_ceiling,
            }
        )
        return metrics

    def episode_metrics(self) -> dict[str, float]:
        metrics = self.metrics()
        utility = metrics["ssa_custody_utility"]
        energy = sum(runtime.energy_consumed_wh for runtime in self.satellites.values())
        ceiling = self.physical_utility_ceiling
        baseline = float(self.config.get("metrics", {}).get("baseline_utility_n1", 0.0))
        metrics.update(
            {
                "utility": utility,
                "mission_goal_utility": 1.0,
                "total_energy_consumed_wh": energy,
                "resource_efficiency": utility / energy if energy else 0.0,
                "utility_fraction_of_physical_ceiling": utility / ceiling if ceiling else 0.0,
                "eta_scale": (
                    (utility / self.constellation_size) / baseline if baseline > 0.0 else 0.0
                ),
            }
        )
        return metrics

    @property
    def delivered_object_ids(self) -> set[str]:
        return {object_id for object_id, records in self.ground_archive.items() if records}

    @property
    def custody_object_ids(self) -> set[str]:
        return {
            object_id
            for object_id, observed_step in self._freshest_ground_steps().items()
            if self.current_step - observed_step <= self.custody_tau_steps
        }

    def satellite_position(self, satellite_id: str, epoch_s: float) -> tuple[float, float, float]:
        if self._position_provider is not None:
            return self._position_provider(satellite_id, epoch_s)
        fixed = self.config["constellation"]["fixed_positions_km"]
        if satellite_id in fixed:
            return tuple(float(value) for value in fixed[satellite_id])
        return propagate_target(self._satellite_orbits[satellite_id], epoch_s)

    def _freshest_ground_steps(self) -> dict[str, int]:
        return {
            object_id: max(record_step(record) for record in records)
            for object_id, records in self.ground_archive.items()
            if records
        }

    def _mean_knowledge_latency(self) -> float:
        latencies: list[int] = []
        for object_id, first_step in self.stats.first_detected_step.items():
            for runtime in self.satellites.values():
                known_step = runtime.first_known_steps.get(object_id, self.current_step)
                latencies.append(max(0, int(known_step) - first_step))
            ground_step = self.stats.delivered_step.get(object_id, self.current_step)
            latencies.append(max(0, int(ground_step) - first_step))
        return sum(latencies) / len(latencies) if latencies else 0.0

    def _build_episode_geometry(self) -> None:
        constellation = self.config["constellation"]
        orbit = self.config["orbit"]
        self._satellite_orbits = build_constellation_orbits(
            self.constellation_size,
            self.seed,
            altitude_km=float(orbit["altitude_km"]),
            inclination_deg=float(orbit["inclination_deg"]),
            eccentricity=float(orbit["eccentricity"]),
            spacing_deg=float(constellation["in_plane_spacing_deg"]),
        )
        target_config = self.config["targets"]
        fixed = target_config["fixed_positions_km"]
        count = len(fixed) if fixed else int(target_config["count"])
        center = (
            next(iter(self._satellite_orbits.values())).raan_deg if self._satellite_orbits else 0.0
        )
        self.targets = generate_catalog(
            count,
            self.seed,
            raan_center_deg=center,
            parent_altitude_km=float(target_config["parent_altitude_km"]),
            parent_inclination_deg=float(target_config["parent_inclination_deg"]),
            raan_spread_deg=float(target_config["raan_spread_deg"]),
            along_track_sigma_ms=float(target_config["along_track_sigma_ms"]),
            normal_sigma_ms=float(target_config["normal_sigma_ms"]),
            size_bounds_m=tuple(target_config["size_bounds_m"]),
        )
        if fixed:
            self.targets = [
                replace(target, object_id=object_id)
                for target, object_id in zip(self.targets, fixed, strict=True)
            ]
        original_count = len(self.targets)
        self.visibility_timeline = self._compute_visibility_timeline()
        accessible = {
            object_id
            for item in self.visibility_timeline
            for object_id in item["visible_target_ids"]
        }
        self.targets = [target for target in self.targets if target.object_id in accessible]
        self.target_ids = [target.object_id for target in self.targets]
        self.target_index = {object_id: index for index, object_id in enumerate(self.target_ids)}
        self.support_cut_count = original_count - len(self.targets)
        allowed = set(self.target_ids)
        self.visibility_timeline = [
            {
                "step": item["step"],
                "visible_target_ids": [
                    object_id for object_id in item["visible_target_ids"] if object_id in allowed
                ],
            }
            for item in self.visibility_timeline
        ]

    def _compute_visibility_timeline(self) -> list[dict[str, Any]]:
        timeline: list[dict[str, Any]] = []
        for step in range(self.max_steps):
            epoch_s = step * self.timestep_s
            positions = self._target_positions(epoch_s)
            visible = {
                access.object_id
                for satellite_id in self.satellite_ids
                for access in self._accesses(satellite_id, epoch_s, positions)
            }
            timeline.append({"step": step, "visible_target_ids": sorted(visible)})
        return timeline

    def _target_positions(self, epoch_s: float) -> dict[str, tuple[float, float, float]]:
        fixed = self.config["targets"]["fixed_positions_km"]
        if fixed:
            allowed = {target.object_id for target in self.targets}
            return {
                object_id: tuple(float(value) for value in position)
                for object_id, position in fixed.items()
                if not allowed or object_id in allowed
            }
        return {target.object_id: propagate_target(target, epoch_s) for target in self.targets}

    def _accesses(
        self,
        satellite_id: str,
        epoch_s: float,
        target_positions: dict[str, tuple[float, float, float]],
        restrict: set[str] | None = None,
    ) -> list[OpticalAccess]:
        position = self.satellite_position(satellite_id, epoch_s)
        fixed = self.config["constellation"]["fixed_positions_km"]
        velocity = (
            tangent_for_static(position)
            if satellite_id in fixed
            else position_and_velocity_hat(self._satellite_orbits[satellite_id], epoch_s)[1]
        )
        positions = (
            target_positions
            if restrict is None
            else {key: value for key, value in target_positions.items() if key in restrict}
        )
        target_config = self.config["targets"]
        return optical_accesses(
            position,
            velocity,
            self.targets,
            positions,
            sun_unit_eci(epoch_s),
            fov_half_angle_deg=float(target_config["fov_half_angle_deg"]),
            boresight_pitch_deg=float(target_config["boresight_pitch_deg"]),
            range_cap_km=float(target_config["range_cap_km"]),
            magnitude_limit=float(target_config["magnitude_limit"]),
            magnitude_sigma=float(target_config["magnitude_sigma"]),
            albedo=float(target_config["albedo"]),
        )
