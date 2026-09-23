"""Canonical mission action, observation, and label vocabularies for traces."""

from __future__ import annotations

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

# Onboard-permitted inputs only; see autops.missions.eventsat.observation.
EVENTSAT_OBSERVATIONS = (
    "position_itrf_x_norm",
    "position_itrf_y_norm",
    "position_itrf_z_norm",
    "velocity_itrf_x_norm",
    "velocity_itrf_y_norm",
    "velocity_itrf_z_norm",
    "sun_itrf_x",
    "sun_itrf_y",
    "sun_itrf_z",
    "station_elevation_sin",
    "station_visible",
    "in_sunlight",
    "battery_soc",
    "obc_fill_log",
    "jetson_fill_log",
    "jetson_compressed_fill_log",
    "health_nominal",
    "uncompressed_observations_log",
    "compression_progress",
    "undetected_observations_log",
    "detection_progress",
    "settling_remaining",
    "last_forced_safe",
    "last_forced_charging",
    "last_action_accepted",
    "last_captured",
    "last_compressed",
    "last_detected",
    "last_obc_transfer_norm",
    "last_downlink_norm",
    "last_platform_energy_norm",
    *(f"current_mode_{mode}" for mode in EVENTSAT_ACTIONS),
    *(f"attitude_target_{mode}" for mode in EVENTSAT_ACTIONS),
)

# Privileged simulator labels for targets and evaluation; never decision inputs.
# Countdowns with no later event inside the episode are censored as -1.
EVENTSAT_STATES = (
    "battery_soc",
    "current_mode_idx",
    "in_sunlight",
    "station_visible",
    "contact_window_active",
    "physical_contact_seconds",
    "time_to_next_eclipse",
    "time_to_next_pass",
    "remaining_pass_duration",
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

__all__ = [
    "EVENTSAT_ACTIONS",
    "EVENTSAT_OBSERVATIONS",
    "EVENTSAT_STATES",
    "SSA_ACTIONS",
    "SSA_COLLECTIVE_FIELDS",
    "SSA_OBSERVATIONS",
    "SSA_STATES",
]
