"""Delivery-aligned EventSat reward consumed by reinforcement learning.

The Individual Negative reward of Juan Oliver et al. (EUCASS 2025), as validated
in the agentic framework: ``alpha * (R_resource + R_action + R_mission)``.
Resource and failed-action penalties shape learning, successful pipeline stages
are neutral by default so a policy is not taught a hand-written operations
sequence, and mission progress counts only data delivered to the ground. No other
representation reads the reward; results record its episode total. Optional
potential-based shaping is a training aid and lives in ``autops.rl``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class EventSatRewardFunction:
    """Resource, action-outcome, and unmet-delivery terms of one EventSat step."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        self.reward_scale = float(cfg.get("reward_scale", 0.01))
        self.resource_penalty_factor = float(cfg.get("resource_penalty_factor", 1.0))
        self.battery_low_threshold = float(cfg.get("battery_low_threshold", 0.3))
        self.storage_high_threshold = float(cfg.get("storage_high_threshold", 0.8))
        self.standby_penalty = float(cfg.get("standby_penalty", 0.0))
        self.safe_penalty = float(cfg.get("safe_penalty", 0.3))
        # Explicit ablation knobs; benchmark defaults do not reward pipeline busywork.
        self.observe_reward = float(cfg.get("observe_reward", 0.0))
        self.compress_reward = float(cfg.get("compress_reward", 0.0))
        self.failed_action_penalty = float(cfg.get("failed_action_penalty", 0.1))
        self.comm_reward_factor = float(cfg.get("comm_reward_factor", 1.0))
        self.comm_reward_cap = float(cfg.get("comm_reward_cap", 5.0))
        # An in-contact downlink of an empty OBC is a failure unless explicitly ablated;
        # an out-of-contact attempt always is.
        self.empty_downlink_is_failure = bool(cfg.get("empty_downlink_is_failure", True))
        self.mission_scale = float(cfg.get("mission_scale", 1.0))
        # Raw observations that never reach the ground are not mission utility.
        self.mission_observation_weight = float(cfg.get("mission_observation_weight", 0.0))
        self.mission_downlink_weight = float(cfg.get("mission_downlink_weight", 1.0))
        if self.mission_observation_weight < 0 or self.mission_downlink_weight < 0:
            raise ValueError("mission reward weights must be non-negative")
        self._mission_weight_sum = self.mission_observation_weight + self.mission_downlink_weight

    def resource_penalty(
        self, battery_soc: float, data_stored_mb: float, storage_capacity_mb: float
    ) -> float:
        """Penalty proportional to battery below, and storage above, their thresholds."""

        penalty = 0.0
        if battery_soc < self.battery_low_threshold:
            penalty += (self.battery_low_threshold - battery_soc) / self.battery_low_threshold
        storage_ratio = data_stored_mb / storage_capacity_mb if storage_capacity_mb > 0 else 0.0
        if storage_ratio > self.storage_high_threshold:
            penalty += (storage_ratio - self.storage_high_threshold) / (
                1.0 - self.storage_high_threshold
            )
        return -self.resource_penalty_factor * penalty

    def action_reward(self, mode: str, action_info: Mapping[str, Any]) -> float:
        """Outcome term of the executed mode; failed actions pay one fixed penalty."""

        if self.is_failed_action(mode, action_info):
            return -self.failed_action_penalty
        if mode == "payload_observe":
            return self.observe_reward
        if mode in {"payload_compress", "payload_detect"}:
            return self.compress_reward
        if mode == "payload_send":
            return self.compress_reward * 0.5
        if mode == "communication":
            downlinked = float(action_info.get("data_downlinked_mb", 0.0))
            return min(self.comm_reward_factor * downlinked, self.comm_reward_cap)
        if mode == "charging":
            return -self.standby_penalty
        if mode == "safe":
            return -self.safe_penalty
        return 0.0

    def is_failed_action(self, mode: str, action_info: Mapping[str, Any]) -> bool:
        """Reward-independent failure classification shared with step diagnostics."""

        if action_info.get("constraint_violation", False):
            return True
        if mode == "payload_observe":
            return bool(action_info.get("storage_overflow", False))
        if mode == "payload_compress":
            return not action_info.get("had_data_to_compress", False)
        if mode == "payload_detect":
            return not action_info.get("had_data_to_detect", False)
        if mode == "payload_send":
            return not action_info.get("had_data_to_send", False)
        if mode == "communication":
            if not action_info.get("pass_active", False):
                return True
            if float(action_info.get("data_downlinked_mb", 0.0)) > 0.0:
                return False
            if action_info.get("communication_failure") == "no_contact":
                return True
            return self.empty_downlink_is_failure
        return False

    def mission_penalty(
        self,
        is_final_step: bool,
        obs_hours: float,
        downlinked_mb: float,
        obs_target_hours: float,
        downlink_target_mb: float,
        episode_steps: int,
        max_mission_steps: int,
    ) -> float:
        """Weighted unmet fraction of the mission targets, growing with episode progress."""

        obs_gap = max(0.0, 1.0 - obs_hours / obs_target_hours) if obs_target_hours > 0 else 0.0
        dl_gap = (
            max(0.0, 1.0 - downlinked_mb / downlink_target_mb) if downlink_target_mb > 0 else 0.0
        )
        unmet_fraction = (
            (self.mission_observation_weight * obs_gap + self.mission_downlink_weight * dl_gap)
            / self._mission_weight_sum
            if self._mission_weight_sum > 0.0
            else 0.0
        )
        if is_final_step:
            return -self.mission_scale * unmet_fraction
        progress = episode_steps / max_mission_steps if max_mission_steps > 0 else 1.0
        return -self.mission_scale * unmet_fraction * progress

    def compute(
        self,
        mode: str,
        battery_soc: float,
        data_stored_mb: float,
        storage_capacity_mb: float,
        action_info: Mapping[str, Any],
        obs_hours: float,
        downlinked_mb: float,
        obs_target_hours: float,
        downlink_target_mb: float,
        episode_step: int,
        max_steps: int,
        is_final_step: bool,
    ) -> float:
        """Scaled sum of the resource, action, and mission terms."""

        return self.reward_scale * (
            self.resource_penalty(battery_soc, data_stored_mb, storage_capacity_mb)
            + self.action_reward(mode, action_info)
            + self.mission_penalty(
                is_final_step=is_final_step,
                obs_hours=obs_hours,
                downlinked_mb=downlinked_mb,
                obs_target_hours=obs_target_hours,
                downlink_target_mb=downlink_target_mb,
                episode_steps=episode_step,
                max_mission_steps=max_steps,
            )
        )


def step_action_info(mode: str, info: Mapping[str, Any]) -> dict[str, Any]:
    """Express one environment step's outcome in the reward function's vocabulary."""

    had_product = bool(info["had_product"])
    return {
        "constraint_violation": info["constraint_violation"],
        "storage_overflow": mode == "payload_observe" and not info["action_accepted"],
        "had_data_to_compress": had_product,
        "had_data_to_detect": had_product,
        "had_data_to_send": had_product,
        "pass_active": info["contact_seconds"] > 0.0,
        "data_downlinked_mb": info["step_downlinked_mb"],
        "communication_failure": info["failure_reason"],
    }


__all__ = ["EventSatRewardFunction", "step_action_info"]
