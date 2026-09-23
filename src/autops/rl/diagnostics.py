"""Per-episode EventSat behaviour counters for training logs (agentic 58b9c5e).

Counters read step info and the onboard last-interval outcome only, so they work
under any reward; physical contact truth is used here, never as a policy input.
RLlib's callback copies them into ``hist_stats/eventsat_*`` for TensorBoard.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ray.rllib.algorithms.callbacks import DefaultCallbacks

FAILED_MODE_KEYS = {
    "payload_observe": "failed_observe",
    "payload_compress": "failed_compress",
    "payload_detect": "failed_detect",
    "payload_send": "failed_send",
    "communication": "failed_comm",
}
DIAGNOSTIC_KEYS = (
    "downlinked_mb",
    "failed_action_penalty",
    "observations",
    "compressions",
    "settling_steps",
    "comm_steps_in_contact",
    "comm_steps_no_contact",
    *FAILED_MODE_KEYS.values(),
    "failed_clamped",
    "final_raw_mb",
    "final_compressed_mb",
    "final_obc_mb",
)


def empty_diagnostics() -> dict[str, float]:
    return dict.fromkeys(DIAGNOSTIC_KEYS, 0.0)


def accumulate(
    diagnostics: dict[str, float], info: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    """Add one EventSat step; downlink and buffer levels are running totals."""

    mode = info["resolved_mode"]
    interval = metadata["last_interval"]
    diagnostics["downlinked_mb"] = float(info["data_downlinked_mb"])
    diagnostics["failed_action_penalty"] += float(info["failed_action_penalty"])
    diagnostics["observations"] += float(interval["captured_mb"] > 0.0)
    diagnostics["compressions"] += float(interval["compressed_products"] > 0)
    diagnostics["settling_steps"] += float(bool(info["in_transition"]))
    if mode == "communication":
        if info["failure_reason"] == "no_contact":
            diagnostics["comm_steps_no_contact"] += 1.0
        elif info["contact_seconds"] > 0.0:
            diagnostics["comm_steps_in_contact"] += 1.0
    if info["failed_action"]:
        # A failed charging step is an operational command clamped to charging.
        diagnostics[FAILED_MODE_KEYS.get(mode, "failed_clamped")] += 1.0
    diagnostics["final_raw_mb"] = float(metadata["jetson_raw_mb"])
    diagnostics["final_compressed_mb"] = float(metadata["jetson_compressed_mb"])
    diagnostics["final_obc_mb"] = float(metadata["obc_data_mb"])


class EpisodeDiagnostics(DefaultCallbacks):
    """Attach the environment's per-episode counters to each completed episode."""

    def on_episode_end(self, *, episode: Any, base_env: Any, env_index: int, **kwargs: Any):
        if isinstance(episode, Exception):
            return
        environment = base_env.get_sub_environments()[env_index]
        for key, value in environment.episode_diagnostics().items():
            episode.hist_data[f"eventsat_{key}"] = [value]


__all__ = ["DIAGNOSTIC_KEYS", "EpisodeDiagnostics", "accumulate", "empty_diagnostics"]
