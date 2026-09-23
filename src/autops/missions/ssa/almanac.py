"""Per-satellite pass and eclipse countdowns, published only to RL representations.

The countdowns are future-event information. The symbolic policy and the SSA world
model do not receive them; the reinforcement-learning representation does, as in
the agentic framework, and results declare that asymmetry. They are computed once
per episode from the episode's pass windows and Keplerian sunlight geometry.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import TYPE_CHECKING, Any

from autops.missions.ssa.geometry import satellite_sunlit

if TYPE_CHECKING:
    from autops.missions.ssa.env import SSAEnvironment


def event_countdowns(
    env: SSAEnvironment, pass_windows: list[dict[str, Any]]
) -> dict[str, dict[str, list[int]]]:
    """Steps to the next pass and eclipse, and remaining pass steps, per satellite.

    A countdown without a later event in the episode is censored at one orbital period.
    """

    steps, period = env.max_steps, env.orbital_period_steps
    countdowns: dict[str, dict[str, list[int]]] = {}
    for satellite_id in env.satellite_ids:
        windows = sorted(
            (int(item["start_step"]), int(item["end_step"]))
            for item in pass_windows
            if item["satellite_id"] == satellite_id
        )
        sunlit = [
            satellite_sunlit(
                env.satellite_position(satellite_id, step * env.timestep_s), step * env.timestep_s
            )
            for step in range(steps)
        ]
        eclipse_starts = [step for step in range(1, steps) if sunlit[step - 1] and not sunlit[step]]
        pass_starts = [start for start, _ in windows]
        remaining = [0] * steps
        for start, end in windows:
            for step in range(start, min(end, steps - 1) + 1):
                remaining[step] = end - step + 1
        countdowns[satellite_id] = {
            "time_to_next_pass": [_next(pass_starts, step, period) for step in range(steps)],
            "time_to_next_eclipse": [_next(eclipse_starts, step, period) for step in range(steps)],
            "remaining_pass_duration": remaining,
        }
    return countdowns


def _next(starts: list[int], step: int, censored: int) -> int:
    index = bisect_right(starts, step)
    return starts[index] - step if index < len(starts) else censored


__all__ = ["event_countdowns"]
