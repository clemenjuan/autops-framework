"""Privileged EventSat event almanac derived from the episode's orbital truth.

These values describe future contact and eclipse timing. They serve ground
planning, the analytical oracle planner, and privileged trace labels; they are
never part of the onboard observation boundary (see ``observation.py``).
"""

from __future__ import annotations

from autops.orbital import OrbitalContext


def event_lookahead(
    orbit: OrbitalContext | None,
    step: int,
    *,
    timestep_s: float,
    period_steps: int,
    downlink_rate_kbps: float,
) -> dict[str, float | bool]:
    """Return next-event countdowns, the current pass, and following gaps."""

    now = step * timestep_s
    eclipses = orbit.eclipses if orbit else ()
    passes = orbit.ground_passes if orbit else ()
    future_eclipses = [item for item in eclipses if item.start_s > now]
    future_passes = [item for item in passes if item.start_s > now]
    current = orbit.get_current_pass(step) if orbit else None
    time_eclipse = (
        int((future_eclipses[0].start_s - now) / timestep_s) if future_eclipses else period_steps
    )
    time_pass = (
        int((future_passes[0].start_s - now) / timestep_s) if future_passes else period_steps
    )
    remaining_s = max(0.0, current.end_s - now) if current else 0.0
    current_contact_s = orbit.contact_seconds(step) if orbit else 0.0
    reference_end = current.end_s if current else now
    # A step can straddle a pass start: contact is already active while the
    # pass itself still begins later in the same step, which leaves the
    # current pass inside future_passes. Measuring the gap against it then
    # yields a negative span, so plans collapse to a single step exactly at
    # the pass entry where ground paradigms do their planning.
    upcoming = [item for item in future_passes if item.start_s >= reference_end]
    next_gap = period_steps
    following_gap = period_steps
    if upcoming:
        next_gap = max(1, int((upcoming[0].start_s - reference_end) / timestep_s))
    if len(upcoming) >= 2:
        following_gap = max(1, int((upcoming[1].start_s - upcoming[0].end_s) / timestep_s))
    capacity_s = orbit.future_pass_contact_s(step, 1) if orbit else 0.0
    return {
        "time_to_next_eclipse": float(time_eclipse),
        "time_to_next_pass": float(time_pass),
        "next_eclipse_known": bool(future_eclipses),
        "next_pass_known": bool(future_passes),
        "remaining_pass_duration": remaining_s / timestep_s,
        "remaining_pass_duration_s": remaining_s,
        "contact_window_seconds": current_contact_s,
        "next_gap_steps": float(next_gap),
        "following_gap_steps": float(following_gap),
        "planning_gap_steps": float(next_gap),
        "future_pass_capacity_mb": capacity_s * downlink_rate_kbps / 8.0 / 1000.0,
    }
