"""Attitude settling shared by every mission with slewed pointing modes."""

from __future__ import annotations


def settle_mode(
    resolved: str,
    target: str,
    remaining: int,
    settling: int,
    maneuver_modes: set[str],
    *,
    mandatory_safe: bool,
) -> tuple[str, str, int, bool]:
    """Return effective mode, attitude target, countdown, and transition flag.

    A slew fixes its target when it starts; commands issued while settling are
    dropped, not queued. Only environment-enforced safety aborts the slew.
    """

    if mandatory_safe:
        return "safe", "safe", 0, False
    if remaining > 0:
        return "charging", target, remaining - 1, True
    maneuver = target != resolved and (resolved in maneuver_modes or target in maneuver_modes)
    if maneuver and settling > 0:
        return "charging", resolved, settling - 1, True
    return resolved, resolved, 0, False


__all__ = ["settle_mode"]
