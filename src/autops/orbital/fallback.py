"""Deterministic simplified orbital-event backend."""

from __future__ import annotations

import math
import random

from .models import EclipseInterval, GroundPass, SimplifiedModel

_DAY_S = 86_400.0


def data_capacity_mb(rate_kbps: float, contact_s: float) -> float:
    """Return transferable decimal megabytes at an effective bit rate."""

    if rate_kbps < 0.0 or contact_s < 0.0:
        raise ValueError("rate_kbps and contact_s must be non-negative")
    return rate_kbps * contact_s / 8_000.0


def simplified_eclipses(
    model: SimplifiedModel,
    duration_s: float,
) -> tuple[EclipseInterval, ...]:
    """Place the configured eclipse fraction at the start of each orbit."""

    if duration_s <= 0.0 or model.eclipse_fraction == 0.0:
        return ()
    shadow_s = model.orbital_period_s * model.eclipse_fraction
    eclipses: list[EclipseInterval] = []
    orbit_start_s = 0.0
    while orbit_start_s < duration_s:
        end_s = min(orbit_start_s + shadow_s, duration_s)
        if end_s > orbit_start_s:
            eclipses.append(EclipseInterval(orbit_start_s, end_s))
        orbit_start_s += model.orbital_period_s
    return tuple(eclipses)


def simplified_ground_passes(
    model: SimplifiedModel,
    *,
    duration_s: float,
    step_s: float,
    downlink_rate_kbps: float,
    seed: int,
) -> tuple[GroundPass, ...]:
    """Generate reproducible, step-aligned AOS times and true pass durations."""

    if duration_s <= 0.0:
        return ()
    if step_s <= 0.0:
        raise ValueError("step_s must be positive")
    rng = random.Random(seed)
    passes: list[GroundPass] = []
    day_count = max(1, math.ceil(duration_s / _DAY_S))

    for day in range(day_count):
        day_start_s = day * _DAY_S
        day_end_s = min((day + 1) * _DAY_S, duration_s)
        if day_start_s >= day_end_s:
            break
        count = rng.randint(model.passes_min_per_day, model.passes_max_per_day)
        for _ in range(count):
            requested_s = rng.uniform(
                model.pass_min_duration_s,
                model.pass_max_duration_s,
            )
            max_start_s = max(day_start_s, day_end_s - requested_s)
            start_index_min = math.ceil(day_start_s / step_s)
            start_index_max = math.floor(max_start_s / step_s)
            start_index = rng.randint(start_index_min, max(start_index_min, start_index_max))
            start_s = start_index * step_s
            end_s = min(start_s + requested_s, day_end_s)
            if end_s <= start_s:
                continue
            contact_s = end_s - start_s
            passes.append(
                GroundPass(
                    start_s=start_s,
                    end_s=end_s,
                    max_elevation_deg=0.0,
                    data_budget_mb=data_capacity_mb(downlink_rate_kbps, contact_s),
                )
            )

    return tuple(sorted(passes, key=lambda ground_pass: ground_pass.start_s))
