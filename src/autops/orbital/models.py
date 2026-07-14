"""Typed inputs and interval outputs for orbital modelling."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class OrbitElements:
    """Classical orbit supplied by a mission configuration.

    Launch-lottery values are explicit fields so an episode can persist the
    sampled RAAN, argument of perigee, and true anomaly in its trace.
    """

    altitude_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_perigee_deg: float
    true_anomaly_deg: float
    epoch: datetime
    propagator: Literal["j2", "keplerian"]

    def __post_init__(self) -> None:
        _positive("altitude_km", self.altitude_km)
        if not 0.0 <= self.eccentricity < 1.0:
            raise ValueError("eccentricity must be in [0, 1)")
        if not 0.0 <= self.inclination_deg <= 180.0:
            raise ValueError("inclination_deg must be in [0, 180]")
        if self.epoch.tzinfo is None or self.epoch.utcoffset() is None:
            raise ValueError("epoch must be timezone-aware")
        for name in ("raan_deg", "arg_perigee_deg", "true_anomaly_deg"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")


def apply_launch_lottery(orbit: OrbitElements, seed: int) -> OrbitElements:
    """Return an orbit with reproducibly sampled orientation angles.

    Sampling order intentionally matches the mission lottery: RAAN, argument
    of perigee, then true anomaly.
    """

    rng = random.Random(seed)
    return replace(
        orbit,
        raan_deg=rng.uniform(0.0, 360.0),
        arg_perigee_deg=rng.uniform(0.0, 360.0),
        true_anomaly_deg=rng.uniform(0.0, 360.0),
    )


@dataclass(frozen=True, slots=True)
class GroundStation:
    """Ground-station geometry supplied by mission configuration."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    min_elevation_deg: float

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("latitude_deg must be in [-90, 90]")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError("longitude_deg must be in [-180, 180]")
        if not 0.0 <= self.min_elevation_deg <= 90.0:
            raise ValueError("min_elevation_deg must be in [0, 90]")


@dataclass(frozen=True, slots=True)
class SimplifiedModel:
    """Configured phase/eclipse and stochastic-pass fallback parameters."""

    orbital_period_s: float
    eclipse_fraction: float
    passes_min_per_day: int
    passes_max_per_day: int
    pass_min_duration_s: float
    pass_max_duration_s: float

    def __post_init__(self) -> None:
        _positive("orbital_period_s", self.orbital_period_s)
        if not 0.0 <= self.eclipse_fraction <= 1.0:
            raise ValueError("eclipse_fraction must be in [0, 1]")
        if self.passes_min_per_day < 0:
            raise ValueError("passes_min_per_day must be non-negative")
        if self.passes_max_per_day < self.passes_min_per_day:
            raise ValueError("passes_max_per_day must be >= passes_min_per_day")
        _positive("pass_min_duration_s", self.pass_min_duration_s)
        if self.pass_max_duration_s < self.pass_min_duration_s:
            raise ValueError("pass_max_duration_s must be >= pass_min_duration_s")


@dataclass(frozen=True, slots=True)
class TimeInterval:
    """Half-open interval ``[start_s, end_s)`` measured from episode start."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s):
            raise ValueError("interval bounds must be finite")
        if self.start_s < 0.0 or self.end_s <= self.start_s:
            raise ValueError("interval must have 0 <= start_s < end_s")

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def overlap_s(self, start_s: float, end_s: float) -> float:
        return max(0.0, min(self.end_s, end_s) - max(self.start_s, start_s))


@dataclass(frozen=True, slots=True)
class EclipseInterval(TimeInterval):
    """An interval in Earth's shadow."""


@dataclass(frozen=True, slots=True)
class GroundPass(TimeInterval):
    """A ground-contact interval with its physical transfer budget."""

    max_elevation_deg: float
    data_budget_mb: float
