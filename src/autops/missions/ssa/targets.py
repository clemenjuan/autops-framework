"""Seeded debris targets, optical access, and paired detection draws."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

EARTH_RADIUS_KM = 6371.0
MU_EARTH_KM3_S2 = 398600.4418
ELEMENT_EPOCH = datetime(2026, 6, 1, tzinfo=UTC)


@dataclass(frozen=True)
class Target:
    object_id: str
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    arg_perigee_deg: float
    true_anomaly_deg: float
    size_m: float = 1.0
    epoch: datetime = ELEMENT_EPOCH


@dataclass(frozen=True)
class OpticalAccess:
    object_id: str
    position_km: tuple[float, float, float]
    range_km: float
    angle_deg: float
    magnitude: float
    probability: float
    quality: float


def phase_function(phase_rad: float) -> float:
    """Diffuse Lambertian-sphere phase function."""

    phase = min(math.pi, max(0.0, float(phase_rad)))
    if phase >= math.pi:
        return 0.0
    value = (2.0 / (3.0 * math.pi**2)) * ((math.pi - phase) * math.cos(phase) + math.sin(phase))
    return max(0.0, value)


def apparent_magnitude(
    size_m: float,
    range_km: float,
    phase_rad: float,
    albedo: float = 0.13,
) -> float:
    if size_m <= 0.0 or range_km <= 0.0 or albedo <= 0.0:
        return math.inf
    phase = phase_function(phase_rad)
    flux_ratio = albedo * size_m**2 / (4.0 * (range_km * 1000.0) ** 2) * phase
    return -26.74 - 2.5 * math.log10(flux_ratio) if flux_ratio > 0.0 else math.inf


def sun_unit_eci(epoch_s: float, epoch: datetime = ELEMENT_EPOCH) -> tuple[float, float, float]:
    """Deterministic low-precision Earth-to-Sun direction in mean-equator ECI."""

    normalized_epoch = epoch.replace(tzinfo=UTC) if epoch.tzinfo is None else epoch
    when = normalized_epoch.astimezone(UTC) + timedelta(seconds=float(epoch_s))
    j2000 = datetime(2000, 1, 1, 12, tzinfo=UTC)
    days = (when - j2000).total_seconds() / 86400.0
    mean_longitude = math.radians((280.460 + 0.9856474 * days) % 360.0)
    mean_anomaly = math.radians((357.528 + 0.9856003 * days) % 360.0)
    longitude = mean_longitude + math.radians(
        1.915 * math.sin(mean_anomaly) + 0.020 * math.sin(2.0 * mean_anomaly)
    )
    obliquity = math.radians(23.439 - 0.0000004 * days)
    return unit(
        (
            math.cos(longitude),
            math.cos(obliquity) * math.sin(longitude),
            math.sin(obliquity) * math.sin(longitude),
        )
    )


def target_sunlit(position_km: Sequence[float], sun_hat: Sequence[float]) -> bool:
    """Cylindrical Earth-shadow gate."""

    position = tuple(float(value) for value in position_km)
    sun = unit(sun_hat)
    projection = dot(position, sun)
    perpendicular = tuple(
        value - projection * axis for value, axis in zip(position, sun, strict=True)
    )
    return projection > 0.0 or norm(perpendicular) > EARTH_RADIUS_KM


def detection_probability(magnitude: float, limit: float = 15.0, sigma: float = 0.5) -> float:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    scaled = (float(limit) - float(magnitude)) / float(sigma)
    if scaled >= 0.0:
        return 1.0 / (1.0 + math.exp(-scaled))
    exponential = math.exp(scaled)
    return exponential / (1.0 + exponential)


def detection_draw(seed: int, object_id: str, satellite_id: str, step: int) -> float:
    """Pure paired draw, invariant to policy history and RNG consumption."""

    key = f"ssa-detection-v1|{seed}|{object_id}|{satellite_id}|{step}"
    raw = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return raw / float(1 << 64)


def optical_accesses(
    observer_position_km: Sequence[float],
    observer_velocity_hat: Sequence[float],
    targets: Iterable[Target],
    target_positions_km: Mapping[str, Sequence[float]],
    sun_hat: Sequence[float],
    *,
    fov_half_angle_deg: float = 1.9,
    boresight_pitch_deg: float = 12.0,
    range_cap_km: float = 150.0,
    magnitude_limit: float = 15.0,
    magnitude_sigma: float = 0.5,
    albedo: float = 0.13,
) -> list[OpticalAccess]:
    observer = tuple(float(value) for value in observer_position_km)
    radial = unit(observer)
    velocity = unit(observer_velocity_hat)
    pitch = math.radians(boresight_pitch_deg)
    boresight = unit(
        tuple(
            math.cos(pitch) * v + math.sin(pitch) * r for v, r in zip(velocity, radial, strict=True)
        )
    )
    sun = unit(sun_hat)
    sizes = {target.object_id: target.size_m for target in targets}
    accesses: list[OpticalAccess] = []
    for object_id in sorted(target_positions_km):
        target_position = tuple(float(value) for value in target_positions_km[object_id])
        relative = tuple(
            target - origin for target, origin in zip(target_position, observer, strict=True)
        )
        distance = norm(relative)
        if distance <= 0.0 or distance > range_cap_km or object_id not in sizes:
            continue
        relative_hat = unit(relative)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, dot(boresight, relative_hat)))))
        if angle > fov_half_angle_deg or not target_sunlit(target_position, sun):
            continue
        phase = math.acos(max(-1.0, min(1.0, dot(sun, tuple(-v for v in relative_hat)))))
        magnitude = apparent_magnitude(sizes[object_id], distance, phase, albedo)
        probability = detection_probability(magnitude, magnitude_limit, magnitude_sigma)
        accesses.append(
            OpticalAccess(
                object_id=object_id,
                position_km=target_position,
                range_km=distance,
                angle_deg=angle,
                magnitude=magnitude,
                probability=probability,
                quality=magnitude_limit - magnitude,
            )
        )
    return accesses


def generate_catalog(
    count: int,
    seed: int,
    *,
    raan_center_deg: float,
    parent_altitude_km: float = 805.0,
    parent_inclination_deg: float = 98.6,
    raan_spread_deg: float = 0.3,
    along_track_sigma_ms: float = 13.0,
    normal_sigma_ms: float = 26.0,
    size_bounds_m: tuple[float, float] = (0.01, 0.10),
) -> list[Target]:
    parent_axis = EARTH_RADIUS_KM + parent_altitude_km
    minimum, maximum = size_bounds_m
    if count < 0 or parent_axis <= 0.0 or not 0.0 < minimum <= maximum:
        raise ValueError("invalid fragmentation catalog configuration")
    if raan_spread_deg < 0.0 or along_track_sigma_ms < 0.0 or normal_sigma_ms < 0.0:
        raise ValueError("catalog dispersions must be non-negative")
    speed_ms = math.sqrt(MU_EARTH_KM3_S2 / parent_axis) * 1000.0
    size_span = 1.0 - (maximum / minimum) ** -2.5
    rng = random.Random(seed)
    catalog: list[Target] = []
    for index in range(count):
        along = rng.gauss(0.0, along_track_sigma_ms)
        normal = rng.gauss(0.0, normal_sigma_ms)
        delta_axis = max(-25.0, min(25.0, 2.0 * parent_axis * along / speed_ms))
        delta_inc = max(-0.2, min(0.2, math.degrees(normal / speed_ms)))
        raan = (raan_center_deg + rng.uniform(-raan_spread_deg, raan_spread_deg)) % 360.0
        eccentricity = rng.uniform(0.0, 0.001)
        argument = rng.uniform(0.0, 360.0)
        anomaly = rng.uniform(0.0, 360.0)
        size = minimum * (1.0 - rng.random() * size_span) ** (-1.0 / 2.5)
        catalog.append(
            Target(
                object_id=f"rso_{index}",
                semi_major_axis_km=parent_axis + delta_axis,
                eccentricity=eccentricity,
                inclination_deg=parent_inclination_deg + delta_inc,
                raan_deg=raan,
                arg_perigee_deg=argument,
                true_anomaly_deg=anomaly,
                size_m=size,
            )
        )
    return catalog


def propagate_target(target: Target, epoch_s: float) -> tuple[float, float, float]:
    """Deterministic two-body fallback used when no orbital backend is available."""

    axis, eccentricity = target.semi_major_axis_km, target.eccentricity
    mean_motion = math.sqrt(MU_EARTH_KM3_S2 / axis**3)
    mean_zero = true_to_mean(math.radians(target.true_anomaly_deg), eccentricity)
    eccentric_anomaly = solve_kepler(
        (mean_zero + mean_motion * epoch_s) % (2.0 * math.pi), eccentricity
    )
    x_perifocal = axis * (math.cos(eccentric_anomaly) - eccentricity)
    y_perifocal = axis * math.sqrt(1.0 - eccentricity**2) * math.sin(eccentric_anomaly)
    return rotate_to_eci(
        x_perifocal,
        y_perifocal,
        math.radians(target.raan_deg),
        math.radians(target.inclination_deg),
        math.radians(target.arg_perigee_deg),
    )


def true_to_mean(true_anomaly: float, eccentricity: float) -> float:
    if eccentricity <= 0.0:
        return true_anomaly % (2.0 * math.pi)
    eccentric_anomaly = 2.0 * math.atan2(
        math.sqrt(1.0 - eccentricity) * math.sin(true_anomaly / 2.0),
        math.sqrt(1.0 + eccentricity) * math.cos(true_anomaly / 2.0),
    )
    return (eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly)) % (2.0 * math.pi)


def solve_kepler(mean_anomaly: float, eccentricity: float) -> float:
    eccentric_anomaly = mean_anomaly
    for _ in range(12):
        delta = (eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly) - mean_anomaly) / (
            1.0 - eccentricity * math.cos(eccentric_anomaly)
        )
        eccentric_anomaly -= delta
        if abs(delta) < 1e-12:
            break
    return eccentric_anomaly


def rotate_to_eci(
    x: float,
    y: float,
    raan: float,
    inclination: float,
    argument_perigee: float,
) -> tuple[float, float, float]:
    cos_o, sin_o = math.cos(raan), math.sin(raan)
    cos_i, sin_i = math.cos(inclination), math.sin(inclination)
    cos_w, sin_w = math.cos(argument_perigee), math.sin(argument_perigee)
    return (
        (cos_o * cos_w - sin_o * sin_w * cos_i) * x + (-cos_o * sin_w - sin_o * cos_w * cos_i) * y,
        (sin_o * cos_w + cos_o * sin_w * cos_i) * x + (-sin_o * sin_w + cos_o * cos_w * cos_i) * y,
        sin_w * sin_i * x + cos_w * sin_i * y,
    )


def norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def unit(vector: Sequence[float]) -> tuple[float, float, float]:
    magnitude = norm(vector)
    if magnitude <= 0.0:
        raise ValueError("cannot normalize a zero vector")
    return tuple(float(value) / magnitude for value in vector)


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
