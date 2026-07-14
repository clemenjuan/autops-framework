"""Constellation, ground-contact, and UHF inter-satellite geometry."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from autops.missions.ssa.targets import (
    EARTH_RADIUS_KM,
    Target,
    dot,
    norm,
    propagate_target,
    sun_unit_eci,
    target_sunlit,
    unit,
)

LIGHT_SPEED_M_S = 3.0e8
BOLTZMANN_J_K = 1.38e-23
EARTH_ROTATION_RAD_S = 7.2921159e-5


@dataclass(frozen=True)
class LinkBudget:
    frequency_hz: float = 437e6
    tx_power_w: float = 2.0
    tx_gain_db: float = 2.15
    rx_gain_db: float = 2.15
    tx_loss_db: float = 3.0
    rx_loss_db: float = 0.5
    bandwidth_hz: float = 9600.0
    symbol_rate_hz: float = 9600.0
    modulation_order: int = 4
    sensitivity_dbw: float = -132.0
    noise_temperature_k: float = 290.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> LinkBudget:
        fields = cls.__dataclass_fields__
        return cls(**{key: values[key] for key in fields if key in values})


def build_constellation_orbits(
    count: int,
    seed: int,
    *,
    altitude_km: float = 775.0,
    inclination_deg: float = 98.6,
    eccentricity: float = 0.001,
    spacing_deg: float = 2.0,
) -> dict[str, Target]:
    """Draw one paired-seed plane and place satellites along it."""

    rng = random.Random(seed)
    raan = rng.uniform(0.0, 360.0)
    argument_perigee = rng.uniform(0.0, 360.0)
    anomaly = rng.uniform(0.0, 360.0)
    return {
        f"sat_{index}": Target(
            object_id=f"sat_{index}",
            semi_major_axis_km=EARTH_RADIUS_KM + altitude_km,
            eccentricity=eccentricity,
            inclination_deg=inclination_deg,
            raan_deg=raan,
            arg_perigee_deg=argument_perigee,
            true_anomaly_deg=(anomaly + index * spacing_deg) % 360.0,
        )
        for index in range(count)
    }


def position_and_velocity_hat(
    orbit: Target,
    epoch_s: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    position = propagate_target(orbit, epoch_s)
    future = propagate_target(orbit, epoch_s + 1.0)
    velocity = tuple(after - before for before, after in zip(position, future, strict=True))
    return position, unit(velocity)


def tangent_for_static(position_km: tuple[float, float, float]) -> tuple[float, float, float]:
    radial = unit(position_km)
    reference = (0.0, 0.0, 1.0) if abs(radial[2]) < 0.9 else (1.0, 0.0, 0.0)
    tangent = (
        radial[1] * reference[2] - radial[2] * reference[1],
        radial[2] * reference[0] - radial[0] * reference[2],
        radial[0] * reference[1] - radial[1] * reference[0],
    )
    return unit(tangent)


def satellite_sunlit(position_km: tuple[float, float, float], epoch_s: float) -> bool:
    return target_sunlit(position_km, sun_unit_eci(epoch_s))


def ground_station_position_eci(
    epoch_s: float,
    latitude_deg: float,
    longitude_deg: float,
) -> tuple[float, float, float]:
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg) + EARTH_ROTATION_RAD_S * epoch_s
    return (
        EARTH_RADIUS_KM * math.cos(latitude) * math.cos(longitude),
        EARTH_RADIUS_KM * math.cos(latitude) * math.sin(longitude),
        EARTH_RADIUS_KM * math.sin(latitude),
    )


def ground_visible(
    satellite_position_km: tuple[float, float, float],
    epoch_s: float,
    *,
    latitude_deg: float = 48.0483,
    longitude_deg: float = 11.6567,
    min_elevation_deg: float = 10.0,
) -> bool:
    station = ground_station_position_eci(epoch_s, latitude_deg, longitude_deg)
    line_of_sight = tuple(
        satellite - site for satellite, site in zip(satellite_position_km, station, strict=True)
    )
    if norm(line_of_sight) <= 0.0:
        return False
    elevation = math.degrees(
        math.asin(max(-1.0, min(1.0, dot(unit(line_of_sight), unit(station)))))
    )
    return elevation >= min_elevation_deg


def ground_contact_seconds(
    position_at: Callable[[str, float], tuple[float, float, float]],
    satellite_id: str,
    start_s: float,
    end_s: float,
    config: Mapping[str, Any],
    *,
    resolution_s: float = 10.0,
) -> float:
    if bool(config.get("always_visible", False)):
        return max(0.0, end_s - start_s)
    contact = 0.0
    cursor = start_s
    while cursor < end_s - 1e-9:
        duration = min(max(1.0, resolution_s), end_s - cursor)
        midpoint = cursor + duration / 2.0
        if ground_visible(
            position_at(satellite_id, midpoint),
            midpoint,
            latitude_deg=float(config.get("latitude_deg", 48.0483)),
            longitude_deg=float(config.get("longitude_deg", 11.6567)),
            min_elevation_deg=float(config.get("min_elevation_deg", 10.0)),
        ):
            contact += duration
        cursor += duration
    return contact


def received_power_dbw(distance_m: float, budget: LinkBudget) -> float:
    if distance_m <= 0.0:
        raise ValueError("distance must be positive")
    transmit_dbw = 10.0 * math.log10(budget.tx_power_w)
    free_space_loss = 20.0 * math.log10(
        4.0 * math.pi * distance_m * budget.frequency_hz / LIGHT_SPEED_M_S
    )
    return (
        transmit_dbw
        + budget.tx_gain_db
        + budget.rx_gain_db
        - budget.tx_loss_db
        - budget.rx_loss_db
        - free_space_loss
    )


def effective_data_rate_bps(distance_m: float, budget: LinkBudget) -> float:
    received = received_power_dbw(distance_m, budget)
    if received < budget.sensitivity_dbw:
        return 0.0
    noise_dbw = 10.0 * math.log10(BOLTZMANN_J_K * budget.noise_temperature_k * budget.bandwidth_hz)
    snr_linear = 10.0 ** ((received - noise_dbw) / 10.0)
    shannon = budget.bandwidth_hz * math.log2(1.0 + snr_linear)
    modulation_cap = budget.symbol_rate_hz * math.log2(budget.modulation_order)
    ideal = min(shannon, modulation_cap)
    efficiency = modulation_cap / budget.bandwidth_hz
    ebn0 = snr_linear / efficiency
    bit_error_rate = math.erfc(math.sqrt(2.0 * ebn0)) / math.log2(budget.modulation_order)
    return ideal * (1.0 - bit_error_rate)


def link_capacity_bytes(
    position_at: Callable[[str, float], tuple[float, float, float]],
    left: str,
    right: str,
    start_s: float,
    end_s: float,
    budget: LinkBudget,
    *,
    resolution_s: float = 10.0,
    cache: MutableMapping[tuple[str, float], tuple[float, float, float]] | None = None,
) -> float:
    """Integrate link rate, reusing each satellite position at each midpoint."""

    def cached_position(satellite_id: str, epoch_s: float) -> tuple[float, float, float]:
        if cache is None:
            return position_at(satellite_id, epoch_s)
        key = (satellite_id, epoch_s)
        if key not in cache:
            cache[key] = position_at(satellite_id, epoch_s)
        return cache[key]

    capacity = 0.0
    cursor = start_s
    while cursor < end_s - 1e-9:
        duration = min(max(1.0, resolution_s), end_s - cursor)
        midpoint = cursor + duration / 2.0
        left_position = cached_position(left, midpoint)
        right_position = cached_position(right, midpoint)
        distance_m = max(
            1.0,
            norm(tuple(b - a for a, b in zip(left_position, right_position, strict=True))) * 1000.0,
        )
        capacity += effective_data_rate_bps(distance_m, budget) * duration / 8.0
        cursor += duration
    return capacity
