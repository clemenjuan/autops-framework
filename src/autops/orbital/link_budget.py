"""Pure ground-link closure helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

_SPHERICAL_EARTH_RADIUS_KM = 6_371.0


@dataclass(frozen=True, slots=True)
class LinkDirection:
    """One configured RF direction; no mission-specific defaults."""

    frequency_mhz: float
    transmit_power_dbm: float
    transmit_amplifier_gain_db: float
    transmit_antenna_gain_dbi: float
    transmit_cable_loss_db: float
    receive_antenna_gain_dbi: float
    receive_cable_loss_db: float
    receive_sensitivity_dbm: float


@dataclass(frozen=True, slots=True)
class GroundLinkConfig:
    downlink: LinkDirection
    uplink: LinkDirection
    atmosphere_loss_db: float
    pointing_loss_db: float

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> GroundLinkConfig:
        """Read the EventSat-style link-budget mapping without numeric defaults."""

        downlink = _mapping(config["downlink"], "downlink")
        uplink = _mapping(config["uplink"], "uplink")
        losses = _mapping(config["losses"], "losses")
        return cls(
            downlink=LinkDirection(
                frequency_mhz=float(downlink["frequency_mhz"]),
                transmit_power_dbm=float(downlink["sat_tx_power_dbm"]),
                transmit_amplifier_gain_db=0.0,
                transmit_antenna_gain_dbi=float(downlink["sat_antenna_gain_dbi"]),
                transmit_cable_loss_db=float(downlink["sat_cable_loss_db"]),
                receive_antenna_gain_dbi=float(downlink["gs_antenna_gain_dbi"]),
                receive_cable_loss_db=float(downlink["gs_cable_loss_db"]),
                receive_sensitivity_dbm=float(downlink["gs_sensitivity_dbm"]),
            ),
            uplink=LinkDirection(
                frequency_mhz=float(uplink["frequency_mhz"]),
                transmit_power_dbm=float(uplink["gs_tx_power_dbm"]),
                transmit_amplifier_gain_db=float(uplink["gs_pa_gain_db"]),
                transmit_antenna_gain_dbi=float(uplink["gs_antenna_gain_dbi"]),
                transmit_cable_loss_db=float(uplink["gs_cable_loss_db"]),
                receive_antenna_gain_dbi=float(uplink["sat_antenna_gain_dbi"]),
                receive_cable_loss_db=float(uplink["sat_cable_loss_db"]),
                receive_sensitivity_dbm=float(uplink["sat_sensitivity_dbm"]),
            ),
            atmosphere_loss_db=float(losses["atmosphere_db"]),
            pointing_loss_db=float(losses["pointing_error_db"]),
        )


@dataclass(frozen=True, slots=True)
class DirectionResult:
    path_loss_db: float
    received_power_dbm: float
    margin_db: float


@dataclass(frozen=True, slots=True)
class GroundLinkResult:
    slant_range_km: float
    downlink: DirectionResult
    uplink: DirectionResult


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def slant_range_km(
    altitude_km: float,
    elevation_deg: float,
    *,
    earth_radius_km: float = _SPHERICAL_EARTH_RADIUS_KM,
) -> float:
    """Range to a satellite over a spherical Earth."""

    if altitude_km <= 0.0 or earth_radius_km <= 0.0:
        raise ValueError("altitude_km and earth_radius_km must be positive")
    if not 0.0 <= elevation_deg <= 90.0:
        raise ValueError("elevation_deg must be in [0, 90]")
    sine = math.sin(math.radians(elevation_deg))
    return (
        math.sqrt(
            (earth_radius_km * sine) ** 2 + 2.0 * earth_radius_km * altitude_km + altitude_km**2
        )
        - earth_radius_km * sine
    )


def free_space_loss_db(distance_km: float, frequency_mhz: float) -> float:
    if distance_km <= 0.0 or frequency_mhz <= 0.0:
        raise ValueError("distance_km and frequency_mhz must be positive")
    return 20.0 * math.log10(distance_km) + 20.0 * math.log10(frequency_mhz) + 32.44


def _direction_result(
    config: LinkDirection,
    *,
    distance_km: float,
    shared_loss_db: float,
) -> DirectionResult:
    path_loss_db = free_space_loss_db(distance_km, config.frequency_mhz)
    received_power_dbm = (
        config.transmit_power_dbm
        + config.transmit_amplifier_gain_db
        + config.transmit_antenna_gain_dbi
        - config.transmit_cable_loss_db
        - path_loss_db
        - shared_loss_db
        + config.receive_antenna_gain_dbi
        - config.receive_cable_loss_db
    )
    return DirectionResult(
        path_loss_db=path_loss_db,
        received_power_dbm=received_power_dbm,
        margin_db=received_power_dbm - config.receive_sensitivity_dbm,
    )


def ground_link_budget(
    config: GroundLinkConfig,
    *,
    altitude_km: float,
    elevation_deg: float,
) -> GroundLinkResult:
    """Evaluate configured downlink and uplink closure at one geometry."""

    distance_km = slant_range_km(altitude_km, elevation_deg)
    shared_loss_db = config.atmosphere_loss_db + config.pointing_loss_db
    return GroundLinkResult(
        slant_range_km=distance_km,
        downlink=_direction_result(
            config.downlink,
            distance_km=distance_km,
            shared_loss_db=shared_loss_db,
        ),
        uplink=_direction_result(
            config.uplink,
            distance_km=distance_km,
            shared_loss_db=shared_loss_db,
        ),
    )
