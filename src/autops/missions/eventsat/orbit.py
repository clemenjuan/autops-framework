"""EventSat orbit, fallback-geometry, and ground-station inputs from the mission file."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from autops.orbital import GroundStation, OrbitElements, SimplifiedModel


def orbit_elements(config: dict[str, Any]) -> OrbitElements:
    orbit = config["orbit"]
    epoch = datetime.fromisoformat(config["simulation"]["epoch"].replace("Z", "+00:00"))
    return OrbitElements(
        altitude_km=float(orbit["altitude_km"]),
        eccentricity=float(orbit["eccentricity"]),
        inclination_deg=float(orbit["inclination_deg"]),
        raan_deg=float(orbit["raan_deg"]),
        arg_perigee_deg=float(orbit["arg_perigee_deg"]),
        true_anomaly_deg=float(orbit["true_anomaly_deg"]),
        epoch=epoch,
        propagator=str(orbit["propagator"]),
    )


def fallback_model(config: dict[str, Any]) -> SimplifiedModel:
    orbit = config["orbit"]
    passes = config["communications"]["passes"]
    return SimplifiedModel(
        orbital_period_s=float(orbit["orbital_period_s"]),
        eclipse_fraction=float(orbit["eclipse_fraction"]),
        passes_min_per_day=int(passes["min_per_day"]),
        passes_max_per_day=int(passes["max_per_day"]),
        pass_min_duration_s=float(passes["min_duration_s"]),
        pass_max_duration_s=float(passes["max_duration_s"]),
    )


def ground_station(config: dict[str, Any]) -> GroundStation:
    station = config["communications"]["ground_station"]
    return GroundStation(
        latitude_deg=float(station["latitude_deg"]),
        longitude_deg=float(station["longitude_deg"]),
        altitude_m=float(station.get("altitude_m", 0.0)),
        min_elevation_deg=float(station["min_elevation_deg"]),
    )


__all__ = ["fallback_model", "ground_station", "orbit_elements"]
