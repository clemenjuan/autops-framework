"""Lazy optional Orekit Eckstein-Hechler backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .fallback import data_capacity_mb
from .models import EclipseInterval, GroundPass, GroundStation, OrbitElements


class OrekitUnavailable(RuntimeError):
    """Raised when the optional Orekit backend cannot be initialized."""


def orekit_data_path() -> Path:
    """Locate the repository-bundled Orekit data archive."""

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "orekit-data.zip"
        if candidate.is_file():
            return candidate
    candidate = Path.cwd() / "orekit-data.zip"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError("orekit-data.zip was not found in the repository tree")


@lru_cache(maxsize=1)
def _bindings() -> SimpleNamespace:
    try:
        import jpype
        import orekit_jpype

        if not jpype.isJVMStarted():
            orekit_jpype.initVM()

        from orekit_jpype.pyhelpers import setup_orekit_data

        setup_orekit_data(filenames=str(orekit_data_path()), from_pip_library=False)

        from org.orekit.bodies import CelestialBodyFactory, GeodeticPoint, OneAxisEllipsoid
        from org.orekit.frames import FramesFactory, TopocentricFrame
        from org.orekit.orbits import KeplerianOrbit, PositionAngleType
        from org.orekit.propagation.analytical import (
            EcksteinHechlerPropagator,
            KeplerianPropagator,
        )
        from org.orekit.time import AbsoluteDate, TimeScalesFactory
        from org.orekit.utils import Constants, IERSConventions
    except Exception as exc:  # optional dependency and JVM errors share one boundary
        raise OrekitUnavailable(f"Orekit initialization failed: {exc}") from exc

    return SimpleNamespace(
        AbsoluteDate=AbsoluteDate,
        CelestialBodyFactory=CelestialBodyFactory,
        Constants=Constants,
        EcksteinHechlerPropagator=EcksteinHechlerPropagator,
        FramesFactory=FramesFactory,
        GeodeticPoint=GeodeticPoint,
        IERSConventions=IERSConventions,
        KeplerianOrbit=KeplerianOrbit,
        KeplerianPropagator=KeplerianPropagator,
        OneAxisEllipsoid=OneAxisEllipsoid,
        PositionAngleType=PositionAngleType,
        TimeScalesFactory=TimeScalesFactory,
        TopocentricFrame=TopocentricFrame,
    )


def is_available() -> bool:
    """Return whether Orekit, Java, and the bundled data can initialize."""

    try:
        _bindings()
    except (OrekitUnavailable, FileNotFoundError):
        return False
    return True


def _absolute_date(bindings: SimpleNamespace, value: datetime) -> Any:
    utc = bindings.TimeScalesFactory.getUTC()
    seconds = value.second + value.microsecond / 1_000_000.0
    return bindings.AbsoluteDate(
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        seconds,
        utc,
    )


@dataclass(frozen=True, slots=True)
class OrekitPropagator:
    """Opaque Orekit propagator with its effective analytical model."""

    raw: Any
    kind: str


def create_propagator(orbit: OrbitElements) -> OrekitPropagator:
    """Create the configured analytical propagator, preferring J2 EH."""

    bindings = _bindings()
    constants = bindings.Constants
    frame = bindings.FramesFactory.getEME2000()
    semi_major_axis_m = constants.WGS84_EARTH_EQUATORIAL_RADIUS + orbit.altitude_km * 1000.0
    keplerian = bindings.KeplerianOrbit(
        semi_major_axis_m,
        orbit.eccentricity,
        math.radians(orbit.inclination_deg),
        math.radians(orbit.arg_perigee_deg),
        math.radians(orbit.raan_deg),
        math.radians(orbit.true_anomaly_deg),
        bindings.PositionAngleType.TRUE,
        frame,
        _absolute_date(bindings, orbit.epoch),
        constants.WGS84_EARTH_MU,
    )
    if orbit.propagator == "keplerian":
        return OrekitPropagator(bindings.KeplerianPropagator(keplerian), "keplerian")

    raw = bindings.EcksteinHechlerPropagator(
        keplerian,
        constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        constants.WGS84_EARTH_MU,
        constants.WGS84_EARTH_C20,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    return OrekitPropagator(raw, "eckstein-hechler-j2")


def position_km(propagator: OrekitPropagator, elapsed_s: float) -> tuple[float, float, float]:
    """Return ECI position at elapsed episode seconds."""

    initial = propagator.raw.getInitialState().getDate()
    position = (
        propagator.raw.propagate(initial.shiftedBy(elapsed_s)).getPVCoordinates().getPosition()
    )
    return position.getX() / 1000.0, position.getY() / 1000.0, position.getZ() / 1000.0


def _sample_times(duration_s: float, sample_s: float) -> tuple[float, ...]:
    times = [index * sample_s for index in range(math.floor(duration_s / sample_s) + 1)]
    if not times or times[-1] < duration_s:
        times.append(duration_s)
    return tuple(times)


def eclipse_intervals(
    propagator: OrekitPropagator,
    *,
    duration_s: float,
    sample_s: float,
) -> tuple[EclipseInterval, ...]:
    """Sample the cylindrical Earth-shadow geometry into intervals."""

    bindings = _bindings()
    sun = bindings.CelestialBodyFactory.getSun()
    earth = _earth(bindings)
    initial = propagator.raw.getInitialState().getDate()
    start_s: float | None = None
    intervals: list[EclipseInterval] = []

    for elapsed_s in _sample_times(duration_s, sample_s):
        state = propagator.raw.propagate(initial.shiftedBy(elapsed_s))
        satellite = state.getPVCoordinates().getPosition()
        sun_position = sun.getPVCoordinates(state.getDate(), state.getFrame()).getPosition()
        cosine = satellite.dotProduct(sun_position) / (satellite.getNorm() * sun_position.getNorm())
        cosine = min(1.0, max(-1.0, cosine))
        in_shadow = cosine < 0.0 and (
            satellite.getNorm() * math.sqrt(max(0.0, 1.0 - cosine * cosine))
            < earth.getEquatorialRadius()
        )
        if in_shadow and start_s is None:
            start_s = elapsed_s
        elif not in_shadow and start_s is not None:
            if elapsed_s > start_s:
                intervals.append(EclipseInterval(start_s, elapsed_s))
            start_s = None

    if start_s is not None and duration_s > start_s:
        intervals.append(EclipseInterval(start_s, duration_s))
    return tuple(intervals)


def _earth(bindings: SimpleNamespace) -> Any:
    frame = bindings.FramesFactory.getITRF(bindings.IERSConventions.IERS_2010, True)
    return bindings.OneAxisEllipsoid(
        bindings.Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        bindings.Constants.WGS84_EARTH_FLATTENING,
        frame,
    )


def _ground_frame(bindings: SimpleNamespace, station: GroundStation) -> Any:
    point = bindings.GeodeticPoint(
        math.radians(station.latitude_deg),
        math.radians(station.longitude_deg),
        station.altitude_m,
    )
    return bindings.TopocentricFrame(_earth(bindings), point, "ground-station")


def _threshold_crossing_s(
    previous_s: float,
    previous_deg: float,
    current_s: float,
    current_deg: float,
    threshold_deg: float,
) -> float:
    if current_deg == previous_deg:
        return current_s
    fraction = (threshold_deg - previous_deg) / (current_deg - previous_deg)
    return previous_s + min(1.0, max(0.0, fraction)) * (current_s - previous_s)


def ground_passes(
    propagator: OrekitPropagator,
    station: GroundStation,
    *,
    duration_s: float,
    sample_s: float,
    downlink_rate_kbps: float,
) -> tuple[GroundPass, ...]:
    """Sample elevation and linearly interpolate AOS/LOS threshold crossings."""

    bindings = _bindings()
    frame = _ground_frame(bindings, station)
    initial = propagator.raw.getInitialState().getDate()
    previous: tuple[float, float] | None = None
    start_s: float | None = None
    max_elevation_deg = 0.0
    passes: list[GroundPass] = []

    def close(end_s: float) -> None:
        nonlocal start_s
        if start_s is None or end_s <= start_s:
            start_s = None
            return
        contact_s = end_s - start_s
        passes.append(
            GroundPass(
                start_s=start_s,
                end_s=end_s,
                max_elevation_deg=max_elevation_deg,
                data_budget_mb=data_capacity_mb(downlink_rate_kbps, contact_s),
            )
        )
        start_s = None

    for elapsed_s in _sample_times(duration_s, sample_s):
        state = propagator.raw.propagate(initial.shiftedBy(elapsed_s))
        position = state.getPVCoordinates().getPosition()
        tracking = frame.getTrackingCoordinates(position, state.getFrame(), state.getDate())
        elevation_deg = math.degrees(tracking.getElevation())

        if elevation_deg >= station.min_elevation_deg:
            if start_s is None:
                start_s = elapsed_s
                if previous is not None and previous[1] < station.min_elevation_deg:
                    start_s = _threshold_crossing_s(
                        previous[0],
                        previous[1],
                        elapsed_s,
                        elevation_deg,
                        station.min_elevation_deg,
                    )
                max_elevation_deg = elevation_deg
            else:
                max_elevation_deg = max(max_elevation_deg, elevation_deg)
        elif start_s is not None:
            end_s = elapsed_s
            if previous is not None and previous[1] >= station.min_elevation_deg:
                end_s = _threshold_crossing_s(
                    previous[0],
                    previous[1],
                    elapsed_s,
                    elevation_deg,
                    station.min_elevation_deg,
                )
            close(end_s)
        previous = elapsed_s, elevation_deg

    if start_s is not None:
        close(duration_s)
    return tuple(passes)
