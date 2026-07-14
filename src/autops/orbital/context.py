"""Episode-level eclipse and ground-contact context."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from .fallback import simplified_eclipses, simplified_ground_passes
from .models import (
    EclipseInterval,
    GroundPass,
    GroundStation,
    OrbitElements,
    SimplifiedModel,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OrbitalContext:
    """Precomputed physical event intervals for one episode."""

    eclipses: tuple[EclipseInterval, ...]
    ground_passes: tuple[GroundPass, ...]
    backend: Literal["orekit", "simplified"]
    propagator_kind: str
    step_s: float
    duration_s: float

    def __post_init__(self) -> None:
        if self.step_s <= 0.0 or self.duration_s <= 0.0:
            raise ValueError("step_s and duration_s must be positive")

    def _step_window(self, step: int) -> tuple[float, float]:
        if step < 0:
            raise ValueError("step must be non-negative")
        start_s = step * self.step_s
        return start_s, min(start_s + self.step_s, self.duration_s)

    def eclipse_seconds(self, step: int) -> float:
        """Shadow seconds overlapping an action step."""

        start_s, end_s = self._step_window(step)
        if start_s >= self.duration_s:
            return 0.0
        return min(
            end_s - start_s,
            sum(interval.overlap_s(start_s, end_s) for interval in self.eclipses),
        )

    def is_in_sunlight(self, step: int) -> bool:
        """Return sunlight state at the start of an action step."""

        instant_s = step * self.step_s
        return not any(interval.start_s <= instant_s < interval.end_s for interval in self.eclipses)

    def contact_seconds(self, step: int) -> float:
        """Physical contact seconds overlapping an action step."""

        start_s, end_s = self._step_window(step)
        if start_s >= self.duration_s:
            return 0.0
        return min(
            end_s - start_s,
            sum(ground_pass.overlap_s(start_s, end_s) for ground_pass in self.ground_passes),
        )

    def is_ground_pass_active(self, step: int) -> bool:
        return self.contact_seconds(step) > 0.0

    def get_current_pass(self, step: int) -> GroundPass | None:
        """Return the first pass overlapping the action step."""

        start_s, end_s = self._step_window(step)
        return next(
            (
                ground_pass
                for ground_pass in self.ground_passes
                if ground_pass.overlap_s(start_s, end_s) > 0.0
            ),
            None,
        )

    def next_pass_contact_s(self, step: int) -> float:
        """Duration of the current pass, or next upcoming pass."""

        instant_s = step * self.step_s
        upcoming = [
            ground_pass for ground_pass in self.ground_passes if ground_pass.end_s > instant_s
        ]
        if not upcoming:
            return 0.0
        return min(upcoming, key=lambda ground_pass: ground_pass.start_s).duration_s

    def future_pass_contact_s(self, step: int, future_pass_number: int = 1) -> float:
        """Duration of a strictly future pass; an ongoing pass is excluded."""

        if future_pass_number < 1:
            raise ValueError("future_pass_number must be >= 1")
        current = self.get_current_pass(step)
        instant_s = step * self.step_s
        if current is None:
            future = [
                ground_pass for ground_pass in self.ground_passes if ground_pass.start_s > instant_s
            ]
        else:
            future = [
                ground_pass
                for ground_pass in self.ground_passes
                if ground_pass.start_s >= current.end_s
            ]
        future.sort(key=lambda ground_pass: ground_pass.start_s)
        if len(future) < future_pass_number:
            return 0.0
        return future[future_pass_number - 1].duration_s

    def remaining_contact_s(self, step: int) -> float:
        """Residual current plus all future contact through episode end."""

        instant_s = step * self.step_s
        return sum(
            ground_pass.overlap_s(instant_s, self.duration_s) for ground_pass in self.ground_passes
        )


def build_orbital_context(
    orbit: OrbitElements,
    fallback: SimplifiedModel,
    ground_station: GroundStation,
    *,
    downlink_rate_kbps: float,
    step_s: float,
    total_steps: int,
    seed: int,
    prefer_orekit: bool = True,
    require_orekit: bool = False,
) -> OrbitalContext:
    """Build Orekit events when possible, otherwise the seeded fallback."""

    if step_s <= 0.0 or total_steps <= 0:
        raise ValueError("step_s and total_steps must be positive")
    if downlink_rate_kbps < 0.0:
        raise ValueError("downlink_rate_kbps must be non-negative")
    duration_s = step_s * total_steps

    if prefer_orekit:
        try:
            from . import orekit

            propagator = orekit.create_propagator(orbit)
            eclipses = orekit.eclipse_intervals(
                propagator,
                duration_s=duration_s,
                sample_s=step_s,
            )
            passes = orekit.ground_passes(
                propagator,
                ground_station,
                duration_s=duration_s,
                sample_s=step_s,
                downlink_rate_kbps=downlink_rate_kbps,
            )
            return OrbitalContext(
                eclipses=eclipses,
                ground_passes=passes,
                backend="orekit",
                propagator_kind=propagator.kind,
                step_s=step_s,
                duration_s=duration_s,
            )
        except Exception as exc:
            if require_orekit:
                raise RuntimeError("required Orekit propagation failed") from exc
            logger.warning("Orekit unavailable or failed; using seeded fallback: %s", exc)
    elif require_orekit:
        raise ValueError("require_orekit cannot be true when prefer_orekit is false")

    return OrbitalContext(
        eclipses=simplified_eclipses(fallback, duration_s),
        ground_passes=simplified_ground_passes(
            fallback,
            duration_s=duration_s,
            step_s=step_s,
            downlink_rate_kbps=downlink_rate_kbps,
            seed=seed,
        ),
        backend="simplified",
        propagator_kind="phase-and-seeded-passes",
        step_s=step_s,
        duration_s=duration_s,
    )
