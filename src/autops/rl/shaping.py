"""Optional potential-based pipeline shaping for EventSat RL training.

A training aid only: ``k * (gamma * Phi(s') - Phi(s))`` with zero terminal
potential preserves optimal policies (Ng, Harada & Russell, ICML 1999), and it is
added by the RLlib bridge, never by the mission reward, so evaluated results do
not change. ``delivery`` credits compressed, OBC, and ground data at 1/3, 2/3, 1;
``raw_progress`` credits raw, compressed, OBC, and ground data at 1/4, 1/2, 3/4, 1
and interpolates compression progress of one raw product. All stages share one
raw-equivalent mission-target cap, filled from the furthest stage backwards.
Ported from the agentic framework (b6446e0).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

POTENTIALS = frozenset({"delivery", "raw_progress"})


@dataclass(frozen=True)
class PipelineShaping:
    potential: str = "delivery"
    scale: float = 1.0
    discount: float = 1.0

    def __post_init__(self) -> None:
        if self.potential not in POTENTIALS:
            raise ValueError("shaping potential must be delivery or raw_progress")
        if not math.isfinite(self.scale) or self.scale < 0.0:
            raise ValueError("shaping scale must be finite and non-negative")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("shaping discount must be between 0 and 1")

    def value(
        self, state: Mapping[str, Any], *, compression_ratio: float, downlink_target_mb: float
    ) -> float:
        """Normalised pipeline progress Phi(s) in [0, 1]."""

        if compression_ratio <= 0.0:
            raise ValueError("compression_ratio must be positive")
        target_raw = max(0.0, float(downlink_target_mb)) * compression_ratio
        if target_raw == 0.0:
            return 0.0
        remaining = target_raw
        credited = []
        for stage in (
            float(state.get("downlink_raw_equivalent_mb", 0.0)),
            float(state.get("obc_raw_equivalent_mb", 0.0)),
            float(state.get("jetson_compressed_mb", 0.0)) * compression_ratio,
        ):
            credit = min(max(0.0, stage), remaining)
            credited.append(credit)
            remaining -= credit
        ground, obc, compressed = credited
        if self.potential == "delivery":
            progress = ground + (2.0 / 3.0) * obc + (1.0 / 3.0) * compressed
            return min(1.0, max(0.0, progress / target_raw))
        raw_mb = max(0.0, float(state.get("jetson_raw_mb", 0.0)))
        raw = min(raw_mb, remaining)
        product_mb = max(0.0, float(state.get("observation_size_mb", 0.0)))
        fraction = min(1.0, max(0.0, float(state.get("compression_progress_fraction", 0.0))))
        processing = (
            min(product_mb, raw)
            if float(state.get("uncompressed_observations", 0.0)) >= 1.0
            and raw_mb >= product_mb > 0.0
            else 0.0
        )
        progress = (
            ground + 0.75 * obc + 0.5 * compressed + 0.25 * raw + 0.25 * fraction * processing
        )
        return min(1.0, max(0.0, progress / target_raw))

    def reward(
        self,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        *,
        compression_ratio: float,
        downlink_target_mb: float,
        is_final_step: bool,
    ) -> float:
        """``k * (gamma * Phi(s') - Phi(s))`` with zero potential after the final step."""

        kwargs = {"compression_ratio": compression_ratio, "downlink_target_mb": downlink_target_mb}
        following = 0.0 if is_final_step else self.value(after, **kwargs)
        return self.scale * (self.discount * following - self.value(before, **kwargs))


def pipeline_state(environment: Any) -> dict[str, Any]:
    """Physical EventSat pipeline plus the fields the potential reads."""

    state = environment.state
    return {
        **state.pipeline(),
        "observation_size_mb": float(environment.config["storage"]["observation_size_mb"]),
        "compression_progress_fraction": min(
            1.0, state.compression_progress / max(1, environment.compression_steps)
        ),
    }


__all__ = ["POTENTIALS", "PipelineShaping", "pipeline_state"]
