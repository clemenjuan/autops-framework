"""Small value objects shared across missions and representations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DecisionContext:
    state: dict[str, Any]
    observation: dict[str, Any]
    memory: Any
    step: int
    role: str = "onboard"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentStep:
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class SpaceSpec:
    """Dependency-free Gymnasium-compatible encoding seam; bounds are scalar or per feature."""

    shape: tuple[int, ...]
    dtype: str
    low: float | int | tuple[float, ...]
    high: float | int | tuple[float, ...]
    labels: tuple[str, ...] = ()
