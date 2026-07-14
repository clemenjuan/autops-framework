"""Representation protocol, decorator registry, and automatic discovery."""

from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, TypeVar

from autops.core.types import DecisionContext, SpaceSpec


class Representation(ABC):
    """Common plugin seam, including the future Gymnasium encoding boundary."""

    observation_space: SpaceSpec | None = None
    action_space: SpaceSpec | None = None

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._last_rationale: str | None = None

    @abstractmethod
    def encode_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def select_action(self, context: DecisionContext) -> dict[str, Any]:
        raise NotImplementedError

    def update(self, transition: dict[str, Any]) -> None:
        """Optional online hook; fixed benchmark representations leave it empty."""

        return None

    @property
    def last_rationale(self) -> str | None:
        return self._last_rationale

    def reset(self, seed: int | None = None) -> None:
        self._last_rationale = None


Plugin = type[Representation]
T = TypeVar("T", bound=Plugin)
_REGISTRY: dict[tuple[str, str, str], Plugin] = {}
_DISCOVERED_PACKAGES: set[str] = set()
_ENTRY_POINTS_DISCOVERED = False


def register(token: str, *, mission: str = "*", role: str = "any") -> Callable[[T], T]:
    key = (mission, token, role)

    def decorate(cls: T) -> T:
        existing = _REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise RuntimeError(f"Duplicate representation plugin {key}")
        _REGISTRY[key] = cls
        return cls

    return decorate


def _discover_package(package_name: str, *, optional: bool = False) -> None:
    if package_name in _DISCOVERED_PACKAGES:
        return
    try:
        package = importlib.import_module(package_name)
    except ModuleNotFoundError as exc:
        if optional and exc.name == package_name:
            return
        raise
    for module in pkgutil.walk_packages(package.__path__, f"{package_name}."):
        importlib.import_module(module.name)
    _DISCOVERED_PACKAGES.add(package_name)


def _discover_entry_points() -> None:
    global _ENTRY_POINTS_DISCOVERED
    if _ENTRY_POINTS_DISCOVERED:
        return
    for item in entry_points(group="autops.representations"):
        item.load()
    _ENTRY_POINTS_DISCOVERED = True


def discover_representations(mission: str | None = None) -> None:
    """Load built-ins and, when requested, one mission's plugin package."""

    _discover_package("autops.representations")
    if mission:
        _discover_package(f"autops.missions.{mission}", optional=True)
    _discover_entry_points()


def create_representation(
    mission: str,
    token: str,
    role: str,
    config: dict[str, Any] | None = None,
) -> Representation:
    discover_representations(mission)
    for key in (
        (mission, token, role),
        (mission, token, "any"),
        ("*", token, role),
        ("*", token, "any"),
    ):
        if plugin := _REGISTRY.get(key):
            return plugin(config)
    raise KeyError(
        f"No representation plugin for mission={mission!r}, token={token!r}, role={role!r}"
    )


def registered_plugins(mission: str | None = None) -> dict[tuple[str, str, str], Plugin]:
    discover_representations(mission)
    return dict(_REGISTRY)
