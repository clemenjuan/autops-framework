"""Single experiment specification and matrix-coordinate expansion.

The matrix is deliberately explicit: representation tokens map to substrate,
action space, shield, and implementation status in ``configs/matrix.yaml``.
There are no generated cell files and no compatibility aliases.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


def repository_root() -> Path:
    override = os.environ.get("AUTOPS_ROOT")
    return Path(override).expanduser().resolve() if override else Path(__file__).parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if not all(parts):
        raise ValueError(f"Invalid override key: {dotted_key!r}")
    target = mapping
    for part in parts[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {dotted_key!r}: {part!r} is not a mapping")
        target = child
    target[parts[-1]] = value


def parse_overrides(items: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        set_nested(result, key.strip(), yaml.safe_load(raw))
    return result


@dataclass(frozen=True)
class Coordinate:
    mission: str
    organisation: str
    paradigm: str
    representation: str | None = None
    onboard: str | None = None
    ground: str | None = None

    @property
    def name(self) -> str:
        if self.paradigm == "ah":
            return f"{self.mission}_{self.organisation}_ah_{self.onboard}_{self.ground}"
        return f"{self.mission}_{self.organisation}_{self.paradigm}_{self.representation}"

    def canonical(self) -> str:
        if self.paradigm == "ah":
            return "/".join(
                [
                    self.mission,
                    self.organisation,
                    self.paradigm,
                    self.onboard or "",
                    self.ground or "",
                ]
            )
        return "/".join([self.mission, self.organisation, self.paradigm, self.representation or ""])


def parse_coordinate(value: str) -> Coordinate:
    parts = [part.strip().lower() for part in value.strip("/").split("/")]
    if len(parts) == 4 and parts[2] == "ah" and "+" in parts[3]:
        onboard, ground = parts[3].split("+", 1)
        return Coordinate(parts[0], parts[1], parts[2], onboard=onboard, ground=ground)
    if len(parts) == 5 and parts[2] == "ah":
        return Coordinate(parts[0], parts[1], parts[2], onboard=parts[3], ground=parts[4])
    if len(parts) == 4 and parts[2] != "ah":
        return Coordinate(parts[0], parts[1], parts[2], representation=parts[3])
    raise ValueError(
        "Coordinate must be mission/organisation/paradigm/representation or "
        "mission/organisation/ah/onboard/ground"
    )


class ExperimentSpec(BaseModel):
    """The only public experiment configuration model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    coordinate: str
    name: str
    mission: str
    organisation: str
    paradigm: str
    representation: str | None = None
    onboard_representation: str | None = None
    ground_representation: str | None = None
    episodes: int = Field(default=1, ge=1)
    steps: int = Field(default=10080, ge=1)
    timestep_s: float = Field(default=60.0, gt=0)
    seeds: list[int] = Field(default_factory=lambda: [42])
    constellation_size: int = Field(default=1, ge=1)
    mission_config: dict[str, Any] = Field(default_factory=dict)
    representation_config: dict[str, Any] = Field(default_factory=dict)
    paradigm_config: dict[str, Any] = Field(default_factory=dict)
    organisation_config: dict[str, Any] = Field(default_factory=dict)
    metrics_config: dict[str, Any] = Field(default_factory=dict)
    output_root: Path = Path("results")

    @model_validator(mode="after")
    def validate_contract(self) -> ExperimentSpec:
        if len(self.seeds) != self.episodes:
            raise ValueError("seeds must contain exactly one paired seed per episode")
        if self.output_root.is_absolute():
            raise ValueError("output_root must be repository-relative")
        if self.paradigm == "ah":
            if not self.onboard_representation or not self.ground_representation:
                raise ValueError("ah requires explicit onboard and ground representations")
            if self.representation is not None:
                raise ValueError("ah must not use the single representation field")
        elif self.representation is None:
            raise ValueError(f"{self.paradigm} requires representation")
        return self

    @property
    def onboard_token(self) -> str | None:
        if self.paradigm == "ah":
            return self.onboard_representation
        return self.representation if self.paradigm == "ao" else None

    @property
    def ground_token(self) -> str | None:
        if self.paradigm == "ah":
            return self.ground_representation
        return self.representation if self.paradigm in {"ag", "conventional"} else None

    @property
    def onboard_uses_jetson(self) -> bool:
        return self.onboard_token in {"rl", "hrl", "lewm-cem"}


def _validate_coordinate(coord: Coordinate, matrix: dict[str, Any]) -> None:
    cell_definitions = matrix.get("representations", {})
    mission = (matrix.get("missions", {}) or {}).get(coord.mission)
    if mission is None:
        raise ValueError(f"Unknown mission {coord.mission!r}")
    if coord.organisation not in mission.get("organisations", []):
        raise ValueError(
            f"Organisation {coord.organisation!r} is not applicable to {coord.mission}"
        )
    rule = (mission.get("paradigms", {}) or {}).get(coord.paradigm)
    if rule is None:
        raise ValueError(f"Paradigm {coord.paradigm!r} is not runnable for {coord.mission}")
    tokens = [coord.onboard, coord.ground] if coord.paradigm == "ah" else [coord.representation]
    for token in tokens:
        definition = cell_definitions.get(token)
        if definition is None:
            raise ValueError(f"Unknown representation {token!r}")
        if not definition.get("implemented", False):
            raise ValueError(f"Representation {token!r} is reserved but not implemented")
    if coord.paradigm == "ah":
        if coord.onboard not in rule.get("onboard", []):
            raise ValueError(f"{coord.onboard!r} is not runnable in the ah onboard slot")
        if coord.ground not in rule.get("ground", []):
            raise ValueError(f"{coord.ground!r} is not runnable in the ah ground slot")
    else:
        slot = "onboard" if coord.paradigm == "ao" else "ground"
        if coord.representation not in rule.get(slot, []):
            raise ValueError(
                f"{coord.representation!r} is not runnable for {coord.mission}/{coord.paradigm}"
            )


def expand_coordinate(
    value: str,
    *,
    episodes: int = 1,
    steps: int | None = None,
    seeds: list[int] | None = None,
    constellation_size: int | None = None,
    overrides: dict[str, Any] | None = None,
    root: Path | None = None,
) -> ExperimentSpec:
    root = root or repository_root()
    matrix = load_yaml(root / "configs" / "matrix.yaml")
    coord = parse_coordinate(value)
    _validate_coordinate(coord, matrix)
    mission_config = load_yaml(root / "configs" / "missions" / f"{coord.mission}.yaml")
    if overrides:
        mission_config = deep_merge(mission_config, overrides.get("mission", {}))
    default_steps = int(mission_config.get("simulation", {}).get("max_steps", 10080))
    default_size = int(mission_config.get("constellation", {}).get("size", 1))
    episode_seeds = list(seeds) if seeds is not None else list(range(42, 42 + episodes))
    if len(episode_seeds) != episodes:
        raise ValueError("Number of seeds must equal episodes")
    return ExperimentSpec(
        coordinate=coord.canonical(),
        name=coord.name,
        mission=coord.mission,
        organisation=coord.organisation,
        paradigm=coord.paradigm,
        representation=coord.representation,
        onboard_representation=coord.onboard,
        ground_representation=coord.ground,
        episodes=episodes,
        steps=default_steps if steps is None else steps,
        timestep_s=float(mission_config.get("simulation", {}).get("timestep_s", 60.0)),
        seeds=episode_seeds,
        constellation_size=default_size if constellation_size is None else constellation_size,
        mission_config=mission_config,
        representation_config=(overrides or {}).get("representation", {}),
        paradigm_config=(overrides or {}).get("paradigm", {}),
        organisation_config=(overrides or {}).get("organisation", {}),
        metrics_config=(overrides or {}).get("metrics", {}),
        output_root=Path((overrides or {}).get("output_root", "results")),
    )
