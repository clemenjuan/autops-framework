"""Public-safe provenance records embedded in world-model traces."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


@dataclass(frozen=True)
class TraceSource:
    """Provenance for one policy/configuration contributing trace episodes."""

    coordinate: str
    config_sha256: str
    source_revision: str
    source_kind: str
    source_dirty: bool
    orbital_backend: str
    episode_count: int
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        parts = self.coordinate.split("/")
        if (
            len(parts) not in {4, 5}
            or any(not part or part != part.strip().lower() for part in parts)
            or self.coordinate != self.coordinate.strip("/")
        ):
            raise ValueError("trace source coordinate must be a non-empty canonical coordinate")
        if not _SHA256_PATTERN.fullmatch(self.config_sha256):
            raise ValueError("trace source config_sha256 must be a lowercase SHA-256")
        if not _REVISION_PATTERN.fullmatch(self.source_revision):
            raise ValueError("trace source revision must be a lowercase Git/package digest")
        if self.source_kind not in {"git", "installed-package"}:
            raise ValueError("trace source kind must be git or installed-package")
        if not isinstance(self.source_dirty, bool):
            raise ValueError("trace source_dirty must be boolean")
        if self.source_kind == "installed-package" and self.source_dirty:
            raise ValueError("installed-package trace sources cannot be dirty")
        mission = parts[0]
        allowed_backends = {
            "eventsat": {"orekit", "simplified"},
            "ssa": {"not-applicable"},
        }
        if self.orbital_backend not in allowed_backends.get(mission, set()):
            raise ValueError("trace orbital_backend does not match its source mission")
        if (
            isinstance(self.episode_count, bool)
            or not isinstance(self.episode_count, int)
            or self.episode_count < 1
        ):
            raise ValueError("trace source episode_count must be a positive integer")
        seeds = tuple(self.seeds)
        if len(seeds) != self.episode_count:
            raise ValueError("trace source seeds must contain one value per episode")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise ValueError("trace source seeds must be non-negative integers")
        object.__setattr__(self, "seeds", seeds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate": self.coordinate,
            "config_sha256": self.config_sha256,
            "source_revision": self.source_revision,
            "source_kind": self.source_kind,
            "source_dirty": self.source_dirty,
            "orbital_backend": self.orbital_backend,
            "episode_count": self.episode_count,
            "seeds": list(self.seeds),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TraceSource:
        if not isinstance(payload, Mapping):
            raise ValueError("trace source must be a mapping")
        allowed = {
            "coordinate",
            "config_sha256",
            "source_revision",
            "source_kind",
            "source_dirty",
            "orbital_backend",
            "episode_count",
            "seeds",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"unknown trace source fields: {sorted(unknown)}")
        missing = allowed - set(payload)
        if missing:
            raise ValueError(f"missing trace source fields: {sorted(missing)}")
        episode_count = payload["episode_count"]
        if isinstance(episode_count, bool) or not isinstance(episode_count, int):
            raise ValueError("trace source episode_count must be an integer")
        seeds = payload["seeds"]
        if not isinstance(seeds, list):
            raise ValueError("trace source seeds must be an array")
        if not isinstance(payload["source_dirty"], bool):
            raise ValueError("trace source_dirty must be boolean")
        string_fields = (
            "coordinate",
            "config_sha256",
            "source_revision",
            "source_kind",
            "orbital_backend",
        )
        if any(not isinstance(payload[name], str) for name in string_fields):
            raise ValueError("trace source string fields must be strings")
        return cls(
            coordinate=payload["coordinate"],
            config_sha256=payload["config_sha256"],
            source_revision=payload["source_revision"],
            source_kind=payload["source_kind"],
            source_dirty=payload["source_dirty"],
            orbital_backend=payload["orbital_backend"],
            episode_count=episode_count,
            seeds=tuple(seeds),
        )


__all__ = ["TraceSource"]
