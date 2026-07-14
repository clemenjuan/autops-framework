"""Read-only-from-the-agent memory used by every runnable comparison cell."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol


class Memory(Protocol):
    def reset(self) -> None: ...

    def record(self, item: dict[str, Any]) -> None: ...

    def recent(self, count: int = 10) -> tuple[dict[str, Any], ...]: ...


@dataclass
class FixedMemory:
    """Bounded run memory with no representation-facing mutation API."""

    capacity: int = 100
    _history: deque[dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.capacity)

    def reset(self) -> None:
        self._history.clear()

    def record(self, item: dict[str, Any]) -> None:
        self._history.append(dict(item))

    def recent(self, count: int = 10) -> tuple[dict[str, Any], ...]:
        return tuple(list(self._history)[-max(0, count) :])
