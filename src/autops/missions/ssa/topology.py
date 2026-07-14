"""Finite hierarchy helpers for organisation-ready SSA experiments."""

from __future__ import annotations


def build_leader_hierarchy(
    members: list[str] | tuple[str, ...],
    branching_factor: int,
) -> list[list[list[str]]]:
    """Group members bottom-up, terminating even for unary branching."""

    if branching_factor < 1:
        raise ValueError("branching_factor must be at least one")
    current = list(members)
    if not current:
        return []
    leaves = [[member] for member in current]
    if branching_factor == 1:
        return [leaves]
    levels = [leaves]
    while len(current) > 1:
        groups = [
            current[index : index + branching_factor]
            for index in range(0, len(current), branching_factor)
        ]
        levels.append(groups)
        current = [group[0] for group in groups]
    return levels
