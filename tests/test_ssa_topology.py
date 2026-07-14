"""Finite hierarchy construction for organisation-ready SSA interfaces."""

from __future__ import annotations

import pytest

from autops.missions.ssa.topology import build_leader_hierarchy


def test_unary_branching_terminates_with_singleton_leaf_groups() -> None:
    members = ["sat_0", "sat_1", "sat_2"]
    assert build_leader_hierarchy(members, 1) == [[["sat_0"], ["sat_1"], ["sat_2"]]]


def test_binary_hierarchy_reaches_one_root_without_dropping_members() -> None:
    levels = build_leader_hierarchy([f"sat_{index}" for index in range(5)], 2)
    assert levels[0] == [["sat_0"], ["sat_1"], ["sat_2"], ["sat_3"], ["sat_4"]]
    assert levels[-1] == [["sat_0", "sat_4"]]
    assert len(levels) == 4


def test_hierarchy_rejects_nonpositive_branching_and_accepts_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_leader_hierarchy(["sat_0"], 0)
    assert build_leader_hierarchy([], 2) == []
