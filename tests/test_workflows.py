from __future__ import annotations

import pytest

from autops.core.workflows import matrix_coordinates


def test_matrix_sweep_expands_runtime_cells_without_reserved_rl() -> None:
    coordinates = matrix_coordinates("eventsat")
    assert "eventsat/sas/ag/symb" in coordinates
    assert "eventsat/sas/ao/lewm-cem" in coordinates
    assert "eventsat/sas/ah/lewm-cem/hllm-a" in coordinates
    assert not any("/rl" in coordinate or "/hrl" in coordinate for coordinate in coordinates)


def test_matrix_sweep_filters_and_fails_when_empty() -> None:
    assert matrix_coordinates("ssa", organisation="imas") == ["ssa/imas/ao/symb"]
    with pytest.raises(ValueError, match="no runnable"):
        matrix_coordinates("ssa", representation="llm-s")
