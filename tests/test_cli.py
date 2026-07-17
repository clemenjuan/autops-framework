from __future__ import annotations

import json
from pathlib import Path

import pytest

from autops.commands import main, parser, seed_values
from autops.wm.schema import load_trace


def test_seed_range_is_inclusive() -> None:
    assert seed_values("42:51") == list(range(42, 52))
    assert seed_values("2,5") == [2, 5]


def test_parser_exposes_all_five_commands() -> None:
    help_text = parser().format_help()
    for command in ("run", "train", "sweep", "export", "board"):
        assert command in help_text


def test_board_defaults_to_paper_b_mission_results() -> None:
    args = parser().parse_args(["board"])

    assert args.manifest == Path("configs/papers/paper_b.yaml")


def test_sweep_dry_run_prints_real_coordinates(capsys) -> None:
    assert main(["sweep", "ssa", "--organisation", "imas", "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["coordinates"] == ["ssa/imas/ao/symb"]
    assert payload["completed"] == 0


def test_sweep_dry_run_still_validates_overrides() -> None:
    with pytest.raises(ValueError, match=r"mission\.no_such_field"):
        main(
            [
                "sweep",
                "ssa",
                "--organisation",
                "imas",
                "--dry-run",
                "--set",
                "mission.no_such_field=1",
            ]
        )


def test_export_command_writes_shared_trace(tmp_path: Path, capsys) -> None:
    output = tmp_path / "trace.npz"
    assert (
        main(
            [
                "export",
                "eventsat/sas/ao/symb",
                "--episodes",
                "1",
                "--steps",
                "3",
                "--no-orekit",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["trace"] == str(output)
    assert load_trace(output).n_steps == 3
