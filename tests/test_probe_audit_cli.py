from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from autops.commands import main, parser
from autops.wm.audit import ProbeAudit, ProbeHeadResult


def test_audit_parser_exposes_frozen_feature_options() -> None:
    args = parser().parse_args(
        [
            "train",
            "audit",
            "trace.npz",
            "--checkpoint",
            "model.pt",
            "--features",
            "obs",
            "--output",
            "audit.json",
            "--window",
            "3",
            "--hidden",
            "32,16",
        ]
    )
    assert args.training_command == "audit"
    assert args.features == "obs"
    assert args.window == 3
    assert args.hidden == "32,16"


def test_audit_serialization_rejects_nonstandard_nan() -> None:
    result = ProbeHeadResult(np.nan, np.nan, np.nan, np.nan, np.nan, degenerate=True)
    payload = ProbeAudit(
        attributes={"constant": result},
        train_episodes=(0,),
        validation_episodes=(1,),
        feature_window=1,
        hidden=(4,),
        mlp_epochs=1,
    ).to_dict()
    encoded = json.dumps(payload, allow_nan=False)
    assert "NaN" not in encoded
    assert payload["attributes"]["constant"]["linear_r2"] is None


def test_observation_audit_command_writes_json(tmp_path: Path, monkeypatch, capsys) -> None:
    output = tmp_path / "audit.json"

    def fake_audit(*args, **kwargs):
        destination = kwargs["output"]
        destination.write_text('{"ok": true}\n', encoding="utf-8")
        return {"ok": True, "output": str(destination)}

    monkeypatch.setattr("autops.commands.audit_probe_decodability", fake_audit)
    assert (
        main(
            [
                "train",
                "audit",
                "trace.npz",
                "--checkpoint",
                "model.pt",
                "--features",
                "obs",
                "--output",
                str(output),
                "--hidden",
                "8,4",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert output.exists()
