from __future__ import annotations

import pytest

from autops.core.probe_audit import audit_probe_decodability


def test_audit_rejects_missing_inputs(tmp_path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        audit_probe_decodability(tmp_path / "missing.npz", checkpoint_path=tmp_path / "missing.pt")
