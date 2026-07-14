from __future__ import annotations

import pytest

from autops.core.probe_audit import audit_probe_decodability


def test_latent_audit_requires_checkpoint(tmp_path) -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        audit_probe_decodability(tmp_path / "missing.npz", features="latents")
