from __future__ import annotations

import pytest

from autops.wm.jepa import LeWMConfig, build_vector_jepa


def test_attention_output_preserves_canonical_dropout() -> None:
    torch = pytest.importorskip("torch")
    model = build_vector_jepa(LeWMConfig(dropout=0.1))
    attention = model.predictor.blocks[0].attention
    assert isinstance(attention.output, torch.nn.Sequential)
    assert isinstance(attention.output[0], torch.nn.Linear)
    assert isinstance(attention.output[1], torch.nn.Dropout)
    assert attention.output[1].p == pytest.approx(0.1)
