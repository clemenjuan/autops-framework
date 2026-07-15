from __future__ import annotations

import numpy as np
import pytest

from autops.wm.audit import compare_probe_heads, rank_auc, stack_feature_history
from autops.wm.dataset import EpisodeSplit
from autops.wm.scoring import candidate_selection_metrics


def test_history_padding_never_crosses_episode_boundaries() -> None:
    features = np.asarray([[[1.0], [2.0]], [[10.0], [20.0]]], dtype=np.float32)
    stacked = stack_feature_history(features, 2)
    assert stacked.tolist() == [[[1.0, 1.0], [1.0, 2.0]], [[10.0, 10.0], [10.0, 20.0]]]


def test_rank_auc_handles_perfect_order_and_ties() -> None:
    assert rank_auc(np.asarray([0.0, 0.1, 0.9, 1.0]), np.asarray([0, 0, 1, 1])) == 1.0
    assert rank_auc(np.ones(4), np.asarray([0, 0, 1, 1])) == 0.5


def test_candidate_selection_metrics_compare_one_shared_bank_to_oracle() -> None:
    oracle = np.asarray([[0.0, 1.0, 2.0, 3.0], [4.0, 3.0, 2.0, 1.0]])
    scores = {
        "terminal_affine": np.asarray([[0.0, 1.0, 3.0, 2.0], [1.0, 2.0, 3.0, 4.0]]),
        "windowed_affine": oracle.copy(),
        "mlp": oracle.copy(),
    }

    evidence = candidate_selection_metrics(scores, oracle, elites=2)

    assert evidence["windowed_affine"]["top_elite_overlap"] == 1.0
    assert evidence["mlp"]["analytical_regret_mean"] == 0.0
    assert evidence["terminal_affine"]["top_elite_overlap"] == pytest.approx(0.5)
    assert evidence["terminal_affine"]["analytical_regret_mean"] == pytest.approx(2.0)


def test_mlp_reveals_nonlinear_xor_gap() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    features = rng.choice([-1.0, 1.0], size=(6, 80, 2)).astype(np.float32)
    targets = (features[..., 0] * features[..., 1] > 0).astype(np.float32)[..., None]
    audit = compare_probe_heads(
        features,
        targets,
        attribute_names=("xor",),
        episodes=EpisodeSplit((0, 1, 2, 3), (4, 5)),
        hidden=(16, 16),
        mlp_epochs=120,
        learning_rate=5e-3,
        seed=3,
    )
    result = audit.attributes["xor"]
    assert result.mlp_r2 > 0.8
    assert result.mlp_minus_linear_r2 > 0.5
    assert result.mlp_auc == 1.0
