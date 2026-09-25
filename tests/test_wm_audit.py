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


def test_only_zero_one_targets_are_scored_as_binary() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    flags = rng.integers(0, 2, size=(4, 60, 1)).astype(np.float32)
    # A science flow completes 60 s of a 3600 s hour per observing step: {0, 1/60}.
    targets = np.concatenate([flags / 60.0, flags], axis=-1)
    audit = compare_probe_heads(
        flags,
        targets,
        attribute_names=("science_progress", "flag"),
        episodes=EpisodeSplit((0, 1), (2, 3)),
        hidden=(8,),
        mlp_epochs=5,
        seed=1,
    )
    science, flag = audit.attributes["science_progress"], audit.attributes["flag"]
    assert science.positive_rate is None and science.linear_auc is None
    assert science.linear_r2 > 0.99
    assert 0.3 < flag.positive_rate < 0.7 and flag.linear_auc == 1.0
