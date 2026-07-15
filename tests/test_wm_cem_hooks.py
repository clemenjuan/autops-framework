from __future__ import annotations

import numpy as np
import pytest

from autops.wm.cem import CEMConfig, categorical_cem


def _score_action_two(sequences: np.ndarray) -> np.ndarray:
    return (sequences == 2).sum(axis=1).astype(np.float64)


def test_guidance_reapplies_and_candidate_seeds_every_iteration() -> None:
    config = CEMConfig(
        horizon=3,
        action_dim=3,
        samples=8,
        elites=2,
        iterations=4,
        plan_hold=1,
        seed=19,
    )
    guidance_inputs: list[np.ndarray] = []
    seeded_inputs: list[np.ndarray] = []
    scored_inputs: list[np.ndarray] = []

    def guide(probabilities: np.ndarray) -> np.ndarray:
        guidance_inputs.append(probabilities.copy())
        guided = np.full_like(probabilities, 0.1)
        guided[:, 2] = 0.8
        return guided

    def seed(sequences: np.ndarray) -> np.ndarray:
        seeded_inputs.append(sequences.copy())
        sequences[0] = 2
        return sequences

    def score(sequences: np.ndarray) -> np.ndarray:
        scored_inputs.append(sequences.copy())
        return _score_action_two(sequences)

    result = categorical_cem(
        score,
        config,
        proposal_guidance=guide,
        seed_candidates=seed,
    )

    assert len(guidance_inputs) == config.iterations + 1
    assert len(seeded_inputs) == config.iterations
    assert len(scored_inputs) == config.iterations
    assert all(np.all(sequences[0] == 2) for sequences in scored_inputs)
    np.testing.assert_allclose(
        result.probabilities,
        np.tile([0.1, 0.1, 0.8], (config.horizon, 1)),
    )
    np.testing.assert_array_equal(result.action_sequence, np.full(config.horizon, 2))


def test_absent_hooks_preserve_the_unmodified_search() -> None:
    config = CEMConfig(
        horizon=4,
        action_dim=3,
        samples=24,
        elites=4,
        iterations=3,
        plan_hold=1,
        seed=23,
    )

    baseline = categorical_cem(_score_action_two, config)
    explicit_none = categorical_cem(
        _score_action_two,
        config,
        proposal_guidance=None,
        seed_candidates=None,
    )

    np.testing.assert_array_equal(baseline.action_sequence, explicit_none.action_sequence)
    np.testing.assert_array_equal(baseline.probabilities, explicit_none.probabilities)
    np.testing.assert_array_equal(baseline.elite_scores, explicit_none.elite_scores)
    assert baseline.score == explicit_none.score


def test_candidate_seed_hook_is_strictly_masked_and_integer() -> None:
    config = CEMConfig(
        horizon=2,
        action_dim=3,
        samples=4,
        elites=1,
        iterations=1,
        plan_hold=1,
    )
    mask = np.asarray([True, False, True])

    def excluded(sequences: np.ndarray) -> np.ndarray:
        sequences[0, 0] = 1
        return sequences

    with pytest.raises(ValueError, match="excluded by the CEM mask"):
        categorical_cem(
            _score_action_two,
            config,
            action_mask=mask,
            seed_candidates=excluded,
        )

    with pytest.raises(ValueError, match="integer array"):
        categorical_cem(
            _score_action_two,
            config,
            seed_candidates=lambda sequences: sequences.astype(np.float64),
        )


def test_projected_candidates_are_the_bank_scored_and_selected() -> None:
    config = CEMConfig(
        horizon=3,
        action_dim=3,
        samples=16,
        elites=2,
        iterations=2,
        plan_hold=1,
        seed=29,
    )
    scored: list[np.ndarray] = []

    def project(sequences: np.ndarray) -> np.ndarray:
        sequences[sequences == 1] = 0
        return sequences

    def score(sequences: np.ndarray) -> np.ndarray:
        scored.append(sequences.copy())
        return (sequences == 2).sum(axis=1).astype(np.float64)

    result = categorical_cem(score, config, project_candidates=project)

    assert all(not np.any(bank == 1) for bank in scored)
    assert not np.any(result.action_sequence == 1)
