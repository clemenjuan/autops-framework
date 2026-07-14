from __future__ import annotations

import numpy as np
import pytest

from autops.wm.cem import CEMConfig, categorical_cem, initial_probabilities


def _score(sequences: np.ndarray) -> np.ndarray:
    return (sequences == 2).sum(axis=1).astype(np.float32)


def test_cem_defaults_match_deployed_search_contract():
    config = CEMConfig()
    assert (config.horizon, config.samples, config.elites, config.iterations) == (12, 256, 32, 4)
    assert config.alpha == 0.7
    assert config.plan_hold == 12


def test_cem_is_seed_deterministic_and_respects_masks():
    config = CEMConfig(horizon=5, samples=64, elites=8, iterations=3, plan_hold=2, seed=17)
    mask = np.array([True, False, True, False, False, False, False])

    first = categorical_cem(_score, config, action_mask=mask)
    second = categorical_cem(_score, config, action_mask=mask)

    np.testing.assert_array_equal(first.action_sequence, second.action_sequence)
    np.testing.assert_array_equal(first.probabilities, second.probabilities)
    assert set(first.action_sequence.tolist()) <= {0, 2}
    assert np.all(first.probabilities[:, ~mask] == 0.0)


def test_first_action_mask_does_not_restrict_future_steps():
    config = CEMConfig(horizon=4, samples=64, elites=8, iterations=3, plan_hold=1, seed=4)
    first_mask = np.array([True, False, False, False, False, False, False])
    result = categorical_cem(_score, config, first_action_mask=first_mask)

    assert result.action_sequence[0] == 0
    assert 2 in result.action_sequence[1:]


def test_warm_start_advances_by_plan_hold_and_resets_without_overlap():
    overlapping = CEMConfig(horizon=5, plan_hold=2)
    previous = np.array([0, 1, 2, 3, 4])
    probabilities = initial_probabilities(overlapping, previous_solution=previous)
    assert np.argmax(probabilities[0]) == 2
    assert np.argmax(probabilities[1]) == 3

    no_overlap = CEMConfig(horizon=5, plan_hold=5)
    reset = initial_probabilities(no_overlap, previous_solution=previous)
    np.testing.assert_allclose(reset, np.full((5, 7), 1.0 / 7.0))


def test_hold_cannot_exceed_horizon():
    with pytest.raises(ValueError, match="plan_hold"):
        CEMConfig(horizon=3, plan_hold=4)
