"""Single categorical cross-entropy search used by LeWM evaluation and control."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

ProposalGuidance = Callable[[np.ndarray], np.ndarray]
CandidateSeeder = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CEMConfig:
    """Deployed EventSat defaults; action_dim remains mission-configurable."""

    horizon: int = 12
    action_dim: int = 7
    samples: int = 256
    elites: int = 32
    iterations: int = 4
    alpha: float = 0.7
    min_probability: float = 1e-4
    plan_hold: int = 12
    seed: int = 0

    def __post_init__(self) -> None:
        integer_fields = {
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "samples": self.samples,
            "elites": self.elites,
            "iterations": self.iterations,
            "plan_hold": self.plan_hold,
        }
        if any(value <= 0 for value in integer_fields.values()):
            raise ValueError(f"CEM integer fields must be positive: {integer_fields}")
        if self.action_dim < 2:
            raise ValueError("categorical CEM requires at least two actions")
        if self.elites > self.samples:
            raise ValueError("elites cannot exceed samples")
        if self.plan_hold > self.horizon:
            raise ValueError("plan_hold cannot exceed horizon")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")
        if not 0.0 < self.min_probability < 1.0:
            raise ValueError("min_probability must lie in (0, 1)")


@dataclass(frozen=True)
class CEMResult:
    action_sequence: np.ndarray
    score: float
    probabilities: np.ndarray
    elite_scores: np.ndarray


def _mask_matrix(
    config: CEMConfig,
    action_mask: np.ndarray | None,
    first_action_mask: np.ndarray | None,
) -> np.ndarray:
    mask = np.ones((config.horizon, config.action_dim), dtype=bool)
    if action_mask is not None:
        supplied = np.asarray(action_mask, dtype=bool)
        if supplied.shape == (config.action_dim,) or supplied.shape == mask.shape:
            mask[:] = supplied
        else:
            raise ValueError(f"action_mask must have shape {(config.action_dim,)} or {mask.shape}")
    if first_action_mask is not None:
        first = np.asarray(first_action_mask, dtype=bool)
        if first.shape != (config.action_dim,):
            raise ValueError(f"first_action_mask must have shape {(config.action_dim,)}")
        mask[0] &= first
    if np.any(mask.sum(axis=1) == 0):
        raise ValueError("action mask leaves a timestep without a valid action")
    return mask


def _normalize(probabilities: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64).copy()
    if values.shape != mask.shape or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and non-negative [horizon, action_dim]")
    values[~mask] = 0.0
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        values = np.where(mask, 1.0, 0.0)
        totals = values.sum(axis=1, keepdims=True)
    return values / totals


def initial_probabilities(
    config: CEMConfig,
    *,
    previous_solution: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Warm-start after the committed hold interval, never from stale actions."""

    valid = np.ones((config.horizon, config.action_dim), dtype=bool) if mask is None else mask
    uniform = np.full((config.horizon, config.action_dim), 1.0 / config.action_dim)
    if previous_solution is None:
        return _normalize(uniform, valid)
    previous = np.asarray(previous_solution, dtype=np.int64).reshape(-1)
    if previous.size == 0 or np.any((previous < 0) | (previous >= config.action_dim)):
        raise ValueError("previous_solution contains an invalid action")
    carry = previous[min(config.plan_hold, previous.size) :]
    if carry.size == 0:
        return _normalize(uniform, valid)
    if carry.size < config.horizon:
        carry = np.concatenate([carry, np.repeat(carry[-1], config.horizon - carry.size)])
    else:
        carry = carry[: config.horizon]
    probabilities = np.full((config.horizon, config.action_dim), 0.04 / (config.action_dim - 1))
    probabilities[np.arange(config.horizon), carry] = 0.96
    return _normalize(probabilities, valid)


def _sample(probabilities: np.ndarray, samples: int, rng: np.random.Generator) -> np.ndarray:
    sequences = np.empty((samples, probabilities.shape[0]), dtype=np.int64)
    actions = np.arange(probabilities.shape[1])
    for timestep, row in enumerate(probabilities):
        sequences[:, timestep] = rng.choice(actions, size=samples, p=row)
    return sequences


def _guided(
    probabilities: np.ndarray,
    mask: np.ndarray,
    proposal_guidance: ProposalGuidance | None,
) -> np.ndarray:
    if proposal_guidance is None:
        return probabilities
    return _normalize(proposal_guidance(probabilities.copy()), mask)


def _seeded(
    sequences: np.ndarray,
    mask: np.ndarray,
    seed_candidates: CandidateSeeder | None,
) -> np.ndarray:
    if seed_candidates is None:
        return sequences
    proposed = np.asarray(seed_candidates(sequences.copy()))
    if proposed.shape != sequences.shape or not np.issubdtype(proposed.dtype, np.integer):
        raise ValueError("seed_candidates must return an integer array matching sampled sequences")
    values = proposed.astype(np.int64, copy=False)
    if np.any((values < 0) | (values >= mask.shape[1])):
        raise ValueError("seed_candidates returned an invalid action")
    allowed = mask[np.arange(mask.shape[0])[None, :], values]
    if not np.all(allowed):
        raise ValueError("seed_candidates returned an action excluded by the CEM mask")
    return values


def categorical_cem(
    score: Callable[[np.ndarray], np.ndarray],
    config: CEMConfig | None = None,
    *,
    action_mask: np.ndarray | None = None,
    first_action_mask: np.ndarray | None = None,
    previous_solution: np.ndarray | None = None,
    initial: np.ndarray | None = None,
    proposal_guidance: ProposalGuidance | None = None,
    seed_candidates: CandidateSeeder | None = None,
    rng: np.random.Generator | None = None,
) -> CEMResult:
    """Maximize one score per integer action sequence.

    proposal_guidance transforms the initial and every post-elite
    distribution. seed_candidates transforms each freshly sampled batch
    before scoring; both hooks are optional and strictly revalidated.
    """

    config = config or CEMConfig()
    mask = _mask_matrix(config, action_mask, first_action_mask)
    probabilities = (
        initial_probabilities(config, previous_solution=previous_solution, mask=mask)
        if initial is None
        else _normalize(initial, mask)
    )
    probabilities = _guided(probabilities, mask, proposal_guidance)
    generator = rng or np.random.default_rng(config.seed)
    best_sequence = np.zeros(config.horizon, dtype=np.int64)
    best_score = -np.inf
    elite_scores = np.empty(0, dtype=np.float32)
    for _ in range(config.iterations):
        sequences = _sample(probabilities, config.samples, generator)
        sequences = _seeded(sequences, mask, seed_candidates)
        scores = np.asarray(score(sequences), dtype=np.float64).reshape(-1)
        if scores.shape != (config.samples,) or not np.isfinite(scores).all():
            raise ValueError("score must return one finite value per CEM sample")
        elite_index = np.argpartition(scores, -config.elites)[-config.elites :]
        elite_scores = scores[elite_index].astype(np.float32)
        iteration_best = int(elite_index[np.argmax(scores[elite_index])])
        if scores[iteration_best] > best_score:
            best_score = float(scores[iteration_best])
            best_sequence = sequences[iteration_best].copy()
        counts = np.full(
            (config.horizon, config.action_dim), config.min_probability, dtype=np.float64
        )
        for timestep in range(config.horizon):
            counts[timestep] += np.bincount(
                sequences[elite_index, timestep], minlength=config.action_dim
            )
        empirical = _normalize(counts, mask)
        probabilities = _normalize(
            config.alpha * empirical + (1.0 - config.alpha) * probabilities,
            mask,
        )
        probabilities = _guided(probabilities, mask, proposal_guidance)
    return CEMResult(
        action_sequence=best_sequence,
        score=best_score,
        probabilities=probabilities.astype(np.float32),
        elite_scores=elite_scores,
    )


def one_hot_sequences(sequences: np.ndarray, action_dim: int) -> np.ndarray:
    """Encode integer CEM candidates for the action-conditioned LeWM."""

    values = np.asarray(sequences, dtype=np.int64)
    if values.ndim != 2 or np.any((values < 0) | (values >= action_dim)):
        raise ValueError("sequences must be valid [sample, horizon] action indices")
    return np.eye(action_dim, dtype=np.float32)[values]


__all__ = [
    "CEMConfig",
    "CEMResult",
    "CandidateSeeder",
    "ProposalGuidance",
    "categorical_cem",
    "initial_probabilities",
    "one_hot_sequences",
]
