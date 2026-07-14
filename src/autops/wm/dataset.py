"""Leakage-resistant window datasets and feature normalization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from autops.wm.schema import TraceDataset


@dataclass(frozen=True)
class EpisodeSplit:
    """Disjoint episode indices used by both training and validation."""

    train: tuple[int, ...]
    validation: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.train or not self.validation:
            raise ValueError("training and validation each require at least one episode")
        if set(self.train) & set(self.validation):
            raise ValueError("training and validation episodes must be disjoint")


@dataclass(frozen=True)
class FeatureNormalizer:
    """Per-feature statistics fitted only on training episodes."""

    obs_mean: np.ndarray
    obs_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def __post_init__(self) -> None:
        pairs = ((self.obs_mean, self.obs_std), (self.action_mean, self.action_std))
        for mean, std in pairs:
            if mean.ndim != 1 or std.shape != mean.shape:
                raise ValueError("normalizer means/stds must be matching vectors")
            if not np.isfinite(mean).all() or not np.isfinite(std).all():
                raise ValueError("normalizer statistics must be finite")
            if np.any(std <= 0.0):
                raise ValueError("normalizer standard deviations must be positive")

    def normalize_obs(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.obs_mean) / self.obs_std).astype(np.float32)

    def normalize_action(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.action_mean) / self.action_std).astype(np.float32)


def split_episodes(
    n_episodes: int, *, train_fraction: float = 0.9, seed: int = 3072
) -> EpisodeSplit:
    """Shuffle episode identities, never individual or overlapping windows."""

    if n_episodes < 2:
        raise ValueError("episode-disjoint validation requires at least two episodes")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between zero and one")
    order = np.random.default_rng(seed).permutation(n_episodes)
    n_train = min(n_episodes - 1, max(1, round(n_episodes * train_fraction)))
    return EpisodeSplit(
        train=tuple(sorted(int(v) for v in order[:n_train])),
        validation=tuple(sorted(int(v) for v in order[n_train:])),
    )


def _episode_rows(values: np.ndarray, episodes: Sequence[int]) -> np.ndarray:
    selected = values[np.asarray(episodes, dtype=np.int64)]
    return selected.reshape(-1, selected.shape[-1]).astype(np.float32)


def _statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def fit_normalizer(trace: TraceDataset, episodes: Sequence[int]) -> FeatureNormalizer:
    """Fit observation/action scales on a declared training episode set."""

    chosen = tuple(int(v) for v in episodes)
    if not chosen or min(chosen) < 0 or max(chosen) >= trace.n_episodes:
        raise ValueError("normalizer episodes must be valid trace episode indices")
    obs_mean, obs_std = _statistics(_episode_rows(trace.obs, chosen))
    action_mean, action_std = _statistics(_episode_rows(trace.action, chosen))
    return FeatureNormalizer(obs_mean, obs_std, action_mean, action_std)


class WindowDataset:
    """NumPy windows over complete selected episodes, including SSA satellite streams."""

    def __init__(
        self,
        trace: TraceDataset,
        episodes: Sequence[int],
        *,
        window: int,
        normalizer: FeatureNormalizer,
    ) -> None:
        if window <= 1 or window > trace.n_steps:
            raise ValueError("window must be in [2, trace.n_steps]")
        self.trace = trace
        self.episodes = tuple(int(v) for v in episodes)
        if not self.episodes or min(self.episodes) < 0 or max(self.episodes) >= trace.n_episodes:
            raise ValueError("window episodes must be valid trace episode indices")
        self.window = int(window)
        self.normalizer = normalizer
        satellites = (
            range(len(trace.metadata.satellite_ids)) if trace.metadata.mission == "ssa" else (None,)
        )
        self.index = [
            (episode, satellite, start)
            for episode in self.episodes
            for satellite in satellites
            for start in range(trace.n_steps - self.window + 1)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        episode, satellite, start = self.index[index]
        stop = start + self.window
        if satellite is None:
            obs = self.trace.obs[episode, start:stop]
            action = self.trace.action[episode, start:stop]
        else:
            obs = self.trace.obs[episode, start:stop, satellite]
            action = self.trace.action[episode, start:stop, satellite]
        return {
            "obs": self.normalizer.normalize_obs(obs),
            "action": self.normalizer.normalize_action(action),
            "episode": episode,
            "start": start,
        }


@dataclass(frozen=True)
class WindowSplit:
    episodes: EpisodeSplit
    normalizer: FeatureNormalizer
    train: WindowDataset
    validation: WindowDataset


def build_window_split(
    trace: TraceDataset,
    *,
    history: int = 3,
    predictions: int = 1,
    train_fraction: float = 0.9,
    seed: int = 3072,
) -> WindowSplit:
    """Create disjoint normalized train/validation windows for LeWM."""

    if history <= 0 or predictions <= 0:
        raise ValueError("history and predictions must be positive")
    episodes = split_episodes(trace.n_episodes, train_fraction=train_fraction, seed=seed)
    normalizer = fit_normalizer(trace, episodes.train)
    window = history + predictions
    return WindowSplit(
        episodes=episodes,
        normalizer=normalizer,
        train=WindowDataset(trace, episodes.train, window=window, normalizer=normalizer),
        validation=WindowDataset(trace, episodes.validation, window=window, normalizer=normalizer),
    )


__all__ = [
    "EpisodeSplit",
    "FeatureNormalizer",
    "WindowDataset",
    "WindowSplit",
    "build_window_split",
    "fit_normalizer",
    "split_episodes",
]
