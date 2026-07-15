"""Small dependency-light LeWM trainer with episode-disjoint validation."""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autops.wm._contract_io import checkpoint_contract_kwargs
from autops.wm._contract_io import exact_fields as _strict_fields
from autops.wm.dataset import (
    EpisodeSplit,
    FeatureNormalizer,
    WindowSplit,
    build_window_split,
    fit_normalizer,
    split_episodes,
)
from autops.wm.jepa import LeWMConfig, build_vector_jepa, require_torch
from autops.wm.schema import (
    EVENTSAT_ACTIONS,
    SSA_ACTIONS,
    TRACE_SCHEMA_VERSION,
    TraceDataset,
    trace_sha256,
)

CHECKPOINT_SCHEMA_VERSION = "autops.lewm.checkpoint/v2"
ValidationCallback = Callable[[int, Mapping[str, float]], None]


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int = 150_000
    warmup_steps: int = 2_000
    batch_size: int = 64
    learning_rate: float = 5e-5
    weight_decay: float = 1e-3
    gradient_clip: float = 1.0
    train_fraction: float = 0.9
    seed: int = 3072
    validation_interval: int = 2_000
    validation_sample_size: int = 512
    train_loss_window: int = 1_000
    device: str = "cpu"

    def __post_init__(self) -> None:
        if (
            min(
                self.max_steps,
                self.batch_size,
                self.validation_interval,
                self.validation_sample_size,
                self.train_loss_window,
            )
            <= 0
        ):
            raise ValueError("training, validation, batch, and loss-window counts must be positive")
        if self.warmup_steps < 0 or self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer configuration")
        if self.gradient_clip <= 0.0 or not 0.0 < self.train_fraction < 1.0:
            raise ValueError("gradient_clip and train_fraction must be positive")


def _normalizer_dict(normalizer: FeatureNormalizer) -> dict[str, list[float]]:
    return {
        name: [float(value) for value in getattr(normalizer, name)]
        for name in ("obs_mean", "obs_std", "action_mean", "action_std")
    }


@dataclass(frozen=True)
class CheckpointContract:
    """Immutable recipe, data identity, split, and training evidence for LeWM."""

    model_config: LeWMConfig
    training_config: TrainingConfig
    mission: str
    observation_names: tuple[str, ...]
    action_names: tuple[str, ...]
    trace_sha256: str
    n_episodes: int
    n_steps: int
    episodes: EpisodeSplit
    normalizer: FeatureNormalizer
    train_loss: float
    validation_loss: float
    best_validation_step: int
    best_validation_loss: float
    validation_history: tuple[tuple[int, float], ...]
    trace_schema_version: str = TRACE_SCHEMA_VERSION
    schema_version: str = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported LeWM checkpoint schema {self.schema_version!r}")
        if self.trace_schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError(f"unsupported training trace schema {self.trace_schema_version!r}")
        if self.mission not in {"eventsat", "ssa"}:
            raise ValueError(f"unsupported training mission {self.mission!r}")
        object.__setattr__(
            self, "observation_names", tuple(str(value) for value in self.observation_names)
        )
        object.__setattr__(self, "action_names", tuple(str(value) for value in self.action_names))
        if (
            len(self.observation_names) != self.model_config.obs_dim
            or not self.observation_names
            or len(set(self.observation_names)) != len(self.observation_names)
        ):
            raise ValueError("checkpoint observation names do not define model.obs_dim")
        if (
            len(self.action_names) != self.model_config.action_dim
            or not self.action_names
            or len(set(self.action_names)) != len(self.action_names)
        ):
            raise ValueError("checkpoint action names do not define model.action_dim")
        expected_actions = {"eventsat": EVENTSAT_ACTIONS, "ssa": SSA_ACTIONS}[self.mission]
        if self.action_names != expected_actions:
            raise ValueError("checkpoint action semantics do not match its mission")
        if re.fullmatch(r"[0-9a-f]{64}", self.trace_sha256) is None:
            raise ValueError("trace_sha256 must be a lowercase SHA-256 digest")
        if (
            self.n_episodes < 2
            or self.n_steps < self.model_config.history + self.model_config.predictions
        ):
            raise ValueError("checkpoint trace dimensions cannot produce its model windows")
        if set(self.episodes.train) | set(self.episodes.validation) != set(range(self.n_episodes)):
            raise ValueError("checkpoint episode split must cover the complete training trace")
        if self.training_config.validation_sample_size < len(self.episodes.validation):
            raise ValueError("validation_sample_size must cover every validation episode")
        expected = split_episodes(
            self.n_episodes,
            train_fraction=self.training_config.train_fraction,
            seed=self.training_config.seed,
        )
        if self.episodes != expected:
            raise ValueError("checkpoint episode split does not match its training recipe")
        if self.normalizer.obs_mean.shape != (
            self.model_config.obs_dim,
        ) or self.normalizer.action_mean.shape != (self.model_config.action_dim,):
            raise ValueError("checkpoint normalizer dimensions do not match its model")
        if not all(
            math.isfinite(value)
            for value in (self.train_loss, self.validation_loss, self.best_validation_loss)
        ):
            raise ValueError("checkpoint training evidence must be finite")
        if (
            not self.validation_history
            or self.validation_history[-1][0] != self.training_config.max_steps
            or any(step <= 0 or not math.isfinite(loss) for step, loss in self.validation_history)
            or any(
                current[0] <= previous[0]
                for previous, current in zip(
                    self.validation_history, self.validation_history[1:], strict=False
                )
            )
        ):
            raise ValueError("checkpoint validation history must end at the final optimizer step")
        if self.validation_loss != self.validation_history[-1][1]:
            raise ValueError("checkpoint validation loss must equal its final history entry")
        expected_best_step, expected_best_loss = min(
            self.validation_history, key=lambda item: (item[1], item[0])
        )
        if (
            self.best_validation_step != expected_best_step
            or self.best_validation_loss != expected_best_loss
        ):
            raise ValueError("checkpoint best validation evidence disagrees with its history")

    def validate_trace(self, trace: TraceDataset) -> None:
        """Reject any trace other than the exact data used to train this checkpoint."""

        if trace.metadata.schema_version != self.trace_schema_version:
            raise ValueError("trace schema does not match checkpoint training data")
        if trace.metadata.mission != self.mission:
            raise ValueError("trace mission does not match checkpoint training data")
        if trace.metadata.observation_names != self.observation_names:
            raise ValueError("trace observation semantics do not match checkpoint training data")
        if trace.metadata.action_names != self.action_names:
            raise ValueError("trace action semantics do not match checkpoint training data")
        if (trace.n_episodes, trace.n_steps) != (self.n_episodes, self.n_steps):
            raise ValueError("trace dimensions do not match checkpoint training data")
        if trace_sha256(trace) != self.trace_sha256:
            raise ValueError("trace SHA-256 does not match checkpoint training data")
        fitted = fit_normalizer(trace, self.episodes.train)
        for name in ("obs_mean", "obs_std", "action_mean", "action_std"):
            if not np.array_equal(getattr(fitted, name), getattr(self.normalizer, name)):
                raise ValueError("checkpoint normalizer does not match its training trace")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_config": asdict(self.model_config),
            "training_config": asdict(self.training_config),
            "data": {
                "trace_schema_version": self.trace_schema_version,
                "mission": self.mission,
                "observation_names": list(self.observation_names),
                "action_names": list(self.action_names),
                "trace_sha256": self.trace_sha256,
                "n_episodes": self.n_episodes,
                "n_steps": self.n_steps,
                "train_episodes": list(self.episodes.train),
                "validation_episodes": list(self.episodes.validation),
                "normalizer": _normalizer_dict(self.normalizer),
            },
            "evidence": {
                "train_loss": self.train_loss,
                "validation_loss": self.validation_loss,
                "best_validation_step": self.best_validation_step,
                "best_validation_loss": self.best_validation_loss,
                "validation_history": [
                    {"step": step, "loss": loss} for step, loss in self.validation_history
                ],
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CheckpointContract:
        return cls(
            **checkpoint_contract_kwargs(
                payload,
                model_config_type=LeWMConfig,
                training_config_type=TrainingConfig,
            )
        )


@dataclass
class TrainingResult:
    model: Any
    model_config: LeWMConfig
    training_config: TrainingConfig
    windows: WindowSplit
    train_loss: float
    validation_loss: float
    best_validation_step: int
    best_validation_loss: float
    validation_history: tuple[tuple[int, float], ...]
    checkpoint_contract: CheckpointContract


def _batch(dataset: Any, indices: np.ndarray, torch: Any, device: Any) -> tuple[Any, Any]:
    rows = [dataset[int(index)] for index in indices]
    obs = torch.from_numpy(np.stack([row["obs"] for row in rows])).to(device)
    action = torch.from_numpy(np.stack([row["action"] for row in rows])).to(device)
    return obs, action


def _learning_rate_factor(step: int, config: TrainingConfig) -> float:
    if step < config.warmup_steps:
        return step / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _validation_loss(
    model: Any, windows: WindowSplit, config: TrainingConfig, torch: Any, device: Any
) -> float:
    """Evaluate a deterministic bounded sample containing every held-out episode."""

    model.eval()
    count = min(len(windows.validation), config.validation_sample_size)
    groups: dict[int, list[int]] = {}
    for index, (episode, _, _) in enumerate(windows.validation.index):
        groups.setdefault(episode, []).append(index)
    chosen: list[int] = []
    episodes = sorted(groups)
    for offset, episode in enumerate(episodes):
        share = count // len(episodes) + (offset < count % len(episodes))
        rows = groups[episode]
        positions = np.linspace(0, len(rows) - 1, num=min(share, len(rows)), dtype=np.int64)
        chosen.extend(rows[int(position)] for position in positions)
    indices = np.asarray(chosen, dtype=np.int64)
    weighted_loss = 0.0
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.manual_seed(config.seed)
        for start in range(0, len(indices), config.batch_size):
            stop = min(start + config.batch_size, len(indices))
            obs, action = _batch(windows.validation, indices[start:stop], torch, device)
            weighted_loss += float(model.loss(obs, action)["loss"]) * (stop - start)
    return weighted_loss / len(indices)


@dataclass(frozen=True)
class _TrainingEvidence:
    train_loss: float
    validation_loss: float
    best_validation_step: int
    best_validation_loss: float
    validation_history: tuple[tuple[int, float], ...]


def _optimize(
    model: Any,
    windows: WindowSplit,
    training: TrainingConfig,
    torch: Any,
    device: Any,
    rng: np.random.Generator,
    on_validation: ValidationCallback | None,
) -> _TrainingEvidence:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _learning_rate_factor(step, training)
    )
    model.train()
    recent_losses: deque[float] = deque(maxlen=training.train_loss_window)
    recent_prediction_losses: deque[float] = deque(maxlen=training.train_loss_window)
    recent_sigreg_losses: deque[float] = deque(maxlen=training.train_loss_window)
    history: list[tuple[int, float]] = []
    best_step, best_loss, best_state = 0, float("inf"), None

    def record_validation(step: int) -> None:
        nonlocal best_state, best_loss, best_step
        loss = _validation_loss(model, windows, training, torch, device)
        if not math.isfinite(loss):
            raise ValueError("validation produced a non-finite loss")
        history.append((step, loss))
        if loss < best_loss:
            best_step, best_loss = step, loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        if on_validation is not None:
            metrics = {
                "train/loss": float(np.mean(recent_losses)),
                "validation/loss": loss,
                "validation/best_loss": best_loss,
                "optimizer/learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            if recent_prediction_losses:
                metrics["train/prediction_loss"] = float(np.mean(recent_prediction_losses))
            if recent_sigreg_losses:
                metrics["train/sigreg_loss"] = float(np.mean(recent_sigreg_losses))
            on_validation(step, metrics)

    for step in range(1, training.max_steps + 1):
        indices = rng.integers(0, len(windows.train), size=training.batch_size)
        obs, action = _batch(windows.train, indices, torch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model.loss(obs, action)
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip)
        optimizer.step()
        scheduler.step()
        recent_losses.append(float(output["loss"].detach()))
        if "prediction_loss" in output:
            recent_prediction_losses.append(float(output["prediction_loss"].detach()))
        if "sigreg_loss" in output:
            recent_sigreg_losses.append(float(output["sigreg_loss"].detach()))
        if step % training.validation_interval == 0:
            record_validation(step)
            model.train()
    if not history or history[-1][0] != training.max_steps:
        record_validation(training.max_steps)
    if best_state is None:
        raise RuntimeError("training completed without a validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return _TrainingEvidence(
        train_loss=float(np.mean(recent_losses)),
        validation_loss=history[-1][1],
        best_validation_step=best_step,
        best_validation_loss=best_loss,
        validation_history=tuple(history),
    )


def train_lewm(
    trace: TraceDataset,
    *,
    model_config: LeWMConfig | None = None,
    training_config: TrainingConfig | None = None,
    on_validation: ValidationCallback | None = None,
) -> TrainingResult:
    """Train the canonical LeWM recipe; small max_steps values provide CPU smoke runs."""

    torch = require_torch()
    training = training_config or TrainingConfig()
    model_cfg = model_config or LeWMConfig(
        obs_dim=len(trace.metadata.observation_names),
        action_dim=len(trace.metadata.action_names),
    )
    if model_cfg.obs_dim != len(trace.metadata.observation_names):
        raise ValueError("model obs_dim does not match trace metadata")
    if model_cfg.action_dim != len(trace.metadata.action_names):
        raise ValueError("model action_dim does not match trace metadata")
    windows = build_window_split(
        trace,
        history=model_cfg.history,
        predictions=model_cfg.predictions,
        train_fraction=training.train_fraction,
        seed=training.seed,
    )
    if training.validation_sample_size < len(windows.episodes.validation):
        raise ValueError("validation_sample_size must cover every validation episode")
    torch.manual_seed(training.seed)
    rng = np.random.default_rng(training.seed)
    device = torch.device(training.device)
    model = build_vector_jepa(model_cfg).to(device)
    evidence = _optimize(model, windows, training, torch, device, rng, on_validation)
    contract = CheckpointContract(
        model_config=model_cfg,
        training_config=training,
        mission=trace.metadata.mission,
        observation_names=trace.metadata.observation_names,
        action_names=trace.metadata.action_names,
        trace_sha256=trace_sha256(trace),
        n_episodes=trace.n_episodes,
        n_steps=trace.n_steps,
        episodes=windows.episodes,
        normalizer=windows.normalizer,
        train_loss=evidence.train_loss,
        validation_loss=evidence.validation_loss,
        best_validation_step=evidence.best_validation_step,
        best_validation_loss=evidence.best_validation_loss,
        validation_history=evidence.validation_history,
    )
    return TrainingResult(
        model=model,
        model_config=model_cfg,
        training_config=training,
        windows=windows,
        train_loss=evidence.train_loss,
        validation_loss=evidence.validation_loss,
        best_validation_step=evidence.best_validation_step,
        best_validation_loss=evidence.best_validation_loss,
        validation_history=evidence.validation_history,
        checkpoint_contract=contract,
    )


def save_checkpoint(path_like: str | Path, result: TrainingResult) -> Path:
    """Persist the model recipe with weights for strict reconstruction."""

    torch = require_torch()
    contract = result.checkpoint_contract
    if (
        contract.model_config != result.model_config
        or contract.training_config != result.training_config
        or contract.train_loss != result.train_loss
        or contract.validation_loss != result.validation_loss
        or contract.best_validation_step != result.best_validation_step
        or contract.best_validation_loss != result.best_validation_loss
        or contract.validation_history != result.validation_history
    ):
        raise ValueError("training result and checkpoint contract disagree")
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "contract": contract.to_dict(),
            "state_dict": result.model.state_dict(),
        },
        path,
    )
    return path


def load_checkpoint(
    path_like: str | Path, *, device: str = "cpu"
) -> tuple[Any, CheckpointContract]:
    """Reconstruct a checkpoint with strict keys; incompatibility is an error."""

    torch = require_torch()
    payload = torch.load(Path(path_like), map_location=device, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("LeWM checkpoint root must be a mapping")
    _strict_fields(payload, {"schema_version", "contract", "state_dict"}, "LeWM checkpoint")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported LeWM checkpoint schema {payload['schema_version']!r}")
    if not isinstance(payload["contract"], Mapping):
        raise ValueError("LeWM checkpoint contract must be a mapping")
    contract = CheckpointContract.from_dict(payload["contract"])
    model = build_vector_jepa(contract.model_config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(torch.device(device)).eval()
    return model, contract


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointContract",
    "TrainingConfig",
    "TrainingResult",
    "load_checkpoint",
    "save_checkpoint",
    "train_lewm",
]
