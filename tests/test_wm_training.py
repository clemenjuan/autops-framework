from __future__ import annotations

import numpy as np
import pytest

import autops.wm.training as training_module
from autops.wm.dataset import build_window_split, split_episodes
from autops.wm.jepa import LeWMConfig
from autops.wm.probes import fit_ridge_probe
from autops.wm.schema import (
    EVENTSAT_ACTIONS,
    EVENTSAT_OBSERVATIONS,
    EVENTSAT_STATES,
    TraceDataset,
    TraceMetadata,
    TraceSource,
    trace_sha256,
)
from autops.wm.training import (
    CHECKPOINT_SCHEMA_VERSION,
    TrainingConfig,
    load_checkpoint,
    save_checkpoint,
    train_lewm,
)


def _source(seeds: tuple[int, ...]) -> TraceSource:
    return TraceSource(
        coordinate="eventsat/sas/ao/symb",
        config_sha256="0" * 64,
        source_revision="0" * 40,
        source_kind="git",
        source_dirty=False,
        orbital_backend="simplified",
        episode_count=len(seeds),
        seeds=seeds,
    )


def _trace(obs: np.ndarray) -> TraceDataset:
    episodes, steps, obs_dim = obs.shape
    action_dim = len(EVENTSAT_ACTIONS)
    action_index = np.arange(episodes * steps).reshape(episodes, steps) % action_dim
    action = np.eye(action_dim, dtype=np.float32)[action_index]
    assert obs_dim == len(EVENTSAT_OBSERVATIONS)
    metadata = TraceMetadata.for_mission(
        "eventsat", timestep_s=60.0, sources=(_source(tuple(range(episodes))),)
    )
    return TraceDataset(
        metadata=metadata,
        obs=obs,
        action=action,
        state=np.zeros((episodes, steps, len(EVENTSAT_STATES)), dtype=np.float32),
        reward=np.zeros((episodes, steps), dtype=np.float32),
        mode=action_index,
        resolved_mode=action_index,
        forced_mode=np.zeros((episodes, steps), dtype=np.float32),
        episode_seed=np.arange(episodes),
        episode_id=np.arange(episodes),
    )


def test_window_split_is_episode_disjoint_and_train_normalized() -> None:
    split = split_episodes(5, train_fraction=0.6, seed=19)
    obs = np.zeros((5, 7, len(EVENTSAT_OBSERVATIONS)), dtype=np.float32)
    obs[np.asarray(split.train)] = 2.0
    obs[np.asarray(split.validation)] = 100.0

    windows = build_window_split(_trace(obs), history=3, predictions=1, train_fraction=0.6, seed=19)

    assert windows.episodes == split
    assert set(windows.train.episodes).isdisjoint(windows.validation.episodes)
    np.testing.assert_allclose(windows.normalizer.obs_mean, 2.0)
    np.testing.assert_allclose(windows.train[0]["obs"], 0.0)
    assert np.asarray(windows.validation[0]["obs"]).mean() > 90.0


def test_probe_validation_is_episode_disjoint_and_flags_degenerate_targets() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(8, 20, 5)).astype(np.float32)
    weight = np.array([1.5, -2.0, 0.4, 3.0, -1.0], dtype=np.float32)
    informative = features @ weight + 4.0
    targets = np.stack([informative, np.full_like(informative, 9.0)], axis=-1)

    with pytest.warns(RuntimeWarning, match="degenerate"):
        fit = fit_ridge_probe(
            features,
            targets,
            attribute_names=("informative", "constant"),
            train_fraction=0.75,
            seed=11,
        )

    assert set(fit.train_episodes).isdisjoint(fit.validation_episodes)
    assert fit.r2["informative"] > 0.999
    assert fit.rmse_over_std["informative"] < 0.01
    assert fit.degenerate == ("constant",)
    assert fit.target_std[1] == 1.0
    assert np.isnan(fit.r2["constant"])


def test_canonical_lewm_recipe_defaults() -> None:
    config = LeWMConfig()
    assert config.embed_dim == 192
    assert config.history == 3
    assert config.encoder_hidden_dim == 256
    assert config.predictor_depth == 4
    assert config.predictor_heads == 8
    assert config.predictor_head_dim == 48
    assert config.predictor_mlp_dim == 512
    assert config.projector_hidden_dim == 512
    assert config.sigreg_weight == 0.09
    assert config.sigreg_knots == 17
    assert config.sigreg_projections == 1024


def test_tiny_cpu_training_and_checkpoint_round_trip(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(23)
    trace = _trace(rng.normal(size=(4, 6, len(EVENTSAT_OBSERVATIONS))).astype(np.float32))
    model_config = LeWMConfig(
        obs_dim=len(EVENTSAT_OBSERVATIONS),
        action_dim=len(EVENTSAT_ACTIONS),
        embed_dim=16,
        encoder_hidden_dim=16,
        predictor_depth=1,
        predictor_heads=2,
        predictor_head_dim=8,
        predictor_mlp_dim=32,
        projector_hidden_dim=32,
        dropout=0.0,
        sigreg_knots=5,
        sigreg_projections=8,
    )
    training_config = TrainingConfig(
        max_steps=2,
        warmup_steps=0,
        batch_size=2,
        validation_interval=1,
        validation_sample_size=4,
        train_loss_window=2,
        train_fraction=0.75,
        seed=13,
    )

    updates: list[tuple[int, dict[str, float]]] = []
    result = train_lewm(
        trace,
        model_config=model_config,
        training_config=training_config,
        on_validation=lambda step, metrics: updates.append((step, dict(metrics))),
    )
    assert np.isfinite(result.train_loss)
    assert np.isfinite(result.validation_loss)
    assert [step for step, _ in updates] == [1, 2]
    assert all("validation/loss" in metrics for _, metrics in updates)
    assert set(result.windows.episodes.train).isdisjoint(result.windows.episodes.validation)

    checkpoint = save_checkpoint(tmp_path / "tiny.ckpt", result)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(payload) == {"schema_version", "contract", "state_dict"}
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    loaded, contract = load_checkpoint(checkpoint)
    assert contract.model_config == model_config
    assert contract.training_config == training_config
    assert contract.mission == "eventsat"
    assert contract.observation_names == trace.metadata.observation_names
    assert contract.action_names == trace.metadata.action_names
    assert contract.trace_sha256 == trace_sha256(trace)
    assert contract.episodes == result.windows.episodes
    np.testing.assert_array_equal(contract.normalizer.obs_mean, result.windows.normalizer.obs_mean)
    assert contract.train_loss == result.train_loss
    assert contract.validation_loss == result.validation_loss
    assert contract.best_validation_step == result.best_validation_step
    assert contract.best_validation_loss == result.best_validation_loss
    assert (contract.best_validation_step, contract.best_validation_loss) == min(
        contract.validation_history, key=lambda item: (item[1], item[0])
    )
    assert contract.validation_history == result.validation_history
    assert [step for step, _ in contract.validation_history] == [1, 2]
    contract.validate_trace(trace)

    changed = _trace(trace.obs.copy())
    changed.reward[0, 0] = 1.0
    with pytest.raises(ValueError, match="SHA-256"):
        contract.validate_trace(changed)

    invalid = tmp_path / "invalid.ckpt"
    torch.save({**payload, "unknown": True}, invalid)
    with pytest.raises(ValueError, match="unknown"):
        load_checkpoint(invalid)

    wrong_schema = tmp_path / "wrong-schema.ckpt"
    torch.save({**payload, "schema_version": "autops.lewm.checkpoint/v1"}, wrong_schema)
    with pytest.raises(ValueError, match="unsupported LeWM checkpoint schema"):
        load_checkpoint(wrong_schema)

    wrong_best = tmp_path / "wrong-best.ckpt"
    evidence = payload["contract"]["evidence"]
    bad_evidence = {**evidence, "best_validation_loss": -1.0}
    bad_contract = {**payload["contract"], "evidence": bad_evidence}
    torch.save({**payload, "contract": bad_contract}, wrong_best)
    with pytest.raises(ValueError, match="best validation evidence"):
        load_checkpoint(wrong_best)
    with torch.no_grad():
        obs = torch.zeros((1, model_config.history + 1, model_config.obs_dim))
        action = torch.zeros((1, model_config.history + 1, model_config.action_dim))
        action[..., 0] = 1.0
        assert torch.isfinite(loaded.loss(obs, action)["loss"])


def test_training_restores_lowest_validation_weights(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    class ScalarModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def loss(self, observations, actions):
            del observations, actions
            return {"loss": (self.weight - 1.0).square()}

    model = ScalarModel()
    validation_weights: list[float] = []
    validation_losses = iter((0.1, 0.9))

    def fake_validation(model, windows, config, torch_module, device):
        del windows, config, torch_module, device
        validation_weights.append(float(model.weight.detach()))
        return next(validation_losses)

    monkeypatch.setattr(training_module, "build_vector_jepa", lambda _: model)
    monkeypatch.setattr(training_module, "_validation_loss", fake_validation)
    trace = _trace(
        np.random.default_rng(71).normal(size=(4, 6, len(EVENTSAT_OBSERVATIONS))).astype(np.float32)
    )
    result = training_module.train_lewm(
        trace,
        model_config=LeWMConfig(obs_dim=25, action_dim=7),
        training_config=TrainingConfig(
            max_steps=2,
            warmup_steps=0,
            batch_size=2,
            learning_rate=0.1,
            weight_decay=0.0,
            train_fraction=0.75,
            validation_interval=1,
            validation_sample_size=4,
            train_loss_window=2,
            seed=29,
        ),
    )

    assert validation_weights[1] != pytest.approx(validation_weights[0])
    assert float(result.model.weight.detach()) == pytest.approx(validation_weights[0])
    assert result.best_validation_step == 1
    assert result.best_validation_loss == pytest.approx(0.1)
    assert result.validation_loss == pytest.approx(0.9)

    checkpoint = save_checkpoint(tmp_path / "best.ckpt", result)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert float(payload["state_dict"]["weight"]) == pytest.approx(validation_weights[0])
