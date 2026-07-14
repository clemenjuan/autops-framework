"""Optional-Torch vector JEPA used by the LeWM transition model.

Torch is imported only when a model is constructed, so the base framework does
not require the ``wm`` optional dependency.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class LeWMConfig:
    obs_dim: int = 25
    action_dim: int = 7
    embed_dim: int = 192
    history: int = 3
    predictions: int = 1
    encoder_hidden_dim: int = 256
    predictor_depth: int = 4
    predictor_heads: int = 8
    predictor_head_dim: int = 48
    predictor_mlp_dim: int = 512
    projector_hidden_dim: int = 512
    dropout: float = 0.1
    embedding_dropout: float = 0.0
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_projections: int = 1024

    def __post_init__(self) -> None:
        dimensions = (
            self.obs_dim,
            self.action_dim,
            self.embed_dim,
            self.history,
            self.predictions,
            self.encoder_hidden_dim,
            self.predictor_depth,
            self.predictor_heads,
            self.predictor_head_dim,
            self.predictor_mlp_dim,
            self.projector_hidden_dim,
            self.sigreg_knots,
            self.sigreg_projections,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("LeWM dimensions and counts must be positive")
        if self.sigreg_knots < 2:
            raise ValueError("sigreg_knots must be at least 2")
        if not 0.0 <= self.dropout < 1.0 or not 0.0 <= self.embedding_dropout < 1.0:
            raise ValueError("LeWM dropout probabilities must lie in [0, 1)")
        if self.sigreg_weight < 0.0:
            raise ValueError("sigreg_weight must be non-negative")


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("LeWM requires the optional dependency: uv sync --extra wm") from exc
    return torch


def _attention_type(torch: Any) -> type:
    nn = torch.nn
    functional = torch.nn.functional

    class Attention(nn.Module):
        def __init__(self, dim: int, heads: int, head_dim: int, dropout: float) -> None:
            super().__init__()
            inner = heads * head_dim
            self.heads = heads
            self.head_dim = head_dim
            self.dropout = dropout
            self.norm = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, 3 * inner, bias=False)
            self.output = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

        def forward(self, values: Any) -> Any:
            batch, steps, _ = values.shape
            qkv = self.qkv(self.norm(values)).chunk(3, dim=-1)
            q, k, v = (
                item.reshape(batch, steps, self.heads, self.head_dim).transpose(1, 2)
                for item in qkv
            )
            output = functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            return self.output(output.transpose(1, 2).reshape(batch, steps, -1))

    return Attention


def _feed_forward_type(torch: Any) -> type:
    nn = torch.nn

    class FeedForward(nn.Module):
        def __init__(self, dim: int, hidden: int, dropout: float) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, dim),
                nn.Dropout(dropout),
            )

        def forward(self, values: Any) -> Any:
            return self.network(values)

    return FeedForward


def _conditional_block_type(torch: Any, attention: type, feed_forward: type) -> type:
    nn = torch.nn

    class ConditionalBlock(nn.Module):
        def __init__(self, config: LeWMConfig) -> None:
            super().__init__()
            dim = config.embed_dim
            self.attention = attention(
                dim, config.predictor_heads, config.predictor_head_dim, config.dropout
            )
            self.feed_forward = feed_forward(dim, config.predictor_mlp_dim, config.dropout)
            self.norm_attention = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.norm_feed_forward = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
            nn.init.zeros_(self.modulation[-1].weight)
            nn.init.zeros_(self.modulation[-1].bias)

        @staticmethod
        def modulate(values: Any, shift: Any, scale: Any) -> Any:
            return values * (1.0 + scale) + shift

        def forward(self, values: Any, condition: Any) -> Any:
            shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(condition).chunk(
                6, dim=-1
            )
            attention_input = self.modulate(self.norm_attention(values), shift_a, scale_a)
            values = values + gate_a * self.attention(attention_input)
            mlp_input = self.modulate(self.norm_feed_forward(values), shift_m, scale_m)
            return values + gate_m * self.feed_forward(mlp_input)

    return ConditionalBlock


def _predictor_type(torch: Any, conditional_block: type) -> type:
    nn = torch.nn

    class Predictor(nn.Module):
        def __init__(self, config: LeWMConfig) -> None:
            super().__init__()
            self.position = nn.Parameter(torch.randn(1, config.history, config.embed_dim))
            self.dropout = nn.Dropout(config.embedding_dropout)
            self.blocks = nn.ModuleList(
                conditional_block(config) for _ in range(config.predictor_depth)
            )
            self.norm = nn.LayerNorm(config.embed_dim)

        def forward(self, embeddings: Any, actions: Any) -> Any:
            steps = embeddings.shape[1]
            if steps > self.position.shape[1]:
                raise ValueError("predictor context exceeds configured history")
            values = self.dropout(embeddings + self.position[:, :steps])
            for block in self.blocks:
                values = block(values, actions)
            return self.norm(values)

    return Predictor


def _sigreg_type(torch: Any) -> type:
    nn = torch.nn

    class SIGReg(nn.Module):
        def __init__(self, knots: int, projections: int) -> None:
            super().__init__()
            self.projections = projections
            points = torch.linspace(0.0, 3.0, knots)
            delta = 3.0 / (knots - 1)
            weights = torch.full((knots,), 2.0 * delta)
            weights[[0, -1]] = delta
            gaussian = torch.exp(-points.square() / 2.0)
            self.register_buffer("points", points)
            self.register_buffer("gaussian", gaussian)
            self.register_buffer("weights", weights * gaussian)

        def forward(self, embeddings: Any) -> Any:
            directions = torch.randn(
                embeddings.shape[-1], self.projections, device=embeddings.device
            )
            directions = directions / directions.norm(p=2, dim=0).clamp_min(1e-12)
            projected = (embeddings @ directions).unsqueeze(-1) * self.points
            error = (projected.cos().mean(-3) - self.gaussian).square()
            error = error + projected.sin().mean(-3).square()
            statistic = (error @ self.weights) * embeddings.shape[-2]
            return statistic.mean()

    return SIGReg


def _vector_jepa_type(torch: Any, predictor: type, sigreg: type) -> type:
    nn = torch.nn

    class VectorJEPA(nn.Module):
        def __init__(self, config: LeWMConfig) -> None:
            super().__init__()
            self.config = config
            self.encoder = nn.Sequential(
                nn.Linear(config.obs_dim, config.encoder_hidden_dim),
                nn.LayerNorm(config.encoder_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.encoder_hidden_dim, config.embed_dim),
            )
            self.projector = nn.Sequential(
                nn.Linear(config.embed_dim, config.projector_hidden_dim),
                nn.GELU(),
                nn.Linear(config.projector_hidden_dim, config.embed_dim),
            )
            self.action_convolution = nn.Conv1d(config.action_dim, config.action_dim, 1)
            self.action_mlp = nn.Sequential(
                nn.Linear(config.action_dim, 4 * config.embed_dim),
                nn.SiLU(),
                nn.Linear(4 * config.embed_dim, config.embed_dim),
            )
            self.predictor = predictor(config)
            self.prediction_projector = nn.Sequential(
                nn.Linear(config.embed_dim, config.projector_hidden_dim),
                nn.GELU(),
                nn.Linear(config.projector_hidden_dim, config.embed_dim),
            )
            self.sigreg = sigreg(config.sigreg_knots, config.sigreg_projections)

        def encode(self, observations: Any) -> Any:
            return self.projector(self.encoder(observations.float()))

        def encode_actions(self, actions: Any) -> Any:
            values = self.action_convolution(actions.float().transpose(1, 2)).transpose(1, 2)
            return self.action_mlp(values)

        def predict(self, embeddings: Any, action_embeddings: Any) -> Any:
            return self.prediction_projector(self.predictor(embeddings, action_embeddings))

        def loss(self, observations: Any, actions: Any) -> dict[str, Any]:
            embeddings = self.encode(observations)
            action_embeddings = self.encode_actions(actions)
            context = embeddings[:, : self.config.history]
            context_actions = action_embeddings[:, : self.config.history]
            target = embeddings[:, self.config.predictions :]
            prediction = self.predict(context, context_actions)
            shared = min(prediction.shape[1], target.shape[1])
            prediction_loss = (prediction[:, :shared] - target[:, :shared]).square().mean()
            sigreg_loss = self.sigreg(embeddings.transpose(0, 1))
            total = prediction_loss + self.config.sigreg_weight * sigreg_loss
            return {"loss": total, "prediction_loss": prediction_loss, "sigreg_loss": sigreg_loss}

        def rollout(self, observations: Any, history_actions: Any, future_actions: Any) -> Any:
            embeddings = self.encode(observations)
            actions = history_actions.clone()
            actions[:, -1] = future_actions[:, 0]
            predictions = []
            for timestep in range(future_actions.shape[1]):
                action_embeddings = self.encode_actions(actions[:, -self.config.history :])
                predicted = self.predict(embeddings[:, -self.config.history :], action_embeddings)[
                    :, -1:
                ]
                predictions.append(predicted[:, 0])
                embeddings = torch.cat([embeddings, predicted], dim=1)
                if timestep + 1 < future_actions.shape[1]:
                    actions = torch.cat(
                        [actions, future_actions[:, timestep + 1 : timestep + 2]], dim=1
                    )
            return torch.stack(predictions, dim=1)

    VectorJEPA.__name__ = "VectorJEPA"
    return VectorJEPA


@lru_cache(maxsize=1)
def _model_type() -> type:
    torch = require_torch()
    attention = _attention_type(torch)
    feed_forward = _feed_forward_type(torch)
    conditional_block = _conditional_block_type(torch, attention, feed_forward)
    predictor = _predictor_type(torch, conditional_block)
    sigreg = _sigreg_type(torch)
    return _vector_jepa_type(torch, predictor, sigreg)


def build_vector_jepa(config: LeWMConfig | None = None) -> Any:
    """Construct the Torch model only when the WM capability is requested."""

    return _model_type()(config or LeWMConfig())


__all__ = ["LeWMConfig", "build_vector_jepa", "require_torch", "torch_available"]
