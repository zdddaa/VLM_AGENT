"""Inject change-probability evidence into temporal feature tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(slots=True)
class ProbabilityInjectionOutput:
    conditioned_tokens: Tensor
    probability_tokens: Tensor
    probability_gate: Tensor
    probability_support: Tensor


def _normalize_probability_map(change_probability: Tensor) -> Tensor:
    if change_probability.ndim == 2:
        change_probability = change_probability.unsqueeze(0).unsqueeze(0)
    elif change_probability.ndim == 3:
        change_probability = change_probability.unsqueeze(1)
    elif change_probability.ndim != 4:
        raise ValueError(
            "change_probability must have shape [H,W], [B,H,W], or [B,1,H,W]"
        )

    if change_probability.shape[1] != 1:
        raise ValueError(
            "change_probability must contain exactly one probability channel, "
            f"got {change_probability.shape[1]}"
        )

    if not torch.is_floating_point(change_probability):
        change_probability = change_probability.float()
    return change_probability.clamp(0.0, 1.0)


class ProbabilityInjection(nn.Module):
    """Condition feature tokens using a soft change-probability prior.

    The probability map is never converted into a hard mask here. It is resized to
    the token grid, encoded as a learnable probability embedding, and used to gate
    the amount of change evidence injected into each token.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        gate_strength: float = 1.0,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if gate_strength < 0:
            raise ValueError("gate_strength must be non-negative")

        self.feature_dim = feature_dim
        self.gate_strength = float(gate_strength)
        self.probability_embedding = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.gate_projection = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        tokens: Tensor,
        change_probability: Tensor,
        *,
        spatial_shape: tuple[int, int] | None,
    ) -> ProbabilityInjectionOutput:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [B,N,C], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected token feature_dim={self.feature_dim}, got {tokens.shape[-1]}"
            )

        probability = _normalize_probability_map(change_probability).to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        batch, token_count, _ = tokens.shape
        if probability.shape[0] not in (1, batch):
            raise ValueError(
                "probability batch dimension must be 1 or match token batch size, "
                f"got {probability.shape[0]} vs {batch}"
            )
        if probability.shape[0] == 1 and batch > 1:
            probability = probability.expand(batch, -1, -1, -1)

        if spatial_shape is not None:
            height, width = spatial_shape
            if height * width != token_count:
                raise ValueError(
                    "spatial_shape is inconsistent with token count: "
                    f"{height}x{width} != {token_count}"
                )
            resized = F.interpolate(
                probability,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            probability_tokens = resized.flatten(2).transpose(1, 2)
        else:
            if probability.shape[-2] * probability.shape[-1] != token_count:
                raise ValueError(
                    "sequence tokens without spatial_shape require probability map "
                    "pixel count to equal token count"
                )
            probability_tokens = probability.flatten(2).transpose(1, 2)

        embedding = self.probability_embedding(probability_tokens)
        learned_gate = self.gate_projection(probability_tokens)
        # Keep the prior soft: low probability attenuates injected evidence but does
        # not erase the visual representation, while high probability strengthens it.
        probability_gate = 1.0 + self.gate_strength * (
            learned_gate * (2.0 * probability_tokens - 1.0)
        )
        conditioned = self.norm(tokens * probability_gate + embedding)

        probability_support = probability_tokens.mean(dim=1).squeeze(-1).clamp(0.0, 1.0)
        return ProbabilityInjectionOutput(
            conditioned_tokens=conditioned,
            probability_tokens=probability_tokens,
            probability_gate=probability_gate,
            probability_support=probability_support,
        )
