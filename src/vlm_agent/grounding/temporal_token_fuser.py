"""Dual-temporal token alignment and change-aware fusion.

The module keeps the temporal endpoints explicit. It does not classify land cover;
it only converts paired T1/T2 feature tokens into a change-conditioned feature
representation that can be consumed by the SAM3 Change Adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class TemporalTokenFusionOutput:
    """Outputs preserved for later geometry and semantic evidence computation."""

    fused_tokens: Tensor
    t1_adapted: Tensor
    t2_adapted: Tensor
    difference_tokens: Tensor
    cosine_difference: Tensor
    spatial_shape: tuple[int, int] | None


class _ResidualAdapter(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(self.norm(x))


def _to_sequence(tokens: Tensor) -> tuple[Tensor, tuple[int, int] | None]:
    """Convert BCHW feature maps to BNC tokens while preserving spatial shape."""

    if tokens.ndim == 3:
        return tokens, None
    if tokens.ndim == 4:
        batch, channels, height, width = tokens.shape
        sequence = tokens.reshape(batch, channels, height * width).transpose(1, 2)
        return sequence, (height, width)
    raise ValueError(
        "tokens must have shape [B, N, C] or [B, C, H, W], "
        f"got {tuple(tokens.shape)}"
    )


class TemporalTokenFusion(nn.Module):
    """Align and fuse paired temporal features without discarding endpoint tokens.

    Fusion uses four complementary terms: T1, T2, absolute difference, and
    element-wise agreement. A learned change gate determines how strongly the
    temporal difference should modify the fused representation.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        adapter_hidden_dim: int | None = None,
        share_temporal_adapter: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")

        hidden_dim = adapter_hidden_dim or max(feature_dim // 2, 32)
        self.feature_dim = feature_dim
        self.share_temporal_adapter = share_temporal_adapter

        self.t1_adapter = _ResidualAdapter(feature_dim, hidden_dim)
        self.t2_adapter = (
            self.t1_adapter
            if share_temporal_adapter
            else _ResidualAdapter(feature_dim, hidden_dim)
        )

        self.fusion = nn.Sequential(
            nn.LayerNorm(feature_dim * 4),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.change_gate = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, t1_tokens: Tensor, t2_tokens: Tensor) -> TemporalTokenFusionOutput:
        t1, spatial_t1 = _to_sequence(t1_tokens)
        t2, spatial_t2 = _to_sequence(t2_tokens)

        if t1.shape != t2.shape:
            raise ValueError(
                "T1 and T2 tokens must have identical shapes after flattening, "
                f"got {tuple(t1.shape)} and {tuple(t2.shape)}"
            )
        if t1.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected token feature_dim={self.feature_dim}, got {t1.shape[-1]}"
            )
        if spatial_t1 != spatial_t2:
            raise ValueError(
                f"T1/T2 spatial shapes differ: {spatial_t1!r} vs {spatial_t2!r}"
            )

        t1_adapted = self.t1_adapter(t1)
        t2_adapted = self.t2_adapter(t2)
        difference = torch.abs(t2_adapted - t1_adapted)
        agreement = t1_adapted * t2_adapted

        base = self.fusion(
            torch.cat((t1_adapted, t2_adapted, difference, agreement), dim=-1)
        )
        gate = self.change_gate(difference)
        fused = self.output_norm(base + gate * difference)

        cosine_similarity = torch.nn.functional.cosine_similarity(
            t1_adapted,
            t2_adapted,
            dim=-1,
            eps=1e-6,
        )
        cosine_difference = ((1.0 - cosine_similarity) * 0.5).clamp(0.0, 1.0)

        return TemporalTokenFusionOutput(
            fused_tokens=fused,
            t1_adapted=t1_adapted,
            t2_adapted=t2_adapted,
            difference_tokens=difference,
            cosine_difference=cosine_difference,
            spatial_shape=spatial_t1,
        )
