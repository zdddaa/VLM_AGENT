"""Auxiliary geometry/semantic heads for the SAM3 Change Adapter.

These heads do not replace the native SAM3 decoder. They provide trainable change-
specific correction signals and normalized evidence that can be fused with a real
SAM3 backend when it is connected.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class GeometrySemanticDecoderOutput:
    mask_logits: Tensor
    boundary_confidence: Tensor
    t1_semantic_logits: Tensor | None
    t2_semantic_logits: Tensor | None
    semantic_refinement_confidence: Tensor | None


class GeometrySemanticDecoder(nn.Module):
    """Decode object geometry plus optional T1/T2 semantic correction logits."""

    def __init__(
        self,
        feature_dim: int,
        *,
        num_semantic_classes: int = 0,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if num_semantic_classes < 0:
            raise ValueError("num_semantic_classes must be non-negative")

        hidden = hidden_dim or max(feature_dim // 2, 32)
        self.feature_dim = feature_dim
        self.num_semantic_classes = num_semantic_classes

        self.mask_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        if num_semantic_classes > 0:
            self.t1_semantic_head: nn.Module | None = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, num_semantic_classes),
            )
            self.t2_semantic_head: nn.Module | None = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, num_semantic_classes),
            )
        else:
            self.t1_semantic_head = None
            self.t2_semantic_head = None

    def forward(
        self,
        conditioned_tokens: Tensor,
        *,
        spatial_shape: tuple[int, int],
        t1_tokens: Tensor | None = None,
        t2_tokens: Tensor | None = None,
    ) -> GeometrySemanticDecoderOutput:
        if conditioned_tokens.ndim != 3:
            raise ValueError(
                "conditioned_tokens must have shape [B,N,C], "
                f"got {tuple(conditioned_tokens.shape)}"
            )
        batch, token_count, channels = conditioned_tokens.shape
        if channels != self.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.feature_dim}, got {channels}"
            )

        height, width = spatial_shape
        if height * width != token_count:
            raise ValueError(
                f"spatial_shape {height}x{width} does not match {token_count} tokens"
            )

        mask_logits_seq = self.mask_head(conditioned_tokens)
        mask_logits = mask_logits_seq.transpose(1, 2).reshape(batch, 1, height, width)

        # Auxiliary confidence = distance from the binary decision boundary. It is
        # intentionally conservative and should later be calibrated/combined with
        # native SAM3 scores, change-probability support, and topology checks.
        probability = torch.sigmoid(mask_logits_seq).squeeze(-1)
        certainty = (2.0 * torch.abs(probability - 0.5)).clamp(0.0, 1.0)
        boundary_confidence = certainty.mean(dim=1)

        t1_semantic_logits: Tensor | None = None
        t2_semantic_logits: Tensor | None = None
        semantic_confidence: Tensor | None = None

        if self.num_semantic_classes > 0:
            if t1_tokens is None or t2_tokens is None:
                raise ValueError(
                    "t1_tokens and t2_tokens are required when semantic heads are enabled"
                )
            if t1_tokens.shape != conditioned_tokens.shape or t2_tokens.shape != conditioned_tokens.shape:
                raise ValueError("T1/T2 semantic tokens must match conditioned token shape")
            assert self.t1_semantic_head is not None
            assert self.t2_semantic_head is not None

            t1_seq = self.t1_semantic_head(t1_tokens)
            t2_seq = self.t2_semantic_head(t2_tokens)
            t1_semantic_logits = t1_seq.transpose(1, 2).reshape(
                batch,
                self.num_semantic_classes,
                height,
                width,
            )
            t2_semantic_logits = t2_seq.transpose(1, 2).reshape(
                batch,
                self.num_semantic_classes,
                height,
                width,
            )

            t1_conf = torch.softmax(t1_seq, dim=-1).amax(dim=-1).mean(dim=1)
            t2_conf = torch.softmax(t2_seq, dim=-1).amax(dim=-1).mean(dim=1)
            semantic_confidence = torch.minimum(t1_conf, t2_conf).clamp(0.0, 1.0)

        return GeometrySemanticDecoderOutput(
            mask_logits=mask_logits,
            boundary_confidence=boundary_confidence,
            t1_semantic_logits=t1_semantic_logits,
            t2_semantic_logits=t2_semantic_logits,
            semantic_refinement_confidence=semantic_confidence,
        )
