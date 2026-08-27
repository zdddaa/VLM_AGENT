"""Change-conditioned SAM3 integration.

The adapter is the only geometry module that is allowed to combine T1/T2 tokens,
change probability, and the change-object box. A concrete SAM3 implementation is
plugged in through ``SAM3Backend``; until then the trainable auxiliary geometry/
semantic decoder provides a testable fallback path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..schemas.change_object import BBox, ChangeObject
from .box_prompt_encoder import BoxPromptEncoder, BoxPromptEncoding
from .geometry_semantic_decoder import (
    GeometrySemanticDecoder,
    GeometrySemanticDecoderOutput,
)
from .probability_injector import ProbabilityInjection, ProbabilityInjectionOutput
from .temporal_token_fuser import TemporalTokenFusion, TemporalTokenFusionOutput


@dataclass(slots=True)
class SAM3BackendOutput:
    """Minimal normalized contract expected from a real SAM3 backend."""

    mask_logits: Tensor
    boundary_confidence: Tensor | None = None
    t1_semantic_logits: Tensor | None = None
    t2_semantic_logits: Tensor | None = None
    semantic_refinement_confidence: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SAM3Backend(Protocol):
    """Backend protocol so local SAM3 APIs can be connected without leaking inward."""

    def decode_change(
        self,
        *,
        conditioned_tokens: Tensor,
        t1_tokens: Tensor,
        t2_tokens: Tensor,
        probability_tokens: Tensor,
        box_prompt_tokens: Tensor,
        box_spatial_prior: Tensor | None,
        spatial_shape: tuple[int, int],
        image_size: tuple[int, int],
    ) -> SAM3BackendOutput: ...


@dataclass(slots=True)
class SAM3ChangeAdapterResult:
    """Normalized result returned to the rest of VLM_AGENT."""

    mask_logits: Tensor
    t1_semantic_logits: Tensor | None
    t2_semantic_logits: Tensor | None
    boundary_confidence: Tensor
    semantic_refinement_confidence: Tensor | None
    semantic_difference: Tensor
    probability_support: Tensor
    box_support: Tensor
    conditioned_tokens: Tensor
    box_prompt_tokens: Tensor
    spatial_shape: tuple[int, int]
    backend_metadata: dict[str, Any] = field(default_factory=dict)

    def apply_to_change_object(
        self,
        obj: ChangeObject,
        *,
        batch_index: int = 0,
        refined_mask_ref: str | None = None,
        semantic_feature_ref: str | None = None,
        polygon_geojson: dict[str, Any] | None = None,
    ) -> None:
        """Write lightweight geometry/evidence back to the canonical object state.

        Tensor payloads are intentionally not serialized into ``ChangeObject``.
        Callers persist them separately and provide references through this method.
        """

        batch = self.mask_logits.shape[0]
        if not 0 <= batch_index < batch:
            raise IndexError(f"batch_index {batch_index} is outside batch size {batch}")

        if refined_mask_ref is not None:
            obj.geometry.refined_mask_ref = refined_mask_ref
        if polygon_geojson is not None:
            obj.geometry.polygon_geojson = polygon_geojson
        obj.geometry.boundary_source = "sam3_change_adapter"
        if semantic_feature_ref is not None:
            obj.semantic_feature_ref = semantic_feature_ref

        evidence_values: dict[str, Any] = {
            "boundary_confidence": _scalar(self.boundary_confidence, batch_index),
            "semantic_difference": _scalar(self.semantic_difference, batch_index),
            "probability_support": _scalar(self.probability_support, batch_index),
            "box_support": _scalar(self.box_support, batch_index),
        }
        if self.semantic_refinement_confidence is not None:
            evidence_values["semantic_refinement_confidence"] = _scalar(
                self.semantic_refinement_confidence,
                batch_index,
            )
        obj.evidence.update(**evidence_values)

        grounding_meta = obj.metadata.setdefault("sam3_change_adapter", {})
        grounding_meta.update(
            {
                "spatial_shape": list(self.spatial_shape),
                "backend_metadata": dict(self.backend_metadata),
            }
        )
        obj.record_trace(
            stage="grounding",
            action="sam3_change_adapter",
            boundary_confidence=evidence_values["boundary_confidence"],
            semantic_difference=evidence_values["semantic_difference"],
            probability_support=evidence_values["probability_support"],
            box_support=evidence_values["box_support"],
        )


def _scalar(value: Tensor, batch_index: int) -> float:
    if value.ndim == 0:
        return float(value.detach().cpu().item())
    return float(value[batch_index].detach().cpu().item())


def _resize_prior(prior: Tensor, target_hw: tuple[int, int]) -> Tensor:
    batch, token_count, channels = prior.shape
    if channels != 1:
        raise ValueError("box spatial prior must have one channel")
    source_side = int(token_count**0.5)
    if source_side * source_side == token_count:
        source_hw = (source_side, source_side)
    else:
        raise ValueError(
            "cannot resize flattened box prior without a known rectangular source shape"
        )
    grid = prior.transpose(1, 2).reshape(batch, 1, *source_hw)
    resized = F.interpolate(grid, size=target_hw, mode="nearest")
    return resized


class SAM3ChangeAdapter(nn.Module):
    """Fuse temporal, probability, and box evidence before SAM3 decoding."""

    def __init__(
        self,
        feature_dim: int,
        *,
        num_semantic_classes: int = 0,
        adapter_hidden_dim: int | None = None,
        prompt_token_count: int = 2,
        probability_gate_strength: float = 1.0,
        backend: SAM3Backend | None = None,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.backend = backend

        self.temporal_fuser = TemporalTokenFusion(
            feature_dim,
            adapter_hidden_dim=adapter_hidden_dim,
        )
        self.probability_injector = ProbabilityInjection(
            feature_dim,
            gate_strength=probability_gate_strength,
        )
        self.box_encoder = BoxPromptEncoder(
            feature_dim,
            prompt_token_count=prompt_token_count,
        )
        self.box_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )
        self.condition_norm = nn.LayerNorm(feature_dim)
        self.decoder = GeometrySemanticDecoder(
            feature_dim,
            num_semantic_classes=num_semantic_classes,
        )

    def forward(
        self,
        t1_tokens: Tensor,
        t2_tokens: Tensor,
        change_probability: Tensor,
        change_boxes: BBox | Tensor | Sequence[float] | Sequence[BBox],
        *,
        image_size: tuple[int, int],
        spatial_shape: tuple[int, int] | None = None,
    ) -> SAM3ChangeAdapterResult:
        temporal = self.temporal_fuser(t1_tokens, t2_tokens)
        resolved_spatial_shape = temporal.spatial_shape or spatial_shape
        if resolved_spatial_shape is None:
            raise ValueError(
                "spatial_shape is required for sequence tokens because geometry decoding "
                "must reconstruct a 2D token grid"
            )
        if resolved_spatial_shape[0] * resolved_spatial_shape[1] != temporal.fused_tokens.shape[1]:
            raise ValueError("resolved spatial_shape does not match fused token count")

        probability = self.probability_injector(
            temporal.fused_tokens,
            change_probability,
            spatial_shape=resolved_spatial_shape,
        )
        box = self.box_encoder(
            change_boxes,
            image_size=image_size,
            batch_size=temporal.fused_tokens.shape[0],
            device=temporal.fused_tokens.device,
            dtype=temporal.fused_tokens.dtype,
            spatial_shape=resolved_spatial_shape,
        )

        conditioned = self._inject_box_prior(probability, box)
        auxiliary = self.decoder(
            conditioned,
            spatial_shape=resolved_spatial_shape,
            t1_tokens=temporal.t1_adapted,
            t2_tokens=temporal.t2_adapted,
        )

        backend_output = self._run_backend(
            conditioned=conditioned,
            temporal=temporal,
            probability=probability,
            box=box,
            spatial_shape=resolved_spatial_shape,
            image_size=image_size,
        )

        if backend_output is None:
            mask_logits = auxiliary.mask_logits
            boundary_confidence = auxiliary.boundary_confidence
            t1_semantic_logits = auxiliary.t1_semantic_logits
            t2_semantic_logits = auxiliary.t2_semantic_logits
            semantic_refinement_confidence = auxiliary.semantic_refinement_confidence
            backend_metadata: dict[str, Any] = {"backend": "auxiliary_decoder"}
        else:
            mask_logits = backend_output.mask_logits
            boundary_confidence = (
                backend_output.boundary_confidence
                if backend_output.boundary_confidence is not None
                else auxiliary.boundary_confidence
            )
            t1_semantic_logits = (
                backend_output.t1_semantic_logits
                if backend_output.t1_semantic_logits is not None
                else auxiliary.t1_semantic_logits
            )
            t2_semantic_logits = (
                backend_output.t2_semantic_logits
                if backend_output.t2_semantic_logits is not None
                else auxiliary.t2_semantic_logits
            )
            semantic_refinement_confidence = (
                backend_output.semantic_refinement_confidence
                if backend_output.semantic_refinement_confidence is not None
                else auxiliary.semantic_refinement_confidence
            )
            backend_metadata = dict(backend_output.metadata)
            backend_metadata.setdefault("backend", type(self.backend).__name__)

        semantic_difference = self._semantic_difference(temporal, box)
        probability_support = self._probability_support(probability, box)
        box_support = self._box_support(
            mask_logits,
            box,
            source_spatial_shape=resolved_spatial_shape,
        )

        return SAM3ChangeAdapterResult(
            mask_logits=mask_logits,
            t1_semantic_logits=t1_semantic_logits,
            t2_semantic_logits=t2_semantic_logits,
            boundary_confidence=boundary_confidence.clamp(0.0, 1.0),
            semantic_refinement_confidence=(
                None
                if semantic_refinement_confidence is None
                else semantic_refinement_confidence.clamp(0.0, 1.0)
            ),
            semantic_difference=semantic_difference,
            probability_support=probability_support,
            box_support=box_support,
            conditioned_tokens=conditioned,
            box_prompt_tokens=box.prompt_tokens,
            spatial_shape=resolved_spatial_shape,
            backend_metadata=backend_metadata,
        )

    def _inject_box_prior(
        self,
        probability: ProbabilityInjectionOutput,
        box: BoxPromptEncoding,
    ) -> Tensor:
        prompt_context = self.box_projection(box.prompt_tokens.mean(dim=1, keepdim=True))
        if box.spatial_prior is None:
            return self.condition_norm(probability.conditioned_tokens + prompt_context)
        return self.condition_norm(
            probability.conditioned_tokens + box.spatial_prior * prompt_context
        )

    def _run_backend(
        self,
        *,
        conditioned: Tensor,
        temporal: TemporalTokenFusionOutput,
        probability: ProbabilityInjectionOutput,
        box: BoxPromptEncoding,
        spatial_shape: tuple[int, int],
        image_size: tuple[int, int],
    ) -> SAM3BackendOutput | None:
        if self.backend is None:
            return None
        return self.backend.decode_change(
            conditioned_tokens=conditioned,
            t1_tokens=temporal.t1_adapted,
            t2_tokens=temporal.t2_adapted,
            probability_tokens=probability.probability_tokens,
            box_prompt_tokens=box.prompt_tokens,
            box_spatial_prior=box.spatial_prior,
            spatial_shape=spatial_shape,
            image_size=image_size,
        )

    @staticmethod
    def _semantic_difference(
        temporal: TemporalTokenFusionOutput,
        box: BoxPromptEncoding,
    ) -> Tensor:
        difference = temporal.cosine_difference
        if box.spatial_prior is None:
            return difference.mean(dim=1).clamp(0.0, 1.0)
        weights = box.spatial_prior.squeeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return ((difference * weights).sum(dim=1) / denominator).clamp(0.0, 1.0)

    @staticmethod
    def _probability_support(
        probability: ProbabilityInjectionOutput,
        box: BoxPromptEncoding,
    ) -> Tensor:
        probability_tokens = probability.probability_tokens.squeeze(-1)
        if box.spatial_prior is None:
            return probability.probability_support
        weights = box.spatial_prior.squeeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return ((probability_tokens * weights).sum(dim=1) / denominator).clamp(0.0, 1.0)

    @staticmethod
    def _box_support(
        mask_logits: Tensor,
        box: BoxPromptEncoding,
        *,
        source_spatial_shape: tuple[int, int],
    ) -> Tensor:
        if box.spatial_prior is None:
            return torch.ones(
                mask_logits.shape[0],
                device=mask_logits.device,
                dtype=mask_logits.dtype,
            )

        probability = torch.sigmoid(mask_logits)
        target_hw = tuple(probability.shape[-2:])
        prior_grid = box.spatial_prior.transpose(1, 2).reshape(
            box.spatial_prior.shape[0],
            1,
            *source_spatial_shape,
        )
        if target_hw != source_spatial_shape:
            prior_grid = F.interpolate(prior_grid, size=target_hw, mode="nearest")

        inside_mass = (probability * prior_grid).sum(dim=(1, 2, 3))
        total_mass = probability.sum(dim=(1, 2, 3)).clamp_min(1e-6)
        return (inside_mass / total_mass).clamp(0.0, 1.0)
