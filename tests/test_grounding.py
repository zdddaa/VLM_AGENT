import pytest


torch = pytest.importorskip("torch")

from vlm_agent import BBox, ChangeObject, GeometryState
from vlm_agent.grounding import (
    BoxPromptEncoder,
    ProbabilityInjection,
    SAM3BackendOutput,
    SAM3ChangeAdapter,
    TemporalTokenFusion,
)


def test_temporal_token_fusion_accepts_feature_maps():
    module = TemporalTokenFusion(feature_dim=8)
    t1 = torch.randn(2, 8, 4, 4)
    t2 = torch.randn(2, 8, 4, 4)

    output = module(t1, t2)

    assert output.fused_tokens.shape == (2, 16, 8)
    assert output.difference_tokens.shape == (2, 16, 8)
    assert output.cosine_difference.shape == (2, 16)
    assert output.spatial_shape == (4, 4)
    assert torch.all((output.cosine_difference >= 0) & (output.cosine_difference <= 1))


def test_probability_injection_resizes_image_probability_to_token_grid():
    module = ProbabilityInjection(feature_dim=8)
    tokens = torch.randn(2, 16, 8)
    probability = torch.rand(2, 1, 64, 64)

    output = module(tokens, probability, spatial_shape=(4, 4))

    assert output.conditioned_tokens.shape == tokens.shape
    assert output.probability_tokens.shape == (2, 16, 1)
    assert output.probability_support.shape == (2,)


def test_box_prompt_encoder_produces_prompt_and_spatial_prior():
    encoder = BoxPromptEncoder(feature_dim=8, prompt_token_count=2)
    output = encoder(
        BBox(8, 8, 56, 56),
        image_size=(64, 64),
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        spatial_shape=(8, 8),
    )

    assert output.prompt_tokens.shape == (1, 2, 8)
    assert output.spatial_prior.shape == (1, 64, 1)
    assert 0 < output.area_ratio.item() <= 1


def test_sam3_change_adapter_runs_end_to_end_without_backend():
    adapter = SAM3ChangeAdapter(feature_dim=8, num_semantic_classes=5)
    t1 = torch.randn(2, 8, 4, 4)
    t2 = torch.randn(2, 8, 4, 4)
    probability = torch.rand(2, 1, 64, 64)
    boxes = torch.tensor(
        [
            [8.0, 8.0, 56.0, 56.0],
            [4.0, 12.0, 52.0, 60.0],
        ]
    )

    result = adapter(
        t1,
        t2,
        probability,
        boxes,
        image_size=(64, 64),
    )

    assert result.mask_logits.shape == (2, 1, 4, 4)
    assert result.t1_semantic_logits.shape == (2, 5, 4, 4)
    assert result.t2_semantic_logits.shape == (2, 5, 4, 4)
    assert result.boundary_confidence.shape == (2,)
    assert result.semantic_difference.shape == (2,)
    assert result.probability_support.shape == (2,)
    assert result.box_support.shape == (2,)
    assert result.backend_metadata["backend"] == "auxiliary_decoder"


class DummySAM3Backend:
    def decode_change(self, **kwargs):
        batch = kwargs["conditioned_tokens"].shape[0]
        device = kwargs["conditioned_tokens"].device
        return SAM3BackendOutput(
            mask_logits=torch.ones(batch, 1, 16, 16, device=device),
            boundary_confidence=torch.full((batch,), 0.95, device=device),
            metadata={"backend": "dummy_sam3"},
        )


def test_backend_output_overrides_auxiliary_geometry_and_writes_back():
    adapter = SAM3ChangeAdapter(
        feature_dim=8,
        num_semantic_classes=3,
        backend=DummySAM3Backend(),
    )
    result = adapter(
        torch.randn(1, 8, 4, 4),
        torch.randn(1, 8, 4, 4),
        torch.rand(1, 1, 64, 64),
        BBox(8, 8, 56, 56),
        image_size=(64, 64),
    )

    obj = ChangeObject(
        object_id="obj-grounding-001",
        geometry=GeometryState(
            bbox=BBox(8, 8, 56, 56),
            initial_mask_ref="memory://initial-mask",
        ),
    )
    result.apply_to_change_object(
        obj,
        refined_mask_ref="memory://sam3-refined-mask",
        semantic_feature_ref="memory://sam3-semantic-features",
    )

    assert result.mask_logits.shape == (1, 1, 16, 16)
    assert result.boundary_confidence.item() == pytest.approx(0.95)
    assert obj.geometry.refined_mask_ref == "memory://sam3-refined-mask"
    assert obj.geometry.boundary_source == "sam3_change_adapter"
    assert obj.semantic_feature_ref == "memory://sam3-semantic-features"
    assert obj.evidence.boundary_confidence == pytest.approx(0.95)
    assert obj.evidence.semantic_difference is not None
    assert obj.evidence.probability_support is not None
    assert obj.evidence.box_support is not None
    assert obj.evidence.semantic_refinement_confidence is not None
    assert obj.trace[-1]["action"] == "sam3_change_adapter"
