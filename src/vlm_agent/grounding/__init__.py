"""Geometry grounding and change-conditioned SAM3 integration.

Primary contract:
T1 local tokens + T2 local tokens + change probability + change bbox
    -> SAM3ChangeAdapter
    -> refined geometry + semantic correction evidence.
"""

from .box_prompt_encoder import BoxPromptEncoder, BoxPromptEncoding
from .geometry_semantic_decoder import (
    GeometrySemanticDecoder,
    GeometrySemanticDecoderOutput,
)
from .probability_injector import ProbabilityInjection, ProbabilityInjectionOutput
from .sam3_change_adapter import (
    SAM3Backend,
    SAM3BackendOutput,
    SAM3ChangeAdapter,
    SAM3ChangeAdapterResult,
)
from .temporal_token_fuser import TemporalTokenFusion, TemporalTokenFusionOutput

__all__ = [
    "BoxPromptEncoder",
    "BoxPromptEncoding",
    "GeometrySemanticDecoder",
    "GeometrySemanticDecoderOutput",
    "ProbabilityInjection",
    "ProbabilityInjectionOutput",
    "SAM3Backend",
    "SAM3BackendOutput",
    "SAM3ChangeAdapter",
    "SAM3ChangeAdapterResult",
    "TemporalTokenFusion",
    "TemporalTokenFusionOutput",
]
