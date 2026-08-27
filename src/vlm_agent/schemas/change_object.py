"""Shared object-centric state used by the full VLM_AGENT pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .evidence import EvidenceState


@dataclass(frozen=True, slots=True)
class BBox:
    """Pixel-space bounding box using inclusive-exclusive coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min")

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max, self.y_max


@dataclass(slots=True)
class GeometryState:
    """Geometry references for one candidate change object.

    Masks and tensors stay outside the JSON state.  The object stores paths/URIs so
    GeoTIFF, NumPy, Torch, or shared-memory backends can be connected later.
    """

    bbox: BBox
    initial_mask_ref: str | None = None
    refined_mask_ref: str | None = None
    polygon_geojson: dict[str, Any] | None = None
    area_m2: float | None = None
    boundary_source: str | None = None

    @property
    def current_mask_ref(self) -> str | None:
        return self.refined_mask_ref or self.initial_mask_ref


@dataclass(slots=True)
class TemporalSemanticState:
    """Structured T1/T2 semantics produced by the temporal VLM or semantic model."""

    t1_class: str | None = None
    t2_class: str | None = None
    t1_label: str | None = None
    t2_label: str | None = None

    @property
    def from_to(self) -> str | None:
        """Derive the transition deterministically once both class codes exist."""

        if self.t1_class is None or self.t2_class is None:
            return None
        return f"{self.t1_class}>{self.t2_class}"


@dataclass(slots=True)
class ChangeObject:
    """Canonical intermediate representation for one semantic change candidate.

    The same object is enriched stage by stage:

    change detector -> SAM3 Change Adapter -> temporal VLM -> evidence controller.
    """

    object_id: str
    geometry: GeometryState

    # References supplied by the change detector / feature extractor.
    change_probability_ref: str | None = None
    t1_token_ref: str | None = None
    t2_token_ref: str | None = None
    t1_context_ref: str | None = None
    t2_context_ref: str | None = None

    semantics: TemporalSemanticState = field(default_factory=TemporalSemanticState)
    evidence: EvidenceState = field(default_factory=EvidenceState)

    development_type: str | None = None
    semantic_feature_ref: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def from_to(self) -> str | None:
        return self.semantics.from_to

    def record_trace(self, stage: str, action: str, **details: Any) -> None:
        self.trace.append(
            {
                "stage": stage,
                "action": action,
                "details": details,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the object without embedding heavy image/tensor payloads."""

        data = asdict(self)
        data["from_to"] = self.from_to
        return data
