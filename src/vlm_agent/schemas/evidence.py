"""Normalized evidence state for object-level semantic change reasoning.

All confidence-like values use [0, 1]. Thresholds do not live in this module;
they belong to the agent policy and should be calibrated on validation data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any


def _validate_probability(name: str, value: float | None) -> None:
    if value is None:
        return
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


@dataclass(slots=True)
class EvidenceState:
    """Evidence collected from change, geometry, semantic, and verification models.

    Values may be ``None`` while a pipeline stage has not produced the evidence yet.
    This allows one ``ChangeObject`` to persist across the full agent state machine.
    """

    cd_confidence: float | None = None
    boundary_confidence: float | None = None
    seed_coverage: float | None = None

    t1_semantic_confidence: float | None = None
    t2_semantic_confidence: float | None = None
    semantic_difference: float | None = None

    pseudo_change_risk: float | None = None
    semantic_conflict: bool = False
    geometry_anomaly: bool = False

    # Filled only after an explicit verification step.  None means unverified.
    real_change_verified: bool | None = None

    def __post_init__(self) -> None:
        for name in (
            "cd_confidence",
            "boundary_confidence",
            "seed_coverage",
            "t1_semantic_confidence",
            "t2_semantic_confidence",
            "semantic_difference",
            "pseudo_change_risk",
        ):
            _validate_probability(name, getattr(self, name))

    @property
    def semantic_confidence(self) -> float | None:
        """Return the conservative T1/T2 semantic confidence.

        The minimum is used because a transition is only as reliable as its weaker
        temporal endpoint.
        """

        values = [
            value
            for value in (self.t1_semantic_confidence, self.t2_semantic_confidence)
            if value is not None
        ]
        if len(values) != 2:
            return None
        return min(values)

    @property
    def mean_semantic_confidence(self) -> float | None:
        values = [
            value
            for value in (self.t1_semantic_confidence, self.t2_semantic_confidence)
            if value is not None
        ]
        if len(values) != 2:
            return None
        return mean(values)

    def missing(self, *fields: str) -> tuple[str, ...]:
        """Return requested evidence fields that are still unavailable."""

        missing_fields: list[str] = []
        for field_name in fields:
            if not hasattr(self, field_name):
                raise AttributeError(f"Unknown evidence field: {field_name}")
            if getattr(self, field_name) is None:
                missing_fields.append(field_name)
        return tuple(missing_fields)

    def update(self, **values: Any) -> None:
        """Update evidence in place while preserving probability validation."""

        for name, value in values.items():
            if not hasattr(self, name):
                raise AttributeError(f"Unknown evidence field: {name}")
            if name in {
                "cd_confidence",
                "boundary_confidence",
                "seed_coverage",
                "t1_semantic_confidence",
                "t2_semantic_confidence",
                "semantic_difference",
                "pseudo_change_risk",
            }:
                _validate_probability(name, value)
            setattr(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
