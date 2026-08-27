"""Evidence-driven controller for VLM_AGENT.

The controller does not inspect imagery and does not generate semantics.  It reads a
``ChangeObject`` and selects the next action from explicit evidence produced by the
change detector, SAM3 Change Adapter, and temporal VLM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from ..schemas.change_object import ChangeObject
from ..schemas.evidence import EvidenceState


class AgentAction(str, Enum):
    ACCEPT = "accept"
    REFINE = "refine"
    VERIFY = "verify"


@dataclass(frozen=True, slots=True)
class AgentPolicy:
    """Routing thresholds.

    Defaults are engineering starting points only.  They must be calibrated on a
    held-out validation set before being used as reported scientific thresholds.
    """

    min_change_confidence: float = 0.60
    high_change_confidence: float = 0.80
    min_boundary_confidence: float = 0.75
    min_semantic_confidence: float = 0.70
    low_semantic_difference: float = 0.30
    high_semantic_difference: float = 0.60
    max_pseudo_change_risk: float = 0.40
    require_complete_evidence: bool = True

    def __post_init__(self) -> None:
        probability_fields = (
            "min_change_confidence",
            "high_change_confidence",
            "min_boundary_confidence",
            "min_semantic_confidence",
            "low_semantic_difference",
            "high_semantic_difference",
            "max_pseudo_change_risk",
        )
        for name in probability_fields:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        if self.high_change_confidence < self.min_change_confidence:
            raise ValueError("high_change_confidence must be >= min_change_confidence")
        if self.high_semantic_difference < self.low_semantic_difference:
            raise ValueError("high_semantic_difference must be >= low_semantic_difference")


@dataclass(frozen=True, slots=True)
class AgentDecision:
    action: AgentAction
    reasons: tuple[str, ...]
    recommended_tool: str | None
    evidence_snapshot: dict[str, Any]


class AgentController:
    """Deterministic evidence router for one ``ChangeObject`` at a time.

    Priority is geometry first, then semantic/temporal verification, then accept.
    This prevents a semantic model from validating an object whose spatial support
    is still unreliable.
    """

    def __init__(self, policy: AgentPolicy | None = None) -> None:
        self.policy = policy or AgentPolicy()

    def decide(self, obj: ChangeObject) -> AgentDecision:
        evidence = obj.evidence

        geometry_reasons = self._geometry_reasons(evidence)
        if geometry_reasons:
            return self._decision(
                AgentAction.REFINE,
                geometry_reasons,
                "sam3_change_adapter",
                evidence,
            )

        verification_reasons = self._verification_reasons(evidence)
        if verification_reasons:
            return self._decision(
                AgentAction.VERIFY,
                verification_reasons,
                "temporal_vlm_verifier",
                evidence,
            )

        # ACCEPT means the current state is resolved and can be decoded.  When an
        # explicit verifier has set real_change_verified=False, the decoder should
        # suppress the candidate from the final change map rather than route again.
        if evidence.real_change_verified is False:
            accept_reasons = ("verification resolved candidate as pseudo-change",)
        elif evidence.real_change_verified is True:
            accept_reasons = ("verification confirmed real change",)
        else:
            accept_reasons = ("change, geometry, and semantic evidence are consistent",)

        return self._decision(
            AgentAction.ACCEPT,
            accept_reasons,
            None,
            evidence,
        )

    def route(self, obj: ChangeObject) -> AgentDecision:
        """Select an action and append the decision to the object's trace."""

        decision = self.decide(obj)
        obj.record_trace(
            stage="agent_controller",
            action=decision.action.value,
            reasons=list(decision.reasons),
            recommended_tool=decision.recommended_tool,
        )
        return decision

    def _geometry_reasons(self, evidence: EvidenceState) -> tuple[str, ...]:
        reasons: list[str] = []
        if evidence.geometry_anomaly:
            reasons.append("geometry anomaly reported")

        if evidence.boundary_confidence is None:
            if self.policy.require_complete_evidence:
                reasons.append("boundary confidence is missing")
        elif evidence.boundary_confidence < self.policy.min_boundary_confidence:
            reasons.append(
                "boundary confidence below policy threshold "
                f"({evidence.boundary_confidence:.3f} < "
                f"{self.policy.min_boundary_confidence:.3f})"
            )

        return tuple(reasons)

    def _verification_reasons(self, evidence: EvidenceState) -> tuple[str, ...]:
        # An explicit verifier result is terminal for routing.  The downstream
        # decoder decides whether to keep or suppress the resolved object.
        if evidence.real_change_verified is not None:
            return ()

        reasons: list[str] = []

        required = (
            "cd_confidence",
            "t1_semantic_confidence",
            "t2_semantic_confidence",
            "semantic_difference",
            "pseudo_change_risk",
        )
        missing = evidence.missing(*required)
        if missing and self.policy.require_complete_evidence:
            reasons.append("missing evidence: " + ", ".join(missing))
            return tuple(reasons)

        if evidence.semantic_conflict:
            reasons.append("T1/T2 semantic evidence is conflicting")

        semantic_confidence = evidence.semantic_confidence
        if (
            semantic_confidence is not None
            and semantic_confidence < self.policy.min_semantic_confidence
        ):
            reasons.append(
                "temporal semantic confidence below policy threshold "
                f"({semantic_confidence:.3f} < {self.policy.min_semantic_confidence:.3f})"
            )

        if (
            evidence.pseudo_change_risk is not None
            and evidence.pseudo_change_risk > self.policy.max_pseudo_change_risk
        ):
            reasons.append(
                "pseudo-change risk above policy threshold "
                f"({evidence.pseudo_change_risk:.3f} > "
                f"{self.policy.max_pseudo_change_risk:.3f})"
            )

        cd = evidence.cd_confidence
        sem_diff = evidence.semantic_difference
        if cd is not None and sem_diff is not None:
            if (
                cd >= self.policy.high_change_confidence
                and sem_diff <= self.policy.low_semantic_difference
            ):
                reasons.append(
                    "high change evidence but weak semantic difference; possible appearance change"
                )
            elif (
                cd < self.policy.min_change_confidence
                and sem_diff >= self.policy.high_semantic_difference
            ):
                reasons.append(
                    "weak change evidence but strong semantic difference; possible missed/underestimated change"
                )
            elif (
                cd < self.policy.min_change_confidence
                and sem_diff <= self.policy.low_semantic_difference
            ):
                reasons.append("both change and semantic-difference evidence are weak")

        return tuple(reasons)

    @staticmethod
    def _decision(
        action: AgentAction,
        reasons: tuple[str, ...],
        recommended_tool: str | None,
        evidence: EvidenceState,
    ) -> AgentDecision:
        return AgentDecision(
            action=action,
            reasons=reasons,
            recommended_tool=recommended_tool,
            evidence_snapshot=asdict(evidence),
        )
