from vlm_agent import (
    AgentAction,
    AgentController,
    BBox,
    ChangeObject,
    EvidenceState,
    GeometryState,
    TemporalSemanticState,
)


def make_object(**evidence_values):
    return ChangeObject(
        object_id="obj-001",
        geometry=GeometryState(
            bbox=BBox(10, 20, 110, 120),
            initial_mask_ref="memory://mask0",
        ),
        semantics=TemporalSemanticState(
            t1_class="A1-02",
            t2_class="A6-03",
        ),
        evidence=EvidenceState(**evidence_values),
    )


def test_from_to_is_deterministic():
    obj = make_object(boundary_confidence=0.9)
    assert obj.from_to == "A1-02>A6-03"


def test_probability_validation():
    try:
        EvidenceState(cd_confidence=1.1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid probability should raise ValueError")


def test_controller_refines_unreliable_geometry():
    obj = make_object(
        cd_confidence=0.9,
        boundary_confidence=0.4,
        t1_semantic_confidence=0.9,
        t2_semantic_confidence=0.9,
        semantic_difference=0.8,
        pseudo_change_risk=0.1,
    )
    decision = AgentController().decide(obj)
    assert decision.action is AgentAction.REFINE
    assert decision.recommended_tool == "sam3_change_adapter"


def test_controller_verifies_semantic_conflict():
    obj = make_object(
        cd_confidence=0.9,
        boundary_confidence=0.9,
        t1_semantic_confidence=0.9,
        t2_semantic_confidence=0.9,
        semantic_difference=0.8,
        pseudo_change_risk=0.1,
        semantic_conflict=True,
    )
    decision = AgentController().decide(obj)
    assert decision.action is AgentAction.VERIFY
    assert decision.recommended_tool == "temporal_vlm_verifier"


def test_controller_accepts_consistent_evidence():
    obj = make_object(
        cd_confidence=0.9,
        boundary_confidence=0.9,
        t1_semantic_confidence=0.9,
        t2_semantic_confidence=0.92,
        semantic_difference=0.8,
        pseudo_change_risk=0.05,
    )
    decision = AgentController().route(obj)
    assert decision.action is AgentAction.ACCEPT
    assert obj.trace[-1]["action"] == "accept"


def test_verified_pseudo_change_is_resolved_not_reverified():
    obj = make_object(
        cd_confidence=0.9,
        boundary_confidence=0.9,
        t1_semantic_confidence=0.9,
        t2_semantic_confidence=0.9,
        semantic_difference=0.1,
        pseudo_change_risk=0.95,
        real_change_verified=False,
    )
    decision = AgentController().decide(obj)
    assert decision.action is AgentAction.ACCEPT
    assert "pseudo-change" in decision.reasons[0]
