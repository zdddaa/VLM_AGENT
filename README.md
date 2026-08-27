# VLM_AGENT

Object-centric remote-sensing semantic change understanding agent.

## Core route

```text
T1/T2 imagery
  -> Change Perception
  -> ChangeObject construction
  -> SAM3 Change Adapter
     (T1/T2 tokens + change probability + change box)
  -> Instruction-tuned temporal VLM
  -> EvidenceState fusion
  -> AgentController
     -> ACCEPT / REFINE / VERIFY
  -> semantic transition decoding and geospatial export
```

The repository is intentionally organized around one shared intermediate representation, `ChangeObject`, one normalized evidence container, `EvidenceState`, and one deterministic routing controller, `AgentController`.

## Current status

Phase 1 is implemented:

- `ChangeObject`: object-level geometry, temporal semantics, model references, trace, and serialization.
- `EvidenceState`: normalized change / geometry / semantic / verification evidence with consistency helpers.
- `AgentController`: evidence-driven `ACCEPT / REFINE / VERIFY` routing with explicit reasons and validation-calibrated policy thresholds.

Next integrations:

1. existing change detector -> `ChangeObject.change_probability_ref` and coarse geometry;
2. SAM3 Change Adapter -> T1/T2 local tokens + change probability + change box -> refined geometry / semantic evidence;
3. instruction-tuned VLM -> T1/T2 semantic recognition, real-vs-pseudo change verification, and structured semantic evidence;
4. deterministic transition decoder -> `T1 class -> T2 class -> from_to`;
5. activity decoder and GeoJSON / semantic-map export.

## Package layout

```text
src/vlm_agent/
├── schemas/
│   ├── change_object.py
│   └── evidence.py
└── agent/
    └── controller.py
```

## Design principles

- Models generate evidence; the agent selects actions.
- SAM3 is change-conditioned, not an unconstrained post-processing step.
- T1/T2 semantic labels are explicit; `from_to` is derived deterministically when both labels are available.
- Large tensors and masks are referenced by URI/path rather than embedded in JSON state.
- Routing thresholds are policy configuration and must be calibrated on validation data rather than treated as fixed scientific constants.
