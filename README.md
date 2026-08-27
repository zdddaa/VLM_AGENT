# VLM_AGENT

Object-centric remote-sensing semantic change understanding agent.

## Core route

```text
T1/T2 imagery
  -> Change Perception
  -> ChangeObject construction
  -> SAM3 Change Adapter
     T1/T2 local tokens
     + change probability
     + change object box
       -> TemporalTokenFusion
       -> ProbabilityInjection
       -> BoxPromptEncoder
       -> SAM3 backend / GeometrySemanticDecoder
       -> refined geometry + semantic correction evidence
  -> Instruction-tuned temporal VLM
  -> EvidenceState fusion
  -> AgentController
     -> ACCEPT / REFINE / VERIFY
  -> semantic transition decoding and geospatial export
```

The repository is organized around one shared intermediate representation, `ChangeObject`, one normalized evidence container, `EvidenceState`, and one deterministic routing controller, `AgentController`.

## Current status

### Phase 1: core state/control — implemented

- `ChangeObject`: object-level geometry, temporal semantics, model references, trace, and serialization.
- `EvidenceState`: normalized change / geometry / semantic / verification evidence.
- `AgentController`: evidence-driven `ACCEPT / REFINE / VERIFY` routing.

### Phase 2: SAM3 change-conditioned grounding — implemented skeleton

- `TemporalTokenFusion`: aligns T1/T2 local tokens and fuses endpoint, difference, and agreement evidence.
- `ProbabilityInjection`: resizes and injects the soft change-probability map into token features.
- `BoxPromptEncoder`: converts `xyxy` change boxes into prompt tokens and a spatial support prior.
- `GeometrySemanticDecoder`: auxiliary trainable geometry head plus optional T1/T2 semantic correction heads.
- `SAM3ChangeAdapter`: orchestrates all grounding signals and exposes a `SAM3Backend` protocol for the real local SAM3 implementation.
- `SAM3ChangeAdapterResult.apply_to_change_object(...)`: writes boundary confidence, semantic difference, probability support, box support, mask references, and trace information back into `ChangeObject / EvidenceState`.

The repository does **not** yet contain the concrete local SAM3 model wrapper/checkpoint API. Until that backend is connected, `SAM3ChangeAdapter` uses its auxiliary decoder so the conditioning pipeline and tests remain executable.

## Grounding contract

```text
F_t1, F_t2, P_change, B_change
             |
             v
     TemporalTokenFusion
             |
             v
     ProbabilityInjection
             |
             v
       BoxPromptEncoder
             |
             v
      conditioned tokens
             |
       +-----+-----+
       |           |
       v           v
  real SAM3    auxiliary decoder
   backend      (fallback/test)
       |           |
       +-----+-----+
             v
 refined mask / boundary confidence
 T1/T2 semantic correction logits
 semantic difference
 probability support
 box support
             |
             v
 ChangeObject + EvidenceState
```

## Installation

Base state/controller code has no heavy ML dependency:

```bash
pip install -e .
```

Grounding modules require PyTorch:

```bash
pip install -e ".[grounding]"
```

Development tests:

```bash
pip install -e ".[dev,grounding]"
pytest -q
```

## Package layout

```text
src/vlm_agent/
├── schemas/
│   ├── change_object.py
│   └── evidence.py
├── grounding/
│   ├── temporal_token_fuser.py
│   ├── probability_injector.py
│   ├── box_prompt_encoder.py
│   ├── geometry_semantic_decoder.py
│   └── sam3_change_adapter.py
└── agent/
    └── controller.py
```

## Next integrations

1. existing change detector -> real `T1/T2 tokens + P_change + B_change`;
2. local SAM3 implementation -> `SAM3Backend.decode_change(...)`;
3. persistence/vectorization -> save refined mask and polygon then write references into `ChangeObject`;
4. instruction-tuned VLM -> T1/T2 semantic recognition and real-vs-pseudo change verification;
5. deterministic transition decoder -> `T1 class -> T2 class -> from_to`;
6. activity decoder and GeoJSON / semantic-map export.

## Design principles

- Models generate evidence; the agent selects actions.
- SAM3 is change-conditioned, not an unconstrained post-processing step.
- Change probability is a soft prior, not a hard replacement mask.
- T1/T2 semantic labels are explicit; `from_to` is derived deterministically when both labels are available.
- Large tensors and masks are referenced by URI/path rather than embedded in JSON state.
- Routing thresholds are policy configuration and must be calibrated on validation data rather than treated as fixed scientific constants.
