# VLM_AGENT Architecture

## 1. Core online route

```text
T1/T2 imagery
  -> Change detector
     -> coarse change mask
     -> change probability P_change
     -> local T1/T2 tokens F_t1 / F_t2
     -> candidate change boxes B_change
  -> ChangeObject
  -> SAM3 Change Adapter
     -> TemporalTokenFusion
     -> ProbabilityInjection
     -> BoxPromptEncoder
     -> conditioned change tokens
     -> real SAM3 backend OR auxiliary GeometrySemanticDecoder
     -> refined mask/polygon + boundary/semantic evidence
  -> instruction-tuned temporal VLM
     -> T1 semantic class + T2 semantic class + real/pseudo-change evidence
  -> EvidenceState
  -> AgentController
     ACCEPT: decode/export current state
     REFINE: invoke SAM3 Change Adapter
     VERIFY: invoke temporal VLM verifier
  -> deterministic transition decoder
     T1 class -> T2 class -> from_to
  -> activity decoder
  -> semantic change map / polygon / GeoJSON / evidence JSON / trace
```

## 2. Canonical ChangeObject

Every model enriches the same object rather than creating parallel incompatible result files.

```text
ChangeObject
├── object_id
├── GeometryState
│   ├── bbox
│   ├── initial_mask_ref
│   ├── refined_mask_ref
│   ├── polygon_geojson
│   └── boundary_source
├── change_probability_ref
├── t1_token_ref
├── t2_token_ref
├── semantic_feature_ref
├── t1_semantic_refinement_ref
├── t2_semantic_refinement_ref
├── TemporalSemanticState
│   ├── t1_class
│   ├── t2_class
│   └── from_to (derived)
├── EvidenceState
│   ├── cd_confidence
│   ├── boundary_confidence
│   ├── seed_coverage
│   ├── probability_support
│   ├── box_support
│   ├── semantic_refinement_confidence
│   ├── t1_semantic_confidence
│   ├── t2_semantic_confidence
│   ├── semantic_difference
│   ├── pseudo_change_risk
│   ├── semantic_conflict
│   └── geometry_anomaly
└── trace
```

Heavy tensors remain outside the JSON state. `ChangeObject` stores paths/URIs to token, mask, semantic-feature, and raster/vector artifacts.

## 3. SAM3 Change Adapter

SAM3 is not treated as an unconstrained mask post-processor. It is conditioned on explicit temporal-change evidence.

```text
F_t1 -----------┐
                │
F_t2 -----------┼--> TemporalTokenFusion
                │        |
                │        v
P_change -------┼--> ProbabilityInjection
                │        |
                │        v
B_change -------┴--> BoxPromptEncoder
                         |
                         v
                 conditioned tokens
                         |
                +--------+--------+
                |                 |
                v                 v
          SAM3Backend       GeometrySemanticDecoder
          (real model)      (auxiliary/fallback)
                |                 |
                +--------+--------+
                         v
            refined geometry + semantic evidence
```

### 3.1 TemporalTokenFusion

Input layouts:

- `[B, N, C]` token sequence, or
- `[B, C, H, W]` feature map.

The module preserves both temporal endpoints and forms four complementary terms:

```text
T1 adapted
T2 adapted
|T2 - T1|
T1 * T2
```

These terms are projected into one change-conditioned representation. The module also returns token-wise cosine semantic difference in `[0, 1]` for evidence fusion.

### 3.2 ProbabilityInjection

`P_change` is treated as a **soft prior**. It is resized to the token grid, embedded, and used as a learnable gate. It never replaces the object mask directly.

The module reports `probability_support`, later recomputed inside the change box by `SAM3ChangeAdapter`.

### 3.3 BoxPromptEncoder

Pixel-space `xyxy` boxes are normalized by image width/height. The encoder outputs:

- learned box prompt tokens;
- a token-grid spatial support prior;
- normalized box coordinates;
- box area ratio.

This keeps the spatial prior explicit and prevents SAM3 from freely expanding to unrelated objects.

### 3.4 GeometrySemanticDecoder

This is an **auxiliary trainable correction head**, not a substitute for the native SAM3 decoder. It provides:

- change-mask logits;
- provisional boundary confidence;
- optional T1 semantic correction logits;
- optional T2 semantic correction logits;
- semantic refinement confidence.

When a concrete SAM3 backend is connected, its outputs take priority; missing backend fields can still fall back to the auxiliary heads.

### 3.5 SAM3Backend protocol

The repository deliberately does not hard-code a specific local SAM3 API. A backend must implement:

```python
decode_change(
    conditioned_tokens=...,
    t1_tokens=...,
    t2_tokens=...,
    probability_tokens=...,
    box_prompt_tokens=...,
    box_spatial_prior=...,
    spatial_shape=...,
    image_size=...,
)
```

and return normalized `SAM3BackendOutput` fields. This isolates future model/checkpoint/API changes from the rest of the agent.

## 4. Grounding evidence written back to the object

`SAM3ChangeAdapterResult.apply_to_change_object(...)` updates lightweight state only:

```text
GeometryState.refined_mask_ref
GeometryState.polygon_geojson
GeometryState.boundary_source
ChangeObject.semantic_feature_ref
EvidenceState.boundary_confidence
EvidenceState.semantic_difference
EvidenceState.probability_support
EvidenceState.box_support
EvidenceState.semantic_refinement_confidence
ChangeObject.trace
```

### Evidence meanings

- `boundary_confidence`: confidence of refined geometry; native SAM3/backend score should supersede the auxiliary estimate when available.
- `semantic_difference`: box-weighted T1/T2 token cosine difference.
- `probability_support`: mean CD probability supported inside the change box.
- `box_support`: proportion of predicted mask probability mass supported by the change box.
- `semantic_refinement_confidence`: auxiliary confidence from the T1/T2 semantic correction heads; it must be calibrated before scientific interpretation.

## 5. Evidence-guided routing

The controller is deterministic in phase 1. It does not inspect images itself.

Priority:

1. unreliable geometry -> `REFINE`;
2. semantic conflict / low semantic confidence / pseudo-change risk / CD-semantic mismatch -> `VERIFY`;
3. otherwise -> `ACCEPT`.

Useful cross-evidence cases:

| Change evidence | Semantic difference | Interpretation |
| --- | --- | --- |
| high | high | consistent real-change evidence |
| high | low | appearance/pseudo-change risk -> verify |
| low | high | underestimated/missed-change risk -> verify |
| low | low | weak candidate -> verify then suppress if confirmed |

All thresholds are configuration values to calibrate on held-out validation data.

## 6. Offline instruction tuning

The text-image instruction corpus is used to adapt the temporal VLM, not to replace pixel-level change detection.

Recommended task families:

1. T1 land-cover semantics;
2. T2 land-cover semantics;
3. dual-temporal change understanding;
4. real-vs-pseudo change verification;
5. semantic conflict correction.

The online pipeline consumes the fine-tuned VLM only. Human labels, instruction generation, and training-data construction remain offline.

## 7. Current implementation boundary

Implemented now:

- ChangeObject / EvidenceState / AgentController;
- TemporalTokenFusion;
- ProbabilityInjection;
- BoxPromptEncoder;
- GeometrySemanticDecoder;
- SAM3ChangeAdapter and backend protocol;
- grounding unit tests and optional PyTorch dependency.

Not yet connected:

- real change-detector token/probability outputs;
- concrete local SAM3 model/checkpoint wrapper;
- mask persistence and polygon vectorization;
- instruction-tuned temporal VLM;
- end-to-end inference runner and validation metrics.
