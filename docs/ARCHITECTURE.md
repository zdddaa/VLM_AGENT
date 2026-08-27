# VLM_AGENT Architecture

## 1. Core online route

```text
T1/T2 imagery
  -> Change detector
     -> coarse change mask
     -> change probability
     -> local T1/T2 tokens
     -> candidate change boxes
  -> ChangeObject
  -> SAM3 Change Adapter
     input: T1 tokens + T2 tokens + change probability + change box
     output: refined mask/polygon + boundary confidence + local semantic evidence
  -> instruction-tuned temporal VLM
     output: T1 semantic class + T2 semantic class + real/pseudo-change evidence
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
│   └── polygon_geojson
├── change_probability_ref
├── t1_token_ref
├── t2_token_ref
├── TemporalSemanticState
│   ├── t1_class
│   ├── t2_class
│   └── from_to (derived)
├── EvidenceState
│   ├── cd_confidence
│   ├── boundary_confidence
│   ├── seed_coverage
│   ├── t1_semantic_confidence
│   ├── t2_semantic_confidence
│   ├── semantic_difference
│   ├── pseudo_change_risk
│   ├── semantic_conflict
│   └── geometry_anomaly
└── trace
```

## 3. SAM3 Change Adapter contract

SAM3 is not treated as an unconstrained mask post-processor.

```text
F_t1, F_t2, P_change, B_change
            |
            v
      Change Adapter
            |
            v
conditioned temporal-change tokens
            |
            v
          SAM3
            |
            +--> refined boundary mask
            +--> polygon
            +--> boundary confidence
            +--> semantic refinement evidence
```

The adapter should combine temporal token alignment/fusion, change-probability injection, and box-prompt encoding. Candidate geometry must be checked before it replaces the coarse geometry.

## 4. Evidence-guided routing

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

## 5. Offline instruction tuning

The text-image instruction corpus is used to adapt the temporal VLM, not to replace pixel-level change detection.

Recommended task families:

1. T1 land-cover semantics;
2. T2 land-cover semantics;
3. dual-temporal change understanding;
4. real-vs-pseudo change verification;
5. semantic conflict correction.

The online pipeline consumes the fine-tuned VLM only. Human labels, instruction generation, and training-data construction remain offline.
