# Round 2 Max-Score Rejection, 2026-07-29

## Public Result

The deterministic artifact below is rejected:

```text
ZIP SHA-256: 21f0a06b4cd9e9e7067b62026a256c6dc2aaf479ce2e9bf3bdee1d6a6e4dd007
score: 25.2683
WER: 68.5798
J_assertion: 35.5632
J_candidates: 12.9332
```

The frozen public reference remains:

```text
ZIP SHA-256: 1b375c092bb5affcbbd830661870bf39c736eed31d4c6dcb6eec2c522e7c558f
score: 33.0750
WER: 63.4134
J_assertion: 41.7462
J_candidates: 23.9379
```

## Score Decomposition

Relative to the public reference, the rejected artifact lost `7.8067` final points:

| Component | Metric regression | Weighted final loss |
| --- | ---: | ---: |
| Entity text/WER | `+5.1664` WER | `1.5499` |
| Assertion | `-6.1830` Jaccard | `1.8549` |
| Candidate | `-11.0047` Jaccard | `4.4019` |

Candidate handling caused about 56% of the final regression.

## Projection Audit

The similar entity counts hid a large identity replacement:

```text
public reference entities: 3,385
rejected entities: 3,269
exact span/type overlap: 1,718
reference-only: 1,667
rejected-only: 1,551
entity Jaccard: 0.3481
same-span type conflicts: 184
```

The candidate policy also changed the task convention:

```text
public reference candidate rows: 591
public reference candidate values: 1,844
rejected candidate rows: 413
rejected candidate values: 413
changed candidates on common exact entities: 333
```

The public scorer rewards useful multi-code ICD recall. The one-code cardinality assumption and
development-derived expected-Jaccard policy were therefore wrong for this hidden-gold regime.

## Root Causes

1. Aggregate Round 2 density was treated as a valid operating-point target. Matching the number of
   entities did not preserve exact spans or types.
2. The proposal verifier was trained on reviewed Round 1 development data whose distribution does
   not reproduce Round 2 hidden gold.
3. Low thresholds promoted many XLM-R-only and Qwen-only proposals while suppressing stronger
   external-reference entities through global overlap resolution.
4. Candidate metadata was copied only from exact selected identities and then compressed to one
   code, so entity drift caused both direct WER loss and secondary candidate/assertion loss.
5. The local DAPT specialization also failed its own gate (`0.7146` exact-span F1 versus `0.7239`);
   it was correctly excluded and did not cause this public regression.

## Locked Policy

- Never submit this artifact or reuse its density thresholds.
- Keep `1b375c...` as the public-score control.
- Entity probes must preserve assertion and candidate projections for every unchanged identity.
- Candidate probes must preserve entity and assertion projections byte-for-byte.
- Do not compress existing ICD lists until an isolated public probe proves that policy.
- Do not promote a union/fusion model from manual-gold score or entity density.
- Future additions must be small, source-attributed, exact-quote projected, and public-probed
  independently before composition.

The next low-risk candidate experiment is the existing 19-row reviewed RxNorm fill-empty control.
It changes no entity, assertion, or existing non-empty candidate:

```text
ZIP SHA-256: d7594bc662657b1a420b46b015bfb33928792bfecbbb55afe4861134046424fe
```
