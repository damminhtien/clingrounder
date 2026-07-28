# Phase 1 Round 2: ICD Top-1 Probe

## Public Result

| Metric | Public-score reference | `C_ICD_TOP1_KEEP_RX` | Delta |
| --- | ---: | ---: | ---: |
| Final | 33.0750 | 32.4501 | -0.6249 |
| WER | 63.4134 | 63.4134 | 0.0000 |
| J_assertion | 41.7462 | 41.7462 | 0.0000 |
| J_candidates | 23.9379 | 22.3758 | -1.5621 |

Submitted ZIP:

```text
outputs/phase1/round2/
20260728T043818Z_round2-33-075-icd-top1-keep-rx_c493af0bd9/
variants/C_ICD_TOP1_KEEP_RX/output.zip
```

SHA-256:

```text
858f025b425981d69ccb1efac531b9bb24b7b9df0ba89d60d1e64dfc615a08ab
```

## Isolation Evidence

- entity count: 3,385, unchanged;
- assertion rows: unchanged;
- RxNorm candidates: unchanged;
- diagnosis rows truncated: 195;
- ICD values removed: 1,253;
- strict validation and candidate isolation issues: 0.

## Decision

Reject. Keep `1b375c092bb5...` as the public-score reference. The public loss
shows that useful diagnosis codes frequently occur below the current lexical
rank one. Do not retry first-candidate ICD truncation; improve candidate
recall/evidence and introduce a calibrated reranker before another ICD depth
probe.
