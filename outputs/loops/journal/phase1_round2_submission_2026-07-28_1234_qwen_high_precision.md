# Phase 1 Round 2: Qwen High-Precision Consensus

## Public Result

| Metric | Previous baseline | Submitted artifact | Delta |
| --- | ---: | ---: | ---: |
| Final | 33.0410 | 33.0750 | +0.0340 |
| WER | 63.4320 | 63.4134 | -0.0186 |
| J_assertion | 41.7386 | 41.7462 | +0.0076 |
| J_candidates | 23.8724 | 23.9379 | +0.0655 |

Submitted ZIP:

```text
outputs/phase1/round2/
20260728T041737Z_round2-33-qwen-high-precision-all-region_5559e1b058/
variants/E_QWEN_HP_CONSENSUS_ADD/output.zip
```

SHA-256:

```text
1b375c092bb5affcbbd830661870bf39c736eed31d4c6dcb6eec2c522e7c558f
```

## Change

The probe preserved all 3,374 baseline entity identities and added 11
non-overlapping entities accepted by the semantic evidence gate. Assertions and
candidates on existing rows were unchanged. Strict schema, raw offset,
terminology, and ZIP validation reported zero issues.

## Reproducibility Status

This ZIP is fully content-addressed and every post-processing step is
reproducible. Its base entity projection came from the imported `friend31`
artifact, whose original producer has not supplied a verified raw-input-to-ZIP
reproduction run. It is therefore an external-teacher public-score reference,
not the repository's reproducible system baseline.

## Decision

Promote this exact ZIP as the frozen public-score reference because all public
components improved. Do not interpret the small delta as proof that broad Qwen
union is safe: the WER improvement is only `0.0186`, far below the campaign
entity gate. Continue with candidate-only probes and the separately trained
Qwen adapter; do not submit the 36-row replacement variant without independent
evidence. The self-owned Qwen pipeline must be measured separately from this
reference before it is called reproducible.
