# Phase 1 Round 2 Rule Baseline - 2026-07-24

## Public Result

- Decision: `reject`
- Primary score: `21.3318`
- WER: `75.3792`
- J assertion: `26.5599`
- J candidates: `14.9439`
- Records scored: `100`
- Artifact:
  `outputs/phase1/round2/20260724T030817Z_round2-rule-frozen-public-43.2014_e1bcf980c2/output.zip`
- SHA-256: `97a471d4d42ebaa12c3db2a1675e98695ca7714792d67c16e5edc7ac2af44a3d`

The SHA matches the deterministic local manifest. The submission had 100 files and zero schema,
offset, code-system, or ZIP validation issue. This rules out packaging failure.

## Interpretation

The prior Round 1 artifact scored `43.2014`, WER `50.8167`, J assertion `49.7245`, and J candidates
`33.8226`. Applying its frozen rule policy to the new input lost `21.8696` primary points and
increased WER by `24.5625`. Since the input round changed, these deltas diagnose distribution
transfer rather than a source-code regression.

The rule output emitted 1,909 entities, including 601 diagnoses, but two documents had no entities.
Inspection of a novelty document showed a missed central G6PD-deficiency diagnosis and repeated
secondary broad diagnoses. The immediate bottleneck is therefore entity recognition and boundary
coverage for mixed Vietnamese health education and question-and-answer text. Candidate or assertion
changes cannot repair those missing spans.

## Controlled Next Step

One already-built metadata probe clears assertions and candidates while preserving the exact entity
projection:

```text
outputs/phase1/round2/20260724T032159Z_round2-rule-empty-metadata-probe_2496ad5dd6/output.zip
SHA-256 68bcf7e8a3ac3beffc8a5c3557d4af710157d296351d6525ab0fc96e00447d00
```

Its WER must remain unchanged because its `(text, type, position)` projection hash is identical.
Submit it only to isolate the Round 2 metadata convention. Do not promote it as an entity fix.

The next entity-bearing experiment is:

1. Complete the pinned five-type XLM-R full run on an authorized Linux/CUDA BF16 host.
2. Calibrate per-type thresholds on the frozen 16-document development split.
3. Compare rule-only, model-only, and dictionary-priority hybrid on the sealed local gate.
4. Build a Round 2 artifact only for a variant that passes the existing missing, spurious,
   boundary, and overall-score gates.

Do not train, pseudo-label, copy prior annotations, or create document-specific rules from Round 2
input.
