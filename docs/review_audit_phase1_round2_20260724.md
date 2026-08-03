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
`33.8226`. The Round 2 submission did **not** reproduce that artifact's composition, so the
`21.8696`-point delta is not an apples-to-apples transfer measurement:

| Component | Round 1 public winner | Submitted Round 2 artifact |
| --- | --- | --- |
| entity source | pipeline + Qwen-derived ensemble, then `E_OVERLAP` | current rule pipeline only |
| entities | 2,809 | 1,909 |
| assertion policy | selective `A_NEG_HIST`, 500 asserted rows | generic pipeline context, 507 asserted rows |
| candidate policy | reviewed RxNorm overlay on 176 drugs; no diagnosis codes | all 601 diagnoses and all 179 drugs received pipeline candidates |
| candidate values | 354 | 899 |

Running the current Round 2 rule configuration on the unchanged Round 1 inputs produced 2,028
entities. It retained only 1,508 of the 2,809 entity identities in the public winner. By contrast,
it retained 1,890 of 2,002 identities from the earlier rule-only proposal source. The rule
implementation is therefore broadly stable relative to the weaker rule baseline, but the submitted
artifact was incorrectly described as the frozen `43.2014` system.

The closest scored rule-only Round 1 artifact had WER `58.2396`; the stronger ensemble reduced that
to `50.8167`. Relative to the rule-only WER, Round 2 still regressed by about `17.14` WER points.
That residual is genuine distribution/annotation transfer failure, while about `7.42` points of the
headline WER difference came from omitting the winning entity ensemble.

The rule output emitted 1,909 entities, including 601 diagnoses, but two documents had no entities.
Inspection of a novelty document showed a missed central G6PD-deficiency diagnosis and repeated
secondary broad diagnoses. The immediate bottleneck is therefore entity recognition and boundary
coverage for mixed Vietnamese health education and question-and-answer text. Candidate or assertion
changes cannot repair those missing spans.

## BTC Convention Audit

The supplied specification has not changed the ZIP layout, five entity types, raw character
positions, metric weights, or wrong-type penalty. The submitted score also exactly satisfies the
published formula, so there is no evidence of a packaging or scorer-display fault.

The official medication example establishes these executable conventions:

- medication spans include strength, route, and frequency but exclude the list number and
  `điều trị`;
- an indication after `điều trị` is a separate symptom span;
- `isHistorical` applies to the pre-admission drugs, but does not automatically propagate to their
  indication symptoms;
- symptom boundaries follow the observed phrase: `sốt đau` is one span, while `lo âu mất ngủ` is
  split into two;
- RxNorm gold is not derivable by exact surface matching alone. For example, text without an
  explicit guaifenesin strength maps to the 800 mg SCD `392085`, and `clonazepam 1.5 mg` maps to
  the 1 mg SCD `197528`.

The last point strongly suggests that some candidate labels were inherited from structured source
records before translation or text corruption. Broad exact/fuzzy linking against the rendered
Vietnamese text cannot be assumed to reproduce that gold.

Round 2 is also structurally different:

- mean length increased from 1,323 to 2,038 characters;
- 54 documents contain question/answer style text, and only 37 are purely clinical style;
- 98 documents reuse at least one exact Round 1 line, but exact old lines cover only `42.3932%` of
  non-empty-line characters;
- 1,387 submitted entities were inside reused exact lines and 522 were outside them;
- 61 documents have less than 50% exact-line coverage from Round 1.

Prediction density fell from 15.33 entities per 1,000 characters for the current rule pipeline on
Round 1 to 9.37 on Round 2. It was 13.51 in clinical-style Round 2 documents but only 4.51 in
question/answer-only documents. This is direct evidence that the rule recognizer under-covers the
new prose style even before the hidden annotation policy is known.

The specification does not state that only the reused clinical block is annotated. The strongest
specification-consistent assumption is therefore to annotate medical concepts throughout the whole
input. However, because almost every document appears to mix reused clinical text with new
educational or question/answer text, a source-label inheritance regime remains plausible and cannot
be ruled out without a controlled public probe or released gold. This uncertainty must be reported,
not treated as fact.

## Controlled Next Step

One already-built metadata probe clears assertions and candidates while preserving the exact entity
projection:

```text
outputs/phase1/round2/20260724T032159Z_round2-rule-empty-metadata-probe_2496ad5dd6/output.zip
SHA-256 68bcf7e8a3ac3beffc8a5c3557d4af710157d296351d6525ab0fc96e00447d00
```

Its WER must remain unchanged because its `(text, type, position)` projection hash is identical.
Submit it only to isolate the Round 2 metadata convention. Do not promote it as an entity fix.

The next probes and entity-bearing experiment are:

1. Use one generic clinical-section-only probe, based only on section/style markers, to distinguish
   whole-document annotation from clinical-block annotation. It must not use Round 1 line hashes,
   annotations, document IDs, or copied offsets.
2. Disable broad diagnosis coding; test empty metadata or a separately gated selective RxNorm
   policy before reintroducing candidates.
3. Complete the pinned five-type XLM-R full run on an authorized Linux/CUDA BF16 host.
4. Calibrate per-type thresholds on the frozen 16-document development split.
5. Compare rule-only, model-only, and dictionary-priority hybrid on the sealed local gate.
6. Build a Round 2 artifact only for a variant that passes the existing missing, spurious,
   boundary, and overall-score gates.

Do not train, pseudo-label, copy prior annotations, or create document-specific rules from Round 2
input.

## Reviewed Recognition Result - 2026-07-25

The reproducible full-pipeline profile in `configs/benchmarks/phase1/submission/round2-full.yaml` added the reviewed VN
clinical lexicon and the train-compiled recognition overlay that had passed the legacy holdout
gate. It retained context, exact full-terminology linking, reranking, and entity KG validation,
while limiting export to one candidate per entity.

Public result for ZIP SHA-256
`2aef0de993189c1cb5a7ea6fc82f4ef54cc2a1350022e17184fa74c7197ac9eb`:

| Measure | Rule baseline | Reviewed recognition | Difference |
| --- | ---: | ---: | ---: |
| primary score | `21.3318` | `22.6633` | `+1.3315` |
| WER | `75.3792` | `72.7063` | `-2.6729` |
| J assertion | `26.5599` | `29.6765` | `+3.1166` |
| J candidates | `14.9439` | `13.9305` | `-1.0134` |

The entity branch is promoted: output density increased from 1,909 to 2,037 entities and public
WER improved. The assertion delta is positive but not isolated because the entity set changed.
The candidate branch is rejected: emitting one exact candidate for 767 entities still reduced the
public candidate metric. The next probe freezes the winning entity/assertion projection and clears
only candidates.

The candidate-empty control was then scored with exactly the same entity/assertion projection:

| Measure | Full candidates | Empty candidates | Difference |
| --- | ---: | ---: | ---: |
| primary score | `22.6633` | `23.7953` | `+1.1320` |
| WER | `72.7063` | `72.7063` | `0.0000` |
| J assertion | `29.6765` | `29.6765` | `0.0000` |
| J candidates | `13.9305` | `16.7606` | `+2.8301` |

Candidate-empty is now the accepted Round 2 baseline. Any candidate reintroduction must preserve
the projection SHA-256
`966c3412904f70cf7f08280223a0445634b1c21bc0f5ded6afdc186fee936506`
and probe one entity/code-system family at a time. Drug-only RxNorm is first because the official
sample requires medication codes while the rejected full artifact assigned 590 diagnosis codes.

The drug-only probe restored 177 one-code RxNorm predictions and kept all diagnosis candidates
empty:

| Measure | Empty candidates | Drug-only RxNorm | Difference |
| --- | ---: | ---: | ---: |
| primary score | `23.7953` | `23.9854` | `+0.1901` |
| WER | `72.7063` | `72.7063` | `0.0000` |
| J assertion | `29.6765` | `29.6765` | `0.0000` |
| J candidates | `16.7606` | `17.2357` | `+0.4751` |

RxNorm drug-only is promoted, but the gain per emitted value is small. The next candidate probe
must remove structurally inconsistent mappings: ingredient or brand codes attached to strength
mentions, route-incompatible SCDF matches, and single-ingredient matches for multi-drug mentions.

The first structure filter removed all three groups together, retaining 153 values. It scored
`23.9547` with J_candidates `17.1591`, decreases of `0.0307` and `0.0766` from the 177-value
drug-only baseline. The grouped filter is rejected. This does not prove every removed group is
useful; it only proves they cannot be removed together. Subsequent probes must isolate SCDF from
ingredient/brand conflicts.

The 166-value probe restored SCDF and removed only 11 ingredient/brand conflicts. It scored
`23.8362` with J_candidates `16.8628`, decreases of `0.1492` and `0.3729` from the accepted
drug-only baseline. Those 11 mappings are therefore restored. Relative to the 153-value grouped
probe, restoring SCDF lowered J_candidates by `0.2963`; the next isolated probe removes only the 13
SCDF mappings from the 177-value baseline.
