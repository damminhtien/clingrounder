# Candidate, RxNorm, And Rulebase Audit - 2026-07-12

This audit closes the actionable findings from the candidate/RxNorm review against the current
implementation. It does not promote a new public candidate regime without an isolated grader win.

## Public Baseline

- Frozen best: `41.8577`
- WER: `50.8167`
- J assertion: `49.7245`
- J candidates: `30.4634`
- Winning modules: `E_OVERLAP + A_NEG + A_HIST`
- Next isolated probe: none approved; return to entity WER/boundary analysis
- Artifact SHA-256: `ab9644f4d635132eee6462929461cf0d9b3bd3b9059e5a598704349410b54644`

`A_NEG_HIST` received public score `41.8577`. Against `A_NEG`, J_assertion increased by `6.7045`
and final score by `2.0114`; WER and J_candidates were unchanged. The public isolation gate passed,
so selective history is promoted rather than treated as a local-only result.

## P0 Status

| Finding | Status | Evidence |
| --- | --- | --- |
| Candidate abstention stopped at `entity.code` | Fixed | Every internal candidate now records `qualified` and `qualification_reason`; Phase 1 exports only qualified candidates. |
| Fixed top-1/top-5 behavior | Fixed | Qualification uses absolute threshold plus relative top-score margin and caps output at five. |
| Eliquis conflict | Fixed | Bare `Eliquis` maps to ingredient-level apixaban `1364430`; explicit `apixaban 5 mg oral tablet` maps to SCD `1364445`. Manual gold 28, audit text, runtime dictionary, and tests agree. |
| RxNorm concept-level policy missing | Fixed | Ingredient/product policy and structured-product `blocked_aliases` are documented and applied to the controlled dictionary. |

Unqualified candidates remain in internal predictions for recall, MRR, source, and error analysis.
Legacy candidate JSON without a qualification field defaults to unqualified.

## P1 Status

| Finding | Status | Evidence |
| --- | --- | --- |
| One global threshold | Infrastructure complete; values not promoted | Pipeline options support global, entity-type, and retrieval-source thresholds. Source overrides type, and type overrides global. Reports expose coverage and rejection reasons. Numeric overrides remain empty until calibrated on held-out evidence. |
| Exact match still ran all retrievers | Fixed | One exact, type-compatible output code returns immediately; ambiguous exact aliases continue through approximate retrieval. |
| Assertion rules dispersed | Mostly fixed | Cue inventory and section priors live in one source JSONL, fallback cue lists were removed, cues have stable rule IDs, and trace counters show matched rules. Generic scope execution and explicit false-positive patterns remain Python engine logic. |
| Phase 1 patched drug spans at export | Fixed | NER creates a structured `MedicationMention` with drug/full/component spans. Export reads the validated full span and contains no medication extension regex. |
| Compound/strength/form tests missing | Fixed | Regression tests cover slash compounds, concatenated products, strength ratios, salts, forms, routes, frequencies, indication stops, and original offsets. Drug overlap resolution favors reviewed combination/product aliases over ingredient splits. |

## P2 Status

| Finding | Decision |
| --- | --- |
| Chemical-aware lexical/subword retrieval | Deferred. Exact/alias, fuzzy, char n-gram, and BM25 already exist; add a dedicated chemical tokenizer only after recall@20 profiling identifies a remaining chemical-name gap. |
| Structured reranking | Partially implemented. Reranking now uses ingredient, strength, unit-bearing strength text, dose form, brand evidence, and RxNorm TTY compatibility. Route/frequency affect the structured mention but do not identify an RxNorm concept by themselves. |
| Dense retrieval/model | Deferred by design. No benchmark currently justifies model/runtime complexity over the deterministic path. |

## Verification

- Ruff: pass
- mypy: pass across `src`
- pytest in sandbox: `275 passed, 1 deselected`
- process-pool test outside sandbox: `1 passed`
- prediction smoke validation: `1` row, `0` issues
- manual gold validation: `75` reviewed files, `2244` entities, `0` issues
- `A_NEG_HIST` Phase 1 ZIP: `100` records, `0` schema/offset/dictionary/ZIP issues

## Promotion Rule

Do not replace the frozen `41.8577` public baseline with full candidate output. Candidate
qualification fixes correctness and enables controlled probes, but public promotion still requires
an isolated `J_candidates` and final-score increase. The prior C3 candidate probe is rejected
evidence, not a threshold-tuning target.

The candidate ZIPs in run `a707b01002de` are not submission probes: their decision sets are empty
because that run had two proposal sources while candidate consensus was enabled only for three-source
runs. The suite now supports an explicit minimum source count: two sources mean exact 2-of-2
agreement, while three sources retain 2-of-3 at the default threshold. Zero-emit candidate variants
cannot be marked probe-ready.

Run `0e4ecf906c8d` was regenerated from the public-winning `A_NEG_HIST` artifact. Its reviewed
candidate compiler rejects non-exact, ambiguous, type-incompatible, and reviewed/dictionary-conflict
mappings. This reduced the first ICD probe to 132 emits and improved local train candidate score by
3.7288 percentage points without changing entities or assertions. The strict-validated next probe is:

`outputs/phase1/top10_probes/top10_0e4ecf906c8d/variants/C_ICD20/output.zip`

SHA-256: `2fb72ebb2fd6b023799576431c8e71354627ecc70601b8535aef9d9c543f061e`.

## C_ICD20 Public Rejection

The public grader scored `C_ICD20` at `41.1189`, with J_candidates `28.6164`. Relative to the
frozen `A_NEG_HIST` baseline, this is `-1.8470` J_candidates and `-0.7388` final score; WER and
J_assertion were unchanged. The candidate isolation gate rejected the probe.

This falsifies the assumption that exact-unique TT06 mappings from the current reviewed train data
generalize to hidden candidate labels. It is consistent with the earlier C3 failure despite a much
smaller and stricter decision set. Keep submission candidates empty and block `C_ICD100`,
`C_RX_ING`, and `C_RX_SCD` until independent evidence reveals the hidden candidate convention.
Do not reinterpret this as a threshold-calibration problem.
