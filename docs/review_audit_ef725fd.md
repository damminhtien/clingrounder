# Review Audit: ef725fd

This audit maps the static review of commit `ef725fd40a1df0a059f92b80510130ceaa22d235`
to the implemented remediation. The reviewed commit is the base snapshot; the changes described
below are committed on top of it.

## P0 Remediation

| Review finding | Status | Implemented evidence |
| --- | --- | --- |
| Reranker only re-sorted the existing score | Fixed | `HeuristicReranker` now uses mention strength, explicit dose form, ingredient, and context-token features; conflicting strengths receive a near-hard penalty. |
| BM25 top result was normalized to `1.0` per query | Fixed | BM25 now uses a fixed saturating calibration and inverted postings; it never divides by the current query maximum. |
| Retrieval source scores were mixed directly | Fixed with conservative defaults | Candidate fusion applies fixed source reliability weights plus weighted reciprocal-rank evidence. Thresholds still require held-out calibration before public promotion. |
| Candidate merge was order-dependent and occurrence-based | Fixed | Evidence is grouped once per unique source, ranked deterministically, and deduplicated by `(code_system, code)`. All source evidence remains on the internal candidate. |
| Linker always assigned top-1 | Fixed | Assignment requires both `link_assignment_threshold` and `link_assignment_margin`; candidate lists remain available when normalization abstains. |
| Dictionary NER discarded the matched concept | Fixed | Exact/toneless dictionary entities pin a candidate and code only when the alias has one compatible output code. Ambiguous aliases remain unlinked. |
| Entity-only submission ran discarded stages | Fixed | `configs/benchmarks/phase1/submission/entity-only.yaml` is a validated `entity_only` contract with context/linking/KG/relation stages disabled. `configs/benchmarks/phase1/submission/full.yaml` is the explicit full mode. |
| Cue scope metadata was ignored | Fixed | Left, right, bidirectional, and section-prior cues are loaded separately; section priors no longer leak into lexical cue lists. |
| LAB_TEST/LAB_RESULT were always PRESENT | Fixed | Lab observations use the same scoped assertion classifier as other entities. Regression cases cover negated, planned, and historical labs. |
| Dose/route/frequency used LAB_RESULT | Fixed | Dedicated medication attribute entity types and typed relations replace LAB_RESULT tails. LAB values remain connected only through `HAS_VALUE`. |
| Local entity matching was greedy | Fixed | The Phase 1 surrogate uses maximum-weight Hungarian assignment with dummy abstention columns. |
| Run hash used random UUID entropy | Fixed | Run directories remain unique, while `content_hash` is deterministic over input contents, Git/working-tree state, resolved config, lockfile, and seed. Manifests include command and Python version. |
| Declared Python support did not match CI/docs | Fixed | Runtime documentation and CI now cover Python 3.11, 3.12, 3.13, and 3.14. |
| Installed wheels could silently lose cue/rule data | Fixed | Assertion cues and false-positive records are package data with explicit resource fallback; missing resources raise instead of silently returning empty rules. |

## P1 Remediation

- Dictionary overlap resolution now uses weighted interval scheduling rather than global
  longest-first greediness.
- False-positive exceptions are data records with stable ids, priority, source, positive examples,
  and counterexamples; duplicate hard-coded branches were removed from `RuleBasedNER`.
- Character n-gram document vectors and norms are precomputed once, and both n-gram and BM25 use
  inverted postings.
- Fuzzy matching uses a trigram posting shortlist instead of scanning every alias for every mention.
- The thread backend shares one immutable `PipelineRunner` and its indexes. The full RxNorm config
  uses threads to avoid multiplying dictionary/index RAM by worker count.
- Relation extraction links the nearest eligible entity in a sentence instead of producing the
  full disease-by-symptom/test Cartesian product, and excludes negated/family endpoints.
- Internal assertion features can represent combinations such as historical+negated. The strict
  internal schema records the primary status, all feature flags, and mandatory rule evidence;
  Phase 1 export emits every supported assertion label selected by the active policy.
- `OntologyReasoner` provides dictionary-backed transitive `is-a` closure, hierarchy distance, and
  reasoning-path provenance. `KGValidator` rejects coded `IS_A` edges that contradict a known
  hierarchy path.

## Deliberate Limits

- Source reliability weights and linker thresholds are conservative deterministic defaults, not a
  learned calibration model. Promotion still requires held-out coverage/accuracy and public gates.
- Medication and clinical relation extraction remains rule/nearest-evidence based; it is no longer
  Cartesian, but it is not a dependency parser.
- Ontology reasoning currently covers hierarchy closure and consistency. Disjointness axioms,
  general rule chaining, and ontology-aware candidate context require curated ontology inputs.
- Process workers still replicate indexes when the process backend is explicitly selected. The
  memory-safe full-dictionary mode uses the shared thread backend.
- The official hidden Phase 1 formula remains unknown. Exact-span/relaxed F1, assertion metrics,
  candidate ranking metrics, coverage, WER proxy, source ablation, and boundary reports must remain
  separate objectives.

## Verification Gates

Current local evidence on 11 July 2026:

- `entity_only`: 100/100 documents, strict directory/ZIP validation with zero issues; manual-gold
  holdout text score `0.533484` and local surrogate score `40.448956`.
- `full`: 100/100 documents, strict directory/ZIP validation with zero issues; the same entity
  spans, holdout assertion score `0.544239`, candidate score `0.580247`, and local surrogate score
  `55.541549`.
- The local gate passed for both modes. These values are diagnostics, not evidence of a public
  candidate/assertion improvement.
- Repository verification: Ruff clean, mypy clean across 106 source files, and 227 pytest tests
  passing.

Repeat before merging or after subsequent changes:

```bash
uv run ruff check .
uv run mypy src
uv run pytest tests
```

For Phase 1, additionally build both mode configs and validate the resulting directory/ZIP before
using any public score as evidence.
