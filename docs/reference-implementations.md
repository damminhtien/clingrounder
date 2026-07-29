# Clinical NLP Reference Implementations

The project studies external systems at pinned revisions without making them runtime
dependencies. Reproduce the checkouts with:

```bash
uv run python scripts/audit_reference_implementations.py --sync
```

The source registry is
`configs/references/clinical_nlp_sources.json`. It records the exact revision, license evidence,
files inspected, patterns adopted, and patterns explicitly rejected. Checkouts live under
`external/reference_repos/` and are ignored by Git.

## Adoption Matrix

| Source | Adopted design | Local destination |
| --- | --- | --- |
| medspaCy | Typed context rules, section metadata, modifier evidence | `context/`, `preprocessing/section_rules.py` |
| NegBio | Optional dependency evidence anchored to a target | future context adapter; linear rules remain fallback |
| cTAKES | Independent assertion attributes, sparse features, per-attribute metrics | `context/features.py`, `evaluation/context_metrics.py` |
| SapBERT | Same-concept synonym curriculum | `training/terminology_pairs.py` |
| biomedical-entity-linking | Shared candidate benchmark and abbreviation-first retrieval | terminology evaluation and query expansion |
| Prompt-BioEL | Cross-candidate/listwise reranking records | `training/listwise_linking.py` |
| MedXN | Drug span plus structured attributes and RxNorm composition | medication mention parser and structured RxNorm linker |
| VietMed-NER | Vietnamese encoder curriculum and noisy-text slices | model mining/training adapters |
| ViHealthBERT | Healthcare-domain representation baseline | optional verifier/encoder benchmark |

## Source-by-Source Findings

### medspaCy

Inspected `context_rule.py`, `context_graph.py`, and the section detector. The useful boundary is
modifier -> target evidence, not spaCy itself. Local adoption:

- `AssertionCue` owns type filters, direction, priority, distance, and termination.
- `ContextGraph` stores one raw-coordinate modifier node and one edge per affected entity.
- `SectionRuleRegistry` owns aliases, category, parent constraints, and maximum character scope.
- Pipeline batch assertion records graph counters while preserving the existing assertion policy.

English resources, spaCy spans, and destructive preprocessing were not imported.

### NegBio

Inspected target-node anchored dependency patterns and semantic graphs. Dependency paths can help
when linear cue scope is ambiguous, but parser errors are a new failure mode in translated or
unsegmented Vietnamese text. No dependency adapter is active. A future adapter must:

1. project parser tokens reversibly to source offsets;
2. add evidence rather than erase linear evidence;
3. beat the linear baseline by assertion attribute and genre;
4. fail open when the parse is absent or invalid.

### Apache cTAKES

Inspected the history, subject, genericity, polarity, and evaluation modules. The adopted design
keeps assertion dimensions independent and separates feature extraction from decoding:

- `AssertionFeatures` stores negated, historical, family, possible, conditional, planned, and
  resolved independently.
- `AssertionModelFeatureExtractor` exposes section, entity type, target position, modifier
  direction/distance, semantic assertion, and bounded rule-ID features.
- `assertion_attribute_metrics()` reports each dimension independently.

The UIMA runtime, Java type system, English lexical windows, and dependency features are excluded.

### SapBERT

Inspected the metric-learning loader, multi-similarity objective, and retrieval evaluation. Local
training pairs are generated from concepts in the pinned terminology repository:

- positives must share one concept ID;
- aliases are bounded per concept to avoid quadratic expansion;
- type and code system remain explicit;
- retrieval recall@k and MRR are measured before reranking.

The joint XLM-R DAPT trainer can consume these pairs as a synonym-contrastive objective while its
Round 2 unlabeled lane remains MLM-only.

### biomedical-entity-linking

Inspected the common model evaluation, candidate-generation failure analysis, and abbreviation
resolution flow. Local adoption is model-neutral:

- `TerminologyQuery` supplies one shared benchmark contract;
- exact, toneless, and FTS modes report recall@5/10/20, MRR, latency, ambiguity, and abstention;
- slices separate seen/unseen aliases and concepts;
- reviewed abbreviation expansion runs before approximate retrieval.

No benchmark framework is allowed to bypass local entity-type or code-system constraints.

### Prompt-BioEL

Inspected candidate generation and prompt reranking. The useful idea is cross-candidate
interaction. `ListwiseLinkingRecord` retains the complete bounded candidate set, supports multiple
valid positive codes, randomizes option order deterministically during training, and evaluates any
reranker with hit@k and MRR. Runtime candidate count remains dynamic rather than the reference
implementation's fixed choice count.

### MedXN

Inspected medication attribute extraction and RxCUI normalization. The local medication contract
separates:

- raw drug span and full medication span;
- strength, administered dose, form, route, frequency, duration, release, and transition;
- ingredient/brand retrieval from product compatibility;
- hard product conflicts from soft administration evidence.

All components retain raw offsets. In particular, `po` and `iv` are routes, not evidence for tablet
or injection products.

### MultiMed / VietMed-NER

Inspected the encoder NER path, word-to-subword alignment, modified span metrics, and noisy ASR
evaluation. VietMed data is used as Vietnamese medical representation/curriculum input with pinned
provenance. Its source labels are auxiliary and never treated as the target five-type schema.
Reference text and ASR-like noise should remain separate evaluation slices.

### ViHealthBERT

Inspected the GitHub NER preprocessing and encoder/CRF implementation at the pinned revision.
ViHealthBERT is eligible as a Vietnamese healthcare representation baseline or verifier
initialization. It does not own exported boundaries because its word-level path requires
segmentation that is not yet reversibly mapped to source text.

## Integration Rules

- English rules and resource inventories are not copied into Vietnamese runtime resources.
- The user has confirmed authorization to study and use every source in this registry. License
  metadata remains recorded for provenance, but it is not an integration blocker for this project.
- Third-party token or word offsets are never exported. Model spans are projected to raw text by
  local offset-safe adapters.
- UMLS, RxNorm, and other terminology data retain their own access and redistribution terms.
- ViHealthBERT requires word segmentation. Until a reversible segmentation map is benchmarked, it
  may initialize a verifier or encoder but cannot own final NER boundaries.
- VietMed-NER source labels are auxiliary supervision. They are not treated as the target schema.

## Audit Notes

medspaCy and NegBio both model a relationship between a context modifier and a target instead of
storing only a sentence-level flag. medspaCy contributes explicit target-type filters,
termination, pseudo rules, scope limits, and evidence edges. NegBio adds dependency paths but also
shows why parsing must be optional: a failed or domain-mismatched parse cannot be allowed to erase
deterministic evidence.

cTAKES trains polarity, uncertainty, subject, history, genericity, and conditional status as
separate attributes. The local schema already follows that principle through `AssertionFeatures`;
new model adapters should preserve independent outputs rather than force a single multiclass label.

SapBERT's useful contribution here is the training contract, not a particular English/UMLS
checkpoint. Positive pairs must share one terminology concept and must remain bounded to avoid
quadratic alias explosion. Prompt-BioEL motivates presenting the complete retrieved candidate set
to a reranker so that close alternatives can interact.

MedXN validates the current medication architecture: preserve the drug span, extract contiguous
SIG components separately, then compare structured mention evidence with RxNorm ingredient,
strength, release type, and dose form. The local implementation additionally keeps every component
at raw offsets and treats route/frequency as administration evidence rather than product form.

## Deliberate Deferrals

- NegBio dependency parsing is deferred until a Vietnamese dependency parser beats the linear
  scope baseline on target-anchored assertion metrics. Parse evidence must remain optional and fail
  open when parsing fails.
- ViHealthBERT boundary ownership is deferred until reversible Vietnamese word segmentation passes
  raw-offset round-trip and boundary benchmarks. It can supply representation features now.
- Prompt-BioEL and ViHealthBERT are architecture references in the core implementation. Their
  model code is not copied; local adapters keep model revision, candidate constraints, and raw
  projection under this repository's contracts.
