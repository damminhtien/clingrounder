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
| medspaCy | Typed context rules, termination, modifier evidence | `context/` rule contracts |
| NegBio | Optional dependency evidence anchored to a target | future context adapter; linear rules remain fallback |
| cTAKES | Independent assertion attributes and per-attribute metrics | `AssertionFeatures`, `evaluation/context_metrics.py` |
| SapBERT | Same-concept synonym curriculum | `training/terminology_pairs.py` |
| biomedical-entity-linking | Shared candidate benchmark and abbreviation-first retrieval | terminology evaluation and query expansion |
| Prompt-BioEL | Cross-candidate/listwise reranking records | `training/listwise_linking.py` |
| MedXN | Drug span plus structured attributes and RxNorm composition | medication mention parser and structured RxNorm linker |
| VietMed-NER | Vietnamese encoder curriculum and noisy-text slices | model mining/training adapters |
| ViHealthBERT | Healthcare-domain representation baseline | optional verifier/encoder benchmark |

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
