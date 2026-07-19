# DailyMed SPL And RxNorm Mappings

## Structured Product Labels

The pinned daily slice contains 105 SPL XML artifacts from the 17 July 2026 catalog snapshot. The
SPL parser renders both narrative labels and compact structured medication records while retaining
exact offsets for product, generic name, ingredient, strength, dosage form, and route.

| Measure | Value |
| --- | ---: |
| SPL artifacts | 105 |
| rendered documents | 239 |
| structured annotations | 841 |
| concept links | 559 |
| medication relations | 707 |
| exact duplicate groups | 1 |
| offset issues | 0 |

The 841 source-structured annotations include 416 drug spans, 148 strengths, 143 routes, and 134
dosage forms. Links preserve NDC, UNII, and NCI provenance. They are not guessed RxNorm codes.

An exact text-only RxNorm crosswalk is diagnostic: among 406 inventory entries, 171 map to multiple
exact codes, 133 have no permitted policy, only 14 are unique, and 88 are unmatched. This is why the
repository does not equate a normalized drug string with an authoritative product code.

## Official SPL-To-RxNorm Mapping

The separate pinned mapping archive has SHA-256
`0d2797b35c31c0651e616d075b8b042591074e66a0c5955d8c4919e50ed9860c`. Compilation produced:

| Measure | Value |
| --- | ---: |
| source rows | 468,456 |
| versioned mappings | 150,925 |
| set/version pairs | 104,292 |
| unique RxCUIs | 24,312 |
| duplicate source rows | 0 |

The pinned 6 July 2026 RxNorm release contains 21,602 of those RxCUIs; 2,710 release-mismatched codes
are rejected. The audit emits 36,076 missing alias pairs for review and never mutates canonical
RxNorm automatically. The daily 105-label slice is newer than the mapping publication and none of
its set IDs is mapped yet.

## Extracted Runtime Knowledge

After source and release checks, DailyMed contributes 35,627 opt-in RxNorm aliases. On a 59-query
medication diagnostic set, canonical/ingredient ranking improved exact hit@1 from about 0.203 to
0.559 and MRR from about 0.336 to 0.576. This benchmark measures retrieval integration, not NER.

Structured relations such as ingredient-strength and product-route are retained with source
evidence. Narrative adverse-effect or treatment claims are not inferred from simple co-occurrence.

## Promotion Boundary

The official mapping-derived alias overlay is runtime opt-in and version-locked. SPL structured
annotations are silver training/evaluation data. Ambiguous exact strings, absent RxCUIs, newly
published unmapped labels, and narrative relations remain review-only.

Artifacts live under `outputs/mining/dailymed-daily-2026-07-17/` and
`outputs/mining/dailymed-rxnorm-2026-07-17/`; compiled opt-in knowledge is under
`outputs/mining/knowledge/dailymed-rxnorm-2026-07-17/`.
