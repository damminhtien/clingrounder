# DailyMed SPL And RxNorm Mappings

## Source Identity And Acquisition

Two distinct source contracts are used and must not be conflated:

1. `dailymed` acquires Structured Product Label (SPL) XML from the DailyMed API v2 catalog.
   The pinned slice is catalog snapshot `catalog-db-2026-07-17T195058-EST`, published
   `Jul 17, 2026 07:50:58PM EST`. The connector fails closed unless the catalog reports exactly
   105 labels for the requested publication date. Config:
   `configs/mining/dailymed-daily-2026-07-17.yaml`.
2. `dailymed_rxnorm_mappings` acquires the official `rxnorm_mappings.zip` release separately.
   The 17 July archive SHA-256 is
   `0d2797b35c31c0651e616d075b8b042591074e66a0c5955d8c4919e50ed9860c`.
   Config: `configs/mining/dailymed-rxnorm-2026-07-17.yaml`.

Both sources are `open_with_terms`, attribution-redistributable records in the registry. Raw files
are immutable content-addressed objects. The SPL slice contains 105 XML artifacts and 12,774,108
source bytes.

## SPL Parsing And Structured Labels

The SPL parser emits the complete narrative label and a compact
`structured_medication_record` for each product. Structured fields are rendered into the document
before annotation, so every product, generic name, ingredient, strength, dosage form and route has
an exact raw `[start,end)` span. Source codes are preserved as supplied; they are not converted to
RxNorm by the parser.

| Measure | Value |
| --- | ---: |
| SPL artifacts | 105 |
| narrative + structured documents | 239 |
| structured medication documents | 134 |
| structured annotations | 841 |
| source concept links | 559 |
| structured relation observations | 707 |
| exact duplicate groups | 1 |
| schema/offset issues | 0 |

The annotations contain 416 drug spans, 148 strengths, 143 routes and 134 dosage forms. Their 559
source links are 134 NDC product IDs, 148 UNII ingredient IDs and 277 NCI Thesaurus form/route IDs.
These identifiers provide source identity and structured supervision; they are not guessed RxNorm
codes.

Source fields are projected by
`medical_kg_nlp.mining.labelers.dailymed:create_dailymed_structured_labeler`. Relation endpoints are
then built from shared product indexes by `create_dailymed_structured_relation_labeler`:

| SPL relation | Observations | Meaning |
| --- | ---: | --- |
| `HAS_ACTIVE_INGREDIENT` | 148 | product to active ingredient |
| `HAS_STRENGTH` | 148 | ingredient to its rendered strength |
| `HAS_ROUTE` | 143 | product to administration route |
| `HAS_DOSAGE_FORM` | 134 | product to dosage form |
| `HAS_GENERIC_NAME` | 134 | product to generic-name span |

Narrative adverse-effect, indication and treatment claims are deliberately not inferred from
co-occurrence. The frozen structured snapshot
`dailymed-daily-2026-07-17-structured-silver-v2-7cbd7e2e3ebc7805` contains 218 grouped train and 21
development documents, 841 annotations and 707 relations.

## Official SPL-To-RxNorm Mapping

The mapping compiler streams the official archive into deduplicated JSONL and SQLite keyed by SPL
set/version, RxCUI, RXSTRING and RXTTY. It produced:

| Measure | Value |
| --- | ---: |
| source rows | 468,456 |
| exact duplicate source rows | 0 |
| versioned mappings | 150,925 |
| set/version pairs | 104,292 |
| unique RxCUIs | 24,312 |
| compiled mapping SHA-256 | `64189f5a092f4240629058bbeff2bced4d7405b3ca1d911b235f94e725cf73d5` |

The pinned 6 July 2026 RxNorm terminology contains 21,602 mapped RxCUIs. The audit rejects 2,710
release-mismatched codes and emits 36,076 alias proposals for review. The 105-label SPL slice is
newer than the mapping publication and none of its set IDs is mapped yet, so structured NDC/UNII
links and official SPL-RxNorm mappings remain separate evidence channels.

An exact text-only crosswalk confirms why the official mapping is needed: among 406 SPL inventory
entries, 171 match multiple exact RxNorm codes, 133 have no permitted policy, 88 are unmatched and
only 14 have one exact concept.

## Extracted Runtime Knowledge

The source-pinned alias compiler checks every proposal against the loaded RxNorm release, requires a
single target and applies alias-shape gates. It promoted 35,627 opt-in aliases targeting 19,100
RxNorm concepts; 445 shapes and four target conflicts were rejected. Overlay SHA-256:
`0b8ad8d80a46f18f7c51cbc94d866e90cbf2ef95ec37ba115f0e5acc4164e8f7`.

On a 59-query medication diagnostic set, canonical/ingredient ranking improved exact hit@1 from
approximately 0.203 to 0.559 and MRR from approximately 0.336 to 0.576. This measures retrieval
integration, not medication NER or end-to-end linking accuracy.

## Structured Evidence Graph

The 707 SPL relation observations were compiled separately from terminology into 445 source nodes,
689 deduplicated edges and 707 evidence rows. Nodes include 275 source-coded concepts and 170
unlinked rendered terms. The graph does not claim that NDC, UNII or NCI codes are canonical RxNorm
nodes.

| Relation benchmark | Indexed edges | Coverage | 8-worker QPS | p95 ms |
| --- | ---: | ---: | ---: | ---: |
| active ingredient | 148 | 100% | 2,741 | 6.63 |
| generic name | 134 | 100% | 2,979 | 3.93 |
| dosage form | 134 | 100% | 3,011 | 4.08 |
| route | 143 | 100% | 2,267 | 4.64 |
| strength | 130 deduplicated | 100% | 2,234 | 5.58 |

All repeated runs were deterministic. The persistent graph lives under
`outputs/mining/knowledge/dailymed-spl-structured-2026-07-17/`. It is suitable for source-evidence
lookup and hard-negative generation; it does not add narrative clinical relations.

## Promotion Boundary

- Runtime opt-in: the official mapping-derived RxNorm alias overlay, version locked to both source
  archive and loaded RxNorm release.
- Training/evaluation: exact SPL structured spans and structured relation endpoints.
- Source-evidence graph: product/ingredient/strength/form/route relations with SPL record evidence.
- Review only: ambiguous exact strings, missing RxCUIs, newly published unmapped labels and all
  narrative adverse-effect/indication claims.
- Never automatic: converting an NDC/UNII/NCI link into RxNorm without an official mapping row.

Artifacts live under `outputs/mining/dailymed-daily-2026-07-17/` and
`outputs/mining/dailymed-rxnorm-2026-07-17/`; compiled aliases and graphs are under
`outputs/mining/knowledge/`.

## Reproduce

```bash
export MEDICAL_KG_ARTIFACT_STORE=/Volumes/medical-kg-mining
uv run medical-kg data run --plan configs/mining/dailymed-daily-2026-07-17.yaml

uv run medical-kg data label propose \
  --documents outputs/mining/dailymed-daily-2026-07-17/documents.jsonl \
  --adapter medical_kg_nlp.mining.labelers.dailymed:create_dailymed_structured_labeler \
  --adapter-config configs/mining/labelers/dailymed-structured.yaml \
  --output outputs/mining/dailymed-daily-2026-07-17/structured_annotations.jsonl

uv run medical-kg data relation propose \
  --documents outputs/mining/dailymed-daily-2026-07-17/documents.jsonl \
  --annotations outputs/mining/dailymed-daily-2026-07-17/structured_annotations.jsonl \
  --adapter medical_kg_nlp.mining.labelers.dailymed:create_dailymed_structured_relation_labeler \
  --adapter-config configs/mining/labelers/dailymed-relations.yaml \
  --output outputs/mining/dailymed-daily-2026-07-17/structured_relations.jsonl

uv run medical-kg data run --plan configs/mining/dailymed-rxnorm-2026-07-17.yaml
uv run medical-kg data mapping compile-dailymed-rxnorm \
  --artifacts outputs/mining/dailymed-rxnorm-2026-07-17/artifacts.jsonl \
  --store "$MEDICAL_KG_ARTIFACT_STORE" \
  --output outputs/mining/dailymed-rxnorm-2026-07-17/compiled_mappings.jsonl \
  --index-output outputs/mining/dailymed-rxnorm-2026-07-17/dailymed_rxnorm.sqlite3 \
  --report-output outputs/mining/dailymed-rxnorm-2026-07-17/compilation_report.json

uv run medical-kg data knowledge compile-graph \
  --documents outputs/mining/dailymed-daily-2026-07-17/documents.jsonl \
  --annotations outputs/mining/dailymed-daily-2026-07-17/structured_annotations.jsonl \
  --relations outputs/mining/dailymed-daily-2026-07-17/structured_relations.jsonl \
  --accepted-layer silver --accepted-review-status proposed --relation-endpoints-only \
  --nodes-output outputs/mining/knowledge/dailymed-spl-structured-2026-07-17/nodes.jsonl \
  --edges-output outputs/mining/knowledge/dailymed-spl-structured-2026-07-17/edges.jsonl \
  --evidence-output outputs/mining/knowledge/dailymed-spl-structured-2026-07-17/evidence.jsonl \
  --report-output outputs/mining/knowledge/dailymed-spl-structured-2026-07-17/compilation_report.json
```
