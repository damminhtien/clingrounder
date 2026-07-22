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

## Processing Status And Full-Release Boundary

The checked-in evidence is intentionally split between work already executed and work only
prepared for the external data plane:

| Lane | Official release | Scale | Repository state |
| --- | --- | ---: | --- |
| Daily API slice | 17 Jul 2026 | 105 labels | processed and measured |
| Human prescription | 21 Jul 2026, 6 ZIP parts | 54,565 files | checksum-pinned; part 6 processed |
| Human prescription part 6 | 21 Jul 2026 | 3,700 SPL files, 1,586,216,624 source bytes | processed and structured-label mined |
| Human OTC | 21 Jul 2026, 11 ZIP parts | 88,764 files | checksum-pinned; not acquired |

Therefore, "DailyMed processed" currently means the 105-label daily tranche, prescription part 6,
and the complete official SPL-to-RxNorm mapping archive. It does **not** mean all SPL narrative
labels have been parsed. The current 17-part human plan covers 143,329 files. Its authoritative
file counts, modification date and MD5 values come from the
[DailyMed full-release manifest](https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm).
`configs/mining/dailymed-full-human-2026-07-21.yaml` records every part. Homeopathic, animal,
device and other non-human lanes are excluded from this plan and need separate quality policies.

The upstream bulk files are mutable URLs. On 22 July, acquisition correctly rejected the former
17 July part-6 MD5 after downloading the replacement bytes. The plan was updated only after the
official manifest reported the 21 July release. This fail-closed event is important: a filename is
not source identity, and a new upstream release must never silently reuse an old snapshot label.

The source publishes MD5 rather than SHA-256. Acquisition validates MD5 while streaming bytes
once into the content-addressed store; the store records SHA-256 as the durable internal identity.
Each part also declares `expected_spl_count`, so a replaced or partial archive fails before its
document manifest is promoted.

## Full-Scale Processing Contract

DailyMed individual packages contain one SPL XML member and optional images. Bulk parts may contain
direct XML members or one level of label ZIP packages. Parser revision `spl_xml:3` applies these
rules:

1. Never read a multi-gigabyte outer ZIP into Python memory.
2. Reject encrypted members, duplicate names, more than 100,000 members, XML over 64 MB, nested ZIP
   over 256 MB, or over 128 GB declared relevant payload per part.
3. Ignore images; parse XML in stable member-name order and retain one bounded XML payload at a
   time.
4. Hash each XML payload and use that digest in document identity. Identical labels repeated across
   release parts become one document family rather than two unrelated records.
5. Materialize stage/final manifests through temporary SQLite, not an in-memory document tuple.
   Duplicate content merges source artifact, archive member and release provenance; an ID collision
   with different text fails closed.
6. Preserve the rendered document as immutable offset text. Any normalization must create a child
   document.

This makes acquisition and parsing resumable on the external volume. Snapshot freezing still loads
the selected document manifest for split/agreement logic, so the full release should first be
materialized, profiled and filtered; only the curated subset should be frozen.

Artifact manifests identify objects as `medical-kg-cas://sha256/<digest>`. The configured local,
external-volume or object-store URI is deliberately absent from persisted records. Consequently,
the same manifest resolves against a different CAS root on another machine without exposing or
depending on the original workstation path.

## Prescription Part-6 Execution Evidence

Part 6 is an execution-sized pilot of the exact connector and parser used by the full 17-part plan;
it is not a hand-selected sample and it does not imply that the other parts were processed.

| Measure | Value |
| --- | ---: |
| Expected and validated SPL files | 3,700 |
| Source archive bytes | 1,586,216,624 |
| Source archive SHA-256 | `3c72512e43c1e298c53874bb1d0884dcd8a695c9234890fc769c5054f58bdeb6` |
| Parsed documents | 9,707 |
| Narrative SPL documents | 3,700 |
| Structured medication documents | 6,007 |
| Document manifest SHA-256 | `15030c63204c1b5183edc5774302fcecefdfd44431bf0906e2685e64b196ce47` |
| Structured annotations | 35,880 |
| Annotation manifest SHA-256 | `81947ff127b0332254cabc2288edbe1a352c8409b00ad21bf58fdbb4df458a67` |
| Duplicate annotation IDs | 0 |
| Schema/offset issues | 0 |

The 35,880 exact source labels contain 17,718 drug spans (6,007 product names, 6,007 generic names,
5,704 active ingredients), 5,704 strengths, 6,007 dosage forms and 6,451 routes. Labeling is
bounded-memory: documents are read in fixed batches and proposals are externally sorted and
deduplicated through temporary SQLite before an atomic JSONL write. A repeated run produced the
same annotation SHA in about 7.2 seconds with approximately 64.4 MiB maximum RSS on the current
machine.

All source, document and annotation manifests were audited for `/Users/` and `/home/` paths; none
remain. `run_result.json` declares `path_base: mining_plan_directory` and stores only paths relative
to that base. The portable release lock additionally binds the source plan, labeler config, source
archive bytes, processed output directory, dependency lock and implementation.

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

## What Has Not Yet Been Mined

- Narrative sections from part 6 and the daily slice have not yet been converted into indication,
  contraindication, warning, adverse-reaction or drug-interaction supervision. The remaining 16
  human release parts have not been acquired or parsed at all.
- Co-occurrence inside narrative prose is not a clinical relation. Section-aware extraction and a
  held-out relation benchmark are required before graph promotion.
- Part 6 has not yet been joined by SPL set/version to the official RxNorm mapping archive. The full
  release has not been coverage-audited against the July RxNorm release, so no claim is made about
  full NDC, ingredient, branded-drug or dose-form coverage.
- Homeopathic, animal and remainder archives are not silently mixed into human medication data.
- Images and OCR are not mined; they are ignored by the SPL parser.

The next DailyMed-specific step is to join part 6 by exact SPL set/version to the official mapping,
compile a leakage-safe drug mention inventory, and benchmark NER/retrieval on a source-held-out
medication set. Only after this pilot passes should the same streaming process expand to the other
16 human parts. Runtime dictionary promotion remains opt-in until that benchmark wins.

Artifacts live under `outputs/mining/dailymed-daily-2026-07-17/` and
`outputs/mining/dailymed-human-rx-part6-2026-07-21/`, while mappings live under
`outputs/mining/dailymed-rxnorm-2026-07-17/`; compiled aliases and graphs are under
`outputs/mining/knowledge/`.

## Reproduce

```bash
export MEDICAL_KG_ARTIFACT_STORE=/mnt/medical-kg/mining-artifacts
uv run medical-kg data run --plan configs/mining/dailymed-daily-2026-07-17.yaml

# Rebuild the executed full-release pilot. A cache hit and a fresh download produce
# identical final manifest hashes.
uv run medical-kg data run \
  --plan configs/mining/dailymed-human-rx-part6-2026-07-21.yaml
uv run medical-kg data label propose \
  --documents outputs/mining/dailymed-human-rx-part6-2026-07-21/documents.jsonl \
  --adapter medical_kg_nlp.mining.labelers.dailymed:create_dailymed_structured_labeler \
  --adapter-config configs/mining/labelers/dailymed-structured.yaml \
  --output outputs/mining/dailymed-human-rx-part6-2026-07-21/structured_annotations.jsonl \
  --batch-size 256
uv run medical-kg data release verify \
  --manifest data/releases/open-ner-retrieval-v1.lock.json --root .

# Requires the external >=250 GB data plane. Only the plan has been validated;
# this command has not been completed for all 17 parts.
uv run medical-kg data run --plan configs/mining/dailymed-full-human-2026-07-21.yaml

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
