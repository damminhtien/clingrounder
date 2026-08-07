# Data Mining

Cross-machine reconstruction and byte-level verification are documented in
[`docs/mining-reproducibility.md`](mining-reproducibility.md). Use a portable release lock for any
dataset, terminology overlay, benchmark, or model artifact that will be reused outside the machine
where it was built.

Source-specific processing evidence is indexed in
[`docs/mining-sources/README.md`](mining-sources/README.md). Use those dossiers to distinguish a
registered connector, an executed snapshot, model proposals, curated labels, and runtime-promoted
knowledge. This document remains the end-to-end command runbook.

`clingrounder.mining` builds reusable clinical NLP datasets without depending on a competition
schema. The lifecycle is:

```text
registered source -> immutable artifact -> parsed document -> proposals
-> review -> coverage analysis -> leakage-safe snapshot
```

## Safety Model

- Source bytes are content-addressed under `objects/sha256/<prefix>/<hash>`.
- Every source has a version, access class, license URL, redistribution rule, retention rule,
  connector, and parser in `data/sources/mining_registry.yaml`.
- Per-article sources such as PMC OA must provide the article license before fetch completes.
- `DUA`, credentialled, private, and quarantine records cannot use hosted labelers.
- `local_only` sources are rejected on remote object stores. DUA/private jobs additionally require
  `encrypted_at_rest: true`; this declaration must reflect verified volume encryption.
- `MinedDocument.text` is immutable. Normalization or translation creates a child document rather
  than rewriting offsets.
- Gold and challenge annotations must be human-accepted. Synthetic records can train models but
  cannot enter challenge snapshots.

Sources with unresolved dataset-level terms remain in `quarantine`. Registering a URL does not
grant permission to redistribute its content.

## Storage

Set an external local volume or S3-compatible URI:

```bash
export CLINGROUNDER_ARTIFACT_STORE=/Volumes/clingrounder-mining
# or: export CLINGROUNDER_ARTIFACT_STORE=s3://bucket/clingrounder-mining
```

JSONL remains the checkpoint and review exchange format. Frozen dataset tables use sharded Parquet;
DuckDB is an optional query catalog. Install these backends with:

```bash
uv sync --extra dev --extra data
```

## Commands

Validate policy before downloading anything:

```bash
uv run clingrounder-research data registry validate \
  --registry data/sources/mining_registry.yaml
```

Synchronize one explicit request. The parameter file contains connector inputs such as `pmc_ids`,
`set_ids`, `nct_ids`, local `paths`, or explicit `artifacts` with checksums:

```bash
uv run clingrounder-research data source sync \
  --source-id pmc_oa \
  --source-version 2026-07-18 \
  --parameters configs/mining/requests/pmc-cases.yaml \
  --store "$CLINGROUNDER_ARTIFACT_STORE" \
  --output outputs/mining/pmc/artifacts.jsonl
```

Parse, label, review, and inspect coverage independently:

```bash
uv run clingrounder-research data dataset build \
  --source-id pmc_oa \
  --artifacts outputs/mining/pmc/artifacts.jsonl \
  --store "$CLINGROUNDER_ARTIFACT_STORE" \
  --output outputs/mining/pmc/documents.jsonl

uv run clingrounder-research data dataset inspect \
  --documents outputs/mining/pmc/documents.jsonl \
  --annotations outputs/mining/pmc/proposals.jsonl \
  --output outputs/mining/pmc/source-profile.json \
  --strict

uv run clingrounder-research data label propose \
  --documents outputs/mining/pmc/documents.jsonl \
  --adapter my_local_plugin:create_labeler \
  --adapter-config configs/models/mining-labeler.yaml \
  --output outputs/mining/pmc/proposals.jsonl

uv run clingrounder-research data review export \
  --documents outputs/mining/pmc/documents.jsonl \
  --proposals outputs/mining/pmc/proposals.jsonl \
  --output outputs/mining/pmc/review.jsonl

uv run clingrounder-research data review quality \
  --documents outputs/mining/pmc/documents.jsonl \
  --proposals outputs/mining/pmc/reviewed.jsonl \
  --relations outputs/mining/pmc/relations.jsonl \
  --output outputs/mining/pmc/review-quality.json

uv run clingrounder-research data coverage report \
  --documents outputs/mining/pmc/documents.jsonl \
  --proposals outputs/mining/pmc/proposals.jsonl \
  --targets configs/mining/coverage_phase2.yaml \
  --snapshot-id phase2-working \
  --output outputs/mining/pmc/coverage.json
```

Hosted teachers are plugin-only and must add `--hosted`. The command checks every document before
invoking the plugin and rejects the whole batch if any source disallows hosted processing. Local
rule and Hugging Face adapters omit this flag.

Run a resumable campaign:

```bash
uv run clingrounder-research data run --plan configs/mining/phase2.yaml
```

Each source stage is keyed by source config, request config, connector revision, and parser revision.
Acquisition checkpoints each artifact. A completed stage is reused on later runs; changing the
source version, parser revision, checksum, or request creates a new stage directory.

## Review Priority

The fixed ranking formula is:

```text
0.30 coverage_gap + 0.25 model_disagreement + 0.20 novelty
+ 0.15 relation_density + 0.10 source_quality
```

Review queues should be grouped by normalized mention/context pattern, then processed for 1-2 hours
per day. Randomly double-review 10% and target agreement of at least 0.90 for spans/types, 0.85 for
assertions, and 0.80 for relations.

## Snapshot Rules

`SnapshotBuilder` unions exact/near duplicate, patient, case, article, template, and concept-family
groups before assigning a split. If any member is held out, the entire connected component becomes
challenge. This prevents a duplicated note or article template from appearing in both train and
evaluation.

Freeze requires an explicit version and timestamp. It rejects invalid offsets, relation endpoints,
unreviewed challenge annotations, synthetic challenge documents, and training snapshots above the
configured synthetic fraction. Existing snapshot directories are immutable; an identical manifest
is an idempotent no-op.

## Current Connector Coverage

Implemented acquisition adapters cover explicit local archives, static URLs, PMC OA, DailyMed, and
ClinicalTrials.gov. Implemented document parsers cover JATS, SPL, ClinicalTrials JSON, FHIR Bundle,
BioC JSON, CodiEsp ZIP, BRAT ZIP, and plain text. LOINC/HPO/Mondo are terminology inputs and should
use `parse_documents: false`.

Source-specific observed counts, fingerprints, failure modes, and promotion decisions live in
[`docs/mining-sources/`](mining-sources/README.md). A connector listed here is implementation
coverage; it is not evidence that a source has been acquired or promoted. The machine-readable
authority for processing state is `data/sources/processing_status.yaml`.

VietBioNER is pinned at Git commit `19ba70a5947d1be72906d407c860b1666b9337e9` under CC BY 4.0.
`configs/mining/vietbioner.yaml` acquires the checksum-pinned archive, preserves each annotator as a
separate document, groups exact duplicate text into one split, and imports source labels as silver
proposals through `clingrounder.mining.labelers.brat`. The broad internal label mapping is an
import convention, not adjudicated clinical gold.

VietMed-NER use for training and inference was confirmed by the data owner on 2026-07-27. The
repository pins the dataset/model revisions and source checksums, projects only text/BIO columns,
and preserves the official train/validation/test split. Redistribution remains prohibited because
the public model card does not state a reusable SPDX license. Its 18-type spoken-medical taxonomy
is retained as source evidence; broad labels such as `DISEASESYMTOM` are never silently narrowed
into a competition label. See
[`docs/mining-sources/vietmed-ner.md`](mining-sources/vietmed-ner.md).

The current VietBioNER snapshot can be reproduced without an implicit download or parser choice:

```bash
uv run clingrounder-research data run --plan configs/mining/vietbioner.yaml

uv run clingrounder-research data label propose \
  --documents outputs/mining/vietbioner-19ba70a/documents.jsonl \
  --adapter clingrounder.mining.labelers.brat:create_brat_archive_labeler \
  --adapter-config configs/mining/labelers/vietbioner.yaml \
  --output outputs/mining/vietbioner-19ba70a/source_annotations.jsonl

uv run clingrounder-research data dataset inspect \
  --documents outputs/mining/vietbioner-19ba70a/documents.jsonl \
  --annotations outputs/mining/vietbioner-19ba70a/source_annotations.jsonl \
  --output outputs/mining/vietbioner-19ba70a/source_profile.json \
  --strict

uv run clingrounder-research data dataset reconcile-duplicates \
  --documents outputs/mining/vietbioner-19ba70a/documents.jsonl \
  --annotations outputs/mining/vietbioner-19ba70a/source_annotations.jsonl \
  --documents-output outputs/mining/vietbioner-19ba70a/reconciled/documents.jsonl \
  --annotations-output outputs/mining/vietbioner-19ba70a/reconciled/training_annotations.jsonl \
  --review-output outputs/mining/vietbioner-19ba70a/reconciled/review_annotations.jsonl \
  --mapping-output outputs/mining/vietbioner-19ba70a/reconciled/document_map.jsonl \
  --report-output outputs/mining/vietbioner-19ba70a/reconciled/agreement_report.json \
  --labeler-id vietbioner-exact-duplicate-consensus:v1

uv run clingrounder-research data review export \
  --documents outputs/mining/vietbioner-19ba70a/reconciled/documents.jsonl \
  --proposals outputs/mining/vietbioner-19ba70a/reconciled/review_annotations.jsonl \
  --output outputs/mining/vietbioner-19ba70a/reconciled/review_queue.jsonl

uv run clingrounder-research data lexicon build \
  --documents outputs/mining/vietbioner-19ba70a/reconciled/documents.jsonl \
  --annotations outputs/mining/vietbioner-19ba70a/reconciled/training_annotations.jsonl \
  --output outputs/mining/vietbioner-19ba70a/reconciled/mention_inventory.jsonl \
  --conflicts-output outputs/mining/vietbioner-19ba70a/reconciled/mention_conflicts.jsonl \
  --report-output outputs/mining/vietbioner-19ba70a/reconciled/mention_inventory_report.json

uv run clingrounder-research data lexicon crosswalk \
  --inventory outputs/mining/vietbioner-19ba70a/reconciled/mention_inventory.jsonl \
  --index .cache/clingrounder/terminology/terminology-0598a6a288ef81ea932f.sqlite3 \
  --source data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl \
  --source data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl \
  --policy configs/mining/crosswalk/vietbioner.yaml \
  --output outputs/mining/vietbioner-19ba70a/reconciled/terminology_crosswalk.jsonl \
  --report-output outputs/mining/vietbioner-19ba70a/reconciled/terminology_crosswalk_report.json \
  --workers 4

uv run clingrounder-research data snapshot freeze \
  --documents outputs/mining/vietbioner-19ba70a/reconciled/documents.jsonl \
  --annotations outputs/mining/vietbioner-19ba70a/reconciled/training_annotations.jsonl \
  --artifacts outputs/mining/vietbioner-19ba70a/artifacts.jsonl \
  --source-fingerprint 3dbc1f703f3b7d8ca080ad9bfb324596b10a1cf7ac82456c04bf594062a1f01d \
  --source-fingerprint fe22b5358e7b974a88a6c75bcc3dc8f876255e3bbc45aff55d9765a59295ab2d \
  --version vietbioner-19ba70a-reconciled-silver-v4 \
  --created-at 2026-07-20T04:49:09Z \
  --development-fraction 0.3 \
  --output-dir outputs/mining/snapshots/vietbioner-19ba70a-reconciled-silver-v4 \
  --skip-agreement-gate
```

The expected import has 70 annotator documents, 3,574 source annotations, seven exact-duplicate
text groups, and no offset mismatch. Exact reconciliation produces 63 unique documents, 3,109
silver training annotations, and 164 disagreement hypotheses in the review queue. Pairwise exact
micro-Jaccard on duplicate annotations is about 0.647, so union labels must not be promoted to gold.

The mention inventory contains unlinked terminology hypotheses only. It must not be concatenated
into runtime dictionaries: VietBioNER merges symptoms and diseases into one source type and is
heavily concentrated on tuberculosis and HIV. Use the conflict report and pinned terminology
repositories to review aliases, then create a separate versioned mapping overlay.

The pinned TT06/RxNorm exact crosswalk currently resolves 8 of 768 inventory entries, covering 22
of 3,109 source occurrences. Another 248 disease/symptom hypotheses are unmatched and 512 entries
are intentionally skipped because no source-label policy permits a terminology query. Every exact
hit remains `review_required`: for example, VietBioNER's broad finding label does not distinguish
the symptom `ho` from a diagnosis even though TT06 has an exact `R05` label. The crosswalk validates
the SQLite source fingerprint, filters entity type and code system before lookup, and never performs
fuzzy matching or mutates a runtime dictionary.

## CodiEsp Source Snapshot

CodiEsp v1.4 is pinned to Zenodo record `3837305`, CC BY 4.0, with source SHA-256
`52b290233906a2eb589ac7b1d9429adeac88f6a9cae8b0d3c180504afbe61688`. The importer reads only
Spanish `text_files` members; machine-translated `text_files_en` members are deliberately excluded.
Run acquisition and source-label import with:

```bash
uv run clingrounder-research data run --plan configs/mining/codiesp.yaml

uv run clingrounder-research data label propose \
  --documents outputs/mining/codiesp-zenodo-3837305/documents.jsonl \
  --adapter clingrounder.mining.labelers.codiesp:create_codiesp_archive_labeler \
  --adapter-config configs/mining/labelers/codiesp.yaml \
  --output outputs/mining/codiesp-zenodo-3837305/source_annotations.jsonl \
  --batch-size 256

uv run clingrounder-research data dataset inspect \
  --documents outputs/mining/codiesp-zenodo-3837305/documents.jsonl \
  --annotations outputs/mining/codiesp-zenodo-3837305/source_annotations.jsonl \
  --output outputs/mining/codiesp-zenodo-3837305/source_profile.json \
  --strict

uv run clingrounder-research data dataset curate-annotations \
  --annotations outputs/mining/codiesp-zenodo-3837305/source_annotations.jsonl \
  --policy configs/mining/curation/codiesp-contiguous-ner.yaml \
  --accepted-output outputs/mining/codiesp-zenodo-3837305/contiguous_training_annotations.jsonl \
  --rejected-output outputs/mining/codiesp-zenodo-3837305/noncontiguous_review_annotations.jsonl \
  --report-output outputs/mining/codiesp-zenodo-3837305/curation_report.json

uv run clingrounder-research data snapshot freeze \
  --documents outputs/mining/codiesp-zenodo-3837305/documents.jsonl \
  --annotations outputs/mining/codiesp-zenodo-3837305/contiguous_training_annotations.jsonl \
  --artifacts outputs/mining/codiesp-zenodo-3837305/artifacts.jsonl \
  --version codiesp-zenodo-3837305-contiguous-silver-v1 \
  --created-at 2026-07-18T17:15:00+07:00 \
  --output-dir outputs/mining/snapshots/codiesp-zenodo-3837305-contiguous-silver-v1 \
  --development-fraction 0 \
  --hash-salt codiesp-zenodo-3837305-contiguous-silver-v1 \
  --skip-agreement-gate
```

The reproducible import contains 3,751 Spanish documents and 18,435 CodiEsp-X source annotations:
14,305 diagnostic/finding records and 4,130 procedure records. All concept links retain the source
ICD-10-CM or ICD-10-PCS provenance. Eleven source rows have a text-reference or zero-length-segment
issue and are marked `needs_review`; no internal raw offset is invalid.

CodiEsp contains 3,707 discontinuous annotations, including 135 whose segments are listed in phrase
order rather than source-offset order. Source JSONL preserves their original segment geometry and a
raw envelope. The contiguous NER view keeps 14,726 exact, issue-free annotations and routes 3,709
records to a separate review/linking file. Do not train a contiguous token classifier directly on
the source envelope file.

The frozen snapshot
`codiesp-zenodo-3837305-contiguous-silver-v1-4b75600fd6d5f751` has manifest SHA-256
`7b8e678e7f83cec058e3cf7299f1cb43e648fbcae625cc5837088e9812983a4f`. It deliberately assigns all
documents to the generic `train` partition: the authoritative CodiEsp `train`, `dev`, `test`, and
`background` split remains in document metadata and must be selected by a CodiEsp evaluation
adapter. This prevents the generic hash splitter from silently redefining the external benchmark.

## PMC Rare-Case Raw Snapshot

`configs/mining/pmc-rare-cases-2026-07-18.yaml` pins ten PMCID values selected from an open-access
rare-case query. Discovery still calls the OA service for each article's current license, while the
article text is checkpointed from the official BioC JSON endpoint. This avoids treating unstable
FTP package links as a reproducible transport and preserves BioC absolute passage offsets.

```bash
export CLINGROUNDER_ARTIFACT_STORE=/Volumes/clingrounder-mining
uv run clingrounder-research data run --plan configs/mining/pmc-rare-cases-2026-07-18.yaml

uv run clingrounder-research data dataset inspect \
  --documents outputs/mining/pmc-rare-cases-2026-07-18/documents.jsonl \
  --output outputs/mining/pmc-rare-cases-2026-07-18/source_profile.json \
  --strict
```

The slice contains ten unique English articles and 232,630 source characters with zero offset or
duplicate issue. Four articles are CC BY, three CC BY-NC, and three CC BY-NC-ND. Snapshot
`pmc-rare-cases-2026-07-18-raw-v2-84000050e3cbf220` has manifest SHA-256
`1c8f50fbb2ae23782797d19f8fee64a9ecbd879e48952920bc0fd855a4cbedd3` and is deliberately marked
non-redistributable because the mixed-source snapshot contains non-commercial and no-derivatives
licenses. Keep raw articles separate from model proposals and derived knowledge overlays.

## DailyMed Structured Medication Snapshot

DailyMed catalog mining uses a fail-closed database publication date and a bounded page range. The
first real slice is the 105 SPL records published on 17 July 2026 from catalog database snapshot
`Jul 17, 2026 07:50:58PM EST`. Set `CLINGROUNDER_ARTIFACT_STORE` to an external volume or S3 prefix;
the full human-label releases are too large for the repository workspace.

```bash
export CLINGROUNDER_ARTIFACT_STORE=/Volumes/clingrounder-mining
uv run clingrounder-research data run --plan configs/mining/dailymed-daily-2026-07-17.yaml

uv run clingrounder-research data label propose \
  --documents outputs/mining/dailymed-daily-2026-07-17/documents.jsonl \
  --adapter clingrounder.mining.labelers.dailymed:create_dailymed_structured_labeler \
  --adapter-config configs/mining/labelers/dailymed-structured.yaml \
  --output outputs/mining/dailymed-daily-2026-07-17/structured_annotations.jsonl

uv run clingrounder-research data relation propose \
  --documents outputs/mining/dailymed-daily-2026-07-17/documents.jsonl \
  --annotations outputs/mining/dailymed-daily-2026-07-17/structured_annotations.jsonl \
  --adapter clingrounder.mining.labelers.dailymed:create_dailymed_structured_relation_labeler \
  --adapter-config configs/mining/labelers/dailymed-relations.yaml \
  --output outputs/mining/dailymed-daily-2026-07-17/structured_relations.jsonl

uv run clingrounder-research data snapshot freeze \
  --documents outputs/mining/dailymed-daily-2026-07-17/documents.jsonl \
  --annotations outputs/mining/dailymed-daily-2026-07-17/structured_annotations.jsonl \
  --relations outputs/mining/dailymed-daily-2026-07-17/structured_relations.jsonl \
  --artifacts outputs/mining/dailymed-daily-2026-07-17/artifacts.jsonl \
  --version dailymed-daily-2026-07-17-structured-silver-v2 \
  --created-at 2026-07-18T19:00:00+07:00 \
  --output-dir outputs/mining/snapshots/dailymed-daily-2026-07-17-structured-silver-v2 \
  --skip-agreement-gate
```

The structured view contains 239 documents, 841 source annotations, 559 source concept links and
707 medication relations. It preserves product, generic name, active ingredient, strength, dosage
form, route, NDC, UNII and NCI identifiers with exact rendered offsets. Snapshot
`dailymed-daily-2026-07-17-structured-silver-v2-7cbd7e2e3ebc7805` has manifest SHA-256
`70bbc54c43f5e8b5cb60f05b90802abe54bfe1bbfecf814b1ea2c9da406b8318`.

The generic exact-name crosswalk in `configs/mining/crosswalk/dailymed-rxnorm.yaml` is diagnostic,
not authoritative. On this slice it produces 406 mention inventory entries and 55 semantic
conflicts: 171 entries have multiple exact RxNorm codes, 133 are skipped by policy, 14 are unique,
and 88 are unmatched. This confirms that exact drug strings alone cannot replace the official
versioned SPL mapping or structured drug parsing.

DailyMed's official SPL-to-RxNorm mapping is acquired separately because it is a versioned source
crosswalk, not inferred text matching. Compile and audit the checksum-pinned release with:

```bash
uv run clingrounder-research data run --plan configs/mining/dailymed-rxnorm-2026-07-17.yaml

uv run clingrounder-research data mapping compile-dailymed-rxnorm \
  --artifacts outputs/mining/dailymed-rxnorm-2026-07-17/artifacts.jsonl \
  --store "$CLINGROUNDER_ARTIFACT_STORE" \
  --output outputs/mining/dailymed-rxnorm-2026-07-17/compiled_mappings.jsonl \
  --index-output outputs/mining/dailymed-rxnorm-2026-07-17/dailymed_rxnorm.sqlite3 \
  --report-output outputs/mining/dailymed-rxnorm-2026-07-17/compilation_report.json

uv run clingrounder-research data mapping audit-dailymed-rxnorm \
  --index outputs/mining/dailymed-rxnorm-2026-07-17/dailymed_rxnorm.sqlite3 \
  --terminology-index .cache/clingrounder/terminology/terminology-0598a6a288ef81ea932f.sqlite3 \
  --source data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl \
  --source data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl \
  --proposals-output outputs/mining/dailymed-rxnorm-2026-07-17/review_alias_proposals.jsonl \
  --report-output outputs/mining/dailymed-rxnorm-2026-07-17/audit_report.json
```

The 17 July mapping archive has SHA-256
`0d2797b35c31c0651e616d075b8b042591074e66a0c5955d8c4919e50ed9860c`, 468,456 source rows,
150,925 versioned mappings and 24,312 RxCUIs. Against the pinned 6 July RxNorm release, 21,602
RxCUIs exist and 2,710 are rejected as release mismatches. The audit emits 36,076 missing alias
pairs as `review_required`; it never mutates the canonical terminology automatically. None of the
105 newly published SPL set IDs is present in this mapping release yet, so those daily records retain
their NDC/UNII/NCI links until an authoritative RxNorm mapping catches up.

## Knowledge Promotion Results

Mined knowledge is promoted independently for recognition and normalization. A source can improve
retrieval without being safe to scan over every clinical note. The active full-terminology config
therefore keeps a compact recognition dictionary and loads reviewed aliases only into the filtered
SQLite normalization repository.

DailyMed contributes 35,627 source-pinned RxNorm aliases. Ranking canonical and ingredient TTYs
ahead of product variants raised exact hit@1 on the 59-query medication diagnostic set from about
0.203 to 0.559 and MRR from about 0.336 to 0.576. CodiEsp contributes 642 Spanish aliases linked to
418 TT06 concepts after rejecting unknown ICD-10-CM/PCS codes and conflicting aliases. Querying
those same 642 overlay rows produces 642/642 exact hits; this is an index integration check, not a
generalization benchmark. The 59-query RxNorm metrics were byte-for-byte unchanged after adding
CodiEsp, which verifies type/code-system isolation.

Reproduce the CodiEsp gate with:

```bash
uv run clingrounder terminology query-set \
  --alias-overlay outputs/mining/knowledge/codiesp-icd10-2026-07-18/alias_overlay.jsonl \
  --output outputs/mining/knowledge/codiesp-icd10-2026-07-18/benchmark/codiesp_tt06_queries.jsonl \
  --manifest-output outputs/mining/knowledge/codiesp-icd10-2026-07-18/benchmark/query_manifest.json

uv run clingrounder terminology build \
  --source data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl \
  --source data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl \
  --alias-overlay outputs/mining/knowledge/dailymed-rxnorm-2026-07-17/alias_overlay.jsonl \
  --alias-overlay outputs/mining/knowledge/codiesp-icd10-2026-07-18/alias_overlay.jsonl \
  --manifest-output outputs/mining/knowledge/codiesp-icd10-2026-07-18/enriched_terminology_manifest.json
```

The resulting index contains 88,837 concepts and 188,488 aliases. Its content-addressed local path
is `.cache/clingrounder/terminology/terminology-a2d5a19e83fbc9e1a305.sqlite3`; runtime composition is
recorded in `configs/pipeline/full_terminology.yaml`. The cache path is derived output, not a source
artifact, and must be rebuilt from the pinned manifests on another machine.

### Leakage-Safe CodiEsp Retrieval Gate

CodiEsp's official `train`, `dev`, and `test` metadata now controls alias promotion. Only train
documents can produce the 393 runtime aliases in
`outputs/mining/knowledge/codiesp-icd10-split-2026-07-18/train/alias_overlay.jsonl`; dev and test
produce query-only artifacts. Queries are sliced by whether the exact alias and target code were
seen in train.

| Query set | Queries | Unknown to TT06 | Exact hit@1 | FTS recall@20 | Unseen-alias recall@20 |
|---|---:|---:|---:|---:|---:|
| Dev, base | 1,443 | 400 | 0.35% | 8.32% | 7.00% |
| Dev, train aliases | 1,443 | 400 | 18.09% | 28.55% | 13.16% |
| Test, base | 1,403 | 376 | 0.36% | 8.13% | 6.95% |
| Test, train aliases | 1,403 | 376 | 19.32% | 30.65% | 14.42% |

The base rows above include the bounded partial-token FTS fallback. Compared with strict phrase/AND
search, train-enriched recall@20 rose from 20.44% to 28.55% on dev and from 22.17% to 30.65% on
test. P95 lexical latency increased from about 5.3 ms to 15-17 ms. Phrase and AND matches retain
priority; OR matches only fill an incomplete top-k and remain type/code-system filtered.

The 400 dev and 376 test expected codes absent from TT06 stay in the report as impossible targets.
They are not imported from ICD-10-CM by code shape and are not silently converted to parent codes.
The full released-source CodiEsp overlay remains available for production terminology coverage,
while model/search evaluation must use the train-only index
`.cache/clingrounder/terminology/terminology-codiesp-train-2026-07-18.sqlite3`.

Build a held-out query set with:

```bash
uv run clingrounder terminology query-set \
  --linked-proposal outputs/mining/knowledge/codiesp-icd10-split-2026-07-18/dev/proposals.jsonl \
  --reference-alias-overlay outputs/mining/knowledge/codiesp-icd10-split-2026-07-18/train/alias_overlay.jsonl \
  --output outputs/mining/knowledge/codiesp-icd10-split-2026-07-18/dev/queries.jsonl \
  --manifest-output outputs/mining/knowledge/codiesp-icd10-split-2026-07-18/dev/query_manifest.json
```

VietBioNER follows a different promotion decision. The materialized v4 split's train-only compiler
accepted 87 recognition concepts. On 12 development documents, the dictionary reached precision
0.540, recall 0.505, and F1 0.522; procedure F1 was only 0.107 and 78 of 166 false positives were
boundary overlaps. The historical split reached F1 0.612, but neither result is strong enough for
runtime promotion. The full source audit and exact commands are in
[`docs/mining-sources/vietbioner.md`](mining-sources/vietbioner.md).

### Phase 1 Reviewed Recognition Mining

The completed Phase 1 manual gold corpus is also a useful small, high-quality mining source. The
benchmark adapter namespaces the documents, marks them `local_private`, validates raw offsets, and
uses only the frozen `train` split (76 documents, 2,112 annotations) to build the inventory. The 24
holdout documents (665 annotations) are read only by the recognition benchmark. Strict aliases from
`data/manual_gold/compiled/phase1_annotation_policy.yaml` act as a source-aware review allowlist;
unsupported or ambiguous inventory rows remain rejected.

The run on 18 July 2026 produced 1,102 train inventory rows and promoted 31 code-free recognition
concepts: 4 diagnoses, 1 drug, 9 lab-test names, and 17 symptoms. Recognition-only exact F1 on the
holdout rose from `0.5631` to `0.5832` (`+14` true positives and `-5` false positives). The end-to-end
Phase 1 manual-gold replay also improved on the same pipeline composition:

| Split | Score | Text | Assertions | Candidates |
|---|---:|---:|---:|---:|
| All, baseline | 52.8175 | 0.4973 | 0.5238 | 0.5546 |
| All, mined recognition | 55.8432 | 0.5320 | 0.5503 | 0.5844 |
| Holdout, baseline | 52.9403 | 0.5065 | 0.5349 | 0.5425 |
| Holdout, mined recognition | 54.1591 | 0.5228 | 0.5440 | 0.5538 |

These are local manual-gold metrics, not a public competition score. The recognition overlay stays
opt-in until a public probe confirms the hidden-label convention. The content-addressed artifact and
all compiler decisions are under:
`outputs/mining/knowledge/phase1-recognition-115886bf9d22/`.

Reproduce the mining gate:

```bash
uv run python scripts/benchmarks/phase1/mine_phase1_recognition_knowledge.py
```

The generated `pipeline_profile_fragment.yaml` can be merged into a pipeline profile. The checked-in
snapshot profile used for the end-to-end replay is
`configs/benchmarks/phase1/pipeline/full_terminology_manual_gold_recognition.yaml`; it intentionally points to the
hashed artifact rather than silently rebuilding or replacing a dictionary at startup.

For model NER, export the reconciled raw spans without changing offsets or leaking the development
split:

```bash
uv run clingrounder-research data dataset export-spans \
  --documents outputs/mining/vietbioner-19ba70a/reconciled/documents.jsonl \
  --annotations outputs/mining/vietbioner-19ba70a/reconciled/model_ner_annotations.jsonl \
  --split-manifest outputs/mining/snapshots/vietbioner-19ba70a-reconciled-silver-v4/manifest.json \
  --output outputs/mining/model_datasets/vietbioner-19ba70a-reconciled-silver-v4/spans.jsonl \
  --manifest-output outputs/mining/model_datasets/vietbioner-19ba70a-reconciled-silver-v4/manifest.json \
  --entity-type FINDING \
  --entity-type PROCEDURE \
  --max-characters 1200
```

### Model-balanced NER snapshot

The source-held-out snapshot above is intentionally a challenge artifact. It is not a suitable
training validation split when an entire source contributes a label that never occurs in training:
the first model snapshot had `FINDING` only in development. A second, immutable model-balanced
snapshot therefore hashes all documents into train/development with a 10% development fraction.
The two artifacts are separate by design:

Rebuild the snapshot and span artifact deterministically with:

```bash
clingrounder-research data snapshot freeze \
  --documents outputs/mining/fused/open-corpus-v1-39106c1cc9d0/documents.jsonl \
  --annotations outputs/mining/fused/open-corpus-v1-39106c1cc9d0/annotations.jsonl \
  --relations outputs/mining/fused/open-corpus-v1-39106c1cc9d0/relations.jsonl \
  --source-fingerprint 39106c1cc9d04e545d4728657cecb5e9702a6c95e548fe4c47f887085e66543d \
  --version open-corpus-v1-balanced-2026-07-18 \
  --created-at 2026-07-18T00:00:00+07:00 \
  --output-dir outputs/mining/snapshots/open-corpus-v1-balanced-2026-07-18 \
  --development-fraction 0.1 \
  --hash-salt open-corpus-v1-balanced-2026-07-18 \
  --skip-agreement-gate
```

Then run the existing `data dataset export-spans` command against the new snapshot manifest and
the fused `curated_annotations.jsonl` input. The source-heldout snapshot must remain untouched.

| artifact | chunks | entities | label coverage |
| --- | ---: | ---: | --- |
| source-held-out | 3,745 | 13,107 | development-only `FINDING` |
| model-balanced | 3,745 | 13,107 | both splits: `DISEASE`, `DRUG`, `FINDING` |

The model-balanced chunk distribution is `train/development`: CodiEsp `2,770/277`, DailyMed
`410/55`, PMC OA `19/2`, and VietBioNER `179/33`. Hashing is performed at document/group level,
so chunks from one document cannot cross the split boundary.

The current model-balanced fingerprints are:

```text
snapshot manifest: ecc593272fb5c0f2dd089976b9c93eb276434b572829491ffd0913af5d97f3c0
span dataset:      aa6cafa2efd1f4f68f927067d41348b16e22da19f5f57f7375b2c5e62ab8aff6
```

Validate it before a model run:

```bash
clingrounder-research model validate-token-dataset \
  --dataset outputs/mining/model-datasets/open-corpus-v1-balanced-2026-07-18/spans.jsonl \
  --dataset-manifest outputs/mining/model-datasets/open-corpus-v1-balanced-2026-07-18/manifest.json
```

Training requires the local `ml` extra and a cached fast tokenizer/model. The command is offline by
construction (`local_files_only=true`) and writes a model fingerprint plus metrics manifest:

```bash
clingrounder-research model train-token-classifier \
  --dataset outputs/mining/model-datasets/open-corpus-v1-balanced-2026-07-18/spans.jsonl \
  --dataset-manifest outputs/mining/model-datasets/open-corpus-v1-balanced-2026-07-18/manifest.json \
  --model-id "$LOCAL_MODEL_ID" \
  --revision "$LOCAL_MODEL_REVISION" \
  --output-dir outputs/models/open-corpus-ner-<run-id>
```

Use `--internal-validation-fraction 0.1` to select a checkpoint from a deterministic document
holdout inside `train` without consuming the source-heldout challenge. Do not use the original
source-heldout `development` split for training or merge its labels into the train vocabulary.
The trainer rejects unseen evaluation labels, tokenizer boundary drift, changed dataset hashes,
and missing model revisions before importing Torch.

### Full-type NER view for later phases

The Phase 1-compatible view deliberately excludes `PROCEDURE`. For later phases, a separate
harmonized view was curated from `harmonized_annotations.jsonl` with the same overlap and offset
gates, adding 2,757 accepted procedure spans:

```text
accepted spans: 15,847
chunks:         3,823 (3,450 train / 373 development)
dataset SHA:    90e1b05cac683046a88e58ab26bb090cd58c7d54077a6845f9fa49ccb59c2ca8
labels:         DISEASE, DRUG, FINDING, PROCEDURE
```

Artifact paths:
`outputs/mining/model-datasets/open-corpus-v1-full-ner-harmonized-2026-07-18/`.
The `LAB_TEST`, `LAB_RESULT`, and `SYMPTOM` proposals currently remain bronze-only in the fused
corpus and are not promoted into this quality-gated view. They must use a separately marked weak-
supervision artifact or human review before becoming evaluation labels; this prevents proposed
PMC annotations from being mistaken for clinical gold.

### Source-pinned co-occurrence evidence

The first relation-mining pass uses only the 500 official CodiEsp train documents. It mines
concept-linked entities in the same sentence and emits the literal symmetric relation
`CO_OCCURS_WITH`; it never promotes a co-occurrence into `TREATS`, `CAUSES`, or `HAS_SYMPTOM`.
Official `corpus_split=train` filtering is part of the policy, independent of later balanced model
splits.

```bash
uv run clingrounder-research data relation mine-cooccurrence \
  --documents outputs/mining/fused/open-corpus-v1-39106c1cc9d0/documents.jsonl \
  --annotations outputs/mining/fused/open-corpus-v1-39106c1cc9d0/harmonized_annotations.jsonl \
  --policy configs/mining/relations/codiesp-train-cooccurrence.yaml \
  --output outputs/mining/relations/codiesp-train-cooccurrence-2026-07-19/relations.jsonl \
  --report-output outputs/mining/relations/codiesp-train-cooccurrence-2026-07-19/report.json
```

This produced 225 semantic pairs and 721 source occurrences from 5,142 eligible annotations. The
relation artifact SHA-256 is
`c80ce0cfd09a2a0c31b4da53889d08327388abe43c5be9d5d33096be5dd4bc50`. Each pair is supported by at
least two train documents; dense sentences, overlapping spans, large character gaps, unlinked
annotations, and dev/test/background records are counted and skipped.

The benchmark graph uses `--relation-endpoints-only` to prevent unrelated annotation aliases from
leaking into the graph and `--canonical-concepts-only` to reject codes absent from TT06. It retained
180 disease-disease edges with 604 evidence occurrences. Another 117 relation occurrences, mainly
ICD-10-PCS procedures, were rejected because no pinned PCS terminology was loaded. They must not be
reintroduced by code shape.

```bash
uv run clingrounder kg benchmark-relations \
  --index .cache/clingrounder/knowledge-graph/knowledge-graph-d2b7d076728655e78e13.sqlite3 \
  --edges outputs/mining/knowledge/codiesp-train-cooccurrence-2026-07-19/edges.jsonl \
  --relation-type CO_OCCURS_WITH \
  --workers 8 \
  --repeats 5 \
  --output outputs/mining/knowledge/codiesp-train-cooccurrence-2026-07-19/relation_benchmark.json
```

The immutable SQLite graph reached 100% edge traversal coverage, deterministic results across eight
workers, about 1,228 queries/second, and p95 latency of 18.69 ms. This is an index consistency and
scaling result, not a clinical relation-quality score. The graph remains an experiment and is not
enabled in the default pipeline until a relation or reranking benchmark shows downstream gain.

## Compiled Knowledge Graph

The terminology and mined relation layers are also compiled into a provenance-bearing graph. The
compiler keeps JSONL as the source of truth and promotes only exact, typed links. In particular,
an RxNorm `ingredient` field is linked only when it resolves to one unique `IN`, `PIN`, or `MIN`
concept. Unresolved or ambiguous fields are counted and skipped; they are never converted into a
guessed code or edge.

```bash
uv run clingrounder-research data knowledge compile-graph \
  --terminology-source data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl \
  --terminology-source data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl \
  --alias-overlay data/dictionaries/vietnamese_medical_alias.jsonl \
  --alias-overlay outputs/mining/knowledge/dailymed-rxnorm-2026-07-17/alias_overlay.jsonl \
  --alias-overlay outputs/mining/knowledge/codiesp-icd10-2026-07-18/alias_overlay.jsonl \
  --documents outputs/mining/dailymed-daily-2026-07-17/documents.jsonl \
  --annotations outputs/mining/dailymed-daily-2026-07-17/structured_annotations.jsonl \
  --relations outputs/mining/dailymed-daily-2026-07-17/structured_relations.jsonl \
  --nodes-output outputs/mining/knowledge/full-graph-rxnorm-2026-07-18/nodes.jsonl \
  --edges-output outputs/mining/knowledge/full-graph-rxnorm-2026-07-18/edges.jsonl \
  --evidence-output outputs/mining/knowledge/full-graph-rxnorm-2026-07-18/evidence.jsonl \
  --report-output outputs/mining/knowledge/full-graph-rxnorm-2026-07-18/compilation_report.json

uv run clingrounder kg build \
  --nodes outputs/mining/knowledge/full-graph-rxnorm-2026-07-18/nodes.jsonl \
  --edges outputs/mining/knowledge/full-graph-rxnorm-2026-07-18/edges.jsonl \
  --evidence outputs/mining/knowledge/full-graph-rxnorm-2026-07-18/evidence.jsonl \
  --manifest-output outputs/mining/knowledge/full-graph-rxnorm-2026-07-18/index_manifest.json
```

The pinned 18 July graph contains 94,098 nodes, 97,232 semantic edges, and 97,250 evidence rows.
Its 89,112 coded nodes come from TT06 and RxNorm; the remaining term nodes represent structured
dosage forms, strengths, and uncoded mined mentions. The graph adds 14,602 unique ingredient
links, 46,804 dosage-form links, 22,630 strength links, 12,919 ICD hierarchy links, and the
707 DailyMed relation observations. The read-only SQLite artifact is
`.cache/clingrounder/knowledge-graph/knowledge-graph-668ddc869e58db0583c0.sqlite3`.

Alias coverage is checked independently before enabling `kg_exact` retrieval:

```bash
uv run clingrounder kg benchmark-aliases \
  --index .cache/clingrounder/knowledge-graph/knowledge-graph-668ddc869e58db0583c0.sqlite3 \
  --alias-overlay data/dictionaries/vietnamese_medical_alias.jsonl \
  --alias-overlay outputs/mining/knowledge/codiesp-icd10-2026-07-18/alias_overlay.jsonl \
  --alias-overlay outputs/mining/knowledge/dailymed-rxnorm-2026-07-17/alias_overlay.jsonl \
  --output outputs/mining/knowledge/full-graph-rxnorm-2026-07-18/alias_benchmark.json
```

CodiEsp and DailyMed resolve at 100% top-1 on this artifact. Vietnamese aliases reach 100% at
top-5 and 95.45% at top-1; `suy tim` (`I50`/`I50.9`) and `hẹp ống sống` (`M48.00`/`M48.0`) remain
ambiguous and must not be auto-emitted. The `kg_exact` retriever is opt-in, exact/toneless only,
and filters entity type and code system before the result limit. It does not traverse the graph or
use FTS to invent candidates. Ontology edges are available for audit, future reranking, and
relation reasoning; they are not an unconstrained NER trigger source.

The current export has 217 chunks and 2,307 entities: 1,880 findings and 427 procedures, with 178
train chunks and 39 development chunks. Chunk limits are soft only when necessary to avoid cutting
an entity. Every local span round-trips to its original document offset, and overlapping labels are
rejected because a BIO token classifier cannot represent them safely.

## Phase 1 Five-Type Model Dataset

The benchmark-owned builder converts only the frozen 76-document manual-gold train split into a
neutral span dataset. It does not read the 24 holdout annotations for model fitting and it rejects
Round 2 documents. Duplicate groups are assigned together using SHA-256 buckets, development
fraction `0.2`, and salt `42`.

```bash
uv run clingrounder-benchmark phase1 model-data build \
  --input-dir data/raw/input \
  --gold-dir data/manual_gold \
  --frozen-split-manifest data/manual_gold/holdout_manifest.json \
  --output-dir outputs/mining/model-datasets/phase1-manual-five-type-v1
```

The materialized build key is
`3bf2365ec1745b9c7fd89cca2ed52035e8f34f3223a4c52d9fd5e8adbb07a1b7`.
It contains 60 train and 16 development documents, 101 chunks, and 2,112 raw-offset spans:

| Internal label | Spans |
| --- | ---: |
| `SYMPTOM` | 912 |
| `DISEASE` | 448 |
| `LAB_TEST` | 317 |
| `LAB_RESULT` | 258 |
| `DRUG` | 177 |

The span JSONL SHA-256 is
`d87384dfdd8ee93bb26f24da0e96f2497acf6b4a3a00e0f27abfd4a0feb64f30`.
Its manifest stores only dataset-relative paths. The public BTC medication example is fingerprinted
as an executable convention test with `included_in_training: false` and
`runtime_lookup_memory: false`.

## Full-Type NER Linux/GPU Run

The full-type run is pinned by
`configs/models/open-corpus-full-type-xlmr-base-2026-07-19.yaml`. It uses the official
`FacebookAI/xlm-roberta-base` checkpoint at immutable revision
`e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`, not a mutable `main` branch. The source dataset has
3,823 chunks and 15,847 entities; the labels present in both train and development are `DISEASE`,
`DRUG`, `FINDING`, and `PROCEDURE`. The manifest-pinned span JSONL SHA-256 is
`90e1b05cac683046a88e58ab26bb090cd58c7d54077a6845f9fa49ccb59c2ca8`.

Inspecting the spec validates dataset offsets, manifest identity, split label compatibility,
checkpoint identity, and the exact Linux/CUDA requirements without importing Torch:

```bash
uv run clingrounder-research model inspect-token-classifier-run \
  --config configs/models/open-corpus-full-type-xlmr-base-2026-07-19.yaml
```

On a networked Linux host, prefetch the immutable checkpoint once. Training itself remains
`local_files_only=true`, so a compute worker cannot silently switch model revisions:

```bash
uv sync --extra ml
uv run hf download FacebookAI/xlm-roberta-base \
  --revision e73636d4f797dec63c3081bb6ed5c7b0bb3f2089

CUDA_VISIBLE_DEVICES=0 uv run clingrounder-research model train-token-classifier-run \
  --config configs/models/open-corpus-full-type-xlmr-base-2026-07-19.yaml
```

The run requires Linux, one CUDA device with at least 16 GiB VRAM, compute capability 8.0, and
BF16 support. It uses batch 4 with gradient accumulation 4. Before loading the model, the command
checks OS, device count, VRAM, compute capability, and BF16 support. The final manifest records GPU
identity, CUDA/Torch versions, run-spec SHA, dataset SHA, model revision, metrics, and saved-model
fingerprint. `run_root` anchors every path independently of the caller's CWD, persisted paths remain
repository-relative, and `full_determinism` is enabled for the checked-in run. The current macOS
x86_64 workspace correctly reports `validated_not_executed`; it is not a Linux/CUDA result.

After training, run inference with an NER-only pipeline config. Set `model_id` to the saved
`final-model` directory and set `revision` to `model.fingerprint` from its `run_manifest.json`:

```yaml
pipeline:
  version: open-corpus-full-type-xlmr-base-2026-07-19
  enable_context: false
  enable_linking: false
  enable_candidate_reranking: false
  enable_graph_evidence_reranking: false
  enable_entity_kg_validation: false
  enable_relations: false
  enable_relation_kg_validation: false

models:
  entity_extractor:
    model_id: outputs/models/open-corpus-full-type-xlmr-base-2026-07-19/final-model
    revision: <model.fingerprint from run_manifest.json>
    device: cuda
    batch_size: 8
    max_length: 512
    stride: 64
```

```bash
uv run clingrounder pipeline run \
  --input outputs/mining/fused/open-corpus-v1-39106c1cc9d0/documents.jsonl \
  --config /path/to/full-type-inference.yaml \
  --output outputs/models/open-corpus-full-type-xlmr-base-2026-07-19/predictions.jsonl \
  --parallel-backend serial
```

Use serial document orchestration because the adapter already batches token windows on one GPU;
process workers would load one model copy per worker. The model adapter rejects offsets that do not
round-trip to source text.

## Graph Evidence Reranker Benchmark

Graph evidence is evaluated only as a bounded reranker feature. It cannot create a candidate and
therefore cannot change recall@K. The benchmark uses TT06 plus the CodiEsp train-only alias overlay,
calibrates the bonus on official dev, and evaluates the selected bonus once on official test. Every
document-backed graph evidence row is checked to belong to `corpus_split=train`.

```bash
uv run clingrounder kg benchmark-reranker \
  --index .cache/clingrounder/knowledge-graph/knowledge-graph-d2b7d076728655e78e13.sqlite3 \
  --nodes outputs/mining/knowledge/codiesp-train-cooccurrence-2026-07-19/nodes.jsonl \
  --edges outputs/mining/knowledge/codiesp-train-cooccurrence-2026-07-19/edges.jsonl \
  --evidence outputs/mining/knowledge/codiesp-train-cooccurrence-2026-07-19/evidence.jsonl \
  --terminology-index .cache/clingrounder/terminology/terminology-codiesp-train-2026-07-18.sqlite3 \
  --terminology-source data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl \
  --terminology-source data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl \
  --terminology-alias-overlay outputs/mining/knowledge/dailymed-rxnorm-2026-07-17/alias_overlay.jsonl \
  --terminology-alias-overlay outputs/mining/knowledge/codiesp-icd10-split-2026-07-18/train/alias_overlay.jsonl \
  --documents outputs/mining/fused/open-corpus-v1-39106c1cc9d0/documents.jsonl \
  --annotations outputs/mining/fused/open-corpus-v1-39106c1cc9d0/harmonized_annotations.jsonl \
  --output outputs/mining/knowledge/codiesp-train-cooccurrence-2026-07-19/graph_reranker_benchmark.json
```

The dev-selected maximum bonus was `0.04`. On 2,052 mapped, contiguous test diagnoses, top-1
accuracy increased from `67.3002%` to `67.7388%` and MRR from `0.686894` to `0.688835`.
Recall@5/10/20 stayed unchanged by design. The feature affected 358 test queries, improved 14 ranks,
and worsened 9. This is a small positive upper bound because same-sentence context links are gold;
production promotion therefore also uses a two-pass benchmark with predicted context links. Repeat
the pinned command above with `--context-mode predicted_exact_unique` and write to
`graph_reranker_predicted_context_benchmark.json` instead of overwriting the oracle report.

This mode still uses gold mention spans to isolate linking behavior, but neighboring codes are not
read from gold. Only one type-compatible ICD-10 output with `exact` provenance can become a context
anchor; toneless, FTS, and ambiguous outputs abstain. On test, 1,614 anchors reached `99.13%`
precision. The selected `0.04` bonus improved top-1 from `67.3002%` to `67.6901%` and MRR from
`0.686894` to `0.688591`; 14 ranks improved and 10 worsened. The predicted-context gain retains
about 89% of the oracle top-1 gain. The feature is suitable for an opt-in second-pass linker, but it
must remain off by default until a full predicted-NER benchmark confirms that span errors do not
erase this small gain.

The reusable pipeline second pass is configured independently of `kg_exact` retrieval:

```yaml
terminology:
  knowledge_graph_index_path: .cache/clingrounder/knowledge-graph/<fingerprint>.sqlite3

pipeline:
  enable_linking: true
  enable_graph_evidence_reranking: true
  graph_evidence_max_bonus: 0.04
  graph_evidence_min_support: 2
  graph_evidence_relation_types: [CO_OCCURS_WITH]
  graph_evidence_cache_size: 4096
```

Runtime order is mention retrieval, optional lexical/model reranking, graph second pass, then code
assignment. Only exact-unique linked neighbors in the same sentence become context anchors.
Unresolved sentence spans are excluded instead of being grouped under a synthetic sentence. The
second pass records `anchor_entities`, `queries_with_context`, `queries_with_graph_feature`, and
`changed_top1` in `PipelineTrace`; it never creates terminology candidates or changes raw spans.
Node and neighbor reads use bounded thread-safe LRU caches, so a long-running worker does not retain
the full graph in process memory.

To remove the remaining gold-span assumption, rerun the held-out benchmark with the pipeline
predictions created above:

```bash
uv run clingrounder kg benchmark-reranker \
  --index .cache/clingrounder/knowledge-graph/knowledge-graph-d2b7d076728655e78e13.sqlite3 \
  --nodes outputs/mining/knowledge/codiesp-train-cooccurrence-2026-07-19/nodes.jsonl \
  --edges outputs/mining/knowledge/codiesp-train-cooccurrence-2026-07-19/edges.jsonl \
  --evidence outputs/mining/knowledge/codiesp-train-cooccurrence-2026-07-19/evidence.jsonl \
  --terminology-index .cache/clingrounder/terminology/terminology-codiesp-train-2026-07-18.sqlite3 \
  --terminology-source data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl \
  --terminology-source data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl \
  --terminology-alias-overlay outputs/mining/knowledge/dailymed-rxnorm-2026-07-17/alias_overlay.jsonl \
  --terminology-alias-overlay outputs/mining/knowledge/codiesp-icd10-split-2026-07-18/train/alias_overlay.jsonl \
  --documents outputs/mining/fused/open-corpus-v1-39106c1cc9d0/documents.jsonl \
  --annotations outputs/mining/fused/open-corpus-v1-39106c1cc9d0/harmonized_annotations.jsonl \
  --predictions outputs/models/open-corpus-full-type-xlmr-base-2026-07-19/predictions.jsonl \
  --context-mode predicted_ner_exact_unique \
  --output outputs/mining/knowledge/codiesp-train-cooccurrence-2026-07-19/graph_reranker_predicted_ner_benchmark.json
```

This regime scores every mapped gold diagnosis. A missed exact disease span receives no candidate
rank, while spurious predicted neighbors remain eligible anchors and therefore reduce reported
anchor precision. The report separates target NER recall, baseline linking, and graph delta.
