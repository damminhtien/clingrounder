# Data Mining

`medical_kg_nlp.mining` builds reusable clinical NLP datasets without depending on a competition
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
export MEDICAL_KG_ARTIFACT_STORE=/Volumes/medical-kg-mining
# or: export MEDICAL_KG_ARTIFACT_STORE=s3://bucket/medical-kg-mining
```

JSONL remains the checkpoint and review exchange format. Frozen dataset tables use sharded Parquet;
DuckDB is an optional query catalog. Install these backends with:

```bash
uv sync --extra dev --extra data
```

## Commands

Validate policy before downloading anything:

```bash
uv run medical-kg data registry validate \
  --registry data/sources/mining_registry.yaml
```

Synchronize one explicit request. The parameter file contains connector inputs such as `pmc_ids`,
`set_ids`, `nct_ids`, local `paths`, or explicit `artifacts` with checksums:

```bash
uv run medical-kg data source sync \
  --source-id pmc_oa \
  --source-version 2026-07-18 \
  --parameters configs/mining/requests/pmc-cases.yaml \
  --store "$MEDICAL_KG_ARTIFACT_STORE" \
  --output outputs/mining/pmc/artifacts.jsonl
```

Parse, label, review, and inspect coverage independently:

```bash
uv run medical-kg data dataset build \
  --source-id pmc_oa \
  --artifacts outputs/mining/pmc/artifacts.jsonl \
  --store "$MEDICAL_KG_ARTIFACT_STORE" \
  --output outputs/mining/pmc/documents.jsonl

uv run medical-kg data dataset inspect \
  --documents outputs/mining/pmc/documents.jsonl \
  --annotations outputs/mining/pmc/proposals.jsonl \
  --output outputs/mining/pmc/source-profile.json \
  --strict

uv run medical-kg data label propose \
  --documents outputs/mining/pmc/documents.jsonl \
  --adapter my_local_plugin:create_labeler \
  --adapter-config configs/models/mining-labeler.yaml \
  --output outputs/mining/pmc/proposals.jsonl

uv run medical-kg data review export \
  --documents outputs/mining/pmc/documents.jsonl \
  --proposals outputs/mining/pmc/proposals.jsonl \
  --output outputs/mining/pmc/review.jsonl

uv run medical-kg data review quality \
  --documents outputs/mining/pmc/documents.jsonl \
  --proposals outputs/mining/pmc/reviewed.jsonl \
  --relations outputs/mining/pmc/relations.jsonl \
  --output outputs/mining/pmc/review-quality.json

uv run medical-kg data coverage report \
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
uv run medical-kg data run --plan configs/mining/phase2.yaml
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

VietBioNER is pinned at Git commit `19ba70a5947d1be72906d407c860b1666b9337e9` under CC BY 4.0.
`configs/mining/vietbioner.yaml` acquires the checksum-pinned archive, preserves each annotator as a
separate document, groups exact duplicate text into one split, and imports source labels as silver
proposals through `medical_kg_nlp.mining.labelers.brat`. The broad internal label mapping is an
import convention, not adjudicated clinical gold. VietMed-NER remains quarantined until an explicit
dataset annotation license is available; do not copy model-card visibility into a redistribution
assumption.

The current VietBioNER snapshot can be reproduced without an implicit download or parser choice:

```bash
uv run medical-kg data run --plan configs/mining/vietbioner.yaml

uv run medical-kg data label propose \
  --documents outputs/mining/vietbioner-19ba70a/documents.jsonl \
  --adapter medical_kg_nlp.mining.labelers.brat:create_brat_archive_labeler \
  --adapter-config configs/mining/labelers/vietbioner.yaml \
  --output outputs/mining/vietbioner-19ba70a/source_annotations.jsonl

uv run medical-kg data dataset inspect \
  --documents outputs/mining/vietbioner-19ba70a/documents.jsonl \
  --annotations outputs/mining/vietbioner-19ba70a/source_annotations.jsonl \
  --output outputs/mining/vietbioner-19ba70a/source_profile.json \
  --strict

uv run medical-kg data dataset reconcile-duplicates \
  --documents outputs/mining/vietbioner-19ba70a/documents.jsonl \
  --annotations outputs/mining/vietbioner-19ba70a/source_annotations.jsonl \
  --documents-output outputs/mining/vietbioner-19ba70a/reconciled/documents.jsonl \
  --annotations-output outputs/mining/vietbioner-19ba70a/reconciled/training_annotations.jsonl \
  --review-output outputs/mining/vietbioner-19ba70a/reconciled/review_annotations.jsonl \
  --mapping-output outputs/mining/vietbioner-19ba70a/reconciled/document_map.jsonl \
  --report-output outputs/mining/vietbioner-19ba70a/reconciled/agreement_report.json \
  --labeler-id vietbioner-exact-duplicate-consensus:v1

uv run medical-kg data review export \
  --documents outputs/mining/vietbioner-19ba70a/reconciled/documents.jsonl \
  --proposals outputs/mining/vietbioner-19ba70a/reconciled/review_annotations.jsonl \
  --output outputs/mining/vietbioner-19ba70a/reconciled/review_queue.jsonl

uv run medical-kg data lexicon build \
  --documents outputs/mining/vietbioner-19ba70a/reconciled/documents.jsonl \
  --annotations outputs/mining/vietbioner-19ba70a/reconciled/training_annotations.jsonl \
  --output outputs/mining/vietbioner-19ba70a/reconciled/mention_inventory.jsonl \
  --conflicts-output outputs/mining/vietbioner-19ba70a/reconciled/mention_conflicts.jsonl \
  --report-output outputs/mining/vietbioner-19ba70a/reconciled/mention_inventory_report.json

uv run medical-kg data lexicon crosswalk \
  --inventory outputs/mining/vietbioner-19ba70a/reconciled/mention_inventory.jsonl \
  --index .cache/medical-kg/terminology/terminology-0598a6a288ef81ea932f.sqlite3 \
  --source data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl \
  --source data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl \
  --policy configs/mining/crosswalk/vietbioner.yaml \
  --output outputs/mining/vietbioner-19ba70a/reconciled/terminology_crosswalk.jsonl \
  --report-output outputs/mining/vietbioner-19ba70a/reconciled/terminology_crosswalk_report.json \
  --workers 4

uv run medical-kg data snapshot freeze \
  --documents outputs/mining/vietbioner-19ba70a/reconciled/documents.jsonl \
  --annotations outputs/mining/vietbioner-19ba70a/reconciled/training_annotations.jsonl \
  --artifacts outputs/mining/vietbioner-19ba70a/artifacts.jsonl \
  --version vietbioner-19ba70a-reconciled-silver-v1 \
  --created-at 2026-07-18T15:22:54+07:00 \
  --output-dir outputs/mining/snapshots/vietbioner-19ba70a-reconciled-silver-v1 \
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
uv run medical-kg data run --plan configs/mining/codiesp.yaml

uv run medical-kg data label propose \
  --documents outputs/mining/codiesp-zenodo-3837305/documents.jsonl \
  --adapter medical_kg_nlp.mining.labelers.codiesp:create_codiesp_archive_labeler \
  --adapter-config configs/mining/labelers/codiesp.yaml \
  --output outputs/mining/codiesp-zenodo-3837305/source_annotations.jsonl \
  --batch-size 256

uv run medical-kg data dataset inspect \
  --documents outputs/mining/codiesp-zenodo-3837305/documents.jsonl \
  --annotations outputs/mining/codiesp-zenodo-3837305/source_annotations.jsonl \
  --output outputs/mining/codiesp-zenodo-3837305/source_profile.json \
  --strict

uv run medical-kg data dataset curate-annotations \
  --annotations outputs/mining/codiesp-zenodo-3837305/source_annotations.jsonl \
  --policy configs/mining/curation/codiesp-contiguous-ner.yaml \
  --accepted-output outputs/mining/codiesp-zenodo-3837305/contiguous_training_annotations.jsonl \
  --rejected-output outputs/mining/codiesp-zenodo-3837305/noncontiguous_review_annotations.jsonl \
  --report-output outputs/mining/codiesp-zenodo-3837305/curation_report.json

uv run medical-kg data snapshot freeze \
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
export MEDICAL_KG_ARTIFACT_STORE=/Volumes/medical-kg-mining
uv run medical-kg data run --plan configs/mining/pmc-rare-cases-2026-07-18.yaml

uv run medical-kg data dataset inspect \
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
`Jul 17, 2026 07:50:58PM EST`. Set `MEDICAL_KG_ARTIFACT_STORE` to an external volume or S3 prefix;
the full human-label releases are too large for the repository workspace.

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

uv run medical-kg data snapshot freeze \
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
uv run medical-kg data run --plan configs/mining/dailymed-rxnorm-2026-07-17.yaml

uv run medical-kg data mapping compile-dailymed-rxnorm \
  --artifacts outputs/mining/dailymed-rxnorm-2026-07-17/artifacts.jsonl \
  --store "$MEDICAL_KG_ARTIFACT_STORE" \
  --output outputs/mining/dailymed-rxnorm-2026-07-17/compiled_mappings.jsonl \
  --index-output outputs/mining/dailymed-rxnorm-2026-07-17/dailymed_rxnorm.sqlite3 \
  --report-output outputs/mining/dailymed-rxnorm-2026-07-17/compilation_report.json

uv run medical-kg data mapping audit-dailymed-rxnorm \
  --index outputs/mining/dailymed-rxnorm-2026-07-17/dailymed_rxnorm.sqlite3 \
  --terminology-index .cache/medical-kg/terminology/terminology-0598a6a288ef81ea932f.sqlite3 \
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
