# RxNorm Full July 2026

## Source And Access Contract

The processed terminology is the NLM RxNorm Full Monthly Release dated 6 July 2026. The source is
registered as `rxnorm_full_2026_07_06` in `data/sources/mining_registry.yaml` with:

- pinned version `2026-07-06`;
- `credentialled` access because the caller must accept NLM/UMLS terms and supply a local archive;
- hosted processing disabled;
- immutable retention in the content-addressed data plane;
- allowed use limited to terminology linking, dictionary enrichment and linking evaluation.

The raw ZIP is not committed and its workstation path is not provenance. Its portable identity is:

| Field | Value |
| --- | --- |
| Filename | `RxNorm_full_07062026.zip` |
| SHA-256 | `53523ee9f1fcd7ee426698edf566aedebe548a6ec8cc372c41271fc5b28e784c` |
| Byte size | 259,313,098 |
| CAS URI | `medical-kg-cas://sha256/53523ee9...e784c` |

The filename is used only as a convenience and cannot select a release. The connector rejects any
different bytes before a downstream parser or index sees them.

## Acquisition And Portability

`configs/mining/rxnorm-full-2026-07-06.yaml` imports an explicitly supplied archive through the
`local_archive` connector. The caller selects paths at runtime:

```bash
export RXNORM_FULL_ARCHIVE=/secure/licensed/RxNorm_full_07062026.zip
export MEDICAL_KG_ARTIFACT_STORE=file:///mnt/medical-kg/mining-artifacts
uv run medical-kg data run --plan configs/mining/rxnorm-full-2026-07-06.yaml
```

Both runtime values must be absolute paths or explicit URIs. Relative values are resolved from the
plan directory so CI and interactive invocations share one path contract. If the stage manifest is
copied but the selected CAS lacks the archive, the runner reacquires the pinned bytes once instead
of reporting a false cache hit.

Discovery uses the actual local file URI only while reading. The canonical source manifest stores
`local-source://rxnorm_full_2026_07_06/RxNorm_full_07062026.zip`; neither it nor CAS metadata contains
`/Users/...`, `/home/...` or the external mount. The same source manifest therefore resolves against
a local disk, copied external volume or S3-compatible store.

When a downstream ZIP/RRF reader needs seekable bytes, hydrate the object atomically:

```bash
uv run medical-kg data artifact materialize \
  --store "$MEDICAL_KG_ARTIFACT_STORE" \
  --sha256 53523ee9f1fcd7ee426698edf566aedebe548a6ec8cc372c41271fc5b28e784c \
  --expected-byte-size 259313098 \
  --output .cache/medical-kg/release-inputs/RxNorm_full_07062026.zip
```

Hydration streams in bounded chunks, hashes while writing and atomically publishes only matching
bytes. The output path is a rebuildable cache and is never release identity.

## Terminology Processing

The importer reads root `rrf/` members and preserves active RxNorm concepts plus structured fields
needed by medication normalization. Canonical outputs are:

- `data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl`;
- `data/standards/rxnorm/processed/rxnorm_full_07062026_import_manifest.json`.

Observed release profile:

| Measure | Value |
| --- | ---: |
| `RXNCONSO.RRF` rows | 1,202,603 |
| active RxNorm rows | 209,650 |
| accepted concepts | 73,912 |
| `RXNREL.RRF` rows | 7,423,180 |
| active RxNorm relation rows | 2,259,114 |
| `RXNSAT.RRF` rows | 7,687,120 |
| active RxNorm attribute rows | 817,115 |
| concepts with ingredient metadata | 54,211 |
| concepts with dose form | 46,670 |
| concepts with strength | 22,500 |
| concepts with brand name | 34,449 |

The canonical concept JSONL contains 73,912 concepts and has SHA-256
`42faaadeb12222ec099f05c398f3ff858866cb694cbb6dc0cb193f0ea23359cb`. SQLite FTS is derived from
this JSONL and is not source of truth.

## Active NDC Extraction

`compile-rxnorm-ndc` streams only `rrf/RXNSAT.RRF`, validates the source SHA and retains active NDC
attributes. It produced:

| Measure | Value |
| --- | ---: |
| active NDC rows | 249,706 |
| unique RxNorm products | 120,469 |
| malformed rows | 0 |
| duplicate active rows | 0 |
| canonical JSONL SHA-256 | `59e6184287d1a4424b8599f6a45774d19768a4c60cf3d7c011a4dd961cc65cee` |

The JSONL and report are portable canonical artifacts. `rxnorm_ndc.sqlite3` is a disposable query
cache used for exact DailyMed joins and may be rebuilt on each machine.

## How It Improves Retrieval

RxNorm is used in three distinct roles; they must not be conflated:

1. Canonical terminology constrains every emitted RxCUI and supplies ingredient, brand, form and
   strength metadata.
2. Active NDC attributes provide independent product identity for exact DailyMed linking.
3. Train-only DailyMed aliases augment lexical retrieval only after code, type, TTY and global
   conflict validation.

On 379 held-out DailyMed product queries, a 130-alias train overlay improved lexical MRR from
`0.1862` to `0.2051`, recall@10 from `0.5831` to `0.6069`, and recall@20 from `0.7810` to `0.7968`.
Only 13 held-out aliases occurred in the overlay, so larger gains require structured attributes or
a learned reranker rather than uncontrolled alias expansion.

## Promotion Boundary

- Runtime canonical source: active concepts from this exact release.
- Runtime opt-in: exact train aliases that resolve to one permitted RxCUI and do not collide with a
  canonical alias.
- Derived cache only: SQLite FTS and NDC indexes.
- Review only: ambiguous aliases, suppressed/deprecated codes, package conflicts and rows where
  DailyMed set/version disagrees with NDC evidence.
- Never automatic: fuzzy selection of one RxCUI from multiple exact products or mapping a disease
  entity to RxNorm.

## Reproduce And Verify

The release spec `configs/mining/releases/open-ner-retrieval-v1.yaml` contains the exact ordered
commands for CAS import, hydration, NDC extraction, DailyMed product linking, alias compilation and
held-out retrieval benchmarks. After materialization:

```bash
uv run medical-kg data release verify \
  --manifest data/releases/open-ner-retrieval-v1.lock.json \
  --root . \
  --store "$MEDICAL_KG_ARTIFACT_STORE" \
  --require-cas-objects
```

Use `--verify-cas-content` when a full streaming rehash of the external archive is required. The
release lock hashes canonical terminology, policies, query sets and reports byte-for-byte but does
not require SQLite cache bytes to match across operating systems.

## Remaining Work

- Re-import the terminology through a mining-native RRF parser so its import manifest directly
  records the CAS URI in addition to the release lock association.
- Benchmark structured strength/form/route features and a local cross-encoder reranker.
- Audit suppressed and retired concepts against downstream snapshots before each monthly upgrade.
- Keep July 2026 frozen for existing experiments; a newer RxNorm release must create a new source
  ID, CAS object, terminology JSONL, index fingerprint and benchmark, never mutate this lane.
