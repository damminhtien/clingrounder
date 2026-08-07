# Dictionaries And Terminology

The toolkit separates canonical terminology, recognition vocabulary, reviewed aliases, and
runtime indexes. Large standards do not need to be loaded into NER memory to be available for
normalization.

## Lifecycle

```text
licensed/public source archive
  -> deterministic importer
  -> canonical ConceptEntry JSONL
  -> reviewed alias overlays
  -> content-addressed SQLite FTS5 index
  -> read-only TerminologyRepository
  -> retrieval, reranking, and constrained assignment
```

JSONL is the source of truth. SQLite and in-memory indexes are derived artifacts and can always be
rebuilt from fingerprinted inputs.

## Concept Records

`ConceptEntry` carries shared terminology fields:

```json
{
  "concept_id": "ICD10:E11",
  "code": "E11",
  "code_system": "ICD-10",
  "canonical_name": "Type 2 diabetes mellitus",
  "official_name_vi": "Đái tháo đường type 2",
  "semantic_type": "DISEASE",
  "aliases": ["tiểu đường type 2"],
  "abbreviations": ["T2DM"],
  "parent_code": "E10-E14",
  "source_ids": ["example_source"]
}
```

Aliases are lookup evidence. They never replace the source substring or its offsets.

## Recognition And Normalization

The pipeline uses two distinct resource roles:

- **Recognition terminology** is small enough for deterministic in-memory matching and controls
  which surfaces can propose entities.
- **Normalization terminology** can contain complete ICD-10 and RxNorm releases in SQLite and is
  queried only after an entity span exists.

This prevents broad standard aliases from becoming unbounded NER rules while preserving full code
coverage during linking.

```yaml
terminology:
  recognition_path: data/dictionaries/seed_concepts.jsonl
  normalization_paths:
    - /authorized/icd10/concepts.jsonl
    - /authorized/rxnorm/concepts.jsonl
  normalization_index_path: .cache/clingrounder/terminology/full.sqlite3
  alias_overlay_path: data/dictionaries/vietnamese_medical_alias.jsonl
```

## ICD-10 Import

The offline importer supports structured Vietnamese tables, WHO ClaML, and reference code files.
TT06 PDF extraction is a separate preprocessing step so runtime code never scrapes PDFs.

```bash
python scripts/extract_tt06_icd10.py \
  --pdf /authorized/06-byt-kem.pdf \
  --tsv /work/06-byt-kem.tsv \
  --output /work/tt06_icd10_extract.jsonl \
  --manifest /work/tt06_icd10_extract_manifest.json

python scripts/import_icd10_dictionary.py \
  --icd10-vn-tt06 /work/tt06_icd10_extract.jsonl \
  --who-claml /authorized/icd102019en.xml.zip \
  --vietnamese-aliases data/dictionaries/vietnamese_medical_alias.jsonl \
  --output /work/icd10_concepts.jsonl \
  --manifest /work/icd10_import_manifest.json
```

The manifest records source fingerprints, parsed counts, hierarchy coverage, and importer version.

## RxNorm Import

The importer reads `RXNCONSO.RRF` and optionally enriches concepts from `RXNREL.RRF` and
`RXNSAT.RRF`. It keeps ingredient, brand, strength, dose form, route, release, status, and TTY as
structured metadata.

```bash
python scripts/import_rxnorm_dictionary.py \
  --prescribable-rxnorm /authorized/RxNorm_full.zip \
  --output /work/rxnorm_prescribable_concepts.jsonl \
  --manifest /work/rxnorm_import_manifest.json
```

Relevant TTYs include `IN`, `PIN`, `MIN`, `BN`, `SCD`, `SBD`, `SCDF`, `SBDF`, `GPCK`, and `BPCK`.
The linker does not force every medication to a product-level concept. A bare ingredient can map to
an ingredient; explicit and compatible product evidence can prefer SCD/SBD.

Product strength is distinct from administered dose. For example, `1.5 mg po qhs` can describe the
amount taken rather than a manufactured strength. Structured compatibility can rank that evidence
without incorrectly hard-rejecting the product.

## Vietnamese Clinical Lexicon

Reviewed symptoms, tests, procedures, and aliases that are not owned by ICD-10 or RxNorm remain in
a separate LOCAL terminology layer:

```bash
python scripts/import_vn_clinical_lexicon.py \
  --input /authorized/reviewed_terms.tsv \
  --output /work/vn_clinical_lexicon_concepts.jsonl \
  --manifest /work/vn_clinical_lexicon_manifest.json
```

Alias overlays must reference an existing concept and matching semantic type. Ambiguous cross-type
aliases are retained as non-exportable proposals until another source resolves the type.

## Build And Query SQLite

```bash
uv run clingrounder terminology build \
  --source /work/icd10_concepts.jsonl \
  --source /work/rxnorm_prescribable_concepts.jsonl \
  --cache-dir .cache/clingrounder/terminology

uv run clingrounder terminology inspect \
  --index .cache/clingrounder/terminology/<fingerprint>.sqlite3 \
  --query "metformin" \
  --entity-type DRUG \
  --code-system RxNorm
```

The cache identity includes:

- ordered source SHA-256 values;
- alias-overlay SHA-256 values;
- index schema version;
- normalization contract version.

Runtime opens query-only, read-only, thread-local connections and never rebuilds an incompatible
index implicitly.

## Candidate Safety

- Filter by entity type and code system before `LIMIT` where the backend allows it.
- Keep retrieval score separate from calibrated emission probability.
- Preserve all evidence sources when fusing retrievers.
- Reject deprecated or suppressed concepts according to the pinned release policy.
- Abstain on ambiguity instead of inventing a parent fallback.
- Validate final codes against the same repository used for linking.

## Source And Release Audit

Source metadata lives in `data/sources/`; local restricted bytes remain outside Git. Import
manifests and `data/provenance/local-artifacts.json` allow an authorized machine to verify exact
inputs without publishing them.

```bash
python scripts/audit_medical_sources.py \
  --input-dir /authorized/source-root \
  --output-dir outputs/source-audit/current
```

Historical benchmark-specific merge and candidate policies are archived in
[`docs/benchmarks/phase1/dictionary-history.md`](benchmarks/phase1/dictionary-history.md).
