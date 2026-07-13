# Dictionaries

The active runtime dictionary is `data/dictionaries/seed_concepts.jsonl`. Each row is loaded into a
`ConceptEntry`, and `ConceptEntry.all_names` feeds rule NER, exact matching, fuzzy matching,
character n-gram retrieval, and BM25 retrieval. Normalized aliases are lookup-only and must not
replace source text offsets.

## ICD-10 Terms

Disease and procedure rows should use this shape:

```json
{
  "concept_id": "ICD10:E11",
  "code": "E11",
  "code_system": "ICD-10",
  "canonical_name": "Type 2 diabetes mellitus",
  "official_name_vi": "Đái tháo đường type 2",
  "official_name_en": "Type 2 diabetes mellitus",
  "semantic_type": "DISEASE",
  "aliases": ["tiểu đường type 2"],
  "synonyms": ["non-insulin-dependent diabetes mellitus"],
  "abbreviations": ["T2DM", "DM2"],
  "parent_code": "E10-E14",
  "parents": ["E10-E14"],
  "blocked_aliases": []
}
```

Use `blocked_aliases` for terms that are too broad or create repeated false positives, such as
single-word fragments. Blocked aliases are excluded from lookup even when present in older alias
lists.

## RxNorm Terms

Drug rows should keep ingredient, generic, brand, and dose-form fields separate:

```json
{
  "concept_id": "RXNORM:6809",
  "code": "6809",
  "code_system": "RxNorm",
  "canonical_name": "Metformin",
  "semantic_type": "DRUG",
  "rxnorm_id": "6809",
  "ingredient": "metformin",
  "generic_name": "metformin",
  "brand_name": "Glucophage",
  "dose_form": "tablet",
  "aliases": ["metformin hydrochloride"],
  "synonyms": [],
  "abbreviations": []
}
```

The linker still enforces type/code-system compatibility: `DRUG` entities can link to `RxNorm` or
`NONE`, never ICD-10.

## Vietnamese Medical Aliases

`data/dictionaries/vietnamese_medical_alias.jsonl` is a curation table for lay terms, abbreviations,
and Vietnamese-English equivalents:

```json
{
  "alias": "cao huyết áp",
  "canonical": "tăng huyết áp",
  "target_concept_id": "ICD10:I10",
  "semantic_type": "DISEASE",
  "notes": "Lay synonym for hypertension."
}
```

The current seed concepts already include these aliases in the merged concept rows so the runtime
pipeline only needs one loaded dictionary. `scripts/build_dictionaries.py` validates that alias-table
targets exist and have matching semantic types.

## Source-Backed Resource Pack

The repository keeps a small committed resource pack instead of downloading large or restricted
medical corpora into git:

- `data/standards/source_versions.json` locks the Phase 1 terminology versions:
  ICD-10 Vietnamese labels come from TT 06/2026/TT-BYT, effective 2026-07-01, and
  drug candidates come from NLM RxNorm June 1 2026. For RxNorm, use Current Prescribable
  Content first and the full monthly release as fallback coverage.
- `data/sources/medical_resource_registry.yaml` records source ids, URLs, access class, license or
  terms, intended use, and notes.
- `data/dictionaries/seed_concepts.jsonl` may include `source_ids` on rows. The runtime loader is
  backward-compatible and ignores this extra provenance field.
- `data/heuristics/assertion_cues.jsonl` stores assertion/context cues with language, scope, and
  source ids. `medical_kg_nlp.context.rules` loads this file when present and falls back to the
  built-in cue tuples if the data file is unavailable.
- `scripts/build_dictionaries.py` validates source ids in the dictionary and cue table against the
  registry, then validates Vietnamese alias targets.

Current committed open/manual-review sources include TT 06/2026/TT-BYT and KCB Vietnamese ICD
labels, WHO ICD-10 hierarchy references, CDC ICD-10-CM as a non-primary reference only,
RxNorm/RxNav identifiers, MedlinePlus topic names, CodiEsp metadata, Synthea metadata, and
NegEx/ConText-style cue provenance. Credentialed or DUA-bound corpora such as MIMIC-IV-Note,
n2c2/i2b2, and NBME are registered for local adapters but are not downloaded or committed.

Run the validation summary:

```bash
python scripts/build_dictionaries.py --config configs/default.yaml
```

## ICD-10 Source Import

Use `scripts/import_icd10_dictionary.py` when you have local ICD source files and need to build a
larger ICD-10 dictionary in the same `ConceptEntry` JSONL shape as `seed_concepts.jsonl`.
For Phase 1, the primary ICD source is a reviewed structured extract of the TT 06/2026/TT-BYT
appendix (`06-byt-kem.pdf`). The importer intentionally accepts JSON/JSONL/CSV/TSV extracts instead
of scraping arbitrary PDFs in the runtime path.

Supported inputs:

- TT 06/2026/TT-BYT structured extracts with `code` and Vietnamese name fields.
- WHO ICD-10 ClaML XML, or a ZIP containing ClaML XML.
- CDC ICD-10-CM code-description TXT/CSV files, or a ZIP containing them.
- CDC ICD-10-CM tabular XML, or a ZIP containing the tabular XML release.
- Curated Vietnamese alias JSONL using either the existing `target_concept_id` alias-table shape or
  rows with `code`, `official_name_vi`, and `aliases`.

Example:

```bash
python scripts/extract_tt06_icd10.py \
  --pdf data/standards/icd10_vn/raw/06-byt-kem.pdf \
  --tsv data/standards/icd10_vn/processed/06-byt-kem.tsv \
  --output data/standards/icd10_vn/processed/tt06_icd10_extract.jsonl \
  --manifest data/standards/icd10_vn/processed/tt06_icd10_extract_manifest.json

python scripts/import_icd10_dictionary.py \
  --icd10-vn-tt06 data/standards/icd10_vn/processed/tt06_icd10_extract.jsonl \
  --who-claml data/raw/icd/icd102019en.xml.zip \
  --vietnamese-aliases data/dictionaries/vietnamese_medical_alias.jsonl \
  --output data/processed/icd10_concepts.jsonl \
  --manifest data/processed/icd10_import_manifest.json
```

The importer is offline and deterministic. It does not download source files, and it does not make
the runtime pipeline depend on WHO, CDC, or KCB availability. It refuses WHO/CDC-only builds unless
`--allow-non-vietnamese-primary` is passed for a reference experiment. After review, selected rows
can be merged into `seed_concepts.jsonl`, then validated with `scripts/build_dictionaries.py` and
indexed with `scripts/build_indexes.py`.

## RxNorm Source Import

Use `scripts/import_rxnorm_dictionary.py` when local NLM RxNorm release files are available. The
reproducible Phase 1 baseline remains locked to:

- primary: `RxNorm_full_prescribe_06012026.zip`;
- fallback: `RxNorm_full_06012026.zip`.

The July 6 full bundle is imported as a versioned promotion candidate, not silently substituted
into that baseline. Its root `rrf/` subtree contains the full release and `prescribe/rrf/` contains
Current Prescribable Content. The importer detects and isolates the requested subtree so rows from
the two products are never mixed by filename alone.

The importer reads `RXNCONSO.RRF`, filters to active `SAB=RXNORM` terms, and prefers TTY values in
this order: `SCD`, `SBD`, `IN`, `PIN`, `MIN`, `SCDF`, `SBDF`, `GPCK`, `BPCK`. Full-release fallback
also accepts active `BN` rows when Prescribable Content omits a legitimate brand concept. It profiles and
uses `RXNREL.RRF` and `RXNSAT.RRF` when present to enrich rows with ingredient, brand, dose-form,
strength, activation, and obsoletion metadata. Strength/status metadata is stored on the JSONL rows
for ontology and QA use; it is not added to alias matching.

```bash
python scripts/import_rxnorm_dictionary.py \
  --prescribable-rxnorm data/standards/rxnorm/raw/RxNorm_full_prescribe_06012026.zip \
  --full-rxnorm data/standards/rxnorm/raw/RxNorm_full_06012026.zip \
  --output data/standards/rxnorm/processed/rxnorm_concepts.jsonl \
  --manifest data/standards/rxnorm/processed/rxnorm_import_manifest.json
```

Import the July bundle into separate candidate layers:

```bash
python scripts/import_rxnorm_dictionary.py \
  --prescribable-rxnorm data/standards/rxnorm/raw/RxNorm_full_07062026.zip \
  --prescribable-source-id rxnorm_prescribable_2026_07_06 \
  --full-source-id rxnorm_full_2026_07_06 \
  --release-date 2026-07-06 \
  --primary-file RxNorm_full_07062026.zip \
  --fallback-file RxNorm_full_07062026.zip \
  --output data/standards/rxnorm/processed/rxnorm_prescribable_07062026_concepts.jsonl \
  --manifest data/standards/rxnorm/processed/rxnorm_prescribable_07062026_import_manifest.json

python scripts/import_rxnorm_dictionary.py \
  --full-rxnorm data/standards/rxnorm/raw/RxNorm_full_07062026.zip \
  --prescribable-source-id rxnorm_prescribable_2026_07_06 \
  --full-source-id rxnorm_full_2026_07_06 \
  --release-date 2026-07-06 \
  --primary-file RxNorm_full_07062026.zip \
  --fallback-file RxNorm_full_07062026.zip \
  --output data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl \
  --manifest data/standards/rxnorm/processed/rxnorm_full_07062026_import_manifest.json
```

`rxnorm data.csv` was profiled but not imported. Its two columns (`RXCUI`, `STR`) omit `SAB`, `TTY`,
`SUPPRESS`, relations, and status attributes, so it cannot enforce the runtime dictionary invariants.
See `data/standards/rxnorm/processed/rxnorm_data_csv_profile.json`.

## Vietnamese Clinical Lexicon Import

Use `scripts/import_vn_clinical_lexicon.py` for reviewed Vietnamese LOCAL terminology that is not
owned by ICD-10 or RxNorm: symptoms, lab names, and future procedure terms. This source stays
separate from `seed_concepts.jsonl` so it can be audited by checksum, regenerated from raw TSV, and
merged into runtime dictionaries only through explicit gates.

```bash
python scripts/import_vn_clinical_lexicon.py \
  --input data/standards/vn_clinical_lexicon/raw/reviewed_terms.tsv \
  --output data/standards/vn_clinical_lexicon/processed/vn_clinical_lexicon_concepts.jsonl \
  --manifest data/standards/vn_clinical_lexicon/processed/vn_clinical_lexicon_import_manifest.json
```

For Phase 1 experiments, avoid merging the full standards directly into runtime NER. Use the
controlled merge command to preserve seed behavior and add only codes whose names occur in the
current input set. Procedure rows are kept in the source layer until a later phase explicitly
exports procedures.

```bash
python scripts/merge_standard_dictionaries.py \
  --base data/dictionaries/seed_concepts.jsonl \
  --standard data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl \
  --standard data/standards/rxnorm/processed/rxnorm_prescribable_06012026_concepts.jsonl \
  --standard data/standards/vn_clinical_lexicon/processed/vn_clinical_lexicon_concepts.jsonl \
  --phase1-input-dir data/raw/input \
  --allow-new-semantic-type DISEASE \
  --allow-new-semantic-type DRUG \
  --allow-new-semantic-type SYMPTOM \
  --allow-new-semantic-type LAB_TEST \
  --allow-new-concept-file data/standards/phase1_reviewed/allowed_standard_concepts.tsv \
  --output data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl

python scripts/build_indexes.py \
  --dictionary data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl \
  --alias-overlay data/dictionaries/vietnamese_medical_alias.jsonl \
  --output data/standards/phase1_seed_tt06_rxnorm_controlled_lexical_index.json
```

## Source Audit And Coverage

Keep full standards separate from runtime dictionaries:

- `data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl` is the full TT06 ICD layer.
- `data/standards/rxnorm/processed/rxnorm_prescribable_06012026_concepts.jsonl` is the full
  imported June RxNorm prescribable layer and locked Phase 1 baseline.
- `data/standards/rxnorm/processed/rxnorm_prescribable_07062026_concepts.jsonl` and
  `rxnorm_full_07062026_concepts.jsonl` are audited July promotion candidates.
- `data/standards/vn_clinical_lexicon/processed/vn_clinical_lexicon_concepts.jsonl` is the
  reviewed Vietnamese LOCAL symptom, lab, and procedure layer.
- `data/standards/phase1_reviewed/allowed_standard_concepts.tsv` is a reviewed exception list for
  exact-match standard concepts that may bypass conservative runtime guards while still requiring
  input evidence and semantic-type gates.
- `data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl` is a runtime-controlled
  dictionary built from seed plus reviewed/input-gated standard rows.

### RxNorm Target Level

Phase 1 uses a conservative concept-level policy:

- bare generic or reviewed bare brand mentions without strength/form map to an ingredient-level
  `IN`, `PIN`, or `MIN` target;
- generic mentions with explicit matching strength and dose form may map to `SCD`;
- branded mentions with explicit matching strength and dose form may map to `SBD`;
- `SCDF`, `SBDF`, `GPCK`, and `BPCK` require the corresponding form/package evidence;
- ambiguous or incomplete mentions abstain instead of assuming a strength or form.

Structured product rows retain ingredient and brand metadata, but underspecified names are written
to `blocked_aliases`. This prevents a bare brand such as `Eliquis` from matching every strength-specific
product while preserving `apixaban 5 mg oral tablet -> 1364445`. Run the policy sanitizer after
manual runtime promotions, then rebuild the index:

```bash
python scripts/merge_standard_dictionaries.py \
  --base data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl \
  --output data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl

python scripts/build_indexes.py \
  --dictionary data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl \
  --alias-overlay data/dictionaries/vietnamese_medical_alias.jsonl \
  --output data/standards/phase1_seed_tt06_rxnorm_controlled_lexical_index.json
```

Run the audit report after adding or regenerating sources:

```bash
python scripts/audit_medical_sources.py \
  --input-dir data/raw/input \
  --output-dir outputs/source_audit/current
```

The report writes:

- `source_manifest.json`: source registry issues, local raw/processed file SHA-256/MD5 checksums,
  file sizes, dictionary coverage, RxNorm `RXNCONSO/RXNREL/RXNSAT` counters, and hierarchy coverage.
- `dictionary_coverage.md`: human-readable coverage by code system, semantic type, source id,
  ICD hierarchy, ambiguous aliases, and broad false-positive candidates.
- `manual_review_queue.jsonl`: alias ambiguity, broad alias, lab/metabolite drug-risk, and unknown
  mention candidates for human review.
- `false_positive_blocklist.jsonl` and `false_positive_blocklist.md`: structured review candidates
  for aliases that should be blocked or context-gated, including short aliases, English single-token
  ICD aliases, and lab/metabolite RxNorm aliases that should require explicit drug context.

The June full RxNorm fallback remains registered as optional missing. A licensed July full bundle is
available locally and has been imported into separate Full and Prescribable Content layers. The raw
247 MB bundle is intentionally ignored by Git because it exceeds GitHub's per-file limit; its SHA-256
is retained in the comparison artifact and the generated source audit. Promotion remains a separate
ablation decision recorded in `data/standards/source_versions.json`.

Mine Vietnamese alias candidates without mutating the runtime dictionary:

```bash
python scripts/mine_vietnamese_aliases.py \
  --input-dir data/raw/input \
  --runtime-dictionary data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl \
  --standard-dictionary data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl \
  --standard-dictionary data/standards/rxnorm/processed/rxnorm_prescribable_06012026_concepts.jsonl \
  --standard-dictionary data/standards/vn_clinical_lexicon/processed/vn_clinical_lexicon_concepts.jsonl \
  --output-dir outputs/source_audit/alias_mining
```

This writes `alias_candidates.jsonl` and `alias_candidates.md`. Treat every row as `needs_review`;
do not bulk-append candidates into `seed_concepts.jsonl` without a regression test or blocklist
decision.
