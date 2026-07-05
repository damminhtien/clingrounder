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
locked Phase 1 version is:

- primary: `RxNorm_full_prescribe_06012026.zip`;
- fallback: `RxNorm_full_06012026.zip`.

The importer reads `RXNCONSO.RRF`, filters to active `SAB=RXNORM` terms, and prefers TTY values in
this order: `SCD`, `SBD`, `IN`, `PIN`, `MIN`, `SCDF`, `SBDF`, `GPCK`, `BPCK`. It also profiles and
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

For Phase 1 experiments, avoid merging the full standards directly into runtime NER. Use the
controlled merge command to preserve seed behavior and add only codes whose names occur in the
current input set:

```bash
python scripts/merge_standard_dictionaries.py \
  --base data/dictionaries/seed_concepts.jsonl \
  --standard data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl \
  --standard data/standards/rxnorm/processed/rxnorm_prescribable_06012026_concepts.jsonl \
  --phase1-input-dir data/raw/input \
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
  imported RxNorm prescribable layer.
- `data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl` is a runtime-controlled
  dictionary built from seed plus reviewed/input-gated standard rows.

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

The full RxNorm monthly release is registered as `rxnorm_full_2026_06_01`, but the official NLM
download redirects to UMLS login without local credentials. The current local source audit therefore
marks `data/standards/rxnorm/raw/RxNorm_full_06012026.zip` as optional missing until a licensed UTS
download is provided.

Mine Vietnamese alias candidates without mutating the runtime dictionary:

```bash
python scripts/mine_vietnamese_aliases.py \
  --input-dir data/raw/input \
  --runtime-dictionary data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl \
  --standard-dictionary data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl \
  --standard-dictionary data/standards/rxnorm/processed/rxnorm_prescribable_06012026_concepts.jsonl \
  --output-dir outputs/source_audit/alias_mining
```

This writes `alias_candidates.jsonl` and `alias_candidates.md`. Treat every row as `needs_review`;
do not bulk-append candidates into `seed_concepts.jsonl` without a regression test or blocklist
decision.
