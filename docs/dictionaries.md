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

- `data/sources/medical_resource_registry.yaml` records source ids, URLs, access class, license or
  terms, intended use, and notes.
- `data/dictionaries/seed_concepts.jsonl` may include `source_ids` on rows. The runtime loader is
  backward-compatible and ignores this extra provenance field.
- `data/heuristics/assertion_cues.jsonl` stores assertion/context cues with language, scope, and
  source ids. `medical_kg_nlp.context.rules` loads this file when present and falls back to the
  built-in cue tuples if the data file is unavailable.
- `scripts/build_dictionaries.py` validates source ids in the dictionary and cue table against the
  registry, then validates Vietnamese alias targets.

Current committed open/manual-review sources include CDC ICD-10-CM, WHO ICD-10 references,
Vietnamese ICD lookup labels, RxNorm/RxNav identifiers, MedlinePlus topic names, CodiEsp metadata,
Synthea metadata, and NegEx/ConText-style cue provenance. Credentialed or DUA-bound corpora such as
MIMIC-IV-Note, n2c2/i2b2, and NBME are registered for local adapters but are not downloaded or
committed.

Run the validation summary:

```bash
python scripts/build_dictionaries.py --config configs/default.yaml
```

## ICD-10 Source Import

Use `scripts/import_icd10_dictionary.py` when you have local ICD source files and need to build a
larger ICD-10 dictionary in the same `ConceptEntry` JSONL shape as `seed_concepts.jsonl`.

Supported inputs:

- WHO ICD-10 ClaML XML, or a ZIP containing ClaML XML.
- CDC ICD-10-CM code-description TXT/CSV files, or a ZIP containing them.
- CDC ICD-10-CM tabular XML, or a ZIP containing the tabular XML release.
- Curated Vietnamese alias JSONL using either the existing `target_concept_id` alias-table shape or
  rows with `code`, `official_name_vi`, and `aliases`.

Example:

```bash
python scripts/import_icd10_dictionary.py \
  --who-claml data/raw/icd/icd102019en.xml.zip \
  --cdc-xml data/raw/icd/icd10cm-April-1-2026-XML.zip \
  --vietnamese-aliases data/dictionaries/vietnamese_medical_alias.jsonl \
  --output data/processed/icd10_concepts.jsonl \
  --manifest data/processed/icd10_import_manifest.json
```

The importer is offline and deterministic. It does not download source files, and it does not make
the runtime pipeline depend on WHO, CDC, or KCB availability. After review, selected rows can be
merged into `seed_concepts.jsonl`, then validated with `scripts/build_dictionaries.py` and indexed
with `scripts/build_indexes.py`.
