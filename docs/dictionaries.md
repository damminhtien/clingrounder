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
