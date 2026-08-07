#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clingrounder.schema.types import CodeSystem, EntityType
from clingrounder.dictionaries.icd10_sources import icd10_chapter_for_code
from clingrounder.utils.io import read_jsonl, write_jsonl
from clingrounder.utils.text import normalize_for_match


_BLOCKED_NEW_PHASE1_ICD_PREFIXES = frozenset({"R", "S", "T", "V", "W", "X", "Y", "Z", "U"})
_DRUG_CONTEXT_CUES = tuple(
    normalize_for_match(cue)
    for cue in (
        "dùng",
        "uống",
        "tiêm",
        "truyền",
        "thuốc",
        "liều",
        "đơn thuốc",
        "home meds",
        "meds",
        "medication",
        "medications",
        "dose",
        "tablet",
        "po",
        "iv",
        "prn",
        "bid",
        "tid",
        "qid",
        "daily",
        "started",
        "given",
        "mg",
    )
)
_BLOCKED_NEW_RXNORM_ALIAS_KEYS = frozenset(
    normalize_for_match(alias)
    for alias in (
        "alanine",
        "aspartate",
        "cholesterol",
        "guaiac",
        "lactate",
        "lipase",
        "succinate",
    )
)
_RXNORM_STRUCTURED_PRODUCT_TTYS = frozenset(
    {"SCD", "SBD", "SCDF", "SBDF", "GPCK", "BPCK"}
)
_STANDARD_SCALAR_ENRICHMENT_FIELDS = (
    "official_name_vi",
    "official_name_en",
    "parent_code",
    "icd10_chapter",
    "icd10_chapter_range",
    "icd10_chapter_name_en",
    "icd10_block",
    "rxnorm_id",
    "rxnorm_tty",
    "ingredient",
    "brand_name",
    "generic_name",
    "dose_form",
    "strength",
    "rxnorm_status",
)
_STANDARD_LIST_ENRICHMENT_FIELDS = (
    "parents",
    "rxnorm_ttys",
    "ingredients",
    "brand_names",
    "dose_forms",
    "strengths",
    "rxnorm_activated",
    "rxnorm_obsoleted",
    "rxnorm_human_drug",
    "rxnorm_vet_drug",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge official standard dictionary rows into seed concepts with Phase 1 controls.",
    )
    parser.add_argument("--base", required=True, help="Base seed ConceptEntry JSONL.")
    parser.add_argument(
        "--standard",
        action="append",
        default=[],
        help="Standard ConceptEntry JSONL to merge. Can be repeated.",
    )
    parser.add_argument("--output", required=True, help="Output merged ConceptEntry JSONL.")
    parser.add_argument(
        "--phase1-input-dir",
        help="If provided, include new standard rows only when an alias/name occurs in these TXT files.",
    )
    parser.add_argument(
        "--min-match-chars",
        type=int,
        default=5,
        help="Minimum normalized alias length for input-gated new rows.",
    )
    parser.add_argument(
        "--include-unmatched-standard",
        action="store_true",
        help="Include all standard rows, not just base-code or input-matched rows.",
    )
    parser.add_argument(
        "--allow-new-semantic-type",
        action="append",
        default=[],
        choices=[entity_type.value for entity_type in EntityType],
        help=(
            "Allow adding new standard rows only for this semantic type. Repeatable. "
            "Existing rows are still enriched by concept_id/code."
        ),
    )
    parser.add_argument(
        "--allow-new-concept-id",
        action="append",
        default=[],
        help=(
            "Reviewed standard concept_id allowed to bypass conservative new-code guards. "
            "Repeatable; still requires semantic type gates and input alias match."
        ),
    )
    parser.add_argument(
        "--allow-new-concept-file",
        action="append",
        default=[],
        help=(
            "TSV/CSV/text file of reviewed concept ids allowed to bypass conservative new-code guards. "
            "Uses a concept_id header when present, otherwise the first column."
        ),
    )
    args = parser.parse_args()

    base_rows = read_jsonl(args.base)
    standard_rows = [row for path in args.standard for row in read_jsonl(path)]
    input_text = _normalized_input_text(Path(args.phase1_input_dir)) if args.phase1_input_dir else None
    merged_rows, summary = merge_standard_rows(
        base_rows,
        standard_rows,
        normalized_input_text=input_text,
        min_match_chars=args.min_match_chars,
        include_unmatched_standard=args.include_unmatched_standard,
        allowed_new_semantic_types=set(args.allow_new_semantic_type) or None,
        allowed_new_concept_ids=_allowed_new_concept_ids(args.allow_new_concept_id, args.allow_new_concept_file),
    )
    write_jsonl(args.output, merged_rows)
    summary["base"] = args.base
    summary["standards"] = args.standard
    summary["output"] = args.output
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def merge_standard_rows(
    base_rows: list[dict[str, Any]],
    standard_rows: list[dict[str, Any]],
    *,
    normalized_input_text: str | None = None,
    min_match_chars: int = 5,
    include_unmatched_standard: bool = False,
    allowed_new_semantic_types: set[str] | None = None,
    allowed_new_concept_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged = {str(row["concept_id"]): dict(row) for row in base_rows}
    merged_by_code = _concept_ids_by_code(merged.values())
    base_concept_ids = set(merged)
    base_alias_keys = _base_alias_keys(base_rows)
    added = 0
    enriched = 0
    code_matched_enriched = 0
    skipped = 0
    matched_alias_examples: list[dict[str, str]] = []
    allowed_new_concept_ids = allowed_new_concept_ids or set()

    for row in standard_rows:
        concept_id = str(row.get("concept_id", "")).strip()
        if not concept_id:
            skipped += 1
            continue
        if concept_id in merged:
            merged[concept_id] = _merge_existing_row(merged[concept_id], row)
            enriched += 1
            continue
        code_key = _row_code_key(row)
        existing_concept_id = merged_by_code.get(code_key) if code_key is not None else None
        if existing_concept_id is not None:
            merged[existing_concept_id] = _merge_existing_row(merged[existing_concept_id], row)
            enriched += 1
            code_matched_enriched += 1
            continue
        if allowed_new_semantic_types is not None and str(row.get("semantic_type", "")) not in allowed_new_semantic_types:
            skipped += 1
            continue
        matched_alias = _matched_alias(
            row,
            normalized_input_text,
            min_match_chars,
            base_alias_keys,
            allow_blocked_icd=concept_id in allowed_new_concept_ids,
        )
        if not include_unmatched_standard and matched_alias is None:
            skipped += 1
            continue
        merged[concept_id] = _new_standard_row(row, matched_alias=matched_alias)
        if code_key is not None:
            merged_by_code.setdefault(code_key, concept_id)
        added += 1
        if matched_alias is not None and len(matched_alias_examples) < 20:
            matched_alias_examples.append(
                {
                    "concept_id": concept_id,
                    "code": str(row.get("code", "")),
                    "alias": matched_alias,
                }
            )

    # Base order is part of deterministic tie-breaking in the current lexical retriever.
    # Preserve it and append newly admitted concepts in standard-source order.
    rows = [_apply_derived_metadata(row) for row in merged.values()]
    summary = {
        "base_rows": len(base_rows),
        "standard_rows": len(standard_rows),
        "output_rows": len(rows),
        "base_concept_ids": len(base_concept_ids),
        "added_rows": added,
        "enriched_rows": enriched,
        "code_matched_enriched_rows": code_matched_enriched,
        "skipped_rows": skipped,
        "matched_alias_examples": matched_alias_examples,
        "allowed_new_semantic_types": sorted(allowed_new_semantic_types) if allowed_new_semantic_types else None,
        "allowed_new_concept_ids": sorted(allowed_new_concept_ids) if allowed_new_concept_ids else [],
        "by_code_system": _count_by(rows, "code_system"),
        "by_semantic_type": _count_by(rows, "semantic_type"),
    }
    return rows, summary


def _merge_existing_row(base: dict[str, Any], standard: dict[str, Any]) -> dict[str, Any]:
    row = dict(base)
    for key in _STANDARD_SCALAR_ENRICHMENT_FIELDS:
        if not row.get(key) and standard.get(key):
            row[key] = standard[key]
    row["aliases"] = _unique_strings([*_string_values(row.get("aliases")), *_standard_aliases(standard)])
    row["synonyms"] = _unique_strings([*_string_values(row.get("synonyms")), *_string_values(standard.get("synonyms"))])
    row["abbreviations"] = _unique_strings(
        [*_string_values(row.get("abbreviations")), *_string_values(standard.get("abbreviations"))]
    )
    for key in _STANDARD_LIST_ENRICHMENT_FIELDS:
        values = _unique_strings([*_string_values(row.get(key)), *_string_values(standard.get(key))])
        if values:
            row[key] = values
    row["source_ids"] = sorted({*_source_ids(row), *_source_ids(standard)})
    if not row.get("source") and standard.get("source"):
        row["source"] = standard["source"]
    return row


def _new_standard_row(standard: dict[str, Any], *, matched_alias: str | None) -> dict[str, Any]:
    row = dict(standard)
    aliases = _standard_aliases(standard)
    if matched_alias is not None:
        aliases.append(matched_alias)
    row["aliases"] = _unique_strings(aliases)
    row.setdefault("synonyms", [])
    row.setdefault("abbreviations", [])
    row.setdefault("source_ids", sorted(_source_ids(row)))
    return row


def _apply_derived_metadata(row: dict[str, Any]) -> dict[str, Any]:
    enriched = _apply_rxnorm_concept_policy(dict(row))
    if enriched.get("code_system") != CodeSystem.ICD10.value:
        return enriched
    code = str(enriched.get("code", "")).strip()
    if code:
        chapter = icd10_chapter_for_code(code)
        if not enriched.get("icd10_chapter") and chapter.get("chapter"):
            enriched["icd10_chapter"] = chapter["chapter"]
        if not enriched.get("icd10_chapter_range") and chapter.get("range"):
            enriched["icd10_chapter_range"] = chapter["range"]
        if not enriched.get("icd10_chapter_name_en") and chapter.get("name_en"):
            enriched["icd10_chapter_name_en"] = chapter["name_en"]
    parent_code = str(enriched.get("parent_code") or "").strip()
    if not enriched.get("icd10_block"):
        block = _derived_icd10_block(code, parent_code)
        if block:
            enriched["icd10_block"] = block
    return enriched


def _apply_rxnorm_concept_policy(row: dict[str, Any]) -> dict[str, Any]:
    if not _structured_rxnorm_product(row):
        return row
    blocked = _string_values(row.get("blocked_aliases"))
    for key in ("ingredient", "brand_name", "generic_name"):
        for value in _string_values(row.get(key)):
            if _underspecified_rxnorm_product_name(value, row):
                blocked.append(value)
    if blocked:
        row["blocked_aliases"] = _unique_strings(blocked)
    return row


def _underspecified_rxnorm_product_name(value: str, row: dict[str, Any]) -> bool:
    normalized = normalize_for_match(value)
    canonical = normalize_for_match(str(row.get("canonical_name") or ""))
    if not normalized or normalized == canonical:
        return False
    if any(character.isdigit() for character in normalized):
        return False
    dose_forms = [
        *_string_values(row.get("dose_form")),
        *_string_values(row.get("dose_forms")),
    ]
    if any(
        normalize_for_match(dose_form) in normalized
        for dose_form in dose_forms
        if normalize_for_match(dose_form)
    ):
        return False
    return len(normalized) < len(canonical)


def _derived_icd10_block(code: str, parent_code: str) -> str | None:
    if parent_code and "-" in parent_code:
        return parent_code
    if parent_code and "." in code:
        return parent_code
    return None


def _standard_aliases(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    keys = [
        "canonical_name",
        "official_name_vi",
        "official_name_en",
        "aliases",
        "synonyms",
        "abbreviations",
    ]
    if not _structured_rxnorm_product(row):
        keys.extend(("ingredient", "brand_name", "generic_name"))
    for key in keys:
        values.extend(_string_values(row.get(key)))
    official_name_vi = str(row.get("official_name_vi") or "").strip()
    if official_name_vi.casefold().startswith("bệnh "):
        values.append(official_name_vi[5:])
    for suffix in (", không xác định", ", không đặc hiệu"):
        if official_name_vi.casefold().endswith(suffix):
            values.append(official_name_vi[: -len(suffix)].strip())
    return _unique_strings(values)


def _structured_rxnorm_product(row: dict[str, Any]) -> bool:
    if row.get("code_system") != CodeSystem.RXNORM.value:
        return False
    tty = str(row.get("rxnorm_tty") or "").strip().upper()
    return tty in _RXNORM_STRUCTURED_PRODUCT_TTYS


def _matched_alias(
    row: dict[str, Any],
    normalized_input_text: str | None,
    min_match_chars: int,
    base_alias_keys: set[str],
    *,
    allow_blocked_icd: bool = False,
) -> str | None:
    if normalized_input_text is None:
        return None
    entity_type = str(row.get("semantic_type", ""))
    code_system = str(row.get("code_system", ""))
    if code_system == CodeSystem.ICD10.value and entity_type != EntityType.DISEASE.value:
        return None
    if (
        code_system == CodeSystem.ICD10.value
        and _blocked_new_icd_code(str(row.get("code", "")))
        and not allow_blocked_icd
    ):
        return None
    if code_system == CodeSystem.RXNORM.value and entity_type != EntityType.DRUG.value:
        return None
    for alias in _standard_aliases(row):
        if code_system == CodeSystem.ICD10.value and _ascii_single_token(alias):
            continue
        normalized_alias = normalize_for_match(alias)
        if len(normalized_alias) < min_match_chars:
            continue
        if normalized_alias in base_alias_keys:
            continue
        if code_system == CodeSystem.RXNORM.value and normalized_alias in _BLOCKED_NEW_RXNORM_ALIAS_KEYS:
            continue
        if code_system == CodeSystem.RXNORM.value and not _has_drug_context(normalized_input_text, normalized_alias):
            continue
        if f" {normalized_alias} " in normalized_input_text:
            return alias
    return None


def _base_alias_keys(rows: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        for alias in _standard_aliases(row):
            normalized_alias = normalize_for_match(alias)
            if normalized_alias:
                keys.add(normalized_alias)
    return keys


def _concept_ids_by_code(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}
    for row in rows:
        code_key = _row_code_key(row)
        concept_id = str(row.get("concept_id", "")).strip()
        if code_key is not None and concept_id:
            mapping.setdefault(code_key, concept_id)
    return mapping


def _row_code_key(row: dict[str, Any]) -> tuple[str, str] | None:
    code_system = str(row.get("code_system", "")).strip()
    code = str(row.get("code", "")).strip()
    if not code_system or not code:
        return None
    return (code_system, code)


def _blocked_new_icd_code(code: str) -> bool:
    return bool(code) and code[0].upper() in _BLOCKED_NEW_PHASE1_ICD_PREFIXES


def _ascii_single_token(alias: str) -> bool:
    stripped = alias.strip()
    return bool(stripped) and stripped.isascii() and " " not in stripped


def _has_drug_context(normalized_input_text: str, normalized_alias: str) -> bool:
    search = f" {normalized_alias} "
    start = 0
    while True:
        index = normalized_input_text.find(search, start)
        if index < 0:
            return False
        left = normalized_input_text[max(0, index - 80) : index]
        right = normalized_input_text[index + len(search) : index + len(search) + 60]
        if any(cue in left or cue in right for cue in _DRUG_CONTEXT_CUES):
            return True
        start = index + 1


def _normalized_input_text(input_dir: Path) -> str:
    chunks = []
    for path in sorted(input_dir.glob("*.txt"), key=lambda item: int(item.stem) if item.stem.isdigit() else item.stem):
        chunks.append(normalize_for_match(path.read_text(encoding="utf-8")))
    return f" {' '.join(chunks)} "


def _allowed_new_concept_ids(inline_ids: list[str], files: list[str]) -> set[str]:
    concept_ids = {concept_id.strip() for concept_id in inline_ids if concept_id.strip()}
    for path in files:
        concept_ids.update(_allowed_concept_ids_from_file(Path(path)))
    return concept_ids


def _allowed_concept_ids_from_file(path: Path) -> set[str]:
    concept_ids: set[str] = set()
    concept_id_column = 0
    header_seen = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        delimiter = "\t" if "\t" in stripped else ","
        columns = [column.strip() for column in stripped.split(delimiter)]
        if not header_seen:
            header_seen = True
            normalized_columns = [column.casefold() for column in columns]
            if "concept_id" in normalized_columns:
                concept_id_column = normalized_columns.index("concept_id")
                continue
        if len(columns) <= concept_id_column:
            continue
        concept_id = columns[concept_id_column].strip()
        if concept_id:
            concept_ids.add(concept_id)
    return concept_ids


def _source_ids(row: dict[str, Any]) -> set[str]:
    source_ids: set[str] = set()
    raw_source_ids = row.get("source_ids")
    if isinstance(raw_source_ids, str):
        source_ids.add(raw_source_ids)
    elif isinstance(raw_source_ids, list | tuple | set):
        source_ids.update(str(source_id) for source_id in raw_source_ids if str(source_id).strip())
    source = row.get("source")
    if isinstance(source, str) and source.strip():
        source_ids.add(source.strip())
    return source_ids


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def _unique_strings(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).split()).strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
