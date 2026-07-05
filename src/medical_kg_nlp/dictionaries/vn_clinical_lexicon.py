from __future__ import annotations

import csv
import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.io import write_jsonl
from medical_kg_nlp.utils.text import normalize_for_match


VN_CLINICAL_LEXICON_SOURCE_ID = "vn_clinical_lexicon_reviewed_2026_07_05"


def parse_vn_clinical_lexicon(
    path: str | Path,
    *,
    source_id: str = VN_CLINICAL_LEXICON_SOURCE_ID,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse a reviewed Vietnamese clinical lexicon TSV/CSV into ConceptEntry rows."""
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8-sig")
    delimiter = "\t" if input_path.suffix.lower() == ".tsv" else _sniff_delimiter(text)
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for line_number, raw in enumerate(reader, start=2):
        try:
            row = _row_to_concept(raw, source_id=source_id)
        except ValueError as exc:
            warnings.append({"line": line_number, "message": str(exc)})
            continue
        rows.append(row)
    warnings.extend(_duplicate_warnings(rows))
    return sorted(rows, key=lambda row: str(row["concept_id"])), warnings


def write_vn_clinical_lexicon_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_jsonl(path, [dict(row) for row in rows])


def write_vn_clinical_lexicon_manifest(
    path: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    source_inputs: Sequence[str],
    parse_warnings: Sequence[Mapping[str, Any]] = (),
    source_id: str = VN_CLINICAL_LEXICON_SOURCE_ID,
) -> dict[str, Any]:
    manifest = build_vn_clinical_lexicon_manifest(
        rows=rows,
        source_inputs=source_inputs,
        parse_warnings=parse_warnings,
        source_id=source_id,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_vn_clinical_lexicon_manifest(
    *,
    rows: Sequence[Mapping[str, Any]],
    source_inputs: Sequence[str],
    parse_warnings: Sequence[Mapping[str, Any]] = (),
    source_id: str = VN_CLINICAL_LEXICON_SOURCE_ID,
) -> dict[str, Any]:
    code_keys = [(str(row.get("code_system", "")), str(row.get("code", ""))) for row in rows]
    concept_ids = [str(row.get("concept_id", "")) for row in rows]
    alias_counts = Counter(
        normalize_for_match(alias)
        for row in rows
        for alias in _concept_aliases(row)
        if normalize_for_match(alias)
    )
    return {
        "schema_version": "vn-clinical-lexicon-import.v1",
        "source_id": source_id,
        "source_inputs": list(source_inputs),
        "concepts": len(rows),
        "unique_concept_ids": len(set(concept_ids)),
        "duplicate_concept_ids": _duplicates(concept_ids),
        "unique_codes": len(set(code_keys)),
        "duplicate_codes": [
            {"code_system": code_system, "code": code, "count": count}
            for (code_system, code), count in _duplicate_counter(code_keys).items()
            if code_system and code
        ],
        "by_semantic_type": _count_by(rows, "semantic_type"),
        "by_code_system": _count_by(rows, "code_system"),
        "alias_count": sum(alias_counts.values()),
        "unique_alias_count": len(alias_counts),
        "ambiguous_aliases": [
            {"normalized_alias": alias, "count": count}
            for alias, count in sorted(alias_counts.items())
            if count > 1
        ],
        "parse_warning_count": len(parse_warnings),
        "parse_warnings": [dict(row) for row in parse_warnings],
    }


def _row_to_concept(raw: Mapping[str, str | None], *, source_id: str) -> dict[str, Any]:
    code = _clean_required(raw, "code").upper()
    semantic_type = EntityType(_clean_required(raw, "semantic_type"))
    canonical_name = _clean_required(raw, "canonical_name")
    official_name_vi = _clean(raw.get("official_name_vi")) or canonical_name
    official_name_en = _clean(raw.get("official_name_en"))
    aliases = _unique_strings(
        [
            official_name_vi,
            *_split_list(raw.get("aliases")),
        ]
    )
    synonyms = _split_list(raw.get("synonyms"))
    abbreviations = _split_list(raw.get("abbreviations"))
    parents = _split_list(raw.get("parents"))
    row: dict[str, Any] = {
        "concept_id": _concept_id(code, semantic_type),
        "code": code,
        "code_system": CodeSystem.LOCAL.value,
        "canonical_name": canonical_name,
        "official_name_vi": official_name_vi,
        "semantic_type": semantic_type.value,
        "aliases": aliases,
        "synonyms": synonyms,
        "abbreviations": abbreviations,
        "parents": parents,
        "source": source_id,
        "source_ids": [source_id],
    }
    if official_name_en:
        row["official_name_en"] = official_name_en
    notes = _clean(raw.get("notes"))
    if notes:
        row["notes"] = notes
    return row


def _concept_id(code: str, semantic_type: EntityType) -> str:
    if code.startswith(("SYMPTOM_", "TEST_", "PROC_")):
        return f"LOCAL:{code}"
    if semantic_type == EntityType.SYMPTOM:
        return f"LOCAL:SYMPTOM_{code}"
    if semantic_type == EntityType.LAB_TEST:
        return f"LOCAL:TEST_{code}"
    if semantic_type == EntityType.PROCEDURE:
        return f"LOCAL:PROC_{code}"
    return f"LOCAL:{code}"


def _duplicate_warnings(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for concept_id, count in _duplicate_counter(str(row.get("concept_id", "")) for row in rows).items():
        if concept_id:
            warnings.append({"kind": "duplicate_concept_id", "concept_id": concept_id, "count": count})
    for (code_system, code), count in _duplicate_counter(
        (str(row.get("code_system", "")), str(row.get("code", ""))) for row in rows
    ).items():
        if code_system and code:
            warnings.append({"kind": "duplicate_code", "code_system": code_system, "code": code, "count": count})
    return warnings


def _concept_aliases(row: Mapping[str, Any]) -> list[str]:
    aliases: list[str] = []
    for key in ("canonical_name", "official_name_vi", "official_name_en", "aliases", "synonyms", "abbreviations"):
        value = row.get(key)
        if isinstance(value, str):
            aliases.append(value)
        elif isinstance(value, list | tuple | set):
            aliases.extend(str(item) for item in value)
    return _unique_strings(aliases)


def _split_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return _unique_strings(value.replace(";", "|").split("|"))


def _clean_required(raw: Mapping[str, str | None], key: str) -> str:
    value = _clean(raw.get(key))
    if not value:
        raise ValueError(f"{key} is required.")
    return value


def _clean(value: str | None) -> str:
    return " ".join(str(value or "").split()).strip()


def _unique_strings(values: Sequence[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean(value)
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def _sniff_delimiter(text: str) -> str:
    sample = text[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return "\t"


def _count_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter(str(row.get(key, "")) for row in rows)
    return dict(sorted(counts.items()))


def _duplicate_counter(values: Sequence[Any] | Any) -> Counter[Any]:
    counts = Counter(values)
    return Counter({value: count for value, count in counts.items() if count > 1})


def _duplicates(values: Sequence[str]) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in sorted(_duplicate_counter(values).items())]
