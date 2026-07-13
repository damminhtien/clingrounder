from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from medical_kg_nlp.utils.text import normalize_for_match


_CANDIDATE_REQUIREMENTS = {
    "CHẨN_ĐOÁN": ("ICD-10", "DISEASE"),
    "THUỐC": ("RxNorm", "DRUG"),
}
_LOCKED_SOURCE_BY_SYSTEM = {
    "ICD-10": "icd10_vn_tt06_2026",
    "RxNorm": "rxnorm_full_2026_07_06",
}


def build_manual_gold_candidate_dictionary(
    gold_dir: str | Path,
    source_paths: Iterable[str | Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a compact standards-backed dictionary for reviewed manual-gold codes.

    This dictionary is deliberately separate from the recognition dictionary. Adding a
    reviewed normalization alias must not make NER start proposing that alias at runtime.
    """
    usage, input_issues = _collect_candidate_usage(Path(gold_dir))
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    source_names: dict[tuple[str, str], set[str]] = defaultdict(set)
    sources = [Path(path) for path in source_paths]

    # Earlier compact dictionaries contribute useful aliases. Later complete releases replace
    # their metadata so provenance always resolves to the locked TT06/RxNorm source.
    for source_path in sources:
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{source_path}:{line_number}: expected a JSON object.")
                key = (str(row.get("code_system", "")), str(row.get("code", "")))
                if key in usage:
                    selected[key] = row
                    for field in (
                        "canonical_name",
                        "official_name_vi",
                        "official_name_en",
                        "aliases",
                        "synonyms",
                        "abbreviations",
                    ):
                        source_names[key].update(_string_values(row.get(field)))

    issues = list(input_issues)
    output_rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = defaultdict(int)
    for key in sorted(usage):
        item = usage[key]
        source_row = selected.get(key)
        if source_row is None:
            issues.append(
                {
                    "kind": "missing_standard_code",
                    "code_system": key[0],
                    "code": key[1],
                    "documents": sorted(item["documents"], key=_document_sort_key),
                }
            )
            continue
        expected_semantic_type = str(item["semantic_type"])
        actual_semantic_type = str(source_row.get("semantic_type", ""))
        if actual_semantic_type != expected_semantic_type:
            issues.append(
                {
                    "kind": "standard_semantic_type_mismatch",
                    "code_system": key[0],
                    "code": key[1],
                    "expected": expected_semantic_type,
                    "actual": actual_semantic_type,
                }
            )
            continue
        required_source = _LOCKED_SOURCE_BY_SYSTEM[key[0]]
        source_ids = set(_string_values(source_row.get("source_ids")))
        if source_row.get("source") != required_source and required_source not in source_ids:
            issues.append(
                {
                    "kind": "unlocked_standard_source",
                    "code_system": key[0],
                    "code": key[1],
                    "required_source": required_source,
                    "actual_source": str(source_row.get("source", "")),
                    "documents": sorted(item["documents"], key=_document_sort_key),
                    "mentions": sorted(item["mentions"]),
                }
            )
            continue

        row = dict(source_row)
        row["aliases"] = _merge_reviewed_aliases(
            row,
            set(item["mentions"]) | source_names[key],
        )
        row["manual_gold_reviewed_aliases"] = sorted(item["mentions"])
        row["manual_gold_document_support"] = len(item["documents"])
        output_rows.append(row)
        source_counts[str(row.get("source", "unknown"))] += 1

    audit = {
        "schema_version": "phase1-manual-gold-candidate-dictionary.v1",
        "gold_dir": str(Path(gold_dir)),
        "source_paths": [str(path) for path in sources],
        "used_code_count": len(usage),
        "compiled_concept_count": len(output_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "issue_count": len(issues),
        "issues": issues,
    }
    return output_rows, audit


def write_manual_gold_candidate_dictionary(
    rows: Iterable[dict[str, Any]],
    path: str | Path,
) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in materialized
        ),
        encoding="utf-8",
    )
    return len(materialized)


def _collect_candidate_usage(
    gold_dir: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    usage: dict[tuple[str, str], dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []
    paths = sorted(
        (path for path in gold_dir.glob("*.json") if path.stem.isdigit()),
        key=lambda path: _document_sort_key(path.stem),
    )
    for gold_path in paths:
        rows = json.loads(gold_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{gold_path}: expected a JSON list.")
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            entity_type = str(row.get("type", ""))
            candidates = row.get("candidates", [])
            if not isinstance(candidates, list) or not candidates:
                continue
            requirement = _CANDIDATE_REQUIREMENTS.get(entity_type)
            if requirement is None:
                issues.append(
                    {
                        "kind": "candidate_on_noncodable_type",
                        "document_id": gold_path.stem,
                        "row_index": row_index,
                        "entity_type": entity_type,
                    }
                )
                continue
            code_system, semantic_type = requirement
            mention = str(row.get("text", "")).strip()
            for candidate in candidates:
                code = str(candidate).strip()
                if not code:
                    continue
                key = (code_system, code)
                item = usage.setdefault(
                    key,
                    {
                        "semantic_type": semantic_type,
                        "mentions": set(),
                        "documents": set(),
                    },
                )
                if item["semantic_type"] != semantic_type:
                    raise ValueError(f"Conflicting semantic types for {code_system}:{code}.")
                if mention:
                    item["mentions"].add(mention)
                item["documents"].add(gold_path.stem)
    return usage, issues


def _merge_reviewed_aliases(row: dict[str, Any], mentions: set[str]) -> list[str]:
    aliases = [str(value) for value in row.get("aliases", [])]
    existing_names = {
        normalize_for_match(str(value))
        for key in (
            "canonical_name",
            "official_name_vi",
            "official_name_en",
            "aliases",
            "synonyms",
            "abbreviations",
        )
        for value in _string_values(row.get(key))
    }
    for mention in sorted(mentions, key=lambda value: (normalize_for_match(value), value)):
        normalized = normalize_for_match(mention)
        if normalized and normalized not in existing_names:
            aliases.append(mention)
            existing_names.add(normalized)
    return aliases


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return [str(value)]


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)
