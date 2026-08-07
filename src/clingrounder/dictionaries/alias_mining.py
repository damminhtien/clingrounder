from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from clingrounder.dictionaries.source_audit import unknown_mention_candidates
from clingrounder.utils.io import read_jsonl, write_jsonl
from clingrounder.utils.text import normalize_for_match

_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹ]+")
_ABBREVIATION_RE = re.compile(r"\b[A-ZĐ]{2,8}\b")
_MIN_ALIAS_CHARS = 4


def mine_vietnamese_alias_candidates(
    *,
    input_dir: str | Path,
    runtime_dictionary_path: str | Path,
    standard_dictionary_paths: Sequence[str | Path] = (),
    top_k_unknown: int = 100,
    top_k_abbreviations: int = 80,
) -> list[dict[str, Any]]:
    """Mine review candidates; never mutates the runtime dictionary.

    Candidates are meant for a human curation queue. The pipeline intentionally separates
    standards/full from runtime/controlled dictionaries so broad standards cannot silently become
    high-recall false positives.
    """
    runtime_rows = read_jsonl(runtime_dictionary_path)
    runtime_alias_keys = {normalize_for_match(alias) for row in runtime_rows for alias in _row_aliases(row)}
    input_text = _input_text(input_dir)
    normalized_input = _searchable_text(input_text)

    candidates: list[dict[str, Any]] = []
    candidates.extend(
        _standard_exact_candidates(
            standard_rows=[row for path in standard_dictionary_paths for row in read_jsonl(path)],
            runtime_alias_keys=runtime_alias_keys,
            normalized_input=normalized_input,
        )
    )
    candidates.extend(
        _unknown_phrase_candidates(
            input_dir=input_dir,
            runtime_rows=runtime_rows,
            top_k=top_k_unknown,
        )
    )
    candidates.extend(
        _abbreviation_candidates(
            input_text=input_text,
            runtime_alias_keys=runtime_alias_keys,
            top_k=top_k_abbreviations,
        )
    )
    return _ranked_unique_candidates(candidates)


def write_alias_mining_outputs(candidates: Sequence[Mapping[str, Any]], output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    write_jsonl(path / "alias_candidates.jsonl", [dict(row) for row in candidates])
    (path / "alias_candidates.md").write_text(render_alias_candidates_markdown(candidates), encoding="utf-8")


def render_alias_candidates_markdown(candidates: Sequence[Mapping[str, Any]]) -> str:
    lines = ["# Vietnamese Alias Mining Candidates", ""]
    by_type = Counter(str(row.get("proposal_type", "")) for row in candidates)
    lines.append(f"- Total candidates: {len(candidates)}")
    lines.append(f"- By proposal type: {dict(sorted(by_type.items()))}")
    lines.extend(["", "## Top Candidates", ""])
    for row in candidates[:80]:
        lines.append(
            "- "
            f"{row.get('priority')} `{row.get('proposal_type')}` "
            f"`{row.get('alias') or row.get('term')}` "
            f"count={row.get('occurrence_count', row.get('count', 0))} "
            f"target={row.get('target_concept_id', '')} "
            f"action={row.get('recommended_action', '')}"
        )
    return "\n".join(lines) + "\n"


def _standard_exact_candidates(
    *,
    standard_rows: Sequence[Mapping[str, Any]],
    runtime_alias_keys: set[str],
    normalized_input: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in standard_rows:
        for alias in _row_aliases(row):
            normalized_alias = normalize_for_match(alias)
            if len(normalized_alias) < _MIN_ALIAS_CHARS or normalized_alias in runtime_alias_keys:
                continue
            if _bad_standard_alias(row, alias, normalized_alias):
                continue
            occurrence_count = normalized_input.count(f" {normalized_alias} ")
            if occurrence_count <= 0:
                continue
            candidates.append(
                {
                    "proposal_type": "standard_exact_missing_runtime",
                    "priority": _priority(row, occurrence_count),
                    "alias": alias,
                    "normalized_alias": normalized_alias,
                    "occurrence_count": occurrence_count,
                    "target_concept_id": str(row.get("concept_id", "")),
                    "code": str(row.get("code", "")),
                    "code_system": str(row.get("code_system", "")),
                    "semantic_type": str(row.get("semantic_type", "")),
                    "source_ids": sorted(_source_ids(row)),
                    "recommended_action": "review_add_alias_or_concept",
                }
            )
    return candidates


def _unknown_phrase_candidates(
    *,
    input_dir: str | Path,
    runtime_rows: Sequence[Mapping[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    rows = []
    for item in unknown_mention_candidates(input_dir, runtime_rows, top_k=top_k):
        rows.append(
            {
                "proposal_type": "unknown_phrase",
                "priority": 30 + int(item.get("count", 0)),
                "term": item["term"],
                "normalized_alias": item["normalized"],
                "count": item["count"],
                "recommended_action": "review_classify_as_alias_section_or_ignore",
            }
        )
    return rows


def _abbreviation_candidates(
    *,
    input_text: str,
    runtime_alias_keys: set[str],
    top_k: int,
) -> list[dict[str, Any]]:
    counts = Counter(match.group(0) for match in _ABBREVIATION_RE.finditer(input_text))
    rows: list[dict[str, Any]] = []
    for token, count in counts.most_common(top_k):
        normalized = normalize_for_match(token)
        if normalized in runtime_alias_keys or count < 2:
            continue
        rows.append(
            {
                "proposal_type": "abbreviation_candidate",
                "priority": 20 + count,
                "alias": token,
                "normalized_alias": normalized,
                "occurrence_count": count,
                "recommended_action": "review_expand_abbreviation",
            }
        )
    return rows


def _ranked_unique_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in candidates:
        key = (
            str(row.get("proposal_type", "")),
            str(row.get("normalized_alias", "")),
            str(row.get("target_concept_id", "")),
        )
        existing = unique.get(key)
        if existing is None or int(row.get("priority", 0)) > int(existing.get("priority", 0)):
            unique[key] = dict(row)
    return sorted(unique.values(), key=lambda row: (-int(row.get("priority", 0)), str(row.get("normalized_alias", ""))))


def _row_aliases(row: Mapping[str, Any]) -> list[str]:
    aliases: list[str] = []
    for key in (
        "canonical_name",
        "official_name_vi",
        "official_name_en",
        "aliases",
        "synonyms",
        "abbreviations",
        "ingredient",
        "brand_name",
        "generic_name",
    ):
        aliases.extend(_string_values(row.get(key)))
    blocked = {normalize_for_match(alias) for alias in _string_values(row.get("blocked_aliases"))}
    unique: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        cleaned = " ".join(alias.split()).strip()
        key = normalize_for_match(cleaned)
        if not cleaned or key in seen or key in blocked:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def _bad_standard_alias(row: Mapping[str, Any], alias: str, normalized_alias: str) -> bool:
    code_system = str(row.get("code_system", ""))
    if code_system == "ICD-10" and alias.strip().isascii() and " " not in alias.strip():
        return True
    if code_system == "RxNorm" and normalized_alias in {"alanine", "aspartate", "cholesterol", "lactate", "lipase"}:
        return True
    return False


def _priority(row: Mapping[str, Any], occurrence_count: int) -> int:
    source_ids = _source_ids(row)
    source_score = 0
    if "icd10_vn_tt06_2026" in source_ids:
        source_score += 50
    if "rxnorm_prescribable_2026_06_01" in source_ids:
        source_score += 45
    return source_score + min(occurrence_count, 50)


def _source_ids(row: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    raw = row.get("source_ids")
    if isinstance(raw, str):
        values.add(raw)
    elif isinstance(raw, list | tuple | set):
        values.update(str(item) for item in raw if str(item).strip())
    source = row.get("source")
    if isinstance(source, str) and source.strip():
        values.add(source)
    return values


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def _input_text(input_dir: str | Path) -> str:
    chunks = []
    for path in sorted(Path(input_dir).glob("*.txt"), key=lambda item: _path_sort_key(item)):
        chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def _searchable_text(text: str) -> str:
    normalized = normalize_for_match(text)
    return f" {re.sub(r'[^0-9A-Za-zÀ-ỹ]+', ' ', normalized)} "


def _path_sort_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (10**12, path.name)
