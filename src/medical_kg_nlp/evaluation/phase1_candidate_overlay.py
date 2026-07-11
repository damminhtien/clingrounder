from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from medical_kg_nlp.utils.text import normalize_for_match


TT06_SOURCE_ID = "icd10_vn_tt06_2026"
_SCALAR_NAME_FIELDS = (
    "canonical_name",
    "official_name_vi",
    "official_name_en",
    "ingredient",
    "brand_name",
    "generic_name",
)
_LIST_NAME_FIELDS = ("aliases", "synonyms", "abbreviations", "brand_names")


@dataclass(frozen=True)
class Phase1CandidateOverlayConfig:
    icd_exact: bool = False
    rxnorm_exact: bool = False
    rxnorm_longest: bool = False


@dataclass(frozen=True)
class Phase1CandidateIndex:
    icd_exact: Mapping[str, str]
    rxnorm_exact: Mapping[str, str]
    rxnorm_longest_aliases: tuple[tuple[str, str], ...]

    @classmethod
    def from_jsonl(cls, path: str | Path) -> Phase1CandidateIndex:
        icd_alias_codes: dict[str, set[str]] = defaultdict(set)
        rxnorm_alias_codes: dict[str, set[str]] = defaultdict(set)
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                code = str(row.get("code", "")).strip()
                if not code:
                    continue
                aliases = _row_aliases(row)
                code_system = row.get("code_system")
                if code_system == "ICD-10" and _has_tt06_provenance(row):
                    for alias in aliases:
                        icd_alias_codes[alias].add(code)
                elif code_system == "RxNorm":
                    for alias in aliases:
                        rxnorm_alias_codes[alias].add(code)

        icd_exact = _unique_code_index(icd_alias_codes)
        rxnorm_exact = _unique_code_index(rxnorm_alias_codes)
        longest_aliases = tuple(
            sorted(
                (
                    (alias, code)
                    for alias, code in rxnorm_exact.items()
                    if _safe_embedded_drug_alias(alias)
                ),
                key=lambda item: (-len(item[0]), item[0], item[1]),
            )
        )
        return cls(
            icd_exact=icd_exact,
            rxnorm_exact=rxnorm_exact,
            rxnorm_longest_aliases=longest_aliases,
        )

    def longest_unique_rxnorm_code(self, mention: str) -> str | None:
        normalized = normalize_for_match(mention)
        matches: list[tuple[int, str]] = []
        best_length = 0
        for alias, code in self.rxnorm_longest_aliases:
            alias_length = len(alias)
            if alias_length < best_length:
                break
            if not _contains_normalized_alias(normalized, alias):
                continue
            if alias_length > best_length:
                best_length = alias_length
                matches = [(alias_length, code)]
            else:
                matches.append((alias_length, code))
        codes = {code for _, code in matches}
        return next(iter(codes)) if len(codes) == 1 else None


def apply_phase1_candidate_overlay(
    rows_by_doc: dict[str, list[dict[str, Any]]],
    index: Phase1CandidateIndex,
    config: Phase1CandidateOverlayConfig,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    overlaid: dict[str, list[dict[str, Any]]] = {}
    counters: Counter[str] = Counter()
    for document_id, rows in rows_by_doc.items():
        output_rows: list[dict[str, Any]] = []
        for row in rows:
            output = dict(row)
            output["candidates"] = []
            entity_type = row.get("type")
            mention = str(row.get("text", ""))
            normalized = normalize_for_match(mention)
            code: str | None = None
            source: str | None = None
            if entity_type == "CHẨN_ĐOÁN" and config.icd_exact:
                code = index.icd_exact.get(normalized)
                source = "icd_exact" if code else None
            elif entity_type == "THUỐC":
                if config.rxnorm_exact:
                    code = index.rxnorm_exact.get(normalized)
                    source = "rxnorm_exact" if code else None
                if code is None and config.rxnorm_longest:
                    code = index.longest_unique_rxnorm_code(mention)
                    source = "rxnorm_longest" if code else None
            if code is not None:
                output["candidates"] = [code]
                counters[source or "unknown"] += 1
            output_rows.append(output)
        overlaid[document_id] = output_rows
    counters["assigned_total"] = sum(counters.values())
    return overlaid, dict(sorted(counters.items()))


def candidate_ablation_passes(
    baseline_metrics: Mapping[str, Any],
    trial_metrics: Mapping[str, Any],
) -> bool:
    return (
        float(trial_metrics["score"]) > float(baseline_metrics["score"])
        and float(trial_metrics["candidates_score"]) > float(baseline_metrics["candidates_score"])
    )


def _row_aliases(row: Mapping[str, Any]) -> set[str]:
    blocked = {normalize_for_match(value) for value in _string_values(row.get("blocked_aliases"))}
    aliases: set[str] = set()
    for field in _SCALAR_NAME_FIELDS:
        value = row.get(field)
        if isinstance(value, str):
            normalized = normalize_for_match(value)
            if normalized and normalized not in blocked:
                aliases.add(normalized)
    for field in _LIST_NAME_FIELDS:
        for value in _string_values(row.get(field)):
            normalized = normalize_for_match(value)
            if normalized and normalized not in blocked:
                aliases.add(normalized)
    return aliases


def _has_tt06_provenance(row: Mapping[str, Any]) -> bool:
    return row.get("source") == TT06_SOURCE_ID or TT06_SOURCE_ID in _string_values(row.get("source_ids"))


def _unique_code_index(alias_codes: Mapping[str, set[str]]) -> dict[str, str]:
    return {
        alias: next(iter(codes))
        for alias, codes in alias_codes.items()
        if len(codes) == 1
    }


def _safe_embedded_drug_alias(alias: str) -> bool:
    alphanumeric = "".join(character for character in alias if character.isalnum())
    return len(alphanumeric) >= 2


def _contains_normalized_alias(mention: str, alias: str) -> bool:
    pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
    return re.search(pattern, mention) is not None


def _string_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return tuple(str(item) for item in value)
    return ()
