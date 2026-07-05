from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.io import write_jsonl


RXNORM_PRESCRIBABLE_2026_06_01_SOURCE_ID = "rxnorm_prescribable_2026_06_01"
RXNORM_FULL_2026_06_01_SOURCE_ID = "rxnorm_full_2026_06_01"
RXNORM_2026_SOURCE = {
    "primary_source_id": RXNORM_PRESCRIBABLE_2026_06_01_SOURCE_ID,
    "fallback_source_id": RXNORM_FULL_2026_06_01_SOURCE_ID,
    "source": "NLM RxNorm",
    "release_date": "2026-06-01",
    "primary_file": "RxNorm_full_prescribe_06012026.zip",
    "fallback_file": "RxNorm_full_06012026.zip",
}
RXNORM_DEFAULT_TTYS = frozenset({"SCD", "SBD", "IN", "PIN", "MIN", "SCDF", "SBDF", "GPCK", "BPCK"})
RXNORM_TTY_PRIORITY: tuple[str, ...] = ("SCD", "SBD", "IN", "PIN", "MIN", "SCDF", "SBDF", "GPCK", "BPCK")
_INGREDIENT_TTYS = frozenset({"IN", "PIN", "MIN"})
_BRAND_TTYS = frozenset({"SBD", "SBDF", "BPCK"})


@dataclass(frozen=True)
class RxNormSourceTerm:
    rxcui: str
    name: str
    tty: str
    sab: str
    source_id: str
    is_preferred: bool = False


def parse_rxnorm_rxnconso(
    path: str | Path,
    *,
    source_id: str = RXNORM_PRESCRIBABLE_2026_06_01_SOURCE_ID,
    allowed_ttys: Iterable[str] = RXNORM_DEFAULT_TTYS,
) -> list[RxNormSourceTerm]:
    allowed = {tty.upper() for tty in allowed_ttys}
    terms: dict[tuple[str, str, str], RxNormSourceTerm] = {}
    for line in _iter_rxnconso_lines(path):
        term = _parse_rxnconso_line(line, source_id=source_id)
        if term is None or term.tty not in allowed:
            continue
        terms[(term.rxcui, term.tty, term.name.casefold())] = term
    return sorted(terms.values(), key=_term_sort_key)


def build_rxnorm_concept_rows(
    terms: Iterable[RxNormSourceTerm],
    *,
    source_priority: Sequence[str] = (
        RXNORM_PRESCRIBABLE_2026_06_01_SOURCE_ID,
        RXNORM_FULL_2026_06_01_SOURCE_ID,
    ),
) -> list[dict[str, Any]]:
    terms_by_rxcui: dict[str, list[RxNormSourceTerm]] = defaultdict(list)
    for term in terms:
        terms_by_rxcui[term.rxcui].append(term)

    rows: list[dict[str, Any]] = []
    for rxcui, grouped_terms in sorted(terms_by_rxcui.items(), key=lambda item: _numeric_key(item[0])):
        best = sorted(grouped_terms, key=lambda term: _rank_term(term, source_priority))[0]
        aliases = _unique(term.name for term in grouped_terms if term.name != best.name)
        ttys = _unique(term.tty for term in grouped_terms)
        row: dict[str, Any] = {
            "concept_id": f"RXNORM:{rxcui}",
            "code": rxcui,
            "code_system": CodeSystem.RXNORM.value,
            "canonical_name": best.name,
            "semantic_type": EntityType.DRUG.value,
            "rxnorm_id": rxcui,
            "aliases": aliases,
            "synonyms": [],
            "abbreviations": [],
            "source": best.source_id,
            "source_ids": sorted({term.source_id for term in grouped_terms}),
            "rxnorm_tty": best.tty,
            "rxnorm_ttys": ttys,
        }
        if best.tty in _INGREDIENT_TTYS:
            row["ingredient"] = best.name
            row["generic_name"] = best.name
        elif best.tty in _BRAND_TTYS:
            row["brand_name"] = best.name
        else:
            row["generic_name"] = best.name
        rows.append(row)
    return rows


def write_rxnorm_concept_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_jsonl(path, [dict(row) for row in rows])


def write_rxnorm_import_manifest(
    path: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    source_inputs: Sequence[str],
    source_parse_counts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    source_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for source_id in _row_source_ids(row):
            source_counts[source_id] += 1
    manifest = {
        "concepts": len(rows),
        "source_policy": rxnorm_source_policy(),
        "source_parse_counts": [dict(count) for count in source_parse_counts],
        "source_inputs": list(source_inputs),
        "source_counts": dict(sorted(source_counts.items())),
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def rxnorm_source_policy() -> dict[str, Any]:
    return {
        "source": dict(RXNORM_2026_SOURCE),
        "preferred_tty_order": list(RXNORM_TTY_PRIORITY),
        "notes": (
            "Use Current Prescribable Content as the primary Phase 1 drug dictionary and the "
            "full monthly release as fallback coverage."
        ),
    }


def _iter_rxnconso_lines(path: str | Path) -> Iterator[str]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            for name in sorted(archive.namelist()):
                if Path(name).name.upper() == "RXNCONSO.RRF":
                    payload = archive.read(name).decode("utf-8", errors="replace")
                    yield from payload.splitlines()
        return
    with input_path.open("r", encoding="utf-8", errors="replace") as handle:
        yield from handle


def _parse_rxnconso_line(line: str, *, source_id: str) -> RxNormSourceTerm | None:
    fields = line.rstrip("\n").split("|")
    if len(fields) < 18:
        return None
    rxcui = fields[0].strip()
    lat = fields[1].strip()
    is_preferred = fields[6].strip().upper() == "Y"
    sab = fields[11].strip().upper()
    tty = fields[12].strip().upper()
    name = fields[14].strip()
    suppress = fields[16].strip().upper()
    if not rxcui or not name or lat != "ENG" or sab != "RXNORM" or suppress in {"Y", "O"}:
        return None
    return RxNormSourceTerm(
        rxcui=rxcui,
        name=name,
        tty=tty,
        sab=sab,
        source_id=source_id,
        is_preferred=is_preferred,
    )


def _term_sort_key(term: RxNormSourceTerm) -> tuple[tuple[int, str], int, str]:
    return (_numeric_key(term.rxcui), _tty_rank(term.tty), term.name.casefold())


def _rank_term(term: RxNormSourceTerm, source_priority: Sequence[str]) -> tuple[int, int, bool, int, str]:
    try:
        source_rank = source_priority.index(term.source_id)
    except ValueError:
        source_rank = len(source_priority)
    return (source_rank, _tty_rank(term.tty), not term.is_preferred, len(term.name), term.name.casefold())


def _tty_rank(tty: str) -> int:
    try:
        return RXNORM_TTY_PRIORITY.index(tty)
    except ValueError:
        return len(RXNORM_TTY_PRIORITY)


def _numeric_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (10**12, value)


def _row_source_ids(row: Mapping[str, Any]) -> set[str]:
    source_ids: set[str] = set()
    raw_source_ids = row.get("source_ids")
    if isinstance(raw_source_ids, str):
        source_ids.add(raw_source_ids)
    elif isinstance(raw_source_ids, list | tuple | set):
        source_ids.update(str(source_id) for source_id in raw_source_ids)
    source = row.get("source")
    if isinstance(source, str) and source:
        source_ids.add(source)
    return source_ids


def _unique(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(value.split()).strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique
