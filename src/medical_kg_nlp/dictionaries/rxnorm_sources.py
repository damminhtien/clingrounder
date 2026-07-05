from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
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
RXNORM_ENRICHMENT_TTYS = frozenset(
    {
        *RXNORM_DEFAULT_TTYS,
        "BN",
        "DF",
        "DFG",
        "PSN",
        "SBDC",
        "SBDFP",
        "SBDG",
        "SCDG",
        "SCDGP",
        "SY",
    }
)
RXNORM_TTY_PRIORITY: tuple[str, ...] = ("SCD", "SBD", "IN", "PIN", "MIN", "SCDF", "SBDF", "GPCK", "BPCK")
_INGREDIENT_TTYS = frozenset({"IN", "PIN", "MIN"})
_BRAND_TTYS = frozenset({"SBD", "SBDF", "BPCK"})
_INGREDIENT_RELAS = frozenset({"HAS_INGREDIENT", "HAS_PRECISE_INGREDIENT", "HAS_INGREDIENTS", "HAS_BOSS"})
_INGREDIENT_INVERSE_RELAS = frozenset({"INGREDIENT_OF", "PRECISE_INGREDIENT_OF", "INGREDIENTS_OF", "BOSS_OF"})
_BRAND_RELAS = frozenset({"HAS_TRADENAME"})
_BRAND_INVERSE_RELAS = frozenset({"TRADENAME_OF"})
_DOSE_FORM_RELAS = frozenset({"HAS_DOSE_FORM", "HAS_DOSEFORMGROUP"})
_DOSE_FORM_INVERSE_RELAS = frozenset({"DOSE_FORM_OF", "DOSEFORMGROUP_OF"})
_STRENGTH_ATTRS = frozenset(
    {
        "RXN_STRENGTH",
        "RXN_AVAILABLE_STRENGTH",
        "RXN_BOSS_STRENGTH_NUM_VALUE",
        "RXN_BOSS_STRENGTH_NUM_UNIT",
        "RXN_BOSS_STRENGTH_DENOM_VALUE",
        "RXN_BOSS_STRENGTH_DENOM_UNIT",
    }
)


@dataclass(frozen=True)
class RxNormSourceTerm:
    rxcui: str
    name: str
    tty: str
    sab: str
    source_id: str
    is_preferred: bool = False


@dataclass(frozen=True)
class RxNormSourceRelation:
    source_rxcui: str
    target_rxcui: str
    rel: str
    rela: str
    source_id: str


@dataclass(frozen=True)
class RxNormSourceAttribute:
    rxcui: str
    attribute: str
    value: str
    source_id: str


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


def parse_rxnorm_rxnrel(
    path: str | Path,
    *,
    source_id: str = RXNORM_PRESCRIBABLE_2026_06_01_SOURCE_ID,
) -> list[RxNormSourceRelation]:
    relations: dict[tuple[str, str, str], RxNormSourceRelation] = {}
    for line in _iter_rrf_lines(path, "RXNREL.RRF"):
        relation = _parse_rxnrel_line(line, source_id=source_id)
        if relation is None:
            continue
        relations[(relation.source_rxcui, relation.target_rxcui, relation.rela)] = relation
    return sorted(relations.values(), key=lambda item: (_numeric_key(item.source_rxcui), item.rela, _numeric_key(item.target_rxcui)))


def parse_rxnorm_rxnsat(
    path: str | Path,
    *,
    source_id: str = RXNORM_PRESCRIBABLE_2026_06_01_SOURCE_ID,
) -> list[RxNormSourceAttribute]:
    attributes: dict[tuple[str, str, str], RxNormSourceAttribute] = {}
    for line in _iter_rrf_lines(path, "RXNSAT.RRF"):
        attribute = _parse_rxnsat_line(line, source_id=source_id)
        if attribute is None:
            continue
        attributes[(attribute.rxcui, attribute.attribute, attribute.value)] = attribute
    return sorted(attributes.values(), key=lambda item: (_numeric_key(item.rxcui), item.attribute, item.value))


def build_rxnorm_concept_rows(
    terms: Iterable[RxNormSourceTerm],
    *,
    enrichment_terms: Iterable[RxNormSourceTerm] = (),
    relations: Iterable[RxNormSourceRelation] = (),
    attributes: Iterable[RxNormSourceAttribute] = (),
    source_priority: Sequence[str] = (
        RXNORM_PRESCRIBABLE_2026_06_01_SOURCE_ID,
        RXNORM_FULL_2026_06_01_SOURCE_ID,
    ),
) -> list[dict[str, Any]]:
    terms_by_rxcui: dict[str, list[RxNormSourceTerm]] = defaultdict(list)
    for term in terms:
        terms_by_rxcui[term.rxcui].append(term)
    name_index = _rxnorm_name_index([*terms, *enrichment_terms], source_priority)
    enrichment = _rxnorm_enrichment(relations, attributes, name_index)

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
        _apply_rxnorm_enrichment(row, enrichment.get(rxcui, {}))
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
    release_profiles: Sequence[Mapping[str, Any]] = (),
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
        "row_enrichment": rxnorm_row_enrichment_summary(rows),
        "release_profiles": [dict(profile) for profile in release_profiles],
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
            "full monthly release as fallback coverage. "
            "RXNREL and RXNSAT are parsed for ingredient, brand, dose-form, strength, and "
            "activation/obsoletion metadata when present."
        ),
    }


def rxnorm_row_enrichment_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "with_ingredient": sum(1 for row in rows if row.get("ingredient") or row.get("ingredients")),
        "with_brand_name": sum(1 for row in rows if row.get("brand_name") or row.get("brand_names")),
        "with_dose_form": sum(1 for row in rows if row.get("dose_form") or row.get("dose_forms")),
        "with_strength": sum(1 for row in rows if row.get("strength") or row.get("strengths")),
        "with_status": sum(1 for row in rows if row.get("rxnorm_status")),
        "inactive_or_obsolete": sum(1 for row in rows if row.get("rxnorm_status") == "inactive"),
    }


def profile_rxnorm_release(
    path: str | Path,
    *,
    allowed_ttys: Iterable[str] = RXNORM_DEFAULT_TTYS,
) -> dict[str, Any]:
    """Return reproducibility/QA counters for a local RxNorm RRF release.

    The runtime dictionary still builds candidates from ``RXNCONSO.RRF`` only. This profile
    intentionally inspects ``RXNREL.RRF`` and ``RXNSAT.RRF`` as evidence that a serious source
    ingest has the relation and attribute files available for later ontology enrichment.
    """
    allowed = {tty.upper() for tty in allowed_ttys}
    return {
        "path": str(path),
        "required_files": {
            "RXNCONSO.RRF": _rrf_file_exists(path, "RXNCONSO.RRF"),
            "RXNREL.RRF": _rrf_file_exists(path, "RXNREL.RRF"),
            "RXNSAT.RRF": _rrf_file_exists(path, "RXNSAT.RRF"),
        },
        "rxnconso": _profile_rxnconso(path, allowed_ttys=allowed),
        "rxnrel": _profile_rxnrel(path),
        "rxnsat": _profile_rxnsat(path),
    }


def _iter_rxnconso_lines(path: str | Path) -> Iterator[str]:
    yield from _iter_rrf_lines(path, "RXNCONSO.RRF")


def _iter_rrf_lines(path: str | Path, target_name: str) -> Iterator[str]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            for name in sorted(archive.namelist()):
                if Path(name).name.upper() == target_name.upper():
                    with archive.open(name) as handle:
                        for raw_line in handle:
                            yield raw_line.decode("utf-8", errors="replace").rstrip("\n")
        return
    with input_path.open("r", encoding="utf-8", errors="replace") as handle:
        yield from handle


def _rrf_file_exists(path: str | Path, target_name: str) -> bool:
    input_path = Path(path)
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            return any(Path(name).name.upper() == target_name.upper() for name in archive.namelist())
    return input_path.name.upper() == target_name.upper() and input_path.exists()


def _profile_rxnconso(path: str | Path, *, allowed_ttys: set[str]) -> dict[str, Any]:
    total_rows = 0
    malformed_rows = 0
    rxnorm_rows = 0
    active_rxnorm_rows = 0
    accepted_rows = 0
    active_rxcuis: set[str] = set()
    accepted_rxcuis: set[str] = set()
    tty_counts: Counter[str] = Counter()
    suppress_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    for line in _iter_rxnconso_lines(path):
        total_rows += 1
        fields = line.rstrip("\n").split("|")
        if len(fields) < 18:
            malformed_rows += 1
            continue
        rxcui = fields[0].strip()
        lat = fields[1].strip()
        sab = fields[11].strip().upper()
        tty = fields[12].strip().upper()
        suppress = fields[16].strip().upper() or "<blank>"
        language_counts[lat or "<blank>"] += 1
        if sab != "RXNORM":
            continue
        rxnorm_rows += 1
        tty_counts[tty or "<blank>"] += 1
        suppress_counts[suppress] += 1
        if rxcui and suppress not in {"Y", "O"}:
            active_rxnorm_rows += 1
            active_rxcuis.add(rxcui)
        if rxcui and lat == "ENG" and tty in allowed_ttys and suppress not in {"Y", "O"}:
            accepted_rows += 1
            accepted_rxcuis.add(rxcui)
    return {
        "total_rows": total_rows,
        "malformed_rows": malformed_rows,
        "rxnorm_rows": rxnorm_rows,
        "active_rxnorm_rows": active_rxnorm_rows,
        "active_concepts": len(active_rxcuis),
        "accepted_term_rows": accepted_rows,
        "accepted_concepts": len(accepted_rxcuis),
        "tty_counts": _counter_dict(tty_counts),
        "suppress_counts": _counter_dict(suppress_counts),
        "language_counts": _counter_dict(language_counts),
    }


def _profile_rxnrel(path: str | Path) -> dict[str, Any]:
    total_rows = 0
    malformed_rows = 0
    rxnorm_rows = 0
    active_rows = 0
    rel_counts: Counter[str] = Counter()
    rela_counts: Counter[str] = Counter()
    suppress_counts: Counter[str] = Counter()
    for line in _iter_rrf_lines(path, "RXNREL.RRF"):
        total_rows += 1
        fields = line.rstrip("\n").split("|")
        if len(fields) < 16:
            malformed_rows += 1
            continue
        sab = fields[10].strip().upper()
        if sab != "RXNORM":
            continue
        rxnorm_rows += 1
        rel = fields[3].strip().upper() or "<blank>"
        rela = fields[7].strip().upper() or "<blank>"
        suppress = fields[14].strip().upper() or "<blank>"
        rel_counts[rel] += 1
        rela_counts[rela] += 1
        suppress_counts[suppress] += 1
        if suppress not in {"Y", "O"}:
            active_rows += 1
    return {
        "total_rows": total_rows,
        "malformed_rows": malformed_rows,
        "rxnorm_rows": rxnorm_rows,
        "active_rows": active_rows,
        "rel_counts": _counter_dict(rel_counts),
        "rela_counts": _counter_dict(rela_counts),
        "suppress_counts": _counter_dict(suppress_counts),
    }


def _profile_rxnsat(path: str | Path) -> dict[str, Any]:
    total_rows = 0
    malformed_rows = 0
    rxnorm_rows = 0
    active_rows = 0
    attr_counts: Counter[str] = Counter()
    suppress_counts: Counter[str] = Counter()
    status_attr_counts: Counter[str] = Counter()
    for line in _iter_rrf_lines(path, "RXNSAT.RRF"):
        total_rows += 1
        fields = line.rstrip("\n").split("|")
        if len(fields) < 13:
            malformed_rows += 1
            continue
        sab = fields[9].strip().upper()
        if sab != "RXNORM":
            continue
        rxnorm_rows += 1
        attr = fields[8].strip().upper() or "<blank>"
        suppress = fields[11].strip().upper() or "<blank>"
        attr_counts[attr] += 1
        suppress_counts[suppress] += 1
        if attr in {"RXN_ACTIVATED", "RXN_OBSOLETED", "RXN_QUANTITY", "TTY"}:
            status_attr_counts[attr] += 1
        if suppress not in {"Y", "O"}:
            active_rows += 1
    return {
        "total_rows": total_rows,
        "malformed_rows": malformed_rows,
        "rxnorm_rows": rxnorm_rows,
        "active_rows": active_rows,
        "attribute_counts": _counter_dict(attr_counts),
        "status_attribute_counts": _counter_dict(status_attr_counts),
        "suppress_counts": _counter_dict(suppress_counts),
    }


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


def _parse_rxnrel_line(line: str, *, source_id: str) -> RxNormSourceRelation | None:
    fields = line.rstrip("\n").split("|")
    if len(fields) < 16:
        return None
    source_rxcui = fields[0].strip()
    target_rxcui = fields[4].strip()
    rel = fields[3].strip().upper()
    rela = fields[7].strip().upper()
    sab = fields[10].strip().upper()
    suppress = fields[14].strip().upper()
    if not source_rxcui or not target_rxcui or sab != "RXNORM" or suppress in {"Y", "O"}:
        return None
    return RxNormSourceRelation(
        source_rxcui=source_rxcui,
        target_rxcui=target_rxcui,
        rel=rel,
        rela=rela,
        source_id=source_id,
    )


def _parse_rxnsat_line(line: str, *, source_id: str) -> RxNormSourceAttribute | None:
    fields = line.rstrip("\n").split("|")
    if len(fields) < 13:
        return None
    rxcui = fields[0].strip()
    attribute = fields[8].strip().upper()
    sab = fields[9].strip().upper()
    value = fields[10].strip()
    suppress = fields[11].strip().upper()
    if not rxcui or not attribute or not value or sab != "RXNORM" or suppress in {"Y", "O"}:
        return None
    return RxNormSourceAttribute(rxcui=rxcui, attribute=attribute, value=value, source_id=source_id)


def _rxnorm_name_index(
    terms: Iterable[RxNormSourceTerm],
    source_priority: Sequence[str],
) -> dict[str, str]:
    grouped: dict[str, list[RxNormSourceTerm]] = defaultdict(list)
    for term in terms:
        grouped[term.rxcui].append(term)
    return {
        rxcui: sorted(grouped_terms, key=lambda term: _rank_term(term, source_priority))[0].name
        for rxcui, grouped_terms in grouped.items()
        if grouped_terms
    }


def _rxnorm_enrichment(
    relations: Iterable[RxNormSourceRelation],
    attributes: Iterable[RxNormSourceAttribute],
    name_index: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(list))
    for relation in relations:
        if relation.rela in _INGREDIENT_RELAS:
            _append_named_relation(rows, relation.target_rxcui, "ingredients", relation.source_rxcui, name_index)
        elif relation.rela in _INGREDIENT_INVERSE_RELAS:
            _append_named_relation(rows, relation.source_rxcui, "ingredients", relation.target_rxcui, name_index)
        elif relation.rela in _BRAND_RELAS:
            _append_named_relation(rows, relation.target_rxcui, "brand_names", relation.source_rxcui, name_index)
        elif relation.rela in _BRAND_INVERSE_RELAS:
            _append_named_relation(rows, relation.source_rxcui, "brand_names", relation.target_rxcui, name_index)
        elif relation.rela in _DOSE_FORM_RELAS:
            _append_named_relation(rows, relation.target_rxcui, "dose_forms", relation.source_rxcui, name_index)
        elif relation.rela in _DOSE_FORM_INVERSE_RELAS:
            _append_named_relation(rows, relation.source_rxcui, "dose_forms", relation.target_rxcui, name_index)
    for attribute in attributes:
        payload = rows[attribute.rxcui]
        if attribute.attribute in _STRENGTH_ATTRS:
            payload["strengths"].append(f"{attribute.attribute}={attribute.value}")
        elif attribute.attribute == "RXN_ACTIVATED":
            payload["rxnorm_activated"].append(attribute.value)
        elif attribute.attribute == "RXN_OBSOLETED":
            payload["rxnorm_obsoleted"].append(attribute.value)
        elif attribute.attribute == "RXN_HUMAN_DRUG":
            payload["rxnorm_human_drug"].append(attribute.value)
        elif attribute.attribute == "RXN_VET_DRUG":
            payload["rxnorm_vet_drug"].append(attribute.value)
    return {rxcui: {key: _unique(values) for key, values in payload.items()} for rxcui, payload in rows.items()}


def _append_named_relation(
    rows: dict[str, dict[str, list[str]]],
    rxcui: str,
    key: str,
    target_rxcui: str,
    name_index: Mapping[str, str],
) -> None:
    target_name = name_index.get(target_rxcui)
    if target_name:
        rows[rxcui][key].append(target_name)


def _apply_rxnorm_enrichment(row: dict[str, Any], enrichment: Mapping[str, Any]) -> None:
    for list_key, scalar_key in (
        ("ingredients", "ingredient"),
        ("brand_names", "brand_name"),
        ("dose_forms", "dose_form"),
        ("strengths", "strength"),
    ):
        values = _unique(str(value) for value in enrichment.get(list_key, []) if str(value).strip())
        if not values:
            continue
        row[list_key] = values
        if not row.get(scalar_key):
            row[scalar_key] = values[0]
    for key in ("rxnorm_activated", "rxnorm_obsoleted", "rxnorm_human_drug", "rxnorm_vet_drug"):
        values = _unique(str(value) for value in enrichment.get(key, []) if str(value).strip())
        if values:
            row[key] = values
    row["rxnorm_status"] = "inactive" if row.get("rxnorm_obsoleted") else "active"


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


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}
