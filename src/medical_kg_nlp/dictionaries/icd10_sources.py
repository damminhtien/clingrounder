from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.io import read_jsonl, write_jsonl


ICD10_VN_TT06_2026_SOURCE_ID = "icd10_vn_tt06_2026"
WHO_ICD10_2019_SOURCE_ID = "who_icd10_2019"
CDC_ICD10CM_2026_SOURCE_ID = "cdc_icd10cm_2026"
ICD_KCB_VN_SOURCE_ID = "icd_kcb_vn"
ICD10_VN_TT06_2026_SOURCE = {
    "source_id": ICD10_VN_TT06_2026_SOURCE_ID,
    "source": "TT 06/2026/TT-BYT",
    "issued_date": "2026-04-02",
    "effective_date": "2026-07-01",
    "primary_file": "06-byt-kem.pdf",
}

_ICD10_CATEGORY_RE = re.compile(r"^[A-Z][0-9]{2}(?:\.[0-9A-Z]+)?$")
_CDC_LINE_RE = re.compile(r"^(?P<code>[A-TV-Z][0-9][0-9A-Z](?:\.?[0-9A-Z]{1,4})?)\s+(?P<name>.+)$")
_SPACE_RE = re.compile(r"\s+")
_FIELD_KEY_RE = re.compile(r"[^0-9a-z]+")
_ICD10_CODE_KEYS = frozenset(
    {
        "code",
        "icd_code",
        "icd10_code",
        "ma",
        "ma_benh",
        "ma_icd",
        "ma_icd10",
        "ma_icd_10",
    }
)
_ICD10_VI_NAME_KEYS = frozenset(
    {
        "official_name_vi",
        "title_vi",
        "name_vi",
        "ten",
        "ten_benh",
        "ten_benh_tat",
        "chan_doan",
        "diagnosis_vi",
        "mo_ta",
    }
)
_ICD10_EN_NAME_KEYS = frozenset(
    {
        "official_name_en",
        "title_en",
        "name_en",
        "english_name",
        "diagnosis_en",
    }
)
_ICD10_ALIAS_KEYS = frozenset({"alias", "aliases", "synonym", "synonyms", "ten_khac"})
_ICD10_PARENT_KEYS = frozenset({"parent", "parent_code", "ma_cha", "block", "chapter"})
_ICD10_CHAPTERS: tuple[tuple[str, str, str, str], ...] = (
    ("A00", "B99", "I", "Certain infectious and parasitic diseases"),
    ("C00", "D48", "II", "Neoplasms"),
    ("D50", "D89", "III", "Diseases of the blood and blood-forming organs and immune mechanism"),
    ("E00", "E90", "IV", "Endocrine, nutritional and metabolic diseases"),
    ("F00", "F99", "V", "Mental and behavioural disorders"),
    ("G00", "G99", "VI", "Diseases of the nervous system"),
    ("H00", "H59", "VII", "Diseases of the eye and adnexa"),
    ("H60", "H95", "VIII", "Diseases of the ear and mastoid process"),
    ("I00", "I99", "IX", "Diseases of the circulatory system"),
    ("J00", "J99", "X", "Diseases of the respiratory system"),
    ("K00", "K93", "XI", "Diseases of the digestive system"),
    ("L00", "L99", "XII", "Diseases of the skin and subcutaneous tissue"),
    ("M00", "M99", "XIII", "Diseases of the musculoskeletal system and connective tissue"),
    ("N00", "N99", "XIV", "Diseases of the genitourinary system"),
    ("O00", "O99", "XV", "Pregnancy, childbirth and the puerperium"),
    ("P00", "P96", "XVI", "Certain conditions originating in the perinatal period"),
    ("Q00", "Q99", "XVII", "Congenital malformations, deformations and chromosomal abnormalities"),
    ("R00", "R99", "XVIII", "Symptoms, signs and abnormal clinical and laboratory findings"),
    ("S00", "T98", "XIX", "Injury, poisoning and certain other consequences of external causes"),
    ("V01", "Y98", "XX", "External causes of morbidity and mortality"),
    ("Z00", "Z99", "XXI", "Factors influencing health status and contact with health services"),
    ("U00", "U99", "XXII", "Codes for special purposes"),
)


@dataclass(frozen=True)
class ICD10SourceConcept:
    code: str
    official_name_en: str
    source_id: str
    parent_code: str | None = None
    aliases: tuple[str, ...] = ()
    official_name_vi: str | None = None


@dataclass(frozen=True)
class ICD10AliasOverlay:
    code: str
    aliases: tuple[str, ...] = ()
    official_name_vi: str | None = None
    source_id: str = ICD_KCB_VN_SOURCE_ID


@dataclass
class _MergedICD10Concept:
    code: str
    canonical_name: str
    official_name_en: str
    source: str
    parent_code: str | None = None
    official_name_vi: str | None = None
    aliases: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    source_ids: set[str] = field(default_factory=set)


def parse_who_icd10_claml(path: str | Path) -> list[ICD10SourceConcept]:
    """Parse WHO ICD-10 ClaML XML or a ZIP containing ClaML XML files."""
    concepts: dict[str, ICD10SourceConcept] = {}
    for root in _iter_xml_roots(path):
        for element in root.iter():
            if _local_name(element.tag) != "Class":
                continue
            code = _format_icd10_code(str(element.attrib.get("code", "")).strip())
            if not _is_icd10_category_code(code):
                continue
            preferred = _rubric_labels(element, "preferred")
            if not preferred:
                continue
            aliases = tuple(_unique(_rubric_labels(element, "inclusion")))
            concepts[code] = ICD10SourceConcept(
                code=code,
                official_name_en=preferred[0],
                source_id=WHO_ICD10_2019_SOURCE_ID,
                parent_code=_super_class_code(element),
                aliases=aliases,
            )
    return sorted(concepts.values(), key=lambda concept: concept.code)


def parse_cdc_icd10cm_descriptions(path: str | Path) -> list[ICD10SourceConcept]:
    """Parse CDC ICD-10-CM code-description TXT files or ZIP releases."""
    concepts: dict[str, ICD10SourceConcept] = {}
    for line in _iter_text_lines(path):
        parsed = _parse_cdc_description_line(line)
        if parsed is None:
            continue
        code, name = parsed
        concepts[code] = ICD10SourceConcept(
            code=code,
            official_name_en=name,
            source_id=CDC_ICD10CM_2026_SOURCE_ID,
            parent_code=_parent_code(code),
        )
    return sorted(concepts.values(), key=lambda concept: concept.code)


def parse_cdc_icd10cm_tabular_xml(path: str | Path) -> list[ICD10SourceConcept]:
    """Parse CDC ICD-10-CM tabular XML files or ZIP releases."""
    concepts: dict[str, ICD10SourceConcept] = {}
    for root in _iter_xml_roots(path):
        for concept in _iter_cdc_diag_concepts(root):
            concepts[concept.code] = concept
    return sorted(concepts.values(), key=lambda concept: concept.code)


def parse_icd10_vn_tt06_table(path: str | Path) -> list[ICD10SourceConcept]:
    """Parse a local structured extract of TT 06/2026/TT-BYT ICD-10 rows.

    The official source is a PDF appendix. This parser intentionally accepts reviewed local
    extracts in JSON/JSONL/CSV/TSV form instead of trying to infer table structure from arbitrary
    PDFs. Required fields are an ICD-10 code and a Vietnamese disease name. Optional fields include
    English name, parent/block, and aliases.
    """
    concepts: dict[str, ICD10SourceConcept] = {}
    for row in _iter_structured_rows(path):
        code = _format_icd10_code(str(_row_value(row, _ICD10_CODE_KEYS) or ""))
        if not _is_icd10_category_code(code):
            continue
        official_name_vi = _optional_clean_string(_row_value(row, _ICD10_VI_NAME_KEYS))
        official_name_en = _optional_clean_string(_row_value(row, _ICD10_EN_NAME_KEYS))
        if official_name_vi is None:
            continue
        aliases = _structured_aliases(row)
        concepts[code] = ICD10SourceConcept(
            code=code,
            official_name_en=official_name_en or official_name_vi,
            official_name_vi=official_name_vi,
            source_id=ICD10_VN_TT06_2026_SOURCE_ID,
            parent_code=_optional_parent_code(_row_value(row, _ICD10_PARENT_KEYS)) or _parent_code(code),
            aliases=tuple(_unique(aliases)),
        )
    return sorted(concepts.values(), key=lambda concept: concept.code)


def load_icd10_vietnamese_overlays(path: str | Path) -> list[ICD10AliasOverlay]:
    """Load curated Vietnamese ICD labels/aliases.

    Supported row shapes:
    - existing runtime alias table rows with ``target_concept_id`` and ``alias``;
    - source-ingestion rows with ``code``, optional ``official_name_vi``, and ``aliases``.
    """
    overlays: dict[str, _OverlayBuilder] = {}
    for row in read_jsonl(path):
        semantic_type = row.get("semantic_type")
        if semantic_type is not None and str(semantic_type) != EntityType.DISEASE.value:
            continue
        code = _code_from_overlay_row(row)
        if code is None:
            continue
        builder = overlays.setdefault(code, _OverlayBuilder(code=code))
        builder.source_id = str(row.get("source", row.get("source_id", ICD_KCB_VN_SOURCE_ID)))
        official_name_vi = _optional_clean_string(row.get("official_name_vi"))
        if official_name_vi is not None:
            builder.official_name_vi = official_name_vi
        builder.aliases.extend(_overlay_aliases(row))
    return [
        ICD10AliasOverlay(
            code=code,
            aliases=tuple(_unique(builder.aliases)),
            official_name_vi=builder.official_name_vi,
            source_id=builder.source_id,
        )
        for code, builder in sorted(overlays.items())
    ]


def build_icd10_concept_rows(
    source_concepts: Iterable[ICD10SourceConcept],
    overlays: Iterable[ICD10AliasOverlay] = (),
    *,
    source_priority: Sequence[str] = (
        ICD10_VN_TT06_2026_SOURCE_ID,
        ICD_KCB_VN_SOURCE_ID,
        WHO_ICD10_2019_SOURCE_ID,
        CDC_ICD10CM_2026_SOURCE_ID,
    ),
) -> list[dict[str, Any]]:
    priority = {source_id: index for index, source_id in enumerate(source_priority)}
    merged: dict[str, _MergedICD10Concept] = {}
    for concept in sorted(
        source_concepts,
        key=lambda item: (priority.get(item.source_id, len(priority)), item.code),
    ):
        current = merged.get(concept.code)
        if current is None:
            current = _MergedICD10Concept(
                code=concept.code,
                canonical_name=concept.official_name_vi or concept.official_name_en,
                official_name_en=concept.official_name_en,
                official_name_vi=concept.official_name_vi,
                source=concept.source_id,
                parent_code=concept.parent_code,
            )
            merged[concept.code] = current
        else:
            if concept.official_name_en != current.official_name_en:
                current.synonyms.append(concept.official_name_en)
            if concept.official_name_vi is not None and current.official_name_vi is None:
                current.official_name_vi = concept.official_name_vi
            if current.parent_code is None:
                current.parent_code = concept.parent_code
        current.source_ids.add(concept.source_id)
        current.aliases.extend(concept.aliases)

    for overlay in overlays:
        current = merged.get(overlay.code)
        if current is None:
            continue
        current.source_ids.add(overlay.source_id)
        current.aliases.extend(overlay.aliases)
        if overlay.official_name_vi is not None:
            current.official_name_vi = overlay.official_name_vi
            current.aliases.append(overlay.official_name_vi)

    rows: list[dict[str, Any]] = []
    for code, merged_concept in sorted(merged.items()):
        parent_code = merged_concept.parent_code or _parent_code(code)
        parents = [parent_code] if parent_code else []
        chapter = icd10_chapter_for_code(code)
        rows.append(
            {
                "concept_id": f"ICD10:{code}",
                "code": code,
                "code_system": CodeSystem.ICD10.value,
                "canonical_name": merged_concept.canonical_name,
                "official_name_en": merged_concept.official_name_en,
                "official_name_vi": merged_concept.official_name_vi,
                "semantic_type": EntityType.DISEASE.value,
                "aliases": _unique(merged_concept.aliases),
                "synonyms": _unique(merged_concept.synonyms),
                "abbreviations": [],
                "parent_code": parent_code,
                "parents": parents,
                "icd10_chapter": chapter.get("chapter"),
                "icd10_chapter_range": chapter.get("range"),
                "icd10_chapter_name_en": chapter.get("name_en"),
                "icd10_block": _icd10_block_code(code, parent_code),
                "source": merged_concept.source,
                "source_ids": sorted(merged_concept.source_ids),
            }
        )
    return rows


def write_icd10_concept_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_jsonl(path, [dict(row) for row in rows])


def write_icd10_import_manifest(
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
        "source_policy": icd10_source_policy(),
        "source_parse_counts": [dict(count) for count in source_parse_counts],
        "source_inputs": list(source_inputs),
        "source_counts": dict(sorted(source_counts.items())),
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return manifest


def icd10_source_policy() -> dict[str, Any]:
    return {
        "primary_source": dict(ICD10_VN_TT06_2026_SOURCE),
        "lookup_source_ids": [ICD_KCB_VN_SOURCE_ID],
        "reference_source_ids": [WHO_ICD10_2019_SOURCE_ID],
        "non_primary_source_ids": [CDC_ICD10CM_2026_SOURCE_ID],
        "notes": (
            "For Phase 1, Vietnamese ICD-10 labels should come from TT 06/2026/TT-BYT "
            "or reviewed KCB extracts. CDC ICD-10-CM is not the primary source."
        ),
    }


def icd10_chapter_for_code(code: str) -> dict[str, str | None]:
    comparable = _icd10_comparable_code(code)
    for start, end, chapter, name in _ICD10_CHAPTERS:
        if _icd10_comparable_code(start) <= comparable <= _icd10_comparable_code(end):
            return {"chapter": chapter, "range": f"{start}-{end}", "name_en": name}
    return {"chapter": None, "range": None, "name_en": None}


@dataclass
class _OverlayBuilder:
    code: str
    aliases: list[str] = field(default_factory=list)
    official_name_vi: str | None = None
    source_id: str = ICD_KCB_VN_SOURCE_ID


def _iter_xml_roots(path: str | Path) -> Iterator[ET.Element]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            for name in sorted(archive.namelist()):
                if name.lower().endswith(".xml"):
                    yield ET.fromstring(archive.read(name))
        return
    yield ET.parse(input_path).getroot()


def _iter_text_lines(path: str | Path) -> Iterator[str]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            for name in sorted(archive.namelist()):
                lower_name = name.lower()
                if lower_name.endswith((".txt", ".csv")):
                    payload = archive.read(name).decode("utf-8-sig", errors="replace")
                    yield from payload.splitlines()
        return
    with input_path.open("r", encoding="utf-8-sig") as handle:
        yield from handle


def _iter_structured_rows(path: str | Path) -> Iterator[Mapping[str, Any]]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            for name in sorted(archive.namelist()):
                if not name.lower().endswith((".json", ".jsonl", ".csv", ".tsv", ".txt")):
                    continue
                text = archive.read(name).decode("utf-8-sig", errors="replace")
                yield from _structured_rows_from_text(name, text)
        return
    yield from _structured_rows_from_text(input_path.name, input_path.read_text(encoding="utf-8-sig"))


def _structured_rows_from_text(name: str, text: str) -> Iterator[Mapping[str, Any]]:
    lower_name = name.lower()
    if lower_name.endswith(".jsonl"):
        for line in text.splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row
        return
    if lower_name.endswith(".json"):
        payload = json.loads(text)
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    yield row
        elif isinstance(payload, dict):
            rows = payload.get("rows")
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        yield row
        return

    delimiter = "\t" if lower_name.endswith(".tsv") else _sniff_delimiter(text)
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    for row in reader:
        yield row


def _sniff_delimiter(text: str) -> str:
    sample = text[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return "\t" if "\t" in sample else ","


def _parse_cdc_description_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    if "\t" in stripped:
        parts = [part.strip() for part in stripped.split("\t") if part.strip()]
        if len(parts) >= 2:
            code = _format_icd10_code(parts[0])
            name = _clean_label(parts[1])
            if _is_icd10_category_code(code) and name:
                return code, name
    match = _CDC_LINE_RE.match(stripped)
    if match is None:
        return None
    code = _format_icd10_code(match.group("code"))
    name = _clean_label(match.group("name"))
    if not _is_icd10_category_code(code) or not name:
        return None
    return code, name


def _iter_cdc_diag_concepts(element: ET.Element, parent_code: str | None = None) -> Iterator[ICD10SourceConcept]:
    for child in element:
        if _local_name(child.tag) != "diag":
            yield from _iter_cdc_diag_concepts(child, parent_code)
            continue
        code = _format_icd10_code(_child_text(child, "name"))
        name = _clean_label(_child_text(child, "desc"))
        effective_parent = parent_code or _parent_code(code)
        if _is_icd10_category_code(code) and name:
            yield ICD10SourceConcept(
                code=code,
                official_name_en=name,
                source_id=CDC_ICD10CM_2026_SOURCE_ID,
                parent_code=effective_parent,
                aliases=tuple(_direct_note_texts(child, "inclusionTerm")),
            )
            yield from _iter_cdc_diag_concepts(child, code)


def _child_text(element: ET.Element, name: str) -> str:
    for child in element:
        if _local_name(child.tag) == name:
            return _clean_label(" ".join(child.itertext()))
    return ""


def _direct_note_texts(element: ET.Element, container_name: str) -> list[str]:
    notes: list[str] = []
    for child in element:
        if _local_name(child.tag) != container_name:
            continue
        for note in child:
            if _local_name(note.tag) == "note":
                text = _clean_label(" ".join(note.itertext()))
                if text:
                    notes.append(text)
    return _unique(notes)


def _rubric_labels(class_element: ET.Element, kind: str) -> list[str]:
    labels: list[str] = []
    for rubric in class_element:
        if _local_name(rubric.tag) != "Rubric" or rubric.attrib.get("kind") != kind:
            continue
        for child in rubric.iter():
            if _local_name(child.tag) == "Label":
                label = _clean_label(" ".join(child.itertext()))
                if label:
                    labels.append(label)
    return _unique(labels)


def _super_class_code(class_element: ET.Element) -> str | None:
    for child in class_element:
        if _local_name(child.tag) != "SuperClass":
            continue
        code = _format_icd10_code(str(child.attrib.get("code", "")).strip())
        return code or None
    return None


def _code_from_overlay_row(row: Mapping[str, Any]) -> str | None:
    raw_code = _optional_clean_string(row.get("code"))
    if raw_code is None:
        concept_id = _optional_clean_string(row.get("target_concept_id"))
        if concept_id is not None and concept_id.startswith("ICD10:"):
            raw_code = concept_id.removeprefix("ICD10:")
    if raw_code is None:
        return None
    code = _format_icd10_code(raw_code)
    return code if _is_icd10_category_code(code) else None


def _row_value(row: Mapping[str, Any], keys: set[str] | frozenset[str]) -> Any:
    normalized = {_normalize_field_key(str(key)): value for key, value in row.items()}
    for key in keys:
        value = normalized.get(key)
        if value not in (None, ""):
            return value
    return None


def _structured_aliases(row: Mapping[str, Any]) -> list[str]:
    aliases: list[str] = []
    normalized = {_normalize_field_key(str(key)): value for key, value in row.items()}
    for key in _ICD10_ALIAS_KEYS:
        value = normalized.get(key)
        aliases.extend(_alias_values(value))
    return [_clean_label(alias) for alias in aliases if _clean_label(alias)]


def _alias_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    return [part for part in re.split(r"[;|]", str(value)) if part.strip()]


def _normalize_field_key(key: str) -> str:
    decomposed = unicodedata.normalize("NFD", key)
    without_accents = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return _FIELD_KEY_RE.sub("_", without_accents.casefold()).strip("_")


def _overlay_aliases(row: Mapping[str, Any]) -> list[str]:
    aliases: list[str] = []
    alias = _optional_clean_string(row.get("alias"))
    if alias is not None:
        aliases.append(alias)
    canonical = _optional_clean_string(row.get("canonical"))
    if canonical is not None:
        aliases.append(canonical)
    for key in ("aliases", "synonyms", "abbreviations"):
        value = row.get(key)
        if isinstance(value, str):
            aliases.append(value)
        elif isinstance(value, list | tuple):
            aliases.extend(str(item) for item in value)
    return [_clean_label(alias) for alias in aliases if _clean_label(alias)]


def _optional_clean_string(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = _clean_label(str(value))
    return cleaned or None


def _optional_parent_code(value: Any) -> str | None:
    raw_value = _optional_clean_string(value)
    if raw_value is None:
        return None
    code = _format_icd10_code(raw_value)
    if _is_icd10_category_code(code) or "-" in code:
        return code
    return None


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


def _format_icd10_code(raw_code: str) -> str:
    code = raw_code.strip().upper().replace(" ", "")
    if not code or "-" in code or "." in code:
        return code
    if len(code) > 3:
        return f"{code[:3]}.{code[3:]}"
    return code


def _parent_code(code: str) -> str | None:
    if "." in code:
        return code.split(".", maxsplit=1)[0]
    return None


def _icd10_block_code(code: str, parent_code: str | None) -> str | None:
    if parent_code and "-" in parent_code:
        return parent_code
    if "." in code:
        return code.split(".", maxsplit=1)[0]
    return None


def _icd10_comparable_code(code: str) -> tuple[str, int]:
    cleaned = _format_icd10_code(code)
    if not cleaned:
        return ("", -1)
    letter = cleaned[0]
    digits = re.sub(r"\D", "", cleaned[1:])
    return (letter, int(digits[:2] or "0"))


def _is_icd10_category_code(code: str) -> bool:
    return bool(_ICD10_CATEGORY_RE.match(code))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _clean_label(label: str) -> str:
    return _SPACE_RE.sub(" ", label).strip()


def _unique(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_label(value)
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique
