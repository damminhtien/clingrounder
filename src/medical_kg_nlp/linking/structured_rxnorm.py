from __future__ import annotations

import re
from dataclasses import dataclass

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.utils.text import normalize_for_match


_STRENGTH_RE = re.compile(
    r"(?<!\w)(?P<value>(?:\d+(?:[.,]\d+)?|[.,]\d+))\s*"
    r"(?P<unit>mcg|µg|μg|mg|g|ml|meq|iu|u)(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_FORM_CUES = {
    "capsule": ("capsule", "viên nang"),
    "inhaler": ("inhaler", "hít", "nebs", "nebulizer", "khí dung"),
    "injection": ("injection", "injectable", "iv", "tiêm", "tĩnh mạch"),
    "solution": ("solution", "syrup", "dung dịch", "dịch truyền"),
    "tablet": ("tablet", "viên", "po", "oral", "uống"),
}
_RELEASE_CUES = {
    "extended": (
        "extended release",
        "sustained release",
        "controlled release",
        "giải phóng kéo dài",
        "xr",
        "er",
        "sr",
        "cr",
    ),
    "immediate": ("immediate release", "giải phóng tức thì", "ir"),
}


@dataclass(frozen=True)
class MedicationStructure:
    strengths: frozenset[str]
    dose_forms: frozenset[str]
    release_types: frozenset[str]

    @property
    def structured(self) -> bool:
        return bool(self.strengths or self.dose_forms or self.release_types)


def parse_medication_structure(text: str) -> MedicationStructure:
    normalized = normalize_for_match(text)
    strengths = frozenset(
        _normalized_strength(match.group("value"), match.group("unit"))
        for match in _STRENGTH_RE.finditer(text)
    )
    return MedicationStructure(
        strengths=strengths,
        dose_forms=_matched_cues(normalized, _FORM_CUES),
        release_types=_matched_cues(normalized, _RELEASE_CUES),
    )


def rxnorm_structure_conflict(
    mention: str,
    entry: ConceptEntry,
) -> str | None:
    mention_structure = parse_medication_structure(mention)
    candidate_text = " ".join(
        value
        for value in (
            *entry.all_names,
            entry.strength,
            entry.dose_form,
        )
        if value
    )
    candidate_structure = parse_medication_structure(candidate_text)
    if (
        mention_structure.strengths
        and candidate_structure.strengths
        and mention_structure.strengths.isdisjoint(candidate_structure.strengths)
    ):
        return "rxnorm_strength_mismatch"
    if (
        mention_structure.release_types
        and candidate_structure.release_types
        and mention_structure.release_types.isdisjoint(candidate_structure.release_types)
    ):
        return "rxnorm_release_mismatch"
    if (
        mention_structure.dose_forms
        and candidate_structure.dose_forms
        and mention_structure.dose_forms.isdisjoint(candidate_structure.dose_forms)
    ):
        return "rxnorm_dose_form_mismatch"
    return None


def _normalized_strength(value: str, unit: str) -> str:
    normalized_value = value.replace(",", ".").lstrip("0") or "0"
    if normalized_value.startswith("."):
        normalized_value = f"0{normalized_value}"
    normalized_value = normalized_value.rstrip("0").rstrip(".") if "." in normalized_value else normalized_value
    normalized_unit = unit.casefold().replace("µ", "u").replace("μ", "u")
    return f"{normalized_value}{normalized_unit}"


def _matched_cues(
    normalized: str,
    cues_by_value: dict[str, tuple[str, ...]],
) -> frozenset[str]:
    return frozenset(
        value
        for value, cues in cues_by_value.items()
        if any(
            re.search(rf"(?<!\w){re.escape(cue)}(?!\w)", normalized) is not None
            for cue in cues
        )
    )
