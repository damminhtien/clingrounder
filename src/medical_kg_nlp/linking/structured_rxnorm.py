from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.utils.text import normalize_for_match


StrengthRole = Literal["mention", "product"]

_STRENGTH_RE = re.compile(
    r"(?<!\w)(?P<value>(?:\d+(?:[.,]\d+)?|[.,]\d+))"
    r"(?P<range>\s*-\s*(?:\d+(?:[.,]\d+)?|[.,]\d+))?\s*"
    r"(?P<unit>mcg|µg|μg|mg|g|ml|meq|iu|u)"
    r"(?:\s*/\s*(?:(?P<den_value>\d+(?:[.,]\d+)?)\s*)?"
    r"(?P<den_unit>ml|l|kg|day|ngày|dose|lần))?(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_FORM_CUES = {
    "capsule": ("capsule", "capsules", "viên nang"),
    "inhaler": ("inhaler", "nebs", "nebulizer", "khí dung"),
    "injection": ("injection", "injectable"),
    "solution": ("solution", "syrup", "dung dịch", "dịch truyền"),
    "suspension": ("suspension", "oral suspension", "hỗn dịch"),
    "tablet": ("tablet", "tablets", "viên nén", "viên"),
}
_ROUTE_CUES = {
    "oral": ("po", "p.o.", "oral", "uống", "đường uống"),
    "intravenous": ("iv", "i.v.", "tiêm tĩnh mạch", "truyền tĩnh mạch", "tĩnh mạch"),
    "intramuscular": ("im", "i.m.", "tiêm bắp"),
    "subcutaneous": ("sc", "s.c.", "tiêm dưới da"),
    "sublingual": ("sl", "s.l.", "ngậm dưới lưỡi"),
    "inhaled": ("hít", "xịt"),
    "topical": ("bôi", "dán", "nhỏ"),
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
        "xl",
    ),
    "immediate": ("immediate release", "giải phóng tức thì", "ir"),
}
_SIG_RE = re.compile(
    r"(?<!\w)(?:bid|tid|qid|qhs|qam|qd|daily|prn|mỗi\s+ngày|hằng\s+ngày|"
    r"hàng\s+ngày|mỗi\s+\d+\s*(?:giờ|phút))(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)
_ADMINISTRATION_RE = re.compile(
    r"(?<!\w)(?:dùng|uống|tiêm|truyền|bổ\s+sung|received|administered|given)(?!\w)",
    flags=re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True)
class MedicationStructure:
    product_strengths: frozenset[str]
    ambiguous_strengths: frozenset[str]
    administered_doses: frozenset[str]
    dose_forms: frozenset[str]
    routes: frozenset[str]
    release_types: frozenset[str]

    @property
    def strengths(self) -> frozenset[str]:
        """All numeric strengths, retained for diagnostics and existing metric code."""
        return self.product_strengths | self.ambiguous_strengths | self.administered_doses

    @property
    def has_product_evidence(self) -> bool:
        return bool(self.product_strengths or self.dose_forms or self.release_types)

    @property
    def structured(self) -> bool:
        return bool(self.strengths or self.dose_forms or self.routes or self.release_types)


def parse_medication_structure(
    text: str,
    *,
    strength_role: StrengthRole = "mention",
) -> MedicationStructure:
    """Parse medication attributes without treating route as dosage form.

    Mention doses are classified conservatively because a number in a SIG may be an administered
    dose rather than the strength of one manufactured product. Candidate terminology strings use
    ``strength_role='product'`` because RxNorm SCD/SBD strengths are product attributes.
    """
    normalized = normalize_for_match(text)
    dose_forms = _matched_cues(normalized, _FORM_CUES)
    routes = _matched_cues(normalized, _ROUTE_CUES)
    release_types = _matched_cues(normalized, _RELEASE_CUES)
    product: set[str] = set()
    ambiguous: set[str] = set()
    administered: set[str] = set()
    for match in _STRENGTH_RE.finditer(text):
        value = _normalized_strength_match(match)
        if strength_role == "product":
            product.add(value)
            continue
        role = _mention_strength_role(text, match, dose_forms, release_types)
        if role == "product":
            product.add(value)
        elif role == "administered":
            administered.add(value)
        else:
            ambiguous.add(value)
    return MedicationStructure(
        product_strengths=frozenset(product),
        ambiguous_strengths=frozenset(ambiguous),
        administered_doses=frozenset(administered),
        dose_forms=dose_forms,
        routes=routes,
        release_types=release_types,
    )


def parse_rxnorm_entry_structure(entry: ConceptEntry) -> MedicationStructure:
    candidate_text = " ".join(
        value
        for value in (*entry.all_names, entry.strength, entry.dose_form)
        if value
    )
    return parse_medication_structure(candidate_text, strength_role="product")


def rxnorm_structure_conflict(mention: str, entry: ConceptEntry) -> str | None:
    mention_structure = parse_medication_structure(mention)
    candidate_structure = parse_rxnorm_entry_structure(entry)
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
    # Only explicit product strength (normally amount + form/release) can hard-reject a product.
    # Route/frequency-only SIG doses remain ambiguous and are handled as soft ranking evidence.
    if (
        mention_structure.product_strengths
        and candidate_structure.product_strengths
        and mention_structure.product_strengths.isdisjoint(candidate_structure.product_strengths)
    ):
        return "rxnorm_product_strength_mismatch"
    return None


def _mention_strength_role(
    text: str,
    match: re.Match[str],
    dose_forms: frozenset[str],
    release_types: frozenset[str],
) -> Literal["product", "ambiguous", "administered"]:
    if match.group("range") or match.group("den_unit") in {"day", "ngày", "dose", "lần"}:
        return "administered"
    left = normalize_for_match(text[max(0, match.start() - 28) : match.start()])
    right = normalize_for_match(text[match.end() : min(len(text), match.end() + 32)])
    local = f"{left} {right}"
    local_forms = _matched_cues(local, _FORM_CUES)
    local_release = _matched_cues(local, _RELEASE_CUES)
    if local_forms or local_release:
        return "product"
    # A route/frequency directly after an amount describes what the patient receives even when
    # terse medication lists omit an administration verb (for example, "1.5 mg po qhs").
    if _SIG_RE.search(right) or _matched_cues(right, _ROUTE_CUES):
        return "administered"
    if _ADMINISTRATION_RE.search(left):
        return "administered"
    if dose_forms or release_types:
        return "product"
    return "ambiguous"


def _normalized_strength_match(match: re.Match[str]) -> str:
    numerator = _normalized_strength(match.group("value"), match.group("unit"))
    denominator_unit = match.group("den_unit")
    if denominator_unit is None:
        return numerator
    denominator_value = _normalized_number(match.group("den_value") or "1")
    return f"{numerator}/{denominator_value}{normalize_for_match(denominator_unit)}"


def _normalized_strength(value: str, unit: str) -> str:
    normalized_unit = unit.casefold().replace("µ", "u").replace("μ", "u")
    return f"{_normalized_number(value)}{normalized_unit}"


def _normalized_number(value: str) -> str:
    normalized = value.replace(",", ".").lstrip("0") or "0"
    if normalized.startswith("."):
        normalized = f"0{normalized}"
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


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
