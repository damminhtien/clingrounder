"""Conservative compositional boundary proposals for clinical mentions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from medical_kg_nlp.ner.contracts import RuleNerContext
from medical_kg_nlp.ner.proposal import EntityProposal
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = ["ClinicalBoundaryProposalExtractor"]


_SYMPTOM_PREFIX_RE = re.compile(
    r"(?P<expand>"
    r"(?:tăng[ \t]+tần[ \t]+suất|cảm[ \t]+giác|cơn|tăng)"
    r"[ \t]+"
    r")$",
    re.IGNORECASE | re.UNICODE,
)
_SYMPTOM_SUFFIX_RE = re.compile(
    r"^(?P<expand>"
    r"[ \t]+(?:"
    r"nhẹ(?:[ \t]+đến[ \t]+[<>]?\d+(?:[.,]\d+)?[ \t]*°?[ \t]*c)?|"
    r"dữ[ \t]+dội|nghiêm[ \t]+trọng|rõ[ \t]+rệt|thoáng[ \t]+qua|"
    r"gián[ \t]+đoạn|kéo[ \t]+dài|tái[ \t]+phát|trở[ \t]+lại|"
    r"toàn[ \t]+thân|hai[ \t]+bên|chi[ \t]+(?:dưới|trên)|"
    r"(?:bên[ \t]+)?(?:trái|phải)|dưới|nhiều"
    r")"
    r")",
    re.IGNORECASE | re.UNICODE,
)
_DISEASE_SUFFIX_RE = re.compile(
    r"^(?P<expand>"
    r"(?:[ \t]*,[ \t]*không[ \t]+đặc[ \t]+hiệu)|"
    r"(?:[ \t]+(?:nghiêm[ \t]+trọng|rõ[ \t]+rệt|tái[ \t]+phát|"
    r"kháng[ \t]+thuốc|nguyên[ \t]+phát|di[ \t]+lệch))"
    r")",
    re.IGNORECASE | re.UNICODE,
)
_COMPOUND_FAINTING_LEFT_RE = re.compile(
    r"(?P<expand>ngất[ \t]+)$",
    re.IGNORECASE | re.UNICODE,
)
_COMPOUND_FAINTING_RIGHT_RE = re.compile(
    r"^(?P<expand>[ \t]+xỉu)",
    re.IGNORECASE | re.UNICODE,
)


@dataclass(frozen=True, slots=True)
class ClinicalBoundaryProposalExtractor:
    """Expand adjacent clinical modifiers without mutating dictionary evidence.

    INVARIANT: expansions consume only spaces or tabs plus an allow-listed modifier. They never
    cross a newline or clause punctuation, and every emitted span remains in raw coordinates.
    """

    score_bonus: float = 0.04
    max_suffixes: int = 3

    def propose(
        self,
        source_text: str,
        context: RuleNerContext,
    ) -> tuple[EntityProposal, ...]:
        proposals: set[EntityProposal] = set()
        for foundation in context.foundation_proposals:
            entity_type = foundation.entity_type
            if entity_type not in {EntityType.DISEASE, EntityType.SYMPTOM}:
                continue
            expanded_span, rule_ids = self._expand(
                source_text,
                foundation.span,
                entity_type,
                context,
                foundation,
            )
            if expanded_span == foundation.span:
                continue
            proposal = EntityProposal(
                span=expanded_span,
                candidate_types=(entity_type,),
                source="clinical_boundary",
                score=min(0.98, foundation.score + self.score_bonus),
                evidence_ids=tuple(
                    sorted(
                        {
                            *foundation.evidence_ids,
                            *(f"clinical_boundary:{rule_id}" for rule_id in rule_ids),
                        }
                    )
                ),
                concept_ids=foundation.concept_ids,
                features=tuple(
                    sorted(
                        {
                            *foundation.features,
                            ("boundary_rules", ",".join(rule_ids)),
                            ("boundary_source", foundation.source),
                            *(
                                {("atomic_clinical_phrase", "true")}
                                if "symptom_compound_fainting" in rule_ids
                                else set()
                            ),
                        }
                    )
                ),
            )
            proposal.validate_offsets(source_text)
            proposals.add(proposal)
        return tuple(sorted(proposals, key=_proposal_order))

    def _expand(
        self,
        source_text: str,
        span: tuple[int, int],
        entity_type: EntityType,
        context: RuleNerContext,
        foundation: EntityProposal,
    ) -> tuple[tuple[int, int], tuple[str, ...]]:
        start, end = span
        rule_ids: list[str] = []
        mention = normalize_for_match(source_text[start:end])

        if entity_type == EntityType.SYMPTOM:
            prefix_match = _last_match(_SYMPTOM_PREFIX_RE, source_text, start)
            if mention == "xỉu":
                prefix_match = _prefer_longer(
                    prefix_match,
                    _last_match(_COMPOUND_FAINTING_LEFT_RE, source_text, start),
                )
            if prefix_match is not None:
                start = prefix_match.start("expand")
                rule_ids.append("symptom_prefix")

            if mention == "ngất":
                end, matched = _consume_suffix(
                    _COMPOUND_FAINTING_RIGHT_RE,
                    source_text,
                    end,
                )
                if matched:
                    rule_ids.append("symptom_compound_fainting")

            for _ in range(self.max_suffixes):
                new_end, matched = _consume_suffix(
                    _SYMPTOM_SUFFIX_RE,
                    source_text,
                    end,
                )
                if not matched:
                    break
                end = new_end
                rule_ids.append("symptom_suffix")

            structured_span = _structured_symptom_span(
                source_text,
                foundation,
                context,
            )
            if structured_span is not None:
                start, end = structured_span
                rule_ids.append("symptom_structured_region")

        if entity_type == EntityType.DISEASE:
            for _ in range(self.max_suffixes):
                new_end, matched = _consume_suffix(
                    _DISEASE_SUFFIX_RE,
                    source_text,
                    end,
                )
                if not matched:
                    break
                end = new_end
                rule_ids.append("disease_suffix")

        return (start, end), tuple(rule_ids)


def _last_match(
    pattern: re.Pattern[str],
    source_text: str,
    boundary: int,
) -> re.Match[str] | None:
    window_start = max(0, boundary - 48)
    matches = list(pattern.finditer(source_text, window_start, boundary))
    return matches[-1] if matches else None


def _prefer_longer(
    first: re.Match[str] | None,
    second: re.Match[str] | None,
) -> re.Match[str] | None:
    if first is None:
        return second
    if second is None:
        return first
    first_length = first.end("expand") - first.start("expand")
    second_length = second.end("expand") - second.start("expand")
    return second if second_length > first_length else first


def _consume_suffix(
    pattern: re.Pattern[str],
    source_text: str,
    boundary: int,
) -> tuple[int, bool]:
    window = source_text[boundary : min(len(source_text), boundary + 96)]
    match = pattern.match(window)
    if match is None:
        return boundary, False
    return boundary + match.end("expand"), True


def _structured_symptom_span(
    source_text: str,
    foundation: EntityProposal,
    context: RuleNerContext,
) -> tuple[int, int] | None:
    """Complete a reviewed short alias inside an already parsed structural region.

    A generic dictionary mention must never absorb its full line. Only contextual aliases carry
    evidence that the short surface is unsafe without structure. The enclosing list item or
    medication indication supplies a bounded phrase whose raw coordinates are already known.
    """

    if foundation.source != "contextual_alias":
        return None
    start, end = foundation.span
    for item in context.medication_items:
        indication = item.indication_span
        if indication is None or not (
            indication[0] <= start and end <= indication[1]
        ):
            continue
        return _trim_structured_span(source_text, indication)

    structure = context.structure
    if structure is None or not structure.starts_list_item(start):
        return None
    line = structure.line_at(start)
    if line is None:
        return None
    return _trim_structured_span(source_text, (start, line.span[1]))


def _trim_structured_span(
    source_text: str,
    span: tuple[int, int],
) -> tuple[int, int] | None:
    """Trim layout punctuation and reject regions too broad for one clinical mention."""

    start, end = span
    while end > start and source_text[end - 1].isspace():
        end -= 1
    while end > start and source_text[end - 1] in ".:;":
        end -= 1
    while end > start and source_text[end - 1].isspace():
        end -= 1
    if end <= start:
        return None
    phrase = source_text[start:end]
    # SCALING: a hard local bound prevents malformed OCR/list lines from creating document-sized
    # proposals and keeps global interval resolution predictable.
    if len(phrase) > 96 or len(phrase.split()) > 18:
        return None
    return start, end


def _proposal_order(proposal: EntityProposal) -> tuple[int, int, str]:
    return proposal.span[0], proposal.span[1], proposal.candidate_types[0].value
