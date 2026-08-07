"""Dictionary and malformed-boundary proposal sources."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from clingrounder.ner.contracts import RuleNerContext
from clingrounder.ner.dictionary_matcher import DictionaryMatch, DictionaryMatcher
from clingrounder.ner.proposal import EntityProposal
from clingrounder.ner.type_resolver import ContextualEntityTypeResolver
from clingrounder.ontology.false_positive import FalsePositiveRule
from clingrounder.schema.types import EntityType

__all__ = [
    "ConcatenatedDrugProposalExtractor",
    "DictionaryProposalExtractor",
]


_CONCATENATED_DRUG_LEFT_PREFIXES = ("dùng", "uống", "tiêm", "truyền")
_CONCATENATED_DRUG_RIGHT_SUFFIXES = ("đã", "và", "trong", "kéo", "iv", "oral", "and")


@dataclass(frozen=True, slots=True)
class DictionaryProposalExtractor:
    """Emit every contextualized dictionary span before overlap resolution."""

    matcher: DictionaryMatcher
    type_resolver: ContextualEntityTypeResolver
    false_positive_rules: tuple[FalsePositiveRule, ...] = ()

    def propose(
        self,
        source_text: str,
        context: RuleNerContext,
    ) -> tuple[EntityProposal, ...]:
        raw_matches = tuple(
            match
            for match in self.matcher.find_candidates(
                source_text,
                require_boundaries=True,
                min_alias_chars=2,
            )
            if not _blocked(
                match,
                source_text,
                self.false_positive_rules,
            )
        )
        matches_by_span: dict[tuple[int, int], list[DictionaryMatch]] = defaultdict(list)
        for match in raw_matches:
            matches_by_span[match.span].append(match)

        indication_spans = tuple(
            item.indication_span
            for item in context.medication_items
            if item.indication_span is not None
        )
        proposals: list[EntityProposal] = []
        for span, matches in matches_by_span.items():
            candidate_types = tuple(
                sorted(
                    {match.entry.semantic_type for match in matches},
                    key=lambda item: item.value,
                )
            )
            resolution = self.type_resolver.resolve(
                source_text,
                span,
                candidate_types,
                medication_indication_spans=indication_spans,
            )
            resolved_types = (
                (resolution.entity_type,)
                if resolution.entity_type is not None
                else candidate_types
            )
            relevant_matches = tuple(
                match
                for match in matches
                if resolution.entity_type is None
                or match.entry.semantic_type == resolution.entity_type
            )
            exact = any(match.match_kind == "exact" for match in relevant_matches)
            features = [("type_resolution", resolution.reason)]
            mention = source_text[span[0] : span[1]]
            if resolved_types == (EntityType.DRUG,) and any(
                separator in mention for separator in ("/", "+")
            ):
                features.append(("atomic_product", "true"))
            if (
                exact
                and len(resolved_types) == 1
                and resolved_types[0]
                in {EntityType.DISEASE, EntityType.DRUG, EntityType.LAB_TEST}
                and (nested_count := _nested_match_count(span, matches_by_span)) > 0
            ):
                # Train characterization shows exact long diagnosis/drug/test terms usually encode
                # one reviewed concept, while splitting them creates false entities. Symptoms are
                # excluded because coordinated symptoms often represent two valid mentions.
                features.append(("atomic_lexical_term_count", str(nested_count)))
            proposals.append(
                EntityProposal(
                    span=span,
                    candidate_types=resolved_types,
                    source="dictionary_exact" if exact else "dictionary_toneless",
                    score=0.78 if exact else 0.76,
                    evidence_ids=tuple(
                        sorted(
                            {
                                f"{match.match_kind}:{match.entry.concept_id}"
                                for match in relevant_matches
                            }
                        )
                    ),
                    concept_ids=tuple(
                        sorted({match.entry.concept_id for match in relevant_matches})
                    ),
                    features=tuple(sorted(features)),
                )
            )
        return tuple(sorted(proposals, key=_proposal_order))


@dataclass(frozen=True, slots=True)
class ConcatenatedDrugProposalExtractor:
    """Recover drug aliases joined to adjacent words without scanning the full alias tuple."""

    matcher: DictionaryMatcher
    false_positive_rules: tuple[FalsePositiveRule, ...] = ()

    def propose(
        self,
        source_text: str,
        context: RuleNerContext,
    ) -> tuple[EntityProposal, ...]:
        del context
        matches = tuple(
            match
            for match in self.matcher.find_candidates(
                source_text,
                require_boundaries=False,
                entity_types={EntityType.DRUG},
                min_alias_chars=4,
            )
            if not _blocked(match, source_text, self.false_positive_rules)
        )
        match_starts = {match.span[0] for match in matches}
        match_ends = {match.span[1] for match in matches}
        lowered = source_text.lower()
        proposals: list[EntityProposal] = []
        for match in matches:
            start, end = match.span
            left_concat = start in match_ends or _left_has_prefix(lowered, start)
            right_concat = end in match_starts or _right_has_suffix(lowered, end)
            left_boundary = (
                start == 0 or not source_text[start - 1].isalnum() or left_concat
            )
            right_boundary = (
                end == len(source_text) or not source_text[end].isalnum() or right_concat
            )
            if not (left_boundary and right_boundary and (left_concat or right_concat)):
                continue
            mention = source_text[start:end]
            features = (
                (("atomic_product", "true"),)
                if any(separator in mention for separator in ("/", "+"))
                else ()
            )
            proposals.append(
                EntityProposal(
                    span=match.span,
                    candidate_types=(EntityType.DRUG,),
                    source="concatenated_drug",
                    score=0.74,
                    evidence_ids=(f"concatenated:{match.entry.concept_id}",),
                    concept_ids=(match.entry.concept_id,),
                    features=features,
                )
            )
        return tuple(sorted(set(proposals), key=_proposal_order))


def _blocked(
    match: DictionaryMatch,
    source_text: str,
    rules: tuple[FalsePositiveRule, ...],
) -> bool:
    return any(rule.blocks(match.alias, source_text, match.span) for rule in rules)


def _left_has_prefix(lowered: str, start: int) -> bool:
    left = lowered[max(0, start - 32) : start]
    return left.endswith(_CONCATENATED_DRUG_LEFT_PREFIXES)


def _right_has_suffix(lowered: str, end: int) -> bool:
    right = lowered[end : end + 32]
    return right.startswith(_CONCATENATED_DRUG_RIGHT_SUFFIXES)


def _proposal_order(proposal: EntityProposal) -> tuple[int, int, str, str]:
    return (
        proposal.span[0],
        proposal.span[1],
        ",".join(item.value for item in proposal.candidate_types),
        proposal.source,
    )


def _nested_match_count(
    outer: tuple[int, int],
    matches_by_span: dict[tuple[int, int], list[DictionaryMatch]],
) -> int:
    return sum(
        inner != outer
        and outer[0] <= inner[0]
        and inner[1] <= outer[1]
        for inner in matches_by_span
    )
