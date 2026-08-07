"""Public composition wrapper for proposal-first deterministic NER."""

from __future__ import annotations

from pathlib import Path

from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.ner.dictionary_matcher import DictionaryMatcher
from clingrounder.ner.document_structure import DocumentStructureAnalyzer
from clingrounder.ner.extractors import (
    AnchoredLabProposalExtractor,
    ClinicalBoundaryProposalExtractor,
    ConcatenatedDrugProposalExtractor,
    ContextualAliasProposalExtractor,
    DictionaryProposalExtractor,
    MedicationAttributeProposalExtractor,
    RegexLabProposalExtractor,
    StructuredLabProposalExtractor,
)
from clingrounder.ner.extractors.contextual_alias import ContextualAliasRule
from clingrounder.ner.lab_observation_extractor import LabObservationExtractor
from clingrounder.ner.medication_attribute_extractor import MedicationAttributeExtractor
from clingrounder.ner.medication_list_parser import MedicationListParser
from clingrounder.ner.medication_mention_parser import MedicationMentionParser
from clingrounder.ner.rule_engine import RuleNerEngine, RuleNerEngineResult
from clingrounder.ner.span_resolver import EvidenceWeightedSpanResolver
from clingrounder.ner.type_resolver import ContextualEntityTypeResolver
from clingrounder.ontology.false_positive import load_false_positive_rules
from clingrounder.schema.annotation import (
    AmbiguousEntityProposal,
    EntityAnnotation,
    EntityExtractionResult,
)
from clingrounder.utils.text import normalize_for_match

__all__ = ["RuleBasedNER"]


class RuleBasedNER:
    """Deterministic NER facade backed by independent proposal extractors."""

    def __init__(
        self,
        store: DictionaryStore,
        *,
        false_positive_path: str | Path | None = None,
        contextual_alias_rules: tuple[ContextualAliasRule, ...] = (),
    ) -> None:
        matcher = DictionaryMatcher(store.aliases_for_ner())
        # Benchmark-calibrated suppression tables are optional dependencies. Keeping the default
        # empty prevents local competition artifacts from changing a reusable pipeline silently.
        false_positive_rules = load_false_positive_rules(false_positive_path)
        type_resolver = ContextualEntityTypeResolver()
        medication_lists = MedicationListParser()
        contextual_aliases = (
            (ContextualAliasProposalExtractor(contextual_alias_rules),)
            if contextual_alias_rules
            else ()
        )
        self.engine = RuleNerEngine(
            foundation_extractors=(
                DictionaryProposalExtractor(
                    matcher=matcher,
                    type_resolver=type_resolver,
                    false_positive_rules=false_positive_rules,
                ),
                ConcatenatedDrugProposalExtractor(
                    matcher=matcher,
                    false_positive_rules=false_positive_rules,
                ),
                *contextual_aliases,
            ),
            dependent_extractors=(
                ClinicalBoundaryProposalExtractor(),
                MedicationAttributeProposalExtractor(MedicationAttributeExtractor()),
                AnchoredLabProposalExtractor(LabObservationExtractor()),
                RegexLabProposalExtractor(),
                StructuredLabProposalExtractor(),
            ),
            span_resolver=EvidenceWeightedSpanResolver(),
            medication_mentions=MedicationMentionParser(),
            medication_lists=medication_lists,
            document_structure=DocumentStructureAnalyzer(),
        )

    def extract(self, text: str) -> list[EntityAnnotation]:
        """Return resolved entities through the standard pipeline contract."""

        return list(self.engine.extract(text).entities)

    def extract_with_trace(self, text: str) -> RuleNerEngineResult:
        """Return resolved entities plus proposal and arbitration lineage."""

        return self.engine.extract(text)

    def extract_with_proposals(self, text: str) -> EntityExtractionResult:
        """Retain unresolved dictionary type evidence for hybrid arbitration."""

        result = self.engine.extract(text)
        ambiguous = tuple(
            AmbiguousEntityProposal(
                span=proposal.span,
                text=text[proposal.span[0] : proposal.span[1]],
                normalized_text=_normalized_text(text, proposal.span),
                candidate_types=proposal.candidate_types,
                concept_ids=proposal.concept_ids,
                confidence=proposal.score,
                source=proposal.source,
            )
            for proposal in result.unresolved_proposals
            if proposal.concept_ids
        )
        return EntityExtractionResult(
            entities=result.entities,
            ambiguous_proposals=ambiguous,
        )


def _normalized_text(text: str, span: tuple[int, int]) -> str:
    return normalize_for_match(text[span[0] : span[1]])
