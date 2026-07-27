"""Public composition wrapper for proposal-first deterministic NER."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.ner.dictionary_matcher import DictionaryMatcher
from medical_kg_nlp.ner.extractors import (
    AnchoredLabProposalExtractor,
    ConcatenatedDrugProposalExtractor,
    DictionaryProposalExtractor,
    MedicationAttributeProposalExtractor,
    RegexLabProposalExtractor,
    StructuredLabProposalExtractor,
)
from medical_kg_nlp.ner.lab_observation_extractor import LabObservationExtractor
from medical_kg_nlp.ner.medication_attribute_extractor import MedicationAttributeExtractor
from medical_kg_nlp.ner.medication_list_parser import MedicationListParser
from medical_kg_nlp.ner.medication_mention_parser import MedicationMentionParser
from medical_kg_nlp.ner.rule_engine import RuleNerEngine, RuleNerEngineResult
from medical_kg_nlp.ner.span_resolver import EvidenceWeightedSpanResolver
from medical_kg_nlp.ner.type_resolver import ContextualEntityTypeResolver
from medical_kg_nlp.ontology.false_positive import (
    DEFAULT_FALSE_POSITIVE_PATH,
    load_false_positive_rules,
)
from medical_kg_nlp.schema.annotation import (
    AmbiguousEntityProposal,
    EntityAnnotation,
    EntityExtractionResult,
)
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = ["RuleBasedNER"]


class RuleBasedNER:
    """Deterministic NER facade backed by independent proposal extractors."""

    def __init__(
        self,
        store: DictionaryStore,
        *,
        false_positive_path: str | Path | None = DEFAULT_FALSE_POSITIVE_PATH,
        disease_symptom_fallback: Literal["disease", "abstain"] = "disease",
    ) -> None:
        matcher = DictionaryMatcher(store.aliases_for_ner())
        false_positive_rules = load_false_positive_rules(false_positive_path)
        type_resolver = ContextualEntityTypeResolver(
            disease_symptom_fallback=disease_symptom_fallback,
        )
        medication_lists = MedicationListParser()
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
            ),
            dependent_extractors=(
                MedicationAttributeProposalExtractor(MedicationAttributeExtractor()),
                AnchoredLabProposalExtractor(LabObservationExtractor()),
                RegexLabProposalExtractor(),
                StructuredLabProposalExtractor(),
            ),
            span_resolver=EvidenceWeightedSpanResolver(),
            medication_mentions=MedicationMentionParser(),
            medication_lists=medication_lists,
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
