"""Typed execution stages used by :mod:`medical_kg_nlp.pipeline.runner`.

Stages own transformations and validation for one concern.  They do not own expensive model or
repository lifecycles; those remain in ``PipelineRuntime``.
"""

from __future__ import annotations

from dataclasses import dataclass

from medical_kg_nlp.pipeline.ports import (
    AssertionClassifierPort,
    BatchAssertionClassifierPort,
    EntityExtractorPort,
)
from medical_kg_nlp.preprocessing.normalizer import NormalizationContract
from medical_kg_nlp.preprocessing.section_rules import split_sections
from medical_kg_nlp.preprocessing.sentence_splitter import split_sentences
from medical_kg_nlp.schema.annotation import (
    AssertionEvidence,
    AssertionFeatures,
    EntityAnnotation,
)
from medical_kg_nlp.schema.document import Section, Sentence
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.schema.validator import PredictionValidationIssue, PredictionValidator
from medical_kg_nlp.validation.profiles import (
    ProfiledValidationIssue,
    ValidationProfile,
    ValidationSeverity,
    apply_validation_profile,
)

__all__ = [
    "AssertionClassificationStage",
    "DocumentPreparationStage",
    "DocumentStructure",
    "EntityExtractionStage",
    "PredictionValidationResult",
    "PredictionValidationStage",
]


@dataclass(frozen=True)
class DocumentStructure:
    """Source-coordinate sections and sentences for one immutable document."""

    sections: tuple[Section, ...]
    sentences: tuple[Sentence, ...]


class DocumentPreparationStage:
    """Prepare source structure without changing the document text or offsets."""

    def diagnostics(
        self,
        source_text: str,
        contract: NormalizationContract,
    ) -> dict[str, int]:
        """Return lookup diagnostics while keeping normalized text out of later stages."""

        mapped_text = contract.prepare(source_text)
        return {
            "original_characters": len(mapped_text.original),
            "normalized_characters": len(mapped_text.normalized),
            "offset_map_entries": len(mapped_text.normalized_to_original),
            "source_coordinate_spans": 1,
            "normalized_text_used_downstream": 0,
        }

    def structure(self, source_text: str) -> DocumentStructure:
        """Detect sections and split sentences in source coordinates."""

        sections = tuple(split_sections(source_text))
        sentences: list[Sentence] = []
        for section in sections:
            sentences.extend(
                split_sentences(
                    section.text,
                    section_title=section.title,
                    base_offset=section.span[0],
                )
            )
        if not sentences:
            sentences.append(Sentence(span=(0, len(source_text)), text=source_text))
        return DocumentStructure(sections=sections, sentences=tuple(sentences))


@dataclass(frozen=True)
class EntityExtractionStage:
    """Run one injected entity extractor and enforce source-coordinate spans."""

    extractor: EntityExtractorPort

    def run(self, source_text: str) -> list[EntityAnnotation]:
        entities = self.extractor.extract(source_text)
        for entity in entities:
            # INVARIANT: a stage may enrich an entity but never move it off the raw source.
            entity.validate_offsets(source_text)
        return entities


@dataclass(frozen=True)
class AssertionClassificationStage:
    """Apply scalar or batch assertion ports to entities inside sentence scope."""

    classifier: AssertionClassifierPort | None

    def run(
        self,
        entities: list[EntityAnnotation],
        sentences: tuple[Sentence, ...],
        counters: dict[str, int],
    ) -> None:
        if self.classifier is None:
            raise RuntimeError("Assertion classifier component is unavailable.")
        classified_ids: set[str] = set()
        if isinstance(self.classifier, BatchAssertionClassifierPort):
            for sentence in sentences:
                sentence_entities = [
                    entity
                    for entity in entities
                    if sentence.span[0] <= entity.span[0] < entity.span[1] <= sentence.span[1]
                ]
                if not sentence_entities:
                    continue
                decisions, graph = self.classifier.classify_batch_with_graph(
                    sentence_entities,
                    sentence,
                )
                for entity in sentence_entities:
                    features, evidence = decisions[entity.id]
                    _apply_assertion_decision(entity, features, evidence, counters)
                    classified_ids.add(entity.id)
                counters["context_graph_targets"] = (
                    counters.get("context_graph_targets", 0) + len(graph.targets)
                )
                counters["context_graph_modifiers"] = (
                    counters.get("context_graph_modifiers", 0) + len(graph.modifiers)
                )
                counters["context_graph_edges"] = (
                    counters.get("context_graph_edges", 0) + len(graph.edges)
                )
        for entity in entities:
            if entity.id in classified_ids:
                continue
            features, evidence = self.classifier.classify_features_with_evidence(
                entity,
                _find_sentence(entity, sentences),
            )
            _apply_assertion_decision(entity, features, evidence, counters)
        counters["classified_entities"] = len(entities)
        counters["matched_rule_events"] = sum(
            count for key, count in counters.items() if key.startswith("rule_")
        )


@dataclass(frozen=True)
class PredictionValidationResult:
    """Profiled validation output consumed by the runner's trace stage."""

    issues: tuple[ProfiledValidationIssue, ...]
    errors: tuple[ProfiledValidationIssue, ...]


@dataclass(frozen=True)
class PredictionValidationStage:
    """Validate a published prediction against the active terminology release."""

    validator: PredictionValidator
    profile: ValidationProfile = ValidationProfile.CORE

    def run(
        self,
        prediction: ClinicalPrediction,
        source_text: str,
    ) -> PredictionValidationResult:
        issues: list[PredictionValidationIssue] = self.validator.validate_prediction(
            prediction,
            source_text=source_text,
        )
        profiled = tuple(apply_validation_profile(issues, self.profile))
        errors = tuple(
            issue for issue in profiled if issue.severity is ValidationSeverity.ERROR
        )
        return PredictionValidationResult(issues=profiled, errors=errors)


def _apply_assertion_decision(
    entity: EntityAnnotation,
    features: AssertionFeatures,
    evidence: tuple[AssertionEvidence, ...],
    counters: dict[str, int],
) -> None:
    entity.assertion_features = features
    entity.assertion = features.primary()
    entity.assertion_evidence = evidence
    for item in evidence:
        key = f"rule_{item.rule_id}"
        counters[key] = counters.get(key, 0) + 1


def _find_sentence(
    entity: EntityAnnotation,
    sentences: tuple[Sentence, ...],
) -> Sentence | None:
    for sentence in sentences:
        if sentence.span[0] <= entity.span[0] and entity.span[1] <= sentence.span[1]:
            return sentence
    return None
