"""Scenario-graph rendering with sentinel-based span projection."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass

from clingrounder.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    ConceptLink,
    MinedDocument,
    RedistributionPolicy,
    RelationProposal,
    ReviewStatus,
)

__all__ = [
    "MinimalPairGenerator",
    "RenderedScenario",
    "ScenarioEntity",
    "ScenarioGraph",
    "ScenarioRelation",
    "SentinelScenarioRenderer",
]

_PLACEHOLDER = re.compile(r"\{\{([a-zA-Z0-9_.-]+)\}\}")
_SENTINEL = re.compile(
    r"<<KG:([a-zA-Z0-9_.-]+)>>(?P<text>.*?)<</KG:\1>>", flags=re.DOTALL
)


@dataclass(frozen=True)
class ScenarioEntity:
    """Graph entity whose semantics remain fixed across surface paraphrases."""

    entity_id: str
    surface: str
    entity_type: str
    assertions: tuple[str, ...] = ()
    concepts: tuple[ConceptLink, ...] = ()


@dataclass(frozen=True)
class ScenarioRelation:
    """Typed graph edge rendered independently from textual relation guessing."""

    relation_id: str
    head_entity_id: str
    tail_entity_id: str
    relation_type: str


@dataclass(frozen=True)
class ScenarioGraph:
    """Canonical semantic input for a generated clinical scenario."""

    scenario_id: str
    entities: tuple[ScenarioEntity, ...]
    relations: tuple[ScenarioRelation, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        entity_ids = [entity.entity_id for entity in self.entities]
        if not self.scenario_id or not entity_ids or len(entity_ids) != len(set(entity_ids)):
            raise ValueError("Scenario requires an ID and unique entity IDs")
        known = set(entity_ids)
        for relation in self.relations:
            if relation.head_entity_id not in known or relation.tail_entity_id not in known:
                raise ValueError(f"Unknown relation endpoint in {relation.relation_id!r}")


@dataclass(frozen=True)
class RenderedScenario:
    """Synthetic document plus graph-derived entity and relation labels."""

    document: MinedDocument
    annotations: tuple[AnnotationProposal, ...]
    relations: tuple[RelationProposal, ...]
    sentinel_text: str


class SentinelScenarioRenderer:
    """Render templates and project sentinels after optional surface paraphrasing."""

    def render_template(
        self,
        graph: ScenarioGraph,
        template: str,
        *,
        note_type: str,
        language: str = "vi",
        source_artifact_id: str = "synthetic:scenario-graphs",
    ) -> RenderedScenario:
        entities = {entity.entity_id: entity for entity in graph.entities}

        def replace(match: re.Match[str]) -> str:
            entity_id = match.group(1)
            entity = entities.get(entity_id)
            if entity is None:
                raise ValueError(f"Template references unknown entity {entity_id!r}")
            return f"<<KG:{entity_id}>>{entity.surface}<</KG:{entity_id}>>"

        sentinel_text = _PLACEHOLDER.sub(replace, template)
        return self.project(
            graph,
            sentinel_text,
            note_type=note_type,
            language=language,
            source_artifact_id=source_artifact_id,
        )

    def project(
        self,
        graph: ScenarioGraph,
        sentinel_text: str,
        *,
        note_type: str,
        language: str = "vi",
        source_artifact_id: str = "synthetic:scenario-graphs",
    ) -> RenderedScenario:
        entities = {entity.entity_id: entity for entity in graph.entities}
        output: list[str] = []
        output_length = 0
        projected: list[tuple[ScenarioEntity, int, int, str, int]] = []
        cursor = 0
        occurrences: dict[str, int] = {}
        for match in _SENTINEL.finditer(sentinel_text):
            prefix = sentinel_text[cursor : match.start()]
            output.append(prefix)
            output_length += len(prefix)
            entity_id = match.group(1)
            entity = entities.get(entity_id)
            if entity is None:
                raise ValueError(f"Sentinel references unknown entity {entity_id!r}")
            surface = match.group("text")
            if not surface or "<<KG:" in surface:
                raise ValueError(f"Invalid or nested sentinel for entity {entity_id!r}")
            start = output_length
            output.append(surface)
            output_length += len(surface)
            end = start + len(surface)
            occurrence = occurrences.get(entity_id, 0)
            occurrences[entity_id] = occurrence + 1
            projected.append((entity, start, end, surface, occurrence))
            cursor = match.end()
        output.append(sentinel_text[cursor:])
        unmatched_text = "".join(output)
        if "<<KG:" in unmatched_text or "<</KG:" in unmatched_text:
            raise ValueError("Malformed sentinel remained after projection")
        missing = sorted(set(entities) - set(occurrences))
        if missing:
            raise ValueError(f"Scenario entities missing from rendered text: {', '.join(missing)}")
        text = unmatched_text

        document_identity = f"{graph.scenario_id}\0{note_type}\0{text}"
        document_id = f"synthetic:{hashlib.sha256(document_identity.encode()).hexdigest()[:24]}"
        document = MinedDocument(
            document_id=document_id,
            text=text,
            language=language,
            note_type=note_type,
            source_artifact_id=source_artifact_id,
            access_class=AccessClass.OPEN,
            redistribution=RedistributionPolicy.ALLOWED,
            hosted_processing_allowed=True,
            group_ids=(f"scenario:{graph.scenario_id}", f"template:{note_type}"),
            metadata={
                "origin": "synthetic",
                "scenario_id": graph.scenario_id,
                **dict(graph.metadata),
            },
        )
        annotations: list[AnnotationProposal] = []
        annotation_by_entity: dict[str, str] = {}
        for entity, start, end, surface, occurrence in projected:
            identity = f"{document_id}\0{entity.entity_id}\0{occurrence}"
            annotation = AnnotationProposal(
                annotation_id=f"synthetic:{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
                document_id=document_id,
                span=(start, end),
                text=surface,
                entity_type=entity.entity_type,
                assertions=entity.assertions,
                concepts=entity.concepts,
                confidence=1.0,
                layer=AnnotationLayer.SILVER,
                label_source="scenario_graph",
                labeler_id="scenario_renderer:v1",
                review_status=ReviewStatus.PROPOSED,
                metadata={"origin": "synthetic", "scenario_entity_id": entity.entity_id},
            )
            annotation.validate_offsets(document)
            annotations.append(annotation)
            annotation_by_entity.setdefault(entity.entity_id, annotation.annotation_id)

        relations = tuple(
            RelationProposal(
                relation_id=f"synthetic:{relation.relation_id}",
                document_id=document_id,
                head_annotation_id=annotation_by_entity[relation.head_entity_id],
                tail_annotation_id=annotation_by_entity[relation.tail_entity_id],
                relation_type=relation.relation_type,
                confidence=1.0,
                layer=AnnotationLayer.SILVER,
                label_source="scenario_graph",
                labeler_id="scenario_renderer:v1",
                review_status=ReviewStatus.PROPOSED,
                metadata={"origin": "synthetic"},
            )
            for relation in graph.relations
        )
        annotation_index = {item.annotation_id: item for item in annotations}
        for relation in relations:
            relation.validate(document, annotation_index)
        return RenderedScenario(document, tuple(annotations), relations, sentinel_text)


class MinimalPairGenerator:
    """Provide named high-risk scenarios without embedding task-specific document IDs."""

    def __init__(self) -> None:
        self._cases = _minimal_pair_cases()

    @property
    def case_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._cases))

    def build(self, case_name: str) -> tuple[ScenarioGraph, str, str]:
        try:
            return self._cases[case_name]
        except KeyError as error:
            raise KeyError(f"Unknown minimal-pair case {case_name!r}") from error


def _minimal_pair_cases() -> Mapping[str, tuple[ScenarioGraph, str, str]]:
    def entity(
        entity_id: str, surface: str, entity_type: str, *assertions: str
    ) -> ScenarioEntity:
        return ScenarioEntity(entity_id, surface, entity_type, tuple(assertions))

    return {
        "family_patient": (
            ScenarioGraph(
                "family_patient",
                (
                    entity("family", "đái tháo đường", "DISEASE", "FAMILY"),
                    entity("patient", "tăng huyết áp", "DISEASE", "PRESENT"),
                ),
            ),
            "Mẹ bệnh nhân có {{family}}. Bệnh nhân hiện mắc {{patient}}.",
            "progress_note",
        ),
        "negated_present": (
            ScenarioGraph(
                "negated_present",
                (
                    entity("absent", "đau ngực", "SYMPTOM", "NEGATED"),
                    entity("present", "khó thở", "SYMPTOM", "PRESENT"),
                ),
            ),
            "Bệnh nhân không {{absent}} nhưng hiện có {{present}}.",
            "progress_note",
        ),
        "possible_confirmed": (
            ScenarioGraph(
                "possible_confirmed",
                (
                    entity("possible", "viêm phổi", "DISEASE", "POSSIBLE"),
                    entity("confirmed", "cúm A", "DISEASE", "PRESENT"),
                ),
            ),
            "Theo dõi {{possible}}; PCR đã xác nhận {{confirmed}}.",
            "discharge_note",
        ),
        "historical_current": (
            ScenarioGraph(
                "historical_current",
                (
                    entity("history", "đột quỵ", "DISEASE", "HISTORICAL"),
                    entity("current", "yếu nửa người", "SYMPTOM", "PRESENT"),
                ),
            ),
            "Tiền sử {{history}}. Hiện tại bệnh nhân {{current}}.",
            "progress_note",
        ),
        "dose_lab_number": (
            ScenarioGraph(
                "dose_lab_number",
                (
                    entity("drug", "metformin 500 mg", "DRUG"),
                    entity("test", "glucose", "LAB_TEST"),
                    entity("result", "500 mg/dL", "LAB_RESULT"),
                ),
            ),
            "Dùng {{drug}}. Xét nghiệm {{test}}: {{result}}.",
            "medication_and_lab",
        ),
        "ambiguous_abbreviation": (
            ScenarioGraph(
                "ambiguous_abbreviation",
                (
                    entity("diagnosis", "MI", "DISEASE", "HISTORICAL"),
                    entity("drug", "Mg", "DRUG"),
                ),
            ),
            "Tiền sử {{diagnosis}}; đã bổ sung {{drug}} đường tĩnh mạch.",
            "progress_note",
        ),
        "repeated_mention": (
            ScenarioGraph(
                "repeated_mention", (entity("symptom", "đau bụng", "SYMPTOM"),)
            ),
            "Lý do vào viện: {{symptom}}. Hiện vẫn {{symptom}}.",
            "progress_note",
        ),
        "mixed_language": (
            ScenarioGraph(
                "mixed_language",
                (
                    entity("disease", "heart failure", "DISEASE", "HISTORICAL"),
                    entity("symptom", "khó thở", "SYMPTOM"),
                ),
            ),
            "History of {{disease}}; hôm nay bệnh nhân {{symptom}}.",
            "progress_note",
        ),
        "toneless": (
            ScenarioGraph("toneless", (entity("symptom", "dau nguc", "SYMPTOM"),)),
            "Benh nhan khai {{symptom}} hai gio nay.",
            "noisy_note",
        ),
        "ocr": (
            ScenarioGraph("ocr", (entity("disease", "viem ph0i", "DISEASE"),)),
            "Chan doan OCR: {{disease}}.",
            "ocr_note",
        ),
        "malformed_list": (
            ScenarioGraph(
                "malformed_list",
                (
                    entity("drug1", "aspirin 81mg", "DRUG"),
                    entity("drug2", "metoprolol 50mg", "DRUG"),
                ),
            ),
            "Thuốc:1.{{drug1}}2){{drug2}} qd",
            "medication_list",
        ),
    }
