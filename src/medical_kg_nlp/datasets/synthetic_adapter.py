from __future__ import annotations
from pathlib import Path

from medical_kg_nlp.schema.annotation import (
    AssertionEvidence,
    AssertionFeatures,
    CandidateConcept,
    EntityAnnotation,
    MedicationComponent,
    MedicationMention,
    RelationAnnotation,
)
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.output import ClinicalPrediction, PredictionMetadata
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType, RelationType
from medical_kg_nlp.utils.io import read_jsonl


class SyntheticDatasetAdapter:
    def load_documents(self, path: str | Path) -> list[ClinicalDocument]:
        return [
            ClinicalDocument(
                document_id=str(row["document_id"]),
                text=str(row["text"]),
                metadata={k: str(v) for k, v in row.get("metadata", {}).items()},
            )
            for row in read_jsonl(path)
        ]

    def load_gold(self, path: str | Path) -> list[ClinicalPrediction]:
        predictions: list[ClinicalPrediction] = []
        for row in read_jsonl(path):
            entities = [
                EntityAnnotation(
                    id=str(entity["id"]),
                    span=(int(entity["span"][0]), int(entity["span"][1])),
                    text=str(entity["text"]),
                    normalized_text=str(entity.get("normalized_text", entity["text"])).lower(),
                    type=EntityType(entity["type"]),
                    assertion=AssertionStatus(entity["assertion"]),
                    code_system=CodeSystem(entity.get("code_system", "NONE")),
                    code=entity.get("code"),
                    confidence=float(entity.get("confidence", 1.0)),
                    candidates=[
                        CandidateConcept(
                            code_system=CodeSystem(candidate["code_system"]),
                            code=candidate.get("code"),
                            name=str(candidate["name"]),
                            retrieval_score=float(candidate["retrieval_score"]),
                            emit_probability=float(candidate["emit_probability"]),
                            concept_id=str(candidate["concept_id"]),
                            source=str(candidate["source"]),
                            evidence_sources=tuple(
                                str(source) for source in candidate["evidence_sources"]
                            ),
                            matched_alias=str(candidate["matched_alias"]),
                            qualified=_required_bool(candidate, "qualified"),
                            qualification_reason=str(candidate["qualification_reason"]),
                        )
                        for candidate in entity.get("candidates", [])
                    ],
                    assertion_features=_assertion_features(entity.get("assertion_features")),
                    assertion_evidence=tuple(
                        AssertionEvidence(
                            rule_id=str(item["rule_id"]),
                            assertion=AssertionStatus(item["assertion"]),
                            cue=str(item["cue"]),
                            scope=str(item["scope"]),
                        )
                        for item in entity["assertion_evidence"]
                    ),
                    medication_mention=_medication_mention(entity.get("medication_mention")),
                )
                for entity in row.get("entities", [])
            ]
            relations = [
                RelationAnnotation(
                    id=str(relation["id"]),
                    head=str(relation["head"]),
                    tail=str(relation["tail"]),
                    type=RelationType(relation["type"]),
                    confidence=float(relation.get("confidence", 1.0)),
                    evidence_span=tuple(relation["evidence_span"])
                    if "evidence_span" in relation
                    else None,
                )
                for relation in row.get("relations", [])
            ]
            predictions.append(
                ClinicalPrediction(
                    document_id=str(row["document_id"]),
                    text_hash=str(row.get("text_hash", "")),
                    entities=entities,
                    relations=relations,
                    metadata=PredictionMetadata(
                        pipeline_version=str(
                            row.get("metadata", {}).get("pipeline_version", "gold")
                        ),
                        created_at=str(
                            row.get("metadata", {}).get("created_at", "1970-01-01T00:00:00+00:00")
                        ),
                    ),
                )
            )
        return predictions


def _assertion_features(payload: object) -> AssertionFeatures:
    if not isinstance(payload, dict):
        return AssertionFeatures()
    return AssertionFeatures(
        negated=payload.get("negated") is True,
        historical=payload.get("historical") is True,
        family=payload.get("family") is True,
        possible=payload.get("possible") is True,
        conditional=payload.get("conditional") is True,
        planned=payload.get("planned") is True,
        resolved=payload.get("resolved") is True,
    )


def _required_bool(payload: dict[str, object], key: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _medication_mention(payload: object) -> MedicationMention | None:
    if not isinstance(payload, dict):
        return None
    drug_span = _two_int_span(payload.get("drug_span"))
    full_span = _two_int_span(payload.get("full_span"))
    components: list[MedicationComponent] = []
    raw_components = payload.get("components", [])
    if isinstance(raw_components, list):
        for component in raw_components:
            if not isinstance(component, dict):
                continue
            span = _two_int_span(component.get("span"))
            kind = str(component.get("kind", ""))
            if span is not None and kind:
                components.append(MedicationComponent(kind=kind, span=span))
    if drug_span is None or full_span is None:
        return None
    return MedicationMention(
        drug_span=drug_span,
        full_span=full_span,
        components=tuple(components),
    )


def _two_int_span(payload: object) -> tuple[int, int] | None:
    if (
        not isinstance(payload, list | tuple)
        or len(payload) != 2
        or isinstance(payload[0], bool)
        or isinstance(payload[1], bool)
        or not isinstance(payload[0], int)
        or not isinstance(payload[1], int)
    ):
        return None
    return payload[0], payload[1]
