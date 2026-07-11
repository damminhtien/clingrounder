from __future__ import annotations
from pathlib import Path

from medical_kg_nlp.schema.annotation import (
    AssertionFeatures,
    CandidateConcept,
    EntityAnnotation,
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
                            code_system=CodeSystem(candidate.get("code_system", "NONE")),
                            code=candidate.get("code"),
                            name=str(candidate.get("name", "")),
                            score=float(candidate.get("score", 0.0)),
                            concept_id=candidate.get("concept_id"),
                            source=candidate.get("source"),
                            matched_alias=candidate.get("matched_alias"),
                        )
                        for candidate in entity.get("candidates", [])
                    ],
                    assertion_features=_assertion_features(entity.get("assertion_features")),
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
