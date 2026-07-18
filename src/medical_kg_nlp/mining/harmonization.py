"""Policy-driven alignment of source labels and codes to internal terminology.

Harmonization is a one-to-one annotation transform: text, raw span, document ID, and
annotation ID are immutable. A source code is promoted only when the pinned target
repository contains a type-compatible concept.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from medical_kg_nlp.mining.policy import MiningQualityGate
from medical_kg_nlp.mining.records import AnnotationProposal, ConceptLink, MinedDocument
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology.ports import TerminologyRepository

__all__ = [
    "AnnotationHarmonizationPolicy",
    "AnnotationHarmonizationRule",
    "AnnotationHarmonizationResult",
    "harmonize_annotations",
    "load_annotation_harmonization_policy",
]


class AnnotationHarmonizationRule(BaseModel):
    """One exact source-label mapping with terminology-backed code promotion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_entity_type: str = Field(min_length=1)
    source_label: str = Field(min_length=1)
    target_entity_type: EntityType
    concept_system_map: dict[str, CodeSystem] = Field(default_factory=dict)
    target_terminology_version: str = Field(min_length=1)
    unmapped_concept_action: Literal["preserve", "drop"] = "preserve"


class AnnotationHarmonizationPolicy(BaseModel):
    """Versioned collection of non-overlapping source-label mappings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["annotation-harmonization-policy.v1"]
    policy_id: str = Field(min_length=1)
    rules: tuple[AnnotationHarmonizationRule, ...]


@dataclass(frozen=True)
class AnnotationHarmonizationResult:
    """Harmonized annotations, row decisions, and aggregate mapping counts."""

    annotations: tuple[AnnotationProposal, ...]
    decisions: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def load_annotation_harmonization_policy(
    path: str | Path,
) -> AnnotationHarmonizationPolicy:
    """Load a strict policy and reject rules that could match the same annotation."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    policy = AnnotationHarmonizationPolicy.model_validate(raw)
    if not policy.rules:
        raise ValueError("Annotation harmonization policy requires at least one rule")
    match_keys = [
        (rule.source_id, rule.source_entity_type, rule.source_label)
        for rule in policy.rules
    ]
    if len(match_keys) != len(set(match_keys)):
        raise ValueError("Annotation harmonization rules must have unique match keys")
    rule_ids = [rule.rule_id for rule in policy.rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("Annotation harmonization rule IDs must be unique")
    return policy


def harmonize_annotations(
    documents: Sequence[MinedDocument],
    annotations: Sequence[AnnotationProposal],
    repository: TerminologyRepository,
    policy: AnnotationHarmonizationPolicy,
) -> AnnotationHarmonizationResult:
    """Align source labels while retaining every unsafe mapping as audit metadata."""

    issues = MiningQualityGate().validate(documents, annotations)
    if issues:
        raise ValueError("Cannot harmonize invalid annotations:\n" + "\n".join(issues))
    documents_by_id = {document.document_id: document for document in documents}
    rules = {
        (rule.source_id, rule.source_entity_type, rule.source_label): rule
        for rule in policy.rules
    }
    output: list[AnnotationProposal] = []
    decisions: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    for annotation in sorted(annotations, key=lambda item: item.annotation_id):
        document = documents_by_id[annotation.document_id]
        source_id = document.metadata.get(
            "source_id", document.source_artifact_id.split(":", 1)[0]
        )
        rule = rules.get(
            (source_id, annotation.entity_type, annotation.source_label or "")
        )
        if rule is None:
            output.append(annotation)
            counters["unchanged_no_rule"] += 1
            continue

        mapped_concepts: list[ConceptLink] = []
        dropped_concepts: list[dict[str, str]] = []
        mapped_count = 0
        for concept in annotation.concepts:
            target_system = rule.concept_system_map.get(concept.code_system)
            if target_system is None:
                mapped_concepts.append(concept)
                continue
            target = repository.get_by_code(target_system, concept.code)
            if target is not None and target.semantic_type is rule.target_entity_type:
                mapped_concepts.append(
                    ConceptLink(
                        code_system=target_system.value,
                        code=target.code or concept.code,
                        terminology_version=rule.target_terminology_version,
                        confidence=concept.confidence,
                    )
                )
                mapped_count += 1
                continue
            dropped_concepts.append(
                {
                    "code_system": concept.code_system,
                    "code": concept.code,
                    "terminology_version": concept.terminology_version,
                    "reason": (
                        "unknown_target_code"
                        if target is None
                        else "target_semantic_type_mismatch"
                    ),
                }
            )
            if rule.unmapped_concept_action == "preserve":
                mapped_concepts.append(concept)

        metadata = {
            **annotation.metadata,
            "harmonization_policy_id": policy.policy_id,
            "harmonization_rule_id": rule.rule_id,
            "harmonization_source_entity_type": annotation.entity_type,
        }
        if dropped_concepts:
            metadata["harmonization_unmapped_concepts"] = json.dumps(
                dropped_concepts,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        harmonized = replace(
            annotation,
            entity_type=rule.target_entity_type.value,
            concepts=tuple(mapped_concepts),
            metadata=metadata,
        )
        # INVARIANT: label harmonization cannot alter the raw character contract.
        harmonized.validate_offsets(document)
        output.append(harmonized)
        rule_counts[rule.rule_id] += 1
        counters["harmonized_annotations"] += 1
        counters["mapped_concepts"] += mapped_count
        counters["unmapped_concepts"] += len(dropped_concepts)
        decisions.append(
            {
                "annotation_id": annotation.annotation_id,
                "document_id": annotation.document_id,
                "rule_id": rule.rule_id,
                "source_entity_type": annotation.entity_type,
                "target_entity_type": harmonized.entity_type,
                "mapped_concept_count": mapped_count,
                "unmapped_concepts": dropped_concepts,
                "decision": "harmonized",
            }
        )

    ordered = tuple(sorted(output, key=lambda item: item.annotation_id))
    output_issues = MiningQualityGate().validate(documents, ordered)
    if output_issues:
        raise ValueError("Harmonization produced invalid annotations:\n" + "\n".join(output_issues))
    report = {
        "schema_version": "annotation-harmonization-report.v1",
        "policy_id": policy.policy_id,
        "input_annotation_count": len(annotations),
        "output_annotation_count": len(ordered),
        "decision_counts": dict(sorted(counters.items())),
        "rule_annotation_counts": dict(sorted(rule_counts.items())),
        "validation_issue_count": 0,
    }
    return AnnotationHarmonizationResult(
        annotations=ordered,
        decisions=tuple(decisions),
        report=report,
    )
