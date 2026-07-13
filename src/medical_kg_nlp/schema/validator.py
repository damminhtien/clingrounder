from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, overload

from medical_kg_nlp.kg.constraints import (
    code_system_valid_for_entity_type,
    entity_code_system_valid,
    relation_type_valid,
)
from medical_kg_nlp.schema.annotation import (
    MEDICATION_COMPONENT_KINDS,
    AssertionFeatures,
    CandidateConcept,
    EntityAnnotation,
    MedicationComponent,
    MedicationMention,
    RelationAnnotation,
)
from medical_kg_nlp.schema.output import ClinicalPrediction, PredictionMetadata
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType, RelationType
from medical_kg_nlp.utils.hashing import sha256_text


class _DictionaryEntry(Protocol):
    code_system: CodeSystem
    code: str | None


class _DictionaryStore(Protocol):
    entries: Sequence[_DictionaryEntry]


@dataclass(frozen=True)
class PredictionValidationIssue:
    kind: str
    path: str
    message: str

    def to_json(self) -> dict[str, str]:
        return {"kind": self.kind, "path": self.path, "message": self.message}


class PredictionValidator:
    def __init__(self, dictionary: _DictionaryStore | None = None) -> None:
        self._allowed_codes: set[tuple[CodeSystem, str]] = set()
        if dictionary is not None:
            self._allowed_codes = {
                (entry.code_system, entry.code)
                for entry in dictionary.entries
                if entry.code is not None
            }

    def validate_payload(
        self,
        payload: Mapping[str, Any],
        source_text: str | None = None,
    ) -> tuple[ClinicalPrediction | None, list[PredictionValidationIssue]]:
        try:
            prediction = prediction_from_json(payload)
        except ValueError as error:
            return None, [PredictionValidationIssue("schema", "$", str(error))]
        return prediction, self.validate_prediction(prediction, source_text)

    def validate_prediction(
        self,
        prediction: ClinicalPrediction,
        source_text: str | None = None,
    ) -> list[PredictionValidationIssue]:
        issues: list[PredictionValidationIssue] = []
        entity_ids: set[str] = set()

        if source_text is not None:
            expected_hash = sha256_text(source_text)
            if prediction.text_hash and prediction.text_hash != expected_hash:
                issues.append(
                    PredictionValidationIssue(
                        "text_hash_mismatch",
                        "$.text_hash",
                        "Prediction text_hash does not match the source document text.",
                    )
                )

        for index, entity in enumerate(prediction.entities):
            path = f"$.entities[{index}]"
            if entity.id in entity_ids:
                issues.append(
                    PredictionValidationIssue(
                        "duplicate_entity_id",
                        f"{path}.id",
                        f"Duplicate entity id {entity.id!r}.",
                    )
                )
            entity_ids.add(entity.id)

            if source_text is not None:
                try:
                    entity.validate_offsets(source_text)
                except ValueError as error:
                    issues.append(PredictionValidationIssue("offset", path, str(error)))

            if not entity_code_system_valid(entity):
                issues.append(
                    PredictionValidationIssue(
                        "invalid_code_system",
                        f"{path}.code_system",
                        f"{entity.type.value} cannot map to {entity.code_system.value}.",
                    )
                )

            if self._allowed_codes and entity.code is not None:
                if (entity.code_system, entity.code) not in self._allowed_codes:
                    issues.append(
                        PredictionValidationIssue(
                            "unknown_dictionary_code",
                            f"{path}.code",
                            (
                                f"{entity.code_system.value} code {entity.code!r} is not "
                                "present in the loaded dictionary."
                            ),
                        )
                    )

            for candidate_index, candidate in enumerate(entity.candidates):
                candidate_path = f"{path}.candidates[{candidate_index}]"
                if not code_system_valid_for_entity_type(entity.type, candidate.code_system):
                    issues.append(
                        PredictionValidationIssue(
                            "invalid_candidate_code_system",
                            f"{candidate_path}.code_system",
                            (
                                f"{entity.type.value} candidate cannot map to "
                                f"{candidate.code_system.value}."
                            ),
                        )
                    )
                if self._allowed_codes and candidate.code is not None:
                    if (candidate.code_system, candidate.code) not in self._allowed_codes:
                        issues.append(
                            PredictionValidationIssue(
                                "unknown_dictionary_code",
                                f"{candidate_path}.code",
                                (
                                    f"{candidate.code_system.value} candidate code "
                                    f"{candidate.code!r} is not present in the loaded dictionary."
                                ),
                            )
                        )

        entities_by_id = {entity.id: entity for entity in prediction.entities}
        relation_ids: set[str] = set()
        for index, relation in enumerate(prediction.relations):
            path = f"$.relations[{index}]"
            if relation.id in relation_ids:
                issues.append(
                    PredictionValidationIssue(
                        "duplicate_relation_id",
                        f"{path}.id",
                        f"Duplicate relation id {relation.id!r}.",
                    )
                )
            relation_ids.add(relation.id)

            if not relation_type_valid(relation, entities_by_id):
                issues.append(
                    PredictionValidationIssue(
                        "invalid_relation",
                        path,
                        (
                            f"Invalid {relation.type.value} relation between "
                            f"{relation.head!r} and {relation.tail!r}."
                        ),
                    )
                )

            if source_text is not None and relation.evidence_span is not None:
                start, end = relation.evidence_span
                if start < 0 or end < start or end > len(source_text):
                    issues.append(
                        PredictionValidationIssue(
                            "invalid_evidence_span",
                            f"{path}.evidence_span",
                            f"Invalid evidence span {relation.evidence_span}.",
                        )
                    )

        return issues


def prediction_from_json(payload: Mapping[str, Any]) -> ClinicalPrediction:
    entities = [
        _entity_from_json(entity, f"$.entities[{index}]")
        for index, entity in enumerate(_sequence(payload, "entities", "$.entities"))
    ]
    relations = [
        _relation_from_json(relation, f"$.relations[{index}]")
        for index, relation in enumerate(_sequence(payload, "relations", "$.relations"))
    ]
    metadata = _mapping(payload, "metadata", "$.metadata")
    return ClinicalPrediction(
        document_id=_string(payload, "document_id", "$.document_id"),
        text_hash=_string(payload, "text_hash", "$.text_hash"),
        entities=entities,
        relations=relations,
        metadata=PredictionMetadata(
            pipeline_version=_string(metadata, "pipeline_version", "$.metadata.pipeline_version"),
            created_at=_string(metadata, "created_at", "$.metadata.created_at"),
        ),
    )


def _entity_from_json(payload: Any, path: str) -> EntityAnnotation:
    row = _ensure_mapping(payload, path)
    return EntityAnnotation(
        id=_string(row, "id", f"{path}.id"),
        span=_span(row, "span", f"{path}.span"),
        text=_string(row, "text", f"{path}.text"),
        normalized_text=_string(row, "normalized_text", f"{path}.normalized_text"),
        type=_enum(EntityType, _string(row, "type", f"{path}.type"), f"{path}.type"),
        assertion=_enum(
            AssertionStatus,
            _string(row, "assertion", f"{path}.assertion"),
            f"{path}.assertion",
        ),
        code_system=_enum(
            CodeSystem,
            _string(row, "code_system", f"{path}.code_system"),
            f"{path}.code_system",
        ),
        code=_optional_string(row.get("code"), f"{path}.code"),
        confidence=_number(row, "confidence", f"{path}.confidence"),
        candidates=[
            _candidate_from_json(candidate, f"{path}.candidates[{index}]")
            for index, candidate in enumerate(
                _optional_sequence(row, "candidates", f"{path}.candidates")
            )
        ],
        assertion_features=_assertion_features(
            row.get("assertion_features"), f"{path}.assertion_features"
        ),
        medication_mention=_medication_mention(
            row.get("medication_mention"),
            f"{path}.medication_mention",
        ),
    )


def _candidate_from_json(payload: Any, path: str) -> CandidateConcept:
    row = _ensure_mapping(payload, path)
    return CandidateConcept(
        code_system=_enum(
            CodeSystem,
            _string(row, "code_system", f"{path}.code_system"),
            f"{path}.code_system",
        ),
        code=_optional_string(row.get("code"), f"{path}.code"),
        name=_string(row, "name", f"{path}.name"),
        score=_number(row, "score", f"{path}.score"),
        concept_id=_optional_string(row.get("concept_id"), f"{path}.concept_id"),
        source=_optional_string(row.get("source"), f"{path}.source"),
        matched_alias=_optional_string(row.get("matched_alias"), f"{path}.matched_alias"),
        qualified=_optional_bool(row, "qualified", f"{path}.qualified", default=False),
        qualification_reason=_optional_string(
            row.get("qualification_reason"), f"{path}.qualification_reason"
        ),
    )


def _assertion_features(payload: Any, path: str) -> AssertionFeatures:
    if payload is None:
        return AssertionFeatures()
    row = _ensure_mapping(payload, path)
    values: dict[str, bool] = {}
    for key in (
        "negated",
        "historical",
        "family",
        "possible",
        "conditional",
        "planned",
        "resolved",
    ):
        value = row.get(key, False)
        if not isinstance(value, bool):
            raise ValueError(f"{path}.{key}: expected boolean")
        values[key] = value
    return AssertionFeatures(**values)


def _medication_mention(payload: Any, path: str) -> MedicationMention | None:
    if payload is None:
        return None
    row = _ensure_mapping(payload, path)
    components: list[MedicationComponent] = []
    for index, component_payload in enumerate(_optional_sequence(row, "components", path)):
        component_path = f"{path}.components[{index}]"
        component = _ensure_mapping(component_payload, component_path)
        kind = _string(component, "kind", f"{component_path}.kind")
        if kind not in MEDICATION_COMPONENT_KINDS:
            raise ValueError(
                f"Unknown medication component kind {kind!r} at {component_path}.kind."
            )
        components.append(
            MedicationComponent(
                kind=kind,
                span=_span(component, "span", f"{component_path}.span"),
            )
        )
    return MedicationMention(
        drug_span=_span(row, "drug_span", f"{path}.drug_span"),
        full_span=_span(row, "full_span", f"{path}.full_span"),
        components=tuple(components),
    )


def _relation_from_json(payload: Any, path: str) -> RelationAnnotation:
    row = _ensure_mapping(payload, path)
    evidence_span = None
    if "evidence_span" in row and row["evidence_span"] is not None:
        evidence_span = _span(row, "evidence_span", f"{path}.evidence_span")
    return RelationAnnotation(
        id=_string(row, "id", f"{path}.id"),
        head=_string(row, "head", f"{path}.head"),
        tail=_string(row, "tail", f"{path}.tail"),
        type=_enum(RelationType, _string(row, "type", f"{path}.type"), f"{path}.type"),
        confidence=_number(row, "confidence", f"{path}.confidence"),
        evidence_span=evidence_span,
    )


def _mapping(payload: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    if key not in payload:
        raise ValueError(f"Missing required field {path}.")
    return _ensure_mapping(payload[key], path)


def _sequence(payload: Mapping[str, Any], key: str, path: str) -> Sequence[Any]:
    if key not in payload:
        raise ValueError(f"Missing required field {path}.")
    return _ensure_sequence(payload[key], path)


def _optional_sequence(payload: Mapping[str, Any], key: str, path: str) -> Sequence[Any]:
    if key not in payload or payload[key] is None:
        return []
    return _ensure_sequence(payload[key], path)


def _string(payload: Mapping[str, Any], key: str, path: str) -> str:
    if key not in payload:
        raise ValueError(f"Missing required field {path}.")
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f"Expected string at {path}.")
    return value


def _optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected string or null at {path}.")
    return value


def _optional_bool(
    payload: Mapping[str, Any], key: str, path: str, *, default: bool
) -> bool:
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean at {path}.")
    return value


def _number(payload: Mapping[str, Any], key: str, path: str) -> float:
    if key not in payload:
        raise ValueError(f"Missing required field {path}.")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected number at {path}.")
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"Expected value between 0 and 1 at {path}.")
    return score


def _span(payload: Mapping[str, Any], key: str, path: str) -> tuple[int, int]:
    if key not in payload:
        raise ValueError(f"Missing required field {path}.")
    value = payload[key]
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"Expected two-item span at {path}.")
    start, end = value
    if isinstance(start, bool) or isinstance(end, bool):
        raise ValueError(f"Expected integer span at {path}.")
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(f"Expected integer span at {path}.")
    if start < 0 or end < start:
        raise ValueError(f"Invalid span {value!r} at {path}.")
    return start, end


@overload
def _enum(enum_type: type[EntityType], value: str, path: str) -> EntityType: ...


@overload
def _enum(enum_type: type[AssertionStatus], value: str, path: str) -> AssertionStatus: ...


@overload
def _enum(enum_type: type[CodeSystem], value: str, path: str) -> CodeSystem: ...


@overload
def _enum(enum_type: type[RelationType], value: str, path: str) -> RelationType: ...


def _enum(
    enum_type: type[EntityType] | type[AssertionStatus] | type[CodeSystem] | type[RelationType],
    value: str,
    path: str,
) -> EntityType | AssertionStatus | CodeSystem | RelationType:
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"Unknown enum value {value!r} at {path}.") from error


def _ensure_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected object at {path}.")
    return value


def _ensure_sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Expected array at {path}.")
    return value
