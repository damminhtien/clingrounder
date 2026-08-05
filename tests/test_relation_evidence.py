from __future__ import annotations

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.evaluation.relation_slices import relation_slice_counts
from medical_kg_nlp.kg.ontology_reasoner import OntologyReasoner
from medical_kg_nlp.kg.validator import KGValidator
from medical_kg_nlp.relations.knowledge import KnownRelationRepository
from medical_kg_nlp.relations.rule_relations import RuleRelationExtractor
from medical_kg_nlp.schema.annotation import EntityAnnotation, RelationAnnotation
from medical_kg_nlp.schema.document import Sentence
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType, RelationType


def test_known_treatment_resource_is_code_based_and_versioned() -> None:
    repository = KnownRelationRepository("data/relations/known_treats.jsonl")

    relation = repository.find(
        CodeSystem.RXNORM,
        "29046",
        RelationType.TREATS,
        CodeSystem.ICD10,
        "I10",
    )

    assert relation is not None
    assert relation.source == "curated_seed"
    assert relation.source_version == "2026-08-05"
    assert relation.review_status == "reviewed"


def test_treatment_relation_uses_codes_not_mention_strings() -> None:
    drug = _entity("D", (0, 8), "unrelated", EntityType.DRUG, CodeSystem.RXNORM, "29046")
    disease = _entity("C", (20, 22), "condition", EntityType.DISEASE, CodeSystem.ICD10, "I10")

    relations = RuleRelationExtractor().extract([drug, disease], [])

    assert [(item.type, item.head, item.tail) for item in relations] == [
        (RelationType.TREATS, "D", "C")
    ]
    assert relations[0].evidence is not None
    assert relations[0].evidence.source == "terminology_ontology_backed"
    assert relations[0].evidence.provenance is not None


def test_same_sentence_proximity_alone_abstains() -> None:
    disease = _entity("D", (0, 7), "viêm phổi", EntityType.DISEASE)
    symptom = _entity("S", (20, 22), "ho", EntityType.SYMPTOM)
    sentence = Sentence((0, 22), "viêm phổi và ho")

    assert RuleRelationExtractor().extract([disease, symptom], [sentence]) == []


def test_explicit_clause_cue_emits_heuristic_relation_with_evidence() -> None:
    disease = _entity("D", (0, 7), "viêm phổi", EntityType.DISEASE)
    symptom = _entity("S", (14, 16), "ho", EntityType.SYMPTOM)
    sentence = Sentence((0, 16), "viêm phổi gây ho")

    relations = RuleRelationExtractor().extract([disease, symptom], [sentence])

    assert len(relations) == 1
    assert relations[0].type == RelationType.HAS_SYMPTOM
    assert relations[0].evidence is not None
    assert relations[0].evidence.source == "sentence_cooccurrence_proposal"
    assert relations[0].evidence.support_score == 0.75


def test_negated_semantic_entity_is_not_related() -> None:
    disease = _entity("D", (0, 7), "viêm phổi", EntityType.DISEASE)
    symptom = _entity(
        "S",
        (14, 16),
        "ho",
        EntityType.SYMPTOM,
        assertion=AssertionStatus.NEGATED,
    )
    sentence = Sentence((0, 16), "viêm phổi gây ho")

    assert RuleRelationExtractor().extract([disease, symptom], [sentence]) == []


def test_relation_serializes_nested_evidence() -> None:
    relation = RuleRelationExtractor().extract(
        [
            _entity("D", (0, 7), "viêm phổi", EntityType.DISEASE),
            _entity("S", (14, 16), "ho", EntityType.SYMPTOM),
        ],
        [Sentence((0, 16), "viêm phổi gây ho")],
    )[0]

    payload = relation.to_json()
    assert payload["evidence"]["source"] == "sentence_cooccurrence_proposal"
    assert payload["evidence"]["evidence_span"] == [0, 16]


def test_validator_reports_unknown_relation_endpoint_membership() -> None:
    store = DictionaryStore.from_jsonl("data/dictionaries/seed_concepts.jsonl")
    drug = _entity("D", (0, 1), "drug", EntityType.DRUG, CodeSystem.RXNORM, "unknown")
    disease = _entity("C", (2, 3), "condition", EntityType.DISEASE, CodeSystem.ICD10, "I10")
    relation = RelationAnnotation("R", "D", "C", RelationType.TREATS, 1.0)

    valid, issues = KGValidator(OntologyReasoner(store)).validate_relations(
        [drug, disease], [relation]
    )

    assert valid == []
    assert [issue.kind for issue in issues] == ["unknown_ontology_membership"]


def test_relation_slice_report_exposes_scope_source_and_domain() -> None:
    drug = _entity("D", (0, 3), "drug", EntityType.DRUG)
    strength = _entity("S", (4, 8), "500mg", EntityType.STRENGTH)
    relation = RuleRelationExtractor().extract(
        [drug, strength], [Sentence((0, 8), "drug 500mg")]
    )[0]

    report = relation_slice_counts([relation], [drug, strength], [Sentence((0, 8), "drug 500mg")])

    assert report["relation_type"] == {"HAS_DOSE": 1}
    assert report["scope"] == {"same_clause": 1}
    assert report["domain"] == {"medication_list": 1}
    assert report["source"] == {"structural_medication_attribute": 1}


def _entity(
    entity_id: str,
    span: tuple[int, int],
    text: str,
    entity_type: EntityType,
    code_system: CodeSystem = CodeSystem.NONE,
    code: str | None = None,
    *,
    assertion: AssertionStatus = AssertionStatus.PRESENT,
) -> EntityAnnotation:
    return EntityAnnotation(
        id=entity_id,
        span=span,
        text=text,
        normalized_text=text.casefold(),
        type=entity_type,
        assertion=assertion,
        code_system=code_system,
        code=code,
    )
