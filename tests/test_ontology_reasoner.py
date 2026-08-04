from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.kg.ontology_reasoner import OntologyReasoner
from medical_kg_nlp.kg.validator import KGValidator
from medical_kg_nlp.schema.annotation import EntityAnnotation, RelationAnnotation
from medical_kg_nlp.schema.types import CodeSystem, EntityType, RelationType


def test_ontology_reasoner_computes_transitive_is_a_path() -> None:
    store = _hierarchy_store()
    reasoner = OntologyReasoner(store)

    assert reasoner.is_a(CodeSystem.ICD10, "C", "A")
    assert reasoner.hierarchy_distance(CodeSystem.ICD10, "C", "A") == 2
    explanation = reasoner.explain_is_a(CodeSystem.ICD10, "C", "A")
    assert explanation is not None
    assert [node[1] for node in explanation.path] == ["C", "B", "A"]


def test_kg_validator_rejects_is_a_relation_that_contradicts_dictionary_hierarchy() -> None:
    store = _hierarchy_store()
    child = _entity("E1", "C")
    parent = _entity("E2", "A")
    valid_relation = RelationAnnotation("R1", "E1", "E2", RelationType.IS_A, 1.0)
    invalid_relation = RelationAnnotation("R2", "E2", "E1", RelationType.IS_A, 1.0)

    reasoner = OntologyReasoner(store)
    valid, valid_issues = KGValidator(reasoner).validate_relations(
        [child, parent], [valid_relation]
    )
    invalid, invalid_issues = KGValidator(reasoner).validate_relations(
        [child, parent], [invalid_relation]
    )

    assert valid == [valid_relation]
    assert valid_issues == []
    assert invalid == []
    assert [issue.kind for issue in invalid_issues] == ["invalid_relation"]


def _hierarchy_store() -> DictionaryStore:
    return DictionaryStore(
        [
            _entry("A", ()),
            _entry("B", ("A",)),
            _entry("C", ("B",)),
        ]
    )


def _entry(code: str, parents: tuple[str, ...]) -> ConceptEntry:
    return ConceptEntry(
        concept_id=f"ICD10:{code}",
        code=code,
        code_system=CodeSystem.ICD10,
        canonical_name=f"Disease {code}",
        semantic_type=EntityType.DISEASE,
        parents=parents,
    )


def _entity(entity_id: str, code: str) -> EntityAnnotation:
    return EntityAnnotation(
        id=entity_id,
        span=(0, 1),
        text=code,
        normalized_text=code.lower(),
        type=EntityType.DISEASE,
        code_system=CodeSystem.ICD10,
        code=code,
    )
