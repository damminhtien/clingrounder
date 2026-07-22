from __future__ import annotations

import json
from pathlib import Path

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.benchmarks.phase1.phase1 import (
    load_phase1_text_documents,
    prediction_to_phase1_entities,
    validate_phase1_entities,
)
from medical_kg_nlp.ner.rule_ner import RuleBasedNER
from medical_kg_nlp.pipeline.factory import PipelineFactory, PipelineFactoryConfig
from medical_kg_nlp.retrieval.rule_factory import build_in_memory_retrieval_pipeline as _retrieval
from medical_kg_nlp.schema.types import AssertionStatus, CodeSystem, EntityType
from medical_kg_nlp.utils.io import read_source_text


FIXTURE = Path("tests/fixtures/phase1/btc_medication_list_crlf.txt")
EXPECTED = Path("tests/fixtures/phase1/btc_medication_list_expected.json")


def test_raw_reader_preserves_btc_crlf_offsets(tmp_path: Path) -> None:
    raw = FIXTURE.read_bytes()
    assert raw.count(b"\r\n") == 11
    assert len(read_source_text(FIXTURE)) == 554

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "1.txt").write_bytes(raw)
    document = load_phase1_text_documents(input_dir)[0]

    assert document.text == raw.decode("utf-8")
    assert len(document.text) == 554


def test_phase1_validator_allows_candidates_to_be_omitted_for_symptom() -> None:
    text = "ho"
    rows = [{"text": "ho", "type": "TRIỆU_CHỨNG", "assertions": [], "position": [0, 2]}]

    assert validate_phase1_entities(rows, text) == []


def test_phase1_validator_requires_candidates_for_codable_types() -> None:
    rows = [
        {"text": "aspirin", "type": "THUỐC", "assertions": [], "position": [0, 7]}
    ]

    issues = validate_phase1_entities(rows, "aspirin")

    assert [(issue.kind, issue.path) for issue in issues] == [
        ("phase1_schema", "$[0].candidates")
    ]


def test_btc_medication_list_boundaries_indications_and_assertions() -> None:
    text = read_source_text(FIXTURE)
    dictionary = _btc_recognition_store()
    entities = RuleBasedNER(dictionary).extract(text)
    runner = PipelineFactory.from_config()
    sections = runner._sections(text)
    sentences = runner._sentences_from_sections(sections, text)
    for entity in entities:
        sentence = runner._find_sentence(entity, sentences)
        classifier = runner.components.assertion_classifier
        assert classifier is not None
        entity.assertion_features, _ = classifier.classify_features_with_evidence(
            entity,
            sentence,
        )

    drugs = [entity for entity in entities if entity.type == EntityType.DRUG]
    symptoms = [entity for entity in entities if entity.type == EntityType.SYMPTOM]
    expected_spans = [
        "amlodipine 10 mg po daily",
        "aspirin 81 mg po daily",
        "metoprolol succinate xl 50 mg po daily",
        "guaifenesin ml po q6h:prn",
        "nystatin oral suspension 5 ml po qid:prn",
        "acetaminophen 325-650 mg po q6h:prn",
        "pravastatin 40 mg po daily",
        "docusate sodium 100 mg po bid",
        "senna 8.6 mg po bid:prn",
        "clonazepam 0.5 mg po qam:prn",
        "clonazepam 1.5 mg po qhs",
    ]
    assert len(drugs) == len(expected_spans)
    for entity, expected in zip(drugs, expected_spans, strict=True):
        assert entity.medication_mention is not None
        start, end = entity.medication_mention.full_span
        assert text[start:end] == expected
        assert AssertionStatus.HISTORICAL in entity.assertion_features.statuses()

    assert [(entity.text, entity.span) for entity in symptoms] == [
        ("ho", (196, 198)),
        ("đau nhức", (254, 262)),
        ("sốt đau", (313, 320)),
        ("táo bón", (397, 404)),
        ("táo bón", (443, 450)),
        ("lo âu", (495, 500)),
        ("lo âu", (541, 546)),
        ("mất ngủ", (547, 554)),
    ]
    assert all(entity.assertion_features.statuses() == () for entity in symptoms)


def test_btc_rxnorm_memory_is_dictionary_constrained() -> None:
    entry = ConceptEntry(
        concept_id="RXNORM:308135",
        code="308135",
        code_system=CodeSystem.RXNORM,
        canonical_name="amlodipine 10 MG Oral Tablet",
        semantic_type=EntityType.DRUG,
        rxnorm_tty="SCD",
    )
    generator = _retrieval(DictionaryStore([entry]), retrieval_sources=("exact",))

    candidates = generator.retrieve(
        "amlodipine 10 mg po daily", EntityType.DRUG
    )

    assert [(candidate.code, candidate.source) for candidate in candidates] == [
        ("308135", "btc_sample")
    ]
    assert (
        _retrieval(DictionaryStore([]), retrieval_sources=("exact",)).retrieve(
            "amlodipine 10 mg po daily", EntityType.DRUG
        )
        == []
    )


def test_btc_sample_is_reproduced_end_to_end() -> None:
    text = read_source_text(FIXTURE)
    resource = "src/medical_kg_nlp/resources/phase1_btc_medication_recognition.jsonl"
    prediction = PipelineFactory.from_config(
        PipelineFactoryConfig(recognition_dictionary_path=str(resource))
    ).process_text("btc", text)
    rows = prediction_to_phase1_entities(prediction, source_text=text)
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))

    assert len(rows) == 19
    drugs = [row for row in rows if row["type"] == "THUỐC"]
    symptoms = [row for row in rows if row["type"] == "TRIỆU_CHỨNG"]
    assert [row["candidates"] for row in drugs] == [
        ["308135"],
        ["243670"],
        ["866436"],
        ["392085"],
        ["7597"],
        ["313782"],
        ["904475"],
        ["1099279"],
        ["312935"],
        ["197527"],
        ["197528"],
    ]
    assert all(row["assertions"] == ["isHistorical"] for row in drugs)
    assert all(row["assertions"] == [] for row in symptoms)
    assert all(text[row["position"][0] : row["position"][1]] == row["text"] for row in rows)
    assert rows == expected
    assert validate_phase1_entities(expected, text, dictionary=DictionaryStore.from_jsonl(resource)) == []


def _btc_recognition_store() -> DictionaryStore:
    names = (
        "amlodipine",
        "aspirin",
        "metoprolol succinate",
        "guaifenesin",
        "nystatin",
        "acetaminophen",
        "pravastatin",
        "docusate sodium",
        "senna",
        "clonazepam",
    )
    return DictionaryStore(
        [
            ConceptEntry(
                concept_id=f"RXNORM:{index}",
                code=str(index),
                code_system=CodeSystem.RXNORM,
                canonical_name=name,
                semantic_type=EntityType.DRUG,
            )
            for index, name in enumerate(names, start=1)
        ]
    )
