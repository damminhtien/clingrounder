"""Contract tests for deterministic under-9B Phase 1 composition."""

from __future__ import annotations

from medical_kg_nlp.benchmarks.phase1.max_score_pipeline import (
    Phase1MaxScorePipeline,
)
from medical_kg_nlp.benchmarks.phase1.proposal_calibration import (
    Phase1ProbabilityCalibrator,
    Phase1ProposalVerifier,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.evaluation.sparse_logistic import SparseLogisticModel
from medical_kg_nlp.ontology.phase1 import PHASE1_ALLOWED_TYPES
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.types import CodeSystem, EntityType


def test_max_score_pipeline_resolves_metadata_and_preserves_offsets() -> None:
    text = "Tiền sử: tăng huyết áp. Hiện tại đau ngực."
    documents = [ClinicalDocument(document_id="1", text=text)]
    diagnosis_start = text.index("tăng huyết áp")
    symptom_start = text.index("đau ngực")
    sources = {
        "rule": {
            "1": [
                _row(
                    "tăng huyết áp",
                    "CHẨN_ĐOÁN",
                    diagnosis_start,
                    candidates=["I10"],
                ),
                _row("đau ngực", "TRIỆU_CHỨNG", symptom_start),
            ]
        },
        "qwen": {
            "1": [
                _row(
                    "tăng huyết áp",
                    "CHẨN_ĐOÁN",
                    diagnosis_start,
                    candidates=[],
                ),
                _row("đau ngực", "TRIỆU_CHỨNG", symptom_start),
            ]
        },
    }
    pipeline = Phase1MaxScorePipeline(
        verifier=_accept_all_verifier(),
        source_roles={
            "rule": ProposalSourceRole.RULE,
            "qwen": ProposalSourceRole.LLM,
        },
        budget_manifest=_budget_manifest(),
        dictionary=_dictionary(),
        candidate_source_priority=("qwen", "rule"),
    )

    result = pipeline.run(documents, sources)

    rows = list(result.rows_by_document["1"])
    assert rows == [
        {
            "text": "tăng huyết áp",
            "type": "CHẨN_ĐOÁN",
            "assertions": ["isHistorical"],
            "position": [diagnosis_start, diagnosis_start + len("tăng huyết áp")],
            "candidates": ["I10"],
        },
        {
            "text": "đau ngực",
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "position": [symptom_start, symptom_start + len("đau ngực")],
            "candidates": [],
        },
    ]
    assert result.counters["entity.selected"] == 2
    assert result.counters["candidate.output_candidate_rows"] == 1
    assert any(
        decision.get("source") == "rule"
        and decision.get("stage") == "candidate_metadata_hydration"
        for decision in result.source_decisions
    )


def test_max_score_pipeline_requires_budget_roles() -> None:
    invalid_budget = _budget_manifest()
    invalid_budget["active"] = [
        {
            "artifact_id": "model",
            "roles": ["ner"],
        }
    ]

    try:
        Phase1MaxScorePipeline(
            verifier=_accept_all_verifier(),
            source_roles={
                "rule": ProposalSourceRole.RULE,
                "qwen": ProposalSourceRole.LLM,
            },
            budget_manifest=invalid_budget,
            dictionary=_dictionary(),
            candidate_source_priority=("rule", "qwen"),
        )
    except ValueError as error:
        assert "missing required roles" in str(error)
    else:
        raise AssertionError("An incomplete inference budget must fail closed")


def _row(
    text: str,
    entity_type: str,
    start: int,
    *,
    candidates: list[str] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "text": text,
        "type": entity_type,
        "assertions": [],
        "position": [start, start + len(text)],
    }
    if candidates is not None:
        row["candidates"] = candidates
    return row


def _accept_all_verifier() -> Phase1ProposalVerifier:
    return Phase1ProposalVerifier(
        model=SparseLogisticModel(
            feature_names=(),
            weights=(),
            bias=20.0,
        ),
        probability_calibrator=Phase1ProbabilityCalibrator(
            method="identity_logistic",
            model=None,
            fold_count=2,
            assignment_sha256="a" * 64,
        ),
        thresholds=tuple(
            (entity_type, 0.5) for entity_type in sorted(PHASE1_ALLOWED_TYPES)
        ),
        training_dataset_sha256="b" * 64,
    )


def _dictionary() -> DictionaryStore:
    return DictionaryStore(
        [
            ConceptEntry(
                concept_id="ICD10:I10",
                code="I10",
                code_system=CodeSystem.ICD10,
                canonical_name="tăng huyết áp",
                semantic_type=EntityType.DISEASE,
                source="TT06",
            )
        ]
    )


def _budget_manifest() -> dict[str, object]:
    return {
        "schema_version": "inference-model-budget.v2",
        "status": "verified",
        "maximum_parameters": 8_900_000_000,
        "total_parameters": 8_800_000_000,
        "active": [
            {
                "artifact_id": "qwen",
                "roles": [
                    "assertion",
                    "candidate_rerank",
                    "recall",
                    "verifier",
                ],
            },
            {
                "artifact_id": "xlmr",
                "roles": ["ner"],
            },
        ],
    }
