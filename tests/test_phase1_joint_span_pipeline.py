"""End-to-end safety tests for learned Phase 1 lattice composition."""

from __future__ import annotations

from collections.abc import Sequence

from medical_kg_nlp.benchmarks.phase1.joint_span import (
    Phase1JointSpanCandidate,
    Phase1JointSpanLabel,
    Phase1JointSpanPrediction,
    Phase1JointSpanSelectionPolicy,
)
from medical_kg_nlp.benchmarks.phase1.joint_span_pipeline import Phase1JointSpanPipeline
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.types import CodeSystem, EntityType


def test_joint_pipeline_selects_generated_full_medication_span_and_relinks_it() -> None:
    text = "Bệnh nhân dùng aspirin 81 mg po daily."
    document = ClinicalDocument(document_id="1", text=text)
    start = text.index("aspirin")
    full = "aspirin 81 mg po daily"
    full_end = start + len(full)
    dictionary = DictionaryStore(
        (
            ConceptEntry(
                concept_id="rx:aspirin-full",
                code="999",
                code_system=CodeSystem.RXNORM,
                canonical_name=full,
                semantic_type=EntityType.DRUG,
                aliases=(full,),
            ),
            ConceptEntry(
                concept_id="rx:aspirin",
                code="1191",
                code_system=CodeSystem.RXNORM,
                canonical_name="aspirin",
                semantic_type=EntityType.DRUG,
                aliases=("aspirin",),
            ),
        )
    )
    result = Phase1JointSpanPipeline(
        verifier=_FullMedicationVerifier(full),
        selection_policy=_policy(),
        source_roles={"rule": ProposalSourceRole.RULE, "qwen": ProposalSourceRole.LLM},
        budget_manifest=_budget(),
        dictionary=dictionary,
        candidate_source_priority=("qwen", "rule"),
        candidate_policy="keep",
    ).run(
        [document],
        {
            "rule": {"1": [_row("aspirin", start, start + len("aspirin"), candidates=["1191"])]},
            "qwen": {"1": [_row("aspirin", start, start + len("aspirin"), candidates=[])]},
        },
    )

    assert result.rows_by_document["1"] == (
        {
            "text": full,
            "type": "THUỐC",
            "position": [start, full_end],
            "candidates": ["999"],
            "assertions": [],
        },
    )


class _FullMedicationVerifier:
    provenance = "test@1"

    def __init__(self, full_text: str) -> None:
        self._full_text = full_text

    def predict(
        self,
        candidates: Sequence[Phase1JointSpanCandidate],
    ) -> Sequence[Phase1JointSpanPrediction]:
        return tuple(
            Phase1JointSpanPrediction(
                candidate.variant.variant_id,
                tuple(
                    (
                        label.value,
                        _probability(label, candidate.variant.text == self._full_text),
                    )
                    for label in Phase1JointSpanLabel
                ),
            )
            for candidate in candidates
        )


def _probability(label: Phase1JointSpanLabel, is_full_medication: bool) -> float:
    if is_full_medication and label is Phase1JointSpanLabel.EXACT_DRUG:
        return 0.99
    if not is_full_medication and label is Phase1JointSpanLabel.SPURIOUS:
        return 0.99
    return 0.01 / (len(Phase1JointSpanLabel) - 1)


def _policy() -> Phase1JointSpanSelectionPolicy:
    return Phase1JointSpanSelectionPolicy(
        genre_type_thresholds=tuple(
            (genre, entity_type, 0.5)
            for genre in ("clinical", "educational", "qa", "unknown")
            for entity_type in (
                "CHẨN_ĐOÁN",
                "KẾT_QUẢ_XÉT_NGHIỆM",
                "TÊN_XÉT_NGHIỆM",
                "THUỐC",
                "TRIỆU_CHỨNG",
            )
        )
    )


def _budget() -> dict[str, object]:
    return {
        "schema_version": "inference-model-budget.v2",
        "status": "verified",
        "total_parameters": 1,
        "maximum_parameters": 9_000_000_000,
        "active": [
            {
                "artifact_id": "joint",
                "roles": ["assertion", "candidate_rerank", "ner", "recall", "verifier"],
            }
        ],
    }


def _row(text: str, start: int, end: int, *, candidates: list[str]) -> dict[str, object]:
    return {
        "text": text,
        "type": "THUỐC",
        "position": [start, end],
        "assertions": [],
        "candidates": candidates,
        "confidence": 0.9,
        "source_label": "test",
    }
