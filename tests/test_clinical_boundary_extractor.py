from __future__ import annotations

from medical_kg_nlp.ner.contracts import RuleNerContext
from medical_kg_nlp.ner.extractors.boundary import ClinicalBoundaryProposalExtractor
from medical_kg_nlp.ner.proposal import EntityProposal
from medical_kg_nlp.schema.types import EntityType


def test_boundary_extractor_composes_symptom_prefix_compound_and_severity() -> None:
    text = "Triệu chứng: cơn ngất xỉu dữ dội."
    start = text.index("ngất")
    foundation = _proposal(text, "ngất", EntityType.SYMPTOM, start)

    proposals = ClinicalBoundaryProposalExtractor().propose(
        text,
        RuleNerContext(foundation_proposals=(foundation,)),
    )

    assert _mentions(text, proposals) == [
        ("cơn ngất xỉu dữ dội", EntityType.SYMPTOM)
    ]
    assert dict(proposals[0].features)["boundary_rules"] == (
        "symptom_prefix,symptom_compound_fainting,symptom_suffix"
    )


def test_boundary_extractor_expands_disease_modifiers() -> None:
    text = "Chẩn đoán: suy tim, không đặc hiệu nghiêm trọng."
    start = text.index("suy tim")
    foundation = _proposal(text, "suy tim", EntityType.DISEASE, start)

    proposals = ClinicalBoundaryProposalExtractor().propose(
        text,
        RuleNerContext(foundation_proposals=(foundation,)),
    )

    assert _mentions(text, proposals) == [
        ("suy tim, không đặc hiệu nghiêm trọng", EntityType.DISEASE)
    ]


def test_boundary_extractor_never_crosses_newline_or_expands_drugs() -> None:
    text = "đau\nnghiêm trọng\naspirin dữ dội"
    symptom = _proposal(text, "đau", EntityType.SYMPTOM, 0)
    drug_start = text.index("aspirin")
    drug = _proposal(text, "aspirin", EntityType.DRUG, drug_start)

    proposals = ClinicalBoundaryProposalExtractor().propose(
        text,
        RuleNerContext(foundation_proposals=(symptom, drug)),
    )

    assert proposals == ()


def test_boundary_extractor_expands_compound_from_xiu_fragment() -> None:
    text = "Có cơn ngất xỉu gián đoạn"
    start = text.index("xỉu")
    foundation = _proposal(text, "xỉu", EntityType.SYMPTOM, start)

    proposals = ClinicalBoundaryProposalExtractor().propose(
        text,
        RuleNerContext(foundation_proposals=(foundation,)),
    )

    assert _mentions(text, proposals) == [
        ("ngất xỉu gián đoạn", EntityType.SYMPTOM)
    ]


def _proposal(
    source_text: str,
    mention: str,
    entity_type: EntityType,
    start: int,
) -> EntityProposal:
    proposal = EntityProposal(
        span=(start, start + len(mention)),
        candidate_types=(entity_type,),
        source="dictionary_exact",
        score=0.78,
        evidence_ids=("exact:C1",),
        concept_ids=("C1",),
    )
    proposal.validate_offsets(source_text)
    return proposal


def _mentions(
    source_text: str,
    proposals: tuple[EntityProposal, ...],
) -> list[tuple[str, EntityType]]:
    output: list[tuple[str, EntityType]] = []
    for proposal in proposals:
        entity_type = proposal.entity_type
        assert entity_type is not None
        output.append(
            (
                source_text[proposal.span[0] : proposal.span[1]],
                entity_type,
            )
        )
    return output
