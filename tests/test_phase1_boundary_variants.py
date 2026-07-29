"""Raw-offset generation and labeling tests for Phase 1 boundary variants."""

from __future__ import annotations

from medical_kg_nlp.benchmarks.phase1.boundary_variants import (
    BoundaryErrorLabel,
    boundary_cross_encoder_text,
    extract_phase1_boundary_features,
    generate_phase1_boundary_variants,
    label_phase1_boundary_variant,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.ner.dictionary_matcher import DictionaryMatcher
from medical_kg_nlp.schema.types import CodeSystem, EntityType


def test_boundary_variants_include_model_dictionary_and_clinical_options() -> None:
    text = "đau ngực trái khi gắng sức dữ dội."
    foundation = _row("đau", "TRIỆU_CHỨNG", 0, "xlmr")
    matcher = DictionaryMatcher(
        [
            (
                "đau ngực trái",
                ConceptEntry(
                    concept_id="symptom:chest-pain",
                    code=None,
                    code_system=CodeSystem.LOCAL,
                    canonical_name="đau ngực trái",
                    semantic_type=EntityType.SYMPTOM,
                ),
            )
        ]
    )

    variants = generate_phase1_boundary_variants(
        "1",
        text,
        [foundation],
        source_roles={"xlmr": ProposalSourceRole.TOKEN_MODEL},
        dictionary_matcher=matcher,
    )
    by_text = {variant.text: variant for variant in variants}

    assert "đau" in by_text
    assert "đau ngực trái" in by_text
    assert "đau ngực trái khi gắng sức" in by_text
    assert "dictionary_trie" in by_text["đau ngực trái"].generators
    assert "model_token" in by_text["đau"].generators
    assert all(text[start:end] == variant.text for variant in variants for start, end in [variant.position])


def test_boundary_variants_do_not_cross_clause_punctuation() -> None:
    text = "đau ngực. sốt cao"
    variants = generate_phase1_boundary_variants(
        "1",
        text,
        [_row("đau", "TRIỆU_CHỨNG", 0, "rule")],
        source_roles={"rule": ProposalSourceRole.RULE},
    )

    assert all("sốt" not in variant.text for variant in variants)


def test_generated_windows_do_not_bridge_distinct_foundation_families() -> None:
    text = "đau ngực khó thở"
    second_start = text.index("khó")
    variants = generate_phase1_boundary_variants(
        "1",
        text,
        [
            _row("đau", "TRIỆU_CHỨNG", 0, "rule"),
            _row("khó thở", "TRIỆU_CHỨNG", second_start, "xlmr"),
        ],
        source_roles={
            "rule": ProposalSourceRole.RULE,
            "xlmr": ProposalSourceRole.TOKEN_MODEL,
        },
    )
    pain = next(
        variant
        for variant in variants
        if variant.position == (0, len("đau"))
    )
    dyspnea = next(
        variant
        for variant in variants
        if variant.position == (second_start, len(text))
    )

    assert pain.family_id != dyspnea.family_id


def test_medication_parser_supplies_full_span_without_indication() -> None:
    text = "aspirin 81 mg po daily điều trị đau đầu"
    variants = generate_phase1_boundary_variants(
        "1",
        text,
        [_row("aspirin", "THUỐC", 0, "rule")],
        source_roles={"rule": ProposalSourceRole.RULE},
    )
    selected = next(
        variant for variant in variants if variant.text == "aspirin 81 mg po daily"
    )

    assert "medication_full_span" in selected.generators
    assert "điều trị" not in selected.text


def test_boundary_labels_distinguish_length_and_wrong_entity() -> None:
    text = "đau ngực trái và sốt"
    rows = [
        _row("đau", "TRIỆU_CHỨNG", 0, "xlmr"),
        _row("đau ngực trái", "TRIỆU_CHỨNG", 0, "qwen"),
        _row("đau ngực trái và sốt", "TRIỆU_CHỨNG", 0, "rule"),
        _row("sốt", "CHẨN_ĐOÁN", text.index("sốt"), "rule"),
    ]
    variants = generate_phase1_boundary_variants(
        "1",
        text,
        rows,
        source_roles={
            "qwen": ProposalSourceRole.LLM,
            "rule": ProposalSourceRole.RULE,
            "xlmr": ProposalSourceRole.TOKEN_MODEL,
        },
    )
    gold = [
        {
            "text": "đau ngực trái",
            "type": "TRIỆU_CHỨNG",
            "position": [0, len("đau ngực trái")],
        }
    ]
    labels = {
        (variant.text, variant.entity_type): label_phase1_boundary_variant(
            variant,
            gold,
        )
        for variant in variants
        if variant.text in {"đau", "đau ngực trái", "đau ngực trái và sốt", "sốt"}
    }

    assert labels[("đau", "TRIỆU_CHỨNG")] is BoundaryErrorLabel.TOO_SHORT
    assert labels[("đau ngực trái", "TRIỆU_CHỨNG")] is BoundaryErrorLabel.CORRECT
    assert (
        labels[("đau ngực trái và sốt", "TRIỆU_CHỨNG")]
        is BoundaryErrorLabel.TOO_LONG
    )
    assert labels[("sốt", "CHẨN_ĐOÁN")] is BoundaryErrorLabel.WRONG_ENTITY


def test_boundary_joint_contract_is_cross_encoder_ready() -> None:
    text = "Triệu chứng hiện tại:\nđau ngực trái"
    start = text.index("đau")
    variants = generate_phase1_boundary_variants(
        "1",
        text,
        [_row("đau ngực", "TRIỆU_CHỨNG", start, "qwen")],
        source_roles={"qwen": ProposalSourceRole.LLM},
    )
    variant = next(item for item in variants if item.text == "đau ngực trái")
    rendered = boundary_cross_encoder_text(variant, text)
    features = extract_phase1_boundary_features(
        variant,
        text,
        {"qwen": ProposalSourceRole.LLM},
        family_size=len(variants),
        base_probability=0.8,
    )

    assert "[SECTION] symptom" in rendered
    assert "[ENTITY] đau ngực trái" in rendered
    assert features["numeric:base_proposal_probability"] == 0.8
    assert features["generator:token_window"] == 1.0
    assert any(name.startswith("hash:joint_bigram:") for name in features)


def _row(
    text: str,
    entity_type: str,
    start: int,
    source: str,
) -> dict[str, object]:
    return {
        "document_id": "1",
        "proposal_id": f"{source}:{start}:{len(text)}:{entity_type}",
        "text": text,
        "type": entity_type,
        "position": [start, start + len(text)],
        "sources": [source],
        "source_count": 1,
        "all_source_agreement": False,
        "status": "source_only",
        "source_evidence": {
            source: {
                "confidence": 0.8,
                "source_labels": [],
                "support_only": False,
            }
        },
    }
