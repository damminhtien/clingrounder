from __future__ import annotations

from clingrounder.benchmarks.phase1.contextual_alias_mining import (
    compile_phase1_contextual_alias_rules,
)
from clingrounder.mining.lexicon import MentionInventoryEntry


def test_contextual_alias_compiler_requires_train_inventory_support() -> None:
    policy = {
        "aliases": {
            "context_required": {
                "TRIỆU_CHỨNG": ["đau", "ra máu"],
                "KẾT_QUẢ_XÉT_NGHIỆM": ["2.4", "dương tính"],
            }
        }
    }
    inventory = (
        _inventory("TRIỆU_CHỨNG", "đau", documents=3),
        _inventory("KẾT_QUẢ_XÉT_NGHIỆM", "2.4", documents=2),
    )

    compilation = compile_phase1_contextual_alias_rules(
        policy,
        inventory,
        inventory_sha256="a" * 64,
    )

    assert len(compilation.artifact["rules"]) == 1
    rule = compilation.artifact["rules"][0]
    assert rule["alias"] == "đau"
    assert rule["entity_type"] == "SYMPTOM"
    assert rule["match_modes"] == ["exact", "toneless"]
    assert rule["required_any"] == [
        "medication_indication",
        "standalone_list_item",
        "symptom_predicate",
        "symptom_section",
    ]
    assert compilation.report["reason_counts"] == {
        "absent_from_train_inventory": 2,
        "eligible": 1,
        "numeric_owned_by_lab_grammar": 1,
    }
    assert all("document_id" not in rule for rule in compilation.artifact["rules"])


def _inventory(
    source_label: str,
    mention: str,
    *,
    documents: int,
) -> MentionInventoryEntry:
    return MentionInventoryEntry(
        term_id=f"term:{source_label}:{mention}",
        normalized_mention=mention,
        entity_type={
            "TRIỆU_CHỨNG": "SYMPTOM",
            "KẾT_QUẢ_XÉT_NGHIỆM": "LAB_RESULT",
        }[source_label],
        source_label=source_label,
        occurrence_count=documents,
        document_count=documents,
        consensus_occurrence_count=0,
        surface_variant_count=1,
        surface_variants=((mention, documents),),
        source_artifact_ids=("artifact",),
        label_sources=(("phase1_manual_gold", documents),),
        example_document_ids=(),
        review_tier="multi_document",
        recommended_use="recognition",
    )
