from __future__ import annotations

from clingrounder.benchmarks.phase1.phase1_entity_gates import (
    Phase1EntityGateConfig,
    apply_phase1_entity_gates,
    compile_boundary_rule_candidates,
)
from clingrounder.benchmarks.phase1.phase1_rule_registry import phase1_rule_registry_from_data


def test_lab_gate_keeps_anchored_values_and_blocks_unanchored_noise() -> None:
    text = "Kali: 2.4. Ngày 03/07/2026. Dùng 80mg. Ghi nhận tăng."
    rows = {
        "1": [
            _row("Kali", "TÊN_XÉT_NGHIỆM", 0),
            _row("2.4", "KẾT_QUẢ_XÉT_NGHIỆM", text.index("2.4")),
            _row("03/07/2026", "KẾT_QUẢ_XÉT_NGHIỆM", text.index("03/07/2026")),
            _row("80mg", "KẾT_QUẢ_XÉT_NGHIỆM", text.index("80mg")),
            _row("tăng", "KẾT_QUẢ_XÉT_NGHIỆM", text.index("tăng")),
        ]
    }

    output, decisions, counts = apply_phase1_entity_gates(
        rows,
        {"1": text},
        config=Phase1EntityGateConfig(lab_gate=True),
    )

    assert [row["text"] for row in output["1"]] == ["Kali", "2.4"]
    assert {row["rule_id"] for row in decisions} == {
        "builtin.lab.date",
        "builtin.lab.medication_attribute",
        "builtin.lab.unanchored_value",
    }
    assert counts["lab_gate.block"] == 2
    assert counts["retype.retype_internal_and_block"] == 1


def test_lab_gate_can_retype_with_reviewed_rule_and_drops_internal_attributes() -> None:
    text = "Liều 80mg"
    registry = phase1_rule_registry_from_data(
        {
            "schema_version": "phase1-rule-registry.v1",
            "rules": [
                {
                    "rule_id": "retype.medication.strength",
                    "stage": "retype",
                    "entity_type": "KẾT_QUẢ_XÉT_NGHIỆM",
                    "normalized_mention": "80mg",
                    "action": "retype",
                    "replacement_type": "MEDICATION_STRENGTH",
                    "review_status": "reviewed",
                }
            ],
        }
    )

    output, decisions, _ = apply_phase1_entity_gates(
        {"1": [_row("80mg", "KẾT_QUẢ_XÉT_NGHIỆM", 5)]},
        {"1": text},
        config=Phase1EntityGateConfig(lab_gate=True),
        registry=registry,
    )

    assert output["1"] == []
    assert decisions[0]["action"] == "retype_internal_and_block"


def test_lab_gate_keeps_self_describing_vitals_and_requires_anchor_for_concentrations() -> None:
    text = "Nhiệt độ 38.8°C. Giá trị 6.3 mg/dL. Glucose 7.1 mg/dL."
    rows = {
        "1": [
            _row("38.8°C", "KẾT_QUẢ_XÉT_NGHIỆM", text.index("38.8°C")),
            _row("6.3 mg/dL", "KẾT_QUẢ_XÉT_NGHIỆM", text.index("6.3 mg/dL")),
            _row("Glucose", "TÊN_XÉT_NGHIỆM", text.index("Glucose")),
            _row("7.1 mg/dL", "KẾT_QUẢ_XÉT_NGHIỆM", text.index("7.1 mg/dL")),
        ]
    }

    output, decisions, _ = apply_phase1_entity_gates(
        rows,
        {"1": text},
        config=Phase1EntityGateConfig(lab_gate=True),
    )

    assert [row["text"] for row in output["1"]] == ["38.8°C", "Glucose", "7.1 mg/dL"]
    blocked = next(row for row in decisions if row["before"]["text"] == "6.3 mg/dL")
    assert blocked["rule_id"] == "builtin.lab.unanchored_value"


def test_lab_gate_uses_same_line_fallback_for_translated_result_lists() -> None:
    text = "- dấu hiệu sinh tồn là: 98.8 65 9360 20 95\n- ghi nhận 350 ml từ ống thông"
    rows = {
        "1": [
            _row("98.8", "KẾT_QUẢ_XÉT_NGHIỆM", text.index("98.8")),
            _row("65", "KẾT_QUẢ_XÉT_NGHIỆM", text.index("65")),
            _row("350 ml", "KẾT_QUẢ_XÉT_NGHIỆM", text.index("350 ml")),
        ]
    }

    output, decisions, _ = apply_phase1_entity_gates(
        rows,
        {"1": text},
        config=Phase1EntityGateConfig(lab_gate=True),
    )

    assert [row["text"] for row in output["1"]] == ["98.8", "65", "350 ml"]
    assert decisions == []


def test_medication_full_span_extends_only_contiguous_sig_attributes() -> None:
    text = "amlodipine 10 mg po daily điều trị tăng huyết áp"
    row = _row("amlodipine", "THUỐC", 0)
    row["assertions"] = ["isHistorical"]
    row["candidates"] = ["308135"]

    output, decisions, counts = apply_phase1_entity_gates(
        {"1": [row]},
        {"1": text},
        config=Phase1EntityGateConfig(medication_full_span=True, resolve_overlaps=False),
    )

    assert output["1"] == [
        {
            "text": "amlodipine 10 mg po daily",
            "type": "THUỐC",
            "assertions": ["isHistorical"],
            "candidates": ["308135"],
            "position": [0, 25],
        }
    ]
    assert decisions[0]["rule_id"] == "builtin.medication.full_span"
    assert decisions[0]["after"]["text"] == "amlodipine 10 mg po daily"
    assert counts["medication_full_span.expand"] == 1


def test_medication_full_span_blocks_expansion_over_another_entity() -> None:
    text = "amlodipine 10 mg"
    rows = {
        "1": [
            _row("amlodipine", "THUỐC", 0),
            _row("10 mg", "KẾT_QUẢ_XÉT_NGHIỆM", text.index("10 mg")),
        ]
    }

    output, decisions, counts = apply_phase1_entity_gates(
        rows,
        {"1": text},
        config=Phase1EntityGateConfig(medication_full_span=True, resolve_overlaps=False),
    )

    assert [row["text"] for row in output["1"]] == ["amlodipine", "10 mg"]
    assert decisions == []
    assert counts["medication_full_span.blocked_overlap"] == 1


def test_strict_exclusions_require_reviewed_type_scoped_registry_rule() -> None:
    text = "phẫu thuật và đau"
    policy = {
        "exclusions": {
            "strict": {"procedure_or_device": ["phẫu thuật"]},
            "review": {"generic_or_vague": ["đau"]},
        }
    }
    rows = {
        "1": [
            _row("phẫu thuật", "CHẨN_ĐOÁN", 0),
            _row("đau", "TRIỆU_CHỨNG", text.index("đau")),
        ]
    }

    without_registry, decisions, _ = apply_phase1_entity_gates(
        rows,
        {"1": text},
        config=Phase1EntityGateConfig(strict_exclusions=True),
        annotation_policy=policy,
    )
    assert [row["text"] for row in without_registry["1"]] == ["phẫu thuật", "đau"]
    assert decisions == []

    registry = phase1_rule_registry_from_data(
        {
            "schema_version": "phase1-rule-registry.v1",
            "rules": [
                {
                    "rule_id": "exclude.reviewed.procedure",
                    "stage": "strict_exclusion",
                    "entity_type": "CHẨN_ĐOÁN",
                    "normalized_mention": "phẫu thuật",
                    "action": "block",
                    "review_status": "reviewed",
                }
            ],
        }
    )
    output, decisions, _ = apply_phase1_entity_gates(
        rows,
        {"1": text},
        config=Phase1EntityGateConfig(strict_exclusions=True),
        annotation_policy=policy,
        registry=registry,
    )

    assert [row["text"] for row in output["1"]] == ["đau"]
    assert decisions[0]["rule_id"] == "exclude.reviewed.procedure"


def test_boundary_expansion_is_guarded_by_punctuation_and_overlap() -> None:
    registry = phase1_rule_registry_from_data(
        {
            "schema_version": "phase1-rule-registry.v1",
            "rules": [
                {
                    "rule_id": "boundary.symptom.chest",
                    "stage": "boundary_symptom_course",
                    "entity_type": "TRIỆU_CHỨNG",
                    "normalized_mention": "đau",
                    "action": "expand",
                    "right_regex": "(?P<expand> ngực)",
                    "review_status": "reviewed",
                }
            ],
        }
    )
    text = "đau ngực"
    output, decisions, _ = apply_phase1_entity_gates(
        {"1": [_row("đau", "TRIỆU_CHỨNG", 0)]},
        {"1": text},
        config=Phase1EntityGateConfig(boundary_stages=("boundary_symptom_course",)),
        registry=registry,
    )
    assert output["1"][0]["text"] == "đau ngực"
    assert output["1"][0]["position"] == [0, 8]
    assert decisions[0]["rule_id"] == "boundary.symptom.chest"

    blocked, _, counts = apply_phase1_entity_gates(
        {"1": [_row("đau", "TRIỆU_CHỨNG", 0), _row("ngực", "TRIỆU_CHỨNG", 4)]},
        {"1": text},
        config=Phase1EntityGateConfig(boundary_stages=("boundary_symptom_course",)),
        registry=registry,
    )
    assert {row["text"] for row in blocked["1"]} == {"đau", "ngực"}
    assert counts["boundary_blocked_overlap"] == 1


def test_boundary_expansion_never_crosses_punctuation() -> None:
    registry = phase1_rule_registry_from_data(
        {
            "schema_version": "phase1-rule-registry.v1",
            "rules": [
                {
                    "rule_id": "boundary.symptom.punctuation",
                    "stage": "boundary_symptom_course",
                    "entity_type": "TRIỆU_CHỨNG",
                    "normalized_mention": "đau",
                    "action": "expand",
                    "right_regex": "(?P<expand>, dữ dội)",
                    "review_status": "reviewed",
                }
            ],
        }
    )
    text = "đau, dữ dội"

    output, decisions, _ = apply_phase1_entity_gates(
        {"1": [_row("đau", "TRIỆU_CHỨNG", 0)]},
        {"1": text},
        config=Phase1EntityGateConfig(boundary_stages=("boundary_symptom_course",)),
        registry=registry,
    )

    assert output["1"][0]["text"] == "đau"
    assert decisions == []


def test_boundary_rule_discovery_requires_repeated_train_pattern_and_stays_draft() -> None:
    gold = {
        "1": [_row("đau ngực", "TRIỆU_CHỨNG", 0)],
        "2": [_row("đau ngực", "TRIỆU_CHỨNG", 0)],
    }
    predictions = {
        "1": [_row("đau", "TRIỆU_CHỨNG", 0)],
        "2": [_row("đau", "TRIỆU_CHỨNG", 0)],
    }

    registry, audit = compile_boundary_rule_candidates(
        gold,
        predictions,
        split="all",
        minimum_document_support=2,
    )

    assert audit["compiled_rule_count"] == 1
    assert registry.rules[0].review_status == "draft"
    assert registry.active_rules() == ()
    assert registry.rules[0].right_regex is not None


def _row(text: str, entity_type: str, start: int) -> dict[str, object]:
    return {
        "text": text,
        "type": entity_type,
        "assertions": [],
        "candidates": [],
        "position": [start, start + len(text)],
    }
