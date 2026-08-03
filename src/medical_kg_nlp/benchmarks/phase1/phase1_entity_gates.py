from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from medical_kg_nlp.benchmarks.phase1.manual_gold import manual_gold_split
from medical_kg_nlp.benchmarks.phase1.phase1_rule_registry import (
    INTERNAL_RETYPE_TYPES,
    Phase1Rule,
    Phase1RuleRegistry,
    phase1_rule_registry_from_data,
)
from medical_kg_nlp.ner.medication_mention_parser import MedicationMentionParser
from medical_kg_nlp.benchmarks.phase1.ontology import PHASE1_TYPE_PRIORITY
from medical_kg_nlp.utils.text import normalize_for_match


LAB_RESULT_TYPE = "KẾT_QUẢ_XÉT_NGHIỆM"
LAB_TEST_TYPE = "TÊN_XÉT_NGHIỆM"
_BOUNDARY_STAGES = (
    "boundary_diagnosis",
    "boundary_symptom_prefix",
    "boundary_symptom_course",
    "boundary_imaging_test",
)
_BOUNDARY_TYPE_BY_STAGE = {
    "boundary_diagnosis": "CHẨN_ĐOÁN",
    "boundary_symptom_prefix": "TRIỆU_CHỨNG",
    "boundary_symptom_course": "TRIỆU_CHỨNG",
    "boundary_imaging_test": "TÊN_XÉT_NGHIỆM",
}
_QUALITATIVE_RESULTS = frozenset(
    {
        "âm tính",
        "dương tính",
        "bình thường",
        "bất thường",
        "tăng",
        "giảm",
        "cao",
        "thấp",
        "không",
    }
)
_LAB_ANCHOR_RE = re.compile(
    r"(?i)(?:\b(?:alt|ast|alp|bilirubin|bun|cr|creatinine|crp|esr|glucose|hba1c|hct|"
    r"hematocrit|hgb|hemoglobin|inr|kali|lactate|natri|nitrite|platelets?|plt|"
    r"troponin|ure|wbc|chem\s*7)\b|huyết áp|nhịp tim|nhịp thở|nhiệt độ|mạch|spo2|"
    r"độ bão hòa oxy|dấu hiệu sinh tồn|xét nghiệm|định lượng|cấy|pcr|sinh thiết|chụp|nội soi)"
)
_VALUE_RE = re.compile(
    r"^\s*(?:[<>≤≥]=?\s*)?[+-]?\d+(?:[.,]\d+)?(?:\s*(?:-|đến)\s*[+-]?\d+(?:[.,]\d+)?)?"
    r"(?:\s*(?:%|mmhg|mmol/l|mg/dl|g/dl|mg/l|ng/ml|meq/l|u/l|lần/phút|°c|ra))?\s*$",
    flags=re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"^\s*(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}[/-]\d{1,2}[/-]\d{1,2})\s*$"
)
_MEDICATION_ATTRIBUTE_RE = re.compile(
    r"^\s*(?:"
    r"\d+(?:[.,]\d+)?\s*(?:mcg|mg|g|kg|ml|l|meq|iu|đơn vị)(?:/\w+)?"
    r"|(?:po|iv|im|sc|sl|uống|tiêm|truyền|tĩnh mạch|dưới da)"
    r"|\d+\s*(?:lần|viên|ống|gói)(?:/ngày)?"
    r"|(?:q\d+h|bid|tid|qid|prn)"
    r")\s*$",
    flags=re.IGNORECASE,
)
_SELF_DESCRIBING_VITAL_RE = re.compile(
    r"^\s*(?:[<>≤≥]=?\s*)?[+-]?\d+(?:[.,]\d+)?\s*"
    r"(?:°c|mmhg|%|lần/phút)\s*$",
    flags=re.IGNORECASE,
)
_MEDICATION_ANCHOR_RE = re.compile(
    r"(?i)\b(?:dùng|liều|thuốc|uống|tiêm|truyền|bổ sung|po|iv|im|sc|sl|"
    r"tablet|capsule|injection)\b"
)
_CLAUSE_DELIMITER_RE = re.compile(r"[;\n!?]|(?<!\d)\.(?!\d)")
_BOUNDARY_PUNCTUATION_RE = re.compile(r"[,;:\n.!?]")


@dataclass(frozen=True)
class Phase1EntityGateConfig:
    lab_gate: bool = False
    medication_full_span: bool = False
    strict_exclusions: bool = False
    boundary_stages: tuple[str, ...] = ()
    resolve_overlaps: bool = True

    def __post_init__(self) -> None:
        invalid = set(self.boundary_stages) - set(_BOUNDARY_STAGES)
        if invalid:
            raise ValueError(f"Unknown boundary stages: {sorted(invalid)}")


def load_annotation_policy(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: annotation policy must be an object.")
    return payload


def apply_phase1_entity_gates(
    rows_by_doc: Mapping[str, list[dict[str, Any]]],
    source_text_by_doc: Mapping[str, str],
    *,
    config: Phase1EntityGateConfig,
    annotation_policy: Mapping[str, Any] | None = None,
    registry: Phase1RuleRegistry | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, int]]:
    """Apply entity-only gates in a fixed, auditable order."""
    decisions: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    output_by_doc: dict[str, list[dict[str, Any]]] = {}
    active_registry = registry or Phase1RuleRegistry(())
    # Compiled exclusions are discovery evidence only. Runtime blocking requires a reviewed,
    # type/context-scoped registry rule.
    strict_exclusions: dict[str, str] = {}

    for document_id, input_rows in rows_by_doc.items():
        source_text = source_text_by_doc.get(document_id)
        if source_text is None:
            raise ValueError(f"Missing source text for document {document_id}.")
        rows = [_copy_row(row) for row in input_rows]
        _validate_raw_offsets(document_id, rows, source_text)

        if config.lab_gate:
            rows = _apply_lab_gate(
                document_id,
                rows,
                source_text,
                active_registry,
                decisions,
                counters,
            )
        if config.medication_full_span:
            rows = _apply_medication_full_span(
                document_id,
                rows,
                source_text,
                decisions,
                counters,
            )
        if config.strict_exclusions:
            rows = _apply_strict_exclusions(
                document_id,
                rows,
                source_text,
                strict_exclusions,
                active_registry,
                decisions,
                counters,
            )
        if config.boundary_stages:
            rows = _apply_boundary_rules(
                document_id,
                rows,
                source_text,
                active_registry,
                config.boundary_stages,
                decisions,
                counters,
            )
        if config.resolve_overlaps:
            rows = _resolve_overlaps(document_id, rows, decisions, counters)
        _validate_raw_offsets(document_id, rows, source_text)
        output_by_doc[document_id] = sorted(rows, key=_entity_sort_key)

    counters["decision_total"] = len(decisions)
    counters["output_entity_total"] = sum(len(rows) for rows in output_by_doc.values())
    return output_by_doc, decisions, dict(sorted(counters.items()))


def write_entity_gate_trace(
    decisions: list[dict[str, Any]],
    counters: Mapping[str, int],
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "entity_gate_decisions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in decisions),
        encoding="utf-8",
    )
    (output / "entity_gate_summary.json").write_text(
        json.dumps(dict(counters), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def compile_boundary_rule_candidates(
    gold_by_doc: Mapping[str, list[dict[str, Any]]],
    predictions_by_doc: Mapping[str, list[dict[str, Any]]],
    *,
    split: str = "train",
    minimum_document_support: int = 2,
    review_status: str = "draft",
) -> tuple[Phase1RuleRegistry, dict[str, Any]]:
    """Discover concept-level under-boundary rules without retaining document selectors."""
    if minimum_document_support < 1:
        raise ValueError("minimum_document_support must be at least 1.")
    evidence: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    signatures_by_mention: dict[tuple[str, str], set[tuple[str, str, str]]] = {}
    rejected: Counter[str] = Counter()
    for document_id, gold_rows in gold_by_doc.items():
        if split != "all" and manual_gold_split(document_id) != split:
            continue
        prediction_rows = predictions_by_doc.get(document_id, [])
        for prediction in prediction_rows:
            entity_type = str(prediction.get("type", ""))
            if entity_type not in {"CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM"}:
                continue
            pred_start, pred_end = _position(prediction)
            matches: list[tuple[float, Mapping[str, Any]]] = []
            for gold in gold_rows:
                if gold.get("type") != entity_type:
                    continue
                gold_start, gold_end = _position(gold)
                if gold_start > pred_start or pred_end > gold_end:
                    continue
                if (gold_start, gold_end) == (pred_start, pred_end):
                    continue
                overlap = pred_end - pred_start
                union = gold_end - gold_start
                matches.append((overlap / union, gold))
            if not matches:
                continue
            _, selected_gold = max(
                matches,
                key=lambda item: (
                    item[0],
                    -(_position(item[1])[1] - _position(item[1])[0]),
                ),
            )
            gold_start, gold_end = _position(selected_gold)
            gold_text = str(selected_gold.get("text", ""))
            prefix = gold_text[: pred_start - gold_start]
            suffix = (
                gold_text[len(gold_text) - (gold_end - pred_end) :]
                if gold_end > pred_end
                else ""
            )
            if _BOUNDARY_PUNCTUATION_RE.search(prefix + suffix):
                rejected["punctuation_crossing"] += 1
                continue
            stage = _boundary_stage_for_type(entity_type, prefix)
            if stage is None:
                continue
            normalized = normalize_for_match(str(prediction.get("text", "")))
            if not normalized:
                continue
            signature = (stage, prefix, suffix)
            signatures_by_mention.setdefault((entity_type, normalized), set()).add(signature)
            key = (stage, entity_type, normalized, prefix, suffix)
            group = evidence.setdefault(key, {"documents": set(), "occurrences": 0})
            group["documents"].add(document_id)
            group["occurrences"] += 1

    rules: list[dict[str, Any]] = []
    for (stage, entity_type, normalized, prefix, suffix), group in sorted(evidence.items()):
        support = len(group["documents"])
        if support < minimum_document_support:
            rejected["insufficient_document_support"] += int(group["occurrences"])
            continue
        if len(signatures_by_mention[(entity_type, normalized)]) != 1:
            rejected["conflicting_boundary_signatures"] += int(group["occurrences"])
            continue
        rule: dict[str, Any] = {
            "rule_id": f"discovered.{stage}.{_short_hash(entity_type + chr(0) + normalized)}",
            "stage": stage,
            "entity_type": entity_type,
            "normalized_mention": normalized,
            "action": "expand",
            "confidence_tier": "high",
            "provenance": {
                "source": f"manual_gold_{split}_boundary_error",
                "occurrence_support": int(group["occurrences"]),
                "document_support": support,
            },
            "review_status": review_status,
            "notes": "Repeated under-boundary correction discovered on the sealed training split.",
        }
        if prefix:
            rule["left_regex"] = rf"(?P<expand>{re.escape(prefix)})$"
        if suffix:
            rule["right_regex"] = rf"^(?P<expand>{re.escape(suffix)})"
        rules.append(rule)
    registry = phase1_rule_registry_from_data(
        {"schema_version": "phase1-rule-registry.v1", "rules": rules},
        source="compiled_boundary_candidates",
    )
    return registry, {
        "schema_version": "phase1-boundary-rule-discovery.v1",
        "split": split,
        "minimum_document_support": minimum_document_support,
        "review_status": review_status,
        "evidence_group_count": len(evidence),
        "compiled_rule_count": len(rules),
        "compiled_rule_count_by_stage": dict(
            sorted(Counter(rule["stage"] for rule in rules).items())
        ),
        "rejected_counts": dict(sorted(rejected.items())),
    }


def _apply_lab_gate(
    document_id: str,
    rows: list[dict[str, Any]],
    source_text: str,
    registry: Phase1RuleRegistry,
    decisions: list[dict[str, Any]],
    counters: Counter[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    registry_rules = registry.active_rules("lab_gate", "retype")
    for row in rows:
        if row.get("type") != LAB_RESULT_TYPE:
            output.append(row)
            continue
        start, end = _position(row)
        explicit = _first_matching_rule(registry_rules, row, source_text)
        if explicit is not None:
            if explicit.action == "keep":
                output.append(row)
                _decision(decisions, counters, document_id, explicit.stage, explicit, "keep", row, row)
                continue
            if explicit.action == "block":
                _decision(decisions, counters, document_id, explicit.stage, explicit, "block", row, None)
                continue
            if explicit.action == "retype":
                if explicit.replacement_type in INTERNAL_RETYPE_TYPES:
                    _decision(
                        decisions,
                        counters,
                        document_id,
                        explicit.stage,
                        explicit,
                        "retype_internal_and_block",
                        row,
                        None,
                    )
                    continue
                retyped = dict(row)
                retyped["type"] = explicit.replacement_type
                retyped["assertions"] = []
                retyped["candidates"] = []
                output.append(retyped)
                _decision(decisions, counters, document_id, explicit.stage, explicit, "retype", row, retyped)
                continue

        mention = str(row.get("text", ""))
        normalized = normalize_for_match(mention)
        if _MEDICATION_ATTRIBUTE_RE.fullmatch(mention) and _has_medication_anchor(
            source_text, start, end
        ):
            _builtin_decision(
                decisions,
                counters,
                document_id,
                "retype",
                "builtin.lab.medication_attribute",
                "retype_internal_and_block",
                "Dose, strength, route, or frequency has a medication anchor.",
                row,
            )
            continue
        if _DATE_RE.fullmatch(mention):
            _builtin_decision(
                decisions,
                counters,
                document_id,
                "lab_gate",
                "builtin.lab.date",
                "block",
                "Date-shaped value cannot be a standalone lab result.",
                row,
            )
            continue
        requires_anchor = (
            bool(_VALUE_RE.fullmatch(mention))
            and _SELF_DESCRIBING_VITAL_RE.fullmatch(mention) is None
        ) or normalized in _QUALITATIVE_RESULTS
        if requires_anchor and not _has_lab_anchor(rows, source_text, start, end):
            _builtin_decision(
                decisions,
                counters,
                document_id,
                "lab_gate",
                "builtin.lab.unanchored_value",
                "block",
                "Numeric or qualitative result has no test/vital anchor in the same clause.",
                row,
            )
            continue
        output.append(row)
    return output


def _apply_medication_full_span(
    document_id: str,
    rows: list[dict[str, Any]],
    source_text: str,
    decisions: list[dict[str, Any]],
    counters: Counter[str],
) -> list[dict[str, Any]]:
    parser = MedicationMentionParser()
    output = [dict(row) for row in rows]
    for index, row in enumerate(output):
        if row.get("type") != "THUỐC":
            continue
        start, end = _position(row)
        mention = parser.parse(source_text, (start, end))
        full_start, full_end = mention.full_span
        if (full_start, full_end) == (start, end):
            continue
        expanded = dict(row)
        expanded["position"] = [full_start, full_end]
        expanded["text"] = source_text[full_start:full_end]
        if any(
            _overlaps(expanded, other)
            for other_index, other in enumerate(output)
            if other_index != index
        ):
            counters["medication_full_span.blocked_overlap"] += 1
            continue

        output[index] = expanded
        _append_decision(
            decisions,
            counters,
            document_id=document_id,
            stage="medication_full_span",
            rule_id="builtin.medication.full_span",
            source="medication_mention_parser",
            action="expand",
            reason="Extend a drug name through contiguous dose, form, route, and frequency attributes.",
            before=row,
            after=expanded,
        )
    return output


def _apply_strict_exclusions(
    document_id: str,
    rows: list[dict[str, Any]],
    source_text: str,
    strict_exclusions: Mapping[str, str],
    registry: Phase1RuleRegistry,
    decisions: list[dict[str, Any]],
    counters: Counter[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    registry_rules = registry.active_rules("strict_exclusion")
    for row in rows:
        explicit = _first_matching_rule(registry_rules, row, source_text)
        normalized = normalize_for_match(str(row.get("text", "")))
        category = strict_exclusions.get(normalized)
        if explicit is None and category is None:
            output.append(row)
            continue
        if explicit is not None:
            _decision(decisions, counters, document_id, explicit.stage, explicit, "block", row, None)
            continue
        rule_id = f"compiled.exclusion.{category}.{_short_hash(normalized)}"
        _builtin_decision(
            decisions,
            counters,
            document_id,
            "strict_exclusion",
            rule_id,
            "block",
            f"Exact normalized mention is a train-compiled strict exclusion ({category}).",
            row,
            source="manual_gold_train",
        )
    return output


def _apply_boundary_rules(
    document_id: str,
    rows: list[dict[str, Any]],
    source_text: str,
    registry: Phase1RuleRegistry,
    stages: tuple[str, ...],
    decisions: list[dict[str, Any]],
    counters: Counter[str],
) -> list[dict[str, Any]]:
    output = [dict(row) for row in rows]
    rules = registry.active_rules(*stages)
    for index, row in enumerate(output):
        for rule in rules:
            if _BOUNDARY_TYPE_BY_STAGE[rule.stage] != row.get("type"):
                continue
            if rule.entity_type is not None and rule.entity_type != row.get("type"):
                continue
            if not rule.mention_matches(str(row.get("text", ""))):
                continue
            expanded = _expand_row(row, source_text, rule)
            if expanded is None:
                continue
            if any(_overlaps(expanded, other) for other_index, other in enumerate(output) if other_index != index):
                counters["boundary_blocked_overlap"] += 1
                continue
            before = row
            output[index] = expanded
            row = expanded
            _decision(decisions, counters, document_id, rule.stage, rule, "expand", before, expanded)
    return output


def _resolve_overlaps(
    document_id: str,
    rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    counters: Counter[str],
) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            -(_position(row)[1] - _position(row)[0]),
            -PHASE1_TYPE_PRIORITY.get(str(row.get("type", "")), 0),
            _position(row)[0],
            _position(row)[1],
        ),
    )
    selected: list[dict[str, Any]] = []
    for row in ranked:
        if any(_overlaps(row, kept) for kept in selected):
            _builtin_decision(
                decisions,
                counters,
                document_id,
                "overlap_resolution",
                "builtin.overlap.longest_type_priority",
                "block",
                "Overlaps a longer or higher-priority selected entity.",
                row,
            )
            continue
        selected.append(row)
    return selected


def _expand_row(row: dict[str, Any], source_text: str, rule: Phase1Rule) -> dict[str, Any] | None:
    start, end = _position(row)
    new_start, new_end = start, end
    if rule.left_regex is not None:
        left = source_text[max(0, start - 160) : start]
        matches = list(rule.left_regex.finditer(left))
        if not matches or matches[-1].end() != len(left):
            return None
        match = matches[-1]
        group_start = match.start("expand") if "expand" in match.re.groupindex else match.start()
        added = left[group_start:]
        if _BOUNDARY_PUNCTUATION_RE.search(added):
            return None
        new_start = start - len(added)
    if rule.right_regex is not None:
        right = source_text[end : min(len(source_text), end + 160)]
        right_match = rule.right_regex.match(right)
        if right_match is None:
            return None
        group_end = (
            right_match.end("expand")
            if "expand" in right_match.re.groupindex
            else right_match.end()
        )
        added = right[:group_end]
        if _BOUNDARY_PUNCTUATION_RE.search(added):
            return None
        new_end = end + len(added)
    if (new_start, new_end) == (start, end):
        return None
    expanded = dict(row)
    expanded["position"] = [new_start, new_end]
    expanded["text"] = source_text[new_start:new_end]
    return expanded


def _has_lab_anchor(rows: list[dict[str, Any]], source_text: str, start: int, end: int) -> bool:
    clause_start, clause_end = _clause_bounds(source_text, start, end)
    if any(
        row.get("type") == LAB_TEST_TYPE
        and _position(row)[0] < clause_end
        and _position(row)[1] > clause_start
        for row in rows
    ):
        return True
    return _LAB_ANCHOR_RE.search(source_text[clause_start:clause_end]) is not None


def _has_medication_anchor(source_text: str, start: int, end: int) -> bool:
    clause_start, clause_end = _clause_bounds(source_text, start, end)
    return _MEDICATION_ANCHOR_RE.search(source_text[clause_start:clause_end]) is not None


def _clause_bounds(source_text: str, start: int, end: int) -> tuple[int, int]:
    left = list(_CLAUSE_DELIMITER_RE.finditer(source_text, 0, start))
    clause_start = left[-1].end() if left else 0
    right = _CLAUSE_DELIMITER_RE.search(source_text, end)
    clause_end = right.start() if right else len(source_text)
    return clause_start, clause_end


def _strict_exclusion_index(policy: Mapping[str, Any]) -> dict[str, str]:
    exclusions = policy.get("exclusions")
    strict = exclusions.get("strict") if isinstance(exclusions, Mapping) else None
    result: dict[str, str] = {}
    if not isinstance(strict, Mapping):
        return result
    for category, values in strict.items():
        if not isinstance(values, list):
            continue
        for value in values:
            normalized = normalize_for_match(str(value))
            if normalized:
                result[normalized] = str(category)
    return result


def _first_matching_rule(
    rules: tuple[Phase1Rule, ...], row: Mapping[str, Any], source_text: str
) -> Phase1Rule | None:
    start, end = _position(row)
    for rule in rules:
        if rule.entity_type is not None and rule.entity_type != row.get("type"):
            continue
        if not rule.mention_matches(str(row.get("text", ""))):
            continue
        if rule.context_matches(source_text, start, end):
            return rule
    return None


def _decision(
    decisions: list[dict[str, Any]],
    counters: Counter[str],
    document_id: str,
    stage: str,
    rule: Phase1Rule,
    action: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any] | None,
) -> None:
    _append_decision(
        decisions,
        counters,
        document_id=document_id,
        stage=stage,
        rule_id=rule.rule_id,
        source=str((rule.provenance or {}).get("source", "rule_registry")),
        action=action,
        reason=rule.notes or f"Reviewed {stage} rule.",
        before=before,
        after=after,
    )


def _builtin_decision(
    decisions: list[dict[str, Any]],
    counters: Counter[str],
    document_id: str,
    stage: str,
    rule_id: str,
    action: str,
    reason: str,
    before: Mapping[str, Any],
    *,
    source: str = "builtin",
) -> None:
    _append_decision(
        decisions,
        counters,
        document_id=document_id,
        stage=stage,
        rule_id=rule_id,
        source=source,
        action=action,
        reason=reason,
        before=before,
        after=None,
    )


def _append_decision(
    decisions: list[dict[str, Any]],
    counters: Counter[str],
    *,
    document_id: str,
    stage: str,
    rule_id: str,
    source: str,
    action: str,
    reason: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any] | None,
) -> None:
    decisions.append(
        {
            "document_id": document_id,
            "stage": stage,
            "rule_id": rule_id,
            "source": source,
            "action": action,
            "reason": reason,
            "before": dict(before),
            "after": dict(after) if after is not None else None,
        }
    )
    counters[f"{stage}.{action}"] += 1
    counters[f"rule.{rule_id}"] += 1


def _copy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output["assertions"] = list(row.get("assertions", [])) if isinstance(row.get("assertions"), list) else []
    output["candidates"] = list(row.get("candidates", [])) if isinstance(row.get("candidates"), list) else []
    position = row.get("position")
    output["position"] = list(position) if isinstance(position, list) else position
    return output


def _validate_raw_offsets(document_id: str, rows: list[dict[str, Any]], source_text: str) -> None:
    for index, row in enumerate(rows):
        start, end = _position(row)
        text = row.get("text")
        if start < 0 or end > len(source_text) or not isinstance(text, str) or source_text[start:end] != text:
            raise ValueError(
                f"Document {document_id} row {index} violates source_text[start:end] == text."
            )


def _position(row: Mapping[str, Any]) -> tuple[int, int]:
    position = row.get("position")
    if (
        not isinstance(position, list)
        or len(position) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in position)
        or position[0] >= position[1]
    ):
        raise ValueError(f"Invalid Phase 1 position: {position!r}")
    return int(position[0]), int(position[1])


def _overlaps(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_start, left_end = _position(left)
    right_start, right_end = _position(right)
    return left_start < right_end and right_start < left_end


def _entity_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    start, end = _position(row)
    return (start, end, -PHASE1_TYPE_PRIORITY.get(str(row.get("type", "")), 0), str(row.get("text", "")))


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def _boundary_stage_for_type(entity_type: str, prefix: str) -> str | None:
    if entity_type == "CHẨN_ĐOÁN":
        return "boundary_diagnosis"
    if entity_type == "TRIỆU_CHỨNG":
        return "boundary_symptom_prefix" if prefix else "boundary_symptom_course"
    if entity_type == "TÊN_XÉT_NGHIỆM":
        return "boundary_imaging_test"
    return None
