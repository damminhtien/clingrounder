from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.benchmarks.phase1.manual_gold import manual_gold_split
from medical_kg_nlp.benchmarks.phase1.phase1_candidate_overlay import TT06_SOURCE_ID
from medical_kg_nlp.benchmarks.phase1.phase1_rule_registry import (
    Phase1Rule,
    Phase1RuleRegistry,
    phase1_rule_registry_from_data,
)
from medical_kg_nlp.benchmarks.phase1.ontology import (
    PHASE1_ASSERTABLE_TYPES,
    PHASE1_CODABLE_TYPES,
    PHASE1_RULE_BY_TYPE,
)
from medical_kg_nlp.utils.text import normalize_for_match


AssertionRegime = Literal["history", "negation", "family"]
CandidateRegime = Literal["icd", "rxnorm_ingredient", "rxnorm_clinical_drug"]

_ASSERTION_STAGE = {
    "history": "assertion_history",
    "negation": "assertion_negation",
    "family": "assertion_family",
}
_ASSERTION_VALUE = {
    "history": "isHistorical",
    "negation": "isNegated",
    "family": "isFamily",
}
_HISTORY_SECTION_RE = re.compile(
    r"(?i)\b(?:tiền sử(?: bệnh| nội khoa| phẫu thuật)?|bệnh lý mạn tính|"
    r"(?:danh sách )?thuốc trước(?: khi)? nhập viện|các sự kiện trước khi nhập viện)\b"
)
_FAMILY_SECTION_RE = re.compile(r"(?i)\b(?:tiền sử gia đình|bệnh sử gia đình)\b")
_GENERIC_SECTION_RE = re.compile(
    r"(?i)\b(?:lý do nhập viện|bệnh sử|triệu chứng hiện tại|khám|cận lâm sàng|"
    r"tình trạng ngay trước khi nhập viện|đánh giá tại bệnh viện|chẩn đoán|điều trị|"
    r"các phát hiện chẩn đoán khác|thủ thuật đã thực hiện)\b"
)
_HISTORY_LOCAL_RE = re.compile(
    r"(?i)(?:\bcách đây\s+(?:(?:khoảng|hơn|gần)\s+)?(?:vài|\d+)\s+(?:tháng|năm)\b|"
    r"\btrước đây\b|\bđã từng\b|\blần nhập viện trước\b|"
    r"\bđợt (?:điều trị|nằm viện|nhập viện) trước\b|\bgần đây nhập viện vì\b)"
)
_NEGATION_RE = re.compile(
    r"(?i)(?:\bkhông có\b|\bkhông ghi nhận\b|\bphủ nhận\b|\bchưa\b|"
    r"\bkhông phát hiện\b|\bkhông thấy\b|\bâm tính với\b|\bloại trừ\b)"
)
_NEGATION_EXCEPTION_RE = re.compile(
    r"(?i)(?:không loại trừ|không thể|không nhớ|không giảm|không đáp ứng|"
    r"không đặc hiệu|không ổn định|không biến chứng)"
)
_SCOPE_TERMINATION_RE = re.compile(r"(?i)\b(?:nhưng|tuy nhiên|ngoại trừ|trừ khi)\b")
_FAMILY_OWNERSHIP_RE = re.compile(
    r"(?i)(?:\b(?:mẹ|cha|bố|anh|chị|em|con trai|con gái|vợ|chồng)\s+"
    r"(?:của bệnh nhân\s+)?(?:bị|mắc|có)\b|\bngười thân\s+(?:bị|mắc|có)\b)"
)
_CLAUSE_DELIMITER_RE = re.compile(r"[;\n.!?]")
_INGREDIENT_TTYS = frozenset({"IN", "PIN", "MIN", "BN"})
_CLINICAL_DRUG_TTYS = frozenset({"SCD", "SBD"})
_STRENGTH_OR_FORM_RE = re.compile(
    r"(?i)(?:\b\d+(?:[.,]\d+)?\s*(?:mcg|mg|g|ml|meq|iu)\b|"
    r"\b(?:tablet|capsule|injection|solution|suspension|cream|ointment|"
    r"viên|ống|dung dịch|tiêm|truyền)\b)"
)


def apply_selective_assertions(
    rows_by_doc: Mapping[str, list[dict[str, Any]]],
    source_text_by_doc: Mapping[str, str],
    *,
    regimes: Iterable[AssertionRegime],
    registry: Phase1RuleRegistry | None = None,
    preserve_existing: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, int]]:
    regime_order = tuple(dict.fromkeys(regimes))
    invalid = set(regime_order) - set(_ASSERTION_STAGE)
    if invalid:
        raise ValueError(f"Unknown assertion regimes: {sorted(invalid)}")
    active_registry = registry or Phase1RuleRegistry(())
    output: dict[str, list[dict[str, Any]]] = {}
    decisions: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for document_id, rows in rows_by_doc.items():
        source_text = source_text_by_doc.get(document_id)
        if source_text is None:
            raise ValueError(f"Missing source text for document {document_id}.")
        document_rows: list[dict[str, Any]] = []
        for row in rows:
            result = _copy_row(row)
            result["assertions"] = list(row.get("assertions", [])) if preserve_existing else []
            if str(row.get("type", "")) not in PHASE1_ASSERTABLE_TYPES:
                document_rows.append(result)
                continue
            start, end = _position(row)
            for regime in regime_order:
                stage = _ASSERTION_STAGE[regime]
                assertion = _ASSERTION_VALUE[regime]
                rule_id = _builtin_assertion_rule(
                    regime,
                    source_text,
                    start,
                    end,
                    str(row["text"]),
                    str(row["type"]),
                )
                rule_source = "builtin_selective"
                if rule_id is None:
                    rule = _first_rule(
                        active_registry.active_rules(stage),
                        row,
                        source_text,
                    )
                    if rule is None or assertion not in rule.assertions:
                        continue
                    rule_id = rule.rule_id
                    rule_source = str((rule.provenance or {}).get("source", "rule_registry"))
                if assertion in result["assertions"]:
                    continue
                result["assertions"].append(assertion)
                decisions.append(
                    {
                        "document_id": document_id,
                        "stage": stage,
                        "rule_id": rule_id,
                        "source": rule_source,
                        "action": "emit",
                        "assertion": assertion,
                        "entity": _identity(row),
                    }
                )
                counters[f"{stage}.emit"] += 1
            document_rows.append(result)
        output[document_id] = document_rows
    _assert_frozen_entities(rows_by_doc, output, field="assertions")
    counters["decision_total"] = len(decisions)
    return output, decisions, dict(sorted(counters.items()))


def compile_reviewed_candidate_registry(
    gold_by_doc: Mapping[str, list[dict[str, Any]]],
    dictionary_path: str | Path,
    *,
    split: Literal["train", "holdout", "all"] = "train",
) -> tuple[Phase1RuleRegistry, dict[str, Any]]:
    code_metadata = _load_code_metadata(dictionary_path)
    dictionary = DictionaryStore.from_jsonl(dictionary_path)
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    rejected: Counter[str] = Counter()
    selected_documents = 0
    for document_id, rows in gold_by_doc.items():
        if split != "all" and manual_gold_split(document_id) != split:
            continue
        selected_documents += 1
        for row in rows:
            entity_type = str(row.get("type", ""))
            if entity_type not in PHASE1_CODABLE_TYPES:
                continue
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != 1:
                rejected["not_single_candidate"] += 1
                continue
            code = str(candidates[0]).strip()
            metadata = code_metadata.get(code)
            expected_system = "ICD-10" if entity_type == "CHẨN_ĐOÁN" else "RxNorm"
            if metadata is None or metadata["code_system"] != expected_system:
                rejected["missing_or_wrong_code_system"] += 1
                continue
            if expected_system == "ICD-10" and not metadata["tt06"]:
                rejected["icd_not_tt06"] += 1
                continue
            expected_type = PHASE1_RULE_BY_TYPE[entity_type].internal_type
            exact_codes = {
                entry.code
                for entry in dictionary.exact_lookup(str(row.get("text", "")))
                if entry.code is not None
                and entry.semantic_type == expected_type
                and entry.code_system.value == expected_system
            }
            if len(exact_codes) != 1:
                rejected["not_exact_unique_dictionary_match"] += 1
                continue
            if code not in exact_codes:
                rejected["reviewed_code_disagrees_with_exact_dictionary"] += 1
                continue
            normalized = normalize_for_match(str(row.get("text", "")))
            if not normalized:
                continue
            key = (normalized, entity_type)
            group = groups.setdefault(
                key,
                {"codes": Counter(), "occurrences": 0, "documents": set()},
            )
            group["codes"][code] += 1
            group["occurrences"] += 1
            group["documents"].add(document_id)

    rules: list[dict[str, Any]] = []
    for (normalized, entity_type), group in sorted(groups.items()):
        if len(group["codes"]) != 1:
            rejected["ambiguous_reviewed_mapping"] += int(group["occurrences"])
            continue
        code = next(iter(group["codes"]))
        metadata = code_metadata[code]
        if entity_type == "CHẨN_ĐOÁN":
            stage = "candidate_icd"
            code_kind = "tt06_exact"
        else:
            ttys = set(metadata["rxnorm_ttys"])
            if ttys & _CLINICAL_DRUG_TTYS:
                stage = "candidate_rxnorm_clinical_drug"
                code_kind = "clinical_drug"
            elif ttys & _INGREDIENT_TTYS or not ttys:
                stage = "candidate_rxnorm_ingredient"
                code_kind = "ingredient_or_brand"
            else:
                rejected["unsupported_rxnorm_tty"] += int(group["occurrences"])
                continue
        rules.append(
            {
                "rule_id": f"reviewed.{stage}.{_short_hash(entity_type + chr(0) + normalized)}",
                "stage": stage,
                "entity_type": entity_type,
                "normalized_mention": normalized,
                "action": "emit",
                "candidates": [code],
                "confidence_tier": "high",
                "provenance": {
                    "source": "manual_gold_train" if split == "train" else f"manual_gold_{split}",
                    "code_system": metadata["code_system"],
                    "code_kind": code_kind,
                    "occurrence_support": int(group["occurrences"]),
                    "document_support": len(group["documents"]),
                    "dictionary_release": metadata["release"],
                },
                "review_status": "reviewed",
            }
        )
    registry = phase1_rule_registry_from_data(
        {"schema_version": "phase1-rule-registry.v1", "rules": rules},
        source="compiled_reviewed_candidates",
    )
    return registry, {
        "schema_version": "phase1-reviewed-candidate-audit.v1",
        "split": split,
        "selected_document_count": selected_documents,
        "reviewed_group_count": len(groups),
        "compiled_rule_count": len(rules),
        "exact_unique_required": True,
        "compiled_rule_count_by_stage": dict(sorted(Counter(rule["stage"] for rule in rules).items())),
        "rejected_counts": dict(sorted(rejected.items())),
    }


def reviewed_candidate_map_rows(
    registry: Phase1RuleRegistry,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stages = (
        "candidate_icd",
        "candidate_rxnorm_ingredient",
        "candidate_rxnorm_clinical_drug",
    )
    for rule in registry.active_rules(*stages):
        if (
            rule.entity_type not in PHASE1_CODABLE_TYPES
            or rule.normalized_mention is None
            or len(rule.candidates) != 1
        ):
            continue
        provenance = dict(rule.provenance or {})
        expected_system = (
            "ICD-10" if rule.entity_type == "CHẨN_ĐOÁN" else "RxNorm"
        )
        code_system = str(provenance.get("code_system", ""))
        if code_system != expected_system:
            raise ValueError(
                f"Rule {rule.rule_id!r} has code_system {code_system!r}; "
                f"expected {expected_system!r}."
            )
        rows.append(
            {
                "normalized_mention": rule.normalized_mention,
                "entity_type": rule.entity_type,
                "candidate": rule.candidates[0],
                "code_system": code_system,
                "candidate_stage": rule.stage,
                "confidence_tier": rule.confidence_tier,
                "occurrence_support": int(provenance.get("occurrence_support", 0)),
                "document_support": int(provenance.get("document_support", 0)),
                "dictionary_release": str(provenance.get("dictionary_release", "")),
                "provenance": str(provenance.get("source", "")),
                "rule_id": rule.rule_id,
                "review_status": rule.review_status,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["entity_type"]),
            str(row["normalized_mention"]),
            str(row["candidate"]),
        ),
    )


def write_reviewed_candidate_map(
    registry: Phase1RuleRegistry,
    path: str | Path,
) -> list[dict[str, Any]]:
    rows = reviewed_candidate_map_rows(registry)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    return rows


def apply_selective_candidates(
    rows_by_doc: Mapping[str, list[dict[str, Any]]],
    registry: Phase1RuleRegistry,
    *,
    regime: CandidateRegime,
    consensus_keys: set[tuple[str, int, int, str]],
    mention_limit: int | None = None,
    preserve_existing: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, int]]:
    stage = {
        "icd": "candidate_icd",
        "rxnorm_ingredient": "candidate_rxnorm_ingredient",
        "rxnorm_clinical_drug": "candidate_rxnorm_clinical_drug",
    }[regime]
    rules = registry.active_rules(stage)
    rule_by_key = {
        (rule.entity_type, rule.normalized_mention): rule
        for rule in rules
        if rule.entity_type is not None and rule.normalized_mention is not None
    }
    frequency: Counter[tuple[str, str]] = Counter()
    for rows in rows_by_doc.values():
        for row in rows:
            key = (str(row.get("type", "")), normalize_for_match(str(row.get("text", ""))))
            if key in rule_by_key:
                frequency[key] += 1
    ranked_keys = sorted(
        frequency,
        key=lambda key: (-frequency[key], key[0], key[1]),
    )
    if mention_limit is not None:
        ranked_keys = ranked_keys[:mention_limit]
    allowed_keys = set(ranked_keys)

    output: dict[str, list[dict[str, Any]]] = {}
    decisions: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for document_id, rows in rows_by_doc.items():
        document_rows: list[dict[str, Any]] = []
        for row in rows:
            result = _copy_row(row)
            result["candidates"] = list(row.get("candidates", [])) if preserve_existing else []
            start, end = _position(row)
            consensus_key = (document_id, start, end, str(row.get("type", "")))
            key = (str(row.get("type", "")), normalize_for_match(str(row.get("text", ""))))
            rule = rule_by_key.get(key)
            if rule is None or key not in allowed_keys:
                document_rows.append(result)
                continue
            if consensus_key not in consensus_keys:
                counters["blocked_without_two_source_consensus"] += 1
                document_rows.append(result)
                continue
            if regime == "rxnorm_clinical_drug" and _STRENGTH_OR_FORM_RE.search(str(row.get("text", ""))) is None:
                counters["blocked_clinical_drug_without_strength_or_form"] += 1
                document_rows.append(result)
                continue
            result["candidates"] = list(rule.candidates)
            decisions.append(
                {
                    "document_id": document_id,
                    "stage": stage,
                    "rule_id": rule.rule_id,
                    "source": str((rule.provenance or {}).get("source", "rule_registry")),
                    "action": "emit",
                    "candidate": rule.candidates[0],
                    "entity": _identity(row),
                }
            )
            counters[f"{stage}.emit"] += 1
            document_rows.append(result)
        output[document_id] = document_rows
    _assert_frozen_entities(rows_by_doc, output, field="candidates")
    counters["eligible_rule_mentions"] = len(allowed_keys)
    counters["decision_total"] = len(decisions)
    return output, decisions, dict(sorted(counters.items()))


def validate_probe_isolation(
    baseline: Mapping[str, list[dict[str, Any]]],
    trial: Mapping[str, list[dict[str, Any]]],
    *,
    module: Literal["entity", "assertion", "candidate", "combined"],
) -> list[str]:
    issues: list[str] = []
    if set(baseline) != set(trial):
        issues.append("document_set_changed")
        return issues
    for document_id in baseline:
        base_rows = baseline[document_id]
        trial_rows = trial[document_id]
        if module in {"assertion", "candidate", "combined"} and [_identity(row) for row in base_rows] != [
            _identity(row) for row in trial_rows
        ]:
            issues.append(f"{document_id}:entity_identity_changed")
            continue
        if module == "assertion" and [row.get("candidates", []) for row in base_rows] != [
            row.get("candidates", []) for row in trial_rows
        ]:
            issues.append(f"{document_id}:candidate_changed")
        if module == "candidate" and [row.get("assertions", []) for row in base_rows] != [
            row.get("assertions", []) for row in trial_rows
        ]:
            issues.append(f"{document_id}:assertion_changed")
        if module == "entity":
            base_by_identity = {_identity_key(row): row for row in base_rows}
            for row in trial_rows:
                base = base_by_identity.get(_identity_key(row))
                if base is None:
                    continue
                if row.get("assertions", []) != base.get("assertions", []):
                    issues.append(f"{document_id}:assertion_changed_on_shared_entity")
                if row.get("candidates", []) != base.get("candidates", []):
                    issues.append(f"{document_id}:candidate_changed_on_shared_entity")
    return issues


def write_overlay_trace(
    decisions: list[dict[str, Any]],
    counters: Mapping[str, int],
    output_dir: str | Path,
    *,
    prefix: str,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{prefix}_decisions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in decisions),
        encoding="utf-8",
    )
    (output / f"{prefix}_summary.json").write_text(
        json.dumps(dict(counters), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _builtin_assertion_rule(
    regime: AssertionRegime,
    source_text: str,
    start: int,
    end: int,
    mention: str,
    entity_type: str,
) -> str | None:
    section = _current_section(source_text, start)
    left_scope = _left_scope(source_text, start)
    if regime == "history":
        if entity_type == "TRIỆU_CHỨNG":
            return None
        if section == "history":
            return "builtin.assertion.history_section"
        if _HISTORY_LOCAL_RE.search(left_scope):
            return "builtin.assertion.history_local_cue"
        return None
    if regime == "family":
        if entity_type != "CHẨN_ĐOÁN":
            return None
        if section == "family":
            return "builtin.assertion.family_section"
        if _FAMILY_OWNERSHIP_RE.search(left_scope):
            return "builtin.assertion.family_ownership"
        return None
    normalized = normalize_for_match(mention)
    if normalized.startswith("không thể") or normalized.startswith("không nhớ"):
        return None
    matches = list(_NEGATION_RE.finditer(left_scope))
    if not matches:
        return None
    cue = matches[-1]
    scoped = left_scope[cue.start() :]
    if _NEGATION_EXCEPTION_RE.search(scoped):
        return None
    termination = _SCOPE_TERMINATION_RE.search(scoped)
    if termination is not None and termination.start() > cue.end() - cue.start():
        return None
    return "builtin.assertion.negation_clause"


def _current_section(source_text: str, start: int) -> str | None:
    section: str | None = None
    for line in source_text[:start].splitlines(keepends=True):
        stripped = line.strip().rstrip(":")
        if _FAMILY_SECTION_RE.search(stripped):
            section = "family"
        elif _HISTORY_SECTION_RE.search(stripped):
            section = "history"
        elif _GENERIC_SECTION_RE.search(stripped):
            section = None
    return section


def _left_scope(source_text: str, start: int, *, max_chars: int = 220) -> str:
    window_start = max(0, start - max_chars)
    window = source_text[window_start:start]
    delimiters = list(_CLAUSE_DELIMITER_RE.finditer(window))
    return window[delimiters[-1].end() :] if delimiters else window


def _first_rule(
    rules: tuple[Phase1Rule, ...], row: Mapping[str, Any], source_text: str
) -> Phase1Rule | None:
    start, end = _position(row)
    for rule in rules:
        if rule.entity_type is not None and rule.entity_type != row.get("type"):
            continue
        if rule.mention_matches(str(row.get("text", ""))) and rule.context_matches(
            source_text, start, end
        ):
            return rule
    return None


def _load_code_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            code = str(row.get("code", "")).strip()
            code_system = str(row.get("code_system", "")).strip()
            if not code or code_system not in {"ICD-10", "RxNorm"}:
                continue
            item = metadata.setdefault(
                code,
                {
                    "code_system": code_system,
                    "tt06": False,
                    "rxnorm_ttys": set(),
                    "release": set(),
                },
            )
            if item["code_system"] != code_system:
                continue
            source_ids = _string_values(row.get("source_ids"))
            item["tt06"] = bool(
                item["tt06"] or row.get("source") == TT06_SOURCE_ID or TT06_SOURCE_ID in source_ids
            )
            item["rxnorm_ttys"].update(_string_values(row.get("rxnorm_ttys")))
            if row.get("rxnorm_tty"):
                item["rxnorm_ttys"].add(str(row["rxnorm_tty"]))
            item["release"].update(source_ids)
            if row.get("source"):
                item["release"].add(str(row["source"]))
    return {
        code: {
            **item,
            "rxnorm_ttys": sorted(item["rxnorm_ttys"]),
            "release": "+".join(sorted(item["release"])),
        }
        for code, item in metadata.items()
    }


def _assert_frozen_entities(
    baseline: Mapping[str, list[dict[str, Any]]],
    trial: Mapping[str, list[dict[str, Any]]],
    *,
    field: str,
) -> None:
    issues = validate_probe_isolation(
        baseline,
        trial,
        module="assertion" if field == "assertions" else "candidate",
    )
    if issues:
        raise ValueError(f"{field} overlay changed frozen entity fields: {issues[:5]}")


def _copy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output["position"] = list(row.get("position", []))
    output["assertions"] = list(row.get("assertions", []))
    output["candidates"] = list(row.get("candidates", []))
    return output


def _identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "text": row.get("text"),
        "type": row.get("type"),
        "position": list(row.get("position", [])),
    }


def _identity_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    position = row.get("position", [])
    return (row.get("text"), row.get("type"), tuple(position) if isinstance(position, list) else ())


def _position(row: Mapping[str, Any]) -> tuple[int, int]:
    position = row.get("position")
    if not isinstance(position, list) or len(position) != 2 or not all(isinstance(x, int) for x in position):
        raise ValueError(f"Invalid position: {position!r}")
    return int(position[0]), int(position[1])


def _string_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple | set):
        return tuple(str(item) for item in value)
    return ()


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
