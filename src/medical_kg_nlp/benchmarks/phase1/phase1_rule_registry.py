from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from medical_kg_nlp.ontology.phase1 import PHASE1_TYPE_PRIORITY
from medical_kg_nlp.utils.text import normalize_for_match


RULE_REGISTRY_SCHEMA_VERSION = "phase1-rule-registry.v1"
RULE_STAGES = frozenset(
    {
        "lab_gate",
        "retype",
        "strict_exclusion",
        "boundary_diagnosis",
        "boundary_symptom_prefix",
        "boundary_symptom_course",
        "boundary_imaging_test",
        "assertion_history",
        "assertion_negation",
        "assertion_family",
        "candidate_icd",
        "candidate_rxnorm_ingredient",
        "candidate_rxnorm_clinical_drug",
    }
)
RULE_ACTIONS = frozenset({"keep", "block", "retype", "expand", "emit"})
RULE_REVIEW_STATUSES = frozenset({"draft", "reviewed", "rejected"})
RULE_CONFIDENCE_TIERS = frozenset({"high", "medium", "low"})
ASSERTION_VALUES = frozenset({"isHistorical", "isNegated", "isFamily"})
PHASE1_ENTITY_TYPES = frozenset(PHASE1_TYPE_PRIORITY)
INTERNAL_RETYPE_TYPES = frozenset(
    {
        "MEDICATION_DOSE",
        "MEDICATION_STRENGTH",
        "MEDICATION_ROUTE",
        "MEDICATION_FREQUENCY",
    }
)

_FORBIDDEN_KEYS = frozenset(
    {
        "document_id",
        "document_ids",
        "doc_id",
        "position",
        "positions",
        "span",
        "spans",
        "start",
        "end",
        "absolute_start",
        "absolute_end",
        "output",
        "output_row",
    }
)
_RULE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")


@dataclass(frozen=True)
class Phase1Rule:
    rule_id: str
    stage: str
    action: str
    entity_type: str | None = None
    normalized_mention: str | None = None
    left_regex: re.Pattern[str] | None = None
    right_regex: re.Pattern[str] | None = None
    window_regex: re.Pattern[str] | None = None
    blocked_window_regex: re.Pattern[str] | None = None
    replacement_type: str | None = None
    assertions: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    confidence_tier: str = "high"
    provenance: Mapping[str, Any] | None = None
    review_status: str = "draft"
    notes: str = ""

    @property
    def active(self) -> bool:
        return self.review_status == "reviewed"

    def mention_matches(self, text: str) -> bool:
        return self.normalized_mention is None or normalize_for_match(text) == self.normalized_mention

    def context_matches(self, source_text: str, start: int, end: int, *, chars: int = 160) -> bool:
        left = source_text[max(0, start - chars) : start]
        right = source_text[end : min(len(source_text), end + chars)]
        window = source_text[max(0, start - chars) : min(len(source_text), end + chars)]
        if self.blocked_window_regex is not None and self.blocked_window_regex.search(window):
            return False
        checks = (
            (self.left_regex, left),
            (self.right_regex, right),
            (self.window_regex, window),
        )
        has_check = False
        for pattern, value in checks:
            if pattern is None:
                continue
            has_check = True
            if pattern.search(value) is None:
                return False
        return has_check or self.normalized_mention is not None


@dataclass(frozen=True)
class Phase1RuleRegistry:
    rules: tuple[Phase1Rule, ...]
    schema_version: str = RULE_REGISTRY_SCHEMA_VERSION

    def active_rules(self, *stages: str) -> tuple[Phase1Rule, ...]:
        stage_set = set(stages)
        return tuple(
            rule for rule in self.rules if rule.active and (not stage_set or rule.stage in stage_set)
        )


def load_phase1_rule_registry(path: str | Path) -> Phase1RuleRegistry:
    registry_path = Path(path)
    if registry_path.suffix.lower() == ".jsonl":
        payload: Any = {
            "schema_version": RULE_REGISTRY_SCHEMA_VERSION,
            "rules": [
                json.loads(line)
                for line in registry_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ],
        }
    else:
        payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    return phase1_rule_registry_from_data(payload, source=str(registry_path))


def phase1_rule_registry_from_data(
    payload: Any,
    *,
    source: str = "<memory>",
) -> Phase1RuleRegistry:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: rule registry must be an object.")
    _reject_forbidden_keys(payload, source)
    schema_version = str(payload.get("schema_version", ""))
    if schema_version != RULE_REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            f"{source}: expected schema_version {RULE_REGISTRY_SCHEMA_VERSION!r}, got {schema_version!r}."
        )
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list):
        raise ValueError(f"{source}: rules must be a list.")
    rules: list[Phase1Rule] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(raw_rules):
        if not isinstance(row, Mapping):
            raise ValueError(f"{source}: rules[{index}] must be an object.")
        rule = _parse_rule(row, f"{source}:rules[{index}]")
        if rule.rule_id in seen_ids:
            raise ValueError(f"{source}: duplicate rule_id {rule.rule_id!r}.")
        seen_ids.add(rule.rule_id)
        rules.append(rule)
    return Phase1RuleRegistry(tuple(rules), schema_version=schema_version)


def write_phase1_rule_registry(
    registry: Phase1RuleRegistry,
    path: str | Path,
) -> None:
    payload = {
        "schema_version": registry.schema_version,
        "rules": [_rule_to_data(rule) for rule in registry.rules],
    }
    Path(path).write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _parse_rule(row: Mapping[str, Any], source: str) -> Phase1Rule:
    rule_id = str(row.get("rule_id", "")).strip()
    if _RULE_ID_RE.fullmatch(rule_id) is None:
        raise ValueError(f"{source}: invalid rule_id {rule_id!r}.")
    stage = str(row.get("stage", "")).strip()
    if stage not in RULE_STAGES:
        raise ValueError(f"{source}: invalid stage {stage!r}.")
    action = str(row.get("action", "")).strip()
    if action not in RULE_ACTIONS:
        raise ValueError(f"{source}: invalid action {action!r}.")
    entity_type = _optional_string(row.get("entity_type"))
    if entity_type is not None and entity_type not in PHASE1_ENTITY_TYPES:
        raise ValueError(f"{source}: invalid entity_type {entity_type!r}.")
    replacement_type = _optional_string(row.get("replacement_type"))
    if replacement_type is not None and replacement_type not in (
        PHASE1_ENTITY_TYPES | INTERNAL_RETYPE_TYPES
    ):
        raise ValueError(f"{source}: invalid replacement_type {replacement_type!r}.")
    normalized_mention = _optional_string(row.get("normalized_mention"))
    if normalized_mention is not None:
        normalized_mention = normalize_for_match(normalized_mention)
    assertions = _string_tuple(row.get("assertions"))
    if not set(assertions) <= ASSERTION_VALUES:
        raise ValueError(f"{source}: invalid assertions {assertions!r}.")
    candidates = _string_tuple(row.get("candidates"))
    if len(candidates) > 1:
        raise ValueError(f"{source}: at most one candidate is allowed.")
    if candidates and entity_type not in {"CHẨN_ĐOÁN", "THUỐC"}:
        raise ValueError(f"{source}: candidates require CHẨN_ĐOÁN or THUỐC entity_type.")
    confidence_tier = str(row.get("confidence_tier", "high"))
    if confidence_tier not in RULE_CONFIDENCE_TIERS:
        raise ValueError(f"{source}: invalid confidence_tier {confidence_tier!r}.")
    review_status = str(row.get("review_status", "draft"))
    if review_status not in RULE_REVIEW_STATUSES:
        raise ValueError(f"{source}: invalid review_status {review_status!r}.")
    if action == "retype" and replacement_type is None:
        raise ValueError(f"{source}: retype action requires replacement_type.")
    if action == "emit" and stage.startswith("assertion_") and not assertions:
        raise ValueError(f"{source}: assertion emit rule requires assertions.")
    if action == "emit" and stage.startswith("candidate_") and not candidates:
        raise ValueError(f"{source}: candidate emit rule requires candidates.")
    if stage.startswith("boundary_") and action != "expand":
        raise ValueError(f"{source}: boundary rules must use action='expand'.")
    if stage.startswith("boundary_") and not row.get("left_regex") and not row.get("right_regex"):
        raise ValueError(f"{source}: boundary rules require left_regex or right_regex.")
    return Phase1Rule(
        rule_id=rule_id,
        stage=stage,
        action=action,
        entity_type=entity_type,
        normalized_mention=normalized_mention,
        left_regex=_compile_optional(row.get("left_regex"), source, "left_regex"),
        right_regex=_compile_optional(row.get("right_regex"), source, "right_regex"),
        window_regex=_compile_optional(row.get("window_regex"), source, "window_regex"),
        blocked_window_regex=_compile_optional(
            row.get("blocked_window_regex"), source, "blocked_window_regex"
        ),
        replacement_type=replacement_type,
        assertions=assertions,
        candidates=candidates,
        confidence_tier=confidence_tier,
        provenance=dict(row.get("provenance", {})) if isinstance(row.get("provenance"), Mapping) else {},
        review_status=review_status,
        notes=str(row.get("notes", "")),
    )


def _reject_forbidden_keys(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in _FORBIDDEN_KEYS:
                raise ValueError(f"{path}: forbidden document-specific field {key!r}.")
            _reject_forbidden_keys(nested, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            _reject_forbidden_keys(nested, f"{path}[{index}]")


def _compile_optional(value: Any, source: str, field: str) -> re.Pattern[str] | None:
    pattern = _optional_string(value)
    if pattern is None:
        return None
    try:
        return re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)
    except re.error as error:
        raise ValueError(f"{source}: invalid {field}: {error}") from error


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Iterable) or isinstance(value, Mapping):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _rule_to_data(rule: Phase1Rule) -> dict[str, Any]:
    row: dict[str, Any] = {
        "rule_id": rule.rule_id,
        "stage": rule.stage,
        "entity_type": rule.entity_type,
        "normalized_mention": rule.normalized_mention,
        "action": rule.action,
        "left_regex": rule.left_regex.pattern if rule.left_regex else None,
        "right_regex": rule.right_regex.pattern if rule.right_regex else None,
        "window_regex": rule.window_regex.pattern if rule.window_regex else None,
        "blocked_window_regex": rule.blocked_window_regex.pattern
        if rule.blocked_window_regex
        else None,
        "replacement_type": rule.replacement_type,
        "assertions": list(rule.assertions),
        "candidates": list(rule.candidates),
        "confidence_tier": rule.confidence_tier,
        "provenance": dict(rule.provenance or {}),
        "review_status": rule.review_status,
        "notes": rule.notes,
    }
    return {key: value for key, value in row.items() if value not in (None, "", [], {})}
