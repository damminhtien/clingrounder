"""Context-sensitive suppression rules for dictionary entity proposals.

The rule engine is reusable; rule inventories are supplied by the application or
benchmark that owns them. Importing the package never discovers repository-local
heuristics implicitly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clingrounder.utils.text import normalize_for_match


__all__ = ["FalsePositiveRule", "load_false_positive_rules"]


@dataclass(frozen=True)
class FalsePositiveRule:
    alias: str
    rule_id: str = ""
    priority: int = 0
    source: str = ""
    examples_positive: tuple[str, ...] = ()
    examples_negative: tuple[str, ...] = ()
    match_mode: str = "exact"
    left_regex: re.Pattern[str] | None = None
    right_regex: re.Pattern[str] | None = None
    context_regex: re.Pattern[str] | None = None
    right_initial_uppercase: bool = False
    context_radius: int = 40
    notes: str = ""

    def blocks(self, alias: str, text: str, span: tuple[int, int]) -> bool:
        normalized_alias = normalize_for_match(alias)
        normalized_target = normalize_for_match(self.alias)
        if self.match_mode == "prefix":
            matches_target = normalized_alias.startswith(normalized_target)
        else:
            matches_target = normalized_alias == normalized_target
        if not matches_target:
            return False
        start, end = span
        left = text[max(0, start - self.context_radius) : start]
        right = text[end : min(len(text), end + self.context_radius)]
        context = text[
            max(0, start - self.context_radius) : min(len(text), end + self.context_radius)
        ]
        if self.left_regex is not None and self.left_regex.search(left):
            return True
        if self.right_regex is not None and self.right_regex.search(right):
            return True
        if self.context_regex is not None and self.context_regex.search(context):
            return True
        if self.right_initial_uppercase:
            right_token = right.lstrip()
            return bool(right_token and right_token[0].isalpha() and right_token[0].isupper())
        return False


def load_false_positive_rules(
    path: str | Path | None = None,
) -> tuple[FalsePositiveRule, ...]:
    """Load an explicitly selected rule table, or no rules when omitted."""

    if path is None:
        return ()
    rule_path = Path(path)
    if not rule_path.exists():
        raise FileNotFoundError(f"False-positive rule file does not exist: {rule_path}")
    with rule_path.open("r", encoding="utf-8") as handle:
        return _load_rules(handle, rule_path)


def _load_rules(handle: Any, path: Path) -> tuple[FalsePositiveRule, ...]:
    rules: list[FalsePositiveRule] = []
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object.")
        rules.append(_rule_from_row(row, path, line_number))
    return tuple(sorted(rules, key=lambda rule: (-rule.priority, rule.rule_id)))


def _rule_from_row(row: dict[str, Any], path: Path, line_number: int) -> FalsePositiveRule:
    alias = str(row.get("alias", "")).strip()
    if not alias:
        raise ValueError(f"{path}:{line_number}: alias must be non-empty.")
    return FalsePositiveRule(
        alias=alias,
        rule_id=str(row.get("rule_id", f"fp-{line_number}")),
        priority=int(row.get("priority", 0)),
        source=str(row.get("source", "")),
        examples_positive=_strings(row.get("examples_positive")),
        examples_negative=_strings(row.get("examples_negative")),
        match_mode=_match_mode(row.get("match_mode"), path, line_number),
        left_regex=_compile_optional(row.get("left_regex"), path, line_number, "left_regex"),
        right_regex=_compile_optional(row.get("right_regex"), path, line_number, "right_regex"),
        context_regex=_compile_optional(
            row.get("context_regex"), path, line_number, "context_regex"
        ),
        right_initial_uppercase=bool(row.get("right_initial_uppercase", False)),
        context_radius=int(row.get("context_radius", 40)),
        notes=str(row.get("notes", "")),
    )


def _compile_optional(
    value: Any, path: Path, line_number: int, field: str
) -> re.Pattern[str] | None:
    if value is None:
        return None
    pattern = str(value).strip()
    if not pattern:
        return None
    try:
        return re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)
    except re.error as error:
        raise ValueError(f"{path}:{line_number}: invalid {field}: {error}") from error


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value)


def _match_mode(value: Any, path: Path, line_number: int) -> str:
    mode = str(value or "exact").strip()
    if mode not in {"exact", "prefix"}:
        raise ValueError(f"{path}:{line_number}: match_mode must be exact or prefix.")
    return mode
