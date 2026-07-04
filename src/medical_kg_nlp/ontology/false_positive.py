from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.utils.text import normalize_for_match


DEFAULT_FALSE_POSITIVE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "heuristics" / "false_positive_blacklist.jsonl"
)


@dataclass(frozen=True)
class FalsePositiveRule:
    alias: str
    left_regex: re.Pattern[str] | None = None
    right_regex: re.Pattern[str] | None = None
    context_regex: re.Pattern[str] | None = None
    right_initial_uppercase: bool = False
    context_radius: int = 40
    notes: str = ""

    def blocks(self, alias: str, text: str, span: tuple[int, int]) -> bool:
        if normalize_for_match(alias) != normalize_for_match(self.alias):
            return False
        start, end = span
        left = text[max(0, start - self.context_radius) : start]
        right = text[end : min(len(text), end + self.context_radius)]
        context = text[max(0, start - self.context_radius) : min(len(text), end + self.context_radius)]
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


def load_false_positive_rules(path: str | Path | None = DEFAULT_FALSE_POSITIVE_PATH) -> tuple[FalsePositiveRule, ...]:
    if path is None:
        return ()
    rule_path = Path(path)
    if not rule_path.exists():
        return ()
    rules: list[FalsePositiveRule] = []
    with rule_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{rule_path}:{line_number}: expected JSON object.")
            rules.append(_rule_from_row(row, rule_path, line_number))
    return tuple(rules)


def _rule_from_row(row: dict[str, Any], path: Path, line_number: int) -> FalsePositiveRule:
    alias = str(row.get("alias", "")).strip()
    if not alias:
        raise ValueError(f"{path}:{line_number}: alias must be non-empty.")
    return FalsePositiveRule(
        alias=alias,
        left_regex=_compile_optional(row.get("left_regex"), path, line_number, "left_regex"),
        right_regex=_compile_optional(row.get("right_regex"), path, line_number, "right_regex"),
        context_regex=_compile_optional(row.get("context_regex"), path, line_number, "context_regex"),
        right_initial_uppercase=bool(row.get("right_initial_uppercase", False)),
        context_radius=int(row.get("context_radius", 40)),
        notes=str(row.get("notes", "")),
    )


def _compile_optional(value: Any, path: Path, line_number: int, field: str) -> re.Pattern[str] | None:
    if value is None:
        return None
    pattern = str(value).strip()
    if not pattern:
        return None
    try:
        return re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)
    except re.error as error:
        raise ValueError(f"{path}:{line_number}: invalid {field}: {error}") from error

