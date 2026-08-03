"""Optional assertion overlays owned by the archived Phase 1 benchmark."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PHASE1_ASSERTION_VALUES = {"isNegated", "isFamily", "isHistorical"}

__all__ = ["Phase1AssertionOverlay", "load_phase1_assertion_overlays"]


@dataclass(frozen=True)
class Phase1AssertionOverlay:
    assertion: str
    entity_types: tuple[str, ...] = ()
    left_regex: re.Pattern[str] | None = None
    right_regex: re.Pattern[str] | None = None
    window_regex: re.Pattern[str] | None = None
    blocked_left_regex: re.Pattern[str] | None = None
    blocked_window_regex: re.Pattern[str] | None = None
    left_chars: int = 120
    right_chars: int = 80
    notes: str = ""

    def matches(self, source_text: str, span: tuple[int, int], *, entity_type: str) -> bool:
        if self.entity_types and entity_type not in self.entity_types:
            return False
        start, end = span
        left = source_text[max(0, start - self.left_chars) : start]
        right = source_text[end : min(len(source_text), end + self.right_chars)]
        window = source_text[max(0, start - self.left_chars) : min(len(source_text), end + self.right_chars)]
        if self.blocked_left_regex is not None and self.blocked_left_regex.search(left):
            return False
        if self.blocked_window_regex is not None and self.blocked_window_regex.search(window):
            return False
        positive_patterns = (
            (self.left_regex, left),
            (self.right_regex, right),
            (self.window_regex, window),
        )
        matched = False
        for pattern, text in positive_patterns:
            if pattern is None:
                continue
            matched = True
            if pattern.search(text) is None:
                return False
        return matched


def load_phase1_assertion_overlays(
    path: str | Path | None,
) -> tuple[Phase1AssertionOverlay, ...]:
    """Load a benchmark-owned overlay without discovering local files implicitly."""

    if path is None:
        return ()
    overlay_path = Path(path)
    if not overlay_path.exists():
        raise FileNotFoundError(f"Phase 1 assertion overlay does not exist: {overlay_path}")
    overlays: list[Phase1AssertionOverlay] = []
    with overlay_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{overlay_path}:{line_number}: expected JSON object.")
            overlays.append(_overlay_from_row(row, overlay_path, line_number))
    return tuple(overlays)


def _overlay_from_row(row: dict[str, Any], path: Path, line_number: int) -> Phase1AssertionOverlay:
    assertion = str(row.get("assertion", "")).strip()
    if assertion not in PHASE1_ASSERTION_VALUES:
        raise ValueError(f"{path}:{line_number}: invalid assertion {assertion!r}.")
    return Phase1AssertionOverlay(
        assertion=assertion,
        entity_types=_string_tuple(row.get("entity_types")),
        left_regex=_compile_optional(row.get("left_regex"), path, line_number, "left_regex"),
        right_regex=_compile_optional(row.get("right_regex"), path, line_number, "right_regex"),
        window_regex=_compile_optional(row.get("window_regex"), path, line_number, "window_regex"),
        blocked_left_regex=_compile_optional(row.get("blocked_left_regex"), path, line_number, "blocked_left_regex"),
        blocked_window_regex=_compile_optional(
            row.get("blocked_window_regex"),
            path,
            line_number,
            "blocked_window_regex",
        ),
        left_chars=int(row.get("left_chars", 120)),
        right_chars=int(row.get("right_chars", 80)),
        notes=str(row.get("notes", "")),
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


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
