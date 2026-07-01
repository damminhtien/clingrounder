from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.schema.types import AssertionStatus


DEFAULT_ASSERTION_CUE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "heuristics" / "assertion_cues.jsonl"
)
VALID_SCOPES = {"left", "right", "bidirectional", "section_prior"}


@dataclass(frozen=True)
class AssertionCue:
    cue: str
    assertion: AssertionStatus
    language: str
    scope: str
    source_ids: tuple[str, ...]
    notes: str = ""


def load_assertion_cues(path: str | Path = DEFAULT_ASSERTION_CUE_PATH) -> list[AssertionCue]:
    cue_path = Path(path)
    cues: list[AssertionCue] = []
    with cue_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{cue_path}:{line_number}: expected JSON object.")
            cues.append(_cue_from_row(row, cue_path, line_number))
    return cues


def load_default_assertion_cues() -> list[AssertionCue]:
    if not DEFAULT_ASSERTION_CUE_PATH.exists():
        return []
    return load_assertion_cues(DEFAULT_ASSERTION_CUE_PATH)


def cues_by_assertion(cues: list[AssertionCue], *, scope: str | None = None) -> dict[AssertionStatus, tuple[str, ...]]:
    grouped: dict[AssertionStatus, list[str]] = {}
    for cue in cues:
        if scope is not None and cue.scope != scope:
            continue
        grouped.setdefault(cue.assertion, [])
        if cue.cue not in grouped[cue.assertion]:
            grouped[cue.assertion].append(cue.cue)
    return {assertion: tuple(values) for assertion, values in grouped.items()}


def section_priors_from_cues(cues: list[AssertionCue]) -> dict[str, AssertionStatus]:
    priors: dict[str, AssertionStatus] = {}
    for cue in cues:
        if cue.scope == "section_prior":
            priors[cue.cue.lower().strip()] = cue.assertion
    return priors


def _cue_from_row(row: dict[str, Any], path: Path, line_number: int) -> AssertionCue:
    cue = str(row.get("cue", "")).strip()
    if not cue:
        raise ValueError(f"{path}:{line_number}: cue must be non-empty.")
    assertion = AssertionStatus(str(row.get("assertion", "")))
    scope = str(row.get("scope", "")).strip()
    if scope not in VALID_SCOPES:
        raise ValueError(f"{path}:{line_number}: invalid scope {scope!r}.")
    source_ids = _source_ids(row.get("source_ids"))
    if not source_ids:
        raise ValueError(f"{path}:{line_number}: source_ids must be non-empty.")
    return AssertionCue(
        cue=cue,
        assertion=assertion,
        language=str(row.get("language", "")).strip() or "unknown",
        scope=scope,
        source_ids=source_ids,
        notes=str(row.get("notes", "")),
    )


def _source_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value if str(item).strip())
