from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from medical_kg_nlp.schema.annotation import AssertionEvidence
from medical_kg_nlp.schema.types import AssertionStatus


DEFAULT_ASSERTION_CUE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "heuristics" / "assertion_cues.jsonl"
)
VALID_SCOPES = {"left", "right", "bidirectional", "section_prior"}


@dataclass(frozen=True)
class AssertionCue:
    rule_id: str
    cue: str
    assertion: AssertionStatus
    language: str
    scope: str
    source_ids: tuple[str, ...]
    notes: str = ""
    priority: int = 100
    max_distance: int = 120


class AssertionRuleRegistry:
    def __init__(self, cues: list[AssertionCue]) -> None:
        self.cues = tuple(cues)
        rule_ids = [cue.rule_id for cue in cues]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("Assertion cue rule_id values must be unique.")

    def evidence(
        self,
        assertion: AssertionStatus,
        cue: str,
        *,
        scope: str,
    ) -> AssertionEvidence:
        candidates = [
            item
            for item in self.cues
            if item.assertion == assertion
            and item.cue.casefold() == cue.casefold()
            and (item.scope == scope or item.scope == "bidirectional")
        ]
        if not candidates:
            raise KeyError(f"No assertion rule for {assertion.value}:{scope}:{cue}")
        selected = sorted(
            candidates,
            key=lambda item: (-item.priority, item.scope != scope, item.rule_id),
        )[0]
        return AssertionEvidence(
            rule_id=selected.rule_id,
            assertion=selected.assertion,
            cue=selected.cue,
            scope=scope,
        )

    def rules(
        self,
        assertion: AssertionStatus,
        *,
        scope: str,
    ) -> tuple[AssertionCue, ...]:
        """Return executable rules in deterministic priority order."""
        rules = [
            item
            for item in self.cues
            if item.assertion == assertion
            and (item.scope == scope or item.scope == "bidirectional")
        ]
        return tuple(
            sorted(
                rules,
                key=lambda item: (-item.priority, item.scope != scope, item.rule_id),
            )
        )

    def section_prior(self, title: str) -> AssertionCue | None:
        normalized = title.casefold().strip()
        candidates = [
            item
            for item in self.cues
            if item.scope == "section_prior" and item.cue.casefold().strip() == normalized
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (-item.priority, item.rule_id))[0]

    @staticmethod
    def evidence_for_rule(rule: AssertionCue, *, scope: str) -> AssertionEvidence:
        return AssertionEvidence(
            rule_id=rule.rule_id,
            assertion=rule.assertion,
            cue=rule.cue,
            scope=scope,
        )


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
    if DEFAULT_ASSERTION_CUE_PATH.exists():
        return load_assertion_cues(DEFAULT_ASSERTION_CUE_PATH)
    resource = files("medical_kg_nlp").joinpath("resources/assertion_cues.jsonl")
    if not resource.is_file():
        raise FileNotFoundError("Packaged assertion cue resource is missing.")
    cues: list[AssertionCue] = []
    with resource.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{resource}:{line_number}: expected JSON object.")
            cues.append(_cue_from_row(row, Path(str(resource)), line_number))
    return cues


def cues_by_assertion(
    cues: list[AssertionCue], *, scope: str | None = None
) -> dict[AssertionStatus, tuple[str, ...]]:
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
    priority = _integer(row.get("priority", 100), path, line_number, "priority", minimum=0)
    max_distance = _integer(
        row.get("max_distance", 120),
        path,
        line_number,
        "max_distance",
        minimum=0,
    )
    return AssertionCue(
        rule_id=str(row.get("rule_id") or _derived_rule_id(assertion, scope, cue)),
        cue=cue,
        assertion=assertion,
        language=str(row.get("language", "")).strip() or "unknown",
        scope=scope,
        source_ids=source_ids,
        notes=str(row.get("notes", "")),
        priority=priority,
        max_distance=max_distance,
    )


def _source_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def _derived_rule_id(assertion: AssertionStatus, scope: str, cue: str) -> str:
    payload = f"{assertion.value}\0{scope}\0{cue.casefold()}".encode()
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"CUE_{assertion.value}_{scope.upper()}_{digest}"


def _integer(
    value: Any,
    path: Path,
    line_number: int,
    field: str,
    *,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path}:{line_number}: {field} must be an integer >= {minimum}.")
    return value
