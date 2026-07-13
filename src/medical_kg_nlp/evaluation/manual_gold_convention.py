from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from medical_kg_nlp.ner.medication_list_parser import MedicationListParser
from medical_kg_nlp.ner.medication_mention_parser import MedicationMentionParser
from medical_kg_nlp.ontology.phase1 import (
    PHASE1_ASSERTABLE_TYPES,
    PHASE1_CODABLE_TYPES,
)
from medical_kg_nlp.utils.io import read_source_text
from medical_kg_nlp.utils.text import normalize_for_match


def audit_manual_gold_convention(
    input_dir: str | Path,
    gold_dir: str | Path,
    *,
    expected_count: int = 100,
    decisions_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit manual Phase 1 labels against conventions demonstrated by BTC samples."""
    input_root = Path(input_dir)
    gold_root = Path(gold_dir)
    medication_lists = MedicationListParser()
    medication_mentions = MedicationMentionParser()
    active_decisions = _load_decisions(
        Path(decisions_path) if decisions_path is not None else gold_root / "convention_decisions.jsonl"
    )
    issues: list[dict[str, Any]] = []
    resolutions: list[dict[str, Any]] = []
    used_decisions: set[tuple[str, str, str]] = set()
    reviewed_ids: list[str] = []
    entity_count = 0
    candidate_evidence: dict[
        tuple[str, str], dict[tuple[str, ...], set[str]]
    ] = defaultdict(lambda: defaultdict(set))

    for index in range(1, expected_count + 1):
        document_id = str(index)
        source_path = input_root / f"{document_id}.txt"
        gold_path = gold_root / f"{document_id}.json"
        if not gold_path.exists():
            issues.append(
                _issue(
                    document_id,
                    "missing_gold",
                    "blocking",
                    "Manual-gold file is missing.",
                )
            )
            continue
        if not source_path.exists():
            issues.append(
                _issue(
                    document_id,
                    "missing_source",
                    "blocking",
                    "Raw source file is missing.",
                )
            )
            continue

        rows = json.loads(gold_path.read_text(encoding="utf-8"))
        source_text = read_source_text(source_path)
        reviewed_ids.append(document_id)
        entity_count += len(rows)
        list_items = medication_lists.items(source_text)
        seen: set[tuple[str, int, int]] = set()
        entity_spans = [
            (row_index, int(span[0]), int(span[1]))
            for row_index, row in enumerate(rows)
            if isinstance((span := row.get("position")), list)
            and len(span) == 2
            and all(isinstance(value, int) for value in span)
        ]

        for row_index, row in enumerate(rows):
            entity_type = str(row.get("type", ""))
            span = row.get("position")
            if not (
                isinstance(span, list)
                and len(span) == 2
                and all(isinstance(value, int) for value in span)
            ):
                continue
            start, end = span
            key = (entity_type, start, end)
            if key in seen:
                issues.append(
                    _row_issue(
                        document_id,
                        row_index,
                        row,
                        "duplicate_span_type",
                        "blocking",
                        "The same span and entity type occurs more than once.",
                    )
                )
            seen.add(key)

            assertions = row.get("assertions", [])
            candidates = row.get("candidates", [])
            if entity_type not in PHASE1_ASSERTABLE_TYPES and assertions:
                issues.append(
                    _row_issue(
                        document_id,
                        row_index,
                        row,
                        "nonassertable_assertion",
                        "blocking",
                        "Assertions are only valid for symptoms, diagnoses, and drugs.",
                    )
                )
            if entity_type not in PHASE1_CODABLE_TYPES and candidates:
                issues.append(
                    _row_issue(
                        document_id,
                        row_index,
                        row,
                        "noncodable_candidate",
                        "blocking",
                        "Candidates are only valid for diagnoses and drugs.",
                    )
                )
            if entity_type in PHASE1_CODABLE_TYPES and isinstance(candidates, list) and len(candidates) > 1:
                # Multi-code rows can be valid for fused drug names or one span that states two diagnoses.
                # Keep them visible for adjudication instead of silently forcing top-1.
                message = (
                    "BTC examples use one exact RxCUI per structured drug; verify that the "
                    "span really fuses multiple drugs."
                    if entity_type == "THUỐC"
                    else "Verify that the diagnosis span explicitly states multiple codable conditions."
                )
                review_issue = _row_issue(
                    document_id,
                    row_index,
                    row,
                    "multi_candidate_review",
                    "review",
                    message,
                )
                decision_key = (
                    "multi_candidate_review",
                    entity_type,
                    normalize_for_match(str(row.get("text", ""))),
                )
                decision = active_decisions.get(decision_key)
                if decision is None:
                    issues.append(review_issue)
                else:
                    used_decisions.add(decision_key)
                    resolutions.append({**review_issue, "decision": decision})
            if entity_type in PHASE1_CODABLE_TYPES and isinstance(candidates, list):
                mention = normalize_for_match(str(row.get("text", "")))
                if mention:
                    # Candidate order is irrelevant to Jaccard; sort before comparing mappings.
                    signature = tuple(sorted(str(candidate) for candidate in candidates))
                    candidate_evidence[(entity_type, mention)][signature].add(document_id)

            if entity_type == "THUỐC":
                parsed = medication_mentions.parse(source_text, (start, end))
                if parsed.full_span[1] > end and not _overlaps_other_entity(
                    parsed.full_span,
                    row_index,
                    entity_spans,
                ):
                    # Glued source tokens such as "morphineiv morphine" can make the parser
                    # consume the next reviewed drug. An overlap is evidence to stop, not expand.
                    issues.append(
                        _row_issue(
                            document_id,
                            row_index,
                            row,
                            "medication_boundary_under",
                            "review",
                            "A contiguous dose/form/route/frequency suffix can extend the medication span.",
                            suggested_position=list(parsed.full_span),
                            suggested_text=source_text[parsed.full_span[0] : parsed.full_span[1]],
                        )
                    )

            for item in list_items:
                # The official BTC medication sample treats each numbered line as one full SIG and
                # keeps the indication outside that medication span.
                if entity_type == "THUỐC" and _contains(item.medication_span, (start, end)):
                    if (start, end) != item.medication_span:
                        issues.append(
                            _row_issue(
                                document_id,
                                row_index,
                                row,
                                "medication_list_boundary",
                                "blocking",
                                "A numbered BTC-style medication item must include the complete medication SIG before the indication.",
                                suggested_position=list(item.medication_span),
                                suggested_text=source_text[item.medication_span[0] : item.medication_span[1]],
                            )
                        )
                    if "isHistorical" not in assertions:
                        issues.append(
                            _row_issue(
                                document_id,
                                row_index,
                                row,
                                "medication_list_history",
                                "blocking",
                                "Drugs in the pre-admission medication list must be historical.",
                            )
                        )
                if (
                    entity_type == "TRIỆU_CHỨNG"
                    and item.indication_span is not None
                    and _contains(item.indication_span, (start, end))
                    and assertions
                ):
                    issues.append(
                        _row_issue(
                            document_id,
                            row_index,
                            row,
                            "indication_assertion",
                            "blocking",
                            "Medication indications in the BTC sample have empty assertions.",
                        )
                    )

        _audit_overlapping_entities(document_id, rows, entity_spans, issues)

    _audit_candidate_mapping_consistency(
        candidate_evidence,
        active_decisions,
        used_decisions,
        issues,
        resolutions,
    )
    for decision_key in sorted(set(active_decisions) - used_decisions):
        issues.append(
            _issue(
                "",
                "unused_convention_decision",
                "review",
                "Convention decision no longer matches any current manual-gold issue.",
                decision_key=list(decision_key),
            )
        )

    severity_counts = Counter(str(issue["severity"]) for issue in issues)
    kind_counts = Counter(str(issue["kind"]) for issue in issues)
    return {
        "schema_version": "phase1-manual-gold-convention.v1",
        "expected_count": expected_count,
        "reviewed_count": len(reviewed_ids),
        "missing_count": expected_count - len(reviewed_ids),
        "entity_count": entity_count,
        "blocking_count": severity_counts["blocking"],
        "review_count": severity_counts["review"],
        "resolved_count": len(resolutions),
        "by_kind": dict(sorted(kind_counts.items())),
        "reviewed_ids": reviewed_ids,
        "issues": issues,
        "resolutions": resolutions,
    }


def write_manual_gold_convention_report(report: dict[str, Any], output_dir: str | Path) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Phase 1 Manual-Gold Convention Audit",
        "",
        f"- Reviewed documents: {report['reviewed_count']}/{report['expected_count']}",
        f"- Entities: {report['entity_count']}",
        f"- Blocking issues: {report['blocking_count']}",
        f"- Review issues: {report['review_count']}",
        f"- Accepted decisions: {report.get('resolved_count', 0)}",
        "",
        "## Issues By Kind",
        "",
    ]
    for kind, count in report["by_kind"].items():
        lines.append(f"- `{kind}`: {count}")
    lines.append("")
    (output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _contains(container: tuple[int, int], inner: tuple[int, int]) -> bool:
    return container[0] <= inner[0] and inner[1] <= container[1]


def _overlaps_other_entity(
    span: tuple[int, int],
    row_index: int,
    entity_spans: list[tuple[int, int, int]],
) -> bool:
    return any(
        other_index != row_index and span[0] < other_end and other_start < span[1]
        for other_index, other_start, other_end in entity_spans
    )


def _audit_overlapping_entities(
    document_id: str,
    rows: list[dict[str, Any]],
    entity_spans: list[tuple[int, int, int]],
    issues: list[dict[str, Any]],
) -> None:
    ordered = sorted(entity_spans, key=lambda item: (item[1], item[2], item[0]))
    for offset, (left_index, left_start, left_end) in enumerate(ordered):
        for right_index, right_start, right_end in ordered[offset + 1 :]:
            if right_start >= left_end:
                break
            left = rows[left_index]
            right = rows[right_index]
            if (left_start, left_end, left.get("type")) == (
                right_start,
                right_end,
                right.get("type"),
            ):
                # duplicate_span_type already reports the exact duplicate with a clearer kind.
                continue
            issues.append(
                _issue(
                    document_id,
                    "overlapping_entities",
                    "blocking",
                    "BTC-style output uses one longest non-overlapping entity for nested spans.",
                    left={
                        "row_index": left_index,
                        "text": left.get("text"),
                        "type": left.get("type"),
                        "position": [left_start, left_end],
                    },
                    right={
                        "row_index": right_index,
                        "text": right.get("text"),
                        "type": right.get("type"),
                        "position": [right_start, right_end],
                    },
                )
            )


def _audit_candidate_mapping_consistency(
    evidence: dict[tuple[str, str], dict[tuple[str, ...], set[str]]],
    decisions: dict[tuple[str, str, str], dict[str, Any]],
    used_decisions: set[tuple[str, str, str]],
    issues: list[dict[str, Any]],
    resolutions: list[dict[str, Any]],
) -> None:
    for (entity_type, mention), mappings in sorted(evidence.items()):
        if len(mappings) <= 1:
            continue
        review_issue = _issue(
            "",
            "candidate_mapping_conflict",
            "review",
            "The same normalized mention has multiple candidate sets; standardize it or document a contextual mapping rule.",
            entity_type=entity_type,
            normalized_mention=mention,
            mappings=[
                {
                    "candidates": list(candidates),
                    "document_ids": sorted(document_ids, key=_document_sort_key),
                }
                for candidates, document_ids in sorted(mappings.items())
            ],
        )
        decision_key = ("candidate_mapping_conflict", entity_type, mention)
        decision = decisions.get(decision_key)
        if decision is None:
            issues.append(review_issue)
            continue
        used_decisions.add(decision_key)
        resolutions.append({**review_issue, "decision": decision})


def _load_decisions(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    decisions: dict[tuple[str, str, str], dict[str, Any]] = {}
    forbidden_fields = {"document_id", "position", "span", "start", "end"}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object.")
            forbidden = forbidden_fields & set(row)
            if forbidden:
                raise ValueError(
                    f"{path}:{line_number}: convention decisions cannot contain "
                    f"document-specific fields {sorted(forbidden)}."
                )
            if row.get("decision") != "allow":
                continue
            kind = str(row.get("kind", "")).strip()
            entity_type = str(row.get("entity_type", "")).strip()
            mention = normalize_for_match(str(row.get("normalized_mention", "")))
            if not kind or not entity_type or not mention or not str(row.get("reason", "")).strip():
                raise ValueError(
                    f"{path}:{line_number}: kind, entity_type, normalized_mention, and reason are required."
                )
            key = (kind, entity_type, mention)
            if key in decisions:
                raise ValueError(f"{path}:{line_number}: duplicate convention decision {key!r}.")
            decisions[key] = row
    return decisions


def _issue(
    document_id: str,
    kind: str,
    severity: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "kind": kind,
        "severity": severity,
        "message": message,
        **details,
    }


def _row_issue(
    document_id: str,
    row_index: int,
    row: dict[str, Any],
    kind: str,
    severity: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    return _issue(
        document_id,
        kind,
        severity,
        message,
        row_index=row_index,
        text=row.get("text"),
        entity_type=row.get("type"),
        position=row.get("position"),
        **details,
    )


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)
