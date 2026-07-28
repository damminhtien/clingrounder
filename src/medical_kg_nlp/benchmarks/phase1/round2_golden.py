"""Build auditable inferred labels for the private Phase 1 Round 2 corpus.

The organizer's public medication-list example is an executable annotation
specification, but it is not enough to recover hidden labels for Round 2.
This module therefore keeps high-confidence consensus labels separate from
the wider review union and marks both as inferred, non-official supervision.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.benchmarks.phase1.phase1 import (
    validate_phase1_submission_documents,
    zip_phase1_output_dir,
)
from medical_kg_nlp.benchmarks.phase1.phase1_selective_overlays import (
    apply_selective_assertions,
)
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.ner.medication_mention_parser import MedicationMentionParser
from medical_kg_nlp.ontology.phase1 import (
    PHASE1_ALLOWED_TYPES,
    PHASE1_CODABLE_TYPES,
    expected_code_system,
)
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = [
    "BTC_PHASE1_INFERRED_GOLD_POLICY",
    "build_phase1_round2_golden",
    "write_phase1_round2_golden",
]


BTC_PHASE1_INFERRED_GOLD_POLICY: dict[str, Any] = {
    "schema_version": "phase1-btc-inferred-gold-policy.v1",
    "official_gold": False,
    "rules": [
        {
            "rule_id": "btc.raw-offset.v1",
            "description": "Every position is a raw [start, end) span and must reproduce text.",
        },
        {
            "rule_id": "btc.medication-full-span.v1",
            "description": (
                "Medication spans include contiguous strength, form, release, route, and "
                "frequency attributes, and stop before the treatment indication."
            ),
        },
        {
            "rule_id": "btc.medication-indication.v1",
            "description": "A symptom after an indication marker is labeled separately.",
        },
        {
            "rule_id": "btc.pre-admission-history.v1",
            "description": "Medication entities in a pre-admission medication list are historical.",
        },
        {
            "rule_id": "btc.typed-candidate.v1",
            "description": (
                "Diagnosis candidates are ICD-10, medication candidates are RxNorm, and "
                "the other three entity types do not emit candidates."
            ),
        },
    ],
}


@dataclass(slots=True)
class _ProposalGroup:
    document_id: str
    text: str
    entity_type: str
    span: tuple[int, int]
    rows_by_source: dict[str, dict[str, Any]]

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted(self.rows_by_source))

    @property
    def support(self) -> int:
        return len(self.rows_by_source)

    @property
    def key(self) -> tuple[int, int, str]:
        return (self.span[0], self.span[1], self.entity_type)


def build_phase1_round2_golden(
    source_text_by_doc: Mapping[str, str],
    sources: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    dictionary: DictionaryStore,
    *,
    minimum_sources: int = 2,
) -> dict[str, Any]:
    """Compile strict consensus and a wider review layer from independent proposals.

    `gold_strict` is suitable only as high-confidence weak supervision. It is
    intentionally sparse when sources disagree. `gold_review` is a proposal
    surface for manual adjudication and must not be used as reviewed training
    data until its queue has been resolved.
    """

    if len(sources) < 2:
        raise ValueError("Round 2 inferred gold requires at least two proposal sources")
    if minimum_sources < 2 or minimum_sources > len(sources):
        raise ValueError("minimum_sources must be between two and the source count")
    source_names = tuple(sorted(sources))
    if any(not name.strip() for name in source_names):
        raise ValueError("Proposal source names must be non-empty")
    expected_documents = set(source_text_by_doc)
    for source_name, rows_by_doc in sources.items():
        if set(rows_by_doc) != expected_documents:
            missing = sorted(expected_documents - set(rows_by_doc), key=_document_sort_key)
            extra = sorted(set(rows_by_doc) - expected_documents, key=_document_sort_key)
            raise ValueError(
                f"Proposal source {source_name!r} does not cover the corpus: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

    strict_by_doc: dict[str, list[dict[str, Any]]] = {}
    review_by_doc: dict[str, list[dict[str, Any]]] = {}
    decisions: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    invalid_proposals: list[dict[str, Any]] = []
    candidate_rejections: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    medication_parser = MedicationMentionParser()

    for document_id in sorted(source_text_by_doc, key=_document_sort_key):
        source_text = source_text_by_doc[document_id]
        groups_by_key: dict[tuple[int, int, str], _ProposalGroup] = {}
        for source_name in source_names:
            seen: set[tuple[int, int, str]] = set()
            for row_index, raw_row in enumerate(sources[source_name][document_id]):
                row, rejected_candidates = _validated_row(
                    raw_row,
                    source_text,
                    dictionary,
                )
                if row is None:
                    invalid_proposals.append(
                        {
                            "document_id": document_id,
                            "source": source_name,
                            "row_index": row_index,
                            "reason": "invalid_schema_type_or_offset",
                            "proposal": raw_row,
                        }
                    )
                    counters["proposal.invalid"] += 1
                    continue
                if rejected_candidates:
                    candidate_rejections.append(
                        {
                            "document_id": document_id,
                            "source": source_name,
                            "row_index": row_index,
                            "entity": _identity(row),
                            "rejected_candidates": rejected_candidates,
                        }
                    )
                    counters["candidate.rejected"] += len(rejected_candidates)
                key = _row_key(row)
                if key in seen:
                    counters["proposal.duplicate_within_source"] += 1
                    continue
                seen.add(key)
                group = groups_by_key.setdefault(
                    key,
                    _ProposalGroup(
                        document_id=document_id,
                        text=str(row["text"]),
                        entity_type=str(row["type"]),
                        span=(int(row["position"][0]), int(row["position"][1])),
                        rows_by_source={},
                    ),
                )
                group.rows_by_source[source_name] = row
                counters[f"source.{source_name}.valid"] += 1

        groups = sorted(groups_by_key.values(), key=_group_sort_key)
        selected_groups: dict[tuple[int, int, str], tuple[_ProposalGroup, Sequence[_ProposalGroup], str]] = {}
        group_reason: dict[tuple[int, int, str], str] = {}
        for component in _overlap_components(groups):
            selected = _select_strict_group(
                source_text,
                component,
                minimum_sources=minimum_sources,
                medication_parser=medication_parser,
            )
            if selected is None:
                reason = _component_review_reason(component, minimum_sources)
                for group in component:
                    group_reason[group.key] = reason
                continue
            winner, evidence_groups, reason = selected
            selected_groups[winner.key] = (winner, evidence_groups, reason)
            group_reason[winner.key] = reason
            for group in component:
                if group.key != winner.key:
                    group_reason[group.key] = (
                        "superseded_by_btc_medication_full_span"
                        if reason == "btc_medication_full_span"
                        else "boundary_conflict"
                    )

        strict_rows: list[dict[str, Any]] = []
        review_rows: list[dict[str, Any]] = []
        for group in groups:
            review_row = _row_from_group(
                group,
                evidence_groups=(group,),
                minimum_sources=minimum_sources,
            )
            review_rows.append(review_row)
            selected = selected_groups.get(group.key)
            if selected is not None:
                winner, evidence_groups, reason = selected
                strict_row = _row_from_group(
                    winner,
                    evidence_groups=evidence_groups,
                    minimum_sources=minimum_sources,
                )
                strict_rows.append(strict_row)
                decisions.append(
                    {
                        "document_id": document_id,
                        "action": "accept_strict",
                        "reason": reason,
                        "sources": list(_component_sources(evidence_groups)),
                        "entity": _identity(strict_row),
                    }
                )
                counters[f"strict.{reason}"] += 1
                continue

            reason = group_reason.get(group.key, "source_only")
            review_queue.append(
                {
                    "document_id": document_id,
                    "reason": reason,
                    "source_count": group.support,
                    "sources": list(group.sources),
                    "text": group.text,
                    "normalized_mention": (
                        normalize_for_match(group.text) or group.text.casefold()
                    ),
                    "type": group.entity_type,
                    "position": list(group.span),
                    "context": _context_window(source_text, group.span),
                    "candidate_suggestions": _candidate_suggestions((group,)),
                    "review_decision": "",
                    "review_notes": "",
                }
            )
            counters[f"review.{reason}"] += 1

        strict_by_doc[document_id] = _deduplicate_rows(strict_rows)
        review_by_doc[document_id] = _deduplicate_rows(review_rows)

    strict_by_doc, strict_assertion_decisions, strict_assertion_counters = (
        apply_selective_assertions(
            strict_by_doc,
            source_text_by_doc,
            regimes=("negation", "history"),
        )
    )
    review_by_doc, _, _ = apply_selective_assertions(
        review_by_doc,
        source_text_by_doc,
        regimes=("negation", "history"),
    )
    strict_by_doc = _apply_btc_external_contract(strict_by_doc)
    review_by_doc = _apply_btc_external_contract(review_by_doc)
    decisions.extend(strict_assertion_decisions)
    counters.update(strict_assertion_counters)
    _attach_review_metadata(review_queue, review_by_doc)
    review_groups = _group_review_queue(review_queue)

    strict_type_counts = Counter(
        str(row["type"]) for rows in strict_by_doc.values() for row in rows
    )
    review_type_counts = Counter(
        str(row["type"]) for rows in review_by_doc.values() for row in rows
    )
    summary = {
        "document_count": len(source_text_by_doc),
        "source_count": len(source_names),
        "source_names": list(source_names),
        "minimum_sources": minimum_sources,
        "strict_entity_count": sum(len(rows) for rows in strict_by_doc.values()),
        "review_entity_count": sum(len(rows) for rows in review_by_doc.values()),
        "review_queue_count": len(review_queue),
        "review_group_count": len(review_groups),
        "invalid_proposal_count": len(invalid_proposals),
        "rejected_candidate_count": sum(
            len(row["rejected_candidates"]) for row in candidate_rejections
        ),
        "strict_candidate_count": sum(
            len(row.get("candidates", []))
            for rows in strict_by_doc.values()
            for row in rows
        ),
        "strict_assertion_count": sum(
            len(row["assertions"]) for rows in strict_by_doc.values() for row in rows
        ),
        "strict_type_counts": dict(sorted(strict_type_counts.items())),
        "review_type_counts": dict(sorted(review_type_counts.items())),
        "counters": dict(sorted(counters.items())),
    }
    return {
        "schema_version": "phase1-round2-inferred-gold.v1",
        "official_gold": False,
        "strict_layer": {
            "review_status": "consensus_inferred",
            "training_eligible": "weak_supervision_only",
            "challenge_eligible": False,
        },
        "review_layer": {
            "review_status": "proposed",
            "training_eligible": False,
            "challenge_eligible": False,
        },
        "policy": BTC_PHASE1_INFERRED_GOLD_POLICY,
        "summary": summary,
        "gold_strict": strict_by_doc,
        "gold_review": review_by_doc,
        "decisions": decisions,
        "review_queue": sorted(review_queue, key=_review_sort_key),
        "review_groups": review_groups,
        "invalid_proposals": invalid_proposals,
        "candidate_rejections": candidate_rejections,
    }


def write_phase1_round2_golden(
    report: Mapping[str, Any],
    output_dir: str | Path,
    *,
    documents: Sequence[ClinicalDocument],
    dictionary: DictionaryStore,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Write inferred layers, validate them, and return an artifact manifest."""

    output = Path(output_dir)
    strict_dir = output / "gold_strict"
    review_dir = output / "gold_review"
    strict = _rows_mapping(report.get("gold_strict"), "gold_strict")
    review = _rows_mapping(report.get("gold_review"), "gold_review")
    _write_rows(strict, strict_dir)
    _write_rows(review, review_dir)

    strict_issues = validate_phase1_submission_documents(
        documents,
        strict_dir,
        dictionary=dictionary,
    )
    review_issues = validate_phase1_submission_documents(
        documents,
        review_dir,
        dictionary=dictionary,
    )
    if strict_issues or review_issues:
        issues = [issue.to_json() for issue in (*strict_issues, *review_issues)]
        raise ValueError(f"Round 2 inferred-gold validation failed: {issues[:10]}")

    strict_zip = output / "gold_strict.zip"
    review_zip = output / "gold_review.zip"
    zip_phase1_output_dir(strict_dir, strict_zip)
    zip_phase1_output_dir(review_dir, review_zip)
    _write_json(output / "policy.json", report["policy"])
    _write_json(output / "summary.json", report["summary"])
    _write_jsonl(output / "decisions.jsonl", report.get("decisions", []))
    _write_jsonl(output / "review_queue.jsonl", report.get("review_queue", []))
    _write_jsonl(output / "review_groups.jsonl", report.get("review_groups", []))
    _write_jsonl(output / "invalid_proposals.jsonl", report.get("invalid_proposals", []))
    _write_jsonl(
        output / "candidate_rejections.jsonl",
        report.get("candidate_rejections", []),
    )
    _write_summary_markdown(output / "summary.md", report["summary"])

    manifest = {
        "schema_version": "phase1-round2-inferred-gold-manifest.v1",
        "official_gold": False,
        "warning": (
            "These labels are inferred from proposal consensus and the public BTC example. "
            "They are not organizer-provided ground truth."
        ),
        "strict_layer": report["strict_layer"],
        "review_layer": report["review_layer"],
        "provenance": dict(provenance),
        "summary": report["summary"],
        "artifacts": {
            "gold_strict_dir": str(strict_dir),
            "gold_strict_zip": str(strict_zip),
            "gold_strict_zip_sha256": sha256_file(strict_zip),
            "gold_review_dir": str(review_dir),
            "gold_review_zip": str(review_zip),
            "gold_review_zip_sha256": sha256_file(review_zip),
            "review_queue": str(output / "review_queue.jsonl"),
            "review_groups": str(output / "review_groups.jsonl"),
            "decisions": str(output / "decisions.jsonl"),
        },
        "validation": {
            "strict_issue_count": 0,
            "review_issue_count": 0,
            "offset_contract": "raw_text[start:end] == text",
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _validated_row(
    raw_row: Mapping[str, Any],
    source_text: str,
    dictionary: DictionaryStore,
) -> tuple[dict[str, Any] | None, list[str]]:
    text = raw_row.get("text")
    entity_type = raw_row.get("type")
    position = raw_row.get("position")
    if (
        not isinstance(text, str)
        or not text
        or entity_type not in PHASE1_ALLOWED_TYPES
        or not isinstance(position, list)
        or len(position) != 2
        or not all(isinstance(value, int) for value in position)
    ):
        return None, []
    start, end = position
    if start < 0 or end <= start or end > len(source_text) or source_text[start:end] != text:
        return None, []

    retained: list[str] = []
    rejected: list[str] = []
    raw_candidates = raw_row.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raw_candidates = []
    code_system = expected_code_system(str(entity_type))
    for value in raw_candidates:
        code = str(value).strip()
        if (
            code
            and code_system is not None
            and (code_system, code) in dictionary.by_code_system_code
        ):
            if code not in retained:
                retained.append(code)
        elif code:
            rejected.append(code)
    row = {
        "text": text,
        "type": str(entity_type),
        "assertions": [],
        "position": [start, end],
    }
    if entity_type in PHASE1_CODABLE_TYPES:
        row["candidates"] = retained
    return row, rejected


def _overlap_components(groups: Sequence[_ProposalGroup]) -> list[list[_ProposalGroup]]:
    if not groups:
        return []
    parents = list(range(len(groups)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left in enumerate(groups):
        for right_index in range(left_index + 1, len(groups)):
            right = groups[right_index]
            if right.span[0] >= left.span[1]:
                break
            if _spans_overlap(left.span, right.span):
                union(left_index, right_index)
    components: dict[int, list[_ProposalGroup]] = {}
    for index, group in enumerate(groups):
        components.setdefault(find(index), []).append(group)
    return [
        sorted(component, key=_group_sort_key)
        for _, component in sorted(
            components.items(),
            key=lambda item: min(group.span[0] for group in item[1]),
        )
    ]


def _select_strict_group(
    source_text: str,
    component: Sequence[_ProposalGroup],
    *,
    minimum_sources: int,
    medication_parser: MedicationMentionParser,
) -> tuple[_ProposalGroup, Sequence[_ProposalGroup], str] | None:
    if len(component) == 1:
        group = component[0]
        if group.support >= minimum_sources:
            return group, (group,), "exact_consensus"
        return None
    if _has_same_span_type_conflict(component):
        return None
    medication = _select_btc_medication_full_span(
        source_text,
        component,
        minimum_sources=minimum_sources,
        medication_parser=medication_parser,
    )
    if medication is not None:
        return medication, component, "btc_medication_full_span"
    return None


def _select_btc_medication_full_span(
    source_text: str,
    component: Sequence[_ProposalGroup],
    *,
    minimum_sources: int,
    medication_parser: MedicationMentionParser,
) -> _ProposalGroup | None:
    if any(group.entity_type != "THUỐC" for group in component):
        return None
    candidates: list[tuple[int, int, _ProposalGroup]] = []
    for long_group in component:
        evidence_sources = set(long_group.sources)
        has_structured_extension = False
        for short_group in component:
            if short_group.key == long_group.key:
                continue
            if not _contains(long_group.span, short_group.span):
                break
            parsed = medication_parser.parse(source_text, short_group.span)
            if parsed.full_span != long_group.span or not parsed.components:
                break
            evidence_sources.update(short_group.sources)
            has_structured_extension = True
        else:
            if has_structured_extension and len(evidence_sources) >= minimum_sources:
                candidates.append(
                    (
                        len(evidence_sources),
                        long_group.span[1] - long_group.span[0],
                        long_group,
                    )
                )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2].text), reverse=True)
    return candidates[0][2]


def _row_from_group(
    group: _ProposalGroup,
    *,
    evidence_groups: Sequence[_ProposalGroup],
    minimum_sources: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "text": group.text,
        "type": group.entity_type,
        "assertions": [],
        "position": list(group.span),
    }
    if group.entity_type in PHASE1_CODABLE_TYPES:
        row["candidates"] = _consensus_candidates(
            evidence_groups,
            minimum_sources=minimum_sources,
        )
    return row


def _consensus_candidates(
    groups: Sequence[_ProposalGroup],
    *,
    minimum_sources: int,
) -> list[str]:
    by_source: dict[str, set[str]] = {}
    for group in groups:
        for source, row in group.rows_by_source.items():
            by_source.setdefault(source, set()).update(
                str(value) for value in row.get("candidates", []) if str(value)
            )
    candidate_values = {code for values in by_source.values() for code in values}
    if len(candidate_values) != 1:
        return []
    code = next(iter(candidate_values))
    support = sum(code in values for values in by_source.values())
    return [code] if support >= minimum_sources else []


def _candidate_suggestions(groups: Sequence[_ProposalGroup]) -> list[str]:
    return sorted(
        {
            str(value)
            for group in groups
            for row in group.rows_by_source.values()
            for value in row.get("candidates", [])
            if str(value)
        }
    )


def _component_review_reason(
    component: Sequence[_ProposalGroup],
    minimum_sources: int,
) -> str:
    if _has_same_span_type_conflict(component):
        return "type_conflict"
    if len(component) > 1:
        return "boundary_conflict"
    if component[0].support < minimum_sources:
        return "source_only"
    return "unresolved"


def _has_same_span_type_conflict(component: Sequence[_ProposalGroup]) -> bool:
    for index, left in enumerate(component):
        for right in component[index + 1 :]:
            if left.span == right.span and left.entity_type != right.entity_type:
                return True
    return False


def _attach_review_metadata(
    queue: list[dict[str, Any]],
    review_by_doc: Mapping[str, list[dict[str, Any]]],
) -> None:
    metadata = {
        (document_id, *_row_key(row)): {
            "assertions": list(row.get("assertions", [])),
            "candidates": list(row.get("candidates", [])),
        }
        for document_id, rows in review_by_doc.items()
        for row in rows
    }
    for item in queue:
        key = (
            str(item["document_id"]),
            int(item["position"][0]),
            int(item["position"][1]),
            str(item["type"]),
        )
        suggestion = metadata.get(key, {"assertions": [], "candidates": []})
        item["assertion_suggestions"] = suggestion["assertions"]
        if not item["candidate_suggestions"]:
            item["candidate_suggestions"] = suggestion["candidates"]


def _group_review_queue(queue: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, tuple[str, ...]], dict[str, Any]] = {}
    for row in queue:
        sources = tuple(str(value) for value in row["sources"])
        key = (
            str(row["normalized_mention"]),
            str(row["type"]),
            str(row["reason"]),
            sources,
        )
        group = groups.setdefault(
            key,
            {
                "normalized_mention": key[0],
                "type": key[1],
                "reason": key[2],
                "sources": list(sources),
                "occurrence_count": 0,
                "document_ids": set(),
                "candidate_suggestions": set(),
                "assertion_suggestions": set(),
                "examples": [],
                "review_decision": "",
                "review_notes": "",
            },
        )
        group["occurrence_count"] += 1
        group["document_ids"].add(str(row["document_id"]))
        group["candidate_suggestions"].update(row["candidate_suggestions"])
        group["assertion_suggestions"].update(row["assertion_suggestions"])
        if len(group["examples"]) < 5:
            group["examples"].append(
                {
                    "document_id": row["document_id"],
                    "text": row["text"],
                    "position": list(row["position"]),
                    "context": row["context"],
                }
            )

    output: list[dict[str, Any]] = []
    for group in groups.values():
        output.append(
            {
                **{
                    key: value
                    for key, value in group.items()
                    if key
                    not in {
                        "document_ids",
                        "candidate_suggestions",
                        "assertion_suggestions",
                    }
                },
                "document_count": len(group["document_ids"]),
                "document_ids": sorted(
                    group["document_ids"],
                    key=_document_sort_key,
                ),
                "candidate_suggestions": sorted(group["candidate_suggestions"]),
                "assertion_suggestions": sorted(group["assertion_suggestions"]),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            -int(row["document_count"]),
            -int(row["occurrence_count"]),
            str(row["reason"]),
            str(row["type"]),
            str(row["normalized_mention"]),
        ),
    )


def _apply_btc_external_contract(
    rows_by_doc: Mapping[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Mirror the public example by omitting candidates on non-codable types."""

    output: dict[str, list[dict[str, Any]]] = {}
    for document_id, rows in rows_by_doc.items():
        transformed: list[dict[str, Any]] = []
        for row in rows:
            copied = dict(row)
            copied["position"] = list(row["position"])
            copied["assertions"] = list(row["assertions"])
            if str(row["type"]) in PHASE1_CODABLE_TYPES:
                copied["candidates"] = list(row.get("candidates", []))
            else:
                copied.pop("candidates", None)
            transformed.append(copied)
        output[document_id] = transformed
    return output


def _deduplicate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rows:
        selected.setdefault(_row_key(row), row)
    return sorted(selected.values(), key=_row_sort_key)


def _component_sources(groups: Sequence[_ProposalGroup]) -> tuple[str, ...]:
    return tuple(sorted({source for group in groups for source in group.sources}))


def _identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "text": row["text"],
        "type": row["type"],
        "position": list(row["position"]),
    }


def _context_window(source_text: str, span: tuple[int, int], radius: int = 100) -> str:
    start, end = span
    return source_text[max(0, start - radius) : min(len(source_text), end + radius)]


def _contains(container: tuple[int, int], inner: tuple[int, int]) -> bool:
    return container[0] <= inner[0] and inner[1] <= container[1]


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _row_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    position = row["position"]
    return int(position[0]), int(position[1]), str(row["type"])


def _group_sort_key(group: _ProposalGroup) -> tuple[int, int, str, str]:
    return group.span[0], group.span[1], group.entity_type, group.text


def _row_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str, str]:
    start, end, entity_type = _row_key(row)
    return start, end, entity_type, str(row["text"])


def _review_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str, str]:
    document_id = str(row["document_id"])
    return (
        int(document_id) if document_id.isdigit() else 10**9,
        int(row["position"][0]),
        str(row["type"]),
        str(row["text"]),
    )


def _document_sort_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (10**9, value)


def _rows_mapping(value: Any, field: str) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a document mapping")
    return {str(key): list(rows) for key, rows in value.items()}


def _write_rows(
    rows_by_doc: Mapping[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for document_id, rows in sorted(
        rows_by_doc.items(),
        key=lambda item: _document_sort_key(item[0]),
    ):
        _write_json(output_dir / f"{document_id}.json", rows, sort_keys=False)


def _write_json(path: Path, value: Any, *, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_summary_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    type_rows = "\n".join(
        f"| {entity_type} | {count} |"
        for entity_type, count in summary["strict_type_counts"].items()
    )
    path.write_text(
        "\n".join(
            [
                "# Round 2 Inferred Golden",
                "",
                "> This is inferred local supervision, not organizer-provided gold.",
                "",
                f"- Documents: {summary['document_count']}",
                f"- Proposal sources: {summary['source_count']}",
                f"- Strict entities: {summary['strict_entity_count']}",
                f"- Review-union entities: {summary['review_entity_count']}",
                f"- Pending review rows: {summary['review_queue_count']}",
                f"- Grouped review decisions: {summary['review_group_count']}",
                f"- Strict candidates: {summary['strict_candidate_count']}",
                f"- Strict assertions: {summary['strict_assertion_count']}",
                "",
                "| Strict entity type | Count |",
                "| --- | ---: |",
                type_rows,
                "",
                "`gold_strict` may be used only as weak supervision. Resolve `review_queue.jsonl` "
                "before treating `gold_review` as human-reviewed data.",
                "",
            ]
        ),
        encoding="utf-8",
    )
