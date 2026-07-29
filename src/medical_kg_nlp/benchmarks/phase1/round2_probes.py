"""Isolated Round 2 assertion, candidate, and region-routed entity probes.

This module operates on complete Phase 1 artifacts rather than composing a new core pipeline.
That boundary is deliberate: public probes must preserve every non-target field, and private
Round 2 text must remain local while external proposal sources are compared.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from medical_kg_nlp.benchmarks.phase1.phase1 import (
    validate_phase1_submission_documents,
    validate_phase1_submission_zip,
    zip_phase1_output_dir,
)
from medical_kg_nlp.benchmarks.phase1.phase1_ensemble import (
    expand_repeated_phase1_mentions,
    load_phase1_output_source,
)
from medical_kg_nlp.benchmarks.phase1.phase1_selective_overlays import (
    apply_selective_assertions,
    validate_probe_isolation,
)
from medical_kg_nlp.benchmarks.phase1.round2 import (
    load_phase1_round2_documents,
)
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.mining.io import load_documents, write_json, write_jsonl
from medical_kg_nlp.ontology.phase1 import (
    PHASE1_ALLOWED_TYPES,
    PHASE1_CODE_SYSTEM_BY_TYPE,
    PHASE1_TYPE_PRIORITY,
)
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.run_output import create_hashed_run_dir
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = [
    "CandidateProbePolicy",
    "Phase1Round2ProbeConfig",
    "Phase1TextRegion",
    "RegionProposalPolicy",
    "apply_reviewed_rxnorm_fill_empty",
    "apply_round2_candidate_policy",
    "align_quoted_phase1_proposals",
    "canonicalize_full_phase1_source",
    "merge_consensus_boundary_replacements",
    "merge_region_routed_proposals",
    "run_phase1_round2_probes",
    "segment_phase1_text_regions",
]

Round2RegionKind = Literal[
    "medication_list",
    "clinical",
    "question_answer",
    "educational",
    "other",
]
CandidateProbePolicy = Literal[
    "icd_top1_keep_rx",
    "rx_only",
    "rx_unique_only",
    "rx_unique_keep_icd",
]

_MEDICATION_HEADING_RE = re.compile(
    r"(?i)^\s*(?:danh sách\s+)?thuốc\s+(?:đang dùng\s+)?trước "
    r"(?:khi\s+)?nhập viện\b"
)
_CLINICAL_HEADING_RE = re.compile(
    r"(?i)^\s*(?:tiền sử(?: bệnh| nội khoa| phẫu thuật)?|bệnh lý mạn tính|"
    r"lý do nhập viện|bệnh sử(?: hiện tại)?|triệu chứng hiện tại|"
    r"tình trạng ngay trước khi nhập viện|đánh giá tại bệnh viện|"
    r"cận lâm sàng|kết quả xét nghiệm|chẩn đoán|điều trị)\b"
)
_QUESTION_ANSWER_HEADING_RE = re.compile(
    r"(?i)^\s*(?:câu hỏi\s+(?:(?:của|từ)\s+)?người dùng|"
    r"câu trả lời\s+(?:của\s+)?(?:bác sĩ|bác sỹ)|"
    r"hỏi\s*:|trả lời\s*:|đáp\s*:)"
)
_EDUCATIONAL_RE = re.compile(
    r"(?i)(?:\blà gì\??\s*$|\bdấu hiệu của\b|\bcó nguy hiểm không\??\s*$|"
    r"^\s*phòng ngừa\b)"
)
_SOURCE_NAME_RE = re.compile(r"[A-Za-z0-9_.-]+")
_ALL_ROUND2_REGIONS: frozenset[Round2RegionKind] = frozenset(
    {"medication_list", "clinical", "question_answer", "educational", "other"}
)


@dataclass(frozen=True)
class Phase1TextRegion:
    """One contiguous region expressed in immutable source coordinates."""

    span: tuple[int, int]
    kind: Round2RegionKind
    text: str

    def validate(self, source_text: str) -> None:
        """Reject region boundaries that no longer address the raw source text."""

        start, end = self.span
        if start < 0 or end < start or end > len(source_text):
            raise ValueError(f"Invalid {self.kind} region span {self.span}")
        if source_text[start:end] != self.text:
            raise ValueError(f"{self.kind} region does not match raw source text")


@dataclass(frozen=True)
class RegionProposalPolicy:
    """Selection rules for adding proposal entities around a frozen baseline."""

    minimum_agreement_sources: int = 2
    allowed_single_source_regions: frozenset[Round2RegionKind] = frozenset(
        {"question_answer", "educational"}
    )

    def __post_init__(self) -> None:
        if self.minimum_agreement_sources < 2:
            raise ValueError("Proposal agreement requires at least two independent sources")


@dataclass(frozen=True)
class Phase1Round2ProbeConfig:
    """Immutable inputs for one hashed Round 2 probe suite."""

    documents_path: Path
    expected_source_archive_sha256: str
    base: Path
    expected_base_sha256: str
    dictionary_paths: tuple[Path, ...]
    proposal_sources: tuple[tuple[str, Path], ...] = ()
    output_root: Path = Path("outputs/phase1/round2")
    run_label: str = "round2-breakthrough-probes"
    expected_count: int = 100
    minimum_agreement_sources: int = 2
    expand_repeated_mentions: bool = True
    full_source_names: tuple[str, ...] = ()
    consensus_source_names: tuple[str, ...] = ()
    candidate_probe_policies: tuple[CandidateProbePolicy, ...] = ()
    reviewed_rxnorm_map_path: Path | None = None
    reviewed_rxnorm_min_occurrence_support: int = 2
    reviewed_rxnorm_min_document_support: int = 1

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.expected_source_archive_sha256):
            raise ValueError(
                "Expected source archive SHA-256 must be 64 lowercase hex characters"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self.expected_base_sha256):
            raise ValueError("Expected baseline SHA-256 must be 64 lowercase hex characters")
        if self.expected_count < 1:
            raise ValueError("Expected document count must be positive")
        if self.minimum_agreement_sources < 2:
            raise ValueError("Proposal agreement requires at least two independent sources")
        if not self.dictionary_paths:
            raise ValueError("At least one validation dictionary is required")
        names = [name for name, _ in self.proposal_sources]
        if len(names) != len(set(names)):
            raise ValueError("Proposal source names must be unique")
        invalid_names = [name for name in names if _SOURCE_NAME_RE.fullmatch(name) is None]
        if invalid_names:
            raise ValueError(f"Invalid proposal source names: {invalid_names}")
        if len(self.full_source_names) != len(set(self.full_source_names)):
            raise ValueError("Full-source names must be unique")
        unknown_full_sources = set(self.full_source_names) - set(names)
        if unknown_full_sources:
            raise ValueError(
                "Full-source variants require matching --source inputs: "
                f"{sorted(unknown_full_sources)}"
            )
        if len(self.consensus_source_names) != len(set(self.consensus_source_names)):
            raise ValueError("Consensus-source names must be unique")
        unknown_consensus_sources = set(self.consensus_source_names) - set(names)
        if unknown_consensus_sources:
            raise ValueError(
                "Consensus-source variants require matching --source inputs: "
                f"{sorted(unknown_consensus_sources)}"
            )
        if len(self.candidate_probe_policies) != len(set(self.candidate_probe_policies)):
            raise ValueError("Candidate probe policies must be unique")
        unsupported_candidate_policies = set(self.candidate_probe_policies) - {
            "icd_top1_keep_rx",
            "rx_only",
            "rx_unique_only",
            "rx_unique_keep_icd",
        }
        if unsupported_candidate_policies:
            raise ValueError(
                "Unsupported candidate probe policies: "
                f"{sorted(unsupported_candidate_policies)}"
            )
        if self.reviewed_rxnorm_min_occurrence_support < 1:
            raise ValueError("Reviewed RxNorm occurrence support must be positive")
        if self.reviewed_rxnorm_min_document_support < 1:
            raise ValueError("Reviewed RxNorm document support must be positive")


@dataclass(frozen=True)
class _ReviewedRxNormRule:
    """One exact medication mapping admitted from the reviewed candidate registry."""

    normalized_mention: str
    candidate: str
    candidate_stage: str
    rule_id: str
    provenance: str
    occurrence_support: int
    document_support: int


def segment_phase1_text_regions(source_text: str) -> tuple[Phase1TextRegion, ...]:
    """Split text into structural regions without normalizing or rewriting characters."""

    if not source_text:
        return ()
    current_kind: Round2RegionKind = _document_default_region(source_text)
    raw_regions: list[tuple[int, int, Round2RegionKind]] = []
    cursor = 0
    for line in source_text.splitlines(keepends=True):
        line_kind = _line_region_kind(line)
        if line_kind is not None:
            current_kind = line_kind
        end = cursor + len(line)
        raw_regions.append((cursor, end, current_kind))
        cursor = end
    if cursor < len(source_text):
        raw_regions.append((cursor, len(source_text), current_kind))

    merged: list[Phase1TextRegion] = []
    for start, end, kind in raw_regions:
        if merged and merged[-1].kind == kind and merged[-1].span[1] == start:
            previous = merged.pop()
            merged.append(
                Phase1TextRegion(
                    span=(previous.span[0], end),
                    kind=kind,
                    text=source_text[previous.span[0] : end],
                )
            )
            continue
        merged.append(
            Phase1TextRegion(
                span=(start, end),
                kind=kind,
                text=source_text[start:end],
            )
        )
    for region in merged:
        region.validate(source_text)
    return tuple(merged)


def align_quoted_phase1_proposals(
    source_text: str,
    proposals: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project exact quoted model output onto every matching raw occurrence.

    A generative model is never trusted to calculate offsets. Optional left/right context narrows
    repeated surfaces; without context every exact occurrence is retained so first-match lookup
    cannot silently lose repeated mentions.
    """

    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for index, proposal in enumerate(proposals):
        text = proposal.get("text")
        entity_type = proposal.get("type")
        left_context = proposal.get("left_context", "")
        right_context = proposal.get("right_context", "")
        if (
            not isinstance(text, str)
            or not text
            or entity_type not in PHASE1_ALLOWED_TYPES
            or not isinstance(left_context, str)
            or not isinstance(right_context, str)
        ):
            rejected.append({"proposal_index": index, "reason": "invalid_schema"})
            continue
        occurrences = _exact_occurrences(source_text, text)
        if left_context:
            occurrences = [
                start
                for start in occurrences
                if source_text[max(0, start - len(left_context)) : start].endswith(
                    left_context
                )
            ]
        if right_context:
            occurrences = [
                start
                for start in occurrences
                if source_text[start + len(text) :].startswith(right_context)
            ]
        if not occurrences:
            rejected.append(
                {
                    "proposal_index": index,
                    "reason": "quote_or_context_not_found",
                    "text": text,
                    "type": entity_type,
                }
            )
            continue
        for start in occurrences:
            key = (entity_type, text, start, start + len(text))
            if key in seen:
                continue
            seen.add(key)
            rows.append(_new_entity_row(text, entity_type, start, start + len(text)))
    rows.sort(key=_row_sort_key)
    return rows, rejected


def merge_region_routed_proposals(
    baseline_by_doc: Mapping[str, list[dict[str, Any]]],
    proposal_sources: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    source_text_by_doc: Mapping[str, str],
    *,
    policy: RegionProposalPolicy | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, int]]:
    """Add non-overlapping proposals using region routing and exact source agreement."""

    active = policy or RegionProposalPolicy()
    expected_ids = set(baseline_by_doc)
    if set(source_text_by_doc) != expected_ids:
        raise ValueError("Source text IDs must exactly match the frozen baseline")
    for source_name, rows_by_doc in proposal_sources.items():
        extra = set(rows_by_doc) - expected_ids
        if extra:
            raise ValueError(
                f"Proposal source {source_name!r} has unexpected documents: {sorted(extra)}"
            )

    output: dict[str, list[dict[str, Any]]] = {}
    decisions: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for document_id in sorted(expected_ids, key=_document_sort_key):
        source_text = source_text_by_doc[document_id]
        regions = segment_phase1_text_regions(source_text)
        baseline = [_copy_row(row) for row in baseline_by_doc[document_id]]
        for row in baseline:
            _validate_entity_identity(row, source_text, document_id=document_id)
        baseline_keys = {_identity_key(row) for row in baseline}

        grouped: dict[
            tuple[str, str, int, int],
            set[str],
        ] = {}
        sample_by_key: dict[tuple[str, str, int, int], dict[str, Any]] = {}
        for source_name, rows_by_doc in sorted(proposal_sources.items()):
            for row in rows_by_doc.get(document_id, []):
                _validate_entity_identity(row, source_text, document_id=document_id)
                key = _identity_key(row)
                grouped.setdefault(key, set()).add(source_name)
                sample_by_key.setdefault(key, row)

        eligible: list[tuple[dict[str, Any], frozenset[str], Round2RegionKind]] = []
        for key, source_name_set in grouped.items():
            if key in baseline_keys:
                counters["proposal.already_in_baseline"] += 1
                continue
            row = sample_by_key[key]
            start, end = _position(row)
            region_kind = _region_for_span(regions, start, end)
            if region_kind is None:
                counters["proposal.crosses_region_boundary"] += 1
                continue
            source_names = frozenset(source_name_set)
            has_consensus = len(source_names) >= active.minimum_agreement_sources
            single_source_allowed = (
                len(source_names) == 1
                and region_kind in active.allowed_single_source_regions
            )
            if not has_consensus and not single_source_allowed:
                counters["proposal.blocked_without_evidence"] += 1
                continue
            eligible.append((row, source_names, region_kind))

        selected: list[dict[str, Any]] = []
        for row, agreement_sources, region_kind in sorted(
            eligible,
            key=_proposal_priority,
        ):
            if any(_rows_overlap(row, existing) for existing in (*baseline, *selected)):
                counters["proposal.blocked_overlap"] += 1
                continue
            start, end = _position(row)
            added = _new_entity_row(
                str(row["text"]),
                str(row["type"]),
                start,
                end,
            )
            selected.append(added)
            reason = (
                "exact_source_consensus"
                if len(agreement_sources) >= active.minimum_agreement_sources
                else "single_source_region_route"
            )
            decisions.append(
                {
                    "document_id": document_id,
                    "stage": "entity_region_router",
                    "action": "add",
                    "reason": reason,
                    "region": region_kind,
                    "sources": sorted(agreement_sources),
                    "entity": _identity_payload(added),
                }
            )
            counters[f"proposal.add.{reason}"] += 1
            counters[f"proposal.add.region.{region_kind}"] += 1
        output[document_id] = sorted((*baseline, *selected), key=_row_sort_key)

    isolation_issues = validate_probe_isolation(
        baseline_by_doc,
        output,
        module="entity",
    )
    if isolation_issues:
        raise ValueError(f"Entity proposal isolation failed: {isolation_issues[:5]}")
    counters["decision_total"] = len(decisions)
    counters["output_entity_total"] = sum(len(rows) for rows in output.values())
    return output, decisions, dict(sorted(counters.items()))


def merge_consensus_boundary_replacements(
    baseline_by_doc: Mapping[str, list[dict[str, Any]]],
    consensus_by_doc: Mapping[str, list[dict[str, Any]]],
    source_text_by_doc: Mapping[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, int]]:
    """Add consensus proposals and replace only one contained same-type baseline span.

    MODEL: this is deliberately a separate public probe from additive recall. Replacement is
    allowed only when one source span contains the other, so crossing spans, type conflicts, and
    one-to-many overlap ambiguity cannot silently rewrite the frozen baseline.
    """

    expected_ids = set(baseline_by_doc)
    if set(consensus_by_doc) != expected_ids or set(source_text_by_doc) != expected_ids:
        raise ValueError(
            "Consensus, source text, and baseline document IDs must match exactly"
        )

    output: dict[str, list[dict[str, Any]]] = {}
    decisions: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for document_id in sorted(expected_ids, key=_document_sort_key):
        source_text = source_text_by_doc[document_id]
        selected = [_copy_row(row) for row in baseline_by_doc[document_id]]
        for row in (*selected, *consensus_by_doc[document_id]):
            _validate_entity_identity(row, source_text, document_id=document_id)

        for proposal in sorted(
            consensus_by_doc[document_id],
            key=lambda row: (
                -PHASE1_TYPE_PRIORITY.get(str(row.get("type")), 0),
                -(_position(row)[1] - _position(row)[0]),
                *_row_sort_key(row),
            ),
        ):
            if any(_identity_key(proposal) == _identity_key(row) for row in selected):
                counters["proposal.already_in_baseline"] += 1
                continue
            overlap_indexes = [
                index for index, row in enumerate(selected) if _rows_overlap(proposal, row)
            ]
            if not overlap_indexes:
                start, end = _position(proposal)
                added = _new_entity_row(
                    str(proposal["text"]),
                    str(proposal["type"]),
                    start,
                    end,
                )
                selected.append(added)
                decisions.append(
                    {
                        "document_id": document_id,
                        "stage": "entity_consensus_boundary",
                        "action": "add",
                        "reason": "internally_consensused_nonoverlap",
                        "entity": _identity_payload(added),
                    }
                )
                counters["proposal.added"] += 1
                continue
            if len(overlap_indexes) != 1:
                counters["proposal.blocked_multiple_overlaps"] += 1
                continue

            overlap_index = overlap_indexes[0]
            existing = selected[overlap_index]
            if proposal.get("type") != existing.get("type"):
                counters["proposal.blocked_type_conflict"] += 1
                continue
            proposal_start, proposal_end = _position(proposal)
            existing_start, existing_end = _position(existing)
            if not (
                (proposal_start <= existing_start and existing_end <= proposal_end)
                or (
                    existing_start <= proposal_start
                    and proposal_end <= existing_end
                )
            ):
                counters["proposal.blocked_crossing_boundary"] += 1
                continue

            replacement = _copy_row(existing)
            replacement["text"] = str(proposal["text"])
            replacement["position"] = [proposal_start, proposal_end]
            if any(
                index != overlap_index and _rows_overlap(replacement, row)
                for index, row in enumerate(selected)
            ):
                counters["proposal.blocked_replacement_overlap"] += 1
                continue
            selected[overlap_index] = replacement
            decisions.append(
                {
                    "document_id": document_id,
                    "stage": "entity_consensus_boundary",
                    "action": "replace",
                    "reason": "single_contained_same_type_overlap",
                    "before": _identity_payload(existing),
                    "after": _identity_payload(replacement),
                }
            )
            counters["proposal.replaced"] += 1

        output[document_id] = sorted(selected, key=_row_sort_key)

    isolation_issues = validate_probe_isolation(
        baseline_by_doc,
        output,
        module="entity",
    )
    if isolation_issues:
        raise ValueError(f"Consensus boundary isolation failed: {isolation_issues[:5]}")
    counters["decision_total"] = len(decisions)
    counters["output_entity_total"] = sum(len(rows) for rows in output.values())
    return output, decisions, dict(sorted(counters.items()))


def canonicalize_full_phase1_source(
    rows_by_doc: Mapping[str, list[dict[str, Any]]],
    source_text_by_doc: Mapping[str, str],
    dictionary: DictionaryStore,
    *,
    preserve_proposal_metadata: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, int]]:
    """Preserve a source entity projection while removing non-canonical candidates.

    This adapter exists for externally produced, already scored artifacts. It does not infer new
    codes or repair spans. Candidate values survive only when the pinned terminology contains the
    expected Phase 1 code system for that entity type. Proposal fusion may opt into retaining
    calibrated evidence fields; ordinary submission paths keep the strict five-field schema.
    """

    if set(rows_by_doc) != set(source_text_by_doc):
        raise ValueError("Full proposal source must contain every Round 2 document exactly once")
    output: dict[str, list[dict[str, Any]]] = {}
    decisions: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    known_codes = set(dictionary.by_code_system_code)
    seen_identities: Counter[tuple[str, str, str, int, int]] = Counter()

    for document_id in sorted(rows_by_doc, key=_document_sort_key):
        source_text = source_text_by_doc[document_id]
        canonical_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(rows_by_doc[document_id]):
            _validate_entity_identity(row, source_text, document_id=document_id)
            entity_type = str(row["type"])
            raw_assertions = row.get("assertions", [])
            raw_candidates = row.get("candidates", [])
            if not isinstance(raw_assertions, list) or not all(
                isinstance(value, str) for value in raw_assertions
            ):
                raise ValueError(
                    f"{document_id}:{row_index}: assertions must be a string list"
                )
            if not isinstance(raw_candidates, list) or not all(
                isinstance(value, str) for value in raw_candidates
            ):
                raise ValueError(
                    f"{document_id}:{row_index}: candidates must be a string list"
                )

            expected_system = PHASE1_CODE_SYSTEM_BY_TYPE.get(entity_type)
            retained: list[str] = []
            for candidate in raw_candidates:
                if expected_system is not None and (
                    expected_system,
                    candidate,
                ) in known_codes:
                    retained.append(candidate)
                    counters["candidate.retained"] += 1
                    continue
                decisions.append(
                    {
                        "document_id": document_id,
                        "stage": "source_candidate_safety",
                        "action": "remove",
                        "reason": (
                            "candidate_not_allowed_for_entity_type"
                            if expected_system is None
                            else "candidate_absent_from_pinned_terminology"
                        ),
                        "candidate": candidate,
                        "entity": _identity_payload(row),
                    }
                )
                counters["candidate.removed"] += 1

            start, end = _position(row)
            canonical = {
                "text": str(row["text"]),
                "type": entity_type,
                "assertions": list(raw_assertions),
                "candidates": retained,
                "position": [start, end],
            }
            if preserve_proposal_metadata:
                # MODEL: confidence, source-task labels, and support-only ownership are verifier
                # features. They are evidence, not Phase 1 output fields, so retention is opt-in.
                for key in ("confidence", "score", "source_label", "support_only"):
                    if key in row:
                        canonical[key] = row[key]
            canonical_rows.append(canonical)
            seen_identities[(document_id, *_identity_key(canonical))] += 1
        output[document_id] = sorted(canonical_rows, key=_row_sort_key)

    counters["duplicate_identity_rows"] = sum(
        count - 1 for count in seen_identities.values() if count > 1
    )
    counters["decision_total"] = len(decisions)
    counters["output_entity_total"] = sum(len(rows) for rows in output.values())
    counters["output_candidate_rows"] = sum(
        bool(row["candidates"]) for rows in output.values() for row in rows
    )
    return output, decisions, dict(sorted(counters.items()))


def apply_round2_candidate_policy(
    rows_by_doc: Mapping[str, list[dict[str, Any]]],
    *,
    policy: CandidateProbePolicy,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, int]]:
    """Apply one abstaining candidate policy without changing entities or assertions.

    `rx_only` retains every existing RxNorm list on medication entities and clears all other
    candidate lists. `rx_unique_only` additionally requires the medication row to contain exactly
    one candidate. `rx_unique_keep_icd` applies that medication uniqueness gate while preserving
    diagnosis candidates, isolating the effect of ambiguous drug lists from ICD abstention.
    `icd_top1_keep_rx` preserves every non-diagnosis list and truncates ranked diagnosis lists to
    their first code, isolating the effect of broad ICD retrieval lists.
    """

    if policy not in {
        "icd_top1_keep_rx",
        "rx_only",
        "rx_unique_only",
        "rx_unique_keep_icd",
    }:
        raise ValueError(f"Unsupported Round 2 candidate policy {policy!r}")
    output: dict[str, list[dict[str, Any]]] = {}
    decisions: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for document_id in sorted(rows_by_doc, key=_document_sort_key):
        transformed: list[dict[str, Any]] = []
        for row in rows_by_doc[document_id]:
            copied = _copy_row(row)
            raw_candidates = copied.get("candidates", [])
            if not isinstance(raw_candidates, list) or not all(
                isinstance(value, str) for value in raw_candidates
            ):
                raise ValueError(f"{document_id}: candidates must be a string list")

            entity_type = str(copied.get("type", ""))
            retained = list(raw_candidates)
            reason: str | None = None
            if (
                policy == "icd_top1_keep_rx"
                and entity_type == "CHẨN_ĐOÁN"
                and len(raw_candidates) > 1
            ):
                # INVARIANT: candidate order is the retriever's frozen rank order. This probe
                # changes only list depth and never re-scores or invents a terminology code.
                retained = raw_candidates[:1]
                reason = "diagnosis_candidate_top1_truncation"
                counters["row.truncated_diagnosis"] += 1
            elif (
                policy != "icd_top1_keep_rx"
                and entity_type != "THUỐC"
                and policy != "rx_unique_keep_icd"
            ):
                retained = []
                if raw_candidates:
                    reason = "non_medication_candidate_abstention"
                    counters["row.cleared_non_medication"] += 1
            elif entity_type == "THUỐC" and policy in {
                "rx_unique_only",
                "rx_unique_keep_icd",
            } and len(raw_candidates) != 1:
                retained = []
                if raw_candidates:
                    reason = "ambiguous_medication_candidate_abstention"
                    counters["row.cleared_ambiguous_medication"] += 1
                else:
                    counters["row.empty_medication"] += 1
            elif entity_type == "THUỐC" and raw_candidates:
                counters["row.retained_medication"] += 1
            elif raw_candidates:
                counters["row.retained_non_medication"] += 1

            copied["candidates"] = retained
            transformed.append(copied)
            counters["candidate.input"] += len(raw_candidates)
            counters["candidate.retained"] += len(retained)
            counters["candidate.removed"] += len(raw_candidates) - len(retained)
            if reason is not None:
                action = (
                    "truncate"
                    if reason == "diagnosis_candidate_top1_truncation"
                    else "clear"
                )
                decisions.append(
                    {
                        "document_id": document_id,
                        "stage": "candidate_abstention",
                        "action": action,
                        "reason": reason,
                        "candidate_count_before": len(raw_candidates),
                        "candidate_count_after": len(retained),
                        "entity": _identity_payload(row),
                    }
                )
        output[document_id] = transformed

    # INVARIANT: this transform is metadata-only; row order, duplicate identities, and assertions
    # stay byte-for-byte equivalent after JSON decoding.
    isolation_issues = validate_probe_isolation(
        rows_by_doc,
        output,
        module="candidate",
    )
    if isolation_issues:
        raise ValueError(f"Candidate policy isolation failed: {isolation_issues[:5]}")
    counters["decision_total"] = len(decisions)
    counters["output_entity_total"] = sum(len(rows) for rows in output.values())
    counters["output_candidate_rows"] = sum(
        bool(row.get("candidates")) for rows in output.values() for row in rows
    )
    return output, decisions, dict(sorted(counters.items()))


def apply_reviewed_rxnorm_fill_empty(
    rows_by_doc: Mapping[str, list[dict[str, Any]]],
    *,
    reviewed_map_path: Path,
    dictionary: DictionaryStore,
    minimum_occurrence_support: int = 2,
    minimum_document_support: int = 1,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, int]]:
    """Fill empty medication candidates from exact, reviewed, pinned RxNorm mappings.

    This deliberately does not perform terminology retrieval. It promotes only mappings already
    reviewed against Phase 1 conventions, and it never replaces a candidate emitted by the frozen
    baseline. Support thresholds make the public probe independent from one-off annotation rows.
    """

    rules, registry_counters = _load_reviewed_rxnorm_rules(
        reviewed_map_path,
        dictionary,
        minimum_occurrence_support=minimum_occurrence_support,
        minimum_document_support=minimum_document_support,
    )
    output: dict[str, list[dict[str, Any]]] = {}
    decisions: list[dict[str, Any]] = []
    counters: Counter[str] = Counter(registry_counters)
    for document_id in sorted(rows_by_doc, key=_document_sort_key):
        transformed: list[dict[str, Any]] = []
        for row in rows_by_doc[document_id]:
            copied = _copy_row(row)
            raw_candidates = copied.get("candidates", [])
            if not isinstance(raw_candidates, list) or not all(
                isinstance(value, str) for value in raw_candidates
            ):
                raise ValueError(f"{document_id}: candidates must be a string list")
            counters["candidate.input"] += len(raw_candidates)
            if raw_candidates:
                # INVARIANT: this probe cannot overwrite or reorder an existing candidate list.
                counters["row.preserved_nonempty"] += 1
                counters["candidate.retained"] += len(raw_candidates)
                transformed.append(copied)
                continue
            if str(copied.get("type", "")) != "THUỐC":
                counters["row.skipped_non_medication"] += 1
                transformed.append(copied)
                continue

            normalized_mention = normalize_for_match(str(copied.get("text", "")))
            rule = rules.get(normalized_mention)
            if rule is None:
                counters["row.no_reviewed_exact_mapping"] += 1
                transformed.append(copied)
                continue

            copied["candidates"] = [rule.candidate]
            transformed.append(copied)
            counters["candidate.added"] += 1
            counters["candidate.retained"] += 1
            counters["row.filled"] += 1
            decisions.append(
                {
                    "document_id": document_id,
                    "stage": rule.candidate_stage,
                    "rule_id": rule.rule_id,
                    "source": rule.provenance,
                    "action": "fill_empty",
                    "reason": "reviewed_exact_rxnorm_mapping",
                    "candidate": rule.candidate,
                    "occurrence_support": rule.occurrence_support,
                    "document_support": rule.document_support,
                    "entity": _identity_payload(row),
                }
            )
        output[document_id] = transformed

    # INVARIANT: candidate enrichment must preserve every entity and assertion byte-for-byte after
    # JSON decoding. The release validator separately enforces code-system and dictionary safety.
    isolation_issues = validate_probe_isolation(
        rows_by_doc,
        output,
        module="candidate",
    )
    if isolation_issues:
        raise ValueError(f"Reviewed RxNorm probe isolation failed: {isolation_issues[:5]}")
    counters["decision_total"] = len(decisions)
    counters["output_entity_total"] = sum(len(rows) for rows in output.values())
    counters["output_candidate_rows"] = sum(
        bool(row.get("candidates")) for rows in output.values() for row in rows
    )
    return output, decisions, dict(sorted(counters.items()))


def _load_reviewed_rxnorm_rules(
    path: Path,
    dictionary: DictionaryStore,
    *,
    minimum_occurrence_support: int,
    minimum_document_support: int,
) -> tuple[dict[str, _ReviewedRxNormRule], dict[str, int]]:
    """Load the reviewed map while rejecting provenance and terminology drift."""

    if minimum_occurrence_support < 1 or minimum_document_support < 1:
        raise ValueError("Reviewed RxNorm support thresholds must be positive")
    rules: dict[str, _ReviewedRxNormRule] = {}
    counters: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            counters["registry.row_input"] += 1
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: reviewed mapping must be an object")
            if (
                row.get("review_status") != "reviewed"
                or row.get("entity_type") != "THUỐC"
                or row.get("code_system") != "RxNorm"
                or row.get("candidate_stage")
                not in {
                    "candidate_rxnorm_ingredient",
                    "candidate_rxnorm_clinical_drug",
                }
            ):
                counters["registry.row_skipped_non_rxnorm_reviewed"] += 1
                continue

            normalized_mention = normalize_for_match(
                str(row.get("normalized_mention", ""))
            )
            candidate = str(row.get("candidate", "")).strip()
            rule_id = str(row.get("rule_id", "")).strip()
            if not normalized_mention or not candidate or not rule_id:
                raise ValueError(
                    f"{path}:{line_number}: reviewed RxNorm mapping is incomplete"
                )
            occurrence_support = _support_value(
                row.get("occurrence_support"),
                path=path,
                line_number=line_number,
                field="occurrence_support",
            )
            document_support = _support_value(
                row.get("document_support"),
                path=path,
                line_number=line_number,
                field="document_support",
            )
            if (
                occurrence_support < minimum_occurrence_support
                or document_support < minimum_document_support
            ):
                counters["registry.row_skipped_low_support"] += 1
                continue

            entry = dictionary.by_code_system_code.get(
                (CodeSystem.RXNORM, candidate)
            )
            if entry is None:
                raise ValueError(
                    f"{path}:{line_number}: RxNorm candidate {candidate!r} is absent "
                    "from the pinned terminology"
                )
            if entry.semantic_type is not EntityType.DRUG:
                raise ValueError(
                    f"{path}:{line_number}: RxNorm candidate {candidate!r} is not a drug"
                )

            rule = _ReviewedRxNormRule(
                normalized_mention=normalized_mention,
                candidate=candidate,
                candidate_stage=str(row["candidate_stage"]),
                rule_id=rule_id,
                provenance=str(row.get("provenance", "reviewed_candidate_map")),
                occurrence_support=occurrence_support,
                document_support=document_support,
            )
            existing = rules.get(normalized_mention)
            if existing is not None and existing.candidate != candidate:
                raise ValueError(
                    f"{path}:{line_number}: conflicting reviewed RxNorm candidates for "
                    f"{normalized_mention!r}: {existing.candidate!r} and {candidate!r}"
                )
            if existing is not None:
                counters["registry.row_duplicate"] += 1
                continue
            rules[normalized_mention] = rule
            counters["registry.row_eligible"] += 1
    return rules, dict(sorted(counters.items()))


def _support_value(
    value: object,
    *,
    path: Path,
    line_number: int,
    field: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path}:{line_number}: {field} must be a non-negative integer")
    return value


def run_phase1_round2_probes(config: Phase1Round2ProbeConfig) -> dict[str, Any]:
    """Build strict, hashed assertion, candidate, and region-routed entity probes."""

    observed_base_sha256 = _path_sha256(config.base)
    if observed_base_sha256 != config.expected_base_sha256:
        raise ValueError(
            "Frozen Round 2 baseline SHA-256 mismatch: "
            f"expected {config.expected_base_sha256}, observed {observed_base_sha256}"
        )
    documents = load_phase1_round2_documents(
        load_documents(config.documents_path),
        expected_archive_sha256=config.expected_source_archive_sha256,
        expected_count=config.expected_count,
    )
    source_text_by_doc = {document.document_id: document.text for document in documents}
    base = load_phase1_output_source(config.base)
    if set(base) != set(source_text_by_doc):
        raise ValueError("Frozen baseline document IDs do not match Round 2 input")
    dictionary = _load_dictionary(config.dictionary_paths)
    _strict_validate_artifact(
        config.base,
        documents=documents,
        dictionary=dictionary,
        expected_count=config.expected_count,
    )

    loaded_sources: dict[str, dict[str, list[dict[str, Any]]]] = {}
    routed_sources: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, path in config.proposal_sources:
        rows = load_phase1_output_source(path)
        loaded_sources[name] = rows
        if config.expand_repeated_mentions:
            rows = expand_repeated_phase1_mentions(rows, source_text_by_doc)
        routed_sources[name] = rows

    run_output = create_hashed_run_dir(
        config.output_root,
        label=config.run_label,
        inputs=(
            config.documents_path,
            config.base,
            *config.dictionary_paths,
            *(
                (config.reviewed_rxnorm_map_path,)
                if config.reviewed_rxnorm_map_path is not None
                else ()
            ),
            *(path for _, path in config.proposal_sources),
        ),
        resolved_config={
            "expected_source_archive_sha256": config.expected_source_archive_sha256,
            "expected_base_sha256": config.expected_base_sha256,
            "expected_count": config.expected_count,
            "minimum_agreement_sources": config.minimum_agreement_sources,
            "expand_repeated_mentions": config.expand_repeated_mentions,
            "proposal_sources": [name for name, _ in config.proposal_sources],
            "full_source_names": list(config.full_source_names),
            "consensus_source_names": list(config.consensus_source_names),
            "candidate_probe_policies": list(config.candidate_probe_policies),
            "reviewed_rxnorm_map_path": (
                str(config.reviewed_rxnorm_map_path)
                if config.reviewed_rxnorm_map_path is not None
                else None
            ),
            "reviewed_rxnorm_min_occurrence_support": (
                config.reviewed_rxnorm_min_occurrence_support
            ),
            "reviewed_rxnorm_min_document_support": (
                config.reviewed_rxnorm_min_document_support
            ),
        },
    )

    variants: list[dict[str, Any]] = []
    assertion_rows, assertion_decisions, assertion_counters = apply_selective_assertions(
        base,
        source_text_by_doc,
        regimes=("negation", "history"),
        preserve_existing=False,
    )
    variants.append(
        _materialize_variant(
            "A_NEG_HIST",
            module="assertion",
            rows=assertion_rows,
            base=base,
            decisions=assertion_decisions,
            counters=assertion_counters,
            run_dir=run_output.run_dir,
            documents=documents,
            dictionary=dictionary,
        )
    )

    for policy in config.candidate_probe_policies:
        candidate_rows, candidate_decisions, candidate_counters = (
            apply_round2_candidate_policy(base, policy=policy)
        )
        variants.append(
            _materialize_variant(
                f"C_{policy.upper()}",
                module="candidate",
                rows=candidate_rows,
                base=base,
                decisions=candidate_decisions,
                counters=candidate_counters,
                run_dir=run_output.run_dir,
                documents=documents,
                dictionary=dictionary,
            )
        )

    if config.reviewed_rxnorm_map_path is not None:
        reviewed_rows, reviewed_decisions, reviewed_counters = (
            apply_reviewed_rxnorm_fill_empty(
                base,
                reviewed_map_path=config.reviewed_rxnorm_map_path,
                dictionary=dictionary,
                minimum_occurrence_support=(
                    config.reviewed_rxnorm_min_occurrence_support
                ),
                minimum_document_support=config.reviewed_rxnorm_min_document_support,
            )
        )
        variants.append(
            _materialize_variant(
                "C_RX_REVIEWED_FILL_EMPTY",
                module="candidate",
                rows=reviewed_rows,
                base=base,
                decisions=reviewed_decisions,
                counters=reviewed_counters,
                run_dir=run_output.run_dir,
                documents=documents,
                dictionary=dictionary,
            )
        )

    for name, rows_by_doc in routed_sources.items():
        routed_rows, decisions, counters = merge_region_routed_proposals(
            base,
            {name: rows_by_doc},
            source_text_by_doc,
            policy=RegionProposalPolicy(
                minimum_agreement_sources=config.minimum_agreement_sources,
            ),
        )
        variants.append(
            _materialize_variant(
                f"E_{name.upper()}_QA_ADD",
                module="entity",
                rows=routed_rows,
                base=base,
                decisions=decisions,
                counters=counters,
                run_dir=run_output.run_dir,
                documents=documents,
                dictionary=dictionary,
            )
        )
        if name in config.consensus_source_names:
            consensus_rows, consensus_decisions, consensus_counters = (
                merge_region_routed_proposals(
                    base,
                    {name: rows_by_doc},
                    source_text_by_doc,
                    policy=RegionProposalPolicy(
                        minimum_agreement_sources=config.minimum_agreement_sources,
                        # MODEL: this source has already passed independent evidence agreement
                        # inside its producer. The separate variant measures whether extending
                        # that evidence beyond Q&A regions improves the public entity metric.
                        allowed_single_source_regions=_ALL_ROUND2_REGIONS,
                    ),
                )
            )
            variants.append(
                _materialize_variant(
                    f"E_{name.upper()}_CONSENSUS_ADD",
                    module="entity",
                    rows=consensus_rows,
                    base=base,
                    decisions=consensus_decisions,
                    counters=consensus_counters,
                    run_dir=run_output.run_dir,
                    documents=documents,
                    dictionary=dictionary,
                )
            )
            (
                consensus_assertion_rows,
                consensus_assertion_decisions,
                consensus_assertion_counters,
            ) = apply_selective_assertions(
                consensus_rows,
                source_text_by_doc,
                regimes=("negation", "history"),
                preserve_existing=True,
            )
            variants.append(
                _materialize_variant(
                    f"E_{name.upper()}_CONSENSUS_ADD_A_NEG_HIST",
                    module="assertion",
                    rows=consensus_assertion_rows,
                    base=consensus_rows,
                    decisions=consensus_assertion_decisions,
                    counters=consensus_assertion_counters,
                    run_dir=run_output.run_dir,
                    documents=documents,
                    dictionary=dictionary,
                )
            )
            (
                replacement_rows,
                replacement_decisions,
                replacement_counters,
            ) = merge_consensus_boundary_replacements(
                base,
                rows_by_doc,
                source_text_by_doc,
            )
            variants.append(
                _materialize_variant(
                    f"E_{name.upper()}_CONSENSUS_REPLACE",
                    module="entity",
                    rows=replacement_rows,
                    base=base,
                    decisions=replacement_decisions,
                    counters=replacement_counters,
                    run_dir=run_output.run_dir,
                    documents=documents,
                    dictionary=dictionary,
                )
            )
        if name in config.full_source_names:
            canonical_rows, source_decisions, source_counters = (
                canonicalize_full_phase1_source(
                    loaded_sources[name],
                    source_text_by_doc,
                    dictionary,
                )
            )
            source_variant_name = f"E_{name.upper()}_FULL_KNOWN"
            variants.append(
                _materialize_variant(
                    source_variant_name,
                    module="source",
                    rows=canonical_rows,
                    base=base,
                    decisions=source_decisions,
                    counters=source_counters,
                    run_dir=run_output.run_dir,
                    documents=documents,
                    dictionary=dictionary,
                )
            )
            combined_rows, combined_decisions, combined_counters = (
                apply_selective_assertions(
                    canonical_rows,
                    source_text_by_doc,
                    regimes=("negation", "history"),
                    preserve_existing=False,
                )
            )
            variants.append(
                _materialize_variant(
                    f"{source_variant_name}_A_NEG_HIST",
                    module="assertion",
                    rows=combined_rows,
                    base=canonical_rows,
                    decisions=combined_decisions,
                    counters=combined_counters,
                    run_dir=run_output.run_dir,
                    documents=documents,
                    dictionary=dictionary,
                )
            )

    if len(routed_sources) >= config.minimum_agreement_sources:
        consensus_rows, decisions, counters = merge_region_routed_proposals(
            base,
            routed_sources,
            source_text_by_doc,
            policy=RegionProposalPolicy(
                minimum_agreement_sources=config.minimum_agreement_sources,
                allowed_single_source_regions=frozenset(),
            ),
        )
        variants.append(
            _materialize_variant(
                "E_SOURCE_CONSENSUS",
                module="entity",
                rows=consensus_rows,
                base=base,
                decisions=decisions,
                counters=counters,
                run_dir=run_output.run_dir,
                documents=documents,
                dictionary=dictionary,
            )
        )

    manifest = json.loads(run_output.manifest_path.read_text(encoding="utf-8"))
    manifest["probe_suite"] = {
        "schema_version": "phase1-round2-breakthrough-probes.v1",
        "baseline": {
            "path": str(config.base),
            "sha256": observed_base_sha256,
            "entity_count": sum(len(rows) for rows in base.values()),
            "entity_projection_sha256": _entity_projection_sha256(base),
        },
        "proposal_source_status": {
            "count": len(loaded_sources),
            "internally_consensused": list(config.consensus_source_names),
            "sources": {
                name: {
                    "path": str(path),
                    "sha256": _path_sha256(path),
                }
                for name, path in config.proposal_sources
            },
        },
        "reviewed_rxnorm_source": (
            {
                "path": str(config.reviewed_rxnorm_map_path),
                "sha256": _path_sha256(config.reviewed_rxnorm_map_path),
                "minimum_occurrence_support": (
                    config.reviewed_rxnorm_min_occurrence_support
                ),
                "minimum_document_support": (
                    config.reviewed_rxnorm_min_document_support
                ),
            }
            if config.reviewed_rxnorm_map_path is not None
            else None
        ),
        "variants": variants,
        "public_promotion_gates": {
            "entity": {"minimum_wer_reduction": 2.0, "final_must_not_decrease": True},
            "assertion": {
                "minimum_j_assertion_gain": 1.0,
                "final_must_not_decrease": True,
            },
            "candidate": {
                "minimum_j_candidate_gain": 0.5,
                "final_must_not_decrease": True,
            },
        },
    }
    write_json(run_output.manifest_path, manifest)
    (run_output.run_dir / "summary.md").write_text(
        _render_summary(manifest["probe_suite"]),
        encoding="utf-8",
    )
    return {
        "run_id": run_output.run_id,
        "run_dir": str(run_output.run_dir),
        "run_manifest": str(run_output.manifest_path),
        "variants": variants,
    }


def _materialize_variant(
    name: str,
    *,
    module: Literal["assertion", "candidate", "entity", "source"],
    rows: Mapping[str, list[dict[str, Any]]],
    base: Mapping[str, list[dict[str, Any]]],
    decisions: list[dict[str, Any]],
    counters: Mapping[str, int],
    run_dir: Path,
    documents: Sequence[ClinicalDocument],
    dictionary: DictionaryStore,
) -> dict[str, Any]:
    variant_dir = run_dir / "variants" / name
    output_dir = variant_dir / "output"
    _write_phase1_rows(rows, output_dir)
    _strict_validate_directory(output_dir, documents, dictionary)
    zip_path = variant_dir / "output.zip"
    zip_phase1_output_dir(output_dir, zip_path)
    zip_issues = validate_phase1_submission_zip(
        zip_path,
        documents=documents,
        dictionary=dictionary,
        expected_count=len(documents),
    )
    if zip_issues:
        raise ValueError(
            f"{name} ZIP validation failed: {[issue.to_json() for issue in zip_issues[:5]]}"
        )
    isolation_issues = (
        []
        if module == "source"
        else validate_probe_isolation(base, rows, module=module)
    )
    if isolation_issues:
        raise ValueError(f"{name} isolation failed: {isolation_issues[:5]}")
    write_jsonl(variant_dir / "decisions.jsonl", decisions)
    report = {
        "name": name,
        "module": module,
        "changed": _change_counts(base, rows),
        "counters": dict(counters),
        "entity_projection_sha256": _entity_projection_sha256(rows),
        "validation_issue_count": 0,
        "isolation_issues": [],
        "output_dir": str(output_dir),
        "zip": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
    }
    write_json(variant_dir / "variant_report.json", report)
    return report


def _strict_validate_artifact(
    artifact: Path,
    *,
    documents: Sequence[ClinicalDocument],
    dictionary: DictionaryStore,
    expected_count: int,
) -> None:
    """Validate the frozen baseline without extracting or rewriting its payload."""

    if artifact.is_file() and artifact.suffix.lower() == ".zip":
        issues = validate_phase1_submission_zip(
            artifact,
            documents=documents,
            dictionary=dictionary,
            expected_count=expected_count,
        )
    elif artifact.is_dir():
        issues = validate_phase1_submission_documents(
            documents,
            artifact,
            dictionary=dictionary,
        )
    else:
        raise ValueError(f"Unsupported Phase 1 artifact: {artifact}")
    if issues:
        raise ValueError(
            "Frozen Round 2 baseline validation failed: "
            f"{[issue.to_json() for issue in issues[:5]]}"
        )


def _strict_validate_directory(
    output_dir: Path,
    documents: Sequence[ClinicalDocument],
    dictionary: DictionaryStore,
) -> None:
    issues = validate_phase1_submission_documents(
        documents,
        output_dir,
        dictionary=dictionary,
    )
    if issues:
        raise ValueError(
            f"Round 2 probe validation failed: {[issue.to_json() for issue in issues[:5]]}"
        )


def _write_phase1_rows(
    rows_by_doc: Mapping[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for document_id, rows in sorted(
        rows_by_doc.items(),
        key=lambda item: _document_sort_key(item[0]),
    ):
        (output_dir / f"{document_id}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _load_dictionary(paths: Sequence[Path]) -> DictionaryStore:
    entries = []
    for path in paths:
        entries.extend(DictionaryStore.load_entries_jsonl(path))
    return DictionaryStore(entries)


def _document_default_region(source_text: str) -> Round2RegionKind:
    if _QUESTION_ANSWER_HEADING_RE.search(source_text):
        return "question_answer"
    if _EDUCATIONAL_RE.search(source_text):
        return "educational"
    return "other"


def _line_region_kind(line: str) -> Round2RegionKind | None:
    if _MEDICATION_HEADING_RE.search(line):
        return "medication_list"
    if _QUESTION_ANSWER_HEADING_RE.search(line):
        return "question_answer"
    if _CLINICAL_HEADING_RE.search(line):
        return "clinical"
    if _EDUCATIONAL_RE.search(line):
        return "educational"
    return None


def _region_for_span(
    regions: Sequence[Phase1TextRegion],
    start: int,
    end: int,
) -> Round2RegionKind | None:
    for region in regions:
        if region.span[0] <= start and end <= region.span[1]:
            return region.kind
    return None


def _exact_occurrences(source_text: str, mention: str) -> list[int]:
    occurrences: list[int] = []
    cursor = 0
    while cursor <= len(source_text) - len(mention):
        start = source_text.find(mention, cursor)
        if start < 0:
            break
        occurrences.append(start)
        cursor = start + max(1, len(mention))
    return occurrences


def _validate_entity_identity(
    row: Mapping[str, Any],
    source_text: str,
    *,
    document_id: str,
) -> None:
    text = row.get("text")
    entity_type = row.get("type")
    if not isinstance(text, str) or entity_type not in PHASE1_ALLOWED_TYPES:
        raise ValueError(f"{document_id}: invalid entity text/type")
    start, end = _position(row)
    # INVARIANT: proposal artifacts may carry no trusted metadata, but their raw identity is exact.
    if start < 0 or end < start or end > len(source_text) or source_text[start:end] != text:
        raise ValueError(
            f"{document_id}: proposal offset mismatch for {text!r} at {(start, end)}"
        )


def _position(row: Mapping[str, Any]) -> tuple[int, int]:
    value = row.get("position")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) for item in value)
    ):
        raise ValueError(f"Invalid Phase 1 position: {value!r}")
    return int(value[0]), int(value[1])


def _new_entity_row(
    text: str,
    entity_type: str,
    start: int,
    end: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "text": text,
        "type": entity_type,
        "assertions": [],
        "candidates": [],
        "position": [start, end],
    }
    return row


def _copy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    for field in ("position", "assertions", "candidates"):
        value = copied.get(field)
        if isinstance(value, list):
            copied[field] = list(value)
    return copied


def _identity_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    start, end = _position(row)
    return str(row.get("type", "")), str(row.get("text", "")), start, end


def _identity_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    start, end = _position(row)
    return {
        "text": row.get("text"),
        "type": row.get("type"),
        "position": [start, end],
    }


def _rows_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_start, left_end = _position(left)
    right_start, right_end = _position(right)
    return left_start < right_end and right_start < left_end


def _proposal_priority(
    item: tuple[dict[str, Any], frozenset[str], Round2RegionKind],
) -> tuple[int, int, int, int, str, str]:
    row, sources, _ = item
    start, end = _position(row)
    return (
        -len(sources),
        -PHASE1_TYPE_PRIORITY.get(str(row.get("type")), 0),
        -(end - start),
        start,
        str(row.get("type", "")),
        str(row.get("text", "")),
    )


def _row_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str, str]:
    start, end = _position(row)
    return start, end, str(row.get("type", "")), str(row.get("text", ""))


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)


def _change_counts(
    base: Mapping[str, list[dict[str, Any]]],
    trial: Mapping[str, list[dict[str, Any]]],
) -> dict[str, int]:
    counters: Counter[str] = Counter()
    for document_id in set(base) | set(trial):
        base_rows = {_identity_key(row): row for row in base.get(document_id, [])}
        trial_rows = {_identity_key(row): row for row in trial.get(document_id, [])}
        counters["entity_added"] += len(set(trial_rows) - set(base_rows))
        counters["entity_removed"] += len(set(base_rows) - set(trial_rows))
        for key in set(base_rows) & set(trial_rows):
            before = base_rows[key]
            after = trial_rows[key]
            if before.get("assertions", []) != after.get("assertions", []):
                counters["assertion_changed"] += 1
            if before.get("candidates", []) != after.get("candidates", []):
                counters["candidate_changed"] += 1
    counters["changed_row_count"] = sum(
        counters[key]
        for key in (
            "entity_added",
            "entity_removed",
            "assertion_changed",
            "candidate_changed",
        )
    )
    return dict(sorted(counters.items()))


def _entity_projection_sha256(
    rows_by_doc: Mapping[str, list[dict[str, Any]]],
) -> str:
    payload = [
        {
            "document_id": document_id,
            "entities": [
                _identity_payload(row)
                for row in sorted(rows, key=_row_sort_key)
            ],
        }
        for document_id, rows in sorted(
            rows_by_doc.items(),
            key=lambda item: _document_sort_key(item[0]),
        )
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _path_sha256(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise ValueError(f"Artifact path does not exist: {path}")
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _render_summary(probe_suite: Mapping[str, Any]) -> str:
    baseline = probe_suite["baseline"]
    lines = [
        "# Phase 1 Round 2 Breakthrough Probes",
        "",
        f"- Baseline SHA-256: `{baseline['sha256']}`",
        f"- Baseline entities: {baseline['entity_count']}",
        f"- Proposal sources: {probe_suite['proposal_source_status']['count']}",
        "",
        "| Variant | Module | Added | Assertion changes | Candidate changes | ZIP SHA-256 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for variant in probe_suite["variants"]:
        changed = variant["changed"]
        lines.append(
            f"| `{variant['name']}` | {variant['module']} | "
            f"{changed.get('entity_added', 0)} | "
            f"{changed.get('assertion_changed', 0)} | "
            f"{changed.get('candidate_changed', 0)} | "
            f"`{variant['zip_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "Public promotion is isolated: entity WER must decrease by at least 2.0, while an "
            "assertion probe must gain at least 1.0 J_assertion and a candidate probe at least "
            "0.5 J_candidates. Final score must not decrease.",
            "",
        ]
    )
    return "\n".join(lines)
