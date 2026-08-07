"""Final Phase 1 composition driven by a learned joint span/type verifier.

Unlike the legacy proposal-calibration path, this pipeline scores a bounded candidate lattice and
then resolves all overlapping spans globally. Assertions and terminology links are deliberately
derived only after the raw entity identity is selected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from clingrounder.benchmarks.phase1.joint_span import (
    Phase1JointSpanSelectionPolicy,
    Phase1JointSpanVerifierPort,
    ScoredPhase1JointSpanCandidate,
    generate_phase1_joint_span_lattice,
    resolve_phase1_joint_span_lattice,
)
from clingrounder.benchmarks.phase1.max_score_pipeline import (
    CandidateMetadataPolicy,
    hydrate_phase1_selected_candidate_metadata,
    validate_phase1_inference_budget,
)
from clingrounder.benchmarks.phase1.phase1 import (
    Phase1ValidationIssue,
    validate_phase1_entities,
)
from clingrounder.benchmarks.phase1.phase1_proposals import (
    build_phase1_proposal_matrix,
)
from clingrounder.benchmarks.phase1.phase1_selective_overlays import (
    AssertionRegime,
    apply_selective_assertions,
)
from clingrounder.benchmarks.phase1.proposal_features import ProposalSourceRole
from clingrounder.benchmarks.phase1.round2_probes import (
    apply_round2_candidate_policy,
    canonicalize_full_phase1_source,
)
from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.ner.dictionary_matcher import DictionaryMatcher
from clingrounder.schema.document import ClinicalDocument

__all__ = ["Phase1JointSpanPipeline", "Phase1JointSpanResult"]


@dataclass(frozen=True, slots=True)
class Phase1JointSpanResult:
    """Rows and decision evidence from one deterministic joint-span composition."""

    rows_by_document: Mapping[str, tuple[Mapping[str, Any], ...]]
    proposal_matrix: Mapping[str, Any]
    joint_scores: tuple[ScoredPhase1JointSpanCandidate, ...]
    source_decisions: tuple[Mapping[str, Any], ...]
    assertion_decisions: tuple[Mapping[str, Any], ...]
    candidate_decisions: tuple[Mapping[str, Any], ...]
    counters: Mapping[str, int]
    budget_manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Phase1JointSpanPipeline:
    """Compose pinned proposal artifacts through a learned global span/type resolver.

    SCALING: model sources can be generated independently and kept on disk. This CPU-only
    composition only runs the compact verifier over bounded candidates, so it remains resumable
    and deterministic for every assertion or linking experiment.
    """

    verifier: Phase1JointSpanVerifierPort
    selection_policy: Phase1JointSpanSelectionPolicy
    source_roles: Mapping[str, ProposalSourceRole]
    budget_manifest: Mapping[str, Any]
    dictionary: DictionaryStore
    candidate_source_priority: tuple[str, ...]
    assertion_regimes: tuple[AssertionRegime, ...] = ("negation", "history")
    candidate_policy: CandidateMetadataPolicy = "rx_unique_keep_icd"

    def __post_init__(self) -> None:
        source_names = frozenset(self.source_roles)
        if len(source_names) < 2:
            raise ValueError("Joint span fusion requires at least two independent sources")
        if len(self.candidate_source_priority) != len(set(self.candidate_source_priority)):
            raise ValueError("Candidate source priority contains duplicates")
        if set(self.candidate_source_priority) != source_names:
            raise ValueError("Candidate source priority must name every source exactly once")
        if self.candidate_policy not in {"keep", "rx_unique_keep_icd"}:
            raise ValueError(f"Unsupported candidate policy {self.candidate_policy!r}")
        validate_phase1_inference_budget(self.budget_manifest)

    def run(
        self,
        documents: Sequence[ClinicalDocument],
        proposal_sources: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    ) -> Phase1JointSpanResult:
        """Select raw span/type identities, then derive all identity-bound metadata."""

        source_text_by_document = _source_text_by_document(documents)
        if set(proposal_sources) != set(self.source_roles):
            raise ValueError("Proposal source names do not match configured source roles")

        canonical_sources: dict[str, dict[str, list[dict[str, Any]]]] = {}
        source_decisions: list[Mapping[str, Any]] = []
        counters: dict[str, int] = {}
        for source_name in sorted(proposal_sources):
            canonical, decisions, source_counters = canonicalize_full_phase1_source(
                {
                    document_id: [dict(row) for row in rows]
                    for document_id, rows in proposal_sources[source_name].items()
                },
                source_text_by_document,
                self.dictionary,
                preserve_proposal_metadata=True,
            )
            canonical_sources[source_name] = canonical
            source_decisions.extend({"source": source_name, **decision} for decision in decisions)
            _merge_counters(counters, f"source.{source_name}", source_counters)

        matrix = build_phase1_proposal_matrix(
            canonical_sources,
            source_text_by_document,
            source_metadata={
                source: {"role": role.value}
                for source, role in sorted(self.source_roles.items())
            },
        )
        eligible = _eligible_matrix_rows(
            matrix,
            self.source_roles,
            source_decisions=source_decisions,
            counters=counters,
        )
        matcher = DictionaryMatcher(self.dictionary.aliases_for_ner())
        candidates = tuple(
            candidate
            for document in documents
            for candidate in generate_phase1_joint_span_lattice(
                document.document_id,
                document.text,
                eligible[document.document_id],
                source_roles=self.source_roles,
                dictionary_matcher=matcher,
            )
        )
        selected, joint_scores = resolve_phase1_joint_span_lattice(
            candidates,
            self.verifier,
            policy=self.selection_policy,
        )
        resolved = {
            document_id: _entity_rows(selected.get(document_id, []))
            for document_id in source_text_by_document
        }
        _merge_counters(
            counters,
            "entity",
            {
                "proposal_total": len(joint_scores),
                "selected": sum(item.selected for item in joint_scores),
                "blocked": sum(not item.selected for item in joint_scores),
            },
        )

        source_identities = {
            (document_id, *_identity_key(row))
            for rows_by_document in canonical_sources.values()
            for document_id, rows in rows_by_document.items()
            for row in rows
        }
        generated_identities = frozenset(
            (document_id, *_identity_key(row))
            for document_id, rows in resolved.items()
            for row in rows
            if (document_id, *_identity_key(row)) not in source_identities
        )
        hydrated, metadata_decisions = hydrate_phase1_selected_candidate_metadata(
            resolved,
            canonical_sources,
            self.candidate_source_priority,
            dictionary=self.dictionary,
            changed_identities=generated_identities,
        )
        source_decisions.extend(metadata_decisions)
        asserted, assertion_decisions, assertion_counters = apply_selective_assertions(
            hydrated,
            source_text_by_document,
            regimes=self.assertion_regimes,
            preserve_existing=False,
        )
        _merge_counters(counters, "assertion", assertion_counters)
        if self.candidate_policy == "rx_unique_keep_icd":
            final_rows, candidate_decisions, candidate_counters = apply_round2_candidate_policy(
                asserted,
                policy="rx_unique_keep_icd",
            )
        else:
            final_rows = {document_id: [dict(row) for row in rows] for document_id, rows in asserted.items()}
            candidate_decisions = []
            candidate_counters = {
                "output_entity_total": sum(len(rows) for rows in final_rows.values()),
                "output_candidate_rows": sum(
                    bool(row.get("candidates"))
                    for rows in final_rows.values()
                    for row in rows
                ),
            }
        _merge_counters(counters, "candidate", candidate_counters)
        _raise_for_validation_issues(documents, final_rows, self.dictionary)
        return Phase1JointSpanResult(
            rows_by_document={
                document_id: tuple(dict(row) for row in rows)
                for document_id, rows in final_rows.items()
            },
            proposal_matrix=matrix,
            joint_scores=joint_scores,
            source_decisions=tuple(source_decisions),
            assertion_decisions=tuple(assertion_decisions),
            candidate_decisions=tuple(candidate_decisions),
            counters=dict(sorted(counters.items())),
            budget_manifest=dict(self.budget_manifest),
        )


def _eligible_matrix_rows(
    matrix: Mapping[str, Any],
    source_roles: Mapping[str, ProposalSourceRole],
    *,
    source_decisions: list[Mapping[str, Any]],
    counters: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    """Exclude support-only proposals while retaining them as exact-span evidence."""

    raw_rows = matrix.get("matrix")
    if not isinstance(raw_rows, list):
        raise ValueError("Proposal matrix did not emit a matrix list")
    output: dict[str, list[dict[str, Any]]] = {}
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError("Proposal matrix row must be an object")
        row = dict(raw_row)
        document_id = str(row.get("document_id", ""))
        sources = row.get("sources")
        if not isinstance(sources, list) or not all(isinstance(source, str) for source in sources):
            raise ValueError("Proposal matrix sources must be a string list")
        if sources and all(source_roles[source] is ProposalSourceRole.VERIFIER for source in sources):
            source_decisions.append(
                {
                    "document_id": document_id,
                    "proposal_id": row.get("proposal_id"),
                    "stage": "proposal_source_eligibility",
                    "action": "block",
                    "reason": "verifier_only_proposal",
                    "entity": _identity_payload(row),
                }
            )
            counters["entity.verifier_only_blocked"] = counters.get("entity.verifier_only_blocked", 0) + 1
            continue
        output.setdefault(document_id, []).append(row)
    return output


def _entity_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Drop resolver-only evidence before entity metadata and schema validation.

    The full scored lattice remains in ``joint_scores``. Carrying provenance fields into a Phase 1
    row would make a structurally valid model decision fail the strict organizer schema.
    """

    return [
        {
            "text": str(row["text"]),
            "type": str(row["type"]),
            "position": list(row["position"]),
            "assertions": [],
            "candidates": [],
        }
        for row in rows
    ]


def _raise_for_validation_issues(
    documents: Sequence[ClinicalDocument],
    rows_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    dictionary: DictionaryStore,
) -> None:
    """Fail before export if learned selection violates hard Phase 1 invariants."""

    expected_ids = {document.document_id for document in documents}
    if set(rows_by_document) != expected_ids:
        raise ValueError("Joint span output document IDs do not match source documents")
    issues: list[Phase1ValidationIssue] = []
    for document in documents:
        issues.extend(
            validate_phase1_entities(
                list(rows_by_document[document.document_id]),
                document.text,
                document_id=document.document_id,
                dictionary=dictionary,
            )
        )
    if issues:
        first = issues[0]
        raise ValueError(
            "Joint span output validation failed: "
            f"{first.document_id}:{first.path}:{first.kind}:{first.message}"
        )


def _source_text_by_document(documents: Sequence[ClinicalDocument]) -> dict[str, str]:
    source_text = {document.document_id: document.text for document in documents}
    if not source_text or len(source_text) != len(documents):
        raise ValueError("Joint span source documents must be non-empty with unique IDs")
    return source_text


def _identity_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    position = row.get("position")
    if (
        not isinstance(position, list)
        or len(position) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in position)
    ):
        raise ValueError("Phase 1 entity position must be two integers")
    return str(row.get("type", "")), str(row.get("text", "")), position[0], position[1]


def _identity_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    entity_type, text, start, end = _identity_key(row)
    return {"text": text, "type": entity_type, "position": [start, end]}


def _merge_counters(target: dict[str, int], prefix: str, values: Mapping[str, int]) -> None:
    for name, value in values.items():
        target[f"{prefix}.{name}"] = target.get(f"{prefix}.{name}", 0) + int(value)
