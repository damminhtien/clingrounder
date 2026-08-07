"""Deterministic composition of calibrated Phase 1 inference artifacts.

Model execution remains behind source adapters. This module consumes their pinned, raw-offset
proposal artifacts and owns only final fusion, metadata policy, validation, and audit records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from clingrounder.benchmarks.phase1.phase1 import (
    Phase1ValidationIssue,
    validate_phase1_entities,
)
from clingrounder.benchmarks.phase1.boundary_overlay import (
    BoundaryPolicy,
    apply_conservative_boundary_overlay,
)
from clingrounder.benchmarks.phase1.boundary_verifier import (
    Phase1BoundaryVerifier,
    ScoredPhase1BoundaryVariant,
    resolve_phase1_boundary_rows,
)
from clingrounder.benchmarks.phase1.phase1_proposals import (
    build_phase1_proposal_matrix,
)
from clingrounder.benchmarks.phase1.phase1_selective_overlays import (
    AssertionRegime,
    apply_selective_assertions,
)
from clingrounder.benchmarks.phase1.proposal_calibration import (
    Phase1ProposalVerifier,
    ScoredPhase1Proposal,
    resolve_phase1_proposal_rows,
)
from clingrounder.benchmarks.phase1.proposal_features import ProposalSourceRole
from clingrounder.benchmarks.phase1.round2_probes import (
    apply_round2_candidate_policy,
    canonicalize_full_phase1_source,
)
from clingrounder.dictionaries.dictionary_store import DictionaryStore
from clingrounder.ner.dictionary_matcher import DictionaryMatcher
from clingrounder.benchmarks.phase1.ontology import (
    PHASE1_CODABLE_TYPES,
    PHASE1_CODE_SYSTEM_BY_TYPE,
    PHASE1_RULE_BY_TYPE,
)
from clingrounder.schema.document import ClinicalDocument

__all__ = [
    "BoundaryPolicy",
    "CandidateMetadataPolicy",
    "Phase1MaxScorePipeline",
    "Phase1MaxScoreResult",
    "hydrate_phase1_selected_candidate_metadata",
    "validate_phase1_inference_budget",
]

CandidateMetadataPolicy = Literal["keep", "rx_unique_keep_icd"]

_REQUIRED_BUDGET_ROLES = frozenset(
    {
        "assertion",
        "candidate_rerank",
        "ner",
        "recall",
        "verifier",
    }
)


@dataclass(frozen=True, slots=True)
class Phase1MaxScoreResult:
    """Final rows plus enough evidence to reproduce every acceptance decision."""

    rows_by_document: Mapping[str, tuple[Mapping[str, Any], ...]]
    proposal_matrix: Mapping[str, Any]
    proposal_scores: tuple[ScoredPhase1Proposal, ...]
    boundary_scores: tuple[ScoredPhase1BoundaryVariant, ...]
    source_decisions: tuple[Mapping[str, Any], ...]
    boundary_decisions: tuple[Mapping[str, Any], ...]
    assertion_decisions: tuple[Mapping[str, Any], ...]
    candidate_decisions: tuple[Mapping[str, Any], ...]
    counters: Mapping[str, int]
    boundary_report: Mapping[str, int | float | None | str]
    budget_manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Phase1MaxScorePipeline:
    """Compose independently generated proposals without loading their base models.

    SCALING: expensive model inference can run once per pinned source and resume independently.
    This final stage is CPU-only and deterministic, so changing an assertion or candidate policy
    never requires reloading Qwen or either XLM-R checkpoint.
    """

    verifier: Phase1ProposalVerifier
    source_roles: Mapping[str, ProposalSourceRole]
    budget_manifest: Mapping[str, Any]
    dictionary: DictionaryStore
    candidate_source_priority: tuple[str, ...]
    assertion_regimes: tuple[AssertionRegime, ...] = ("negation", "history")
    candidate_policy: CandidateMetadataPolicy = "rx_unique_keep_icd"
    boundary_verifier: Phase1BoundaryVerifier | None = None
    boundary_policy: BoundaryPolicy = BoundaryPolicy()

    def __post_init__(self) -> None:
        source_names = frozenset(self.source_roles)
        if len(source_names) < 2:
            raise ValueError("Max-score fusion requires at least two independent sources")
        if len(self.candidate_source_priority) != len(
            set(self.candidate_source_priority)
        ):
            raise ValueError("Candidate source priority contains duplicates")
        if set(self.candidate_source_priority) != source_names:
            raise ValueError(
                "Candidate source priority must name every proposal source exactly once"
            )
        if self.candidate_policy not in {"keep", "rx_unique_keep_icd"}:
            raise ValueError(f"Unsupported candidate policy {self.candidate_policy!r}")
        if (
            self.boundary_policy.mode == "conservative_replacement"
            and self.boundary_verifier is None
        ):
            raise ValueError("Boundary replacement requires a pinned boundary verifier")
        if self.boundary_verifier is not None:
            if self.boundary_policy.mode != "conservative_replacement":
                raise ValueError("A boundary verifier requires an active boundary policy")
            if self.boundary_verifier.resolution_policy != "conservative_replacement":
                raise ValueError("Max-score pipeline refuses open-ranker boundary output")
            if not self.boundary_verifier.requires_base_probability:
                raise ValueError(
                    "Max-score boundary replacement requires proposal-conditioned scoring"
                )
        _validate_budget_manifest(self.budget_manifest)

    def run(
        self,
        documents: Sequence[ClinicalDocument],
        proposal_sources: Mapping[
            str,
            Mapping[str, Sequence[Mapping[str, Any]]],
        ],
    ) -> Phase1MaxScoreResult:
        """Resolve entities, derive assertions, select candidates, and validate output."""

        source_text_by_document = _source_text_by_document(documents)
        if set(proposal_sources) != set(self.source_roles):
            raise ValueError("Proposal source names do not match the configured source roles")

        canonical_sources: dict[str, dict[str, list[dict[str, Any]]]] = {}
        source_decisions: list[Mapping[str, Any]] = []
        counters: dict[str, int] = {}
        for source_name in sorted(proposal_sources):
            raw_rows = {
                document_id: [dict(row) for row in rows]
                for document_id, rows in proposal_sources[source_name].items()
            }
            canonical, decisions, source_counters = canonicalize_full_phase1_source(
                raw_rows,
                source_text_by_document,
                self.dictionary,
                preserve_proposal_metadata=True,
            )
            canonical_sources[source_name] = canonical
            source_decisions.extend(
                {"source": source_name, **decision} for decision in decisions
            )
            _merge_counters(counters, f"source.{source_name}", source_counters)

        matrix = build_phase1_proposal_matrix(
            canonical_sources,
            source_text_by_document,
            source_metadata={
                source: {"role": role.value}
                for source, role in sorted(self.source_roles.items())
            },
        )
        eligible_proposals: list[Mapping[str, Any]] = []
        for row in matrix["matrix"]:
            sources = row.get("sources", [])
            if not isinstance(sources, list):
                raise ValueError("Proposal matrix sources must be a list")
            if sources and all(
                self.source_roles[str(source)] is ProposalSourceRole.VERIFIER
                for source in sources
            ):
                # MODEL: VietMed is source-task evidence. It may strengthen an exact proposal
                # from a target-task source but cannot own a Phase 1 entity by itself.
                source_decisions.append(
                    {
                        "document_id": row.get("document_id"),
                        "proposal_id": row.get("proposal_id"),
                        "stage": "proposal_source_eligibility",
                        "action": "block",
                        "reason": "verifier_only_proposal",
                        "entity": _identity_payload(row),
                    }
                )
                counters["entity.verifier_only_blocked"] = (
                    counters.get("entity.verifier_only_blocked", 0) + 1
                )
                continue
            eligible_proposals.append(row)
        resolved, proposal_scores = resolve_phase1_proposal_rows(
            eligible_proposals,
            source_text_by_document,
            self.verifier,
            source_roles=self.source_roles,
        )
        _merge_counters(
            counters,
            "entity",
            {
                "proposal_total": len(proposal_scores),
                "selected": sum(item.selected for item in proposal_scores),
                "blocked": sum(not item.selected for item in proposal_scores),
            },
        )

        boundary_scores: tuple[ScoredPhase1BoundaryVariant, ...] = ()
        boundary_decisions: tuple[Mapping[str, Any], ...] = ()
        boundary_report: Mapping[str, int | float | None | str] = {
            "replacement_count": 0,
            "replacement_correct": None,
            "replacement_precision": None,
            "base_errors_fixed": None,
            "correct_bases_destroyed": None,
            "net_exact_span_gain": None,
            "label_status": "boundary_policy_disabled",
        }
        repaired = resolved
        changed_identities: frozenset[tuple[str, str, str, int, int]] = frozenset()
        if self.boundary_verifier is not None:
            # MODEL: the boundary model scores alternatives around all eligible proposals, while
            # the overlay remains anchored to the proposal verifier's accepted base identities.
            boundary_resolved, boundary_scores = resolve_phase1_boundary_rows(
                eligible_proposals,
                source_text_by_document,
                self.boundary_verifier,
                proposal_verifier=self.verifier,
                source_roles=self.source_roles,
                dictionary_matcher=DictionaryMatcher(self.dictionary.aliases_for_ner()),
            )
            overlay = apply_conservative_boundary_overlay(
                base_rows=resolved,
                boundary_rows=boundary_resolved,
                scored_variants=boundary_scores,
                source_text_by_document=source_text_by_document,
                verifier=self.boundary_verifier,
                policy=self.boundary_policy,
            )
            repaired = {
                document_id: [dict(row) for row in rows]
                for document_id, rows in overlay.rows_by_document.items()
            }
            changed_identities = overlay.changed_identities
            boundary_decisions = overlay.decisions
            boundary_report = overlay.diagnostic_report
            _merge_counters(counters, "boundary", overlay.counters)

        hydrated, metadata_decisions = _hydrate_selected_candidates(
            repaired,
            canonical_sources,
            self.candidate_source_priority,
            dictionary=self.dictionary,
            changed_identities=changed_identities,
        )
        source_decisions.extend(metadata_decisions)
        asserted, assertion_decisions, assertion_counters = (
            apply_selective_assertions(
                hydrated,
                source_text_by_document,
                regimes=self.assertion_regimes,
                preserve_existing=False,
            )
        )
        _merge_counters(counters, "assertion", assertion_counters)

        if self.candidate_policy == "rx_unique_keep_icd":
            final_rows, candidate_decisions, candidate_counters = (
                apply_round2_candidate_policy(
                    asserted,
                    policy="rx_unique_keep_icd",
                )
            )
        else:
            final_rows = {
                document_id: [dict(row) for row in rows]
                for document_id, rows in asserted.items()
            }
            candidate_decisions = []
            candidate_counters = {
                "output_entity_total": sum(
                    len(rows) for rows in final_rows.values()
                ),
                "output_candidate_rows": sum(
                    bool(row.get("candidates"))
                    for rows in final_rows.values()
                    for row in rows
                ),
            }
        _merge_counters(counters, "candidate", candidate_counters)

        validation_issues = _validate_rows(
            documents,
            final_rows,
            self.dictionary,
        )
        if validation_issues:
            first = validation_issues[0]
            raise ValueError(
                "Max-score output validation failed: "
                f"{first.document_id}:{first.path}:{first.kind}:{first.message}"
            )
        return Phase1MaxScoreResult(
            rows_by_document={
                document_id: tuple(dict(row) for row in rows)
                for document_id, rows in final_rows.items()
            },
            proposal_matrix=matrix,
            proposal_scores=proposal_scores,
            boundary_scores=boundary_scores,
            source_decisions=tuple(source_decisions),
            boundary_decisions=boundary_decisions,
            assertion_decisions=tuple(assertion_decisions),
            candidate_decisions=tuple(candidate_decisions),
            counters=dict(sorted(counters.items())),
            boundary_report=boundary_report,
            budget_manifest=dict(self.budget_manifest),
        )


def _hydrate_selected_candidates(
    resolved: Mapping[str, Sequence[Mapping[str, Any]]],
    sources: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    source_priority: Sequence[str],
    *,
    dictionary: DictionaryStore,
    changed_identities: frozenset[tuple[str, str, str, int, int]],
) -> tuple[dict[str, list[dict[str, Any]]], list[Mapping[str, Any]]]:
    """Hydrate unchanged source metadata and relink changed boundary identities.

    INVARIANT: a candidate list is never copied from a span with different raw offsets. Boundary
    replacements use an exact unique terminology pass instead of inheriting old metadata.
    """

    indexes = {
        source_name: {
            (document_id, *_identity_key(row)): row
            for document_id, rows in source_rows.items()
            for row in rows
        }
        for source_name, source_rows in sources.items()
    }
    output: dict[str, list[dict[str, Any]]] = {}
    decisions: list[Mapping[str, Any]] = []
    for document_id, rows in resolved.items():
        hydrated_rows: list[dict[str, Any]] = []
        for row in rows:
            hydrated = dict(row)
            if str(row.get("type", "")) not in PHASE1_CODABLE_TYPES:
                hydrated_rows.append(hydrated)
                continue
            identity = (document_id, *_identity_key(row))
            hydrated["candidates"] = []
            if identity in changed_identities:
                candidates = _relink_changed_identity(row, dictionary)
                hydrated["candidates"] = candidates
                decisions.append(
                    {
                        "document_id": document_id,
                        "stage": "candidate_metadata_relink",
                        "action": (
                            "emit_exact_unique" if candidates else "abstain_ambiguous_or_missing"
                        ),
                        "candidate_count": len(candidates),
                        "entity": _identity_payload(row),
                    }
                )
                hydrated_rows.append(hydrated)
                continue
            for source_name in source_priority:
                source_row = indexes[source_name].get(identity)
                if source_row is None:
                    continue
                raw_candidates = source_row.get("candidates", [])
                if not isinstance(raw_candidates, list) or not all(
                    isinstance(value, str) for value in raw_candidates
                ):
                    raise ValueError(
                        f"{source_name}:{document_id}: candidates must be a string list"
                    )
                if not raw_candidates:
                    continue
                hydrated["candidates"] = list(raw_candidates)
                decisions.append(
                    {
                        "document_id": document_id,
                        "stage": "candidate_metadata_hydration",
                        "action": "copy_exact_identity",
                        "source": source_name,
                        "candidate_count": len(raw_candidates),
                        "entity": _identity_payload(row),
                    }
                )
                break
            hydrated_rows.append(hydrated)
        output[document_id] = hydrated_rows
    return output, decisions


def hydrate_phase1_selected_candidate_metadata(
    resolved: Mapping[str, Sequence[Mapping[str, Any]]],
    sources: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    source_priority: Sequence[str],
    *,
    dictionary: DictionaryStore,
    changed_identities: frozenset[tuple[str, str, str, int, int]],
) -> tuple[dict[str, list[dict[str, Any]]], list[Mapping[str, Any]]]:
    """Hydrate candidates only for raw identities accepted by a resolver.

    This public boundary is shared by the legacy and joint span pipelines. A generated lattice
    variant is passed through ``changed_identities`` so it receives a fresh exact lookup instead
    of metadata from the shorter or longer proposal that generated it.
    """

    return _hydrate_selected_candidates(
        resolved,
        sources,
        source_priority,
        dictionary=dictionary,
        changed_identities=changed_identities,
    )


def _relink_changed_identity(
    row: Mapping[str, Any],
    dictionary: DictionaryStore,
) -> list[str]:
    """Rerun exact type-filtered linking for a boundary-replaced span.

    The final candidate policy still runs after this function. Here we only recover a single
    exact terminology code; fuzzy retrieval would turn a boundary-only probe into a linking probe.
    """

    entity_type = str(row.get("type", ""))
    expected_system = PHASE1_CODE_SYSTEM_BY_TYPE[entity_type]
    expected_internal_type = PHASE1_RULE_BY_TYPE[entity_type].internal_type
    codes = {
        entry.code
        for entry in dictionary.exact_lookup(str(row.get("text", "")))
        if entry.code is not None
        and entry.code_system is expected_system
        and entry.semantic_type is expected_internal_type
    }
    return sorted(codes) if len(codes) == 1 else []


def _validate_budget_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed before any source is fused under the competition budget."""

    if manifest.get("schema_version") != "inference-model-budget.v2":
        raise ValueError("Max-score pipeline requires a verified v2 inference budget")
    if manifest.get("status") != "verified":
        raise ValueError("Inference budget manifest is not verified")
    total = manifest.get("total_parameters")
    maximum = manifest.get("maximum_parameters")
    if (
        not isinstance(total, int)
        or isinstance(total, bool)
        or not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or total > maximum
    ):
        raise ValueError("Inference model budget is invalid or exceeded")
    raw_active = manifest.get("active")
    if not isinstance(raw_active, list):
        raise ValueError("Inference budget active artifacts must be a list")
    artifact_ids: set[str] = set()
    roles: set[str] = set()
    for item in raw_active:
        if not isinstance(item, Mapping):
            raise ValueError("Inference budget active artifact must be an object")
        artifact_id = str(item.get("artifact_id", "")).strip()
        if not artifact_id or artifact_id in artifact_ids:
            raise ValueError("Inference budget artifact IDs must be non-empty and unique")
        artifact_ids.add(artifact_id)
        raw_roles = item.get("roles")
        if not isinstance(raw_roles, list) or not all(
            isinstance(role, str) and role.strip() for role in raw_roles
        ):
            raise ValueError("Inference budget artifact roles must be strings")
        roles.update(raw_roles)
    missing_roles = _REQUIRED_BUDGET_ROLES - roles
    if missing_roles:
        raise ValueError(
            f"Inference budget is missing required roles: {sorted(missing_roles)}"
        )


def validate_phase1_inference_budget(manifest: Mapping[str, Any]) -> None:
    """Validate an already-verified budget before either final resolver consumes it."""

    _validate_budget_manifest(manifest)


def _source_text_by_document(
    documents: Sequence[ClinicalDocument],
) -> dict[str, str]:
    source_text = {document.document_id: document.text for document in documents}
    if not source_text:
        raise ValueError("Max-score pipeline requires source documents")
    if len(source_text) != len(documents):
        raise ValueError("Max-score source documents contain duplicate IDs")
    return source_text


def _validate_rows(
    documents: Sequence[ClinicalDocument],
    rows_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    dictionary: DictionaryStore,
) -> list[Phase1ValidationIssue]:
    expected_ids = {document.document_id for document in documents}
    if set(rows_by_document) != expected_ids:
        raise ValueError("Max-score output document IDs do not match source documents")
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
    return issues


def _identity_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    position = row.get("position")
    if (
        not isinstance(position, list)
        or len(position) != 2
        or not all(isinstance(value, int) for value in position)
    ):
        raise ValueError("Phase 1 entity position must be two integers")
    return (
        str(row.get("type", "")),
        str(row.get("text", "")),
        position[0],
        position[1],
    )


def _identity_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    entity_type, text, start, end = _identity_key(row)
    return {
        "text": text,
        "type": entity_type,
        "position": [start, end],
    }


def _merge_counters(
    target: dict[str, int],
    prefix: str,
    values: Mapping[str, int],
) -> None:
    for name, value in values.items():
        target[f"{prefix}.{name}"] = target.get(f"{prefix}.{name}", 0) + int(value)
