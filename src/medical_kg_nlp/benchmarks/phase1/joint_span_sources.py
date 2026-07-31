"""Convert offset-safe model outputs into auditable joint span proposal sources."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from medical_kg_nlp.benchmarks.phase1.phase1_proposals import (
    build_phase1_proposal_matrix,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole
from medical_kg_nlp.benchmarks.phase1.reviewed_corpus import Phase1ReviewedCorpus
from medical_kg_nlp.benchmarks.phase1.split_contract import phase1_document_sort_key
from medical_kg_nlp.ontology.phase1 import PHASE1_TYPE_BY_ENTITY_TYPE
from medical_kg_nlp.schema.annotation import EntityAnnotation

__all__ = [
    "EntityProposalExtractorPort",
    "build_phase1_joint_span_proposal_matrix",
    "build_phase1_token_model_proposal_rows",
]


class EntityProposalExtractorPort(Protocol):
    """Emit exact raw-text entity proposals without any benchmark output policy."""

    def extract(self, source_text: str) -> Sequence[EntityAnnotation]:
        """Return independently projected proposal entities for one source text."""


def build_phase1_token_model_proposal_rows(
    corpus: Phase1ReviewedCorpus,
    extractor: EntityProposalExtractorPort,
    *,
    source_name: str,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Materialize one token model as a target-task proposal source.

    INVARIANT: each model span is rechecked against the corpus source before it becomes a lattice
    row. This prevents a tokenizer projection error from being learned or exported downstream.
    """

    if not source_name.strip():
        raise ValueError("Token model proposal source name must be non-empty")
    rows_by_document: dict[str, tuple[dict[str, Any], ...]] = {}
    for document_id in sorted(corpus.source_texts, key=phase1_document_sort_key):
        source_text = corpus.source_texts[document_id]
        rows: list[dict[str, Any]] = []
        seen: set[tuple[int, int, str]] = set()
        for index, entity in enumerate(extractor.extract(source_text)):
            entity.validate_offsets(source_text)
            phase1_type = PHASE1_TYPE_BY_ENTITY_TYPE.get(entity.type)
            if phase1_type is None:
                continue
            start, end = entity.span
            identity = (start, end, phase1_type)
            if identity in seen:
                continue
            seen.add(identity)
            if not 0.0 <= entity.confidence <= 1.0:
                raise ValueError("Token model proposal confidence must be within [0, 1]")
            rows.append(
                {
                    "document_id": document_id,
                    "proposal_id": f"{document_id}:{source_name}:{index}:{start}:{end}:{phase1_type}",
                    "text": entity.text,
                    "type": phase1_type,
                    "position": [start, end],
                    "confidence": entity.confidence,
                    "source_label": entity.type.value,
                }
            )
        rows_by_document[document_id] = tuple(
            sorted(rows, key=lambda row: (row["position"], row["type"], row["proposal_id"]))
        )
    return rows_by_document


def build_phase1_joint_span_proposal_matrix(
    corpus: Phase1ReviewedCorpus,
    sources: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    source_roles: Mapping[str, ProposalSourceRole | str],
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Align every raw source before bounded lattice generation.

    The matrix retains exact agreement and individual confidence evidence. It makes no selection
    decision, so all learned thresholding remains inside the joint verifier and global resolver.
    """

    if len(sources) < 2:
        raise ValueError("Joint span training requires at least two proposal sources")
    if set(sources) != set(source_roles):
        raise ValueError("Joint span sources and source roles must match exactly")
    expected_documents = set(corpus.source_texts)
    normalized_sources: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for source_name, rows_by_document in sources.items():
        if set(rows_by_document) != expected_documents:
            raise ValueError(f"Proposal source {source_name!r} does not cover the full corpus")
        ProposalSourceRole(source_roles[source_name])
        normalized_sources[source_name] = {
            document_id: [dict(row) for row in rows]
            for document_id, rows in rows_by_document.items()
        }
    matrix = build_phase1_proposal_matrix(
        normalized_sources,
        corpus.source_texts,
        source_metadata={
            source_name: {"role": ProposalSourceRole(source_roles[source_name]).value}
            for source_name in sorted(source_roles)
        },
    )
    raw_matrix = matrix.get("matrix")
    if not isinstance(raw_matrix, list):
        raise ValueError("Proposal matrix did not emit a matrix list")
    matrix_rows_by_document: dict[str, list[dict[str, Any]]] = {
        document_id: [] for document_id in corpus.source_texts
    }
    for row in raw_matrix:
        if not isinstance(row, Mapping):
            raise ValueError("Proposal matrix row must be an object")
        document_id = str(row.get("document_id", ""))
        if document_id not in matrix_rows_by_document:
            raise ValueError("Proposal matrix references an unknown document")
        matrix_rows_by_document[document_id].append(dict(row))
    return {
        document_id: tuple(
            sorted(
                rows,
                key=lambda row: (
                    row["position"],
                    str(row["type"]),
                    str(row["proposal_id"]),
                ),
            )
        )
        for document_id, rows in matrix_rows_by_document.items()
    }
