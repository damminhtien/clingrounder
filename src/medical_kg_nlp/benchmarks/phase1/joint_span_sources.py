"""Convert offset-safe model outputs into auditable joint span proposal sources."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from medical_kg_nlp.benchmarks.phase1.phase1_proposals import (
    build_phase1_proposal_matrix,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole
from medical_kg_nlp.benchmarks.phase1.reviewed_corpus import Phase1ReviewedCorpus
from medical_kg_nlp.benchmarks.phase1.split_contract import phase1_document_sort_key
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.ner.medication_list_parser import MedicationListParser
from medical_kg_nlp.ner.medication_mention_parser import MedicationMentionParser
from medical_kg_nlp.ner.rule_ner import RuleBasedNER
from medical_kg_nlp.ontology.phase1 import PHASE1_ALLOWED_TYPES
from medical_kg_nlp.ontology.phase1 import PHASE1_TYPE_BY_ENTITY_TYPE
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.types import EntityType

__all__ = [
    "EntityProposalExtractorPort",
    "build_phase1_joint_span_proposal_matrix",
    "build_phase1_medication_parser_source_rows",
    "build_phase1_rule_source_rows",
    "build_phase1_token_model_proposal_rows",
    "load_phase1_joint_span_source_rows",
]


class EntityProposalExtractorPort(Protocol):
    """Emit exact raw-text entity proposals without any benchmark output policy."""

    def extract(self, source_text: str) -> Sequence[EntityAnnotation]:
        """Return independently projected proposal entities for one source text."""


def build_phase1_rule_source_rows(
    corpus: Phase1ReviewedCorpus,
    dictionary: DictionaryStore,
    *,
    source_name: str = "rule",
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Materialize unresolved RuleNER evidence in the common source-row contract.

    Each candidate type is retained independently. This prevents the historical disease fallback
    from hiding a disease/symptom conflict before the joint verifier can classify it.
    """

    if not source_name.strip():
        raise ValueError("Rule proposal source name must be non-empty")
    ner = RuleBasedNER(dictionary, disease_symptom_fallback="abstain")
    rows_by_document: dict[str, tuple[dict[str, Any], ...]] = {}
    for document_id in sorted(corpus.source_texts, key=phase1_document_sort_key):
        source_text = corpus.source_texts[document_id]
        rows: list[dict[str, Any]] = []
        for proposal_index, proposal in enumerate(ner.extract_with_trace(source_text).trace.proposals):
            proposal.validate_offsets(source_text)
            start, end = proposal.span
            for entity_type in proposal.candidate_types:
                phase1_type = PHASE1_TYPE_BY_ENTITY_TYPE.get(entity_type)
                if phase1_type is None:
                    continue
                rows.append(
                    {
                        "text": source_text[start:end],
                        "type": phase1_type,
                        "position": [start, end],
                        "confidence": proposal.score,
                        "source_label": proposal.source,
                        "proposal_id": (
                            f"{document_id}:{source_name}:{proposal_index}:{start}:{end}:"
                            f"{phase1_type}"
                        ),
                    }
                )
        rows_by_document[document_id] = tuple(
            sorted(rows, key=lambda row: (row["position"], row["type"], row["proposal_id"]))
        )
    return rows_by_document


def build_phase1_medication_parser_source_rows(
    corpus: Phase1ReviewedCorpus,
    dictionary: DictionaryStore,
    *,
    source_name: str = "medication_parser",
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Materialize structured full-SIG spans from independent medication parsing.

    The parser does not discover a drug name. It uses the RuleNER drug recognition proposal as an
    anchor, then emits only a longer contiguous medication span. Keeping this as a separate source
    lets the joint verifier learn when exact medication structure is useful without silently
    widening every drug in the final resolver.

    INVARIANT: each emitted full span is clamped to the parsed list item's medication region and
    is sliced from the untouched source text.
    """

    if not source_name.strip():
        raise ValueError("Medication parser source name must be non-empty")
    ner = RuleBasedNER(dictionary, disease_symptom_fallback="abstain")
    mention_parser = MedicationMentionParser()
    list_parser = MedicationListParser()
    rows_by_document: dict[str, tuple[dict[str, Any], ...]] = {}
    for document_id in sorted(corpus.source_texts, key=phase1_document_sort_key):
        source_text = corpus.source_texts[document_id]
        list_items = list_parser.items(source_text)
        rows: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for proposal_index, proposal in enumerate(ner.extract_with_trace(source_text).trace.proposals):
            if EntityType.DRUG not in proposal.candidate_types:
                continue
            proposal.validate_offsets(source_text)
            base_span = proposal.span
            full_span = mention_parser.parse(source_text, base_span).full_span
            containing_item = next(
                (
                    item
                    for item in list_items
                    if item.medication_span[0] <= base_span[0]
                    and base_span[1] <= item.medication_span[1]
                ),
                None,
            )
            if containing_item is not None:
                full_span = (full_span[0], min(full_span[1], containing_item.medication_span[1]))
            if full_span == base_span or full_span in seen:
                continue
            start, end = full_span
            if start != base_span[0] or end <= start or end > len(source_text):
                raise ValueError("Medication parser emitted an invalid full span")
            seen.add(full_span)
            rows.append(
                {
                    "text": source_text[start:end],
                    "type": "THUỐC",
                    "position": [start, end],
                    "confidence": min(0.99, proposal.score + 0.04),
                    "source_label": "medication_full_span",
                    "proposal_id": (
                        f"{document_id}:{source_name}:{proposal_index}:{start}:{end}:THUỐC"
                    ),
                }
            )
        rows_by_document[document_id] = tuple(
            sorted(rows, key=lambda row: (row["position"], row["proposal_id"]))
        )
    return rows_by_document


def load_phase1_joint_span_source_rows(
    source_dir: str | Path,
    corpus: Phase1ReviewedCorpus,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Load a persisted source by explicit corpus ID, including governed prefixed IDs.

    Phase 1 submission loaders only accept numeric filenames. Joint training also consumes the
    owner-authorized source IDs, so this loader validates a named source directory directly.
    """

    root = Path(source_dir)
    document_root = root / "consensus" if (root / "consensus").is_dir() else root
    if not document_root.is_dir():
        raise FileNotFoundError(document_root)
    rows_by_document: dict[str, tuple[dict[str, Any], ...]] = {}
    for document_id in sorted(corpus.source_texts, key=phase1_document_sort_key):
        path = document_root / f"{document_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Joint span source is missing {path.name}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or any(not isinstance(row, Mapping) for row in raw):
            raise ValueError(f"Joint span source must contain an entity list: {path}")
        source_text = corpus.source_texts[document_id]
        rows: list[dict[str, Any]] = []
        for index, raw_row in enumerate(raw):
            row = dict(raw_row)
            _validate_source_row(row, source_text, path, index)
            rows.append(row)
        rows_by_document[document_id] = tuple(
            sorted(rows, key=lambda row: (row["position"], row["type"], str(row.get("source_label", ""))))
        )
    expected_names = {f"{document_id}.json" for document_id in corpus.source_texts}
    unexpected = {
        path.name
        for path in document_root.glob("*.json")
        if path.name != "manifest.json" and path.name not in expected_names
    }
    if unexpected:
        raise ValueError(f"Joint span source has unexpected documents: {sorted(unexpected)}")
    return rows_by_document


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


def _validate_source_row(
    row: Mapping[str, Any],
    source_text: str,
    path: Path,
    index: int,
) -> None:
    """Reject invalid persisted source evidence before source fusion can inspect it."""

    text = row.get("text")
    entity_type = row.get("type")
    position = row.get("position")
    if not isinstance(text, str) or not text or entity_type not in PHASE1_ALLOWED_TYPES:
        raise ValueError(f"{path}:{index}: joint span source has invalid text/type")
    if (
        not isinstance(position, list)
        or len(position) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in position)
    ):
        raise ValueError(f"{path}:{index}: joint span source has invalid position")
    start, end = position
    # INVARIANT: persisted model evidence is never re-aligned or normalized during loading.
    if start < 0 or end <= start or end > len(source_text) or source_text[start:end] != text:
        raise ValueError(f"{path}:{index}: joint span source violates raw offsets")
