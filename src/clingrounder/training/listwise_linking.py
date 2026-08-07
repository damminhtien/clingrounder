"""Model-neutral listwise records for candidate reranker training.

Independent cross-encoders cannot compare close ontology candidates directly. This
contract presents one mention, its context, and the complete bounded candidate set
as a single training example while retaining dictionary/type invariants.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from clingrounder.linking.candidate import Candidate
from clingrounder.linking.listwise import (
    ListwiseCandidateOption,
    ListwiseStructuredMention,
    build_listwise_linking_query,
    render_listwise_candidate,
)
from clingrounder.schema.types import EntityType
from clingrounder.terminology.ports import TerminologyRepository

__all__ = [
    "CandidateRecallError",
    "ListwiseCandidateOption",
    "ListwiseLinkingRecord",
    "PairwiseLinkingExample",
    "build_pairwise_linking_examples",
    "build_listwise_linking_record",
    "evaluate_listwise_scores",
    "render_listwise_input",
    "shuffle_listwise_candidates",
]


class CandidateRecallError(ValueError):
    """Raised when retrieval omitted every supervised positive candidate."""


@dataclass(frozen=True, slots=True)
class ListwiseLinkingRecord:
    """One mention-level listwise training or evaluation example."""

    query_id: str
    mention: str
    context: str
    entity_type: EntityType
    candidates: tuple[ListwiseCandidateOption, ...]
    positive_indices: tuple[int, ...]
    structured_mention: ListwiseStructuredMention = field(
        default_factory=ListwiseStructuredMention
    )

    def __post_init__(self) -> None:
        if not self.query_id.strip() or not self.mention.strip():
            raise ValueError("Listwise records require query_id and mention.")
        if len(self.candidates) < 2:
            raise ValueError("Listwise records require at least two candidates.")
        identities = [
            (candidate.code_system, candidate.code, candidate.concept_id)
            for candidate in self.candidates
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Listwise candidate identities must be unique.")
        if (
            not self.positive_indices
            or len(self.positive_indices) != len(set(self.positive_indices))
            or any(not 0 <= index < len(self.candidates) for index in self.positive_indices)
        ):
            raise ValueError("positive_indices must be unique valid candidate indices.")

    def to_json(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "mention": self.mention,
            "context": self.context,
            "entity_type": self.entity_type.value,
            "structured_mention": self.structured_mention.to_json(),
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "positive_indices": list(self.positive_indices),
        }


@dataclass(frozen=True, slots=True)
class PairwiseLinkingExample:
    """One XLM-R-compatible mention/candidate pair expanded from a listwise record."""

    query_id: str
    candidate_index: int
    query_text: str
    candidate_text: str
    label: int

    def to_json(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "candidate_index": self.candidate_index,
            "query_text": self.query_text,
            "candidate_text": self.candidate_text,
            "label": self.label,
        }


def build_listwise_linking_record(
    *,
    query_id: str,
    mention: str,
    context: str,
    entity_type: EntityType,
    candidates: Sequence[Candidate],
    positive_codes: Sequence[str],
    repository: TerminologyRepository | None = None,
) -> ListwiseLinkingRecord:
    """Convert constrained retrieval output into one positive-set record."""

    if not positive_codes:
        raise ValueError("positive_codes must be non-empty.")
    query = build_listwise_linking_query(
        query_id=query_id,
        mention=mention,
        context=context,
        entity_type=entity_type,
        candidates=candidates,
        repository=repository,
    )
    expected = set(positive_codes)
    positive_indices = tuple(
        index for index, candidate in enumerate(query.candidates) if candidate.code in expected
    )
    if not positive_indices:
        raise CandidateRecallError(
            "No positive code is present in the retrieved candidates; fix recall before reranking."
        )
    return ListwiseLinkingRecord(
        query_id=query_id,
        mention=mention,
        context=context,
        entity_type=entity_type,
        candidates=query.candidates,
        positive_indices=positive_indices,
        structured_mention=query.structured_mention,
    )


def build_pairwise_linking_examples(
    records: Sequence[ListwiseLinkingRecord],
) -> tuple[PairwiseLinkingExample, ...]:
    """Expand listwise supervision for a cheap sequence-classification baseline."""

    examples: list[PairwiseLinkingExample] = []
    for record in records:
        positives = set(record.positive_indices)
        query_text = (
            f"[TYPE] {record.entity_type.value}\n"
            f"[MENTION] {record.mention}\n[CONTEXT] {record.context}"
        )
        examples.extend(
            PairwiseLinkingExample(
                query_id=record.query_id,
                candidate_index=index,
                query_text=query_text,
                candidate_text=render_listwise_candidate(candidate),
                label=int(index in positives),
            )
            for index, candidate in enumerate(record.candidates)
        )
    return tuple(examples)


def shuffle_listwise_candidates(
    record: ListwiseLinkingRecord,
    *,
    seed: int,
) -> ListwiseLinkingRecord:
    """Deterministically randomize training order and remap positive indices."""

    derived_seed = int.from_bytes(
        hashlib.sha256(f"{seed}\0{record.query_id}".encode("utf-8")).digest()[:8],
        "big",
    )
    order = list(range(len(record.candidates)))
    random.Random(derived_seed).shuffle(order)
    old_positive = set(record.positive_indices)
    return replace(
        record,
        candidates=tuple(record.candidates[index] for index in order),
        positive_indices=tuple(
            new_index
            for new_index, old_index in enumerate(order)
            if old_index in old_positive
        ),
    )


def render_listwise_input(record: ListwiseLinkingRecord) -> str:
    """Render a language-neutral cross-candidate input with stable option IDs."""

    lines = [
        f"[TYPE] {record.entity_type.value}",
        f"[MENTION] {record.mention}",
        f"[CONTEXT] {record.context}",
        "[STRUCTURED_MENTION] "
        + json.dumps(record.structured_mention.to_json(), ensure_ascii=False, sort_keys=True),
        "[CANDIDATES]",
    ]
    for index, candidate in enumerate(record.candidates):
        alias = candidate.matched_alias or ""
        code = candidate.code or ""
        lines.append(
            f"[{index}] {candidate.code_system}|{code}|"
            f"{candidate.canonical_name}|{alias}"
        )
    return "\n".join(lines)


def evaluate_listwise_scores(
    records: Sequence[ListwiseLinkingRecord],
    scores: dict[str, Sequence[float]],
    *,
    recall_cutoffs: tuple[int, ...] = (1, 5, 10, 20),
    abstained_query_ids: Sequence[str] = (),
    order_consistency: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate positive-set rank metrics for any listwise model adapter."""

    if not records:
        raise ValueError("At least one listwise record is required.")
    ranks: list[int] = []
    rows: list[dict[str, Any]] = []
    abstained = set(abstained_query_ids)
    consistency = order_consistency or {}
    jaccards: list[float] = []
    caught_errors = 0
    abstention_count = 0
    for record in records:
        row_scores = scores.get(record.query_id)
        if row_scores is None or len(row_scores) != len(record.candidates):
            raise ValueError(f"Missing or invalid scores for {record.query_id}.")
        ranked_indices = sorted(
            range(len(row_scores)),
            key=lambda index: (-float(row_scores[index]), index),
        )
        positive = set(record.positive_indices)
        rank = next(
            rank
            for rank, candidate_index in enumerate(ranked_indices, start=1)
            if candidate_index in positive
        )
        ranks.append(rank)
        is_abstained = record.query_id in abstained
        emitted = set() if is_abstained else {ranked_indices[0]}
        union = emitted | positive
        jaccard = len(emitted & positive) / len(union) if union else 1.0
        jaccards.append(jaccard)
        if is_abstained:
            abstention_count += 1
            caught_errors += int(rank > 1)
        rows.append(
            {
                "query_id": record.query_id,
                "best_positive_rank": rank,
                "top_candidate_index": ranked_indices[0],
                "top1_correct": rank == 1,
                "abstained": is_abstained,
                "jaccard_after_emission": jaccard,
                "order_consistency": consistency.get(record.query_id),
            }
        )
    total = len(ranks)
    return {
        "schema_version": "listwise-linking-evaluation.v1",
        "query_count": total,
        "hit_at": {
            str(cutoff): sum(rank <= cutoff for rank in ranks) / total
            for cutoff in recall_cutoffs
        },
        "mrr": sum(1.0 / rank for rank in ranks) / total,
        "top1_accuracy": sum(rank == 1 for rank in ranks) / total,
        "jaccard_after_emission": sum(jaccards) / total,
        "order_consistency": (
            sum(consistency.values()) / len(consistency) if consistency else None
        ),
        "abstention_precision": (
            caught_errors / abstention_count if abstention_count else None
        ),
        "rows": rows,
    }


def record_to_jsonl(record: ListwiseLinkingRecord) -> str:
    """Serialize one record for framework-specific training adapters."""

    return json.dumps(record.to_json(), ensure_ascii=False, sort_keys=True)
