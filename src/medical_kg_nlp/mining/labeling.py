"""Composable proposal batching, policy enforcement, and multi-labeler consensus."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence

from medical_kg_nlp.mining.ports import ProposalLabelerPort
from medical_kg_nlp.mining.records import (
    AnnotationLayer,
    AnnotationProposal,
    ConceptLink,
    MinedDocument,
    ReviewStatus,
)

__all__ = [
    "BatchedProposalLabelerAdapter",
    "ConsensusProposalLabeler",
    "PolicyAwareProposalLabelerAdapter",
]


class BatchedProposalLabelerAdapter:
    """Bound model memory while preserving deterministic input and output order."""

    def __init__(self, delegate: ProposalLabelerPort, *, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.delegate = delegate
        self.batch_size = batch_size

    def propose(self, documents: Sequence[MinedDocument]) -> Iterable[AnnotationProposal]:
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            yield from self.delegate.propose(batch)


class PolicyAwareProposalLabelerAdapter:
    """Block documents that cannot be sent to the wrapped hosted labeler."""

    def __init__(
        self,
        delegate: ProposalLabelerPort,
        *,
        allow_document: Callable[[MinedDocument], bool],
    ) -> None:
        self.delegate = delegate
        self.allow_document = allow_document

    def propose(self, documents: Sequence[MinedDocument]) -> Iterable[AnnotationProposal]:
        blocked = sorted(
            document.document_id
            for document in documents
            if not self.allow_document(document)
        )
        if blocked:
            # PRIVACY: fail the whole batch instead of silently leaking a restricted subset.
            raise PermissionError(f"Labeler policy rejected documents: {', '.join(blocked)}")
        yield from self.delegate.propose(documents)


class ConsensusProposalLabeler:
    """Merge independent entity proposals while retaining disagreement for review."""

    def __init__(
        self,
        labelers: Sequence[ProposalLabelerPort],
        *,
        min_votes: int = 2,
        labeler_id: str = "consensus:v1",
    ) -> None:
        if not labelers:
            raise ValueError("Consensus requires at least one labeler")
        if not 1 <= min_votes <= len(labelers):
            raise ValueError("min_votes must be between one and the labeler count")
        self.labelers = tuple(labelers)
        self.min_votes = min_votes
        self.labeler_id = labeler_id

    def propose(self, documents: Sequence[MinedDocument]) -> Iterable[AnnotationProposal]:
        documents_by_id = {document.document_id: document for document in documents}
        grouped: dict[
            tuple[str, tuple[int, int], str, str], dict[str, AnnotationProposal]
        ] = defaultdict(dict)
        for source_index, labeler in enumerate(self.labelers):
            for proposal in labeler.propose(documents):
                document = documents_by_id.get(proposal.document_id)
                if document is None:
                    raise ValueError(
                        f"Labeler {source_index} returned unknown document {proposal.document_id!r}"
                    )
                proposal.validate_offsets(document)
                key = (proposal.document_id, proposal.span, proposal.text, proposal.entity_type)
                previous = grouped[key].get(proposal.labeler_id)
                if previous is None or proposal.confidence > previous.confidence:
                    grouped[key][proposal.labeler_id] = proposal

        for key in sorted(grouped):
            votes = tuple(grouped[key].values())
            vote_count = len(votes)
            document_id, span, text, entity_type = key
            assertions = _majority_assertions(votes, self.min_votes)
            concepts = _majority_concepts(votes, self.min_votes)
            identity = f"{document_id}\0{span}\0{text}\0{entity_type}\0{self.labeler_id}"
            consensus = vote_count >= self.min_votes
            yield AnnotationProposal(
                annotation_id=f"consensus:{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
                document_id=document_id,
                span=span,
                text=text,
                entity_type=entity_type,
                assertions=assertions,
                concepts=concepts,
                confidence=sum(value.confidence for value in votes) / vote_count,
                layer=AnnotationLayer.SILVER if consensus else AnnotationLayer.BRONZE,
                label_source="multi_labeler_consensus",
                labeler_id=self.labeler_id,
                review_status=(
                    ReviewStatus.PROPOSED if consensus else ReviewStatus.NEEDS_REVIEW
                ),
                source_label=_majority_optional_text(
                    [value.source_label for value in votes], self.min_votes
                ),
                metadata={
                    "vote_count": str(vote_count),
                    "source_labelers": ",".join(sorted(grouped[key])),
                    "consensus": str(consensus).lower(),
                },
            )


def _majority_assertions(
    proposals: Sequence[AnnotationProposal], threshold: int
) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for proposal in proposals:
        counts.update(set(proposal.assertions))
    return tuple(sorted(value for value, count in counts.items() if count >= threshold))


def _majority_concepts(
    proposals: Sequence[AnnotationProposal], threshold: int
) -> tuple[ConceptLink, ...]:
    grouped: dict[tuple[str, str, str], list[ConceptLink]] = defaultdict(list)
    for proposal in proposals:
        for concept in {
            (item.code_system, item.code, item.terminology_version): item
            for item in proposal.concepts
        }.values():
            grouped[(concept.code_system, concept.code, concept.terminology_version)].append(
                concept
            )
    result = []
    for key, values in sorted(grouped.items()):
        if len(values) >= threshold:
            result.append(
                ConceptLink(
                    code_system=key[0],
                    code=key[1],
                    terminology_version=key[2],
                    confidence=sum(value.confidence for value in values) / len(values),
                )
            )
    return tuple(result)


def _majority_optional_text(values: Sequence[str | None], threshold: int) -> str | None:
    counts = Counter(value for value in values if value)
    if not counts:
        return None
    value, count = counts.most_common(1)[0]
    return value if count >= threshold else None
