"""Coverage-cube measurement and review prioritization."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from clingrounder.mining.records import (
    AnnotationLayer,
    AnnotationProposal,
    CoverageCell,
    CoverageReport,
    MinedDocument,
    ReviewStatus,
)

__all__ = ["CoverageCubePlanner", "CoverageTarget", "ReviewPriority"]


@dataclass(frozen=True)
class CoverageTarget:
    """Minimum support requested for one exact coverage-cube slice."""

    dimensions: tuple[tuple[str, str], ...]
    target: int

    def __post_init__(self) -> None:
        if self.target < 0:
            raise ValueError("Coverage target must be non-negative")
        if not self.dimensions:
            raise ValueError("Coverage target requires at least one dimension")
        if any(
            not isinstance(dimension, tuple)
            or len(dimension) != 2
            or not all(isinstance(value, str) and value for value in dimension)
            for dimension in self.dimensions
        ):
            raise ValueError("Coverage dimensions must be non-empty (key, value) pairs")


@dataclass(frozen=True)
class ReviewPriority:
    """Auditable components of the fixed review-priority formula."""

    document_id: str
    score: float
    coverage_gap: float
    model_disagreement: float
    novelty: float
    relation_density: float
    source_quality: float


class CoverageCubePlanner:
    """Measure configured slices and rank documents under a constrained review budget."""

    def __init__(self, targets: Sequence[CoverageTarget]) -> None:
        self.targets = tuple(targets)

    def report(
        self,
        snapshot_id: str,
        documents: Sequence[MinedDocument],
        annotations: Sequence[AnnotationProposal],
    ) -> CoverageReport:
        documents_by_id = {document.document_id: document for document in documents}
        cells: list[CoverageCell] = []
        for target in self.targets:
            matching = [
                annotation
                for annotation in annotations
                if (document := documents_by_id.get(annotation.document_id)) is not None
                and _matches(target.dimensions, document, annotation)
            ]
            cells.append(
                CoverageCell(
                    dimensions=target.dimensions,
                    observed=len(matching),
                    target=target.target,
                    human_reviewed=sum(
                        annotation.review_status is ReviewStatus.ACCEPTED
                        and annotation.layer in {AnnotationLayer.GOLD, AnnotationLayer.CHALLENGE}
                        for annotation in matching
                    ),
                    synthetic=sum(
                        annotation.metadata.get("origin") == "synthetic"
                        for annotation in matching
                    ),
                )
            )
        return CoverageReport(snapshot_id=snapshot_id, cells=tuple(cells))

    def rank(
        self,
        documents: Sequence[MinedDocument],
        annotations: Sequence[AnnotationProposal],
    ) -> Sequence[str]:
        return [item.document_id for item in self.priorities(documents, annotations)]

    def priorities(
        self,
        documents: Sequence[MinedDocument],
        annotations: Sequence[AnnotationProposal],
    ) -> tuple[ReviewPriority, ...]:
        annotations_by_document: dict[str, list[AnnotationProposal]] = defaultdict(list)
        mention_document_frequency: Counter[tuple[str, str]] = Counter()
        for annotation in annotations:
            annotations_by_document[annotation.document_id].append(annotation)
        for values in annotations_by_document.values():
            mention_document_frequency.update(
                set((value.text.casefold(), value.entity_type) for value in values)
            )

        report = self.report("ranking", documents, annotations)
        gap_by_dimensions = {cell.dimensions: cell.gap_ratio for cell in report.cells}
        priorities: list[ReviewPriority] = []
        for document in documents:
            values = annotations_by_document.get(document.document_id, [])
            target_gaps = [
                gap
                for dimensions, gap in gap_by_dimensions.items()
                if any(_matches(dimensions, document, annotation) for annotation in values)
                or _matches_document(dimensions, document)
            ]
            coverage_gap = _mean(target_gaps)
            model_disagreement = _model_disagreement(values)
            novelty = _mean(
                [
                    1.0 / mention_document_frequency[(value.text.casefold(), value.entity_type)]
                    for value in values
                ]
            )
            relation_density = _bounded_metadata_float(document.metadata, "relation_density")
            source_quality = _bounded_metadata_float(
                document.metadata, "source_quality", default=0.5
            )
            score = (
                0.30 * coverage_gap
                + 0.25 * model_disagreement
                + 0.20 * novelty
                + 0.15 * relation_density
                + 0.10 * source_quality
            )
            priorities.append(
                ReviewPriority(
                    document_id=document.document_id,
                    score=score,
                    coverage_gap=coverage_gap,
                    model_disagreement=model_disagreement,
                    novelty=novelty,
                    relation_density=relation_density,
                    source_quality=source_quality,
                )
            )
        return tuple(sorted(priorities, key=lambda item: (-item.score, item.document_id)))


def _matches(
    dimensions: tuple[tuple[str, str], ...],
    document: MinedDocument,
    annotation: AnnotationProposal,
) -> bool:
    values = _dimension_values(document, annotation)
    return all(values.get(key) == expected for key, expected in dimensions)


def _matches_document(
    dimensions: tuple[tuple[str, str], ...], document: MinedDocument
) -> bool:
    values = _dimension_values(document, None)
    return all(values.get(key) == expected for key, expected in dimensions)


def _dimension_values(
    document: MinedDocument,
    annotation: AnnotationProposal | None,
) -> dict[str, str]:
    values = {
        "language": document.language,
        "note_type": document.note_type,
        "specialty": document.metadata.get("specialty", "unknown"),
        "noise": document.metadata.get("noise", "clean"),
        "formatting": document.metadata.get("formatting", "plain"),
        "source": document.metadata.get("source_id", document.source_artifact_id.split(":", 1)[0]),
    }
    if annotation is not None:
        values["entity_type"] = annotation.entity_type
        values["assertion"] = "+".join(annotation.assertions) or "present"
        values["code_system"] = (
            annotation.concepts[0].code_system if annotation.concepts else "unlinked"
        )
        values["terminology_chapter"] = (
            annotation.concepts[0].code[:1] if annotation.concepts else "unlinked"
        )
    return values


def _model_disagreement(values: Sequence[AnnotationProposal]) -> float:
    grouped: dict[tuple[int, int], set[tuple[str, str]]] = defaultdict(set)
    labelers: set[str] = set()
    for value in values:
        grouped[value.span].add((value.entity_type, value.text))
        labelers.add(value.labeler_id)
    if not grouped:
        return 0.0
    conflicts = sum(len(proposals) > 1 for proposals in grouped.values())
    single_source = sum(
        value.metadata.get("vote_count") == "1" or value.review_status is ReviewStatus.NEEDS_REVIEW
        for value in values
    )
    if len(labelers) <= 1 and single_source == 0:
        return 0.0
    return min(1.0, (conflicts + single_source) / max(len(grouped), 1))


def _bounded_metadata_float(
    metadata: Mapping[str, str], key: str, *, default: float = 0.0
) -> float:
    try:
        value = float(metadata.get(key, default))
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, value))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
