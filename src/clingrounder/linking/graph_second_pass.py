"""Document-level second-pass linking with exact-unique graph anchors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from clingrounder.linking.candidate import Candidate
from clingrounder.linking.graph_evidence import GraphContextConcept, GraphEvidenceReranker
from clingrounder.schema.annotation import EntityAnnotation
from clingrounder.schema.document import Sentence
from clingrounder.schema.types import CodeSystem

__all__ = ["GraphEvidenceSecondPass"]


@dataclass(frozen=True)
class _Anchor:
    entity_id: str
    sentence_index: int
    concept: GraphContextConcept


class GraphEvidenceSecondPass:
    """Rerank document candidates using only exact-unique neighboring links."""

    def __init__(self, reranker: GraphEvidenceReranker) -> None:
        self.reranker = reranker

    def rerank_document(
        self,
        entities: list[EntityAnnotation],
        candidates_by_entity: Mapping[str, list[Candidate]],
        sentences: list[Sentence],
        mentions_by_entity: Mapping[str, str],
    ) -> tuple[dict[str, list[Candidate]], dict[str, int]]:
        """Return new rankings plus integer counters suitable for PipelineTrace."""

        sentence_by_entity = {
            entity.id: _containing_sentence(entity, sentences) for entity in entities
        }
        anchors = self._anchors(candidates_by_entity, sentence_by_entity)
        anchors_by_sentence: dict[int, list[_Anchor]] = {}
        for anchor in anchors:
            anchors_by_sentence.setdefault(anchor.sentence_index, []).append(anchor)

        output: dict[str, list[Candidate]] = {}
        queries_with_context = 0
        queries_with_graph_feature = 0
        changed_top1 = 0
        context_events = 0
        for entity in entities:
            candidates = candidates_by_entity.get(entity.id)
            if candidates is None:
                continue
            sentence_index = sentence_by_entity[entity.id]
            # INVARIANT: an unresolved sentence is not shared context. Grouping all orphan
            # spans under the sentinel index would leak evidence across unrelated text.
            context = (
                _context_for_target(
                    entity.id,
                    anchors_by_sentence.get(sentence_index, ()),
                )
                if sentence_index >= 0
                else ()
            )
            context_events += len(context)
            queries_with_context += int(bool(context))
            reranked = self.reranker.rerank(
                candidates,
                mention=mentions_by_entity.get(entity.id, entity.text),
                context_concepts=context,
            )
            queries_with_graph_feature += int(
                any("graph_reranker" in candidate.sources for candidate in reranked)
            )
            changed_top1 += int(_top_code(candidates) != _top_code(reranked))
            output[entity.id] = reranked
        return output, {
            "anchor_entities": len(anchors),
            "context_events": context_events,
            "queries_with_context": queries_with_context,
            "queries_with_graph_feature": queries_with_graph_feature,
            "changed_top1": changed_top1,
            "reranked_entities": len(output),
        }

    @staticmethod
    def _anchors(
        candidates_by_entity: Mapping[str, list[Candidate]],
        sentence_by_entity: dict[str, int],
    ) -> tuple[_Anchor, ...]:
        anchors: list[_Anchor] = []
        for entity_id, candidates in candidates_by_entity.items():
            sentence_index = sentence_by_entity.get(entity_id, -1)
            if sentence_index < 0:
                continue
            candidate = _exact_unique_candidate(candidates)
            if candidate is None or candidate.code is None:
                continue
            anchors.append(
                _Anchor(
                    entity_id=entity_id,
                    sentence_index=sentence_index,
                    concept=GraphContextConcept(candidate.code_system, candidate.code),
                )
            )
        return tuple(anchors)


def _exact_unique_candidate(candidates: list[Candidate]) -> Candidate | None:
    exact = [
        candidate
        for candidate in candidates
        if candidate.code is not None
        and candidate.code_system != CodeSystem.NONE
        and "exact" in candidate.sources
    ]
    keys = {(candidate.code_system, candidate.code) for candidate in exact}
    return exact[0] if len(keys) == 1 else None


def _containing_sentence(entity: EntityAnnotation, sentences: list[Sentence]) -> int:
    for index, sentence in enumerate(sentences):
        if sentence.span[0] <= entity.span[0] and entity.span[1] <= sentence.span[1]:
            return index
    return -1


def _context_for_target(
    target_entity_id: str,
    anchors: list[_Anchor] | tuple[_Anchor, ...],
) -> tuple[GraphContextConcept, ...]:
    unique = {
        (anchor.concept.code_system.value, anchor.concept.code): anchor.concept
        for anchor in anchors
        if anchor.entity_id != target_entity_id
    }
    return tuple(unique[key] for key in sorted(unique))


def _top_code(candidates: list[Candidate]) -> tuple[CodeSystem, str | None] | None:
    if not candidates:
        return None
    return candidates[0].code_system, candidates[0].code
