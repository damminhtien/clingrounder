"""Local generative adapter for position-robust listwise candidate reranking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING, Sequence

from medical_kg_nlp.adapters.generative.runtime import (
    ChatMessage,
    GenerationConfig,
    GenerativeModelPort,
)
from medical_kg_nlp.adapters.generative.structured import (
    StructuredResponseError,
    parse_structured_response,
)
from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.linking.listwise import (
    ListwiseCandidateOrder,
    ListwiseLinkingQuery,
    ListwiseOrderRanking,
    ListwiseRerankDecision,
    aggregate_listwise_rankings,
    build_listwise_candidate_orders,
    build_listwise_linking_query,
    render_listwise_candidate,
)
from medical_kg_nlp.terminology.ports import TerminologyRepository

if TYPE_CHECKING:
    from medical_kg_nlp.pipeline.ports import CandidateRerankerPort

__all__ = [
    "GenerativeListwiseRerankerAdapter",
    "LISTWISE_RERANKER_PROMPT_VERSION",
    "listwise_reranker_prompt_hash",
]

LISTWISE_RERANKER_PROMPT_VERSION = "medical-kg-listwise-reranker.v1"
_SYSTEM_PROMPT = """Bạn là bộ xếp hạng entity linking y khoa.
Chỉ xếp hạng các candidate đã cung cấp. Không tạo mã, tên hay candidate mới.
So sánh mention, context, loại entity, aliases, parent và structured attributes.
Trả về duy nhất một JSON object theo schema:
{"ranking":["B","A","C"],"abstain":false}
ranking phải chứa mỗi nhãn candidate đúng một lần. Dùng abstain=true khi evidence không đủ.
"""


class GenerativeListwiseRerankerAdapter:
    """Aggregate three order-controlled rankings from one pinned local chat model."""

    def __init__(
        self,
        runtime: GenerativeModelPort,
        repository: TerminologyRepository,
        *,
        generation: GenerationConfig,
        base_reranker: CandidateRerankerPort | None = None,
        candidate_limit: int = 12,
        model_weight: float = 0.75,
        shuffle_seed: int = 42,
        structured_retries: int = 1,
    ) -> None:
        if not 2 <= candidate_limit <= 12:
            raise ValueError("candidate_limit must be between 2 and 12")
        if not 0.0 <= model_weight <= 1.0:
            raise ValueError("model_weight must be between 0 and 1")
        if structured_retries < 0:
            raise ValueError("structured_retries cannot be negative")
        self.runtime = runtime
        self.repository = repository
        self.generation = generation
        self.base_reranker = base_reranker
        self.candidate_limit = candidate_limit
        self.model_weight = model_weight
        self.shuffle_seed = shuffle_seed
        self.structured_retries = structured_retries

    def rerank(
        self,
        candidates: list[Candidate],
        context_window: str = "",
        mention: str = "",
    ) -> list[Candidate]:
        """Rerank a bounded head and preserve retrieval order when the model abstains."""

        base = (
            list(candidates)
            if self.base_reranker is None
            else self.base_reranker.rerank(
                candidates,
                context_window=context_window,
                mention=mention,
            )
        )
        if len(base) < 2:
            return base
        bounded = base[: self.candidate_limit]
        query = build_listwise_linking_query(
            query_id=_query_id(mention, context_window, bounded),
            mention=mention,
            context=context_window,
            entity_type=bounded[0].semantic_type,
            candidates=bounded,
            repository=self.repository,
        )
        decision = self.rank_query(query)
        if decision.abstain:
            return base

        rescored = [
            replace(
                candidate,
                score=_bounded_probability(
                    (1.0 - self.model_weight) * candidate.score
                    + self.model_weight * decision.aggregate_scores[index]
                ),
            )
            for index, candidate in enumerate(bounded)
        ]
        # MODEL: the listwise stage owns only the bounded candidate head. Retrieval recall outside
        # this set must be fixed before promotion rather than hidden behind an unscored tail.
        return sorted(
            rescored,
            key=lambda candidate: (
                -candidate.score,
                candidate.code_system.value,
                candidate.code or "",
                candidate.concept_id,
            ),
        )

    def close(self) -> None:
        """Close the generative runtime and nested reranker when applicable."""

        close = getattr(self.runtime, "close", None)
        if callable(close):
            close()
        nested_close = getattr(self.base_reranker, "close", None)
        if callable(nested_close):
            nested_close()

    def rank_query(self, query: ListwiseLinkingQuery) -> ListwiseRerankDecision:
        """Run retrieval, reverse, and seeded-shuffle prompts for one query."""

        rankings = tuple(
            self._rank_order(query, order, order_index=index)
            for index, order in enumerate(
                build_listwise_candidate_orders(query, seed=self.shuffle_seed)
            )
        )
        return aggregate_listwise_rankings(query, rankings)

    def _rank_order(
        self,
        query: ListwiseLinkingQuery,
        order: ListwiseCandidateOrder,
        *,
        order_index: int,
    ) -> ListwiseOrderRanking:
        messages = _listwise_messages(query, order)
        active_messages = list(messages)
        raw_response = ""
        for attempt in range(self.structured_retries + 1):
            raw_response = self.runtime.generate(
                active_messages,
                replace(self.generation, seed=self.generation.seed + order_index),
            )
            try:
                return _parse_order_ranking(raw_response, order)
            except (StructuredResponseError, TypeError, ValueError) as error:
                if attempt >= self.structured_retries:
                    break
                active_messages.extend(
                    (
                        ChatMessage(role="assistant", content=raw_response or "{}"),
                        ChatMessage(
                            role="user",
                            content=(
                                "JSON không hợp lệ: "
                                f"{error}. Trả lại đúng schema và chỉ dùng nhãn đã cho."
                            ),
                        ),
                    )
                )
        # INVARIANT: malformed generation is an abstention. It cannot alter candidate identity.
        return ListwiseOrderRanking(
            order_id=order.order_id,
            ranked_candidate_indices=(),
            abstain=True,
            valid=False,
        )


def listwise_reranker_prompt_hash() -> str:
    """Fingerprint behavior-bearing prompt text without mention or terminology content."""

    return hashlib.sha256(
        json.dumps(
            {
                "version": LISTWISE_RERANKER_PROMPT_VERSION,
                "system": _SYSTEM_PROMPT,
                "orders": ["retrieval", "reverse", "shuffled"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _listwise_messages(
    query: ListwiseLinkingQuery,
    order: ListwiseCandidateOrder,
) -> tuple[ChatMessage, ...]:
    structured = json.dumps(
        query.structured_mention.to_json(),
        ensure_ascii=False,
        sort_keys=True,
    )
    lines = [
        f"[MENTION]\n{query.mention}",
        f"[CONTEXT]\n{query.context or '-'}",
        f"[ENTITY_TYPE]\n{query.entity_type.value}",
        f"[STRUCTURED_MENTION]\n{structured}",
        "[CANDIDATES]",
    ]
    for display_index, candidate_index in enumerate(order.candidate_indices):
        label = _option_label(display_index)
        lines.append(f"{label}. {render_listwise_candidate(query.candidates[candidate_index])}")
    return (
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user", content="\n".join(lines)),
    )


def _parse_order_ranking(
    raw_response: str,
    order: ListwiseCandidateOrder,
) -> ListwiseOrderRanking:
    value = parse_structured_response(raw_response)
    if not isinstance(value, dict):
        raise TypeError("Listwise output must be a JSON object")
    raw_ranking = value.get("ranking")
    abstain = value.get("abstain")
    if not isinstance(raw_ranking, list) or not all(
        isinstance(item, str) for item in raw_ranking
    ):
        raise TypeError("ranking must be an array of labels")
    if not isinstance(abstain, bool):
        raise TypeError("abstain must be boolean")
    expected_labels = tuple(_option_label(index) for index in range(len(order.candidate_indices)))
    ranking_labels = tuple(raw_ranking)
    if len(ranking_labels) != len(expected_labels) or set(ranking_labels) != set(
        expected_labels
    ):
        raise ValueError("ranking must contain every provided label exactly once")
    display_index_by_label = {label: index for index, label in enumerate(expected_labels)}
    ranked_indices = tuple(
        order.candidate_indices[display_index_by_label[label]] for label in ranking_labels
    )
    return ListwiseOrderRanking(
        order_id=order.order_id,
        ranked_candidate_indices=ranked_indices,
        abstain=abstain,
    )


def _query_id(mention: str, context: str, candidates: Sequence[Candidate]) -> str:
    payload = {
        "mention": mention,
        "context": context,
        "candidate_ids": [candidate.concept_id for candidate in candidates],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _option_label(index: int) -> str:
    if not 0 <= index < 26:
        raise ValueError("Listwise option index exceeds alphabetic label range")
    return chr(ord("A") + index)


def _bounded_probability(value: float) -> float:
    return min(1.0, max(0.0, value))
