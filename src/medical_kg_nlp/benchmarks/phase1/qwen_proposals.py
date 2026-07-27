"""Qwen proposal, projection, consensus, and adjudication for Phase 1.

The model never supplies offsets. It returns exact source quotes and one of the five task labels;
this module projects the quote back to immutable raw text and records the evidence as
``EntityProposal`` objects.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from medical_kg_nlp.adapters.generative import (
    ChatMessage,
    GenerationConfig,
    GenerativeModelPort,
    StructuredResponseError,
    parse_structured_response,
)
from medical_kg_nlp.ner.proposal import EntityProposal
from medical_kg_nlp.schema.types import EntityType

__all__ = [
    "PHASE1_QWEN_PROMPT_VERSION",
    "Phase1AdjudicationCandidate",
    "Phase1AdjudicationDecision",
    "Phase1QwenAdapter",
    "Phase1QwenPassResult",
    "Phase1QuotedProposal",
    "Phase1ReviewEntity",
    "RawTextWindow",
    "apply_phase1_adjudication",
    "build_phase1_qwen_adjudication_messages",
    "build_phase1_qwen_extraction_messages",
    "build_phase1_qwen_review_messages",
    "phase1_qwen_prompt_hash",
    "project_phase1_quoted_proposals",
    "select_qwen_confirmed_proposals",
    "split_raw_text_windows",
]

PHASE1_QWEN_PROMPT_VERSION = "phase1-qwen-extraction.v2"
PHASE1_QWEN_REVIEW_PROMPT_VERSION = "phase1-qwen-review-missing.v2"
Phase1Label = Literal[
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
]
AdjudicationAction = Literal["KEEP", "DROP", "REPLACE"]
_StructuredT = TypeVar("_StructuredT")

_LABEL_TO_ENTITY_TYPE: dict[str, EntityType] = {
    "TRIỆU_CHỨNG": EntityType.SYMPTOM,
    "TÊN_XÉT_NGHIỆM": EntityType.LAB_TEST,
    "KẾT_QUẢ_XÉT_NGHIỆM": EntityType.LAB_RESULT,
    "CHẨN_ĐOÁN": EntityType.DISEASE,
    "THUỐC": EntityType.DRUG,
}
_ENTITY_TYPE_TO_LABEL = {value: key for key, value in _LABEL_TO_ENTITY_TYPE.items()}
_MODEL_PROPOSAL_CONFIDENCE = 0.90
_MAX_CONTEXT_CHARACTERS = 120
_SYSTEM_PROMPT = """Bạn là bộ gán nhãn thực thể y khoa tiếng Việt.
Chỉ dùng đúng 5 nhãn: TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC.
Mỗi text phải là chuỗi trích nguyên văn, liên tục trong SOURCE. Không tự tính offset.
Không tự chấm confidence; pipeline sẽ hiệu chỉnh theo agreement giữa các pass.
THUỐC giữ full span nếu SOURCE có strength, dạng, route hoặc frequency đi cùng thuốc.
Chỉ lấy kết quả định lượng/định tính khi có xét nghiệm hay dấu hiệu sinh tồn làm anchor.
Không biến liều thuốc, route, frequency, ngày tháng hay số hành chính thành kết quả xét nghiệm.
Không thêm giải thích ngoài JSON theo schema được yêu cầu."""
_REVIEW_SYSTEM_PROMPT = """Bạn là chuyên gia review gán nhãn thực thể y khoa tiếng Việt.
SOURCE đã có một danh sách EXISTING_ENTITIES. Chỉ tìm thực thể y khoa CÒN THIẾU trong SOURCE.
Không trả lại entity đã có cùng text và type.
Chỉ dùng đúng 5 nhãn: TRIỆU_CHỨNG, TÊN_XÉT_NGHIỆM, KẾT_QUẢ_XÉT_NGHIỆM, CHẨN_ĐOÁN, THUỐC.
Mỗi text phải là chuỗi trích nguyên văn, liên tục trong SOURCE. Không tự tính offset.
Không tự chấm confidence; pipeline sẽ hiệu chỉnh theo agreement giữa các pass.
THUỐC giữ full span nếu SOURCE có strength, dạng, route hoặc frequency đi cùng thuốc.
Không biến liều thuốc, route, frequency, ngày tháng hay số hành chính thành kết quả xét nghiệm.
Không thêm giải thích ngoài JSON theo schema được yêu cầu."""


@dataclass(frozen=True, slots=True)
class RawTextWindow:
    """One overlapping model window expressed in source-text coordinates."""

    span: tuple[int, int]
    text: str

    def validate(self, source_text: str) -> None:
        start, end = self.span
        if start < 0 or end <= start or end > len(source_text):
            raise ValueError(f"Invalid raw-text window span: {self.span}")
        if source_text[start:end] != self.text:
            raise ValueError("Raw-text window no longer matches the immutable source")


@dataclass(frozen=True, slots=True)
class Phase1QuotedProposal:
    """Validated model proposal before local offset projection."""

    text: str
    entity_type: Phase1Label
    confidence: float
    left_context: str = ""
    right_context: str = ""

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("Quoted proposal text must be non-empty")
        if self.entity_type not in _LABEL_TO_ENTITY_TYPE:
            raise ValueError(f"Unsupported Phase 1 label: {self.entity_type}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Proposal confidence must be between zero and one")
        if (
            len(self.left_context) > _MAX_CONTEXT_CHARACTERS
            or len(self.right_context) > _MAX_CONTEXT_CHARACTERS
        ):
            raise ValueError("Proposal context anchors must not exceed 120 characters")


@dataclass(frozen=True, slots=True)
class Phase1ReviewEntity:
    """One existing label shown to the missing-entity reviewer."""

    text: str
    entity_type: Phase1Label
    span: tuple[int, int]

    def validate(self, source_text: str) -> None:
        """Reject stale or normalized coordinates before they enter a model prompt."""

        start, end = self.span
        if self.entity_type not in _LABEL_TO_ENTITY_TYPE:
            raise ValueError(f"Unsupported Phase 1 review label: {self.entity_type}")
        if start < 0 or end <= start or end > len(source_text):
            raise ValueError(f"Invalid Phase 1 review span: {self.span}")
        if source_text[start:end] != self.text:
            raise ValueError("Review entity no longer matches immutable source text")


@dataclass(frozen=True, slots=True)
class Phase1QwenPassResult:
    """Auditable output from one extraction pass over one document."""

    pass_id: str
    prompt_hash: str
    proposals: tuple[EntityProposal, ...]
    rejected: tuple[dict[str, Any], ...]
    response_sha256: tuple[str, ...]
    raw_responses: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Phase1AdjudicationCandidate:
    """One locally identified candidate shown to Qwen for adjudication."""

    proposal_id: str
    text: str
    entity_type: Phase1Label
    span: tuple[int, int]
    sources: tuple[str, ...]
    confidence: float

    def __post_init__(self) -> None:
        if not self.proposal_id.strip() or not self.text:
            raise ValueError("Adjudication candidates require proposal_id and text")
        if self.entity_type not in _LABEL_TO_ENTITY_TYPE:
            raise ValueError("Adjudication candidate has an invalid Phase 1 type")
        start, end = self.span
        if start < 0 or end <= start:
            raise ValueError("Adjudication candidate requires a non-empty span")
        if self.sources != tuple(sorted(set(self.sources))) or not self.sources:
            raise ValueError("Adjudication sources must be non-empty, unique, and sorted")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Adjudication confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class Phase1AdjudicationDecision:
    """Structured Qwen decision whose replacement remains subject to local projection."""

    proposal_id: str
    action: AdjudicationAction
    confidence: float
    evidence_quote: str
    replacement_text: str | None = None
    replacement_type: Phase1Label | None = None

    def __post_init__(self) -> None:
        if not self.proposal_id.strip() or not self.evidence_quote:
            raise ValueError("Adjudication decisions require proposal_id and evidence_quote")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Adjudication confidence must be between zero and one")
        has_replacement = self.replacement_text is not None or self.replacement_type is not None
        if self.action == "REPLACE":
            if not self.replacement_text or self.replacement_type not in _LABEL_TO_ENTITY_TYPE:
                raise ValueError("REPLACE requires replacement_text and replacement_type")
        elif has_replacement:
            raise ValueError("KEEP and DROP must not include replacement fields")


class Phase1QwenAdapter:
    """Run reusable Qwen passes while keeping Phase 1 projection deterministic."""

    def __init__(
        self,
        runtime: GenerativeModelPort,
        *,
        max_window_characters: int = 12_000,
        window_overlap_characters: int = 800,
        structured_retries: int = 1,
    ) -> None:
        if structured_retries < 0:
            raise ValueError("structured_retries cannot be negative")
        self._runtime = runtime
        self._max_window_characters = max_window_characters
        self._window_overlap_characters = window_overlap_characters
        self._structured_retries = structured_retries

    def extract(
        self,
        source_text: str,
        *,
        pass_id: str,
        target_types: Sequence[Phase1Label],
        generation: GenerationConfig,
    ) -> Phase1QwenPassResult:
        """Run one recall/targeted pass and project every exact occurrence."""

        if not pass_id.strip():
            raise ValueError("pass_id must be non-empty")
        normalized_types = tuple(sorted(set(target_types)))
        if not normalized_types or any(item not in _LABEL_TO_ENTITY_TYPE for item in normalized_types):
            raise ValueError("target_types must contain supported Phase 1 labels")
        prompt_hash = phase1_qwen_prompt_hash(
            pass_id=pass_id,
            target_types=normalized_types,
        )
        proposals: list[EntityProposal] = []
        rejected: list[dict[str, Any]] = []
        response_hashes: list[str] = []
        raw_responses: list[str] = []
        for window_index, window in enumerate(
            split_raw_text_windows(
                source_text,
                max_characters=self._max_window_characters,
                overlap_characters=self._window_overlap_characters,
            )
        ):
            messages = build_phase1_qwen_extraction_messages(
                window.text,
                pass_id=pass_id,
                target_types=normalized_types,
            )
            (quoted, parse_rejections), raw_response = self._generate_structured(
                messages,
                generation,
                parser=_parse_quoted_proposals,
            )
            response_hashes.append(_text_sha256(raw_response))
            raw_responses.append(raw_response)
            projected, projection_rejections = project_phase1_quoted_proposals(
                window.text,
                quoted,
                source=f"qwen.{pass_id}",
                evidence_id=f"{pass_id}.window-{window_index}",
                source_offset=window.span[0],
            )
            proposals.extend(projected)
            rejected.extend(
                {
                    **row,
                    "pass_id": pass_id,
                    "window_index": window_index,
                }
                for row in (*parse_rejections, *projection_rejections)
            )
        return Phase1QwenPassResult(
            pass_id=pass_id,
            prompt_hash=prompt_hash,
            proposals=_deduplicate_entity_proposals(proposals),
            rejected=tuple(rejected),
            response_sha256=tuple(response_hashes),
            raw_responses=tuple(raw_responses),
        )

    def review_missing(
        self,
        source_text: str,
        existing_entities: Sequence[Phase1ReviewEntity],
        *,
        generation: GenerationConfig,
        max_rounds: int = 2,
    ) -> Phase1QwenPassResult:
        """Iteratively extract only labels absent from an existing projection.

        MODEL: repeated review is useful only when every round is projected locally. The model
        never supplies offsets, and a repeated quote is expanded to every still-unlabeled raw
        occurrence rather than the first ``str.index`` match.
        """

        if not 1 <= max_rounds <= 5:
            raise ValueError("Missing-entity review rounds must be between one and five")
        for entity in existing_entities:
            entity.validate(source_text)
        known = {
            (*entity.span, _LABEL_TO_ENTITY_TYPE[entity.entity_type])
            for entity in existing_entities
        }
        cumulative: list[EntityProposal] = []
        rejected: list[dict[str, Any]] = []
        response_hashes: list[str] = []
        raw_responses: list[str] = []

        for round_index in range(max_rounds):
            added_this_round = 0
            for window_index, window in enumerate(
                split_raw_text_windows(
                    source_text,
                    max_characters=self._max_window_characters,
                    overlap_characters=self._window_overlap_characters,
                )
            ):
                visible_existing = _review_entities_for_window(
                    source_text,
                    window,
                    existing_entities,
                    cumulative,
                )
                messages = build_phase1_qwen_review_messages(
                    window.text,
                    visible_existing,
                    round_index=round_index,
                )
                (quoted, parse_rejections), raw_response = self._generate_structured(
                    messages,
                    generation,
                    parser=_parse_quoted_proposals,
                )
                response_hashes.append(_text_sha256(raw_response))
                raw_responses.append(raw_response)
                projected, projection_rejections = project_phase1_quoted_proposals(
                    window.text,
                    quoted,
                    source="qwen.review-missing",
                    evidence_id=f"review.round-{round_index}.window-{window_index}",
                    source_offset=window.span[0],
                )
                for proposal in projected:
                    entity_type = proposal.entity_type
                    if entity_type is None:
                        continue
                    identity = (*proposal.span, entity_type)
                    if identity in known:
                        continue
                    known.add(identity)
                    cumulative.append(proposal)
                    added_this_round += 1
                rejected.extend(
                    {
                        **row,
                        "pass_id": "review-missing",
                        "round_index": round_index,
                        "window_index": window_index,
                    }
                    for row in (*parse_rejections, *projection_rejections)
                )
            if added_this_round == 0:
                break

        prompt_hash = _text_sha256(
            json.dumps(
                {
                    "prompt_version": PHASE1_QWEN_REVIEW_PROMPT_VERSION,
                    "system": _REVIEW_SYSTEM_PROMPT,
                    "max_rounds": max_rounds,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return Phase1QwenPassResult(
            pass_id="review-missing",
            prompt_hash=prompt_hash,
            proposals=_deduplicate_entity_proposals(cumulative),
            rejected=tuple(rejected),
            response_sha256=tuple(response_hashes),
            raw_responses=tuple(raw_responses),
        )

    def adjudicate(
        self,
        source_text: str,
        candidates: Sequence[Phase1AdjudicationCandidate],
        *,
        generation: GenerationConfig,
    ) -> tuple[tuple[Phase1AdjudicationDecision, ...], str]:
        """Ask Qwen to keep, drop, or replace already projected candidates."""

        for candidate in candidates:
            start, end = candidate.span
            if source_text[start:end] != candidate.text:
                raise ValueError(
                    f"Adjudication candidate {candidate.proposal_id} violates raw offsets"
                )
        messages = build_phase1_qwen_adjudication_messages(source_text, candidates)
        decisions, raw_response = self._generate_structured(
            messages,
            generation,
            parser=_parse_adjudication_decisions,
        )
        known_ids = {candidate.proposal_id for candidate in candidates}
        if any(decision.proposal_id not in known_ids for decision in decisions):
            raise ValueError("Adjudication response references an unknown proposal_id")
        return decisions, _text_sha256(raw_response)

    def _generate_structured(
        self,
        messages: Sequence[ChatMessage],
        generation: GenerationConfig,
        *,
        parser: Callable[[Any], _StructuredT],
    ) -> tuple[_StructuredT, str]:
        """Generate and validate one task schema inside the bounded retry loop."""

        active_messages = list(messages)
        last_error: Exception | None = None
        for attempt in range(self._structured_retries + 1):
            raw_response = self._runtime.generate(active_messages, generation)
            try:
                parsed = parse_structured_response(raw_response)
                return parser(parsed), raw_response
            except (StructuredResponseError, TypeError, ValueError) as error:
                last_error = error
                if attempt >= self._structured_retries:
                    break
                active_messages.extend(
                    (
                        ChatMessage(role="assistant", content=raw_response or "{}"),
                        ChatMessage(
                            role="user",
                            content=(
                                "Output trước không đúng JSON schema. Lỗi validator: "
                                f"{error}. Hãy trả lại duy nhất JSON đúng schema, không giải thích."
                            ),
                        ),
                    )
                )
        raise StructuredResponseError(
            f"Structured generation failed after retries: {last_error}"
        )


def phase1_qwen_prompt_hash(
    *,
    pass_id: str,
    target_types: Sequence[Phase1Label],
) -> str:
    """Hash all behavior-bearing prompt components without including private source text."""

    payload = {
        "version": PHASE1_QWEN_PROMPT_VERSION,
        "system": _SYSTEM_PROMPT,
        "pass_id": pass_id,
        "target_types": sorted(set(target_types)),
    }
    return _text_sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def split_raw_text_windows(
    source_text: str,
    *,
    max_characters: int,
    overlap_characters: int,
) -> tuple[RawTextWindow, ...]:
    """Split long text at a nearby line/space while preserving raw coordinates."""

    if max_characters < 256:
        raise ValueError("max_characters must be at least 256")
    if overlap_characters < 0 or overlap_characters >= max_characters:
        raise ValueError("overlap_characters must be in [0, max_characters)")
    if not source_text:
        return ()
    windows: list[RawTextWindow] = []
    start = 0
    while start < len(source_text):
        proposed_end = min(len(source_text), start + max_characters)
        end = proposed_end
        if proposed_end < len(source_text):
            search_start = max(start + max_characters // 2, proposed_end - 800)
            newline = source_text.rfind("\n", search_start, proposed_end)
            space = source_text.rfind(" ", search_start, proposed_end)
            boundary = max(newline, space)
            if boundary > start:
                end = boundary + (1 if source_text[boundary] == "\n" else 0)
        window = RawTextWindow(span=(start, end), text=source_text[start:end])
        window.validate(source_text)
        windows.append(window)
        if end == len(source_text):
            break
        next_start = max(start + 1, end - overlap_characters)
        start = next_start
    return tuple(windows)


def project_phase1_quoted_proposals(
    source_text: str,
    proposals: Sequence[Phase1QuotedProposal],
    *,
    source: str,
    evidence_id: str,
    source_offset: int = 0,
) -> tuple[list[EntityProposal], list[dict[str, Any]]]:
    """Project quotes to all matching occurrences and preserve optional context anchors."""

    projected: list[EntityProposal] = []
    rejected: list[dict[str, Any]] = []
    for index, proposal in enumerate(proposals):
        exact_occurrences = _exact_occurrences(source_text, proposal.text)
        occurrences = exact_occurrences
        if proposal.left_context:
            occurrences = [
                start
                for start in occurrences
                if source_text[max(0, start - len(proposal.left_context)) : start].endswith(
                    proposal.left_context
                )
            ]
        if proposal.right_context:
            occurrences = [
                start
                for start in occurrences
                if source_text[start + len(proposal.text) :].startswith(proposal.right_context)
            ]
        context_projection = "matched"
        if not occurrences and exact_occurrences:
            # MODEL: exact quotation is the offset authority. Context anchors are only optional
            # disambiguation hints, and local chat checkpoints sometimes copy a whole clause or
            # invent spacing around an otherwise exact mention. Agreement gates still prevent a
            # single recovered quote from entering final output.
            occurrences = exact_occurrences
            context_projection = "fallback_to_all_exact_occurrences"
        if not occurrences:
            rejected.append(
                {
                    "proposal_index": index,
                    "reason": "quote_or_context_not_found",
                    "text": proposal.text,
                    "type": proposal.entity_type,
                }
            )
            continue
        for start in occurrences:
            global_start = source_offset + start
            global_end = global_start + len(proposal.text)
            entity_type = _LABEL_TO_ENTITY_TYPE[proposal.entity_type]
            item = EntityProposal(
                span=(global_start, global_end),
                candidate_types=(entity_type,),
                source=source,
                score=proposal.confidence,
                evidence_ids=(evidence_id,),
                features=tuple(
                    sorted(
                        (
                            ("left_context", proposal.left_context),
                            ("quoted_text", proposal.text),
                            ("right_context", proposal.right_context),
                            ("context_projection", context_projection),
                        )
                    )
                ),
            )
            # INVARIANT: offsets are calculated only from an exact source quote.
            item.validate_offsets(
                (" " * source_offset) + source_text
                if source_offset
                else source_text
            )
            projected.append(item)
    return projected, rejected


def select_qwen_confirmed_proposals(
    proposal_sources: Mapping[str, Sequence[EntityProposal]],
    *,
    thresholds: Mapping[EntityType, float] | None = None,
    minimum_sources: int = 2,
) -> tuple[EntityProposal, ...]:
    """Select exact span/type consensus while requiring at least one Qwen source."""

    if minimum_sources < 2:
        raise ValueError("Qwen consensus requires at least two independent sources or passes")
    active_thresholds: defaultdict[EntityType, float] = defaultdict(lambda: 0.0)
    if thresholds:
        active_thresholds.update(thresholds)
    grouped: dict[tuple[int, int, EntityType], list[EntityProposal]] = defaultdict(list)
    for source_name, proposals in proposal_sources.items():
        for proposal in proposals:
            if proposal.entity_type is None:
                continue
            if proposal.source != source_name:
                raise ValueError("Proposal mapping key must equal EntityProposal.source")
            grouped[(*proposal.span, proposal.entity_type)].append(proposal)
    selected: list[EntityProposal] = []
    for (start, end, entity_type), evidence in grouped.items():
        sources = {proposal.source for proposal in evidence}
        qwen_sources = {source for source in sources if source.startswith("qwen.")}
        if not qwen_sources or len(sources) < minimum_sources:
            continue
        qwen_score = max(
            proposal.score
            for proposal in evidence
            if proposal.source in qwen_sources
        )
        if qwen_score < active_thresholds[entity_type]:
            continue
        evidence_ids = tuple(
            sorted(
                {
                    evidence_id
                    for proposal in evidence
                    for evidence_id in proposal.evidence_ids
                }
            )
        )
        selected.append(
            EntityProposal(
                span=(start, end),
                candidate_types=(entity_type,),
                source="qwen.consensus",
                score=max(proposal.score for proposal in evidence),
                evidence_ids=evidence_ids,
                features=(
                    ("agreement_count", str(len(sources))),
                    ("agreement_sources", ",".join(sorted(sources))),
                ),
            )
        )
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                item.span[0],
                item.span[1],
                item.entity_type.value if item.entity_type else "",
            ),
        )
    )


def apply_phase1_adjudication(
    source_text: str,
    candidates: Sequence[Phase1AdjudicationCandidate],
    decisions: Sequence[Phase1AdjudicationDecision],
    *,
    minimum_confidence: float,
) -> tuple[EntityProposal, ...]:
    """Apply decisions locally; replacement quotes must overlap the original candidate."""

    by_id = {candidate.proposal_id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("Adjudication proposal_id values must be unique")
    decision_by_id = {decision.proposal_id: decision for decision in decisions}
    if len(decision_by_id) != len(decisions):
        raise ValueError("Adjudication may decide each proposal at most once")
    output: list[EntityProposal] = []
    for proposal_id, candidate in by_id.items():
        decision = decision_by_id.get(proposal_id)
        if decision is None or decision.action == "DROP":
            continue
        if decision.confidence < minimum_confidence:
            continue
        if decision.evidence_quote not in source_text:
            continue
        if decision.action == "KEEP":
            output.append(
                EntityProposal(
                    span=candidate.span,
                    candidate_types=(_LABEL_TO_ENTITY_TYPE[candidate.entity_type],),
                    source="qwen.adjudicated",
                    score=decision.confidence,
                    evidence_ids=(proposal_id,),
                    features=(("action", "KEEP"),),
                )
            )
            continue
        assert decision.replacement_text is not None
        assert decision.replacement_type is not None
        replacement_span = _overlapping_quote_span(
            source_text,
            decision.replacement_text,
            candidate.span,
        )
        if replacement_span is None:
            continue
        output.append(
            EntityProposal(
                span=replacement_span,
                candidate_types=(_LABEL_TO_ENTITY_TYPE[decision.replacement_type],),
                source="qwen.adjudicated",
                score=decision.confidence,
                evidence_ids=(proposal_id,),
                features=(("action", "REPLACE"),),
            )
        )
    return _deduplicate_entity_proposals(output)


def build_phase1_qwen_extraction_messages(
    source_text: str,
    *,
    pass_id: str,
    target_types: Sequence[Phase1Label],
) -> tuple[ChatMessage, ...]:
    schema = (
        '{"entities":[{"text":"exact quote","type":"'
        + '|'.join(target_types)
        + '","left_context":"","right_context":""}]}'
    )
    return (
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=(
                f"PASS={pass_id}\nChỉ tìm các type: {', '.join(target_types)}.\n"
                f"Schema: {schema}\nSOURCE_START\n{source_text}\nSOURCE_END"
            ),
        ),
    )


def build_phase1_qwen_review_messages(
    source_text: str,
    existing_entities: Sequence[Phase1ReviewEntity],
    *,
    round_index: int,
) -> tuple[ChatMessage, ...]:
    """Build the missing-only reviewer prompt without exposing trusted offsets."""

    if round_index < 0:
        raise ValueError("Review round index cannot be negative")
    for entity in existing_entities:
        entity.validate(source_text)
    serialized = [
        {"text": entity.text, "type": entity.entity_type}
        for entity in sorted(
            existing_entities,
            key=lambda item: (item.span[0], item.span[1], item.entity_type),
        )
    ]
    schema = (
        '{"entities":[{"text":"exact missing quote",'
        '"type":"TRIỆU_CHỨNG|TÊN_XÉT_NGHIỆM|KẾT_QUẢ_XÉT_NGHIỆM|'
        'CHẨN_ĐOÁN|THUỐC","left_context":"","right_context":""}]}'
    )
    return (
        ChatMessage(role="system", content=_REVIEW_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=(
                f"REVIEW_ROUND={round_index + 1}\n"
                f"Schema: {schema}\n"
                f"EXISTING_ENTITIES={json.dumps(serialized, ensure_ascii=False)}\n"
                f"SOURCE_START\n{source_text}\nSOURCE_END"
            ),
        ),
    )


def build_phase1_qwen_adjudication_messages(
    source_text: str,
    candidates: Sequence[Phase1AdjudicationCandidate],
) -> tuple[ChatMessage, ...]:
    serialized = [
        {
            "proposal_id": item.proposal_id,
            "text": item.text,
            "type": item.entity_type,
            "sources": list(item.sources),
            "confidence": item.confidence,
        }
        for item in candidates
    ]
    return (
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=(
                "Xét từng proposal. KEEP nếu span/type đúng; DROP nếu không phải entity; "
                "REPLACE chỉ khi có exact replacement quote trong SOURCE. evidence_quote phải "
                "là exact quote cùng clause. Trả JSON "
                '{"decisions":[{"proposal_id":"...","action":"KEEP|DROP|REPLACE",'
                '"confidence":0.0,"evidence_quote":"...",'
                '"replacement_text":null,"replacement_type":null}]}.\n'
                f"PROPOSALS={json.dumps(serialized, ensure_ascii=False)}\n"
                f"SOURCE_START\n{source_text}\nSOURCE_END"
            ),
        ),
    )


def _review_entities_for_window(
    source_text: str,
    window: RawTextWindow,
    seed_entities: Sequence[Phase1ReviewEntity],
    added_proposals: Sequence[EntityProposal],
) -> tuple[Phase1ReviewEntity, ...]:
    start, end = window.span
    visible: dict[tuple[int, int, str], Phase1ReviewEntity] = {}
    for entity in seed_entities:
        entity_start, entity_end = entity.span
        if start <= entity_start and entity_end <= end:
            local = Phase1ReviewEntity(
                text=entity.text,
                entity_type=entity.entity_type,
                span=(entity_start - start, entity_end - start),
            )
            local.validate(window.text)
            visible[(*local.span, local.entity_type)] = local
    for proposal in added_proposals:
        entity_type = proposal.entity_type
        if entity_type is None or entity_type not in _ENTITY_TYPE_TO_LABEL:
            continue
        entity_start, entity_end = proposal.span
        if start <= entity_start and entity_end <= end:
            local = Phase1ReviewEntity(
                text=source_text[entity_start:entity_end],
                entity_type=_ENTITY_TYPE_TO_LABEL[entity_type],  # type: ignore[arg-type]
                span=(entity_start - start, entity_end - start),
            )
            local.validate(window.text)
            visible[(*local.span, local.entity_type)] = local
    return tuple(
        sorted(
            visible.values(),
            key=lambda item: (item.span[0], item.span[1], item.entity_type),
        )
    )


def _parse_quoted_proposals(
    value: Any,
) -> tuple[list[Phase1QuotedProposal], list[dict[str, Any]]]:
    rows: Any
    if isinstance(value, list):
        # MODEL: some local chat checkpoints omit the requested top-level wrapper but still emit
        # the exact entity row schema. Row validation below remains authoritative.
        rows = value
    elif isinstance(value, Mapping):
        container_keys = (
            "entities",
            "missing_entities",
            "new_entities",
            "results",
            "items",
        )
        selected = [key for key in container_keys if isinstance(value.get(key), list)]
        if len(selected) != 1:
            raise StructuredResponseError(
                "Extraction response requires exactly one recognized entity array"
            )
        rows = value[selected[0]]
    else:
        raise StructuredResponseError(
            "Extraction response must be an entity array or an object containing one"
        )
    output: list[Phase1QuotedProposal] = []
    rejected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            rejected.append({"proposal_index": index, "reason": "not_an_object"})
            continue
        try:
            entity_type = str(row.get("type", ""))
            if entity_type not in _LABEL_TO_ENTITY_TYPE:
                raise ValueError("invalid_type")
            output.append(
                Phase1QuotedProposal(
                    text=str(row.get("text", "")),
                    entity_type=entity_type,  # type: ignore[arg-type]
                    # MODEL: self-reported confidence copied the schema placeholder in Qwen3.
                    # Exact quote projection receives a fixed source score; independent pass or
                    # support agreement remains the actual promotion signal.
                    confidence=_MODEL_PROPOSAL_CONFIDENCE,
                    left_context=_bounded_context(
                        row.get("left_context", ""),
                        side="left",
                    ),
                    right_context=_bounded_context(
                        row.get("right_context", ""),
                        side="right",
                    ),
                )
            )
        except (TypeError, ValueError) as error:
            rejected.append(
                {
                    "proposal_index": index,
                    "reason": f"invalid_proposal:{error}",
                }
            )
    return output, rejected


def _bounded_context(value: Any, *, side: Literal["left", "right"]) -> str:
    """Retain only the context nearest the quote without trusting model offsets."""

    text = str(value)
    if len(text) <= _MAX_CONTEXT_CHARACTERS:
        return text
    if side == "left":
        return text[-_MAX_CONTEXT_CHARACTERS:]
    return text[:_MAX_CONTEXT_CHARACTERS]


def _parse_adjudication_decisions(
    value: Any,
) -> tuple[Phase1AdjudicationDecision, ...]:
    if not isinstance(value, Mapping) or not isinstance(value.get("decisions"), list):
        raise StructuredResponseError("Adjudication response requires a decisions array")
    decisions: list[Phase1AdjudicationDecision] = []
    for row in value["decisions"]:
        if not isinstance(row, Mapping):
            raise StructuredResponseError("Adjudication decisions must be objects")
        action = str(row.get("action", ""))
        if action not in {"KEEP", "DROP", "REPLACE"}:
            raise StructuredResponseError(f"Invalid adjudication action: {action}")
        replacement_type = row.get("replacement_type")
        if replacement_type is not None and replacement_type not in _LABEL_TO_ENTITY_TYPE:
            raise StructuredResponseError("Invalid adjudication replacement_type")
        decisions.append(
            Phase1AdjudicationDecision(
                proposal_id=str(row.get("proposal_id", "")),
                action=action,  # type: ignore[arg-type]
                confidence=float(row.get("confidence", 0.0)),
                evidence_quote=str(row.get("evidence_quote", "")),
                replacement_text=(
                    None
                    if row.get("replacement_text") is None
                    else str(row["replacement_text"])
                ),
                replacement_type=replacement_type,
            )
        )
    return tuple(decisions)


def _deduplicate_entity_proposals(
    proposals: Sequence[EntityProposal],
) -> tuple[EntityProposal, ...]:
    best: dict[tuple[int, int, tuple[EntityType, ...], str], EntityProposal] = {}
    for proposal in proposals:
        key = (*proposal.span, proposal.candidate_types, proposal.source)
        current = best.get(key)
        if current is None or proposal.score > current.score:
            best[key] = proposal
    return tuple(
        sorted(
            best.values(),
            key=lambda item: (
                item.span[0],
                item.span[1],
                tuple(entity_type.value for entity_type in item.candidate_types),
                item.source,
            ),
        )
    )


def _exact_occurrences(source_text: str, quote: str) -> list[int]:
    starts: list[int] = []
    cursor = 0
    while True:
        start = source_text.find(quote, cursor)
        if start < 0:
            return starts
        starts.append(start)
        cursor = start + 1


def _overlapping_quote_span(
    source_text: str,
    quote: str,
    original_span: tuple[int, int],
) -> tuple[int, int] | None:
    spans = [(start, start + len(quote)) for start in _exact_occurrences(source_text, quote)]
    overlapping = [
        span
        for span in spans
        if span[0] < original_span[1] and original_span[0] < span[1]
    ]
    if len(overlapping) != 1:
        return None
    return overlapping[0]


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
