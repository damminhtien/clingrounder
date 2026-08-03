"""Conservative boundary replacement for frozen Phase 1 entity selections.

The boundary ranker proposes alternatives, but it does not own final entities. This module
applies only high-evidence replacements to an already selected entity set and records why every
candidate was accepted or rejected.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from medical_kg_nlp.benchmarks.phase1.boundary_verifier import (
    Phase1BoundaryVerifier,
    ScoredPhase1BoundaryVariant,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import (
    is_phase1_heading_only_proposal,
)
from medical_kg_nlp.ner.document_structure import DocumentStructureAnalyzer
from medical_kg_nlp.benchmarks.phase1.ontology import PHASE1_CODABLE_TYPES, PHASE1_TYPE_PRIORITY

__all__ = [
    "BoundaryPolicy",
    "BoundaryOverlayResult",
    "apply_conservative_boundary_overlay",
]

BoundaryMode = Literal["disabled", "conservative_replacement"]

# The first public boundary probe is intentionally limited to independently corroborated spans.
_MIN_REPLACEMENT_SOURCE_SUPPORT = 2


@dataclass(frozen=True, slots=True)
class BoundaryPolicy:
    """Runtime controls for boundary repair after proposal fusion.

    ``conservative_replacement`` never adds or deletes an entity. It replaces one selected base
    identity only after the boundary verifier independently clears its threshold and margin.
    """

    mode: BoundaryMode = "disabled"
    require_same_type: bool = True
    require_base_selected: bool = True
    preserve_unmodified_identity: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "conservative_replacement"}:
            raise ValueError(f"Unsupported boundary policy mode {self.mode!r}")
        if self.mode == "conservative_replacement" and not all(
            (
                self.require_same_type,
                self.require_base_selected,
                self.preserve_unmodified_identity,
            )
        ):
            raise ValueError(
                "Conservative boundary replacement requires all identity safeguards"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BoundaryPolicy":
        """Parse an explicit run-spec policy without accepting unreviewed controls."""

        expected = {
            "mode",
            "require_same_type",
            "require_base_selected",
            "preserve_unmodified_identity",
        }
        unknown = set(payload) - expected
        missing = expected - set(payload)
        if unknown or missing:
            raise ValueError(
                "Boundary policy fields are invalid: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        values = {
            key: payload[key]
            for key in (
                "require_same_type",
                "require_base_selected",
                "preserve_unmodified_identity",
            )
        }
        if not all(isinstance(value, bool) for value in values.values()):
            raise ValueError("Boundary policy safeguards must be boolean")
        raw_mode = payload["mode"]
        if (
            not isinstance(raw_mode, str)
            or raw_mode not in {"disabled", "conservative_replacement"}
        ):
            raise ValueError("Boundary policy mode is invalid")
        return cls(
            mode=cast(BoundaryMode, raw_mode),
            require_same_type=bool(values["require_same_type"]),
            require_base_selected=bool(values["require_base_selected"]),
            preserve_unmodified_identity=bool(
                values["preserve_unmodified_identity"]
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize only controls that change the submission decision boundary."""

        return {
            "mode": self.mode,
            "require_same_type": self.require_same_type,
            "require_base_selected": self.require_base_selected,
            "preserve_unmodified_identity": self.preserve_unmodified_identity,
        }


@dataclass(frozen=True, slots=True)
class BoundaryOverlayResult:
    """Rows and auditable evidence emitted by conservative boundary repair."""

    rows_by_document: Mapping[str, tuple[Mapping[str, Any], ...]]
    changed_identities: frozenset[tuple[str, str, str, int, int]]
    decisions: tuple[Mapping[str, Any], ...]
    counters: Mapping[str, int]
    diagnostic_report: Mapping[str, int | float | None | str]


def apply_conservative_boundary_overlay(
    *,
    base_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    boundary_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    scored_variants: Sequence[ScoredPhase1BoundaryVariant],
    source_text_by_document: Mapping[str, str],
    verifier: Phase1BoundaryVerifier,
    policy: BoundaryPolicy,
) -> BoundaryOverlayResult:
    """Replace only a selected same-type base row with a stronger raw-exact variant.

    INVARIANT: changed identities receive empty metadata here. Candidate linking and assertion
    classification run again later; this prevents stale metadata from surviving a span change.
    """

    expected_ids = set(source_text_by_document)
    if set(base_rows) != expected_ids or set(boundary_rows) != expected_ids:
        raise ValueError("Boundary overlay document IDs must match source documents")
    if policy.mode == "disabled":
        return _disabled_result(base_rows)
    if verifier.resolution_policy != "conservative_replacement":
        raise ValueError("Boundary overlay requires a conservative boundary verifier")
    if not verifier.requires_base_probability:
        raise ValueError("Boundary overlay requires a proposal-conditioned verifier")

    boundary_selected = {
        (document_id, *_identity_key(row))
        for document_id, rows in boundary_rows.items()
        for row in rows
    }
    base_items_by_family: dict[str, list[ScoredPhase1BoundaryVariant]] = {}
    for item in scored_variants:
        if item.base_selected:
            base_items_by_family.setdefault(item.variant.family_id, []).append(item)

    output = {
        document_id: [_copy_row(row) for row in rows]
        for document_id, rows in base_rows.items()
    }
    decisions: list[Mapping[str, Any]] = []
    counters: Counter[str] = Counter()
    changed: set[tuple[str, str, str, int, int]] = set()
    structures = {
        document_id: DocumentStructureAnalyzer().analyze(source_text)
        for document_id, source_text in source_text_by_document.items()
    }

    candidates = [
        item
        for item in scored_variants
        if item.selected and item.replacement_selected
    ]
    for item in sorted(candidates, key=_replacement_sort_key):
        counters["replacement.considered"] += 1
        variant = item.variant
        document_id = variant.document_id
        source_text = source_text_by_document.get(document_id)
        if source_text is None:
            raise ValueError(f"Boundary variant references unknown document {document_id!r}")
        base_items = base_items_by_family.get(variant.family_id, [])
        base_item = base_items[0] if len(base_items) == 1 else None
        rejected = _replacement_rejection(
            item,
            base_item,
            output[document_id],
            boundary_selected,
            source_text,
            structures[document_id],
            verifier,
        )
        if rejected is not None:
            counters[f"replacement.rejected.{rejected}"] += 1
            decisions.append(
                _decision(item, action="reject", reason=rejected, base_item=base_item)
            )
            continue

        assert base_item is not None
        base_index = _base_row_index(output[document_id], base_item)
        if base_index is None:
            counters["replacement.rejected.base_not_in_resolved_output"] += 1
            decisions.append(
                _decision(
                    item,
                    action="reject",
                    reason="base_not_in_resolved_output",
                    base_item=base_item,
                )
            )
            continue
        if any(
            index != base_index and _overlap(variant.position, _position(existing))
            for index, existing in enumerate(output[document_id])
        ):
            # A conservative replacement never displaces a second accepted entity. The proposal
            # resolver remains the owner of cross-entity conflict selection.
            counters["replacement.rejected.conflicts_with_accepted_entity"] += 1
            decisions.append(
                _decision(
                    item,
                    action="reject",
                    reason="conflicts_with_accepted_entity",
                    base_item=base_item,
                )
            )
            continue

        before = output[document_id][base_index]
        replacement = _replacement_row(variant.text, variant.entity_type, variant.position)
        output[document_id][base_index] = replacement
        changed.add((document_id, *_identity_key(replacement)))
        counters["replacement.applied"] += 1
        decisions.append(
            _decision(
                item,
                action="replace",
                reason="high_margin_multisource_same_type",
                base_item=base_item,
                before=before,
                after=replacement,
            )
        )

    for document_id, rows in output.items():
        for row in rows:
            _validate_raw_row(row, source_text_by_document[document_id])
        rows.sort(key=_row_sort_key)
    counters["replacement.changed_identity_count"] = len(changed)
    counters["replacement.output_entity_total"] = sum(
        len(rows) for rows in output.values()
    )
    return BoundaryOverlayResult(
        rows_by_document={
            document_id: tuple(dict(row) for row in rows)
            for document_id, rows in output.items()
        },
        changed_identities=frozenset(changed),
        decisions=tuple(decisions),
        counters=dict(sorted(counters.items())),
        diagnostic_report={
            "replacement_count": counters["replacement.applied"],
            # Official labels are not available during Round 2 inference. These values are
            # intentionally unknown rather than fabricated from local manual-gold telemetry.
            "replacement_correct": None,
            "replacement_precision": None,
            "base_errors_fixed": None,
            "correct_bases_destroyed": None,
            "net_exact_span_gain": None,
            "label_status": "official_submission_required",
        },
    )


def _disabled_result(
    base_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> BoundaryOverlayResult:
    rows = {
        document_id: tuple(_copy_row(row) for row in values)
        for document_id, values in base_rows.items()
    }
    return BoundaryOverlayResult(
        rows_by_document=rows,
        changed_identities=frozenset(),
        decisions=(),
        counters={"replacement.disabled": 1},
        diagnostic_report={
            "replacement_count": 0,
            "replacement_correct": None,
            "replacement_precision": None,
            "base_errors_fixed": None,
            "correct_bases_destroyed": None,
            "net_exact_span_gain": None,
            "label_status": "boundary_policy_disabled",
        },
    )


def _replacement_rejection(
    item: ScoredPhase1BoundaryVariant,
    base_item: ScoredPhase1BoundaryVariant | None,
    rows: Sequence[Mapping[str, Any]],
    boundary_selected: set[tuple[str, str, str, int, int]],
    source_text: str,
    structure: Any,
    verifier: Phase1BoundaryVerifier,
) -> str | None:
    variant = item.variant
    if base_item is None:
        return "ambiguous_base_family"
    if not item.replacement_selected or not base_item.base_selected:
        return "base_or_replacement_not_selected"
    if variant.document_id != base_item.variant.document_id:
        return "document_mismatch"
    if variant.entity_type != base_item.variant.entity_type:
        return "type_mismatch"
    if len(variant.sources) < _MIN_REPLACEMENT_SOURCE_SUPPORT:
        return "insufficient_source_support"
    if (variant.document_id, *_identity_key_from_variant(variant)) not in boundary_selected:
        return "replacement_not_in_boundary_resolution"
    start, end = variant.position
    if source_text[start:end] != variant.text:
        return "raw_offset_mismatch"
    if is_phase1_heading_only_proposal(
        variant.to_proposal_row(),
        source_text,
        structure=structure,
    ):
        return "structural_heading"
    threshold = verifier.threshold_for(variant.entity_type, genre=item.genre)
    if item.probability < threshold:
        return "below_boundary_threshold"
    margin = item.probability - base_item.probability
    if margin < verifier.replacement_margin_by_type[variant.entity_type]:
        return "below_replacement_margin"
    if _base_row_index(rows, base_item) is None:
        return "base_not_in_resolved_output"
    return None


def _base_row_index(
    rows: Sequence[Mapping[str, Any]],
    base_item: ScoredPhase1BoundaryVariant,
) -> int | None:
    expected = _identity_key_from_variant(base_item.variant)
    for index, row in enumerate(rows):
        if _identity_key(row) == expected:
            return index
    return None


def _replacement_row(
    text: str,
    entity_type: str,
    position: tuple[int, int],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "text": text,
        "type": entity_type,
        "assertions": [],
        "position": [position[0], position[1]],
    }
    if entity_type in PHASE1_CODABLE_TYPES:
        row["candidates"] = []
    return row


def _decision(
    item: ScoredPhase1BoundaryVariant,
    *,
    action: str,
    reason: str,
    base_item: ScoredPhase1BoundaryVariant | None,
    before: Mapping[str, Any] | None = None,
    after: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    margin = (
        None if base_item is None else item.probability - base_item.probability
    )
    payload: dict[str, Any] = {
        "document_id": item.variant.document_id,
        "stage": "boundary_conservative_overlay",
        "action": action,
        "reason": reason,
        "variant_probability": item.probability,
        "base_probability": None if base_item is None else base_item.probability,
        "replacement_margin": margin,
        "source_count": len(item.variant.sources),
        "entity": _identity_payload_from_variant(item.variant),
    }
    if before is not None:
        payload["before"] = _identity_payload(before)
    if after is not None:
        payload["after"] = _identity_payload(after)
    return payload


def _replacement_sort_key(
    item: ScoredPhase1BoundaryVariant,
) -> tuple[float, float, int, str, int, int, str]:
    start, end = item.variant.position
    return (
        -item.probability,
        -len(item.variant.sources),
        -PHASE1_TYPE_PRIORITY[item.variant.entity_type],
        item.variant.document_id,
        start,
        end,
        item.variant.variant_id,
    )


def _copy_row(row: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    for field in ("position", "assertions", "candidates"):
        if isinstance(copied.get(field), list):
            copied[field] = list(copied[field])
    return copied


def _position(row: Mapping[str, Any]) -> tuple[int, int]:
    raw = row.get("position")
    if (
        not isinstance(raw, list | tuple)
        or len(raw) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in raw)
        or raw[0] < 0
        or raw[1] <= raw[0]
    ):
        raise ValueError("Boundary overlay row position is invalid")
    return int(raw[0]), int(raw[1])


def _identity_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    start, end = _position(row)
    return str(row.get("type", "")), str(row.get("text", "")), start, end


def _identity_key_from_variant(variant: Any) -> tuple[str, str, int, int]:
    return (
        str(variant.entity_type),
        str(variant.text),
        int(variant.position[0]),
        int(variant.position[1]),
    )


def _identity_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    entity_type, text, start, end = _identity_key(row)
    return {"text": text, "type": entity_type, "position": [start, end]}


def _identity_payload_from_variant(variant: Any) -> dict[str, Any]:
    entity_type, text, start, end = _identity_key_from_variant(variant)
    return {"text": text, "type": entity_type, "position": [start, end]}


def _validate_raw_row(row: Mapping[str, Any], source_text: str) -> None:
    start, end = _position(row)
    if source_text[start:end] != row.get("text"):
        raise ValueError("Boundary overlay violates the raw offset invariant")


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _row_sort_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    start, end = _position(row)
    entity_type = str(row.get("type", ""))
    return start, end, -PHASE1_TYPE_PRIORITY.get(entity_type, 0), entity_type
