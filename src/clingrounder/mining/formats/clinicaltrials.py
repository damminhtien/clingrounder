"""Offset-safe rendering of ClinicalTrials.gov condition/intervention records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from clingrounder.schema.types import EntityType

__all__ = [
    "ClinicalTrialRenderedField",
    "ClinicalTrialRenderedStudy",
    "render_clinical_trial",
]


@dataclass(frozen=True)
class ClinicalTrialRenderedField:
    """One source-structured field projected onto immutable rendered text."""

    span: tuple[int, int]
    text: str
    entity_type: str
    source_label: str
    role: str
    group_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "span": [self.span[0], self.span[1]],
            "text": self.text,
            "entity_type": self.entity_type,
            "source_label": self.source_label,
            "role": self.role,
            "group_id": self.group_id,
        }


@dataclass(frozen=True)
class ClinicalTrialRenderedStudy:
    """Rendered trial text plus exact fields used by source labelers."""

    text: str
    fields: tuple[ClinicalTrialRenderedField, ...]


def render_clinical_trial(protocol: Mapping[str, Any]) -> ClinicalTrialRenderedStudy:
    """Render API v2 protocol fields without weak matching the resulting text."""

    identification = _mapping(protocol.get("identificationModule"))
    description = _mapping(protocol.get("descriptionModule"))
    conditions_module = _mapping(protocol.get("conditionsModule"))
    arms = _mapping(protocol.get("armsInterventionsModule"))
    outcomes = _mapping(protocol.get("outcomesModule"))
    blocks: list[str] = []
    fields: list[ClinicalTrialRenderedField] = []

    _append_plain_block(blocks, "Title", identification.get("briefTitle"))
    _append_plain_block(blocks, "Summary", description.get("briefSummary"))

    conditions = _strings(conditions_module.get("conditions"))
    condition_rows = tuple(
        (value, value, EntityType.DISEASE, "CTGOV_CONDITION", f"condition:{index}")
        for index, value in enumerate(conditions)
    )
    _append_field_block(blocks, fields, "Conditions", condition_rows, role="condition")

    intervention_rows: list[tuple[str, str, EntityType, str, str]] = []
    interventions = arms.get("interventions", ())
    if isinstance(interventions, Sequence) and not isinstance(interventions, (str, bytes)):
        for index, intervention in enumerate(interventions):
            if not isinstance(intervention, Mapping):
                continue
            name = str(intervention.get("name", "")).strip()
            if not name:
                continue
            description_text = str(intervention.get("description", "")).strip()
            rendered = ": ".join(
                value for value in (name, description_text) if value
            )
            source_type = str(intervention.get("type", "OTHER")).strip().upper()
            entity_type = _INTERVENTION_ENTITY_TYPES.get(source_type, EntityType.OTHER)
            intervention_rows.append(
                (
                    rendered,
                    name,
                    entity_type,
                    f"CTGOV_INTERVENTION_{source_type}",
                    f"intervention:{index}",
                )
            )
    _append_field_block(
        blocks,
        fields,
        "Interventions",
        tuple(intervention_rows),
        role="intervention",
    )

    for key, title, role in (
        ("primaryOutcomes", "Primary outcomes", "primary_outcome"),
        ("secondaryOutcomes", "Secondary outcomes", "secondary_outcome"),
    ):
        outcome_rows: list[tuple[str, str, EntityType, str, str]] = []
        raw_outcomes = outcomes.get(key, ())
        if isinstance(raw_outcomes, Sequence) and not isinstance(
            raw_outcomes, (str, bytes)
        ):
            for index, outcome in enumerate(raw_outcomes):
                if not isinstance(outcome, Mapping):
                    continue
                measure = str(outcome.get("measure", "")).strip()
                if measure:
                    outcome_rows.append(
                        (
                            measure,
                            measure,
                            EntityType.OTHER,
                            f"CTGOV_{role.upper()}",
                            f"{role}:{index}",
                        )
                    )
        _append_field_block(
            blocks,
            fields,
            title,
            tuple(outcome_rows),
            role=role,
        )

    text = "\n\n".join(blocks)
    if not text:
        raise ValueError("ClinicalTrials study produced no text sections")
    # INVARIANT: source fields are validated only after final text assembly.
    for field in fields:
        if text[field.span[0] : field.span[1]] != field.text:
            raise AssertionError(f"ClinicalTrials offset mismatch for {field.text!r}")
    return ClinicalTrialRenderedStudy(text=text, fields=tuple(fields))


def _append_plain_block(blocks: list[str], title: str, value: Any) -> None:
    text = "" if value is None else str(value).strip()
    if text:
        blocks.append(f"{title}\n{text}")


def _append_field_block(
    blocks: list[str],
    fields: list[ClinicalTrialRenderedField],
    title: str,
    rows: tuple[tuple[str, str, EntityType, str, str], ...],
    *,
    role: str,
) -> None:
    if not rows:
        return
    block_start = sum(len(block) for block in blocks) + 2 * len(blocks)
    lines = [title]
    relative_cursor = len(title)
    for rendered, annotation_text, entity_type, source_label, group_id in rows:
        prefix = "\n- "
        annotation_start = block_start + relative_cursor + len(prefix)
        lines.append(f"- {rendered}")
        fields.append(
            ClinicalTrialRenderedField(
                span=(annotation_start, annotation_start + len(annotation_text)),
                text=annotation_text,
                entity_type=entity_type.value,
                source_label=source_label,
                role=role,
                group_id=group_id,
            )
        )
        relative_cursor += len(prefix) + len(rendered)
    blocks.append("\n".join(lines))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(text for item in value if (text := str(item).strip()))


_INTERVENTION_ENTITY_TYPES = {
    "BEHAVIORAL": EntityType.PROCEDURE,
    "BIOLOGICAL": EntityType.DRUG,
    "COMBINATION_PRODUCT": EntityType.DRUG,
    "DEVICE": EntityType.PROCEDURE,
    "DIAGNOSTIC_TEST": EntityType.LAB_TEST,
    "DIETARY_SUPPLEMENT": EntityType.DRUG,
    "DRUG": EntityType.DRUG,
    "GENETIC": EntityType.PROCEDURE,
    "PROCEDURE": EntityType.PROCEDURE,
    "RADIATION": EntityType.PROCEDURE,
}
