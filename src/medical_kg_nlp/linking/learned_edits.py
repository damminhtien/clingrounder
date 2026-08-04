"""Learn conservative mention-to-terminology edits from supervised alignment pairs.

Learned edits are lookup transformations, never source-text rewrites. They expand a normalized
query and resolve the transformed text through the canonical terminology repository, preserving
all entity offsets and dictionary constraints.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal, cast

from medical_kg_nlp.linking.candidate import Candidate
from medical_kg_nlp.retrieval.constraints import allowed_code_systems
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.terminology.ports import TerminologyRepository
from medical_kg_nlp.utils.text import normalize_for_match, strip_vietnamese_tones
from medical_kg_nlp.utils.io import read_jsonl, write_jsonl

__all__ = [
    "EditContextConstraints",
    "LearnedEditModel",
    "LearnedEditObservation",
    "LearnedEditRetrieverAdapter",
    "LearnedEditRule",
    "LearnedEditVariant",
    "learn_edit_transformations",
    "load_learned_edit_model",
    "write_learned_edit_model",
]

EditKind = Literal["whole", "token", "restore_diacritics"]


@dataclass(frozen=True, slots=True, order=True)
class EditContextConstraints:
    """Optional structural scope required before an edit may execute."""

    genre: str | None = None
    section: str | None = None

    def matches(self, *, genre: str | None, section: str | None) -> bool:
        return (self.genre is None or self.genre == genre) and (
            self.section is None or self.section == section
        )


@dataclass(frozen=True, slots=True)
class LearnedEditObservation:
    """One gold mention aligned to a terminology alias or canonical title."""

    mention: str
    terminology_alias: str
    entity_type: EntityType
    genre: str | None = None
    section: str | None = None

    def __post_init__(self) -> None:
        if not normalize_for_match(self.mention):
            raise ValueError("Learned-edit observation requires mention text")
        if not normalize_for_match(self.terminology_alias):
            raise ValueError("Learned-edit observation requires terminology_alias")


@dataclass(frozen=True, slots=True)
class LearnedEditRule:
    """Auditable edit accepted by support and precision gates."""

    rule_id: str
    kind: EditKind
    source: str
    target: str
    support: int
    correct_count: int
    precision: float
    entity_type: EntityType
    context_constraints: EditContextConstraints = EditContextConstraints()

    def __post_init__(self) -> None:
        if not self.rule_id.strip() or not self.source or not self.target:
            raise ValueError("Learned edit requires rule_id, source, and target")
        if self.source == self.target:
            raise ValueError("Learned edit must transform its source")
        if self.support < 1 or not 0 <= self.correct_count <= self.support:
            raise ValueError("Learned edit has invalid support counts")
        if not 0.0 <= self.precision <= 1.0:
            raise ValueError("Learned edit precision must be between 0 and 1")

    def to_json(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "kind": self.kind,
            "source": self.source,
            "target": self.target,
            "support": self.support,
            "correct_count": self.correct_count,
            "precision": self.precision,
            "entity_type": self.entity_type.value,
            "context_constraints": {
                "genre": self.context_constraints.genre,
                "section": self.context_constraints.section,
            },
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> "LearnedEditRule":
        raw_constraints = payload.get("context_constraints", {})
        if not isinstance(raw_constraints, Mapping):
            raise ValueError("Learned edit context_constraints must be an object")
        raw_kind = str(payload.get("kind", ""))
        if raw_kind not in {"whole", "token", "restore_diacritics"}:
            raise ValueError(f"Unknown learned edit kind {raw_kind!r}")
        return cls(
            rule_id=str(payload.get("rule_id", "")),
            kind=cast(EditKind, raw_kind),
            source=str(payload.get("source", "")),
            target=str(payload.get("target", "")),
            support=_integer_payload(payload, "support"),
            correct_count=_integer_payload(payload, "correct_count"),
            precision=_float_payload(payload, "precision"),
            entity_type=EntityType(str(payload.get("entity_type", ""))),
            context_constraints=EditContextConstraints(
                _optional_payload_text(raw_constraints.get("genre")),
                _optional_payload_text(raw_constraints.get("section")),
            ),
        )


@dataclass(frozen=True, slots=True)
class LearnedEditVariant:
    """One transformed normalized query and the rule that produced it."""

    text: str
    rule_id: str
    precision: float


@dataclass(frozen=True, slots=True)
class LearnedEditModel:
    """Immutable rule set with deterministic application order."""

    rules: tuple[LearnedEditRule, ...]
    minimum_support: int = 3
    minimum_precision: float = 0.90

    def __post_init__(self) -> None:
        if self.minimum_support < 1:
            raise ValueError("Learned-edit minimum_support must be positive")
        if not 0.0 <= self.minimum_precision <= 1.0:
            raise ValueError("Learned-edit minimum_precision must be within [0, 1]")
        if len({rule.rule_id for rule in self.rules}) != len(self.rules):
            raise ValueError("Learned-edit model contains duplicate rule IDs")
        if any(
            rule.support < self.minimum_support
            or rule.precision < self.minimum_precision
            for rule in self.rules
        ):
            raise ValueError("Learned-edit model contains a rule below its activation gate")

    def transform(
        self,
        mention: str,
        entity_type: EntityType,
        *,
        genre: str | None = None,
        section: str | None = None,
        limit: int = 10,
    ) -> tuple[LearnedEditVariant, ...]:
        """Return unique transformed queries without mutating the source mention."""

        if limit < 1:
            raise ValueError("Learned-edit transform limit must be positive")
        normalized = normalize_for_match(mention)
        variants: list[LearnedEditVariant] = []
        seen = {normalized}
        for rule in self.rules:
            if rule.entity_type is not entity_type or not rule.context_constraints.matches(
                genre=genre,
                section=section,
            ):
                continue
            transformed = _apply_rule(rule, normalized)
            if transformed is None or transformed in seen:
                continue
            seen.add(transformed)
            variants.append(
                LearnedEditVariant(
                    text=transformed,
                    rule_id=rule.rule_id,
                    precision=rule.precision,
                )
            )
            if len(variants) >= limit:
                break
        return tuple(variants)


@dataclass(frozen=True, slots=True)
class LearnedEditRetrieverAdapter:
    """Resolve learned query variants through exact, type-filtered terminology lookup."""

    model: LearnedEditModel
    repository: TerminologyRepository
    genre: str | None = None
    section: str | None = None
    source: str = "learned_edit"
    terminal_on_match: bool = False
    unique_output_short_circuit: bool = False

    def retrieve(
        self,
        mention: str,
        entity_type: EntityType,
        context_window: str,
        limit: int,
    ) -> list[Candidate]:
        del context_window
        variants = self.model.transform(
            mention,
            entity_type,
            genre=self.genre,
            section=self.section,
            limit=limit,
        )
        systems = allowed_code_systems(entity_type)
        output: list[Candidate] = []
        seen: set[str] = set()
        for variant in variants:
            for entry in self.repository.exact_lookup(
                variant.text,
                entity_type=entity_type,
                code_systems=systems,
                limit=limit,
            ):
                if entry.concept_id in seen:
                    continue
                seen.add(entry.concept_id)
                output.append(
                    Candidate(
                        concept_id=entry.concept_id,
                        code=entry.code,
                        code_system=entry.code_system,
                        canonical_name=entry.canonical_name,
                        semantic_type=entry.semantic_type,
                        score=variant.precision,
                        source=self.source,
                        matched_alias=variant.text,
                    )
                )
        return output[:limit]


@dataclass(frozen=True, slots=True, order=True)
class _RuleSignature:
    kind: EditKind
    source: str
    target: str
    entity_type: EntityType
    constraints: EditContextConstraints


def learn_edit_transformations(
    observations: Iterable[LearnedEditObservation],
    *,
    minimum_support: int = 3,
    minimum_precision: float = 0.90,
) -> LearnedEditModel:
    """Learn whole, token-level, and diacritic-restoration transformations.

    Precision is measured over every compatible observation where the rule can execute, rather
    than only over examples that originally proposed that rule. This guards common short forms
    such as ``k`` or ``tha`` from becoming broad false-positive expansions.
    """

    if minimum_support < 1:
        raise ValueError("minimum_support must be positive")
    if not 0.0 <= minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be within [0, 1]")
    values = tuple(observations)
    signatures: set[_RuleSignature] = set()
    for observation in values:
        for signature in _signatures_for_observation(observation):
            signatures.add(signature)

    accepted: list[LearnedEditRule] = []
    for signature in sorted(signatures, key=_signature_order):
        support = 0
        correct = 0
        for observation in values:
            if observation.entity_type is not signature.entity_type:
                continue
            if not signature.constraints.matches(
                genre=observation.genre,
                section=observation.section,
            ):
                continue
            mention = normalize_for_match(observation.mention)
            transformed = _apply_signature(signature, mention)
            if transformed is None:
                continue
            support += 1
            if transformed == normalize_for_match(observation.terminology_alias):
                correct += 1
        precision = correct / support if support else 0.0
        if support < minimum_support or precision < minimum_precision:
            continue
        accepted.append(
            LearnedEditRule(
                rule_id=_rule_id(signature),
                kind=signature.kind,
                source=signature.source,
                target=signature.target,
                support=support,
                correct_count=correct,
                precision=precision,
                entity_type=signature.entity_type,
                context_constraints=signature.constraints,
            )
        )
    accepted.sort(key=_rule_order)
    return LearnedEditModel(
        rules=tuple(accepted),
        minimum_support=minimum_support,
        minimum_precision=minimum_precision,
    )


def write_learned_edit_model(model: LearnedEditModel, path: str | Path) -> None:
    """Write model metadata followed by deterministic rule rows."""

    write_jsonl(
        path,
        (
            {
                "record_type": "metadata",
                "schema_version": "learned-edits.v1",
                "minimum_support": model.minimum_support,
                "minimum_precision": model.minimum_precision,
            },
            *(
                {"record_type": "rule", **rule.to_json()}
                for rule in model.rules
            ),
        ),
    )


def load_learned_edit_model(path: str | Path) -> LearnedEditModel:
    """Load a prebuilt learned-edit artifact without inducing rules at startup."""

    rows = read_jsonl(path)
    if not rows or rows[0].get("record_type") != "metadata":
        raise ValueError("Learned-edit artifact requires a leading metadata row")
    metadata = rows[0]
    if metadata.get("schema_version") != "learned-edits.v1":
        raise ValueError("Unsupported learned-edit artifact schema")
    rules = tuple(
        LearnedEditRule.from_json(row)
        for row in rows[1:]
        if row.get("record_type") == "rule"
    )
    if len(rules) != len(rows) - 1:
        raise ValueError("Learned-edit artifact contains an unknown record type")
    return LearnedEditModel(
        rules=rules,
        minimum_support=_integer_payload(metadata, "minimum_support"),
        minimum_precision=_float_payload(metadata, "minimum_precision"),
    )


def _signatures_for_observation(
    observation: LearnedEditObservation,
) -> tuple[_RuleSignature, ...]:
    source = normalize_for_match(observation.mention)
    target = normalize_for_match(observation.terminology_alias)
    if source == target:
        return ()
    constraints = [EditContextConstraints()]
    if observation.genre is not None or observation.section is not None:
        constraints.append(
            EditContextConstraints(observation.genre, observation.section)
        )
    candidates: set[tuple[EditKind, str, str]] = {("whole", source, target)}
    if strip_vietnamese_tones(source) == strip_vietnamese_tones(target):
        candidates.add(("restore_diacritics", strip_vietnamese_tones(source), target))
    source_tokens = source.split()
    target_tokens = target.split()
    matcher = SequenceMatcher(a=source_tokens, b=target_tokens, autojunk=False)
    for tag, source_start, source_end, target_start, target_end in matcher.get_opcodes():
        if tag != "replace" or source_start == source_end or target_start == target_end:
            continue
        candidates.add(
            (
                "token",
                " ".join(source_tokens[source_start:source_end]),
                " ".join(target_tokens[target_start:target_end]),
            )
        )
    return tuple(
        _RuleSignature(kind, old, new, observation.entity_type, constraint)
        for constraint in constraints
        for kind, old, new in candidates
        if old and new and old != new
    )


def _apply_rule(rule: LearnedEditRule, normalized_mention: str) -> str | None:
    signature = _RuleSignature(
        rule.kind,
        rule.source,
        rule.target,
        rule.entity_type,
        rule.context_constraints,
    )
    return _apply_signature(signature, normalized_mention)


def _apply_signature(signature: _RuleSignature, normalized_mention: str) -> str | None:
    if signature.kind == "whole":
        return signature.target if normalized_mention == signature.source else None
    if signature.kind == "restore_diacritics":
        return (
            signature.target
            if strip_vietnamese_tones(normalized_mention) == signature.source
            else None
        )
    tokens = normalized_mention.split()
    source_tokens = signature.source.split()
    for start in range(0, len(tokens) - len(source_tokens) + 1):
        if tokens[start : start + len(source_tokens)] != source_tokens:
            continue
        replacement = signature.target.split()
        return " ".join((*tokens[:start], *replacement, *tokens[start + len(source_tokens) :]))
    return None


def _rule_id(signature: _RuleSignature) -> str:
    payload = "\x1f".join(
        (
            signature.kind,
            signature.entity_type.value,
            signature.source,
            signature.target,
            signature.constraints.genre or "*",
            signature.constraints.section or "*",
        )
    )
    return f"learned-edit-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def _signature_order(signature: _RuleSignature) -> tuple[str, str, str, str, str, str]:
    return (
        signature.entity_type.value,
        signature.kind,
        signature.source,
        signature.target,
        signature.constraints.genre or "",
        signature.constraints.section or "",
    )


def _rule_order(rule: LearnedEditRule) -> tuple[int, float, int, int, str]:
    specificity = int(rule.context_constraints.genre is not None) + int(
        rule.context_constraints.section is not None
    )
    kind_priority = {"whole": 0, "restore_diacritics": 1, "token": 2}[rule.kind]
    return (-specificity, -rule.precision, -rule.support, kind_priority, rule.rule_id)


def _integer_payload(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Learned edit {field} must be an integer")
    return value


def _float_payload(payload: Mapping[str, object], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Learned edit {field} must be numeric")
    return float(value)


def _optional_payload_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
