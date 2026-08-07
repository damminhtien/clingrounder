"""Mine explicit abbreviation definitions without leaking evaluation text into runtime knowledge.

The miner recognizes definitions such as ``magnetic resonance imaging (MRI)`` and
``MRI (magnetic resonance imaging)``. It uses backward character alignment instead of a broad
parenthesis regex, which removes leading prose and rejects most section-number or citation noise.
Only definitions from policy-declared knowledge splits can produce a runtime table; held-out
definitions remain evaluation evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from clingrounder.dictionaries.synonym_table import ConceptEntry
from clingrounder.mining.records import MinedDocument, SourceArtifact
from clingrounder.preprocessing.normalizer import NORMALIZATION_CONTRACT_VERSION
from clingrounder.schema.types import CodeSystem
from clingrounder.terminology.ports import TerminologyRepository
from clingrounder.utils.text import normalize_for_match

__all__ = [
    "AbbreviationMiningPolicy",
    "AbbreviationMiningResult",
    "benchmark_abbreviation_knowledge",
    "build_runtime_abbreviation_table",
    "load_abbreviation_mining_policy",
    "load_snapshot_splits",
    "mine_abbreviations",
]

_POLICY_SCHEMA_VERSION = "abbreviation-mining-policy.v1"
_MINER_REVISION = "explicit-parenthetical-v2"
_PARENTHETICAL_RE = re.compile(r"\(([^()\n]{2,180})\)")
_PRECEDING_TOKEN_RE = re.compile(r"([A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9+./\-]{1,19})\s*$")
_CLAUSE_BOUNDARY_RE = re.compile(r"[\n.;:!?]")
_ROMAN_NUMERAL_RE = re.compile(r"[IVXLCDM]+")
_WORD_RE = re.compile(r"[^\W_]+(?:[-/][^\W_]+)*", flags=re.UNICODE)
_INITIALISM_STOPWORDS = frozenset(
    {"a", "an", "and", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}
)


@dataclass(frozen=True)
class AbbreviationMiningPolicy:
    """Source, split, shape, and support gates for one abbreviation experiment."""

    policy_id: str
    accepted_source_ids: tuple[str, ...]
    accepted_source_versions: tuple[str, ...]
    accepted_source_sha256: tuple[str, ...]
    accepted_languages: tuple[str, ...]
    knowledge_splits: tuple[str, ...]
    evaluation_splits: tuple[str, ...]
    min_supporting_documents: int = 2
    min_supporting_groups: int = 2
    min_character_alignment_documents: int = 5
    min_character_alignment_groups: int = 5
    min_short_characters: int = 2
    max_short_characters: int = 20
    max_long_characters: int = 180
    max_long_words: int = 24
    max_examples: int = 20
    reject_pure_roman_numerals: bool = True

    def __post_init__(self) -> None:
        required = (
            self.policy_id,
            *self.accepted_source_ids,
            *self.accepted_source_versions,
            *self.accepted_source_sha256,
            *self.accepted_languages,
            *self.knowledge_splits,
            *self.evaluation_splits,
        )
        if any(not value.strip() for value in required):
            raise ValueError("Abbreviation policy fields must be explicit and non-empty")
        if set(self.knowledge_splits) & set(self.evaluation_splits):
            raise ValueError("Knowledge and evaluation splits must be disjoint")
        support_thresholds = (
            self.min_supporting_documents,
            self.min_supporting_groups,
            self.min_character_alignment_documents,
            self.min_character_alignment_groups,
        )
        if any(value < 1 for value in support_thresholds):
            raise ValueError("Abbreviation support thresholds must be positive")
        if not 2 <= self.min_short_characters <= self.max_short_characters:
            raise ValueError("Abbreviation short-form limits are invalid")
        if self.max_long_characters < 4 or self.max_long_words < 1:
            raise ValueError("Abbreviation long-form limits are invalid")
        if self.max_examples < 1:
            raise ValueError("Abbreviation max_examples must be positive")


@dataclass(frozen=True)
class AbbreviationMiningResult:
    """Occurrence evidence, aggregate decisions, and a conflict-free runtime table."""

    definitions: tuple[dict[str, Any], ...]
    candidates: tuple[dict[str, Any], ...]
    abbreviation_table: tuple[dict[str, Any], ...]
    conflicts: tuple[dict[str, Any], ...]
    report: dict[str, Any]


@dataclass(frozen=True)
class _Definition:
    definition_id: str
    document_id: str
    split: str
    abbreviation: str
    expansion: str
    abbreviation_span: tuple[int, int]
    expansion_span: tuple[int, int]
    direction: str
    alignment_method: str
    group_id: str
    source_id: str
    source_version: str
    source_sha256: str

    @property
    def normalized_abbreviation(self) -> str:
        return normalize_for_match(self.abbreviation)

    @property
    def normalized_expansion(self) -> str:
        return normalize_for_match(self.expansion)

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition_id": self.definition_id,
            "document_id": self.document_id,
            "split": self.split,
            "abbreviation": self.abbreviation,
            "normalized_abbreviation": self.normalized_abbreviation,
            "expansion": self.expansion,
            "normalized_expansion": self.normalized_expansion,
            "abbreviation_span": list(self.abbreviation_span),
            "expansion_span": list(self.expansion_span),
            "direction": self.direction,
            "alignment_method": self.alignment_method,
            "group_id": self.group_id,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "source_sha256": self.source_sha256,
        }


def load_abbreviation_mining_policy(path: str | Path) -> AbbreviationMiningPolicy:
    """Load a fail-closed, versioned abbreviation policy."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Abbreviation mining policy must be an object")
    if raw.get("schema_version") != _POLICY_SCHEMA_VERSION:
        raise ValueError("Unsupported abbreviation mining policy schema version")
    return AbbreviationMiningPolicy(
        policy_id=_required_string(raw, "policy_id"),
        accepted_source_ids=_string_tuple(raw, "accepted_source_ids"),
        accepted_source_versions=_string_tuple(raw, "accepted_source_versions"),
        accepted_source_sha256=_string_tuple(raw, "accepted_source_sha256"),
        accepted_languages=_string_tuple(raw, "accepted_languages"),
        knowledge_splits=_string_tuple(raw, "knowledge_splits"),
        evaluation_splits=_string_tuple(raw, "evaluation_splits"),
        min_supporting_documents=int(raw.get("min_supporting_documents", 2)),
        min_supporting_groups=int(raw.get("min_supporting_groups", 2)),
        min_character_alignment_documents=int(raw.get("min_character_alignment_documents", 5)),
        min_character_alignment_groups=int(raw.get("min_character_alignment_groups", 5)),
        min_short_characters=int(raw.get("min_short_characters", 2)),
        max_short_characters=int(raw.get("max_short_characters", 20)),
        max_long_characters=int(raw.get("max_long_characters", 180)),
        max_long_words=int(raw.get("max_long_words", 24)),
        max_examples=int(raw.get("max_examples", 20)),
        reject_pure_roman_numerals=bool(raw.get("reject_pure_roman_numerals", True)),
    )


def load_snapshot_splits(path: str | Path) -> dict[str, str]:
    """Load immutable document split assignments from a dataset snapshot manifest."""

    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("splits"), Mapping):
        raise ValueError(f"{source}: snapshot manifest has no split assignments")
    splits = {str(key): str(value) for key, value in raw["splits"].items()}
    if any(not key or not value for key, value in splits.items()):
        raise ValueError(f"{source}: split assignments must be non-empty")
    return splits


def mine_abbreviations(
    documents: Iterable[MinedDocument],
    artifacts: Sequence[SourceArtifact],
    split_by_document: Mapping[str, str],
    policy: AbbreviationMiningPolicy,
    *,
    base_abbreviation_rows: Sequence[Mapping[str, Any]] = (),
) -> AbbreviationMiningResult:
    """Extract definitions and compile only conflict-free knowledge-split abbreviations."""

    artifacts_by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    if len(artifacts_by_id) != len(artifacts):
        raise ValueError("Abbreviation mining artifacts must have unique IDs")
    accepted_splits = set(policy.knowledge_splits) | set(policy.evaluation_splits)
    reason_counts: Counter[str] = Counter()
    definitions: list[_Definition] = []
    document_count = 0
    selected_document_count = 0
    parenthetical_count = 0

    for document in documents:
        document_count += 1
        split = split_by_document.get(document.document_id)
        artifact = artifacts_by_id.get(document.source_artifact_id)
        reason = _document_rejection_reason(document, artifact, split, accepted_splits, policy)
        if reason is not None:
            reason_counts[reason] += 1
            continue
        assert artifact is not None and split is not None
        selected_document_count += 1
        for parenthetical in _PARENTHETICAL_RE.finditer(document.text):
            parenthetical_count += 1
            extracted, extraction_reason = _definition_from_parenthetical(
                document,
                artifact,
                split,
                parenthetical,
                policy,
            )
            reason_counts[extraction_reason] += 1
            if extracted is not None:
                definitions.append(extracted)

    ordered_definitions = tuple(
        definition.to_dict()
        for definition in sorted(
            definitions,
            key=lambda value: (
                value.document_id,
                value.abbreviation_span,
                value.expansion_span,
            ),
        )
    )
    candidates, table, conflicts, aggregation_counts = _aggregate_definitions(
        definitions,
        policy,
        base_abbreviation_rows=base_abbreviation_rows,
    )
    split_counts = Counter(definition.split for definition in definitions)
    alignment_counts = Counter(definition.alignment_method for definition in definitions)
    report: dict[str, Any] = {
        "schema_version": "abbreviation-mining-report.v1",
        "miner_revision": _MINER_REVISION,
        "normalization_version": NORMALIZATION_CONTRACT_VERSION,
        "policy_id": policy.policy_id,
        "document_count": document_count,
        "selected_document_count": selected_document_count,
        "parenthetical_count": parenthetical_count,
        "definition_count": len(ordered_definitions),
        "unique_definition_count": len(
            {
                (row["normalized_abbreviation"], row["normalized_expansion"])
                for row in ordered_definitions
            }
        ),
        "definition_split_counts": dict(sorted(split_counts.items())),
        "definition_alignment_counts": dict(sorted(alignment_counts.items())),
        "candidate_count": len(candidates),
        "abbreviation_table_count": len(table),
        "conflict_count": len(conflicts),
        "reason_counts": dict(sorted(reason_counts.items())),
        "aggregation_reason_counts": dict(sorted(aggregation_counts.items())),
        "split_contract": {
            "knowledge_splits": list(policy.knowledge_splits),
            "evaluation_splits": list(policy.evaluation_splits),
            "evaluation_used_for_table": False,
        },
    }
    return AbbreviationMiningResult(
        definitions=ordered_definitions,
        candidates=candidates,
        abbreviation_table=table,
        conflicts=conflicts,
        report=report,
    )


def benchmark_abbreviation_knowledge(
    definitions: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    mined_rows: Sequence[Mapping[str, Any]],
    *,
    evaluation_splits: Sequence[str],
    repository: TerminologyRepository | None = None,
    retrieval_limit: int = 20,
) -> dict[str, Any]:
    """Compare baseline and mined expansion tables on held-out explicit definitions.

    When a terminology repository is supplied, only held-out expansions resolving to exactly one
    concept become retrieval cases. This avoids inventing gold codes from ambiguous terminology.
    """

    if retrieval_limit < 1:
        raise ValueError("retrieval_limit must be positive")
    selected_splits = set(evaluation_splits)
    evaluation = [row for row in definitions if str(row.get("split", "")) in selected_splits]
    unique_evaluation = list(
        {
            (
                str(row["normalized_abbreviation"]),
                str(row["normalized_expansion"]),
            ): row
            for row in evaluation
        }.values()
    )
    baseline = _abbreviation_mapping(baseline_rows)
    enriched = _merge_abbreviation_mappings(baseline, _abbreviation_mapping(mined_rows))
    report: dict[str, Any] = {
        "schema_version": "abbreviation-benchmark.v1",
        "evaluation_splits": list(evaluation_splits),
        "evaluation_definition_count": len(evaluation),
        "evaluation_unique_pair_count": len(unique_evaluation),
        "baseline": _expansion_metrics(evaluation, baseline),
        "enriched": _expansion_metrics(evaluation, enriched),
        "unique_pairs": {
            "baseline": _expansion_metrics(unique_evaluation, baseline),
            "enriched": _expansion_metrics(unique_evaluation, enriched),
        },
    }
    if repository is not None:
        report["retrieval"] = _retrieval_metrics(
            evaluation,
            baseline,
            enriched,
            repository,
            limit=retrieval_limit,
        )
    return report


def build_runtime_abbreviation_table(
    baseline_rows: Sequence[Mapping[str, Any]],
    mined_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Merge reviewed and mined tables deterministically without losing ambiguity."""

    grouped: dict[str, dict[str, Any]] = {}
    for source_layer, rows in (("baseline", baseline_rows), ("mined", mined_rows)):
        for row in rows:
            surface = str(row.get("abbreviation", "")).strip()
            expansions = row.get("expansions", ())
            if (
                not surface
                or not isinstance(expansions, Sequence)
                or isinstance(expansions, (str, bytes))
            ):
                continue
            key = normalize_for_match(surface)
            target = grouped.setdefault(
                key,
                {
                    "abbreviation": surface,
                    "normalized_abbreviation": key,
                    "expansions": {},
                    "source_layers": set(),
                },
            )
            target["source_layers"].add(source_layer)
            for expansion_value in expansions:
                expansion = str(expansion_value).strip()
                if expansion:
                    target["expansions"].setdefault(normalize_for_match(expansion), expansion)

    output: list[dict[str, Any]] = []
    for key, value in sorted(grouped.items()):
        expansions = value["expansions"]
        if not expansions:
            continue
        output.append(
            {
                "abbreviation": value["abbreviation"],
                "normalized_abbreviation": key,
                "expansions": [expansions[name] for name in sorted(expansions)],
                "normalized_expansions": sorted(expansions),
                "source_layers": sorted(value["source_layers"]),
            }
        )
    return tuple(output)


def _document_rejection_reason(
    document: MinedDocument,
    artifact: SourceArtifact | None,
    split: str | None,
    accepted_splits: set[str],
    policy: AbbreviationMiningPolicy,
) -> str | None:
    if artifact is None:
        return "unknown_source_artifact"
    if artifact.source_id not in policy.accepted_source_ids:
        return "source_not_allowed"
    if artifact.source_version not in policy.accepted_source_versions:
        return "source_version_not_allowed"
    if artifact.object.sha256 not in policy.accepted_source_sha256:
        return "source_fingerprint_not_allowed"
    if document.language not in policy.accepted_languages:
        return "language_not_allowed"
    if split is None:
        return "missing_split_assignment"
    if split not in accepted_splits:
        return "split_not_selected"
    return None


def _definition_from_parenthetical(
    document: MinedDocument,
    artifact: SourceArtifact,
    split: str,
    parenthetical: re.Match[str],
    policy: AbbreviationMiningPolicy,
) -> tuple[_Definition | None, str]:
    inner_start, inner_end = parenthetical.span(1)
    inner = document.text[inner_start:inner_end].strip()
    inner_trim = len(document.text[inner_start:inner_end]) - len(
        document.text[inner_start:inner_end].lstrip()
    )
    inner_start += inner_trim
    inner_end = inner_start + len(inner)

    if _is_short_form(inner, policy):
        candidate_start, candidate = _preceding_long_candidate(
            document.text,
            parenthetical.start(),
            short_form=inner,
            policy=policy,
        )
        selected = _select_long_form(inner, candidate)
        if selected is None:
            return None, "long_short_alignment_failed"
        expansion, relative_start, alignment_method = selected
        shape_reason = _long_form_rejection_reason(inner, expansion, policy)
        if shape_reason is not None:
            return None, shape_reason
        expansion_start = candidate_start + relative_start
        return (
            _make_definition(
                document,
                artifact,
                split,
                abbreviation=inner,
                expansion=expansion,
                abbreviation_span=(inner_start, inner_end),
                expansion_span=(expansion_start, expansion_start + len(expansion)),
                direction="long_short",
                alignment_method=alignment_method,
                policy=policy,
            ),
            "accepted_long_short",
        )

    preceding = document.text[max(0, parenthetical.start() - 32) : parenthetical.start()]
    token_match = _PRECEDING_TOKEN_RE.search(preceding)
    if token_match is None:
        return None, "no_short_form"
    abbreviation = token_match.group(1)
    if not _is_short_form(abbreviation, policy):
        return None, "invalid_short_form"
    selected = _select_long_form(abbreviation, inner)
    if selected is None:
        return None, "short_long_alignment_failed"
    expansion, relative_start, alignment_method = selected
    shape_reason = _long_form_rejection_reason(abbreviation, expansion, policy)
    if shape_reason is not None:
        return None, shape_reason
    abbreviation_start = parenthetical.start() - len(preceding) + token_match.start(1)
    expansion_start = inner_start + relative_start
    return (
        _make_definition(
            document,
            artifact,
            split,
            abbreviation=abbreviation,
            expansion=expansion,
            abbreviation_span=(abbreviation_start, abbreviation_start + len(abbreviation)),
            expansion_span=(expansion_start, expansion_start + len(expansion)),
            direction="short_long",
            alignment_method=alignment_method,
            policy=policy,
        ),
        "accepted_short_long",
    )


def _preceding_long_candidate(
    text: str,
    parenthetical_start: int,
    *,
    short_form: str,
    policy: AbbreviationMiningPolicy,
) -> tuple[int, str]:
    max_window = min(policy.max_long_characters * 2, 512)
    window_start = max(0, parenthetical_start - max_window)
    window = text[window_start:parenthetical_start].rstrip()
    boundaries = list(_CLAUSE_BOUNDARY_RE.finditer(window))
    if boundaries:
        boundary_end = boundaries[-1].end()
        suffix = window[boundary_end:]
        leading_whitespace = len(suffix) - len(suffix.lstrip())
        window_start += boundary_end + leading_whitespace
        window = suffix.lstrip()
    words = list(re.finditer(r"\S+", window))
    alphanumeric_count = sum(character.isalnum() for character in short_form)
    max_words = min(
        policy.max_long_words,
        max(alphanumeric_count + 5, alphanumeric_count * 2),
    )
    if len(words) > max_words:
        relative_start = words[-max_words].start()
        window_start += relative_start
        window = window[relative_start:]
    return window_start, window.strip()


def _select_long_form(short_form: str, candidate: str) -> tuple[str, int, str] | None:
    """Return the shortest candidate suffix aligning every short-form character backward."""

    if not candidate:
        return None
    initialism_match = _select_initialism_long_form(short_form, candidate)
    if initialism_match is not None:
        return (*initialism_match, "token_initials")
    short_index = len(short_form) - 1
    long_index = len(candidate) - 1
    while short_index >= 0:
        short_character = short_form[short_index].casefold()
        if not short_character.isalnum():
            short_index -= 1
            continue
        while True:
            while long_index >= 0 and candidate[long_index].casefold() != short_character:
                long_index -= 1
            if long_index < 0:
                return None
            # The first abbreviation character must start a token. Rewinding an in-word match to
            # the token start created truncated expansions such as ``Research Center Inventories``
            # for ARCI even when ``Addiction`` was present immediately before it.
            if short_index == 0 and long_index > 0 and candidate[long_index - 1].isalnum():
                long_index -= 1
                continue
            break
        short_index -= 1
        long_index -= 1
    start = long_index + 1
    while start < len(candidate) and candidate[start].isspace():
        start += 1
    selected = candidate[start:].strip(" \t,;:-")
    if not selected:
        return None
    selected_start = candidate.find(selected, start)
    return selected, selected_start, "backward_characters"


def _select_initialism_long_form(short_form: str, candidate: str) -> tuple[str, int] | None:
    """Prefer exact token initials before permissive character-level alignment.

    This recovers full forms such as ``Proton Pump Inhibitors`` for ``PPIs``. The fallback remains
    necessary for valid forms where one word contributes multiple letters, such as ``NAAED``.
    """

    characters = [character.casefold() for character in short_form if character.isalnum()]
    if (
        len(characters) > 2
        and short_form.endswith("s")
        and any(character.isupper() for character in short_form[:-1])
    ):
        characters.pop()
    if len(characters) < 2:
        return None
    expected = "".join(characters)
    tokens = list(_WORD_RE.finditer(candidate))
    for start_index in range(len(tokens) - 1, -1, -1):
        suffix = tokens[start_index:]
        content_suffix = [
            token for token in suffix if token.group().casefold() not in _INITIALISM_STOPWORDS
        ]
        initial_variants = {
            "".join(token.group()[0].casefold() for token in suffix),
            "".join(_token_initials(token.group()) for token in suffix),
            "".join(token.group()[0].casefold() for token in content_suffix),
            "".join(_token_initials(token.group()) for token in content_suffix),
        }
        if expected not in initial_variants:
            continue
        start = suffix[0].start()
        selected = candidate[start:].strip(" \t,;:-")
        return selected, candidate.find(selected, start)
    return None


def _token_initials(token: str) -> str:
    """Return initials for a word, including meaningful hyphen/slash components."""

    return "".join(component[0].casefold() for component in re.split(r"[-/]", token) if component)


def _is_short_form(value: str, policy: AbbreviationMiningPolicy) -> bool:
    if not policy.min_short_characters <= len(value) <= policy.max_short_characters:
        return False
    if any(character.isspace() for character in value):
        return False
    letters = [character for character in value if character.isalpha()]
    if not letters or not any(character.isupper() for character in letters):
        return False
    # Common title-cased words inside parentheses are not abbreviations. Keep mixed forms such as
    # pH/Cr and acronym plurals such as AEDs while rejecting forms such as ``Oral``.
    uppercase_count = sum(character.isupper() for character in letters)
    if len(letters) > 2 and uppercase_count < 2:
        return False
    if not value[0].isalnum() or not value[-1].isalnum():
        return False
    if policy.reject_pure_roman_numerals and _ROMAN_NUMERAL_RE.fullmatch(value):
        return False
    return all(character.isalnum() or character in "+./-" for character in value)


def _long_form_rejection_reason(
    short_form: str,
    expansion: str,
    policy: AbbreviationMiningPolicy,
) -> str | None:
    """Return a stable rejection reason for aligned but implausible expansions."""

    if len(expansion) > policy.max_long_characters:
        return "long_form_too_long"
    if len(expansion.split()) > policy.max_long_words:
        return "long_form_too_many_words"
    if any(character in expansion for character in "()[]{}"):
        return "long_form_contains_bracket"
    if normalize_for_match(short_form) == normalize_for_match(expansion):
        return "short_form_equals_long_form"
    first_token = _WORD_RE.search(expansion)
    if first_token is not None:
        first_component = re.split(r"[-/]", first_token.group(), maxsplit=1)[0]
        if normalize_for_match(first_component) == normalize_for_match(short_form):
            return "long_form_starts_with_short_form"
    short_size = sum(character.isalnum() for character in short_form)
    expansion_size = sum(character.isalnum() for character in expansion)
    if expansion_size <= short_size:
        return "long_form_not_longer_than_short_form"
    return None


def _make_definition(
    document: MinedDocument,
    artifact: SourceArtifact,
    split: str,
    *,
    abbreviation: str,
    expansion: str,
    abbreviation_span: tuple[int, int],
    expansion_span: tuple[int, int],
    direction: str,
    alignment_method: str,
    policy: AbbreviationMiningPolicy,
) -> _Definition:
    # INVARIANT: mined spans always point into immutable source text, even though normalized
    # strings are used for conflict detection and retrieval.
    if document.text[abbreviation_span[0] : abbreviation_span[1]] != abbreviation:
        raise ValueError(f"Abbreviation offset mismatch in {document.document_id}")
    if document.text[expansion_span[0] : expansion_span[1]] != expansion:
        raise ValueError(f"Expansion offset mismatch in {document.document_id}")
    if len(expansion) > policy.max_long_characters:
        raise ValueError("Aligned expansion exceeds policy maximum")
    if len(expansion.split()) > policy.max_long_words:
        raise ValueError("Aligned expansion exceeds policy word maximum")
    identity = "\x1f".join(
        (
            policy.policy_id,
            document.document_id,
            str(abbreviation_span[0]),
            str(expansion_span[0]),
            normalize_for_match(abbreviation),
            normalize_for_match(expansion),
        )
    )
    return _Definition(
        definition_id=f"abbr-definition:{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
        document_id=document.document_id,
        split=split,
        abbreviation=abbreviation,
        expansion=expansion,
        abbreviation_span=abbreviation_span,
        expansion_span=expansion_span,
        direction=direction,
        alignment_method=alignment_method,
        group_id=min(document.group_ids) if document.group_ids else document.document_id,
        source_id=artifact.source_id,
        source_version=artifact.source_version,
        source_sha256=artifact.object.sha256,
    )


def _aggregate_definitions(
    definitions: Sequence[_Definition],
    policy: AbbreviationMiningPolicy,
    *,
    base_abbreviation_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    Counter[str],
]:
    knowledge = [value for value in definitions if value.split in policy.knowledge_splits]
    grouped: dict[tuple[str, str], list[_Definition]] = defaultdict(list)
    expansions_by_abbreviation: dict[str, set[str]] = defaultdict(set)
    for definition in knowledge:
        key = (definition.normalized_abbreviation, definition.normalized_expansion)
        grouped[key].append(definition)
        expansions_by_abbreviation[key[0]].add(key[1])
    base = _abbreviation_mapping(base_abbreviation_rows)
    candidates: list[dict[str, Any]] = []
    table: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for abbreviation in sorted(expansions_by_abbreviation):
        expansion_keys = sorted(expansions_by_abbreviation[abbreviation])
        if len(expansion_keys) != 1:
            conflict = {
                "normalized_abbreviation": abbreviation,
                "reason": "mined_expansion_conflict",
                "expansions": [
                    _expansion_summary(grouped[(abbreviation, expansion)], policy)
                    for expansion in expansion_keys
                ],
            }
            conflicts.append(conflict)
            reason_counts["mined_expansion_conflict"] += 1
            for expansion in expansion_keys:
                candidates.append(
                    _candidate_row(
                        grouped[(abbreviation, expansion)],
                        policy,
                        decision="rejected",
                        reason="mined_expansion_conflict",
                    )
                )
            continue

        values = grouped[(abbreviation, expansion_keys[0])]
        document_count = len({value.document_id for value in values})
        group_count = len({value.group_id for value in values})
        character_alignment_only = all(
            value.alignment_method == "backward_characters" for value in values
        )
        normalized_expansion = expansion_keys[0]
        if document_count < policy.min_supporting_documents:
            reason = "insufficient_document_support"
        elif group_count < policy.min_supporting_groups:
            reason = "insufficient_group_support"
        elif character_alignment_only and document_count < policy.min_character_alignment_documents:
            reason = "insufficient_character_alignment_document_support"
        elif character_alignment_only and group_count < policy.min_character_alignment_groups:
            reason = "insufficient_character_alignment_group_support"
        elif abbreviation in base and normalized_expansion not in base[abbreviation]:
            reason = "base_table_conflict"
        elif abbreviation in base:
            reason = "already_present"
        else:
            reason = "promoted"
        decision = (
            "promoted"
            if reason == "promoted"
            else ("skipped" if reason == "already_present" else "rejected")
        )
        candidates.append(_candidate_row(values, policy, decision=decision, reason=reason))
        reason_counts[reason] += 1
        if reason == "base_table_conflict":
            conflicts.append(
                {
                    "normalized_abbreviation": abbreviation,
                    "reason": reason,
                    "base_expansions": sorted(base[abbreviation]),
                    "expansions": [_expansion_summary(values, policy)],
                }
            )
        if reason != "promoted":
            continue
        surfaces = Counter(value.abbreviation for value in values)
        expansions = Counter(value.expansion for value in values)
        table.append(
            {
                "abbreviation": _preferred_surface(surfaces),
                "expansions": [_preferred_surface(expansions)],
                "normalized_abbreviation": abbreviation,
                "normalized_expansions": [normalized_expansion],
                "supporting_document_count": document_count,
                "supporting_group_count": group_count,
                "occurrence_count": len(values),
                "alignment_method_counts": dict(
                    sorted(Counter(value.alignment_method for value in values).items())
                ),
                "definition_ids": sorted(value.definition_id for value in values)[
                    : policy.max_examples
                ],
                "policy_id": policy.policy_id,
                "source_ids": sorted({value.source_id for value in values}),
                "source_versions": sorted({value.source_version for value in values}),
                "source_sha256": sorted({value.source_sha256 for value in values}),
            }
        )
    return (
        tuple(sorted(candidates, key=lambda row: str(row["normalized_abbreviation"]))),
        tuple(sorted(table, key=lambda row: str(row["normalized_abbreviation"]))),
        tuple(sorted(conflicts, key=lambda row: str(row["normalized_abbreviation"]))),
        reason_counts,
    )


def _candidate_row(
    values: Sequence[_Definition],
    policy: AbbreviationMiningPolicy,
    *,
    decision: str,
    reason: str,
) -> dict[str, Any]:
    summary = _expansion_summary(values, policy)
    return {**summary, "decision": decision, "reason": reason}


def _expansion_summary(
    values: Sequence[_Definition], policy: AbbreviationMiningPolicy
) -> dict[str, Any]:
    abbreviations = Counter(value.abbreviation for value in values)
    expansions = Counter(value.expansion for value in values)
    alignment_methods = Counter(value.alignment_method for value in values)
    return {
        "candidate_id": "abbreviation-candidate:"
        + hashlib.sha256(
            "\x1f".join(
                (
                    policy.policy_id,
                    values[0].normalized_abbreviation,
                    values[0].normalized_expansion,
                )
            ).encode()
        ).hexdigest()[:24],
        "abbreviation": _preferred_surface(abbreviations),
        "normalized_abbreviation": values[0].normalized_abbreviation,
        "expansion": _preferred_surface(expansions),
        "normalized_expansion": values[0].normalized_expansion,
        "supporting_document_count": len({value.document_id for value in values}),
        "supporting_group_count": len({value.group_id for value in values}),
        "occurrence_count": len(values),
        "alignment_method_counts": dict(sorted(alignment_methods.items())),
        "definition_ids": sorted(value.definition_id for value in values)[: policy.max_examples],
        "policy_id": policy.policy_id,
    }


def _preferred_surface(counts: Counter[str]) -> str:
    return min(counts, key=lambda value: (-counts[value], len(value), value.casefold(), value))


def _abbreviation_mapping(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    output: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        abbreviation = str(row.get("abbreviation", "")).strip()
        expansions = row.get("expansions", [])
        if (
            not abbreviation
            or not isinstance(expansions, Sequence)
            or isinstance(expansions, (str, bytes))
        ):
            continue
        key = normalize_for_match(abbreviation)
        output[key].update(
            normalize_for_match(str(value)) for value in expansions if str(value).strip()
        )
    return {key: tuple(sorted(values)) for key, values in sorted(output.items())}


def _merge_abbreviation_mappings(
    left: Mapping[str, Sequence[str]], right: Mapping[str, Sequence[str]]
) -> dict[str, tuple[str, ...]]:
    keys = set(left) | set(right)
    return {
        key: tuple(sorted(set(left.get(key, ())) | set(right.get(key, ())))) for key in sorted(keys)
    }


def _expansion_metrics(
    definitions: Sequence[Mapping[str, Any]], table: Mapping[str, Sequence[str]]
) -> dict[str, float | int]:
    known = 0
    exact = 0
    for definition in definitions:
        abbreviation = str(definition["normalized_abbreviation"])
        expansion = str(definition["normalized_expansion"])
        candidates = table.get(abbreviation, ())
        known += bool(candidates)
        exact += expansion in candidates
    total = len(definitions)
    return {
        "known_abbreviation_count": known,
        "exact_expansion_count": exact,
        "coverage": known / total if total else 0.0,
        "exact_recall": exact / total if total else 0.0,
        "conditional_accuracy": exact / known if known else 0.0,
    }


def _retrieval_metrics(
    definitions: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Sequence[str]],
    enriched: Mapping[str, Sequence[str]],
    repository: TerminologyRepository,
    *,
    limit: int,
) -> dict[str, Any]:
    cases: list[tuple[Mapping[str, Any], ConceptEntry]] = []
    ambiguous_gold_count = 0
    seen: set[tuple[str, str]] = set()
    for definition in definitions:
        key = (
            str(definition["normalized_abbreviation"]),
            str(definition["normalized_expansion"]),
        )
        if key in seen:
            continue
        seen.add(key)
        concepts = repository.exact_lookup(key[1], limit=100)
        concepts_by_id = {concept.concept_id: concept for concept in concepts}
        eligible = [
            concept
            for concept in concepts_by_id.values()
            if concept.code is not None and concept.code_system != CodeSystem.NONE
        ]
        if len(eligible) != 1:
            ambiguous_gold_count += 1
            continue
        cases.append((definition, eligible[0]))

    baseline_ranks = [
        _retrieval_rank(definition, concept, baseline, repository, limit=limit)
        for definition, concept in cases
    ]
    enriched_ranks = [
        _retrieval_rank(definition, concept, enriched, repository, limit=limit)
        for definition, concept in cases
    ]
    case_rows = [
        {
            "abbreviation": str(definition["abbreviation"]),
            "normalized_abbreviation": str(definition["normalized_abbreviation"]),
            "expansion": str(definition["expansion"]),
            "expected_concept_id": concept.concept_id,
            "expected_code": concept.code,
            "expected_code_system": concept.code_system.value,
            "expected_semantic_type": concept.semantic_type.value,
            "baseline_rank": baseline_rank,
            "enriched_rank": enriched_rank,
        }
        for (definition, concept), baseline_rank, enriched_rank in zip(
            cases, baseline_ranks, enriched_ranks, strict=True
        )
    ]
    return {
        "resolvable_case_count": len(cases),
        "ambiguous_or_missing_gold_count": ambiguous_gold_count,
        "improved_case_count": sum(
            enriched_rank is not None and (baseline_rank is None or enriched_rank < baseline_rank)
            for baseline_rank, enriched_rank in zip(baseline_ranks, enriched_ranks, strict=True)
        ),
        "regressed_case_count": sum(
            baseline_rank is not None and (enriched_rank is None or enriched_rank > baseline_rank)
            for baseline_rank, enriched_rank in zip(baseline_ranks, enriched_ranks, strict=True)
        ),
        "baseline": _rank_metrics(baseline_ranks),
        "enriched": _rank_metrics(enriched_ranks),
        "cases": case_rows,
    }


def _retrieval_rank(
    definition: Mapping[str, Any],
    expected: ConceptEntry,
    table: Mapping[str, Sequence[str]],
    repository: TerminologyRepository,
    *,
    limit: int,
) -> int | None:
    abbreviation = str(definition["normalized_abbreviation"])
    candidates: list[Any] = []
    # Exact expansion evidence is ranked before broad lexical search. The expansion itself comes
    # only from the knowledge split or the reviewed baseline table, never from held-out text.
    for expansion in table.get(abbreviation, ()):
        candidates.extend(
            repository.exact_lookup(
                expansion,
                entity_type=expected.semantic_type,
                code_systems=(expected.code_system,),
                limit=limit,
            )
        )
    candidates.extend(
        repository.search(
            str(definition["abbreviation"]),
            entity_type=expected.semantic_type,
            code_systems=(expected.code_system,),
            limit=limit,
        )
    )
    ordered: list[str] = []
    seen_concept_ids: set[str] = set()
    for candidate in candidates:
        if candidate.concept_id not in seen_concept_ids:
            ordered.append(candidate.concept_id)
            seen_concept_ids.add(candidate.concept_id)
        if len(ordered) >= limit:
            break
    try:
        return ordered.index(expected.concept_id) + 1
    except ValueError:
        return None


def _rank_metrics(ranks: Sequence[int | None]) -> dict[str, float | int]:
    total = len(ranks)
    found = [rank for rank in ranks if rank is not None]
    return {
        "hit_at_1": sum(rank <= 1 for rank in found) / total if total else 0.0,
        "recall_at_5": sum(rank <= 5 for rank in found) / total if total else 0.0,
        "recall_at_10": sum(rank <= 10 for rank in found) / total if total else 0.0,
        "recall_at_20": sum(rank <= 20 for rank in found) / total if total else 0.0,
        "mrr": sum(1.0 / rank for rank in found) / total if total else 0.0,
        "matched_count": len(found),
        "missing_count": total - len(found),
    }


def _required_string(raw: Mapping[str, Any], key: str) -> str:
    value = str(raw.get(key, "")).strip()
    if not value:
        raise ValueError(f"Abbreviation policy requires {key}")
    return value


def _string_tuple(raw: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Abbreviation policy {key} must be an array")
    result = tuple(str(item).strip() for item in value)
    if not result or any(not item for item in result):
        raise ValueError(f"Abbreviation policy {key} must contain non-empty values")
    return result
