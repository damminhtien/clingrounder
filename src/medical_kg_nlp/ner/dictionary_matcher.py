from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.text import normalize_for_match, strip_vietnamese_tones


_ALLOWED_NORMALIZED_CHAR_RE = re.compile(r"[\w\s/+.-]", flags=re.UNICODE)
_MIN_TONELESS_ALIAS_CHARS = 8


@dataclass(frozen=True)
class DictionaryMatch:
    span: tuple[int, int]
    text: str
    normalized_text: str
    alias: str
    entry: ConceptEntry
    match_kind: str
    priority: int


@dataclass(frozen=True)
class _AliasPayload:
    key: str
    alias: str
    entry: ConceptEntry
    match_kind: str
    priority: int


@dataclass
class _AutomatonNode:
    transitions: dict[str, int] = field(default_factory=dict)
    fail: int = 0
    outputs: list[_AliasPayload] = field(default_factory=list)


@dataclass(frozen=True)
class _NormalizedText:
    text: str
    original_offsets: tuple[int, ...]


class _AhoCorasick:
    def __init__(self, payloads: Iterable[_AliasPayload]) -> None:
        self.nodes = [_AutomatonNode()]
        for payload in payloads:
            self._add(payload)
        self._build_failures()

    def find(self, text: str) -> Iterable[tuple[int, int, _AliasPayload]]:
        state = 0
        for index, char in enumerate(text):
            while state and char not in self.nodes[state].transitions:
                state = self.nodes[state].fail
            state = self.nodes[state].transitions.get(char, 0)
            for payload in self.nodes[state].outputs:
                start = index - len(payload.key) + 1
                if start >= 0:
                    yield start, index + 1, payload

    def _add(self, payload: _AliasPayload) -> None:
        state = 0
        for char in payload.key:
            next_state = self.nodes[state].transitions.get(char)
            if next_state is None:
                next_state = len(self.nodes)
                self.nodes[state].transitions[char] = next_state
                self.nodes.append(_AutomatonNode())
            state = next_state
        self.nodes[state].outputs.append(payload)

    def _build_failures(self) -> None:
        queue: deque[int] = deque()
        for next_state in self.nodes[0].transitions.values():
            queue.append(next_state)
            self.nodes[next_state].fail = 0

        while queue:
            state = queue.popleft()
            for char, next_state in self.nodes[state].transitions.items():
                queue.append(next_state)
                fail_state = self.nodes[state].fail
                while fail_state and char not in self.nodes[fail_state].transitions:
                    fail_state = self.nodes[fail_state].fail
                self.nodes[next_state].fail = self.nodes[fail_state].transitions.get(char, 0)
                self.nodes[next_state].outputs.extend(
                    self.nodes[self.nodes[next_state].fail].outputs
                )


class DictionaryMatcher:
    def __init__(self, aliases: Iterable[tuple[str, ConceptEntry]]) -> None:
        exact_payloads: list[_AliasPayload] = []
        toneless_payloads: list[_AliasPayload] = []
        seen: set[tuple[str, str, str]] = set()
        for alias, entry in aliases:
            cleaned = " ".join(alias.split()).strip()
            if len(cleaned) < 2:
                continue
            exact_key = normalize_for_match(cleaned)
            toneless_key = normalize_for_match(cleaned, strip_diacritics=True)
            if exact_key:
                key = ("exact", exact_key, entry.concept_id)
                if key not in seen:
                    exact_payloads.append(
                        _AliasPayload(
                            key=exact_key,
                            alias=cleaned,
                            entry=entry,
                            match_kind="exact",
                            priority=0,
                        )
                    )
                    seen.add(key)
            if (
                toneless_key
                and toneless_key != exact_key
                and len(toneless_key) >= _MIN_TONELESS_ALIAS_CHARS
                and not _compact_alias_is_upper(cleaned)
            ):
                key = ("toneless", toneless_key, entry.concept_id)
                if key not in seen:
                    toneless_payloads.append(
                        _AliasPayload(
                            key=toneless_key,
                            alias=cleaned,
                            entry=entry,
                            match_kind="toneless",
                            priority=1,
                        )
                    )
                    seen.add(key)
        self._exact = _AhoCorasick(exact_payloads)
        self._toneless = _AhoCorasick(toneless_payloads)

    def find_candidates(
        self,
        text: str,
        *,
        require_boundaries: bool = True,
        entity_types: set[EntityType] | None = None,
        min_alias_chars: int = 2,
    ) -> list[DictionaryMatch]:
        exact_text = normalize_text_with_offsets(text)
        toneless_text = normalize_text_with_offsets(text, strip_diacritics=True)
        matches = [
            *self._matches_for_normalized_text(
                source_text=text,
                normalized_text=exact_text,
                automaton=self._exact,
                require_boundaries=require_boundaries,
                entity_types=entity_types,
                min_alias_chars=min_alias_chars,
            ),
            *self._matches_for_normalized_text(
                source_text=text,
                normalized_text=toneless_text,
                automaton=self._toneless,
                require_boundaries=require_boundaries,
                entity_types=entity_types,
                min_alias_chars=min_alias_chars,
            ),
        ]
        return _deduplicate_matches(matches)

    @staticmethod
    def resolve_longest(matches: Iterable[DictionaryMatch]) -> list[DictionaryMatch]:
        """Select the maximum-weight non-overlapping match set.

        A small per-entity utility lets two independently useful concepts beat
        one broad overlapping alias, while covered characters and exact-match
        priority preserve the usual longest-match behavior for nested aliases.
        """

        ordered = sorted(
            matches,
            key=lambda match: (
                match.span[1],
                match.span[0],
                match.priority,
                match.entry.concept_id,
            ),
        )
        if not ordered:
            return []
        ends = [match.span[1] for match in ordered]
        predecessors = [
            bisect_right(ends, match.span[0], hi=index) - 1 for index, match in enumerate(ordered)
        ]
        states: list[tuple[int, int, tuple[int, ...]]] = [(0, 0, ())]
        for index, match in enumerate(ordered):
            previous = states[predecessors[index] + 1]
            length = match.span[1] - match.span[0]
            weight = length + 4 + (2 if match.match_kind == "exact" else 0)
            include = (previous[0] + weight, previous[1] + length, (*previous[2], index))
            exclude = states[-1]
            include_key = (include[0], include[1], -len(include[2]))
            exclude_key = (exclude[0], exclude[1], -len(exclude[2]))
            states.append(include if include_key > exclude_key else exclude)
        selected = [ordered[index] for index in states[-1][2]]
        return sorted(
            selected, key=lambda match: (match.span[0], match.span[1], match.entry.concept_id)
        )

    def _matches_for_normalized_text(
        self,
        *,
        source_text: str,
        normalized_text: _NormalizedText,
        automaton: _AhoCorasick,
        require_boundaries: bool,
        entity_types: set[EntityType] | None,
        min_alias_chars: int,
    ) -> list[DictionaryMatch]:
        matches: list[DictionaryMatch] = []
        for start_norm, end_norm, payload in automaton.find(normalized_text.text):
            if len(payload.alias.strip()) < min_alias_chars:
                continue
            if entity_types is not None and payload.entry.semantic_type not in entity_types:
                continue
            span = _original_span(normalized_text, start_norm, end_norm)
            if span is None:
                continue
            span = _extend_trailing_alias_punctuation(payload.alias, source_text, span)
            if _crosses_disallowed_separator(payload.alias, source_text[span[0] : span[1]]):
                continue
            if require_boundaries and not _valid_alias_boundaries(payload.alias, source_text, span):
                continue
            mention = source_text[span[0] : span[1]]
            matches.append(
                DictionaryMatch(
                    span=span,
                    text=mention,
                    normalized_text=normalize_for_match(mention),
                    alias=payload.alias,
                    entry=payload.entry,
                    match_kind=payload.match_kind,
                    priority=payload.priority,
                )
            )
        return matches


def normalize_text_with_offsets(text: str, *, strip_diacritics: bool = False) -> _NormalizedText:
    chars: list[str] = []
    offsets: list[int] = []
    previous_was_space = False
    for original_index, char in enumerate(text):
        normalized_char = unicodedata.normalize("NFKC", char).lower()
        if strip_diacritics:
            normalized_char = strip_vietnamese_tones(normalized_char)
        for unit in normalized_char:
            output = unit if _ALLOWED_NORMALIZED_CHAR_RE.fullmatch(unit) else " "
            if output.isspace():
                if previous_was_space:
                    continue
                chars.append(" ")
                offsets.append(original_index)
                previous_was_space = True
                continue
            chars.append(output)
            offsets.append(original_index)
            previous_was_space = False
    return _NormalizedText(text="".join(chars), original_offsets=tuple(offsets))


def _original_span(
    normalized_text: _NormalizedText, start: int, end: int
) -> tuple[int, int] | None:
    if start < 0 or end <= start or end > len(normalized_text.original_offsets):
        return None
    start_original = normalized_text.original_offsets[start]
    end_original = normalized_text.original_offsets[end - 1] + 1
    if end_original <= start_original:
        return None
    return start_original, end_original


def _extend_trailing_alias_punctuation(
    alias: str, text: str, span: tuple[int, int]
) -> tuple[int, int]:
    end = span[1]
    stripped_alias = alias.rstrip()
    for punctuation in (")", "]"):
        if stripped_alias.endswith(punctuation) and end < len(text) and text[end] == punctuation:
            end += 1
    return span[0], end


def _crosses_disallowed_separator(alias: str, mention: str) -> bool:
    return any(
        separator in mention and separator not in alias for separator in (",", ";", "\n", "\r")
    )


def _deduplicate_matches(matches: Iterable[DictionaryMatch]) -> list[DictionaryMatch]:
    unique: dict[tuple[int, int, str], DictionaryMatch] = {}
    for match in sorted(matches, key=lambda item: (item.priority, item.span[0], item.span[1])):
        key = (match.span[0], match.span[1], match.entry.concept_id)
        unique.setdefault(key, match)
    return list(unique.values())


def _valid_alias_boundaries(alias: str, text: str, span: tuple[int, int]) -> bool:
    start, end = span
    if start > 0 and _is_word_char(text[start - 1]):
        return False
    if end >= len(text):
        return True
    next_char = text[end]
    if _compact_alias_is_upper(alias):
        return not _is_word_char(next_char)
    return not _is_word_char(next_char) or next_char.isupper()


def _compact_alias_is_upper(alias: str) -> bool:
    compact = re.sub(r"[\W_]+", "", alias, flags=re.UNICODE)
    return bool(compact) and compact.upper() == compact


def _is_word_char(char: str) -> bool:
    return char == "_" or char.isalnum()
