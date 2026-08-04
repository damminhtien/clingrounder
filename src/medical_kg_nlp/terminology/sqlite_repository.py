"""Read-only, thread-local SQLite terminology repository."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Sequence
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote

from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.preprocessing.normalizer import NORMALIZATION_CONTRACT_VERSION
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.terminology.index_builder import (
    TERMINOLOGY_INDEX_SCHEMA_VERSION,
    input_fingerprint,
    source_fingerprint,
)
from medical_kg_nlp.terminology.ports import TerminologySearchHit
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = ["SQLiteTerminologyRepository"]

_CONCEPT_COLUMNS = (
    "c.concept_id, c.code, c.code_system, c.canonical_name, c.semantic_type, "
    "c.tty, c.parent_code, c.parents_json, c.aliases_json, c.synonyms_json, "
    "c.abbreviations_json, c.blocked_aliases_json, c.official_name_vi, "
    "c.official_name_en, c.source, c.rxnorm_id, c.ingredient, c.brand_name, "
    "c.generic_name, c.dose_form, c.strength"
)

# RxNorm TTY only breaks lexical ties. It must never outrank a better FTS match.
_TTY_RANK_SQL = """
CASE c.tty
    WHEN 'IN' THEN 0
    WHEN 'PIN' THEN 1
    WHEN 'MIN' THEN 2
    WHEN 'BN' THEN 3
    WHEN 'SCD' THEN 4
    WHEN 'SBD' THEN 5
    WHEN 'SCDF' THEN 6
    WHEN 'SBDF' THEN 7
    WHEN 'GPCK' THEN 8
    WHEN 'BPCK' THEN 9
    ELSE 20
END
"""

# Product rows repeat their ingredient as an alias. Prefer a concept's own label so
# an ingredient query returns the IN concept before every SCD/SBD containing it.
_ALIAS_SOURCE_RANK_SQL = """
CASE a.source
    WHEN 'canonical_name' THEN 0
    WHEN 'official_name_vi' THEN 1
    WHEN 'official_name_en' THEN 1
    WHEN 'alias' THEN 2
    WHEN 'synonym' THEN 2
    WHEN 'abbreviation' THEN 2
    WHEN 'generic_name' THEN 3
    WHEN 'brand_name' THEN 3
    WHEN 'ingredient' THEN 4
    ELSE 2
END
"""


class SQLiteTerminologyRepository:
    """Query a prebuilt index without loading the terminology release into RAM."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        expected_source_paths: tuple[str | Path, ...] | None = None,
        expected_alias_overlay_paths: tuple[str | Path, ...] | None = None,
        expected_normalization_version: str = NORMALIZATION_CONTRACT_VERSION,
    ) -> None:
        self.index_path = Path(index_path).resolve()
        if not self.index_path.is_file():
            raise FileNotFoundError(
                f"Terminology index does not exist: {self.index_path}. Build it explicitly first."
            )
        self._local = threading.local()
        self.metadata = self._load_metadata()
        self._validate_metadata(
            expected_source_paths,
            expected_alias_overlay_paths,
            expected_normalization_version,
        )

    def get_by_concept_id(self, concept_id: str) -> ConceptEntry | None:
        row = self._connection().execute(
            f"SELECT {_CONCEPT_COLUMNS} FROM concepts c WHERE c.concept_id = ?",
            (concept_id,),
        ).fetchone()
        return _entry_from_row(row) if row is not None else None

    def get_by_code(self, code_system: CodeSystem, code: str) -> ConceptEntry | None:
        row = self._connection().execute(
            f"""
            SELECT {_CONCEPT_COLUMNS}
            FROM concepts c
            WHERE c.code_system = ? AND c.code = ?
            ORDER BY c.concept_id
            LIMIT 1
            """,
            (code_system.value, code),
        ).fetchone()
        return _entry_from_row(row) if row is not None else None

    def contains(self, code_system: CodeSystem, code: str) -> bool:
        """Check the indexed release key without hydrating a complete concept row."""

        row = self._connection().execute(
            "SELECT 1 FROM concepts WHERE code_system = ? AND code = ? LIMIT 1",
            (code_system.value, code),
        ).fetchone()
        # SCALING: concepts_code_idx makes validation O(log n) per distinct code.
        return row is not None

    def exact_lookup(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        return self._alias_lookup(
            "a.normalized",
            normalize_for_match(mention),
            entity_type,
            code_systems,
            limit,
        )

    def toneless_lookup(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        return self._alias_lookup(
            "a.toneless",
            normalize_for_match(mention, strip_diacritics=True),
            entity_type,
            code_systems,
            limit,
        )

    def search(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[ConceptEntry]:
        return [
            hit.entry
            for hit in self.search_scored(
                mention,
                entity_type=entity_type,
                code_systems=code_systems,
                limit=limit,
            )
        ]

    def search_scored(
        self,
        mention: str,
        *,
        entity_type: EntityType | None = None,
        code_systems: Sequence[CodeSystem] | None = None,
        limit: int = 20,
    ) -> list[TerminologySearchHit]:
        _validate_limit(limit)
        normalized = normalize_for_match(mention)
        if len(normalized) < 3:
            return [
                TerminologySearchHit(
                    entry=entry,
                    score=1.0,
                    matched_alias=mention,
                    match_kind="exact",
                )
                for entry in self.exact_lookup(
                    mention,
                    entity_type=entity_type,
                    code_systems=code_systems,
                    limit=limit,
                )
            ]
        conditions, parameters = _concept_filters(entity_type, code_systems)
        phrase_query = _quoted_fts_term(normalized)
        output = self._fts_search_scored(
            normalized,
            phrase_query,
            match_kind="phrase",
            conditions=conditions,
            parameters=parameters,
            limit=limit,
        )
        seen = {hit.entry.concept_id for hit in output}
        token_and_query, token_or_query = _token_queries(normalized)
        fallbacks = (
            (token_and_query, "all_tokens", limit * 4),
            (token_or_query, "partial_tokens", limit * 2),
        )
        for fallback_query, match_kind, query_limit in fallbacks:
            if len(output) >= limit:
                break
            if fallback_query is None or fallback_query == phrase_query:
                continue
            # SCALING: strict phrase/AND hits stay first. The bounded OR query only fills an
            # incomplete top-k, improving unseen-surface recall without widening exact lookup.
            for hit in self._fts_search_scored(
                normalized,
                fallback_query,
                match_kind=match_kind,
                conditions=conditions,
                parameters=parameters,
                limit=query_limit,
            ):
                if hit.entry.concept_id in seen:
                    continue
                output.append(hit)
                seen.add(hit.entry.concept_id)
                if len(output) == limit:
                    break
        return output

    def _fts_search_scored(
        self,
        normalized_mention: str,
        match_query: str,
        *,
        match_kind: str,
        conditions: str,
        parameters: Sequence[str],
        limit: int,
    ) -> list[TerminologySearchHit]:
        sql = f"""
            SELECT {_CONCEPT_COLUMNS},
                   aliases_fts.surface AS matched_alias,
                   aliases_fts.normalized AS matched_normalized,
                   bm25(aliases_fts) AS lexical_rank
            FROM aliases_fts
            JOIN concepts c ON c.concept_id = aliases_fts.concept_id
            WHERE aliases_fts MATCH ? {conditions}
            ORDER BY lexical_rank, {_TTY_RANK_SQL}, c.code_system,
                     COALESCE(c.code, ''), c.concept_id
            LIMIT ?
        """
        rows = self._connection().execute(
            sql,
            (match_query, *parameters, max(limit, limit * 8)),
        )
        return _deduplicate_scored_rows(
            rows,
            normalized_mention=normalized_mention,
            match_kind=match_kind,
            limit=limit,
        )

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            del self._local.connection

    def _alias_lookup(
        self,
        column: str,
        value: str,
        entity_type: EntityType | None,
        code_systems: Sequence[CodeSystem] | None,
        limit: int,
    ) -> list[ConceptEntry]:
        _validate_limit(limit)
        conditions, parameters = _concept_filters(entity_type, code_systems)
        sql = f"""
            SELECT {_CONCEPT_COLUMNS}, MIN({_ALIAS_SOURCE_RANK_SQL}) AS alias_rank
            FROM aliases a
            JOIN concepts c ON c.concept_id = a.concept_id
            WHERE {column} = ? {conditions}
            GROUP BY c.concept_id
            ORDER BY alias_rank, {_TTY_RANK_SQL}, c.code_system,
                     COALESCE(c.code, ''), c.concept_id
            LIMIT ?
        """
        rows = self._connection().execute(sql, (value, *parameters, limit))
        return [_entry_from_row(row) for row in rows]

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            # SCALING: one immutable read connection per worker avoids a global lock and pool.
            uri = f"file:{quote(str(self.index_path))}?mode=ro&immutable=1"
            connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            self._local.connection = connection
        return connection

    def _load_metadata(self) -> dict[str, str]:
        uri = f"file:{quote(str(self.index_path))}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        try:
            return {
                str(key): str(value)
                for key, value in connection.execute("SELECT key, value FROM metadata")
            }
        except sqlite3.DatabaseError as error:
            raise ValueError(f"Invalid terminology index: {self.index_path}") from error
        finally:
            connection.close()

    def _validate_metadata(
        self,
        expected_source_paths: tuple[str | Path, ...] | None,
        expected_alias_overlay_paths: tuple[str | Path, ...] | None,
        expected_normalization_version: str,
    ) -> None:
        if self.metadata.get("schema_version") != TERMINOLOGY_INDEX_SCHEMA_VERSION:
            raise ValueError("Terminology index schema version is stale")
        if self.metadata.get("normalization_version") != expected_normalization_version:
            raise ValueError("Terminology index normalization contract is stale")
        if expected_source_paths is not None:
            current = source_fingerprint(expected_source_paths)
            if self.metadata.get("source_fingerprint") != current:
                raise ValueError("Terminology index source fingerprint is stale")
            overlays = expected_alias_overlay_paths or ()
            expected_input = input_fingerprint(expected_source_paths, overlays)
            if self.metadata.get("input_fingerprint") != expected_input:
                raise ValueError("Terminology index alias-overlay fingerprint is stale")
        elif expected_alias_overlay_paths is not None:
            raise ValueError(
                "Expected canonical source paths are required when validating alias overlays"
            )


def _concept_filters(
    entity_type: EntityType | None,
    code_systems: Sequence[CodeSystem] | None,
) -> tuple[str, list[str]]:
    clauses: list[str] = []
    parameters: list[str] = []
    if entity_type is not None:
        clauses.append("c.semantic_type = ?")
        parameters.append(entity_type.value)
    if code_systems is not None:
        systems = [code_system.value for code_system in code_systems]
        if not systems:
            return " AND 0", []
        clauses.append(f"c.code_system IN ({','.join('?' for _ in systems)})")
        parameters.extend(systems)
    return (" AND " + " AND ".join(clauses) if clauses else ""), parameters


def _validate_limit(limit: int) -> None:
    if limit < 1:
        raise ValueError("limit must be at least 1")


def _deduplicate_scored_rows(
    rows: sqlite3.Cursor,
    *,
    normalized_mention: str,
    match_kind: str,
    limit: int,
) -> list[TerminologySearchHit]:
    by_concept: dict[str, tuple[TerminologySearchHit, int]] = {}
    for row_number, row in enumerate(rows):
        concept_id = str(row["concept_id"])
        matched_alias = str(row["matched_alias"])
        score = _lexical_similarity(
            normalized_mention,
            str(row["matched_normalized"]),
        )
        hit = TerminologySearchHit(
            entry=_entry_from_row(row),
            score=score,
            matched_alias=matched_alias,
            match_kind=match_kind,
            lexical_rank=float(row["lexical_rank"]),
        )
        current = by_concept.get(concept_id)
        if current is None or _search_hit_order(hit, row_number) < _search_hit_order(
            current[0],
            current[1],
        ):
            by_concept[concept_id] = (hit, row_number)
    ordered = sorted(
        by_concept.values(),
        key=lambda item: _search_hit_order(item[0], item[1]),
    )
    return [hit for hit, _ in ordered[:limit]]


def _search_hit_order(
    hit: TerminologySearchHit,
    row_number: int,
) -> tuple[float, float, int, str, str]:
    return (
        -hit.score,
        hit.lexical_rank if hit.lexical_rank is not None else 0.0,
        row_number,
        hit.entry.code or "",
        hit.entry.concept_id,
    )


def _lexical_similarity(normalized_mention: str, normalized_alias: str) -> float:
    """Measure surface agreement without pretending FTS rank is a probability."""

    mention = normalize_for_match(normalized_mention, strip_diacritics=True)
    alias = normalize_for_match(normalized_alias, strip_diacritics=True)
    if not mention or not alias:
        return 0.0
    if mention == alias:
        return 1.0
    character_score = SequenceMatcher(a=mention, b=alias, autojunk=False).ratio()
    mention_tokens = set(mention.split())
    alias_tokens = set(alias.split())
    overlap = len(mention_tokens & alias_tokens)
    token_score = (
        (2.0 * overlap) / (len(mention_tokens) + len(alias_tokens))
        if mention_tokens and alias_tokens
        else 0.0
    )
    score = max(character_score, token_score)

    # INVARIANT: contradictory numeric subtypes or strengths must not qualify merely
    # because the surrounding disease/drug words are similar.
    mention_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", mention))
    alias_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", alias))
    if mention_numbers and alias_numbers and mention_numbers.isdisjoint(alias_numbers):
        score = min(score, 0.49)
    return min(1.0, max(0.0, score))


def _quoted_fts_term(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def _token_queries(value: str) -> tuple[str | None, str | None]:
    # FTS5's trigram tokenizer cannot use one- or two-character query terms. Keep only stable
    # alphanumeric terms and require at least two so an OR query cannot become a broad scan.
    raw_tokens = re.findall(r"[^\W_]+", value)
    # INVARIANT: a short number often distinguishes clinical subtypes (for example type 1/2).
    # Dropping it can rank a contradictory concept, so leave that query as an exact-only miss.
    if any(token.isdigit() and len(token) < 3 for token in raw_tokens):
        return None, None
    tokens = tuple(dict.fromkeys(token for token in raw_tokens if len(token) >= 3))
    if len(tokens) < 2:
        return None, None
    return (
        " AND ".join(_quoted_fts_term(token) for token in tokens),
        " OR ".join(_quoted_fts_term(token) for token in tokens),
    )


def _entry_from_row(row: sqlite3.Row) -> ConceptEntry:
    return ConceptEntry(
        concept_id=str(row["concept_id"]),
        code=str(row["code"]) if row["code"] is not None else None,
        code_system=CodeSystem(str(row["code_system"])),
        canonical_name=str(row["canonical_name"]),
        semantic_type=EntityType(str(row["semantic_type"])),
        aliases=_json_tuple(row["aliases_json"]),
        official_name_vi=_optional_string(row["official_name_vi"]),
        official_name_en=_optional_string(row["official_name_en"]),
        synonyms=_json_tuple(row["synonyms_json"]),
        abbreviations=_json_tuple(row["abbreviations_json"]),
        parents=_json_tuple(row["parents_json"]),
        parent_code=_optional_string(row["parent_code"]),
        source=str(row["source"]),
        rxnorm_id=_optional_string(row["rxnorm_id"]),
        ingredient=_optional_string(row["ingredient"]),
        brand_name=_optional_string(row["brand_name"]),
        generic_name=_optional_string(row["generic_name"]),
        dose_form=_optional_string(row["dose_form"]),
        rxnorm_tty=_optional_string(row["tty"]),
        strength=_optional_string(row["strength"]),
        blocked_aliases=_json_tuple(row["blocked_aliases_json"]),
    )


def _json_tuple(value: object) -> tuple[str, ...]:
    payload = json.loads(str(value))
    if not isinstance(payload, list):
        raise ValueError("Expected JSON array in terminology index")
    return tuple(str(item) for item in payload)


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None
