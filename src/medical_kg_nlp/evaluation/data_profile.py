from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, median
from typing import Any

from medical_kg_nlp.context.rules import (
    FAMILY_CUES,
    HISTORICAL_CUES,
    NEGATION_CUES,
    PLANNED_CUES,
    POSSIBLE_CUES,
    RESOLVED_CUES,
)
from medical_kg_nlp.datasets.synthetic_adapter import SyntheticDatasetAdapter
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.schema.annotation import EntityAnnotation
from medical_kg_nlp.schema.document import ClinicalDocument, Section
from medical_kg_nlp.schema.output import ClinicalPrediction
from medical_kg_nlp.preprocessing.section_splitter import split_sections
from medical_kg_nlp.preprocessing.sentence_splitter import split_sentences
from medical_kg_nlp.utils.text import normalize_for_match

_TOKEN_RE = re.compile(r"[\w.+/-]+", flags=re.UNICODE)
_UPPER_ACRONYM_RE = re.compile(r"^[A-ZĐ]{2,}[A-ZĐ0-9.+/-]*$")
_SHORT_CODELIKE_RE = re.compile(r"^(?=.*\d)[A-Za-zĐđ0-9.+/-]{2,12}$")


def profile_paths(
    documents_path: str | Path,
    gold_path: str | Path,
    *,
    dictionary_path: str | Path | None = None,
    reference_gold_path: str | Path | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    adapter = SyntheticDatasetAdapter()
    documents = adapter.load_documents(documents_path)
    gold = adapter.load_gold(gold_path)
    reference_gold = adapter.load_gold(reference_gold_path) if reference_gold_path else None
    dictionary = DictionaryStore.from_jsonl(dictionary_path) if dictionary_path else None
    return profile_dataset(
        documents=documents,
        gold=gold,
        dictionary=dictionary,
        reference_gold=reference_gold,
        top_k=top_k,
    )


def profile_dataset(
    *,
    documents: list[ClinicalDocument],
    gold: list[ClinicalPrediction],
    dictionary: DictionaryStore | None = None,
    reference_gold: list[ClinicalPrediction] | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    documents_by_id = {document.document_id: document for document in documents}
    gold_by_id = {prediction.document_id: prediction for prediction in gold}
    sections_by_document = _sections_by_document(documents)

    entities = [entity for prediction in gold for entity in prediction.entities]
    coded_entities = [entity for entity in entities if entity.code is not None]
    relations = [relation for prediction in gold for relation in prediction.relations]

    span_lengths = [entity.span[1] - entity.span[0] for entity in entities]
    note_lengths = [len(document.text) for document in documents]
    sentences_per_document = [
        sum(
            len(split_sentences(section.text, section_title=section.title, base_offset=section.span[0]))
            for section in sections
        )
        for sections in sections_by_document.values()
    ]
    missing_gold_documents = sorted(set(documents_by_id) - set(gold_by_id))
    missing_source_documents = sorted(set(gold_by_id) - set(documents_by_id))

    offset_issues = _offset_issues(documents_by_id, gold)
    dictionary_coverage = _dictionary_coverage(coded_entities, dictionary)
    code_overlap = _code_overlap(gold, reference_gold)

    return {
        "documents": {
            "count": len(documents),
            "missing_gold_documents": missing_gold_documents,
            "missing_source_documents": missing_source_documents,
            "note_length": _number_summary(note_lengths),
            "sentences_per_document": _number_summary(sentences_per_document),
            "section_titles": _counter_items(_section_title_counts(sections_by_document.values()), top_k),
        },
        "entities": {
            "count": len(entities),
            "coded_count": len(coded_entities),
            "by_type": _counter_items(Counter(entity.type.value for entity in entities), top_k),
            "by_assertion": _counter_items(Counter(entity.assertion.value for entity in entities), top_k),
            "by_code_system": _counter_items(Counter(entity.code_system.value for entity in entities), top_k),
            "span_length": _number_summary(span_lengths),
            "top_mentions": _counter_items(Counter(normalize_for_match(entity.text) for entity in entities), top_k),
            "top_codes": _counter_items(Counter(_code_key(entity) for entity in coded_entities), top_k),
            "singleton_codes": _singleton_codes(coded_entities),
            "mention_code_pairs": _mention_code_pairs(coded_entities, top_k),
            "ambiguous_mentions": _ambiguous_mentions(coded_entities, top_k),
            "abbreviation_like_mentions": _counter_items(_abbreviation_like_mentions(entities), top_k),
            "by_section": _counter_items(_entity_section_counts(gold, sections_by_document), top_k),
        },
        "context_cues": _context_cue_counts((document.text for document in documents), top_k),
        "relations": {
            "count": len(relations),
            "by_type": _counter_items(Counter(relation.type.value for relation in relations), top_k),
        },
        "dictionary_coverage": dictionary_coverage,
        "code_overlap": code_overlap,
        "offsets": {
            "issue_count": len(offset_issues),
            "issues": offset_issues[:top_k],
        },
    }


def render_markdown(profile: dict[str, Any]) -> str:
    documents = _mapping(profile["documents"])
    entities = _mapping(profile["entities"])
    relations = _mapping(profile["relations"])
    coverage = _mapping(profile["dictionary_coverage"])
    offsets = _mapping(profile["offsets"])

    lines = [
        "# Data Profile",
        "",
        "## Summary",
        "",
        f"- Documents: {documents['count']}",
        f"- Entities: {entities['count']}",
        f"- Coded entities: {entities['coded_count']}",
        f"- Relations: {relations['count']}",
        f"- Dictionary coverage: {coverage['coverage']}",
        f"- Offset issues: {offsets['issue_count']}",
        "",
        "## Top Entity Types",
        "",
        *_markdown_counter_table(_list(entities["by_type"])),
        "",
        "## Top Codes",
        "",
        *_markdown_counter_table(_list(entities["top_codes"])),
        "",
        "## Context Cues",
        "",
        *_markdown_counter_table(_list(profile["context_cues"])),
    ]
    return "\n".join(lines) + "\n"


def _sections_by_document(documents: list[ClinicalDocument]) -> dict[str, list[Section]]:
    return {document.document_id: split_sections(document.text) for document in documents}


def _section_title_counts(section_groups: Iterable[list[Section]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for sections in section_groups:
        counts.update(section.title for section in sections)
    return counts


def _entity_section_counts(
    gold: list[ClinicalPrediction],
    sections_by_document: dict[str, list[Section]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for prediction in gold:
        sections = sections_by_document.get(prediction.document_id, [])
        for entity in prediction.entities:
            counts[_section_title_for_entity(entity, sections)] += 1
    return counts


def _section_title_for_entity(entity: EntityAnnotation, sections: list[Section]) -> str:
    for section in sections:
        if section.span[0] <= entity.span[0] and entity.span[1] <= section.span[1]:
            return section.title
    return "UNKNOWN"


def _offset_issues(
    documents_by_id: dict[str, ClinicalDocument],
    gold: list[ClinicalPrediction],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for prediction in gold:
        document = documents_by_id.get(prediction.document_id)
        if document is None:
            continue
        for entity in prediction.entities:
            try:
                entity.validate_offsets(document.text)
            except ValueError as error:
                issues.append(
                    {
                        "document_id": prediction.document_id,
                        "entity_id": entity.id,
                        "message": str(error),
                    }
                )
    return issues


def _dictionary_coverage(
    coded_entities: list[EntityAnnotation],
    dictionary: DictionaryStore | None,
) -> dict[str, Any]:
    if dictionary is None:
        return {
            "checked": False,
            "coded_entities": len(coded_entities),
            "known_codes": 0,
            "unknown_codes": 0,
            "coverage": None,
            "unknown_code_items": [],
        }
    allowed_codes = {
        (entry.code_system.value, entry.code)
        for entry in dictionary.entries
        if entry.code is not None
    }
    unknown_counter: Counter[str] = Counter()
    known = 0
    for entity in coded_entities:
        key = (entity.code_system.value, entity.code)
        if key in allowed_codes:
            known += 1
        else:
            unknown_counter[_code_key(entity)] += 1
    total = len(coded_entities)
    return {
        "checked": True,
        "dictionary_codes": len(allowed_codes),
        "coded_entities": total,
        "known_codes": known,
        "unknown_codes": total - known,
        "coverage": round(known / total, 6) if total else 1.0,
        "unknown_code_items": _counter_items(unknown_counter, 20),
    }


def _code_overlap(
    gold: list[ClinicalPrediction],
    reference_gold: list[ClinicalPrediction] | None,
) -> dict[str, Any]:
    target_codes = {_code_key(entity) for prediction in gold for entity in prediction.entities if entity.code is not None}
    if reference_gold is None:
        return {
            "checked": False,
            "target_unique_codes": len(target_codes),
            "reference_unique_codes": 0,
            "unseen_codes": [],
            "unseen_code_count": 0,
        }
    reference_codes = {
        _code_key(entity)
        for prediction in reference_gold
        for entity in prediction.entities
        if entity.code is not None
    }
    unseen = sorted(target_codes - reference_codes)
    return {
        "checked": True,
        "target_unique_codes": len(target_codes),
        "reference_unique_codes": len(reference_codes),
        "unseen_code_count": len(unseen),
        "unseen_codes": unseen,
    }


def _context_cue_counts(texts: Iterable[str], top_k: int) -> list[dict[str, Any]]:
    cue_groups = {
        "possible": POSSIBLE_CUES,
        "negation": NEGATION_CUES,
        "historical": HISTORICAL_CUES,
        "family": FAMILY_CUES,
        "planned": PLANNED_CUES,
        "resolved": RESOLVED_CUES,
    }
    counts: Counter[str] = Counter()
    for text in texts:
        lowered = text.lower()
        for group, cues in cue_groups.items():
            for cue in cues:
                cue_text = cue.strip().lower()
                if cue_text and cue_text in lowered:
                    counts[f"{group}:{cue_text}"] += lowered.count(cue_text)
    return _counter_items(counts, top_k)


def _abbreviation_like_mentions(entities: list[EntityAnnotation]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for entity in entities:
        for token in _TOKEN_RE.findall(entity.text):
            if _is_abbreviation_like(token):
                counts[token] += 1
    return counts


def _is_abbreviation_like(token: str) -> bool:
    stripped = token.strip(".,;:()[]{}")
    if len(stripped) < 2 or len(stripped) > 12:
        return False
    if _UPPER_ACRONYM_RE.match(stripped):
        return True
    return bool(_SHORT_CODELIKE_RE.match(stripped))


def _mention_code_pairs(coded_entities: list[EntityAnnotation], top_k: int) -> list[dict[str, Any]]:
    pairs: Counter[tuple[str, str]] = Counter(
        (normalize_for_match(entity.text), _code_key(entity)) for entity in coded_entities
    )
    return [
        {"mention": mention, "code": code, "count": count}
        for (mention, code), count in pairs.most_common(top_k)
    ]


def _ambiguous_mentions(coded_entities: list[EntityAnnotation], top_k: int) -> list[dict[str, Any]]:
    codes_by_mention: dict[str, set[str]] = defaultdict(set)
    counts_by_mention: Counter[str] = Counter()
    for entity in coded_entities:
        mention = normalize_for_match(entity.text)
        codes_by_mention[mention].add(_code_key(entity))
        counts_by_mention[mention] += 1
    ambiguous = [
        (mention, sorted(codes), counts_by_mention[mention])
        for mention, codes in codes_by_mention.items()
        if len(codes) > 1
    ]
    ambiguous.sort(key=lambda item: (-item[2], item[0]))
    return [
        {"mention": mention, "codes": codes, "count": count}
        for mention, codes, count in ambiguous[:top_k]
    ]


def _singleton_codes(coded_entities: list[EntityAnnotation]) -> list[str]:
    counts = Counter(_code_key(entity) for entity in coded_entities)
    return sorted(code for code, count in counts.items() if count == 1)


def _code_key(entity: EntityAnnotation) -> str:
    return f"{entity.code_system.value}:{entity.code}"


def _number_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values), 6),
        "median": round(median(values), 6),
    }


def _counter_items(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    return [{"key": key, "count": count} for key, count in counter.most_common(limit)]


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Expected object-like JSON value.")
    return value


def _list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError("Expected array-like JSON value.")
    return value


def _markdown_counter_table(rows: list[Any]) -> list[str]:
    table = ["| Key | Count |", "| --- | ---: |"]
    for row in rows:
        item = _mapping(row)
        table.append(f"| {item['key']} | {item['count']} |")
    return table
