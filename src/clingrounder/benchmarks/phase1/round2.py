"""Audit Phase 1 Round 2 distribution without creating runtime annotation memory.

The audit compares corpus shape and duplicate evidence only. It deliberately emits no source
text, annotation span, entity type, or concept candidate, and the pipeline never imports this
benchmark-owned module.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from clingrounder.benchmarks.phase1.manual_gold import load_phase1_directory
from clingrounder.mining.dedup import StableTextDeduplicator
from clingrounder.mining.io import write_json, write_jsonl
from clingrounder.mining.profile import build_dataset_profile
from clingrounder.mining.records import (
    AccessClass,
    MinedDocument,
    RedistributionPolicy,
)
from clingrounder.schema.document import ClinicalDocument
from clingrounder.utils.hashing import sha256_file
from clingrounder.utils.io import read_source_text

__all__ = [
    "ROUND2_NOVELTY_SOURCE_IDS",
    "build_phase1_round2_audit",
    "load_phase1_round2_documents",
    "write_phase1_round2_audit",
]

ROUND2_NOVELTY_SOURCE_IDS = (
    "1",
    "24",
    "40",
    "48",
    "76",
    "79",
    "81",
    "83",
    "84",
    "94",
)

_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_CLINICAL_MARKERS = (
    "tiền sử bệnh",
    "lý do nhập viện",
    "đánh giá tại bệnh viện",
)
_QA_MARKERS = (
    "câu hỏi từ người dùng",
    "câu trả lời của bác sĩ",
    "hỏi :",
    "trả lời :",
)
_EDUCATIONAL_MARKERS = (
    " là gì?",
    "dấu hiệu của",
    "có nguy hiểm không?",
    "phòng ngừa ",
)


def load_phase1_round2_documents(
    documents: Sequence[MinedDocument],
    *,
    expected_archive_sha256: str,
    expected_count: int = 100,
) -> list[ClinicalDocument]:
    """Convert a private mined manifest into benchmark documents after provenance checks."""

    if not re.fullmatch(r"[0-9a-f]{64}", expected_archive_sha256):
        raise ValueError("Expected Round 2 archive SHA-256 must be 64 lowercase hex characters")
    by_source_id = _documents_by_numeric_source_id(documents)
    expected_ids = {str(index) for index in range(1, expected_count + 1)}
    if set(by_source_id) != expected_ids:
        missing = sorted(expected_ids - set(by_source_id), key=int)
        extra = sorted(set(by_source_id) - expected_ids, key=int)
        raise ValueError(f"Round 2 source IDs differ: missing={missing}, extra={extra}")
    archive_sha256 = _single_archive_sha256(documents)
    if archive_sha256 != expected_archive_sha256:
        raise ValueError(
            "Round 2 archive fingerprint differs: "
            f"expected {expected_archive_sha256}, observed {archive_sha256}"
        )

    output: list[ClinicalDocument] = []
    for source_id in sorted(by_source_id, key=int):
        document = by_source_id[source_id]
        if document.access_class is not AccessClass.AUTHORIZED_PRIVATE:
            raise ValueError(f"Round 2 document {source_id} is not authorized_private")
        if document.redistribution is not RedistributionPolicy.PROHIBITED:
            raise ValueError(f"Round 2 document {source_id} permits redistribution")
        if not document.hosted_processing_allowed:
            raise ValueError(
                f"Round 2 document {source_id} does not permit hosted processing"
            )
        raw_bytes_sha256 = document.metadata.get("raw_bytes_sha256")
        # INVARIANT: re-encoding must recover the exact imported UTF-8 bytes. This also proves
        # no newline normalization occurred between archive parsing and model inference.
        if raw_bytes_sha256 != document.text_sha256:
            raise ValueError(f"Round 2 raw-byte hash mismatch for document {source_id}")
        output.append(
            ClinicalDocument(
                document_id=source_id,
                text=document.text,
                metadata={
                    "source_manifest_document_id": document.document_id,
                    "source_archive_sha256": archive_sha256,
                    "archive_member": document.metadata.get("archive_member", ""),
                    "raw_bytes_sha256": raw_bytes_sha256,
                    "access_class": document.access_class.value,
                },
            )
        )
    return output


def build_phase1_round2_audit(
    documents: Sequence[MinedDocument],
    *,
    reference_input_dir: str | Path,
    reference_gold_dir: str | Path,
    reference_split_manifest: str | Path,
    novelty_source_ids: Sequence[str] = ROUND2_NOVELTY_SOURCE_IDS,
) -> dict[str, Any]:
    """Build profile, duplicate evidence, and an explicitly audit-only novelty queue."""

    by_source_id = _documents_by_numeric_source_id(documents)
    reference_texts = _load_reference_texts(reference_input_dir)
    split_manifest = json.loads(
        Path(reference_split_manifest).read_text(encoding="utf-8")
    )
    split_by_reference = {
        str(row["document_id"]): str(row["split"])
        for row in split_manifest.get("assignments", [])
    }
    if set(reference_texts) != set(split_by_reference):
        raise ValueError("Reference text IDs do not match the frozen split manifest")
    reference_gold = load_phase1_directory(reference_gold_dir)
    if set(reference_gold) != set(reference_texts):
        raise ValueError("Reference gold IDs do not match reference text IDs")

    profile = build_dataset_profile(documents)
    profile["round2"] = {
        "schema_version": "phase1-round2-profile.v1",
        "source_archive_sha256": _single_archive_sha256(documents),
        "source_document_id_range": [min(map(int, by_source_id)), max(map(int, by_source_id))],
        "style_counts": _style_counts(documents),
        "masked_document_count": sum("***" in document.text for document in documents),
        "bullet_document_count": sum(_has_bullet(document.text) for document in documents),
        "runtime_eligible": False,
    }

    duplicate_report = _duplicate_report(
        by_source_id,
        reference_texts,
        split_by_reference,
        reference_gold,
    )
    context_hits = duplicate_report["cross_round_context_overlap"]["by_source_document"]
    requested_novelty = tuple(str(value) for value in novelty_source_ids)
    unknown_novelty = sorted(set(requested_novelty) - set(by_source_id), key=int)
    if unknown_novelty:
        raise ValueError(f"Novelty queue references unknown Round 2 IDs: {unknown_novelty}")
    novelty_queue = tuple(
        {
            "source_document_id": source_id,
            "document_id": by_source_id[source_id].document_id,
            "character_count": len(by_source_id[source_id].text),
            "exact_reference_context_hits": int(context_hits[source_id]),
            "reason": "priority_novelty_review",
            "runtime_eligible": False,
        }
        for source_id in sorted(requested_novelty, key=int)
    )
    return {
        "schema_version": "phase1-round2-audit.v1",
        "profile": profile,
        "duplicate_report": duplicate_report,
        "novelty_queue": novelty_queue,
        "policy": {
            "purpose": "distribution_and_novelty_audit_only",
            "runtime_eligible": False,
            "contains_source_text": False,
            "contains_annotations": False,
            "contains_candidates": False,
        },
        "reference": {
            "corpus_fingerprint_sha256": str(
                split_manifest.get("corpus", {}).get("fingerprint_sha256", "")
            ),
            "document_count": len(reference_texts),
        },
    }


def write_phase1_round2_audit(
    audit: Mapping[str, Any],
    output_dir: str | Path,
    *,
    documents_manifest_path: str | Path,
) -> dict[str, Any]:
    """Write deterministic audit artifacts and a fingerprint-only manifest."""

    target = Path(output_dir)
    profile = _mapping(audit.get("profile"), "profile")
    duplicate_report = _mapping(audit.get("duplicate_report"), "duplicate_report")
    novelty_queue = audit.get("novelty_queue")
    if not isinstance(novelty_queue, Sequence) or isinstance(novelty_queue, (str, bytes)):
        raise ValueError("novelty_queue must be a sequence")
    profile_sha256 = write_json(target / "profile.json", profile)
    duplicate_sha256 = write_json(target / "duplicate_report.json", duplicate_report)
    novelty_sha256 = write_jsonl(
        target / "novelty_queue.jsonl",
        (_mapping(row, "novelty row") for row in novelty_queue),
    )
    manifest = {
        "schema_version": "phase1-round2-audit-manifest.v1",
        "runtime_eligible": False,
        "inputs": {
            "documents_sha256": sha256_file(documents_manifest_path),
            "source_archive_sha256": profile["round2"]["source_archive_sha256"],
            "reference_corpus_fingerprint_sha256": audit["reference"][
                "corpus_fingerprint_sha256"
            ],
        },
        "outputs": {
            "profile.json": profile_sha256,
            "duplicate_report.json": duplicate_sha256,
            "novelty_queue.jsonl": novelty_sha256,
        },
    }
    write_json(target / "audit_manifest.json", manifest)
    return manifest


def _documents_by_numeric_source_id(
    documents: Sequence[MinedDocument],
) -> dict[str, MinedDocument]:
    result: dict[str, MinedDocument] = {}
    for document in documents:
        source_id = document.metadata.get("source_document_id", "")
        if not source_id.isdigit() or str(int(source_id)) != source_id:
            raise ValueError(f"Round 2 document has invalid numeric source ID {source_id!r}")
        if source_id in result:
            raise ValueError(f"Duplicate Round 2 source document ID {source_id!r}")
        result[source_id] = document
    if not result:
        raise ValueError("Round 2 audit requires at least one document")
    return result


def _load_reference_texts(input_dir: str | Path) -> dict[str, str]:
    paths = tuple(Path(input_dir).glob("*.txt"))
    return {
        path.stem: read_source_text(path)
        for path in sorted(paths, key=lambda item: int(item.stem))
        if path.stem.isdigit()
    }


def _duplicate_report(
    documents: Mapping[str, MinedDocument],
    reference_texts: Mapping[str, str],
    split_by_reference: Mapping[str, str],
    reference_gold: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    round2_documents = tuple(documents.values())
    duplicate_groups = StableTextDeduplicator().describe_groups(round2_documents)
    internal_groups = [
        {
            "group_id": group.group_id,
            "kind": group.kind.value,
            "source_document_ids": sorted(
                (
                    _source_id_for_document_id(round2_documents, document_id)
                    for document_id in group.document_ids
                ),
                key=int,
            ),
        }
        for group in duplicate_groups
        if len(group.document_ids) > 1
    ]

    reference_shingles = {
        source_id: _word_shingles(text, size=8)
        for source_id, text in reference_texts.items()
    }
    reference_lines = {
        line.strip()
        for text in reference_texts.values()
        for line in text.splitlines()
        if line.strip()
    }
    cross_rows: list[dict[str, Any]] = []
    exact_line_documents = 0
    exact_shingle_documents = 0
    weighted_line_characters = 0
    weighted_total_characters = 0
    for source_id, document in sorted(documents.items(), key=lambda item: int(item[0])):
        shingles = _word_shingles(document.text, size=8)
        best_reference_id, best_score, best_overlap = _best_reference_match(
            shingles,
            reference_shingles,
        )
        nonempty_lines = [line.strip() for line in document.text.splitlines() if line.strip()]
        matched_line_characters = sum(
            len(line) for line in nonempty_lines if line in reference_lines
        )
        total_line_characters = sum(len(line) for line in nonempty_lines)
        weighted_line_characters += matched_line_characters
        weighted_total_characters += total_line_characters
        has_exact_line = matched_line_characters > 0
        has_exact_shingle = best_overlap > 0
        exact_line_documents += has_exact_line
        exact_shingle_documents += has_exact_shingle
        cross_rows.append(
            {
                "source_document_id": source_id,
                "best_reference_document_id": best_reference_id,
                "best_reference_split": split_by_reference[best_reference_id],
                "shingle_jaccard": round(best_score, 6),
                "shared_shingle_count": best_overlap,
                "exact_line_character_fraction": round(
                    matched_line_characters / max(total_line_characters, 1),
                    6,
                ),
            }
        )

    contexts = _reference_contexts(reference_texts, reference_gold, window=32)
    context_hits = {
        source_id: sum(context in document.text for context in contexts)
        for source_id, document in sorted(documents.items(), key=lambda item: int(item[0]))
    }
    zero_context_ids = sorted(
        (source_id for source_id, count in context_hits.items() if count == 0),
        key=int,
    )
    return {
        "schema_version": "phase1-round2-duplicate-audit.v1",
        "policy": {
            "purpose": "audit_only",
            "runtime_eligible": False,
            "annotation_transfer_permitted": False,
        },
        "within_round2": {
            "near_duplicate_group_count": len(internal_groups),
            "groups": internal_groups,
            "top_pairs": _top_internal_pairs(documents, minimum_jaccard=0.5, limit=25),
        },
        "cross_round": {
            "exact_line_document_count": exact_line_documents,
            "exact_shingle_document_count": exact_shingle_documents,
            "weighted_exact_line_character_fraction": round(
                weighted_line_characters / max(weighted_total_characters, 1),
                6,
            ),
            "strong_match_count_ge_0_25": sum(
                float(row["shingle_jaccard"]) >= 0.25 for row in cross_rows
            ),
            "very_strong_match_count_ge_0_50": sum(
                float(row["shingle_jaccard"]) >= 0.5 for row in cross_rows
            ),
            "by_source_document": cross_rows,
        },
        "cross_round_context_overlap": {
            "window_characters": 32,
            "reference_context_count": len(contexts),
            "zero_context_overlap_source_ids": zero_context_ids,
            "by_source_document": context_hits,
            "runtime_eligible": False,
        },
    }


def _single_archive_sha256(documents: Sequence[MinedDocument]) -> str:
    values = {document.metadata.get("source_archive_sha256", "") for document in documents}
    if len(values) != 1:
        raise ValueError("Round 2 documents must share one source archive SHA-256")
    value = next(iter(values))
    if len(value) != 64:
        raise ValueError("Round 2 source archive SHA-256 is invalid")
    return value


def _style_counts(documents: Sequence[MinedDocument]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for document in documents:
        normalized = document.text.casefold()
        clinical = any(marker in normalized for marker in _CLINICAL_MARKERS)
        qa = any(marker in normalized for marker in _QA_MARKERS)
        educational = any(marker in normalized for marker in _EDUCATIONAL_MARKERS)
        labels = [
            name
            for name, present in (
                ("clinical", clinical),
                ("question_answer", qa),
                ("educational", educational),
            )
            if present
        ]
        counts["+".join(labels) if labels else "other"] += 1
    return dict(sorted(counts.items()))


def _has_bullet(text: str) -> bool:
    return any(line.lstrip().startswith(("-", "•", "*")) for line in text.splitlines())


def _word_shingles(text: str, *, size: int) -> frozenset[str]:
    tokens = _TOKEN.findall(text.casefold())
    if len(tokens) < size:
        return frozenset(tokens)
    return frozenset(
        " ".join(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    )


def _best_reference_match(
    shingles: frozenset[str],
    references: Mapping[str, frozenset[str]],
) -> tuple[str, float, int]:
    best = ("", -1.0, -1)
    for source_id, candidate in references.items():
        overlap = len(shingles & candidate)
        union = len(shingles | candidate)
        score = overlap / union if union else 0.0
        rank = (score, overlap, -int(source_id))
        current = (best[1], best[2], -int(best[0])) if best[0] else (-1.0, -1, 0)
        if rank > current:
            best = (source_id, score, overlap)
    if not best[0]:
        raise ValueError("Reference corpus is empty")
    return best


def _top_internal_pairs(
    documents: Mapping[str, MinedDocument],
    *,
    minimum_jaccard: float,
    limit: int,
) -> list[dict[str, Any]]:
    shingles = {
        source_id: _word_shingles(document.text, size=8)
        for source_id, document in documents.items()
    }
    pairs: list[tuple[float, str, str, int]] = []
    ordered = sorted(shingles, key=int)
    for left_index, left_id in enumerate(ordered):
        for right_id in ordered[left_index + 1 :]:
            overlap = len(shingles[left_id] & shingles[right_id])
            union = len(shingles[left_id] | shingles[right_id])
            score = overlap / union if union else 0.0
            if score >= minimum_jaccard:
                pairs.append((score, left_id, right_id, overlap))
    return [
        {
            "left_source_document_id": left_id,
            "right_source_document_id": right_id,
            "shingle_jaccard": round(score, 6),
            "shared_shingle_count": overlap,
        }
        for score, left_id, right_id, overlap in sorted(
            pairs,
            key=lambda item: (-item[0], int(item[1]), int(item[2])),
        )[:limit]
    ]


def _reference_contexts(
    reference_texts: Mapping[str, str],
    reference_gold: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    window: int,
) -> frozenset[str]:
    contexts: set[str] = set()
    for source_id, rows in reference_gold.items():
        text = reference_texts[source_id]
        for row in rows:
            position = row.get("position")
            if not isinstance(position, list) or len(position) != 2:
                raise ValueError(f"Reference gold {source_id} has an invalid position")
            start, end = int(position[0]), int(position[1])
            context = text[max(0, start - window) : min(len(text), end + window)]
            if context:
                contexts.add(context)
    return frozenset(contexts)


def _source_id_for_document_id(
    documents: Sequence[MinedDocument],
    document_id: str,
) -> str:
    for document in documents:
        if document.document_id == document_id:
            return document.metadata["source_document_id"]
    raise KeyError(document_id)


def _mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return {str(key): item for key, item in value.items()}
