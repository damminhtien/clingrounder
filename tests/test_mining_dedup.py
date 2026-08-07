"""Duplicate grouping contracts for leakage-safe mined datasets."""

from __future__ import annotations

from clingrounder.mining.dedup import DuplicateGroupKind, StableTextDeduplicator
from clingrounder.mining.records import (
    AccessClass,
    MinedDocument,
    RedistributionPolicy,
)


def _document(document_id: str, text: str) -> MinedDocument:
    return MinedDocument(
        document_id=document_id,
        text=text,
        language="vi",
        note_type="clinical_note",
        source_artifact_id="fixture:artifact",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ALLOWED,
        hosted_processing_allowed=True,
    )


def test_dedup_report_distinguishes_safe_collapse_from_split_only_groups() -> None:
    documents = (
        _document("raw-a", "Sốt và ho."),
        _document("raw-b", "Sốt và ho."),
        _document("normalized-a", " Đau   đầu "),
        _document("normalized-b", "đau đầu"),
        _document("single", "Không đau ngực."),
    )

    groups = StableTextDeduplicator(hamming_threshold=0).describe_groups(documents)
    by_members = {group.document_ids: group for group in groups}

    assert by_members[("raw-a", "raw-b")].kind is DuplicateGroupKind.RAW_EXACT
    assert (
        by_members[("normalized-a", "normalized-b")].kind
        is DuplicateGroupKind.NORMALIZED_EXACT
    )
    assert by_members[("single",)].kind is DuplicateGroupKind.SINGLETON


def test_group_api_is_derived_from_auditable_group_records() -> None:
    documents = (
        _document("left", "đau đầu dữ dội kéo dài " * 5 + "ba ngày"),
        _document("right", "đau đầu dữ dội kéo dài " * 5 + "bốn ngày"),
    )
    deduplicator = StableTextDeduplicator(hamming_threshold=16, bands=4)

    assignments = deduplicator.group(documents)
    groups = deduplicator.describe_groups(documents)

    assert len(groups) == 1
    assert groups[0].kind is DuplicateGroupKind.NEAR
    assert assignments == {"left": groups[0].group_id, "right": groups[0].group_id}


def test_exact_mode_skips_near_matching_but_keeps_normalized_duplicates() -> None:
    documents = (
        _document("left", "đau đầu dữ dội kéo dài " * 5 + "ba ngày"),
        _document("right", "đau đầu dữ dội kéo dài " * 5 + "bốn ngày"),
        _document("normalized-a", " Sốt   và ho "),
        _document("normalized-b", "sốt và ho"),
    )

    groups = StableTextDeduplicator(include_near=False).describe_groups(documents)
    by_members = {group.document_ids: group for group in groups}

    assert by_members[("left",)].kind is DuplicateGroupKind.SINGLETON
    assert by_members[("right",)].kind is DuplicateGroupKind.SINGLETON
    assert (
        by_members[("normalized-a", "normalized-b")].kind
        is DuplicateGroupKind.NORMALIZED_EXACT
    )
