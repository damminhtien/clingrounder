"""Create split-frozen RxNorm alias proposals from exact DailyMed product links.

The upstream linker requires agreement between the official SPL/version mapping and RxNorm NDC
evidence. This module does not perform another terminology guess: it only aggregates those exact
links into the neutral alias-proposal contract consumed by the strict knowledge compiler.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from medical_kg_nlp.utils.text import normalize_for_match

__all__ = [
    "DAILYMED_PRODUCT_ALIAS_SOURCE",
    "DailyMedProductAliasResult",
    "build_dailymed_product_alias_proposals",
]

DAILYMED_PRODUCT_ALIAS_SOURCE = "DailyMed exact product identity"
_REQUIRED_EVIDENCE = "exact_set_version_ndc_intersection"


@dataclass(frozen=True)
class DailyMedProductAliasResult:
    """Split-specific proposals, link decisions, and aggregate audit metrics."""

    proposals: tuple[dict[str, Any], ...]
    decisions: tuple[dict[str, Any], ...]
    report: dict[str, Any]


@dataclass(frozen=True)
class _ProductLink:
    link_id: str
    document_id: str
    surface: str
    normalized_alias: str
    rxcui: str
    rxttys: tuple[str, ...]
    source_version: str


def build_dailymed_product_alias_proposals(
    links: Sequence[Mapping[str, Any]],
    *,
    selected_document_ids: frozenset[str],
    split_name: str,
    links_sha256: str,
    split_manifest_sha256: str,
) -> DailyMedProductAliasResult:
    """Aggregate one frozen split while preserving all target conflicts for audit.

    A normalized product surface may target several strengths and therefore several RxCUIs. The
    function deliberately emits each alias/code hypothesis; the generic knowledge compiler then
    rejects conflicting aliases instead of silently choosing one code.
    """

    if not selected_document_ids:
        raise ValueError("DailyMed alias proposal split cannot be empty")
    if not split_name.strip():
        raise ValueError("DailyMed alias proposal split name must be non-empty")
    _validate_sha256(links_sha256, "links_sha256")
    _validate_sha256(split_manifest_sha256, "split_manifest_sha256")

    selected = tuple(
        _decode_link(raw)
        for raw in links
        if str(raw.get("document_id", "")) in selected_document_ids
    )
    if not selected:
        raise ValueError(f"No DailyMed product links belong to split {split_name!r}")
    link_ids = [link.link_id for link in selected]
    if len(link_ids) != len(set(link_ids)):
        raise ValueError("DailyMed product alias input contains duplicate link IDs")

    grouped: dict[tuple[str, str], list[_ProductLink]] = defaultdict(list)
    targets_by_alias: dict[str, set[str]] = defaultdict(set)
    for link in selected:
        grouped[(link.normalized_alias, link.rxcui)].append(link)
        targets_by_alias[link.normalized_alias].add(link.rxcui)

    proposals: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for (normalized_alias, rxcui), members in sorted(grouped.items()):
        surfaces = Counter(member.surface for member in members)
        target_count = len(targets_by_alias[normalized_alias])
        proposal_id = _proposal_id(split_name, normalized_alias, rxcui)
        proposals.append(
            {
                "proposal_id": proposal_id,
                "normalized_alias": normalized_alias,
                "code_system": "RxNorm",
                "code": rxcui,
                "semantic_type": "DRUG",
                "source": DAILYMED_PRODUCT_ALIAS_SOURCE,
                "source_version": "+".join(
                    sorted({member.source_version for member in members})
                ),
                # INVARIANT: the derived exact-link manifest, rather than a mutable local path,
                # is the source identity accepted by the downstream promotion policy.
                "source_sha256": links_sha256,
                "review_status": "source_verified",
                "supporting_record_count": len(
                    {member.document_id for member in members}
                ),
                "occurrence_count": len(members),
                "surface_variants": [
                    {
                        "surface": surface,
                        "count": count,
                        "ttys": sorted(
                            {tty for member in members for tty in member.rxttys}
                        ),
                    }
                    for surface, count in sorted(
                        surfaces.items(),
                        key=lambda item: (-item[1], len(item[0]), item[0].casefold()),
                    )
                ],
                "source_annotation_ids": sorted(member.link_id for member in members),
                "source_split": split_name,
                "target_count_for_alias": target_count,
                "evidence": _REQUIRED_EVIDENCE,
            }
        )
        reason = "unique_alias_target" if target_count == 1 else "ambiguous_alias_target"
        for member in members:
            decisions.append(
                {
                    "link_id": member.link_id,
                    "document_id": member.document_id,
                    "normalized_alias": normalized_alias,
                    "rxcui": rxcui,
                    "decision": "proposed",
                    "reason": reason,
                    "proposal_id": proposal_id,
                    "target_count_for_alias": target_count,
                }
            )

    conflict_aliases = {
        alias: sorted(targets)
        for alias, targets in sorted(targets_by_alias.items())
        if len(targets) > 1
    }
    ordered_proposals = tuple(
        sorted(
            proposals,
            key=lambda row: (str(row["normalized_alias"]), str(row["code"])),
        )
    )
    ordered_decisions = tuple(
        sorted(decisions, key=lambda row: (str(row["link_id"]), str(row["rxcui"])))
    )
    return DailyMedProductAliasResult(
        proposals=ordered_proposals,
        decisions=ordered_decisions,
        report={
            "schema_version": "dailymed-product-alias-proposal-report.v1",
            "source": DAILYMED_PRODUCT_ALIAS_SOURCE,
            "split": split_name,
            "input_link_count": len(links),
            "selected_link_count": len(selected),
            "selected_document_count": len({link.document_id for link in selected}),
            "proposal_count": len(ordered_proposals),
            "normalized_alias_count": len(targets_by_alias),
            "unique_target_alias_count": sum(
                len(targets) == 1 for targets in targets_by_alias.values()
            ),
            "ambiguous_target_alias_count": len(conflict_aliases),
            "ambiguous_targets": conflict_aliases,
            "links_sha256": links_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "evidence_contract": _REQUIRED_EVIDENCE,
            "promotion_contract": (
                "split-frozen, two-source-exact-linked, conflict-preserving"
            ),
        },
    )


def _decode_link(raw: Mapping[str, Any]) -> _ProductLink:
    evidence = _required_string(raw, "evidence")
    if evidence != _REQUIRED_EVIDENCE:
        raise ValueError(f"Unsupported DailyMed product-link evidence: {evidence!r}")
    surface = _required_string(raw, "text")
    normalized_alias = normalize_for_match(surface)
    if not normalized_alias:
        raise ValueError("DailyMed product-link surface normalizes to an empty alias")
    raw_ttys = raw.get("rxttys")
    if not isinstance(raw_ttys, list) or not all(
        isinstance(value, str) and value.strip() for value in raw_ttys
    ):
        raise ValueError("DailyMed product link rxttys must be a string array")
    source_version = "+".join(
        (
            _required_string(raw, "dailymed_source_version"),
            _required_string(raw, "rxnorm_source_version"),
        )
    )
    return _ProductLink(
        link_id=_required_string(raw, "link_id"),
        document_id=_required_string(raw, "document_id"),
        surface=surface,
        normalized_alias=normalized_alias,
        rxcui=_required_string(raw, "rxcui"),
        rxttys=tuple(sorted(set(raw_ttys))),
        source_version=source_version,
    )


def _required_string(raw: Mapping[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"DailyMed product link {field!r} must be a non-empty string")
    return value.strip()


def _validate_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 value")


def _proposal_id(split: str, normalized_alias: str, rxcui: str) -> str:
    identity = f"{split}\0{normalized_alias}\0{rxcui}"
    return f"dailymed-product-alias:{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
