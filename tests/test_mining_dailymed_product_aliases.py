"""DailyMed exact product-link alias proposal tests."""

from __future__ import annotations

import pytest

from clingrounder.mining.dailymed_product_aliases import (
    DAILYMED_PRODUCT_ALIAS_SOURCE,
    build_dailymed_product_alias_proposals,
)

_SHA = "a" * 64


def _link(
    link_id: str,
    document_id: str,
    text: str,
    rxcui: str,
) -> dict[str, object]:
    return {
        "link_id": link_id,
        "document_id": document_id,
        "text": text,
        "rxcui": rxcui,
        "rxttys": ["SCD", "PSN"],
        "evidence": "exact_set_version_ndc_intersection",
        "dailymed_source_version": "daily-v1",
        "rxnorm_source_version": "rxnorm-v1",
    }


def test_product_alias_proposals_are_split_frozen_and_aggregate_support() -> None:
    links = (
        _link("link-1", "train-1", "Drug A", "100"),
        _link("link-2", "train-2", "DRUG A", "100"),
        _link("link-dev", "dev-1", "Drug B", "200"),
    )

    result = build_dailymed_product_alias_proposals(
        links,
        selected_document_ids=frozenset({"train-1", "train-2"}),
        split_name="train",
        links_sha256=_SHA,
        split_manifest_sha256=_SHA,
    )

    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal["normalized_alias"] == "drug a"
    assert proposal["code"] == "100"
    assert proposal["source"] == DAILYMED_PRODUCT_ALIAS_SOURCE
    assert proposal["supporting_record_count"] == 2
    assert result.report["selected_link_count"] == 2
    assert result.report["ambiguous_target_alias_count"] == 0


def test_product_alias_proposals_preserve_ambiguous_targets_for_compiler_rejection() -> None:
    links = (
        _link("link-1", "train-1", "Drug A", "100"),
        _link("link-2", "train-2", "Drug A", "200"),
    )

    result = build_dailymed_product_alias_proposals(
        links,
        selected_document_ids=frozenset({"train-1", "train-2"}),
        split_name="train",
        links_sha256=_SHA,
        split_manifest_sha256=_SHA,
    )

    assert [proposal["code"] for proposal in result.proposals] == ["100", "200"]
    assert result.report["ambiguous_target_alias_count"] == 1
    assert {row["reason"] for row in result.decisions} == {
        "ambiguous_alias_target"
    }


def test_product_alias_proposals_reject_non_intersection_evidence() -> None:
    link = _link("link-1", "train-1", "Drug A", "100")
    link["evidence"] = "ndc_only"

    with pytest.raises(ValueError, match="Unsupported DailyMed product-link evidence"):
        build_dailymed_product_alias_proposals(
            (link,),
            selected_document_ids=frozenset({"train-1"}),
            split_name="train",
            links_sha256=_SHA,
            split_manifest_sha256=_SHA,
        )
