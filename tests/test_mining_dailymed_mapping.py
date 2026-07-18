"""Compilation tests for the full official DailyMed-RxNorm crosswalk."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from medical_kg_nlp.mining.mappings.dailymed_rxnorm import (
    DailyMedRxNormMappingRepository,
    audit_dailymed_rxnorm_mapping,
    compile_dailymed_rxnorm_mapping,
)
from medical_kg_nlp.mining.records import (
    AccessClass,
    RedistributionPolicy,
    SourceArtifact,
)
from medical_kg_nlp.mining.storage import LocalArtifactStore
from medical_kg_nlp.terminology import build_terminology_index


def test_dailymed_mapping_compiler_deduplicates_and_builds_read_only_index(
    tmp_path: Path,
) -> None:
    payload = _mapping_zip(
        [
            "set-a|1|100|Drug 100|SCD",
            "set-a|1|100|Drug 100|SCD",
            "set-a|1|100|Drug One Hundred|SY",
            "set-a|1|200|Brand 200|SBD",
            "set-b|2|300|Drug 300|PSN",
        ]
    )
    artifact = _artifact(tmp_path, payload)
    output = tmp_path / "mappings.jsonl"
    index = tmp_path / "mappings.sqlite3"
    report_path = tmp_path / "report.json"

    report = compile_dailymed_rxnorm_mapping(
        artifact,
        io.BytesIO(payload),
        output_path=output,
        index_path=index,
        report_path=report_path,
    )

    assert report["input_row_count"] == 5
    assert report["unique_source_row_count"] == 4
    assert report["duplicate_source_row_count"] == 1
    assert report["mapping_count"] == 3
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["rxstrings"] == [
        {"text": "Drug 100", "ttys": ["SCD"]},
        {"text": "Drug One Hundred", "ttys": ["SY"]},
    ]

    repository = DailyMedRxNormMappingRepository(
        index,
        expected_source_sha256=artifact.object.sha256,
    )
    try:
        concepts = repository.lookup("SET-A", 1)
        assert [concept.rxcui for concept in concepts] == ["100", "200"]
        assert concepts[0].rxttys == ("SCD", "SY")
    finally:
        repository.close()
    with pytest.raises(ValueError, match="fingerprint changed"):
        DailyMedRxNormMappingRepository(index, expected_source_sha256="0" * 64)


def test_dailymed_mapping_audit_excludes_unknown_codes_and_existing_aliases(
    tmp_path: Path,
) -> None:
    payload = _mapping_zip(
        [
            "set-a|1|100|Drug 100|SCD",
            "set-a|1|100|Drug One Hundred|SY",
            "set-a|1|200|Brand 200|SBD",
            "set-b|2|300|Unknown Drug|PSN",
        ]
    )
    artifact = _artifact(tmp_path, payload)
    mapping_index = tmp_path / "mappings.sqlite3"
    compile_dailymed_rxnorm_mapping(
        artifact,
        io.BytesIO(payload),
        output_path=tmp_path / "mappings.jsonl",
        index_path=mapping_index,
        report_path=tmp_path / "compile-report.json",
    )
    terminology_source = _terminology_source(tmp_path / "concepts.jsonl")
    terminology_index = tmp_path / "terminology.sqlite3"
    build_terminology_index((terminology_source,), output_path=terminology_index)

    proposals_path = tmp_path / "proposals.jsonl"
    report = audit_dailymed_rxnorm_mapping(
        mapping_index,
        terminology_index,
        proposals_path=proposals_path,
        report_path=tmp_path / "audit-report.json",
    )

    assert report["mapping_rxcui_count"] == 3
    assert report["known_rxcui_count"] == 2
    assert report["unknown_rxcui_count"] == 1
    assert report["mapping_alias_pair_count"] == 4
    assert report["existing_alias_pair_count"] == 2
    assert report["review_alias_proposal_count"] == 1
    proposals = [
        json.loads(line)
        for line in proposals_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [(row["code"], row["normalized_alias"]) for row in proposals] == [
        ("100", "drug one hundred")
    ]
    assert proposals[0]["review_status"] == "review_required"
    assert report["absent_code_samples"] == [
        {"rxcui": "300", "example_rxstring": "Unknown Drug"}
    ]


def _mapping_zip(rows: list[str]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "rxnorm_mappings.txt",
            "SETID|SPL_VERSION|RXCUI|RXSTRING|RXTTY\n" + "\n".join(rows) + "\n",
        )
    return output.getvalue()


def _artifact(tmp_path: Path, payload: bytes) -> SourceArtifact:
    store = LocalArtifactStore(tmp_path / "objects")
    stored = store.put_stream(io.BytesIO(payload), metadata={})
    return SourceArtifact(
        artifact_id="dailymed_rxnorm_mappings:fixture",
        source_id="dailymed_rxnorm_mappings",
        source_version="2026-07-17",
        source_uri="memory://rxnorm_mappings.zip",
        object=stored,
        media_type="application/zip",
        license_id="nlm_dailymed_terms",
        access_class=AccessClass.OPEN_WITH_TERMS,
        redistribution=RedistributionPolicy.ATTRIBUTION,
        hosted_processing_allowed=True,
        retrieved_at="2026-07-18T00:00:00+00:00",
    )


def _terminology_source(path: Path) -> Path:
    rows = [
        {
            "concept_id": "RX:100",
            "code": "100",
            "code_system": "RxNorm",
            "canonical_name": "Drug 100",
            "semantic_type": "DRUG",
            "rxnorm_tty": "SCD",
            "source": "test-rxnorm",
        },
        {
            "concept_id": "RX:200",
            "code": "200",
            "code_system": "RxNorm",
            "canonical_name": "Brand 200",
            "semantic_type": "DRUG",
            "rxnorm_tty": "SBD",
            "source": "test-rxnorm",
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path
