"""Offline parser fixtures for structured mining sources."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from medical_kg_nlp.mining.parsers import (
    BiocJsonParser,
    ClinicalTrialsJsonParser,
    CodiEspArchiveParser,
    FhirBundleParser,
    JatsXmlParser,
    PlainTextArchiveParser,
    PmcOaParser,
    SplXmlParser,
)
from medical_kg_nlp.mining.labelers import (
    ClinicalTrialsStructuredLabelerAdapter,
    ClinicalTrialsStructuredRelationLabelerAdapter,
)
from medical_kg_nlp.mining.records import (
    AccessClass,
    RedistributionPolicy,
    SourceArtifact,
)
from medical_kg_nlp.mining.storage import LocalArtifactStore


def _artifact(
    tmp_path: Path,
    payload: bytes,
    *,
    source_id: str,
    media_type: str,
    metadata: dict[str, str] | None = None,
) -> tuple[SourceArtifact, LocalArtifactStore]:
    store = LocalArtifactStore(tmp_path / source_id)
    stored = store.put_stream(io.BytesIO(payload), metadata={})
    return (
        SourceArtifact(
            artifact_id=f"{source_id}:fixture",
            source_id=source_id,
            source_version="fixture-v1",
            source_uri=f"memory://{source_id}",
            object=stored,
            media_type=media_type,
            license_id="fixture-license",
            access_class=AccessClass.OPEN,
            redistribution=RedistributionPolicy.ALLOWED,
            hosted_processing_allowed=True,
            retrieved_at="2026-07-16T00:00:00+00:00",
            metadata=metadata or {},
        ),
        store,
    )


def test_jats_parser_renders_article_sections(tmp_path: Path) -> None:
    payload = b"""<article xml:lang="en"><front><article-meta>
      <article-id pub-id-type="pmc">PMC42</article-id>
      <title-group><article-title>Rare case</article-title></title-group>
      <abstract><p>Patient had <italic>fever</italic>.</p></abstract>
    </article-meta></front><body><sec><title>Case</title><p>A finding.</p></sec></body></article>"""
    artifact, store = _artifact(
        tmp_path, payload, source_id="pmc_oa", media_type="application/xml"
    )

    document = next(iter(JatsXmlParser().parse(artifact, store=store)))

    assert document.text == "Rare case\n\nPatient had fever.\n\nCase\n\nA finding."
    assert document.group_ids == ("article:PMC42",)
    assert document.metadata["parser_revision"] == "2"
    assert document.metadata["source_block_format"] == "jats"
    blocks = json.loads(document.metadata["source_blocks"])
    assert [block["section_path"] for block in blocks] == [
        [],
        ["Abstract"],
        ["Case"],
        ["Case"],
    ]
    assert [document.text[start:end] for start, end in (block["span"] for block in blocks)] == [
        "Rare case",
        "Patient had fever.",
        "Case",
        "A finding.",
    ]


def test_spl_parser_renders_label_sections(tmp_path: Path) -> None:
    payload = b"""<document xmlns="urn:hl7-org:v3" xml:lang="en">
      <setId root="set-42"/><title>Example drug</title><component><structuredBody>
      <component><section><title>Warnings</title><text>May cause nausea.</text></section></component>
      </structuredBody></component></document>"""
    artifact, store = _artifact(
        tmp_path, payload, source_id="dailymed", media_type="application/xml"
    )

    document = next(iter(SplXmlParser().parse(artifact, store=store)))

    assert document.text == "Example drug\n\nWarnings\nMay cause nausea."
    assert document.metadata["external_id"] == "set-42"


def test_clinical_trials_parser_renders_relation_fields(tmp_path: Path) -> None:
    payload = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT42", "briefTitle": "Trial title"},
            "descriptionModule": {"briefSummary": "A short summary."},
            "conditionsModule": {"conditions": ["Rare disease"]},
            "armsInterventionsModule": {
                "interventions": [
                    {"type": "DRUG", "name": "Drug A", "description": "Oral"}
                ]
            },
            "outcomesModule": {"primaryOutcomes": [{"measure": "Survival"}]},
        }
    }
    artifact, store = _artifact(
        tmp_path,
        json.dumps(payload).encode(),
        source_id="clinicaltrials_v2",
        media_type="application/json",
    )

    document = next(iter(ClinicalTrialsJsonParser().parse(artifact, store=store)))

    assert "Conditions\n- Rare disease" in document.text
    assert "Interventions\n- Drug A: Oral" in document.text
    assert "Primary outcomes\n- Survival" in document.text
    annotations = tuple(
        ClinicalTrialsStructuredLabelerAdapter(labeler_id="fixture@1").propose(
            (document,)
        )
    )
    relations = tuple(
        ClinicalTrialsStructuredRelationLabelerAdapter(
            labeler_id="fixture-relations@1"
        ).propose((document,), annotations)
    )

    assert [(item.text, item.entity_type) for item in annotations] == [
        ("Rare disease", "DISEASE"),
        ("Drug A", "DRUG"),
        ("Survival", "OTHER"),
    ]
    assert all(document.text[item.span[0] : item.span[1]] == item.text for item in annotations)
    assert [relation.relation_type for relation in relations] == [
        "STUDIES_INTERVENTION"
    ]


def test_fhir_parser_is_deterministic_and_patient_grouped(tmp_path: Path) -> None:
    payload = {
        "resourceType": "Bundle",
        "id": "bundle-42",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "patient-1"}},
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "condition-1",
                    "code": {"text": "Hypertension"},
                }
            },
        ],
    }
    artifact, store = _artifact(
        tmp_path,
        json.dumps(payload).encode(),
        source_id="synthea",
        media_type="application/fhir+json",
    )

    document = next(iter(FhirBundleParser().parse(artifact, store=store)))

    assert document.text.startswith("Condition/condition-1: Hypertension")
    assert document.group_ids == ("patient:patient-1",)


def test_bioc_parser_preserves_absolute_passage_offsets(tmp_path: Path) -> None:
    payload = {
        "documents": [
            {
                "id": "BioRED-42",
                "passages": [
                    {"offset": 0, "text": "Title"},
                    {"offset": 8, "text": "Disease and gene."},
                ],
            }
        ]
    }
    artifact, store = _artifact(
        tmp_path,
        json.dumps(payload).encode(),
        source_id="biored",
        media_type="application/json",
    )

    document = next(iter(BiocJsonParser().parse(artifact, store=store)))

    assert document.text[8:25] == "Disease and gene."
    assert document.text == "Title   Disease and gene."


def test_pmc_parser_accepts_real_bioc_collection_shape(tmp_path: Path) -> None:
    payload = [
        {
            "bioctype": "BioCCollection",
            "source": "PMC",
            "documents": [
                {
                    "id": "13373952",
                    "passages": [
                        {"offset": 0, "text": "Rare case"},
                        {"offset": 12, "text": "Clinical finding."},
                        {
                            "offset": 30,
                            "text": "",
                            "infons": {"section_type": "REF", "type": "ref"},
                        },
                    ],
                }
            ],
        }
    ]
    artifact, store = _artifact(
        tmp_path,
        json.dumps(payload).encode(),
        source_id="pmc_oa",
        media_type="application/json",
    )

    document = next(iter(PmcOaParser().parse(artifact, store=store)))

    assert document.text == "Rare case   Clinical finding."
    assert document.group_ids == ("article:13373952",)
    assert document.metadata["parser_id"] == "bioc_json"


def test_codiesp_parser_reads_notes_without_extracting_paths(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("root/train/text_files/2.txt", "Segundo caso")
        archive.writestr("root/train/text_files/1.txt", "Primer caso")
        archive.writestr("root/train/text_files_en/1.txt", "First case")
        archive.writestr("root/background/text_files/3.txt", "Caso de fondo")
    artifact, store = _artifact(
        tmp_path, buffer.getvalue(), source_id="codiesp", media_type="application/zip"
    )

    documents = list(CodiEspArchiveParser().parse(artifact, store=store))

    assert [document.metadata["external_id"] for document in documents] == [
        "train:1",
        "train:2",
        "background:3",
    ]
    assert [document.text for document in documents] == [
        "Primer caso",
        "Segundo caso",
        "Caso de fondo",
    ]
    assert all(document.language == "es" for document in documents)


def test_plain_text_archive_preserves_bytes_newlines_and_numeric_identity(
    tmp_path: Path,
) -> None:
    first = "\ufeffDòng một\r\nDòng hai\r\n".encode()
    second = "Sốt cao\n".encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("input/2.txt", second)
        archive.writestr("input/1.txt", first)
        archive.writestr("input/README.md", "ignored")
    artifact, store = _artifact(
        tmp_path,
        buffer.getvalue(),
        source_id="phase1_round2_input",
        media_type="application/zip",
    )

    documents = list(PlainTextArchiveParser().parse(artifact, store=store))

    assert [item.metadata["source_document_id"] for item in documents] == ["1", "2"]
    assert documents[0].text.encode() == first
    assert documents[0].metadata == {
        "archive_member": "input/1.txt",
        "external_id": "1",
        "newline_normalization": "none",
        "parser_id": "plain_text_archive",
        "parser_revision": "1",
        "raw_byte_size": str(len(first)),
        "raw_bytes_sha256": documents[0].text_sha256,
        "raw_encoding": "utf-8",
        "source_archive_sha256": artifact.object.sha256,
        "source_document_id": "1",
        "source_unit_sha256": documents[0].text_sha256,
    }
    assert documents[0].group_ids == ("source_record:phase1_round2_input:1",)


@pytest.mark.parametrize(
    "unsafe_name",
    ("../1.txt", "/absolute/1.txt", "C:/input/1.txt", "input/../../1.txt"),
)
def test_plain_text_archive_rejects_unsafe_paths(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(unsafe_name, "Sốt")
    artifact, store = _artifact(
        tmp_path,
        buffer.getvalue(),
        source_id="phase1_round2_input",
        media_type="application/zip",
    )

    with pytest.raises(ValueError, match="Unsafe ZIP member path"):
        tuple(PlainTextArchiveParser().parse(artifact, store=store))


def test_plain_text_archive_rejects_duplicate_canonical_numeric_ids(
    tmp_path: Path,
) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("input/1.txt", "Sốt")
        archive.writestr("other/01.txt", "Ho")
    artifact, store = _artifact(
        tmp_path,
        buffer.getvalue(),
        source_id="phase1_round2_input",
        media_type="application/zip",
    )

    with pytest.raises(ValueError, match="Duplicate source document ID '1'"):
        tuple(PlainTextArchiveParser().parse(artifact, store=store))


def test_plain_text_archive_rejects_high_expansion_ratio(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("input/1.txt", "A" * 16_384)
    artifact, store = _artifact(
        tmp_path,
        buffer.getvalue(),
        source_id="phase1_round2_input",
        media_type="application/zip",
    )
    parser = PlainTextArchiveParser(
        max_member_bytes=32_768,
        max_total_uncompressed_bytes=32_768,
        max_compression_ratio=2.0,
    )

    with pytest.raises(ValueError, match="unsafe compression ratio"):
        tuple(parser.parse(artifact, store=store))
