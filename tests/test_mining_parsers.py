"""Offline parser fixtures for structured mining sources."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from medical_kg_nlp.mining.parsers import (
    BiocJsonParser,
    ClinicalTrialsJsonParser,
    CodiEspArchiveParser,
    FhirBundleParser,
    JatsXmlParser,
    SplXmlParser,
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
                "interventions": [{"name": "Drug A", "description": "Oral"}]
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
