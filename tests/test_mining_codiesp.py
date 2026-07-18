"""CodiEsp source import tests for language, spans, codes, and source defects."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from medical_kg_nlp.mining.labelers import (
    CodiEspArchiveLabelerAdapter,
    CodiEspLabelMapping,
)
from medical_kg_nlp.mining.parsers import CodiEspArchiveParser
from medical_kg_nlp.mining.records import (
    AccessClass,
    RedistributionPolicy,
    ReviewStatus,
    SourceArtifact,
)
from medical_kg_nlp.mining.storage import LocalArtifactStore


def _fixture(tmp_path: Path) -> tuple[SourceArtifact, LocalArtifactStore]:
    text = "Dolor y fiebre. TAC torácico."
    rows = "\n".join(
        (
            "case-1\tDIAGNOSTICO\tr52\tDolor\t0 5",
            "case-1\tPROCEDIMIENTO\tbw24\ttorácico TAC\t16 16;20 28;16 19",
            # This punctuation mismatch exists in the real corpus too. The raw span
            # remains valid, but the row must be routed to source review.
            "case-1\tDIAGNOSTICO\tr50.9\tfiebre.\t8 14",
        )
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("root/train/text_files/case-1.txt", text)
        archive.writestr("root/train/text_files_en/case-1.txt", "Pain and fever.")
        archive.writestr("root/train/trainX.tsv", rows)
        archive.writestr("root/background/text_files/background-1.txt", "Sin etiqueta")

    store = LocalArtifactStore(tmp_path / "store")
    stored = store.put_stream(io.BytesIO(buffer.getvalue()), metadata={})
    artifact = SourceArtifact(
        artifact_id="codiesp:fixture",
        source_id="codiesp",
        source_version="zenodo-3837305",
        source_uri="memory://codiesp",
        object=stored,
        media_type="application/zip",
        license_id="CC-BY-4.0",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ATTRIBUTION,
        hosted_processing_allowed=True,
        retrieved_at="2026-07-18T00:00:00+00:00",
    )
    return artifact, store


def test_codiesp_parser_excludes_machine_translation_and_preserves_split(
    tmp_path: Path,
) -> None:
    artifact, store = _fixture(tmp_path)

    documents = tuple(CodiEspArchiveParser().parse(artifact, store=store))

    assert len(documents) == 2
    assert [document.text for document in documents] == [
        "Dolor y fiebre. TAC torácico.",
        "Sin etiqueta",
    ]
    assert documents[0].metadata["corpus_split"] == "train"
    assert documents[0].metadata["codiesp_case_id"] == "case-1"
    assert documents[0].metadata["annotation_member"] == "root/train/trainX.tsv"
    assert documents[1].metadata["corpus_split"] == "background"
    assert "annotation_member" not in documents[1].metadata


def test_codiesp_labeler_preserves_codes_segments_and_review_issues(
    tmp_path: Path,
) -> None:
    artifact, store = _fixture(tmp_path)
    documents = tuple(CodiEspArchiveParser().parse(artifact, store=store))
    labeler = CodiEspArchiveLabelerAdapter(
        artifacts=(artifact,),
        store=store,
        label_map={
            "DIAGNOSTICO": CodiEspLabelMapping("FINDING", "ICD-10-CM"),
            "PROCEDIMIENTO": CodiEspLabelMapping("PROCEDURE", "ICD-10-PCS"),
        },
        labeler_id="codiesp-fixture",
        terminology_version="codiesp-v1.4-source-cie10",
    )

    proposals = tuple(labeler.propose(documents))

    assert len(proposals) == 3
    diagnosis = next(proposal for proposal in proposals if proposal.text == "Dolor")
    assert diagnosis.entity_type == "FINDING"
    assert diagnosis.concepts[0].code == "R52"
    assert diagnosis.concepts[0].code_system == "ICD-10-CM"
    procedure = next(proposal for proposal in proposals if proposal.source_label == "PROCEDIMIENTO")
    assert procedure.span == (16, 28)
    assert procedure.text == "TAC torácico"
    assert procedure.metadata["codiesp_raw_segments"] == "[[16,16],[20,28],[16,19]]"
    assert procedure.metadata["codiesp_segments"] == "[[20,28],[16,19]]"
    assert procedure.metadata["source_annotated_text"] == "torácico TAC"
    assert procedure.metadata["import_issues"] == '["zero_length_segment:16"]'
    assert procedure.review_status is ReviewStatus.NEEDS_REVIEW
    mismatch = next(proposal for proposal in proposals if proposal.text == "fiebre")
    assert mismatch.metadata["source_text_match"] == "false"
    assert mismatch.review_status is ReviewStatus.NEEDS_REVIEW
    assert len({proposal.annotation_id for proposal in proposals}) == 3
    source_document = documents[0]
    for proposal in proposals:
        proposal.validate_offsets(source_document)
