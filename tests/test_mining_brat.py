"""BRAT source import tests covering raw offsets and discontinuous spans."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from clingrounder.mining.labelers import BratArchiveLabelerAdapter
from clingrounder.mining.parsers import BratArchiveParser
from clingrounder.mining.records import (
    AccessClass,
    AnnotationLayer,
    RedistributionPolicy,
    ReviewStatus,
    SourceArtifact,
)
from clingrounder.mining.storage import LocalArtifactStore


def _brat_fixture(tmp_path: Path) -> tuple[SourceArtifact, LocalArtifactStore]:
    # Source bytes use CRLF while BRAT offsets follow browser-normalized LF text.
    text = "Bệnh lao\r\nphổi và GeneXpert."
    annotations = "\n".join(
        (
            "T1\tSymptom_and_Disease 0 8;9 13\tBệnh lao phổi",
            "T2\tDiagnosticProcedure 17 26\tGeneXpert",
            "T3\tAnnotatorNotes T2\tneeds review",
        )
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for annotator in ("Annotator_A", "Annotator_B"):
            archive.writestr(f"root/data_brat/{annotator}/dup_1.txt", text)
            archive.writestr(
                f"root/data_brat/{annotator}/dup_1.ann",
                annotations,
            )
        archive.writestr("root/data_supervised_learning/train.txt", "ignored")

    store = LocalArtifactStore(tmp_path / "store")
    stored = store.put_stream(io.BytesIO(buffer.getvalue()), metadata={})
    artifact = SourceArtifact(
        artifact_id="vietbioner:fixture",
        source_id="vietbioner",
        source_version="fixture-v1",
        source_uri="memory://vietbioner",
        object=stored,
        media_type="application/zip",
        license_id="CC-BY-4.0",
        access_class=AccessClass.OPEN,
        redistribution=RedistributionPolicy.ATTRIBUTION,
        hosted_processing_allowed=True,
        retrieved_at="2026-07-18T00:00:00+00:00",
    )
    return artifact, store


def test_brat_parser_pairs_only_annotated_text_and_groups_duplicates(
    tmp_path: Path,
) -> None:
    artifact, store = _brat_fixture(tmp_path)

    documents = tuple(
        BratArchiveParser(language="vi", note_type="biomedical_literature").parse(
            artifact,
            store=store,
        )
    )

    assert len(documents) == 2
    assert documents[0].text == "Bệnh lao\nphổi và GeneXpert."
    assert documents[0].group_ids == documents[1].group_ids
    assert documents[0].metadata["annotation_member"].endswith("dup_1.ann")
    assert documents[0].metadata["newline_normalization"] == "universal_lf"
    assert all(document.language == "vi" for document in documents)


def test_brat_labeler_preserves_discontinuous_segments_and_raw_envelope(
    tmp_path: Path,
) -> None:
    artifact, store = _brat_fixture(tmp_path)
    documents = tuple(BratArchiveParser(language="vi").parse(artifact, store=store))
    labeler = BratArchiveLabelerAdapter(
        artifacts=(artifact,),
        store=store,
        label_map={
            "Symptom_and_Disease": "FINDING",
            "DiagnosticProcedure": "PROCEDURE",
        },
        labeler_id="vietbioner-fixture",
    )

    proposals = tuple(labeler.propose(documents))

    assert len(proposals) == 4
    first = next(
        proposal
        for proposal in proposals
        if proposal.document_id == documents[0].document_id
        and proposal.source_label == "Symptom_and_Disease"
    )
    assert first.span == (0, 13)
    assert first.text == "Bệnh lao\nphổi"
    assert first.metadata["brat_segments"] == "[[0,8],[9,13]]"
    assert first.metadata["source_annotated_text"] == "Bệnh lao phổi"
    assert first.metadata["discontinuous"] == "true"
    assert first.layer is AnnotationLayer.SILVER
    assert first.review_status is ReviewStatus.PROPOSED
    assert len({proposal.annotation_id for proposal in proposals}) == 4
    for proposal in proposals:
        document = next(item for item in documents if item.document_id == proposal.document_id)
        proposal.validate_offsets(document)
