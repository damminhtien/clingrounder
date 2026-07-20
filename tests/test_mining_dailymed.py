"""DailyMed SPL projection tests for structured drug mining."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from medical_kg_nlp.mining.labelers.dailymed import (
    DailyMedStructuredLabelerAdapter,
    DailyMedStructuredRelationLabelerAdapter,
)
from medical_kg_nlp.mining.parsers.xml import SplXmlParser
from medical_kg_nlp.mining.records import (
    AccessClass,
    RedistributionPolicy,
    SourceArtifact,
)
from medical_kg_nlp.mining.storage import LocalArtifactStore


def test_spl_parser_and_labeler_project_structured_medication_fields(
    tmp_path: Path,
) -> None:
    artifact, store = _spl_artifact(tmp_path)

    documents = tuple(SplXmlParser().parse(artifact, store=store))

    assert [document.note_type for document in documents] == [
        "structured_product_label",
        "structured_medication_record",
    ]
    structured = documents[1]
    assert structured.text == (
        "Product: AMOXICILLIN\n"
        "Generic name: amoxicillin\n"
        "Active ingredient: AMOXICILLIN\n"
        "Strength: 500 mg\n"
        "Dosage form: CAPSULE\n"
        "Route: ORAL\n"
        "NDC: 00000-001"
    )
    assert structured.metadata["dailymed_spl_version"] == "7"
    assert structured.metadata["spl_ndc"] == "00000-001"

    proposals = tuple(
        DailyMedStructuredLabelerAdapter(labeler_id="dailymed-spl:v1").propose(documents)
    )

    assert [proposal.entity_type for proposal in proposals] == [
        "DRUG",
        "DRUG",
        "DRUG",
        "STRENGTH",
        "DOSAGE_FORM",
        "ROUTE",
    ]
    assert [proposal.source_label for proposal in proposals] == [
        "SPL_PRODUCT_NAME",
        "SPL_GENERIC_NAME",
        "SPL_ACTIVE_INGREDIENT",
        "SPL_INGREDIENT_STRENGTH",
        "SPL_DOSAGE_FORM",
        "SPL_ROUTE",
    ]
    assert [
        (concept.code_system, concept.code)
        for proposal in proposals
        for concept in proposal.concepts
    ] == [
        ("NDC", "00000-001"),
        ("UNII", "9EM05410Q9"),
        ("NCI_THESAURUS", "C25158"),
        ("NCI_THESAURUS", "C38288"),
    ]
    for proposal in proposals:
        proposal.validate_offsets(structured)

    relations = tuple(
        DailyMedStructuredRelationLabelerAdapter(
            labeler_id="dailymed-spl-relations:v1"
        ).propose(documents, proposals)
    )
    assert sorted(relation.relation_type for relation in relations) == [
        "HAS_ACTIVE_INGREDIENT",
        "HAS_DOSAGE_FORM",
        "HAS_GENERIC_NAME",
        "HAS_ROUTE",
        "HAS_STRENGTH",
    ]
    strength = next(
        relation for relation in relations if relation.relation_type == "HAS_STRENGTH"
    )
    by_id = {proposal.annotation_id: proposal for proposal in proposals}
    assert by_id[strength.head_annotation_id].source_label == "SPL_ACTIVE_INGREDIENT"
    assert by_id[strength.tail_annotation_id].source_label == "SPL_INGREDIENT_STRENGTH"


def test_spl_parser_streams_zip_members_and_skips_exact_duplicates(tmp_path: Path) -> None:
    first = _spl_payload(set_id="set-42")
    second = _spl_payload(set_id="set-43")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("labels/set-42.xml", first)
        output.writestr("labels/set-42-copy.xml", first)
        output.writestr("labels/set-43.xml", second)
        output.writestr("labels/image.jpg", b"not parsed")
    artifact, store = _artifact_from_payload(
        tmp_path,
        archive.getvalue(),
        artifact_id="dailymed:zip-fixture",
        source_uri="memory://dailymed/labels.zip",
        media_type="application/zip",
    )

    documents = tuple(SplXmlParser().parse(artifact, store=store))

    assert len(documents) == 4
    assert {document.metadata["dailymed_set_id"] for document in documents} == {
        "set-42",
        "set-43",
    }
    assert {
        document.metadata["archive_member"] for document in documents
    } == {"labels/set-42-copy.xml", "labels/set-43.xml"}


def test_spl_member_identity_is_independent_of_archive_wrapper(tmp_path: Path) -> None:
    payload = _spl_payload(set_id="set-42")
    direct, direct_store = _artifact_from_payload(
        tmp_path / "direct",
        payload,
        artifact_id="dailymed:direct",
        source_uri="memory://dailymed/set-42.xml",
        media_type="application/xml",
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w") as output:
        output.writestr("different/path/set-42.xml", payload)
    wrapped, wrapped_store = _artifact_from_payload(
        tmp_path / "wrapped",
        archive.getvalue(),
        artifact_id="dailymed:wrapped",
        source_uri="memory://dailymed/part-01.zip",
        media_type="application/zip",
    )

    direct_ids = [
        document.document_id for document in SplXmlParser().parse(direct, store=direct_store)
    ]
    wrapped_ids = [
        document.document_id
        for document in SplXmlParser().parse(wrapped, store=wrapped_store)
    ]

    assert direct_ids == wrapped_ids


def test_spl_parser_rejects_zip_without_xml(tmp_path: Path) -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w") as output:
        output.writestr("label.jpg", b"image")
    artifact, store = _artifact_from_payload(
        tmp_path,
        archive.getvalue(),
        artifact_id="dailymed:no-xml",
        source_uri="memory://dailymed/no-xml.zip",
        media_type="application/zip",
    )

    with pytest.raises(ValueError, match="contains no SPL XML member"):
        tuple(SplXmlParser().parse(artifact, store=store))


def test_spl_parser_rejects_oversized_xml_member(tmp_path: Path) -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w") as output:
        output.writestr("label.xml", _spl_payload(set_id="set-42"))
    artifact, store = _artifact_from_payload(
        tmp_path,
        archive.getvalue(),
        artifact_id="dailymed:oversized",
        source_uri="memory://dailymed/oversized.zip",
        media_type="application/zip",
    )
    parser = SplXmlParser()
    parser.max_xml_member_bytes = 32

    with pytest.raises(ValueError, match="exceeds size limit"):
        tuple(parser.parse(artifact, store=store))


def test_spl_parser_validates_source_published_label_count(tmp_path: Path) -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w") as output:
        output.writestr("label.xml", _spl_payload(set_id="set-42"))
    artifact, store = _artifact_from_payload(
        tmp_path,
        archive.getvalue(),
        artifact_id="dailymed:count-mismatch",
        source_uri="memory://dailymed/part-01.zip",
        media_type="application/zip",
        metadata={"expected_spl_count": "2"},
    )

    with pytest.raises(ValueError, match="SPL count mismatch"):
        tuple(SplXmlParser().parse(artifact, store=store))


def _spl_artifact(tmp_path: Path) -> tuple[SourceArtifact, LocalArtifactStore]:
    return _artifact_from_payload(
        tmp_path,
        _spl_payload(set_id="set-42"),
        artifact_id="dailymed:fixture",
        source_uri="memory://dailymed/set-42.xml",
        media_type="application/xml",
    )


def _spl_payload(*, set_id: str) -> bytes:
    return f"""<document xmlns="urn:hl7-org:v3" xml:lang="en">
      <setId root="{set_id}"/><versionNumber value="7"/><effectiveTime value="20260717"/>
      <title>Amoxicillin label</title><component><structuredBody><component><section>
        <title>Product data</title><text>Amoxicillin product information.</text>
        <subject><manufacturedProduct><manufacturedProduct>
          <code code="00000-001" codeSystem="2.16.840.1.113883.6.69"/>
          <name>AMOXICILLIN</name>
          <formCode code="C25158" displayName="CAPSULE"
                    codeSystem="2.16.840.1.113883.3.26.1.1"/>
          <asEntityWithGeneric><genericMedicine><name>amoxicillin</name>
          </genericMedicine></asEntityWithGeneric>
          <ingredient classCode="ACTIB"><quantity><numerator value="500" unit="mg"/>
          <denominator value="1"/></quantity><ingredientSubstance>
          <code code="9EM05410Q9" codeSystem="2.16.840.1.113883.4.9"/>
          <name>AMOXICILLIN</name></ingredientSubstance></ingredient>
          <ingredient classCode="IACT"><ingredientSubstance><name>STARCH</name>
          </ingredientSubstance></ingredient>
        </manufacturedProduct><consumedIn><substanceAdministration>
          <routeCode code="C38288" displayName="ORAL"
                     codeSystem="2.16.840.1.113883.3.26.1.1"/>
        </substanceAdministration></consumedIn></manufacturedProduct></subject>
      </section></component></structuredBody></component>
    </document>""".encode()


def _artifact_from_payload(
    tmp_path: Path,
    payload: bytes,
    *,
    artifact_id: str,
    source_uri: str,
    media_type: str,
    metadata: dict[str, str] | None = None,
) -> tuple[SourceArtifact, LocalArtifactStore]:
    store = LocalArtifactStore(tmp_path / "objects")
    stored = store.put_stream(io.BytesIO(payload), metadata={})
    artifact = SourceArtifact(
        artifact_id=artifact_id,
        source_id="dailymed",
        source_version="catalog-2026-07-17",
        source_uri=source_uri,
        object=stored,
        media_type=media_type,
        license_id="nlm_dailymed_terms",
        access_class=AccessClass.OPEN_WITH_TERMS,
        redistribution=RedistributionPolicy.ATTRIBUTION,
        hosted_processing_allowed=True,
        retrieved_at="2026-07-18T00:00:00+00:00",
        metadata={
            "published_date": "Jul 17, 2026",
            "set_id": "set-42",
            **(metadata or {}),
        },
    )
    return artifact, store
