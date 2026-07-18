"""DailyMed SPL projection tests for structured drug mining."""

from __future__ import annotations

import io
from pathlib import Path

from medical_kg_nlp.mining.labelers.dailymed import DailyMedStructuredLabelerAdapter
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


def _spl_artifact(tmp_path: Path) -> tuple[SourceArtifact, LocalArtifactStore]:
    payload = b"""<document xmlns="urn:hl7-org:v3" xml:lang="en">
      <setId root="set-42"/><versionNumber value="7"/><effectiveTime value="20260717"/>
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
    </document>"""
    store = LocalArtifactStore(tmp_path / "objects")
    stored = store.put_stream(io.BytesIO(payload), metadata={})
    artifact = SourceArtifact(
        artifact_id="dailymed:fixture",
        source_id="dailymed",
        source_version="catalog-2026-07-17",
        source_uri="memory://dailymed/set-42.xml",
        object=stored,
        media_type="application/xml",
        license_id="nlm_dailymed_terms",
        access_class=AccessClass.OPEN_WITH_TERMS,
        redistribution=RedistributionPolicy.ATTRIBUTION,
        hosted_processing_allowed=True,
        retrieved_at="2026-07-18T00:00:00+00:00",
        metadata={"published_date": "Jul 17, 2026", "set_id": "set-42"},
    )
    return artifact, store
