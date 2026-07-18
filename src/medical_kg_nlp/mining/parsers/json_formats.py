"""ClinicalTrials, FHIR, and BioC JSON parsers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from html.parser import HTMLParser
from typing import Any

from medical_kg_nlp.mining.parsers.base import ArtifactParserAdapter
from medical_kg_nlp.mining.ports import ArtifactStorePort
from medical_kg_nlp.mining.records import MinedDocument, SourceArtifact

__all__ = ["BiocJsonParser", "ClinicalTrialsJsonParser", "FhirBundleParser"]


class ClinicalTrialsJsonParser(ArtifactParserAdapter):
    """Render API v2 study records as condition/intervention/outcome documents."""

    parser_id = "clinicaltrials_json"
    parser_revision = "1"

    def parse(
        self,
        artifact: SourceArtifact,
        *,
        store: ArtifactStorePort,
    ) -> Iterable[MinedDocument]:
        payload = json.loads(self.read_artifact(artifact, store))
        studies = payload.get("studies") if isinstance(payload, Mapping) else None
        if studies is None:
            studies = [payload]
        if not isinstance(studies, Sequence):
            raise ValueError("ClinicalTrials payload has no study records")
        for study in studies:
            if not isinstance(study, Mapping):
                raise ValueError("ClinicalTrials study must be an object")
            protocol = _mapping(study.get("protocolSection", {}), "protocolSection")
            identification = _mapping(protocol.get("identificationModule", {}), "identification")
            nct_id = str(identification.get("nctId", "")).strip()
            if not nct_id:
                raise ValueError("ClinicalTrials study has no nctId")
            sections = _clinical_trial_sections(protocol)
            yield self.make_document(
                artifact,
                external_id=nct_id,
                text="\n\n".join(sections),
                language="en",
                note_type="clinical_trial",
                group_ids=(f"clinical_trial:{nct_id}",),
            )


class FhirBundleParser(ArtifactParserAdapter):
    """Render a FHIR bundle into a deterministic trajectory document."""

    parser_id = "fhir_bundle"
    parser_revision = "1"

    def parse(
        self,
        artifact: SourceArtifact,
        *,
        store: ArtifactStorePort,
    ) -> Iterable[MinedDocument]:
        bundle = json.loads(self.read_artifact(artifact, store))
        if not isinstance(bundle, Mapping) or bundle.get("resourceType") != "Bundle":
            raise ValueError("FHIR parser expects a Bundle resource")
        bundle_id = str(bundle.get("id") or artifact.object.sha256[:16])
        resources = _fhir_resources(bundle)
        patient_ids = sorted(
            str(resource.get("id"))
            for resource in resources
            if resource.get("resourceType") == "Patient" and resource.get("id")
        )
        blocks = [_render_fhir_resource(resource) for resource in resources]
        text = "\n\n".join(block for block in blocks if block)
        group_ids = tuple(f"patient:{value}" for value in patient_ids) or (
            f"fhir_bundle:{bundle_id}",
        )
        yield self.make_document(
            artifact,
            external_id=bundle_id,
            text=text,
            language="en",
            note_type="fhir_trajectory",
            group_ids=group_ids,
        )


class BiocJsonParser(ArtifactParserAdapter):
    """Reconstruct BioC passage offsets so imported annotations remain projectable."""

    parser_id = "bioc_json"
    parser_revision = "2"

    def parse(
        self,
        artifact: SourceArtifact,
        *,
        store: ArtifactStorePort,
    ) -> Iterable[MinedDocument]:
        payload = json.loads(self.read_artifact(artifact, store))
        documents = _bioc_documents(payload)
        for document in documents:
            if not isinstance(document, Mapping):
                raise ValueError("BioC document must be an object")
            external_id = str(document.get("id", "")).strip()
            if not external_id:
                raise ValueError("BioC document has no id")
            passages = document.get("passages", [])
            if not isinstance(passages, Sequence):
                raise ValueError("BioC passages must be a sequence")
            text = _reconstruct_bioc_text(passages)
            yield self.make_document(
                artifact,
                external_id=external_id,
                text=text,
                language=str(document.get("language", "en")),
                note_type="biomedical_literature",
                group_ids=(f"article:{external_id}",),
            )


def _bioc_documents(payload: Any) -> tuple[Mapping[str, Any], ...]:
    """Flatten BioC object, document-list, and collection-list serializations."""

    raw_documents: Any = payload.get("documents") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_documents, Sequence) or isinstance(raw_documents, (str, bytes)):
        raise ValueError("BioC JSON must contain a documents sequence")
    documents: list[Mapping[str, Any]] = []
    for item in raw_documents:
        if not isinstance(item, Mapping):
            raise ValueError("BioC document or collection must be an object")
        collection_documents = item.get("documents")
        if collection_documents is None:
            documents.append(item)
            continue
        if not isinstance(collection_documents, Sequence) or isinstance(
            collection_documents, (str, bytes)
        ):
            raise ValueError("BioC collection documents must be a sequence")
        for document in collection_documents:
            if not isinstance(document, Mapping):
                raise ValueError("BioC document must be an object")
            documents.append(document)
    return tuple(documents)


def _clinical_trial_sections(protocol: Mapping[str, Any]) -> list[str]:
    identification = _mapping(protocol.get("identificationModule", {}), "identification")
    description = _mapping(protocol.get("descriptionModule", {}), "description")
    conditions = _mapping(protocol.get("conditionsModule", {}), "conditions")
    arms = _mapping(protocol.get("armsInterventionsModule", {}), "arms")
    outcomes = _mapping(protocol.get("outcomesModule", {}), "outcomes")
    sections: list[str] = []
    _append_section(sections, "Title", identification.get("briefTitle"))
    _append_section(sections, "Summary", description.get("briefSummary"))
    _append_list_section(sections, "Conditions", conditions.get("conditions"))
    interventions = arms.get("interventions", [])
    if isinstance(interventions, Sequence):
        rendered = []
        for intervention in interventions:
            if isinstance(intervention, Mapping):
                name = str(intervention.get("name", "")).strip()
                description_text = str(intervention.get("description", "")).strip()
                rendered.append(": ".join(value for value in (name, description_text) if value))
        _append_list_section(sections, "Interventions", rendered)
    for key, title in (("primaryOutcomes", "Primary outcomes"), ("secondaryOutcomes", "Secondary outcomes")):
        values = outcomes.get(key, [])
        if isinstance(values, Sequence):
            rendered = [
                str(item.get("measure", "")).strip()
                for item in values
                if isinstance(item, Mapping) and item.get("measure")
            ]
            _append_list_section(sections, title, rendered)
    if not sections:
        raise ValueError("ClinicalTrials study produced no text sections")
    return sections


def _fhir_resources(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries = bundle.get("entry", [])
    if not isinstance(entries, Sequence):
        raise ValueError("FHIR Bundle.entry must be a sequence")
    resources: list[Mapping[str, Any]] = []
    for entry in entries:
        if isinstance(entry, Mapping) and isinstance(entry.get("resource"), Mapping):
            resources.append(entry["resource"])
    return sorted(resources, key=lambda item: (str(item.get("resourceType", "")), str(item.get("id", ""))))


def _render_fhir_resource(resource: Mapping[str, Any]) -> str:
    resource_type = str(resource.get("resourceType", "Resource"))
    identifier = str(resource.get("id", ""))
    values: list[str] = []
    if resource_type == "Composition":
        sections = resource.get("section", [])
        if isinstance(sections, Sequence):
            for section in sections:
                if not isinstance(section, Mapping):
                    continue
                title = str(section.get("title", "")).strip()
                narrative = section.get("text", {})
                div = narrative.get("div", "") if isinstance(narrative, Mapping) else ""
                rendered = _strip_html(str(div))
                values.append(": ".join(value for value in (title, rendered) if value))
    else:
        for key in ("code", "medicationCodeableConcept", "valueCodeableConcept"):
            value = resource.get(key)
            if isinstance(value, Mapping) and (text := _codeable_text(value)):
                values.append(text)
        for key in ("valueString", "status", "clinicalStatus"):
            value = resource.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.strip())
    header = "/".join(value for value in (resource_type, identifier) if value)
    body = "; ".join(dict.fromkeys(value for value in values if value))
    return f"{header}: {body}" if body else header


def _reconstruct_bioc_text(passages: Sequence[Any]) -> str:
    normalized: list[tuple[int, str]] = []
    max_end = 0
    for passage in passages:
        if not isinstance(passage, Mapping):
            raise ValueError("BioC passage must be an object")
        offset = int(passage.get("offset", 0))
        text = str(passage.get("text", ""))
        if offset < 0:
            raise ValueError("BioC passage requires a non-negative offset")
        if not text:
            # BioC PMC exports can include empty reference placeholders. They carry no
            # characters or annotations, so excluding them does not alter any raw offset.
            continue
        normalized.append((offset, text))
        max_end = max(max_end, offset + len(text))
    if not normalized:
        raise ValueError("BioC document has no text passages")
    characters = [" "] * max_end
    occupied = [False] * max_end
    for offset, text in sorted(normalized):
        for index, character in enumerate(text, start=offset):
            if occupied[index] and characters[index] != character:
                raise ValueError("Overlapping BioC passages disagree on source text")
            characters[index] = character
            occupied[index] = True
    # INVARIANT: leading and inter-passage spaces retain BioC's absolute passage offsets.
    return "".join(characters)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.fragments: list[str] = []

    def handle_data(self, data: str) -> None:
        self.fragments.append(data)


def _strip_html(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    return " ".join("".join(parser.fragments).split())


def _codeable_text(value: Mapping[str, Any]) -> str:
    if text := str(value.get("text", "")).strip():
        return text
    coding = value.get("coding", [])
    if isinstance(coding, Sequence):
        for item in coding:
            if isinstance(item, Mapping) and item.get("display"):
                return str(item["display"]).strip()
    return ""


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _append_section(sections: list[str], title: str, value: Any) -> None:
    text = "" if value is None else str(value).strip()
    if text:
        sections.append(f"{title}\n{text}")


def _append_list_section(sections: list[str], title: str, values: Any) -> None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return
    rendered = [str(value).strip() for value in values if str(value).strip()]
    if rendered:
        sections.append(f"{title}\n" + "\n".join(f"- {value}" for value in rendered))
