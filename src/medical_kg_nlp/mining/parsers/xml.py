"""JATS and Structured Product Label XML document parsers."""

from __future__ import annotations

import io
import json
import tarfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable

from medical_kg_nlp.mining.formats.dailymed import (
    extract_spl_products,
    render_spl_product,
)
from medical_kg_nlp.mining.formats.jats import render_jats_article
from medical_kg_nlp.mining.parsers.base import ArtifactParserAdapter
from medical_kg_nlp.mining.ports import ArtifactStorePort
from medical_kg_nlp.mining.records import MinedDocument, SourceArtifact

__all__ = ["JatsXmlParser", "SplXmlParser"]


class JatsXmlParser(ArtifactParserAdapter):
    """Render PMC JATS XML or OA tarballs into immutable article text."""

    parser_id = "jats_xml"
    parser_revision = "2"
    max_xml_member_bytes = 64 * 1024 * 1024

    def parse(
        self,
        artifact: SourceArtifact,
        *,
        store: ArtifactStorePort,
    ) -> Iterable[MinedDocument]:
        payload = self.read_artifact(artifact, store)
        for member_name, xml_payload in _jats_payloads(payload, artifact.media_type):
            root = ET.fromstring(xml_payload)
            rendered = render_jats_article(root, fallback_id=member_name)
            yield self.make_document(
                artifact,
                external_id=rendered.article_id,
                text=rendered.text,
                language=_xml_language(root, default="en"),
                note_type="case_report_article",
                group_ids=(f"article:{rendered.article_id}",),
                metadata={
                    "archive_member": member_name,
                    "article_type": rendered.article_type,
                    "journal_title": rendered.journal_title,
                    "publication_year": rendered.publication_year,
                    "keywords": json.dumps(rendered.keywords, ensure_ascii=False),
                    "subjects": json.dumps(rendered.subjects, ensure_ascii=False),
                    "source_block_format": "jats",
                    "source_blocks": json.dumps(
                        [block.to_dict() for block in rendered.blocks],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            )


class SplXmlParser(ArtifactParserAdapter):
    """Render DailyMed narrative text and source-structured product records."""

    parser_id = "spl_xml"
    parser_revision = "2"

    def parse(
        self,
        artifact: SourceArtifact,
        *,
        store: ArtifactStorePort,
    ) -> Iterable[MinedDocument]:
        root = ET.fromstring(self.read_artifact(artifact, store))
        set_id_node = _first_local(root, "setId")
        set_id = ""
        if set_id_node is not None:
            set_id = (set_id_node.get("root") or _node_text(set_id_node)).strip()
        set_id = set_id or artifact.metadata.get("set_id") or artifact.object.sha256[:16]
        version_node = _first_local(root, "versionNumber")
        spl_version = "" if version_node is None else (version_node.get("value") or "").strip()
        effective_node = _first_local(root, "effectiveTime")
        effective_time = (
            "" if effective_node is None else (effective_node.get("value") or "").strip()
        )
        blocks: list[str] = []
        title = _first_local(root, "title")
        if title is not None and (text := _node_text(title)):
            blocks.append(text)
        for section in _all_local(root, "section"):
            section_blocks: list[str] = []
            section_title = next(
                (child for child in section if _local_name(child.tag) == "title"),
                None,
            )
            if section_title is not None and (text := _node_text(section_title)):
                section_blocks.append(text)
            section_text = next(
                (child for child in section if _local_name(child.tag) == "text"),
                None,
            )
            if section_text is not None and (text := _node_text(section_text)):
                section_blocks.append(text)
            if section_blocks:
                blocks.append("\n".join(section_blocks))
        yield self.make_document(
            artifact,
            external_id=set_id,
            text="\n\n".join(_deduplicate_adjacent(blocks)),
            language=_xml_language(root, default="en"),
            note_type="structured_product_label",
            group_ids=(f"drug_label:{set_id}",),
            metadata={
                "dailymed_set_id": set_id,
                "dailymed_source_version": artifact.source_version,
                "dailymed_spl_version": spl_version or artifact.metadata.get("spl_version", ""),
                "effective_time": effective_time,
                "published_date": artifact.metadata.get("published_date", ""),
            },
        )
        for product_index, product in enumerate(extract_spl_products(root)):
            rendered = render_spl_product(product)
            yield self.make_document(
                artifact,
                external_id=f"{set_id}:product:{product_index}",
                text=rendered.text,
                language=_xml_language(root, default="en"),
                note_type="structured_medication_record",
                group_ids=(f"drug_label:{set_id}",),
                metadata={
                    "dailymed_set_id": set_id,
                    "dailymed_source_version": artifact.source_version,
                    "dailymed_spl_version": (
                        spl_version or artifact.metadata.get("spl_version", "")
                    ),
                    "effective_time": effective_time,
                    "published_date": artifact.metadata.get("published_date", ""),
                    "spl_fields": json.dumps(
                        [field.to_dict() for field in rendered.fields],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    "spl_ndc": product.ndc,
                    "spl_product_index": str(product_index),
                },
            )


def _jats_payloads(payload: bytes, media_type: str) -> tuple[tuple[str, bytes], ...]:
    if media_type not in {"application/gzip", "application/x-gzip"} and not payload.startswith(
        b"\x1f\x8b"
    ):
        return (("article.xml", payload),)
    members: list[tuple[str, bytes]] = []
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in sorted(archive.getmembers(), key=lambda item: item.name):
            if not member.isfile() or not member.name.lower().endswith((".nxml", ".xml")):
                continue
            if member.size > JatsXmlParser.max_xml_member_bytes:
                raise ValueError(f"JATS member {member.name!r} exceeds size limit")
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            members.append((member.name, extracted.read()))
    if not members:
        raise ValueError("PMC OA package contains no JATS XML member")
    return tuple(members)


def _node_text(node: ET.Element) -> str:
    # Parsing XML creates the canonical document. Inline tags are joined before whitespace cleanup.
    return " ".join("".join(node.itertext()).split())


def _first_text(root: ET.Element, path: str) -> str:
    node = root.find(path)
    return "" if node is None else _node_text(node)


def _first_local(root: ET.Element, local_name: str) -> ET.Element | None:
    return next((node for node in root.iter() if _local_name(node.tag) == local_name), None)


def _all_local(root: ET.Element, local_name: str) -> tuple[ET.Element, ...]:
    return tuple(node for node in root.iter() if _local_name(node.tag) == local_name)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml_language(root: ET.Element, *, default: str) -> str:
    return root.get("{http://www.w3.org/XML/1998/namespace}lang", default)


def _deduplicate_adjacent(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if not result or result[-1] != value:
            result.append(value)
    return result
