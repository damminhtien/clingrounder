"""JATS and Structured Product Label XML document parsers."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable
from typing import BinaryIO

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
    """Render DailyMed XML or multipart ZIP releases without loading an archive into RAM.

    DailyMed distributes individual labels as ZIP files containing one SPL XML document and
    optional images. Full human releases are multi-gigabyte ZIP parts. The parser therefore
    bounds every XML member and aggregate declared size while retaining only one label payload
    in memory at a time.
    """

    parser_id = "spl_xml"
    parser_revision = "3"
    max_archive_members = 100_000
    max_xml_member_bytes = 64 * 1024 * 1024
    max_nested_archive_bytes = 256 * 1024 * 1024
    max_total_member_bytes = 128 * 1024 * 1024 * 1024

    def parse(
        self,
        artifact: SourceArtifact,
        *,
        store: ArtifactStorePort,
    ) -> Iterable[MinedDocument]:
        seen_payloads: set[str] = set()
        with store.open(artifact.object.sha256) as stream:
            if _is_zip_artifact(artifact, stream):
                for member_name, payload in self._zip_payloads(stream, artifact.source_uri):
                    source_unit_sha256 = hashlib.sha256(payload).hexdigest()
                    if source_unit_sha256 in seen_payloads:
                        continue
                    seen_payloads.add(source_unit_sha256)
                    yield from self._parse_payload(
                        artifact,
                        payload,
                        archive_member=member_name,
                        source_unit_sha256=source_unit_sha256,
                    )
                return
            payload = _read_stream_bounded(stream, self.max_xml_member_bytes)
        yield from self._parse_payload(
            artifact,
            payload,
            archive_member="",
            source_unit_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def _parse_payload(
        self,
        artifact: SourceArtifact,
        payload: bytes,
        *,
        archive_member: str,
        source_unit_sha256: str,
    ) -> Iterable[MinedDocument]:
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as error:
            location = archive_member or artifact.source_uri
            raise ValueError(f"Invalid DailyMed SPL XML in {location!r}: {error}") from error
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
                "archive_member": archive_member,
            },
            source_unit_sha256=source_unit_sha256,
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
                    "archive_member": archive_member,
                },
                source_unit_sha256=source_unit_sha256,
            )

    def _zip_payloads(
        self,
        stream: BinaryIO,
        source_name: str,
    ) -> Iterable[tuple[str, bytes]]:
        if not stream.seekable():
            raise ValueError(
                "DailyMed ZIP parsing requires a seekable artifact stream; stage remote "
                "archives in the content-addressed cache before parsing"
            )
        try:
            with zipfile.ZipFile(stream) as archive:
                yield from self._archive_xml_payloads(
                    archive,
                    source_name=source_name,
                    depth=0,
                )
        except zipfile.BadZipFile as error:
            raise ValueError(f"Invalid DailyMed ZIP artifact {source_name!r}") from error

    def _archive_xml_payloads(
        self,
        archive: zipfile.ZipFile,
        *,
        source_name: str,
        depth: int,
    ) -> Iterable[tuple[str, bytes]]:
        infos = archive.infolist()
        if len(infos) > self.max_archive_members:
            raise ValueError(
                f"DailyMed archive {source_name!r} has {len(infos)} members; "
                f"limit is {self.max_archive_members}"
            )
        names: set[str] = set()
        declared_bytes = 0
        xml_members = 0
        for info in sorted(infos, key=lambda item: item.filename):
            if info.is_dir():
                continue
            if info.filename in names:
                raise ValueError(
                    f"DailyMed archive {source_name!r} contains duplicate member "
                    f"{info.filename!r}"
                )
            names.add(info.filename)
            if info.flag_bits & 0x1:
                raise ValueError(
                    f"DailyMed archive member {info.filename!r} is encrypted"
                )
            suffix = info.filename.lower()
            if not suffix.endswith((".xml", ".zip")):
                continue
            declared_bytes += info.file_size
            if declared_bytes > self.max_total_member_bytes:
                raise ValueError(
                    f"DailyMed archive {source_name!r} exceeds the declared extraction limit"
                )
            if suffix.endswith(".xml"):
                xml_members += 1
                payload = _read_zip_member(
                    archive,
                    info,
                    limit=self.max_xml_member_bytes,
                )
                yield info.filename, payload
                continue
            if depth >= 1:
                raise ValueError(
                    f"DailyMed archive nesting exceeds one level at {info.filename!r}"
                )
            nested_payload = _read_zip_member(
                archive,
                info,
                limit=self.max_nested_archive_bytes,
            )
            try:
                with zipfile.ZipFile(io.BytesIO(nested_payload)) as nested:
                    nested_source = f"{source_name}!{info.filename}"
                    for nested_name, payload in self._archive_xml_payloads(
                        nested,
                        source_name=nested_source,
                        depth=depth + 1,
                    ):
                        xml_members += 1
                        yield f"{info.filename}!{nested_name}", payload
            except zipfile.BadZipFile as error:
                raise ValueError(
                    f"Invalid nested DailyMed ZIP member {info.filename!r}"
                ) from error
        if xml_members == 0:
            raise ValueError(f"DailyMed archive {source_name!r} contains no SPL XML member")


def _is_zip_artifact(artifact: SourceArtifact, stream: BinaryIO) -> bool:
    if artifact.media_type.lower() in {
        "application/zip",
        "application/x-zip-compressed",
    } or artifact.source_uri.lower().endswith(".zip"):
        return True
    if not stream.seekable():
        return False
    position = stream.tell()
    signature = stream.read(4)
    stream.seek(position)
    return signature.startswith(b"PK\x03\x04")


def _read_stream_bounded(stream: BinaryIO, limit: int) -> bytes:
    payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"DailyMed SPL XML exceeds member limit of {limit} bytes")
    return payload


def _read_zip_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    limit: int,
) -> bytes:
    if info.file_size > limit:
        raise ValueError(
            f"DailyMed archive member {info.filename!r} exceeds size limit of {limit} bytes"
        )
    # SCALING: read one bounded label package at a time; the multi-gigabyte outer release is
    # never copied into process memory.
    with archive.open(info) as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(
            f"DailyMed archive member {info.filename!r} exceeds size limit of {limit} bytes"
        )
    if len(payload) != info.file_size:
        raise ValueError(
            f"DailyMed archive member {info.filename!r} size differs from its ZIP metadata"
        )
    return payload


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
