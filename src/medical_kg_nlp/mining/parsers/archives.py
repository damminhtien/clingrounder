"""Bounded text/archive parsers for local plain-text and CodiEsp corpora."""

from __future__ import annotations

import hashlib
import io
import re
import stat
import zipfile
from collections.abc import Iterable
from pathlib import PurePosixPath

from medical_kg_nlp.mining.formats.codiesp import read_codiesp_archive
from medical_kg_nlp.mining.parsers.base import ArtifactParserAdapter
from medical_kg_nlp.mining.ports import ArtifactStorePort
from medical_kg_nlp.mining.records import MinedDocument, SourceArtifact

__all__ = ["CodiEspArchiveParser", "PlainTextArchiveParser", "PlainTextParser"]

_NUMERIC_ID = re.compile(r"[0-9]+")


class PlainTextParser(ArtifactParserAdapter):
    """Parse one UTF-8 text artifact with source-defined language and note type."""

    parser_id = "plain_text"
    parser_revision = "1"

    def __init__(self, *, language: str = "vi", note_type: str = "clinical_text") -> None:
        self.language = language
        self.note_type = note_type

    def parse(
        self,
        artifact: SourceArtifact,
        *,
        store: ArtifactStorePort,
    ) -> Iterable[MinedDocument]:
        text = self.read_artifact(artifact, store).decode("utf-8-sig")
        external_id = artifact.metadata.get("filename", artifact.object.sha256[:16])
        yield self.make_document(
            artifact,
            external_id=external_id,
            text=text,
            language=self.language,
            note_type=self.note_type,
            group_ids=(f"source_record:{external_id}",),
        )


class PlainTextArchiveParser(ArtifactParserAdapter):
    """Parse bounded UTF-8 ``.txt`` members without extracting archive paths.

    The parser is intentionally strict because a local archive is still untrusted input.
    Limits are checked from the ZIP directory before any member is decompressed and again
    while each member is streamed. Numeric IDs are canonicalized, so ``1.txt`` and
    ``01.txt`` cannot silently become two records for the same source document.
    """

    parser_id = "plain_text_archive"
    parser_revision = "1"

    def __init__(
        self,
        *,
        language: str = "vi",
        note_type: str = "clinical_text",
        require_numeric_ids: bool = True,
        max_members: int = 10_000,
        max_member_bytes: int = 8 * 1024 * 1024,
        max_total_uncompressed_bytes: int = 256 * 1024 * 1024,
        max_compression_ratio: float = 200.0,
    ) -> None:
        if not language.strip() or not note_type.strip():
            raise ValueError("Plain-text archive language and note type must be non-empty")
        if max_members < 1 or max_member_bytes < 1 or max_total_uncompressed_bytes < 1:
            raise ValueError("Plain-text archive limits must be positive")
        if max_compression_ratio < 1.0:
            raise ValueError("Plain-text archive compression ratio must be at least one")
        self.language = language
        self.note_type = note_type
        self.require_numeric_ids = require_numeric_ids
        self.max_members = max_members
        self.max_member_bytes = max_member_bytes
        self.max_total_uncompressed_bytes = max_total_uncompressed_bytes
        self.max_compression_ratio = max_compression_ratio

    def parse(
        self,
        artifact: SourceArtifact,
        *,
        store: ArtifactStorePort,
    ) -> Iterable[MinedDocument]:
        payload = self.read_artifact(artifact, store)
        try:
            archive = zipfile.ZipFile(io.BytesIO(payload))
        except zipfile.BadZipFile as error:
            raise ValueError("Plain-text artifact is not a valid ZIP archive") from error

        with archive:
            members = self._validated_members(archive)
            for source_id, info in members:
                raw = _read_zip_member(archive, info, limit=self.max_member_bytes)
                try:
                    # INVARIANT: decode without ``utf-8-sig`` so BOM and every newline remain
                    # part of the immutable offset coordinate system.
                    text = raw.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ValueError(
                        f"ZIP member {info.filename!r} is not valid UTF-8"
                    ) from error
                member_sha256 = hashlib.sha256(raw).hexdigest()
                yield self.make_document(
                    artifact,
                    external_id=source_id,
                    text=text,
                    language=self.language,
                    note_type=self.note_type,
                    group_ids=(f"source_record:{artifact.source_id}:{source_id}",),
                    metadata={
                        "archive_member": info.filename,
                        "source_document_id": source_id,
                        "source_archive_sha256": artifact.object.sha256,
                        "raw_bytes_sha256": member_sha256,
                        "raw_byte_size": str(len(raw)),
                        "raw_encoding": "utf-8",
                        "newline_normalization": "none",
                    },
                    source_unit_sha256=member_sha256,
                )

    def _validated_members(
        self,
        archive: zipfile.ZipFile,
    ) -> tuple[tuple[str, zipfile.ZipInfo], ...]:
        infos = archive.infolist()
        if len(infos) > self.max_members:
            raise ValueError(
                f"ZIP archive has {len(infos)} members; limit is {self.max_members}"
            )

        total_uncompressed = 0
        by_source_id: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            _validate_zip_path(info.filename)
            if _is_zip_symlink(info):
                raise ValueError(f"ZIP member {info.filename!r} is a symbolic link")
            if info.flag_bits & 0x1:
                raise ValueError(f"ZIP member {info.filename!r} is encrypted")
            if info.is_dir():
                continue
            if PurePosixPath(info.filename).suffix.casefold() != ".txt":
                continue
            if info.file_size > self.max_member_bytes:
                raise ValueError(
                    f"ZIP member {info.filename!r} exceeds {self.max_member_bytes} bytes"
                )
            total_uncompressed += info.file_size
            if total_uncompressed > self.max_total_uncompressed_bytes:
                raise ValueError(
                    "ZIP text members exceed total uncompressed limit of "
                    f"{self.max_total_uncompressed_bytes} bytes"
                )
            compressed_size = max(info.compress_size, 1)
            ratio = info.file_size / compressed_size
            if ratio > self.max_compression_ratio:
                raise ValueError(
                    f"ZIP member {info.filename!r} has unsafe compression ratio {ratio:.1f}"
                )
            source_id = PurePosixPath(info.filename).stem
            if self.require_numeric_ids:
                if _NUMERIC_ID.fullmatch(source_id) is None:
                    raise ValueError(
                        f"ZIP member {info.filename!r} does not have a numeric document ID"
                    )
                source_id = str(int(source_id))
            if source_id in by_source_id:
                previous = by_source_id[source_id].filename
                raise ValueError(
                    f"Duplicate source document ID {source_id!r}: {previous!r} and "
                    f"{info.filename!r}"
                )
            by_source_id[source_id] = info

        if not by_source_id:
            raise ValueError("ZIP archive contains no eligible .txt members")
        return tuple(
            sorted(
                by_source_id.items(),
                key=lambda item: (
                    int(item[0]) if self.require_numeric_ids else item[0],
                    item[1].filename,
                ),
            )
        )


class CodiEspArchiveParser(ArtifactParserAdapter):
    """Parse only Spanish source cases from a bounded CodiEsp archive."""

    parser_id = "codiesp"
    parser_revision = "2"

    def parse(
        self,
        artifact: SourceArtifact,
        *,
        store: ArtifactStorePort,
    ) -> Iterable[MinedDocument]:
        bundle = read_codiesp_archive(
            self.read_artifact(artifact, store),
            include_annotations=False,
        )
        for source_document in bundle.documents:
            external_id = f"{source_document.split}:{source_document.case_id}"
            metadata = {
                "archive_member": source_document.text_member,
                "codiesp_case_id": source_document.case_id,
                "corpus_split": source_document.split,
            }
            if source_document.annotation_member is not None:
                metadata["annotation_member"] = source_document.annotation_member
            yield self.make_document(
                artifact,
                external_id=external_id,
                text=source_document.text,
                language="es",
                note_type="clinical_case",
                group_ids=(f"codiesp_case:{source_document.case_id}",),
                metadata=metadata,
            )


def _validate_zip_path(filename: str) -> None:
    """Reject paths that could escape if a future caller extracted the archive."""

    normalized = filename.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized)
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"Unsafe ZIP member path {filename!r}")


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(unix_mode)


def _read_zip_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    limit: int,
) -> bytes:
    with archive.open(info, "r") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"ZIP member {info.filename!r} exceeds {limit} bytes")
    if len(payload) != info.file_size:
        raise ValueError(f"ZIP member {info.filename!r} size changed while reading")
    return payload
