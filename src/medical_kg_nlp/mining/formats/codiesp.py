"""Bounded CodiEsp archive reader for Spanish documents and CodiEsp-X labels."""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

__all__ = [
    "CodiEspArchiveBundle",
    "CodiEspDocumentBundle",
    "CodiEspSpanAnnotation",
    "read_codiesp_archive",
]

_SPLITS = ("train", "dev", "test", "background")
_SPLIT_ORDER = {value: index for index, value in enumerate(_SPLITS)}


@dataclass(frozen=True)
class CodiEspDocumentBundle:
    """One immutable Spanish clinical case inside the source archive."""

    split: str
    case_id: str
    text_member: str
    annotation_member: str | None
    text: str


@dataclass(frozen=True)
class CodiEspSpanAnnotation:
    """One source CodiEsp-X row with validated raw coordinate segments."""

    split: str
    case_id: str
    annotation_member: str
    row_number: int
    source_label: str
    source_code: str
    annotated_text: str
    raw_segments: tuple[tuple[int, int], ...]
    segments: tuple[tuple[int, int], ...]
    segment_issues: tuple[str, ...]
    source_text_matches: bool

    @property
    def envelope(self) -> tuple[int, int]:
        """Return the contiguous raw envelope required by annotation contracts."""

        # CodiEsp may list segments in phrase order instead of source-offset order.
        return min(start for start, _ in self.segments), max(end for _, end in self.segments)


@dataclass(frozen=True)
class CodiEspArchiveBundle:
    """Validated Spanish documents plus optional source span annotations."""

    documents: tuple[CodiEspDocumentBundle, ...]
    annotations: tuple[CodiEspSpanAnnotation, ...]


def read_codiesp_archive(
    payload: bytes,
    *,
    include_annotations: bool = True,
    max_members: int = 100_000,
    max_member_bytes: int = 32 * 1024 * 1024,
    max_total_bytes: int = 512 * 1024 * 1024,
) -> CodiEspArchiveBundle:
    """Decode only source-language cases and optionally validate CodiEsp-X rows."""

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        files = _validated_files(
            archive,
            max_members=max_members,
            max_member_bytes=max_member_bytes,
            max_total_bytes=max_total_bytes,
        )
        by_name = {member.filename: member for member in files}
        documents = _read_documents(archive, files, by_name)
        annotations = _read_annotations(archive, files, documents) if include_annotations else ()
    return CodiEspArchiveBundle(documents=documents, annotations=annotations)


def _validated_files(
    archive: zipfile.ZipFile,
    *,
    max_members: int,
    max_member_bytes: int,
    max_total_bytes: int,
) -> tuple[zipfile.ZipInfo, ...]:
    members = archive.infolist()
    if len(members) > max_members:
        raise ValueError("CodiEsp archive exceeds member count limit")
    files = tuple(member for member in members if not member.is_dir())
    names = [member.filename for member in files]
    if len(names) != len(set(names)):
        raise ValueError("CodiEsp archive contains duplicate member names")
    if sum(member.file_size for member in files) > max_total_bytes:
        raise ValueError("CodiEsp archive exceeds total uncompressed size limit")
    for member in files:
        if member.flag_bits & 0x1:
            raise ValueError(f"Encrypted CodiEsp member is not supported: {member.filename!r}")
        if member.file_size > max_member_bytes:
            raise ValueError(f"CodiEsp member {member.filename!r} exceeds size limit")
    return files


def _read_documents(
    archive: zipfile.ZipFile,
    files: tuple[zipfile.ZipInfo, ...],
    by_name: dict[str, zipfile.ZipInfo],
) -> tuple[CodiEspDocumentBundle, ...]:
    documents = []
    seen_keys: set[tuple[str, str]] = set()
    for member in files:
        identity = _document_identity(member.filename)
        if identity is None:
            continue
        split, case_id = identity
        key = (split, case_id)
        if key in seen_keys:
            raise ValueError(f"Duplicate CodiEsp case {split}/{case_id}")
        seen_keys.add(key)
        parts = PurePosixPath(member.filename).parts
        annotation_name = str(PurePosixPath(*parts[:-3], split, f"{split}X.tsv"))
        documents.append(
            CodiEspDocumentBundle(
                split=split,
                case_id=case_id,
                text_member=member.filename,
                annotation_member=(annotation_name if annotation_name in by_name else None),
                # INVARIANT: offsets target the decoded Spanish source member exactly;
                # the machine-translated text_files_en tree is never considered here.
                text=archive.read(member).decode("utf-8-sig"),
            )
        )
    if not documents:
        raise ValueError("CodiEsp archive contains no Spanish text_files documents")
    return tuple(
        sorted(
            documents,
            key=lambda item: (_SPLIT_ORDER[item.split], item.case_id),
        )
    )


def _read_annotations(
    archive: zipfile.ZipFile,
    files: tuple[zipfile.ZipInfo, ...],
    documents: tuple[CodiEspDocumentBundle, ...],
) -> tuple[CodiEspSpanAnnotation, ...]:
    documents_by_key = {(item.split, item.case_id): item for item in documents}
    annotation_members = []
    for member in files:
        split = _annotation_split(member.filename)
        if split is not None:
            annotation_members.append((split, member))
    annotations = []
    for split, member in sorted(
        annotation_members,
        key=lambda item: (_SPLIT_ORDER[item[0]], item[1].filename),
    ):
        payload = archive.read(member).decode("utf-8-sig")
        reader = csv.reader(io.StringIO(payload), delimiter="\t")
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != 5:
                raise ValueError(
                    f"Invalid CodiEsp-X row {member.filename}:{row_number}: "
                    "expected five tab fields"
                )
            case_id, source_label, source_code, annotated_text, raw_position = row
            document = documents_by_key.get((split, case_id))
            if document is None:
                raise ValueError(f"CodiEsp-X row references missing case {split}/{case_id}")
            raw_segments, segments, segment_issues = _parse_segments(
                raw_position,
                source_text=document.text,
                member=member.filename,
                row_number=row_number,
            )
            source_segments_text = " ".join(document.text[start:end] for start, end in segments)
            annotations.append(
                CodiEspSpanAnnotation(
                    split=split,
                    case_id=case_id,
                    annotation_member=member.filename,
                    row_number=row_number,
                    source_label=_required_value(
                        source_label, member.filename, row_number, "label"
                    ),
                    source_code=_required_value(source_code, member.filename, row_number, "code"),
                    annotated_text=_required_value(
                        annotated_text, member.filename, row_number, "text reference"
                    ),
                    raw_segments=raw_segments,
                    segments=segments,
                    segment_issues=segment_issues,
                    source_text_matches=source_segments_text == annotated_text,
                )
            )
    if not annotations:
        raise ValueError("CodiEsp archive contains no CodiEsp-X annotations")
    return tuple(annotations)


def _document_identity(member_name: str) -> tuple[str, str] | None:
    parts = PurePosixPath(member_name).parts
    if len(parts) < 3 or parts[-2] != "text_files" or not parts[-1].endswith(".txt"):
        return None
    split = parts[-3]
    if split not in _SPLIT_ORDER:
        return None
    case_id = parts[-1].removesuffix(".txt")
    return (split, case_id) if case_id else None


def _annotation_split(member_name: str) -> str | None:
    parts = PurePosixPath(member_name).parts
    if len(parts) < 2:
        return None
    split = parts[-2]
    if split not in {"train", "dev", "test"}:
        return None
    return split if parts[-1] == f"{split}X.tsv" else None


def _parse_segments(
    value: str,
    *,
    source_text: str,
    member: str,
    row_number: int,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[str, ...],
]:
    raw_segments = []
    segments = []
    issues = []
    for raw_segment in value.split(";"):
        coordinates = raw_segment.split()
        if len(coordinates) != 2:
            raise ValueError(f"Invalid CodiEsp segment {member}:{row_number}: {raw_segment!r}")
        try:
            start, end = (int(coordinates[0]), int(coordinates[1]))
        except ValueError as error:
            raise ValueError(
                f"Non-numeric CodiEsp segment {member}:{row_number}: {raw_segment!r}"
            ) from error
        raw_segments.append((start, end))
        if start < 0 or end < start or end > len(source_text):
            raise ValueError(f"Out-of-range CodiEsp segment {member}:{row_number}: {(start, end)}")
        if start == end:
            # Some source rows contain an empty first segment. Keep it in raw_segments
            # for audit, but never create a zero-width internal annotation segment.
            issues.append(f"zero_length_segment:{start}")
            continue
        segments.append((start, end))
    if not segments:
        raise ValueError(f"CodiEsp row {member}:{row_number} has no usable segment")
    ordered_segments = sorted(segments)
    for previous, current in zip(ordered_segments, ordered_segments[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                f"Overlapping CodiEsp segments {member}:{row_number}: {previous} and {current}"
            )
    return tuple(raw_segments), tuple(segments), tuple(issues)


def _required_value(value: str, member: str, row_number: int, field: str) -> str:
    if not value.strip():
        raise ValueError(f"Empty CodiEsp {field} at {member}:{row_number}")
    return value
