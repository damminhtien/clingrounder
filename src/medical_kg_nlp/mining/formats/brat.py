"""Bounded BRAT archive and text-bound annotation decoding."""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass

__all__ = [
    "BratDocumentBundle",
    "BratTextBoundAnnotation",
    "parse_brat_text_bound_annotations",
    "read_brat_archive",
]

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class BratDocumentBundle:
    """One paired BRAT source document and annotation member."""

    text_member: str
    annotation_member: str
    text: str
    annotations: str
    newline_normalization: str


@dataclass(frozen=True)
class BratTextBoundAnnotation:
    """One BRAT text-bound annotation with one or more raw source segments."""

    annotation_id: str
    source_label: str
    segments: tuple[tuple[int, int], ...]
    annotated_text: str

    @property
    def envelope(self) -> tuple[int, int]:
        """Return the contiguous envelope required by downstream span contracts."""

        return self.segments[0][0], self.segments[-1][1]


def read_brat_archive(
    payload: bytes,
    *,
    max_members: int = 100_000,
    max_member_bytes: int = 32 * 1024 * 1024,
    max_total_bytes: int = 512 * 1024 * 1024,
) -> tuple[BratDocumentBundle, ...]:
    """Read paired ``.txt``/``.ann`` members without extracting archive paths."""

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        if len(members) > max_members:
            raise ValueError("BRAT archive exceeds member count limit")
        files = [member for member in members if not member.is_dir()]
        names = [member.filename for member in files]
        if len(names) != len(set(names)):
            raise ValueError("BRAT archive contains duplicate member names")
        total_size = sum(member.file_size for member in files)
        if total_size > max_total_bytes:
            raise ValueError("BRAT archive exceeds total uncompressed size limit")
        for member in files:
            if member.flag_bits & 0x1:
                raise ValueError(f"Encrypted BRAT member is not supported: {member.filename!r}")
            if member.file_size > max_member_bytes:
                raise ValueError(f"BRAT member {member.filename!r} exceeds size limit")

        by_name = {member.filename: member for member in files}
        pairs = [
            (member, by_name[f"{member.filename[:-4]}.ann"])
            for member in files
            if member.filename.endswith(".txt") and f"{member.filename[:-4]}.ann" in by_name
        ]
        if not pairs:
            raise ValueError("BRAT archive contains no paired .txt/.ann documents")

        bundles: list[BratDocumentBundle] = []
        for text_member, annotation_member in sorted(pairs, key=lambda pair: pair[0].filename):
            # SECURITY: members are read in-place and never extracted to filesystem paths.
            annotations = archive.read(annotation_member).decode("utf-8")
            decoded_text = archive.read(text_member).decode("utf-8")
            text, newline_normalization = _select_annotated_text(
                decoded_text,
                annotations,
                member_name=text_member.filename,
            )
            bundles.append(
                BratDocumentBundle(
                    text_member=text_member.filename,
                    annotation_member=annotation_member.filename,
                    text=text,
                    annotations=annotations,
                    newline_normalization=newline_normalization,
                )
            )
        return tuple(bundles)


def parse_brat_text_bound_annotations(
    payload: str,
    *,
    source_text: str,
) -> tuple[BratTextBoundAnnotation, ...]:
    """Parse and validate BRAT text-bound annotations against immutable source text."""

    annotations: list[BratTextBoundAnnotation] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line or raw_line.startswith(("#", "A", "M", "N", "R", "E", "*")):
            continue
        fields = raw_line.split("\t", 2)
        if len(fields) != 3:
            raise ValueError(f"Invalid BRAT line {line_number}: expected three tab fields")
        annotation_id, descriptor, annotated_text = fields
        if not annotation_id.startswith("T"):
            continue
        source_label, separator, raw_segments = descriptor.partition(" ")
        if not separator or not source_label or not raw_segments:
            raise ValueError(f"Invalid BRAT text-bound descriptor on line {line_number}")
        # Some historical corpora encode annotator notes with a T-prefixed ID. They are
        # valid BRAT metadata but are not text-bound training labels.
        if source_label == "AnnotatorNotes":
            continue
        segments = _parse_segments(raw_segments, line_number=line_number)
        _validate_segments(segments, source_text=source_text, line_number=line_number)
        source_segments_text = " ".join(source_text[start:end] for start, end in segments)
        if _normalized_whitespace(source_segments_text) != _normalized_whitespace(annotated_text):
            raise ValueError(
                f"BRAT annotated text mismatch on line {line_number}: {annotation_id!r}"
            )
        annotations.append(
            BratTextBoundAnnotation(
                annotation_id=annotation_id,
                source_label=source_label,
                segments=segments,
                annotated_text=annotated_text,
            )
        )
    return tuple(annotations)


def _parse_segments(
    raw_segments: str,
    *,
    line_number: int,
) -> tuple[tuple[int, int], ...]:
    segments: list[tuple[int, int]] = []
    for raw_segment in raw_segments.split(";"):
        coordinates = raw_segment.split()
        if len(coordinates) != 2:
            raise ValueError(f"Invalid BRAT segment on line {line_number}: {raw_segment!r}")
        try:
            start, end = int(coordinates[0]), int(coordinates[1])
        except ValueError as error:
            raise ValueError(
                f"Non-numeric BRAT segment on line {line_number}: {raw_segment!r}"
            ) from error
        segments.append((start, end))
    return tuple(segments)


def _validate_segments(
    segments: tuple[tuple[int, int], ...],
    *,
    source_text: str,
    line_number: int,
) -> None:
    previous_end = -1
    for start, end in segments:
        if start < 0 or end <= start or end > len(source_text):
            raise ValueError(f"Invalid BRAT offsets on line {line_number}: {(start, end)}")
        if start < previous_end:
            raise ValueError(f"Overlapping BRAT segments on line {line_number}")
        previous_end = end


def _normalized_whitespace(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def _select_annotated_text(
    decoded_text: str,
    annotations: str,
    *,
    member_name: str,
) -> tuple[str, str]:
    """Select the newline representation that the BRAT offsets actually target."""

    candidates = (
        ("none", decoded_text),
        ("universal_lf", decoded_text.replace("\r\n", "\n").replace("\r", "\n")),
    )
    failures: list[str] = []
    seen: set[str] = set()
    for mode, candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parse_brat_text_bound_annotations(annotations, source_text=candidate)
        except ValueError as error:
            failures.append(f"{mode}: {error}")
            continue
        # INVARIANT: the chosen parsed text is the exact coordinate system used by
        # source annotators. The original archive bytes remain immutable in CAS.
        return candidate, mode
    raise ValueError(
        f"BRAT offsets do not match any supported newline contract for {member_name!r}: "
        + "; ".join(failures)
    )
