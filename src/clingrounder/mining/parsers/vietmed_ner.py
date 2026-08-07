"""Parse pinned VietMed-NER Parquet shards without materializing audio columns."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Iterable, Sequence
from typing import Any

from clingrounder.mining.parsers.base import ArtifactParserAdapter
from clingrounder.mining.ports import ArtifactStorePort
from clingrounder.mining.records import MinedDocument, SourceArtifact

__all__ = ["VietMedNerParquetParser", "align_vietmed_tokens"]

_PROJECTED_COLUMNS = ("words", "labels", "text", "duration")


class VietMedNerParquetParser(ArtifactParserAdapter):
    """Project immutable transcript/BIO columns from one VietMed-NER shard."""

    parser_id = "vietmed_ner_parquet"
    parser_revision = "1"
    max_artifact_bytes = 1024 * 1024 * 1024

    def parse(
        self,
        artifact: SourceArtifact,
        *,
        store: ArtifactStorePort,
    ) -> Iterable[MinedDocument]:
        parquet = _load_parquet_module()
        split = artifact.metadata.get("split", "").strip()
        if split not in {"train", "validation", "test"}:
            raise ValueError("VietMed-NER artifact metadata requires a known split")
        with store.open(artifact.object.sha256) as stream:
            parquet_file = parquet.ParquetFile(stream)
            row_index = 0
            # SCALING: column projection skips the embedded audio bytes. Batches bound memory
            # while the content-addressed Parquet remains the immutable source artifact.
            for batch in parquet_file.iter_batches(
                columns=list(_PROJECTED_COLUMNS),
                batch_size=256,
            ):
                for raw in batch.to_pylist():
                    text = str(raw.get("text", ""))
                    words = _string_sequence(raw.get("words"), field="words")
                    labels = _string_sequence(raw.get("labels"), field="labels")
                    if len(words) != len(labels):
                        raise ValueError(
                            f"VietMed-NER words/labels length mismatch at {split}:{row_index}"
                        )
                    offsets = align_vietmed_tokens(text, words)
                    external_id = f"{split}:{row_index:06d}"
                    unit_sha256 = hashlib.sha256(
                        (
                            f"{artifact.object.sha256}\0{external_id}\0"
                            f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
                        ).encode("utf-8")
                    ).hexdigest()
                    yield self.make_document(
                        artifact,
                        external_id=external_id,
                        text=text,
                        language="vi",
                        note_type="spoken_medical_transcript",
                        group_ids=(f"vietmed_ner:{external_id}",),
                        source_unit_sha256=unit_sha256,
                        metadata={
                            "source_split": split,
                            "token_offsets": json.dumps(
                                offsets,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            "source_bio_labels": json.dumps(
                                labels,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            "duration_seconds": str(raw.get("duration", "")),
                        },
                    )
                    row_index += 1


def align_vietmed_tokens(
    source_text: str,
    words: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    """Align source words monotonically to the untouched transcript."""

    offsets: list[tuple[int, int]] = []
    cursor = 0
    for index, word in enumerate(words):
        if not word:
            raise ValueError(f"VietMed-NER word {index} is empty")
        start = source_text.find(word, cursor)
        if start < 0:
            raise ValueError(
                f"Cannot align VietMed-NER word {word!r} after character {cursor}"
            )
        gap = source_text[cursor:start]
        if gap and not gap.isspace():
            raise ValueError(
                f"Non-whitespace token alignment gap before word {index}: {gap!r}"
            )
        end = start + len(word)
        # INVARIANT: every source token remains an exact slice of the immutable transcript.
        if source_text[start:end] != word:
            raise ValueError(f"VietMed-NER token offset mismatch for {word!r}")
        offsets.append((start, end))
        cursor = end
    if source_text[cursor:].strip():
        raise ValueError("VietMed-NER words do not cover the transcript suffix")
    return tuple(offsets)


def _load_parquet_module() -> Any:
    try:
        return importlib.import_module("pyarrow.parquet")
    except ImportError as error:
        raise RuntimeError(
            "VietMed-NER Parquet parsing requires the project data extra"
        ) from error


def _string_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"VietMed-NER {field} must be a sequence")
    return tuple(str(item) for item in value)
