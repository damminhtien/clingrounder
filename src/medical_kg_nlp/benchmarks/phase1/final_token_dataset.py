"""Compile all authorized Phase 1 supervision into final-fit token-classifier records.

This adapter is benchmark-owned because it understands the five external Phase 1 labels. The
generic training package only receives neutral ``MinedDocument`` and ``AnnotationProposal``
records with immutable source offsets.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.benchmarks.phase1.final_supervision import (
    Phase1FinalSupervisionCorpus,
)
from medical_kg_nlp.benchmarks.phase1.model_dataset import PHASE1_FIVE_TYPE_LABELS
from medical_kg_nlp.mining.io import write_json, write_jsonl
from medical_kg_nlp.mining.model_dataset import SpanDatasetConfig, export_span_dataset
from medical_kg_nlp.mining.records import (
    AccessClass,
    AnnotationLayer,
    AnnotationProposal,
    MinedDocument,
    RedistributionPolicy,
    ReviewStatus,
)
from medical_kg_nlp.benchmarks.phase1.ontology import PHASE1_RULE_BY_TYPE
from medical_kg_nlp.utils.hashing import sha256_file, sha256_text

__all__ = [
    "Phase1FinalTokenDatasetConfig",
    "build_phase1_final_token_dataset",
]

_SCHEMA_VERSION = "phase1-final-token-dataset.v1"
_LABEL_SOURCE = "phase1_final_authorized_supervision"


@dataclass(frozen=True, slots=True)
class Phase1FinalTokenDatasetConfig:
    """Deterministic chunking controls for an all-authorized final NER fit."""

    max_characters: int = 1600
    include_empty_chunks: bool = True
    empty_chunk_rate: float = 1.0

    def __post_init__(self) -> None:
        SpanDatasetConfig(
            max_characters=self.max_characters,
            entity_types=PHASE1_FIVE_TYPE_LABELS,
            include_empty_chunks=self.include_empty_chunks,
            empty_chunk_rate=self.empty_chunk_rate,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize every data-shaping choice into the immutable build key."""

        return {
            "max_characters": self.max_characters,
            "include_empty_chunks": self.include_empty_chunks,
            "empty_chunk_rate": self.empty_chunk_rate,
        }


def build_phase1_final_token_dataset(
    corpus: Phase1FinalSupervisionCorpus,
    output_dir: str | Path,
    *,
    config: Phase1FinalTokenDatasetConfig | None = None,
) -> dict[str, Any]:
    """Write an atomic five-type training dataset from all owner-authorized supervision.

    INVARIANT: ``corpus.reviewed`` already materializes LF child documents for the authorized
    CRLF bundle. Every exported offset therefore addresses precisely the text passed to the
    tokenizer, never the archive's original byte-oriented coordinate system.
    """

    active = config or Phase1FinalTokenDatasetConfig()
    documents, annotations = _to_neutral_records(corpus)
    label_counts = Counter(annotation.entity_type for annotation in annotations)
    if set(label_counts) != set(PHASE1_FIVE_TYPE_LABELS):
        missing = sorted(set(PHASE1_FIVE_TYPE_LABELS) - set(label_counts))
        raise ValueError(f"Final token dataset is missing Phase 1 labels: {missing}")

    build_contract = {
        "schema_version": _SCHEMA_VERSION,
        "final_supervision_fingerprint_sha256": corpus.manifest["fingerprint_sha256"],
        "document_count": len(documents),
        "source_document_counts": dict(sorted(Counter(corpus.source_by_document.values()).items())),
        "labels": list(PHASE1_FIVE_TYPE_LABELS),
        "config": active.to_dict(),
        "round2_included": False,
        "friend31_included": False,
    }
    build_key = sha256_text(_stable_json(build_contract))
    output = Path(output_dir)
    existing = _load_existing_build(output, build_key)
    if existing is not None:
        return existing

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        documents_path = staging / "documents.jsonl"
        annotations_path = staging / "annotations.jsonl"
        split_manifest_path = staging / "split_manifest.json"
        spans_path = staging / "spans.jsonl"
        dataset_manifest_path = staging / "manifest.json"
        write_jsonl(documents_path, (document.to_dict() for document in documents))
        write_jsonl(annotations_path, (annotation.to_dict() for annotation in annotations))
        split_manifest = {
            "schema_version": "phase1-final-token-split.v1",
            "purpose": "final_fit_all_authorized_supervision",
            "splits": {document.document_id: "train" for document in documents},
            "round2_included": False,
            "friend31_included": False,
        }
        write_json(split_manifest_path, split_manifest)
        dataset_manifest = export_span_dataset(
            documents,
            annotations,
            {document.document_id: "train" for document in documents},
            SpanDatasetConfig(entity_types=PHASE1_FIVE_TYPE_LABELS, **active.to_dict()),
            output_path=spans_path,
            manifest_path=dataset_manifest_path,
            documents_path=documents_path,
            annotations_path=annotations_path,
            split_manifest_path=split_manifest_path,
            manifest_root=staging,
        )
        if dataset_manifest["entity_count"] != len(annotations):
            raise RuntimeError("Final token dataset dropped an authorized annotation")
        if dataset_manifest["document_count"] != len(documents):
            raise RuntimeError("Final token dataset dropped an authorized document")
        output_hashes = {
            path.name: sha256_file(path)
            for path in (
                documents_path,
                annotations_path,
                split_manifest_path,
                spans_path,
                dataset_manifest_path,
            )
        }
        report = {
            "schema_version": _SCHEMA_VERSION,
            "build_key": build_key,
            "build_contract": build_contract,
            "dataset": {
                "document_count": len(documents),
                "annotation_count": len(annotations),
                "chunk_count": dataset_manifest["chunk_count"],
                "entity_type_counts": dataset_manifest["entity_type_counts"],
                "split_chunk_counts": dataset_manifest["split_chunk_counts"],
            },
            "outputs": dict(sorted(output_hashes.items())),
        }
        write_json(staging / "build_manifest.json", report)
        staging.replace(output)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _to_neutral_records(
    corpus: Phase1FinalSupervisionCorpus,
) -> tuple[tuple[MinedDocument, ...], tuple[AnnotationProposal, ...]]:
    fingerprint = str(corpus.manifest["fingerprint_sha256"])
    documents: list[MinedDocument] = []
    annotations: list[AnnotationProposal] = []
    for document_id, source_text in corpus.reviewed.source_texts.items():
        source_dataset = corpus.source_by_document[document_id]
        document = MinedDocument(
            document_id=document_id,
            text=source_text,
            language="vi",
            note_type="phase1_final_supervision",
            source_artifact_id=f"phase1-final-supervision:{fingerprint}",
            access_class=AccessClass.AUTHORIZED_PRIVATE,
            redistribution=RedistributionPolicy.PROHIBITED,
            hosted_processing_allowed=True,
            group_ids=(f"phase1-final:{document_id}",),
            metadata={
                "source_dataset": source_dataset,
                "supervision_fingerprint": fingerprint,
            },
        )
        documents.append(document)
        for row_index, row in enumerate(corpus.reviewed.gold_rows[document_id]):
            phase1_type = str(row.get("type", ""))
            rule = PHASE1_RULE_BY_TYPE.get(phase1_type)
            if rule is None:
                raise ValueError(
                    f"Final supervision {document_id}:{row_index} has invalid type {phase1_type!r}"
                )
            raw_position = row.get("position")
            if not isinstance(raw_position, list) or len(raw_position) != 2:
                raise ValueError(f"Final supervision {document_id}:{row_index} has invalid position")
            start, end = int(raw_position[0]), int(raw_position[1])
            text = str(row.get("text", ""))
            identity = f"{document_id}\0{row_index}\0{start}\0{end}\0{phase1_type}\0{text}"
            annotation = AnnotationProposal(
                annotation_id=(
                    "phase1-final:"
                    f"{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"
                ),
                document_id=document_id,
                span=(start, end),
                text=text,
                entity_type=rule.internal_type.value,
                assertions=tuple(str(value) for value in row.get("assertions", [])),
                concepts=(),
                confidence=1.0,
                layer=AnnotationLayer.GOLD,
                label_source=_LABEL_SOURCE,
                labeler_id="phase1_final_authorized_supervision",
                review_status=ReviewStatus.ACCEPTED,
                source_label=phase1_type,
                metadata={
                    "source_dataset": source_dataset,
                    "supervision_fingerprint": fingerprint,
                },
            )
            annotation.validate_offsets(document)
            annotations.append(annotation)
    return tuple(documents), tuple(annotations)


def _load_existing_build(output: Path, build_key: str) -> dict[str, Any] | None:
    if not output.exists():
        return None
    manifest_path = output / "build_manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"Dataset output exists without a build manifest: {output}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("build_key") != build_key:
        raise FileExistsError(f"Dataset output belongs to a different build: {output}")
    outputs = raw.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Existing final token dataset has no output fingerprints")
    for relative_path, expected_sha256 in outputs.items():
        path = output / str(relative_path)
        if not path.is_file() or sha256_file(path) != str(expected_sha256):
            raise ValueError(f"Existing final token dataset output failed fingerprint check: {path}")
    return {str(key): value for key, value in raw.items()}


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
