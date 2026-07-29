"""Build provenance-separated corpora for domain-adaptive pretraining.

The builder writes one immutable JSONL file per lane. Competition input may be
used as an explicitly authorized *unlabeled MLM lane*, but it is never merged
into supervised or terminology-pair records.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from medical_kg_nlp.mining.io import load_documents, write_json, write_jsonl
from medical_kg_nlp.mining.records import AccessClass, MinedDocument
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.io import read_yaml

__all__ = [
    "DaptCorpusBuildSpec",
    "DaptCorpusLaneKind",
    "DaptCorpusLaneSpec",
    "build_dapt_corpus",
    "load_dapt_corpus_build_spec",
]

_LANE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{1,63}")


class DaptCorpusLaneKind(StrEnum):
    """Training-policy class for one corpus lane."""

    OPEN_UNLABELED = "open_unlabeled"
    ROUND2_UNLABELED = "round2_unlabeled"


@dataclass(frozen=True, slots=True)
class DaptCorpusLaneSpec:
    """One source-pinned document lane used only for language modeling."""

    lane_id: str
    kind: DaptCorpusLaneKind
    documents_path: Path
    source_manifest_path: Path
    languages: tuple[str, ...] = ("vi",)
    sampling_weight: float = 1.0
    minimum_characters: int = 20

    def __post_init__(self) -> None:
        if _LANE_ID.fullmatch(self.lane_id) is None:
            raise ValueError(f"Invalid DAPT lane_id: {self.lane_id!r}")
        if not self.documents_path.is_file():
            raise ValueError(f"DAPT documents do not exist: {self.documents_path}")
        if not self.source_manifest_path.is_file():
            raise ValueError(
                f"DAPT source manifest does not exist: {self.source_manifest_path}"
            )
        if not self.languages or any(not value.strip() for value in self.languages):
            raise ValueError("DAPT lane languages must be non-empty")
        if self.sampling_weight <= 0:
            raise ValueError("DAPT lane sampling_weight must be positive")
        if self.minimum_characters < 1:
            raise ValueError("DAPT lane minimum_characters must be positive")


@dataclass(frozen=True, slots=True)
class DaptCorpusBuildSpec:
    """Portable identity for a multi-lane DAPT corpus build."""

    schema_version: str
    build_id: str
    config_path: Path
    run_root: Path
    output_dir: Path
    lanes: tuple[DaptCorpusLaneSpec, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "xlmr-dapt-corpus-build.v1":
            raise ValueError("Unsupported DAPT corpus-build schema")
        if not self.build_id.strip():
            raise ValueError("DAPT build_id must be non-empty")
        if not self.lanes:
            raise ValueError("DAPT corpus requires at least one lane")
        lane_ids = [lane.lane_id for lane in self.lanes]
        if len(lane_ids) != len(set(lane_ids)):
            raise ValueError("DAPT lane_id values must be unique")
        self.relative_path(self.config_path)
        self.relative_path(self.output_dir)
        for lane in self.lanes:
            self.relative_path(lane.documents_path)
            self.relative_path(lane.source_manifest_path)

    def relative_path(self, path: str | Path) -> str:
        """Return one path relative to the portable run root."""

        try:
            return Path(path).resolve().relative_to(self.run_root).as_posix()
        except ValueError as error:
            raise ValueError(f"DAPT path escapes run_root: {path}") from error


def load_dapt_corpus_build_spec(path: str | Path) -> DaptCorpusBuildSpec:
    """Load a strict DAPT corpus YAML without importing model dependencies."""

    config_path = Path(path).resolve()
    raw = read_yaml(config_path)
    run_root = _resolve(config_path.parent, _required_string(raw, "run_root"))
    lanes_raw = raw.get("lanes")
    if not isinstance(lanes_raw, list):
        raise ValueError("DAPT corpus lanes must be a list")
    lanes = tuple(_lane_spec(value, run_root) for value in lanes_raw)
    return DaptCorpusBuildSpec(
        schema_version=_required_string(raw, "schema_version"),
        build_id=_required_string(raw, "build_id"),
        config_path=config_path,
        run_root=run_root,
        output_dir=_resolve(run_root, _required_string(raw, "output_dir")),
        lanes=lanes,
    )


def build_dapt_corpus(spec: DaptCorpusBuildSpec) -> dict[str, Any]:
    """Write deterministic, deduplicated lane files and their provenance."""

    output = spec.output_dir
    lanes_dir = output / "lanes"
    lanes_dir.mkdir(parents=True, exist_ok=True)
    seen_text: dict[str, str] = {}
    lane_reports: list[dict[str, Any]] = []
    total_records = 0

    # PRIVACY: open text wins cross-lane duplicates so restricted Round 2 text
    # cannot silently become the canonical copy of an already-open document.
    ordered_lanes = sorted(
        spec.lanes,
        key=lambda lane: (
            lane.kind == DaptCorpusLaneKind.ROUND2_UNLABELED,
            lane.lane_id,
        ),
    )
    for lane in ordered_lanes:
        rows, counters = _lane_rows(lane, seen_text)
        lane_path = lanes_dir / f"{lane.lane_id}.jsonl"
        lane_sha256 = write_jsonl(lane_path, rows)
        total_records += len(rows)
        lane_reports.append(
            {
                "lane_id": lane.lane_id,
                "kind": lane.kind.value,
                "path": spec.relative_path(lane_path),
                "sha256": lane_sha256,
                "record_count": len(rows),
                "sampling_weight": lane.sampling_weight,
                "languages": list(lane.languages),
                "source": {
                    "documents": spec.relative_path(lane.documents_path),
                    "documents_sha256": sha256_file(lane.documents_path),
                    "manifest": spec.relative_path(lane.source_manifest_path),
                    "manifest_sha256": sha256_file(lane.source_manifest_path),
                },
                "counters": dict(sorted(counters.items())),
            }
        )

    manifest = {
        "schema_version": "xlmr-dapt-corpus.v1",
        "build_id": spec.build_id,
        "build_spec": {
            "path": spec.relative_path(spec.config_path),
            "sha256": sha256_file(spec.config_path),
        },
        "record_count": total_records,
        "lanes": lane_reports,
        "round2_unlabeled_policy": {
            "lane_ids": [
                lane.lane_id
                for lane in ordered_lanes
                if lane.kind == DaptCorpusLaneKind.ROUND2_UNLABELED
            ],
            "supervision": "none",
            "allowed_objectives": ["masked_language_modeling"],
            "forbidden_objectives": [
                "entity_supervision",
                "pseudo_labeling",
                "synonym_contrastive",
                "threshold_calibration",
            ],
        },
        "deduplication": {
            "identity": "raw_text_sha256",
            "cross_lane_precedence": "open_before_round2",
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _lane_rows(
    lane: DaptCorpusLaneSpec,
    seen_text: dict[str, str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for document in sorted(
        load_documents(lane.documents_path),
        key=lambda item: item.document_id,
    ):
        counters["documents_seen"] += 1
        reason = _document_rejection_reason(document, lane)
        if reason is not None:
            counters[f"rejected.{reason}"] += 1
            continue
        previous_lane = seen_text.get(document.text_sha256)
        if previous_lane is not None:
            counters[
                "deduplicated.within_lane"
                if previous_lane == lane.lane_id
                else "deduplicated.cross_lane"
            ] += 1
            continue
        seen_text[document.text_sha256] = lane.lane_id
        rows.append(_corpus_row(document, lane))
        counters["accepted"] += 1
    return rows, counters


def _document_rejection_reason(
    document: MinedDocument,
    lane: DaptCorpusLaneSpec,
) -> str | None:
    if document.language not in lane.languages:
        return "language"
    if len(document.text.strip()) < lane.minimum_characters:
        return "too_short"
    if lane.kind == DaptCorpusLaneKind.ROUND2_UNLABELED:
        if document.access_class != AccessClass.AUTHORIZED_PRIVATE:
            return "round2_access_class"
        if not document.hosted_processing_allowed:
            return "round2_hosted_processing"
    elif document.access_class not in {
        AccessClass.OPEN,
        AccessClass.OPEN_WITH_TERMS,
    }:
        return "not_open"
    return None


def _corpus_row(
    document: MinedDocument,
    lane: DaptCorpusLaneSpec,
) -> dict[str, Any]:
    identity = hashlib.sha256(
        f"{lane.lane_id}\0{document.document_id}\0{document.text_sha256}".encode()
    ).hexdigest()[:24]
    return {
        "record_id": f"dapt:{identity}",
        "lane_id": lane.lane_id,
        "lane_kind": lane.kind.value,
        "document_id": document.document_id,
        "source_artifact_id": document.source_artifact_id,
        "text": document.text,
        "text_sha256": document.text_sha256,
        "language": document.language,
        "note_type": document.note_type,
        # INVARIANT: DAPT corpus records carry no annotation or pseudo-label field.
        "supervision": "none",
        "objective": "masked_language_modeling",
    }


def _lane_spec(value: object, run_root: Path) -> DaptCorpusLaneSpec:
    if not isinstance(value, dict):
        raise ValueError("Each DAPT corpus lane must be a mapping")
    languages = value.get("languages", ["vi"])
    if not isinstance(languages, list):
        raise ValueError("DAPT lane languages must be a list")
    return DaptCorpusLaneSpec(
        lane_id=_required_string(value, "lane_id"),
        kind=DaptCorpusLaneKind(_required_string(value, "kind")),
        documents_path=_resolve(
            run_root,
            _required_string(value, "documents"),
        ),
        source_manifest_path=_resolve(
            run_root,
            _required_string(value, "source_manifest"),
        ),
        languages=tuple(str(item) for item in languages),
        sampling_weight=float(value.get("sampling_weight", 1.0)),
        minimum_characters=int(value.get("minimum_characters", 20)),
    )


def _required_string(raw: dict[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()
