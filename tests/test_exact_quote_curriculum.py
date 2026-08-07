"""Tests for licensed, source-label-preserving exact-quote curricula."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from clingrounder.mining.exact_quote_curriculum import (
    ExactQuoteCurriculumConfig,
    build_exact_quote_curriculum,
)


def test_curriculum_uses_train_only_and_preserves_broad_labels(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path, source_id="vietbioner", quarantine=False)
    spans, manifest = _write_spans(tmp_path)

    report = build_exact_quote_curriculum(
        ExactQuoteCurriculumConfig(
            source_id="vietbioner",
            source_registry_path=registry,
            spans_path=spans,
            spans_manifest_path=manifest,
            output_dir=tmp_path / "output",
            allowed_labels=("FINDING", "PROCEDURE"),
        )
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "output" / "curriculum.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["targets"] == [
        {"label": "PROCEDURE", "text": "X-quang"},
        {"label": "FINDING", "text": "lao phổi"},
    ]
    assert "development mention" not in json.dumps(rows, ensure_ascii=False)
    assert report["policy"]["target_task_crosswalk_applied"] is False
    assert report["output"]["label_counts"] == {"FINDING": 1, "PROCEDURE": 1}


def test_curriculum_rejects_quarantined_source(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path, source_id="vietmed_ner", quarantine=True)
    spans, manifest = _write_spans(tmp_path)

    with pytest.raises(ValueError, match="Quarantined source"):
        build_exact_quote_curriculum(
            ExactQuoteCurriculumConfig(
                source_id="vietmed_ner",
                source_registry_path=registry,
                spans_path=spans,
                spans_manifest_path=manifest,
                output_dir=tmp_path / "output",
                allowed_labels=("FINDING", "PROCEDURE"),
            )
        )


def test_curriculum_rejects_offset_drift(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path, source_id="vietbioner", quarantine=False)
    spans, manifest = _write_spans(tmp_path, corrupt_offset=True)

    with pytest.raises(ValueError, match="Raw span mismatch"):
        build_exact_quote_curriculum(
            ExactQuoteCurriculumConfig(
                source_id="vietbioner",
                source_registry_path=registry,
                spans_path=spans,
                spans_manifest_path=manifest,
                output_dir=tmp_path / "output",
                allowed_labels=("FINDING", "PROCEDURE"),
            )
        )


def _write_registry(
    root: Path,
    *,
    source_id: str,
    quarantine: bool,
) -> Path:
    path = root / "registry.yaml"
    access_class = "quarantine" if quarantine else "open"
    allowed_uses = "[license_review]" if quarantine else "[entity_training]"
    path.write_text(
        f"""schema_version: medical-source-registry.v2
resources:
  - id: {source_id}
    name: Test source
    category: vietnamese_medical_ner
    version: pinned-v1
    version_policy: pinned
    access_class: {access_class}
    license_id: {"verify_license" if quarantine else "CC-BY-4.0"}
    license_url: https://example.test/license
    redistribution: {"unknown" if quarantine else "attribution"}
    hosted_processing_allowed: false
    retention: {"local_only" if quarantine else "immutable"}
    connector: local_archive
    parser: brat
    allowed_uses: {allowed_uses}
""",
        encoding="utf-8",
    )
    return path


def _write_spans(
    root: Path,
    *,
    corrupt_offset: bool = False,
) -> tuple[Path, Path]:
    spans = root / "spans.jsonl"
    train_text = "X-quang xác nhận lao phổi"
    rows = [
        {
            "document_id": "vi:train",
            "entities": [
                {
                    "annotation_id": "a1",
                    "start": 0,
                    "end": 7,
                    "text": "X-quang",
                    "label": "PROCEDURE",
                },
                {
                    "annotation_id": "a2",
                    "start": 16 if corrupt_offset else 17,
                    "end": 25,
                    "text": "lao phổi",
                    "label": "FINDING",
                },
            ],
            "record_id": "r1",
            "source_artifact_id": "artifact:1",
            "split": "train",
            "text": train_text,
            "text_sha256": hashlib.sha256(train_text.encode()).hexdigest(),
        },
        {
            "document_id": "vi:development",
            "entities": [],
            "record_id": "r2",
            "source_artifact_id": "artifact:1",
            "split": "development",
            "text": "development mention",
            "text_sha256": hashlib.sha256(b"development mention").hexdigest(),
        },
    ]
    spans.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(spans.read_bytes()).hexdigest()
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "mined-span-dataset.v1",
                "output_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    return spans, manifest
