"""Opt-in checks for isolated terminology startup and memory measurements."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clingrounder.terminology.benchmark import benchmark_terminology_repositories
from clingrounder.terminology.index_builder import build_terminology_index


@pytest.mark.benchmark
def test_benchmark_compares_identical_merged_sources(tmp_path: Path) -> None:
    first = _write_source(tmp_path / "icd.jsonl", "ICD:E11.9", "E11.9", "ICD-10")
    second = _write_source(tmp_path / "rx.jsonl", "RX:6809", "6809", "RxNorm")
    manifest = build_terminology_index((first, second), cache_dir=tmp_path / "cache")

    report = benchmark_terminology_repositories((first, second), manifest.index_path)

    assert report["memory"]["concept_count"] == 2
    assert report["sqlite"]["concept_count"] == 2
    assert report["sources"] == [str(first), str(second)]


def _write_source(path: Path, concept_id: str, code: str, system: str) -> Path:
    semantic_type = "DRUG" if system == "RxNorm" else "DISEASE"
    row = {
        "concept_id": concept_id,
        "code": code,
        "code_system": system,
        "canonical_name": concept_id,
        "semantic_type": semantic_type,
        "source": "benchmark-test",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path
