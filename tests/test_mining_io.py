"""Atomic streaming IO tests for large mining manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from medical_kg_nlp.mining.io import write_jsonl


def test_write_jsonl_streams_with_the_same_deterministic_fingerprint(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    expected = b'{"a": 1, "b": 2}\n{"text": "thuoc"}\n'

    fingerprint = write_jsonl(
        target,
        (row for row in ({"b": 2, "a": 1}, {"text": "thuoc"})),
    )

    assert target.read_bytes() == expected
    assert fingerprint == hashlib.sha256(expected).hexdigest()


def test_write_jsonl_keeps_previous_file_when_stream_fails(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    target.write_text("previous\n", encoding="utf-8")

    def broken_rows():
        yield {"row": 1}
        raise RuntimeError("source failed")

    with pytest.raises(RuntimeError, match="source failed"):
        write_jsonl(target, broken_rows())

    assert target.read_text(encoding="utf-8") == "previous\n"
