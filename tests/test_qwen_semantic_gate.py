"""Reviewed semantic gate tests for high-precision Qwen additions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medical_kg_nlp.benchmarks.phase1.qwen_semantic_gate import (
    filter_high_precision_qwen_proposals,
)


def test_gate_blocks_noise_retypes_concepts_and_requires_bp_context(
    tmp_path: Path,
) -> None:
    text = "G6PD; bệnh dại; tăng HA; đo HA: 130/76; tăng men gan; thuốc"
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        _row(text, "G6PD", "THUỐC"),
        _row(text, "bệnh dại", "TRIỆU_CHỨNG"),
        _row(text, "HA", "TÊN_XÉT_NGHIỆM", occurrence=0),
        _row(text, "HA", "CHẨN_ĐOÁN", occurrence=1),
        _row(text, "tăng men gan", "CHẨN_ĐOÁN"),
        _row(text, "thuốc", "THUỐC"),
    ]
    (source / "1.json").write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "review" / "proposals"

    report = filter_high_precision_qwen_proposals(
        source,
        {"1": text},
        output,
    )
    accepted = json.loads((output / "1.json").read_text(encoding="utf-8"))

    assert [(row["text"], row["type"]) for row in accepted] == [
        ("bệnh dại", "CHẨN_ĐOÁN"),
        ("HA", "TÊN_XÉT_NGHIỆM"),
        ("tăng men gan", "KẾT_QUẢ_XÉT_NGHIỆM"),
    ]
    assert report["policy"]["document_specific_rules"] is False
    assert report["counts"]["output"] == 3


def test_gate_rejects_offset_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "1.json").write_text(
        json.dumps(
            [
                {
                    "text": "bệnh dại",
                    "type": "CHẨN_ĐOÁN",
                    "position": [1, 9],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="raw offset mismatch"):
        filter_high_precision_qwen_proposals(
            source,
            {"1": "bệnh dại"},
            tmp_path / "output",
        )


def _row(
    text: str,
    mention: str,
    entity_type: str,
    *,
    occurrence: int = 0,
) -> dict[str, object]:
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(mention, start + 1)
    return {
        "text": mention,
        "type": entity_type,
        "assertions": [],
        "candidates": [],
        "position": [start, start + len(mention)],
    }
