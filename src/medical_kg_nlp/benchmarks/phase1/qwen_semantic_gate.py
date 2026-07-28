"""Reviewed semantic gate for high-precision exact-quote Qwen proposals.

The gate is deliberately surface and context based. It never records a
document identifier or absolute offset, so reviewed decisions remain reusable
across notes instead of becoming a hidden answer overlay.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from medical_kg_nlp.mining.io import write_json, write_text
from medical_kg_nlp.ontology.phase1 import PHASE1_ALLOWED_TYPES
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = ["filter_high_precision_qwen_proposals"]

_BLOCKED_MENTIONS = frozenset({"g6pd", "thuốc", "****"})
_REVIEWED_TYPES = {
    "béo phì": "CHẨN_ĐOÁN",
    "bệnh dại": "CHẨN_ĐOÁN",
    "bệnh gout": "CHẨN_ĐOÁN",
    "bệnhgout": "CHẨN_ĐOÁN",
    "bệnh mạch vành": "CHẨN_ĐOÁN",
    "hẹp lòng mạch": "CHẨN_ĐOÁN",
    "loãng xương": "CHẨN_ĐOÁN",
    "mày đay vô căn": "CHẨN_ĐOÁN",
    "nấm bẹn": "CHẨN_ĐOÁN",
    "rối loạn lipid máu": "CHẨN_ĐOÁN",
    "sâu răng": "CHẨN_ĐOÁN",
    "tiểu đường": "CHẨN_ĐOÁN",
    "ba stent mật kim loại đã được đặt": "KẾT_QUẢ_XÉT_NGHIỆM",
    "men gan tăng": "KẾT_QUẢ_XÉT_NGHIỆM",
    "tăng men gan": "KẾT_QUẢ_XÉT_NGHIỆM",
    "siêu âm tim": "TÊN_XÉT_NGHIỆM",
    "nitramyl": "THUỐC",
    "chảy máu cam": "TRIỆU_CHỨNG",
    "da dầu": "TRIỆU_CHỨNG",
    "giảm lượng nước tiểu": "TRIỆU_CHỨNG",
    "mụn": "TRIỆU_CHỨNG",
    "sốt": "TRIỆU_CHỨNG",
    "trứng cá": "TRIỆU_CHỨNG",
    "đau bụng": "TRIỆU_CHỨNG",
}
_REVIEWED_PREFIX_TYPES = {
    "st chênh xuống": "KẾT_QUẢ_XÉT_NGHIỆM",
}
_BLOOD_PRESSURE_LEFT = re.compile(r"(?i)\bđo\s*$")
_BLOOD_PRESSURE_RIGHT = re.compile(
    r"(?i)^\s*(?:(?::|=)\s*)?\d{2,3}/\d{2,3}\b"
)


def filter_high_precision_qwen_proposals(
    source_dir: str | Path,
    source_text_by_doc: Mapping[str, str],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Keep only reviewed semantic families and preserve exact raw spans.

    INVARIANT: accepted text and position are copied only after validating
    ``source_text[start:end] == text``. Metadata fields begin empty because
    this is an entity-only WER probe.
    """

    source_root = Path(source_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    accepted_by_type: Counter[str] = Counter()
    document_count = 0
    for document_id in sorted(source_text_by_doc, key=_document_sort_key):
        source_path = source_root / f"{document_id}.json"
        if not source_path.is_file():
            raise ValueError(f"Missing Qwen proposal file: {source_path}")
        rows = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Qwen proposal file must contain a list: {source_path}")
        source_text = source_text_by_doc[document_id]
        output: list[dict[str, Any]] = []
        seen: set[tuple[int, int, str]] = set()
        for row_index, row in enumerate(rows):
            counters["input"] += 1
            accepted = _review_row(
                row,
                source_text,
                document_id=document_id,
                row_index=row_index,
            )
            if accepted is None:
                counters["blocked"] += 1
                continue
            position = accepted["position"]
            identity = (position[0], position[1], accepted["type"])
            if identity in seen:
                counters["duplicate"] += 1
                continue
            seen.add(identity)
            output.append(accepted)
            accepted_by_type[accepted["type"]] += 1
        output.sort(
            key=lambda row: (
                row["position"][0],
                row["position"][1],
                row["type"],
            )
        )
        write_text(
            destination / f"{document_id}.json",
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        )
        counters["output"] += len(output)
        document_count += 1

    manifest = {
        "schema_version": "phase1-qwen-semantic-gate.v1",
        "source": str(source_root),
        "source_sha256": _directory_fingerprint(source_root),
        "document_count": document_count,
        "policy": {
            "blocked_mentions": sorted(_BLOCKED_MENTIONS),
            "reviewed_exact_mentions": dict(sorted(_REVIEWED_TYPES.items())),
            "reviewed_prefix_mentions": dict(
                sorted(_REVIEWED_PREFIX_TYPES.items())
            ),
            "document_specific_rules": False,
        },
        "counts": dict(sorted(counters.items())),
        "output_type_counts": dict(sorted(accepted_by_type.items())),
    }
    write_json(destination.parent / "review_manifest.json", manifest)
    return manifest


def _review_row(
    raw: object,
    source_text: str,
    *,
    document_id: str,
    row_index: int,
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        raise ValueError(f"{document_id}:{row_index}: proposal must be a mapping")
    position = raw.get("position")
    text = raw.get("text")
    if (
        not isinstance(position, list)
        or len(position) != 2
        or not isinstance(text, str)
    ):
        raise ValueError(f"{document_id}:{row_index}: invalid text/position")
    start, end = int(position[0]), int(position[1])
    if start < 0 or end <= start or source_text[start:end] != text:
        raise ValueError(f"{document_id}:{row_index}: raw offset mismatch")
    normalized = normalize_for_match(text)
    if normalized in _BLOCKED_MENTIONS or set(normalized) <= {"*"}:
        return None
    entity_type = _REVIEWED_TYPES.get(normalized)
    if entity_type is None:
        entity_type = next(
            (
                candidate_type
                for prefix, candidate_type in _REVIEWED_PREFIX_TYPES.items()
                if normalized.startswith(prefix)
            ),
            None,
        )
    if normalized in {"ha", "huyết áp"}:
        left = source_text[max(0, start - 8) : start]
        right = source_text[end : min(len(source_text), end + 16)]
        if _BLOOD_PRESSURE_LEFT.search(left) or _BLOOD_PRESSURE_RIGHT.search(right):
            entity_type = "TÊN_XÉT_NGHIỆM"
    if entity_type is None:
        return None
    if entity_type not in PHASE1_ALLOWED_TYPES:
        raise ValueError(f"Reviewed type is outside Phase 1 schema: {entity_type}")
    return {
        "text": text,
        "type": entity_type,
        "assertions": [],
        "candidates": [],
        "position": [start, end],
    }


def _directory_fingerprint(path: Path) -> str:
    digest_rows = [
        f"{child.name}\0{sha256_file(child)}"
        for child in sorted(path.glob("*.json"))
        if child.stem.isdigit()
    ]
    if not digest_rows:
        raise ValueError(f"No numeric Qwen proposal files found under {path}")
    import hashlib

    return hashlib.sha256("\n".join(digest_rows).encode("utf-8")).hexdigest()


def _document_sort_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (10**9, value)
