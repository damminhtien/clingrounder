"""Load reviewed Phase 1 documents under an explicit training-governance contract."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.benchmarks.phase1.split_contract import (
    Phase1SplitContract,
    phase1_document_sort_key,
)
from medical_kg_nlp.benchmarks.phase1.training_governance import (
    load_phase1_training_governance,
)
from medical_kg_nlp.utils.hashing import sha256_file

__all__ = ["Phase1ReviewedCorpus", "load_phase1_reviewed_corpus"]


@dataclass(frozen=True, slots=True)
class Phase1ReviewedCorpus:
    """Raw source text and reviewed rows for explicitly allowed split IDs."""

    source_texts: Mapping[str, str]
    gold_rows: Mapping[str, tuple[Mapping[str, Any], ...]]
    split_by_document: Mapping[str, str]

    def document_ids(self, split: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                (
                    document_id
                    for document_id, value in self.split_by_document.items()
                    if value == split
                ),
                key=phase1_document_sort_key,
            )
        )


def load_phase1_reviewed_corpus(
    contract: Phase1SplitContract,
    *,
    input_dir: str | Path,
    gold_dir: str | Path,
    frozen_manifest_path: str | Path,
    splits: Sequence[str] = ("train", "development"),
    training_governance_path: str | Path | None = None,
) -> Phase1ReviewedCorpus:
    """Load reviewed records while preventing accidental sealed-holdout access.

    The legacy split stays available for diagnostics. Final-fit callers may request all three
    splits only after pinning the governance policy that authorizes `manual_gold: train_all`.
    """

    invalid = set(splits) - {"train", "development", "holdout"}
    if invalid:
        raise ValueError(f"Reviewed corpus cannot open splits {sorted(invalid)}")
    if "holdout" in splits:
        if training_governance_path is None:
            raise ValueError("Opening holdout labels requires Phase 1 training governance")
        governance = load_phase1_training_governance(training_governance_path)
        if governance.manual_gold.usage != "train_all":
            raise ValueError("Training governance does not authorize full manual-gold fit")
    frozen_path = Path(frozen_manifest_path)
    if sha256_file(frozen_path) != contract.frozen_manifest_sha256:
        raise ValueError("Frozen split manifest changed after contract verification")
    frozen = _read_mapping(frozen_path)
    assignments = _assignment_by_document(frozen)
    source_root = Path(input_dir)
    gold_root = Path(gold_dir)
    source_texts: dict[str, str] = {}
    gold_rows: dict[str, tuple[Mapping[str, Any], ...]] = {}
    split_by_document: dict[str, str] = {}

    for split in splits:
        for document_id in contract.ids(split):
            assignment = assignments.get(document_id)
            if assignment is None:
                raise ValueError(f"Frozen manifest has no assignment for {document_id}")
            source_path = source_root / f"{document_id}.txt"
            gold_path = gold_root / f"{document_id}.json"
            _verify_hash(
                source_path,
                str(assignment.get("document_sha256", "")),
            )
            _verify_hash(gold_path, str(assignment.get("gold_sha256", "")))
            source_texts[document_id] = source_path.read_text(encoding="utf-8")
            gold_rows[document_id] = tuple(_read_rows(gold_path))
            split_by_document[document_id] = split
    return Phase1ReviewedCorpus(
        source_texts=source_texts,
        gold_rows=gold_rows,
        split_by_document=split_by_document,
    )


def _assignment_by_document(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    raw = manifest.get("assignments")
    if not isinstance(raw, list):
        raise ValueError("Frozen split manifest has no assignments")
    assignments: dict[str, Mapping[str, Any]] = {}
    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError("Frozen assignment must be an object")
        document_id = str(row.get("document_id", ""))
        if not document_id or document_id in assignments:
            raise ValueError("Frozen assignments contain a missing or duplicate ID")
        assignments[document_id] = row
    return assignments


def _verify_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not expected or sha256_file(path) != expected:
        raise ValueError(f"Frozen source fingerprint mismatch for {path}")


def _read_rows(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(row, Mapping) for row in payload
    ):
        raise ValueError(f"{path}: expected a JSON entity list")
    return payload


def _read_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return payload
