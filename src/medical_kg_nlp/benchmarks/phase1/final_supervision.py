"""Compose the governed all-manual and authorized-ground-truth corpus for final Phase 1 fit."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.benchmarks.phase1.authorized_ground_truth import (
    load_phase1_authorized_ground_truth,
)
from medical_kg_nlp.benchmarks.phase1.reviewed_corpus import (
    Phase1ReviewedCorpus,
    load_phase1_reviewed_corpus,
)
from medical_kg_nlp.benchmarks.phase1.split_contract import (
    load_phase1_split_contract,
    phase1_document_sort_key,
)
from medical_kg_nlp.utils.hashing import sha256_file

__all__ = ["Phase1FinalSupervisionCorpus", "load_phase1_final_supervision_corpus"]


@dataclass(frozen=True, slots=True)
class Phase1FinalSupervisionCorpus:
    """Final-fit source texts, exact labels, and source provenance for every authorized record."""

    reviewed: Phase1ReviewedCorpus
    source_by_document: Mapping[str, str]
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        if set(self.reviewed.source_texts) != set(self.source_by_document):
            raise ValueError("Final supervision source provenance must cover every document")
        if set(self.reviewed.split_by_document.values()) != {"train"}:
            raise ValueError("Final supervision corpus must assign every authorized row to final train")
        if any(source not in {"manual_gold", "authorized_ground_truth"} for source in self.source_by_document.values()):
            raise ValueError("Final supervision corpus has an unknown source")


def load_phase1_final_supervision_corpus(
    *,
    governance_path: str | Path,
    model_split_manifest_path: str | Path,
    frozen_split_manifest_path: str | Path,
    manual_input_dir: str | Path = "data/raw/input",
    manual_gold_dir: str | Path = "data/manual_gold",
    authorized_archive_path: str | Path | None = None,
) -> Phase1FinalSupervisionCorpus:
    """Load final-fit supervision without allowing Round 2 or external prediction artifacts.

    The manual corpus is read through its frozen split contract solely to validate source hashes.
    The final model consumes all 100 manually reviewed records, as authorized by governance.
    """

    contract = load_phase1_split_contract(
        model_split_manifest_path,
        frozen_split_manifest_path,
    )
    manual = load_phase1_reviewed_corpus(
        contract,
        input_dir=manual_input_dir,
        gold_dir=manual_gold_dir,
        frozen_manifest_path=frozen_split_manifest_path,
        splits=("train", "development", "holdout"),
        training_governance_path=governance_path,
    )
    authorized = load_phase1_authorized_ground_truth(
        governance_path,
        archive_path=authorized_archive_path,
    )
    overlap = set(manual.source_texts) & set(authorized.source_texts)
    if overlap:
        raise ValueError("Final supervision sources must have disjoint document IDs")
    source_texts = {**manual.source_texts, **authorized.source_texts}
    gold_rows = {**manual.gold_rows, **authorized.gold_rows}
    split_by_document = {document_id: "train" for document_id in source_texts}
    source_by_document = {
        **{document_id: "manual_gold" for document_id in manual.source_texts},
        **{
            document_id: "authorized_ground_truth"
            for document_id in authorized.source_texts
        },
    }
    if len(manual.source_texts) != 100 or len(authorized.source_texts) != 100:
        raise ValueError("Final supervision must contain 100 manual and 100 authorized documents")
    reviewed = Phase1ReviewedCorpus(
        source_texts=dict(sorted(source_texts.items(), key=lambda item: phase1_document_sort_key(item[0]))),
        gold_rows=dict(sorted(gold_rows.items(), key=lambda item: phase1_document_sort_key(item[0]))),
        split_by_document=dict(
            sorted(split_by_document.items(), key=lambda item: phase1_document_sort_key(item[0]))
        ),
    )
    manifest = {
        "schema_version": "phase1-final-supervision-corpus.v1",
        "document_count": len(reviewed.source_texts),
        "source_document_counts": {
            "manual_gold": len(manual.source_texts),
            "authorized_ground_truth": len(authorized.source_texts),
        },
        "source_annotation_counts": {
            "manual_gold": sum(len(rows) for rows in manual.gold_rows.values()),
            "authorized_ground_truth": authorized.source_annotation_count,
            "authorized_ground_truth_duplicate_identities": authorized.duplicate_identity_count,
        },
        "inputs": {
            "governance_sha256": sha256_file(Path(governance_path)),
            "model_split_manifest_sha256": sha256_file(Path(model_split_manifest_path)),
            "frozen_split_manifest_sha256": sha256_file(Path(frozen_split_manifest_path)),
            "authorized_archive_sha256": authorized.archive_sha256,
            "authorized_input_zip_sha256": authorized.input_zip_sha256,
            "authorized_gt_zip_sha256": authorized.gt_zip_sha256,
        },
        "round2_included": False,
        "friend31_included": False,
        "final_fit_split": "train",
    }
    manifest["fingerprint_sha256"] = _manifest_fingerprint(manifest)
    return Phase1FinalSupervisionCorpus(
        reviewed=reviewed,
        source_by_document=source_by_document,
        manifest=manifest,
    )


def _manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    import json

    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
