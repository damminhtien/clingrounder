"""Tests for the neutral benchmark evidence audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from clingrounder.evaluation.dataset_audit import audit_dataset


ROOT = Path("benchmarks/vi_clinical_grounding_v1")


def test_synthetic_pilot_is_explicitly_not_clinical_evidence() -> None:
    report = audit_dataset(ROOT)

    assert report.status == "synthetic_pilot"
    assert report.human_reviewed is False
    assert report.eligible_for_clinical_claim is False
    assert "clinical_claim_requires_human_review" in report.warnings
    assert report.checks["template_groups_disjoint"] is True
    assert report.checks["normalized_text_splits_disjoint"] is True


def test_audit_rejects_cross_split_template_and_text_leakage(tmp_path: Path) -> None:
    row = {
        "document_id": "doc-1",
        "text": "Bệnh nhân sốt.",
        "metadata": {"template_group": "same-template", "human_reviewed": True},
        "entities": [],
        "relations": [],
    }
    for split in ("train", "test"):
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps({**row, "document_id": f"{split}-1"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    manifest = _manifest(tmp_path, status="human_reviewed", human_reviewed=True)
    (tmp_path / "dataset_manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    report = audit_dataset(tmp_path)

    assert "template_group_overlap" in report.issues
    assert "normalized_text_overlap" in report.issues
    assert report.eligible_for_clinical_claim is False


def test_reviewed_dataset_passes_the_public_evidence_gate(tmp_path: Path) -> None:
    rows = {
        "train": {"document_id": "train-1", "text": "Đau đầu."},
        "test": {"document_id": "test-1", "text": "Khó thở."},
    }
    for split, base in rows.items():
        row = {
            **base,
            "metadata": {"template_group": f"{split}-template", "human_reviewed": True},
            "entities": [],
            "relations": [],
        }
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    manifest = _manifest(tmp_path, status="human_reviewed", human_reviewed=True)
    (tmp_path / "dataset_manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    report = audit_dataset(tmp_path)

    assert report.eligible_for_clinical_claim is True
    assert not report.issues


def _manifest(tmp_path: Path, *, status: str, human_reviewed: bool) -> dict[str, object]:
    splits: dict[str, dict[str, object]] = {}
    for split in ("train", "test"):
        path = tmp_path / f"{split}.jsonl"
        splits[split] = {
            "path": path.name,
            "documents": 1,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "schema_version": "clingrounder.dataset-manifest.v1",
        "dataset": {
            "id": "audit-fixture",
            "version": "1.0.0",
            "status": status,
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "human_reviewed": human_reviewed,
        },
        "splits": splits,
        "policy": {
            "template_groups_disjoint": True,
            "test_used_for_development": False,
            "private_data": False,
        },
        "review": {
            "reviewers_required": 2,
            "double_review_fraction": 0.1,
            "agreement_targets": {"span_type": 0.9, "assertion": 0.85},
        },
    }
