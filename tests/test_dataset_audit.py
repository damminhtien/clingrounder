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
    assert report.eligible_for_engineering_use is True
    assert report.eligible_for_clinical_claim is False
    assert report.clinical_claim_blockers == (
        "synthetic_source",
        "human_review_required",
        "release_status_not_reviewed",
    )
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
    assert report.agreement is not None
    assert report.checks["review_agreement_meets_targets"] is True


def test_reviewed_dataset_requires_measured_agreement(tmp_path: Path) -> None:
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
    manifest["review"]["agreement_report_sha256"] = "0" * 64  # type: ignore[index]
    (tmp_path / "dataset_manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    report = audit_dataset(tmp_path)

    assert report.eligible_for_clinical_claim is False
    assert "agreement_report_fingerprint_mismatch" in report.issues


def test_reviewed_dataset_rejects_agreement_below_manifest_target(tmp_path: Path) -> None:
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
    agreement_path = tmp_path / "agreement.json"
    agreement = json.loads(agreement_path.read_text(encoding="utf-8"))
    agreement["span_type_agreement"] = 0.80
    agreement_path.write_text(
        json.dumps(agreement, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["review"]["agreement_report_sha256"] = hashlib.sha256(  # type: ignore[index]
        agreement_path.read_bytes()
    ).hexdigest()
    (tmp_path / "dataset_manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    report = audit_dataset(tmp_path)

    assert report.eligible_for_clinical_claim is False
    assert "agreement_below_target:span_type" in report.issues


def test_audit_rejects_invalid_annotation_structure(tmp_path: Path) -> None:
    invalid = {
        "document_id": "train-1",
        "text": "Sốt.",
        "metadata": {"template_group": "train-template"},
        "entities": [
            {
                "id": "e1",
                "span": [0, 99],
                "text": "Sốt",
                "type": "SYMPTOM",
                "assertion": "PRESENT",
                "code_system": "NONE",
                "code": "fabricated",
            }
        ],
        "relations": [],
    }
    valid = {
        "document_id": "test-1",
        "text": "Ho.",
        "metadata": {"template_group": "test-template"},
        "entities": [],
        "relations": [],
    }
    (tmp_path / "train.jsonl").write_text(json.dumps(invalid) + "\n", encoding="utf-8")
    (tmp_path / "test.jsonl").write_text(json.dumps(valid) + "\n", encoding="utf-8")
    manifest = _manifest(tmp_path, status="synthetic_pilot", human_reviewed=False)
    (tmp_path / "dataset_manifest.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    report = audit_dataset(tmp_path)

    assert report.checks["annotation_structure_valid"] is False
    assert "invalid_entity:train:1" in report.issues
    assert report.eligible_for_clinical_claim is False


def _manifest(tmp_path: Path, *, status: str, human_reviewed: bool) -> dict[str, object]:
    splits: dict[str, dict[str, object]] = {}
    for split in ("train", "test"):
        path = tmp_path / f"{split}.jsonl"
        splits[split] = {
            "path": path.name,
            "documents": 1,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    agreement = {
        "schema_version": "clingrounder.review-agreement.v1",
        "dataset_id": "audit-fixture",
        "dataset_version": "1.0.0",
        "reviewed_document_count": 2,
        "double_reviewed_document_count": 1,
        "double_review_fraction": 0.5,
        "span_type_agreement": 0.95,
        "assertion_agreement": 0.90,
        "relation_agreement": 0.85,
    }
    agreement_path = tmp_path / "agreement.json"
    agreement_path.write_text(
        json.dumps(agreement, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
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
            "agreement_report": agreement_path.name,
            "agreement_report_sha256": hashlib.sha256(agreement_path.read_bytes()).hexdigest(),
        },
    }
