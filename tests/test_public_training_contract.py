from pathlib import Path
import hashlib

import pytest
import yaml

from clingrounder.cli.parser import build_parser
from clingrounder.training import load_public_training_contract


def test_checked_in_contract_is_inspectable_but_not_ready() -> None:
    contract = load_public_training_contract("configs/training/vi_clinical_ner_v1.yaml")

    assert contract.status == "pending_public_snapshot"
    assert contract.model_id == "FacebookAI/xlm-roberta-base"
    assert contract.train_manifest is None


def test_contract_has_a_research_cli_inspection_command() -> None:
    args = build_parser(scope="research").parse_args(
        ["model", "inspect-public-training-contract", "--config", "contract.yaml"]
    )

    assert args.handler == "model_inspect_public_training_contract"


def test_ready_contract_requires_two_distinct_manifests(tmp_path: Path) -> None:
    payload = _payload(status="ready")
    (tmp_path / "train.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "dev.json").write_text("{}\n", encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text('{"eligible_for_clinical_claim": true}\n', encoding="utf-8")
    payload["data"] = {
        "train_manifest": "train.json",
        "validation_manifest": "dev.json",
        "train_manifest_sha256": _sha256(tmp_path / "train.json"),
        "validation_manifest_sha256": _sha256(tmp_path / "dev.json"),
        "dataset_audit_report": "audit.json",
        "dataset_audit_report_sha256": _sha256(audit),
    }
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    contract = load_public_training_contract(path)

    assert contract.status == "ready"
    assert contract.validation_manifest == (tmp_path / "dev.json").resolve()


def test_ready_contract_rejects_unreviewed_dataset_audit(tmp_path: Path) -> None:
    payload = _payload(status="ready")
    (tmp_path / "train.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "dev.json").write_text("{}\n", encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text('{"eligible_for_clinical_claim": false}\n', encoding="utf-8")
    payload["data"] = {
        "train_manifest": "train.json",
        "validation_manifest": "dev.json",
        "train_manifest_sha256": _sha256(tmp_path / "train.json"),
        "validation_manifest_sha256": _sha256(tmp_path / "dev.json"),
        "dataset_audit_report": "audit.json",
        "dataset_audit_report_sha256": _sha256(audit),
    }
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="eligible dataset audit"):
        load_public_training_contract(path)


def test_contract_rejects_manifest_checksum_drift(tmp_path: Path) -> None:
    payload = _payload(status="pending_public_snapshot")
    train = tmp_path / "train.json"
    train.write_text("{}\n", encoding="utf-8")
    payload["data"] = {
        "train_manifest": train.name,
        "validation_manifest": None,
        "train_manifest_sha256": "0" * 64,
        "validation_manifest_sha256": None,
        "dataset_audit_report": None,
        "dataset_audit_report_sha256": None,
    }
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="train_manifest SHA-256 mismatch"):
        load_public_training_contract(path)


def test_contract_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    payload = _payload(status="pending_public_snapshot")
    payload["data"] = {"train_manifest": "../outside.json", "validation_manifest": None}
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes contract directory"):
        load_public_training_contract(path)


def test_contract_rejects_unpinned_model_revision(tmp_path: Path) -> None:
    payload = _payload(status="pending_public_snapshot")
    payload["model"]["revision"] = "main"
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="40-character"):
        load_public_training_contract(path)


def _payload(*, status: str) -> dict[str, object]:
    return {
        "schema_version": "clingrounder.public-training-contract.v1",
        "run_id": "test-run",
        "status": status,
        "task": "entity-extraction",
        "labels": ["DISEASE", "SYMPTOM"],
        "model": {"id": "local/test", "revision": "a" * 40},
        "data": {
            "train_manifest": None,
            "validation_manifest": None,
            "train_manifest_sha256": None,
            "validation_manifest_sha256": None,
            "dataset_audit_report": None,
            "dataset_audit_report_sha256": None,
        },
        "training": {
            "seed": 42,
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
        },
        "selection": {"primary_metric": "entity_f1", "minimum_primary_metric": 0.0},
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
