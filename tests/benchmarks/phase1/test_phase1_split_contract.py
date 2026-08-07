import hashlib
import json
from pathlib import Path

from clingrounder.benchmarks.phase1.split_contract import (
    load_phase1_split_contract,
)
from clingrounder.utils.hashing import sha256_file


def _ids_sha256(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def test_split_contract_keeps_holdout_sealed(tmp_path: Path) -> None:
    frozen_path = tmp_path / "holdout.json"
    frozen_path.write_text(
        json.dumps(
            {
                "schema_version": "phase1-manual-gold-split.v1",
                "corpus": {"fingerprint_sha256": "c" * 64},
                "splits": {
                    "train": {"document_ids": ["1", "2"]},
                    "holdout": {"document_ids": ["3"]},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "schema_version": "phase1-model-training-split.v1",
                "round2_included": False,
                "source_split_manifest_sha256": sha256_file(frozen_path),
                "source_corpus_fingerprint_sha256": "c" * 64,
                "source_document_ids": {
                    "train": ["1"],
                    "development": ["2"],
                },
                "excluded_holdout": {
                    "document_ids_sha256": _ids_sha256(["3"])
                },
                "split_groups": {
                    "phase1-manual-gold:1": "group:1",
                    "phase1-manual-gold:2": "group:2",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    contract = load_phase1_split_contract(model_path, frozen_path)

    assert contract.train_ids == ("1",)
    assert contract.development_ids == ("2",)
    assert contract.holdout_ids == ("3",)
    assert set(contract.train_ids + contract.development_ids).isdisjoint(
        contract.holdout_ids
    )
