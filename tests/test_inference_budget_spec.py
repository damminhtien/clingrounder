"""Contracts for measured active models and fail-closed parameter reservations."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from medical_kg_nlp.adapters.generative import (
    load_inference_budget_spec,
    safetensors_parameter_count,
    verify_inference_budget_spec,
)
from medical_kg_nlp.cli.main import main
from medical_kg_nlp.utils.hashing import sha256_file


def test_inference_budget_verifies_manifest_and_safetensors_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "model.json"
    manifest.write_text('{"model":{"parameter_count":7}}', encoding="utf-8")
    tensors = tmp_path / "adapter.safetensors"
    _write_safetensors(tensors, {"a": [2, 3], "b": [1]})
    config = _write_budget_config(
        tmp_path,
        manifest_sha256=sha256_file(manifest),
        tensors_sha256=sha256_file(tensors),
    )
    output = tmp_path / "verified.json"

    assert (
        main(
            [
                "model",
                "inspect-inference-budget",
                "--config",
                str(config),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert json.loads(capsys.readouterr().out) == report
    assert report["active_parameters"] == 14
    assert report["reserved_parameters"] == 5
    assert report["total_parameters"] == 19
    assert report["remaining_parameters"] == 1
    assert report["active"][1]["measured_parameter_count"] == 7
    assert report["reservations"][0]["status"] == "reserved"


def test_inference_budget_rejects_changed_artifact_bytes(tmp_path: Path) -> None:
    manifest = tmp_path / "model.json"
    manifest.write_text('{"model":{"parameter_count":7}}', encoding="utf-8")
    tensors = tmp_path / "adapter.safetensors"
    _write_safetensors(tensors, {"a": [7]})
    config = _write_budget_config(
        tmp_path,
        manifest_sha256=sha256_file(manifest),
        tensors_sha256=sha256_file(tensors),
    )
    spec = load_inference_budget_spec(config)
    manifest.write_text('{"model":{"parameter_count":8}}', encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_inference_budget_spec(spec)


def test_inference_budget_rejects_reservations_above_limit(tmp_path: Path) -> None:
    manifest = tmp_path / "model.json"
    manifest.write_text('{"model":{"parameter_count":7}}', encoding="utf-8")
    tensors = tmp_path / "adapter.safetensors"
    _write_safetensors(tensors, {"a": [7]})
    config = _write_budget_config(
        tmp_path,
        manifest_sha256=sha256_file(manifest),
        tensors_sha256=sha256_file(tensors),
        reservation=7,
    )

    with pytest.raises(ValueError, match="plus reservations exceeded"):
        load_inference_budget_spec(config)


def test_safetensors_parameter_count_rejects_invalid_header(tmp_path: Path) -> None:
    path = tmp_path / "invalid.safetensors"
    path.write_bytes(struct.pack("<Q", 100) + b"{}")

    with pytest.raises(ValueError, match="header length is invalid"):
        safetensors_parameter_count(path)


def _write_safetensors(path: Path, shapes: dict[str, list[int]]) -> None:
    offset = 0
    header: dict[str, object] = {}
    for name, shape in shapes.items():
        size = 4
        for dimension in shape:
            size *= dimension
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def _write_budget_config(
    root: Path,
    *,
    manifest_sha256: str,
    tensors_sha256: str,
    reservation: int = 5,
) -> Path:
    path = root / "budget.yaml"
    path.write_text(
        f"""\
schema_version: inference-model-budget-spec.v1
run_root: .
maximum_parameters: 20
active:
  - artifact_id: base
    model_id: example/base
    revision: "{"1" * 40}"
    parameter_count: 7
    kind: base
    roles: [recall]
    evidence:
      kind: manifest
      path: model.json
      sha256: {manifest_sha256}
      parameter_field: model.parameter_count
  - artifact_id: adapter
    model_id: example/adapter
    revision: "{"2" * 40}"
    parameter_count: 7
    kind: adapter
    roles: [verifier]
    evidence:
      kind: safetensors
      path: adapter.safetensors
      sha256: {tensors_sha256}
reservations:
  - artifact_id: future-head
    maximum_parameters: {reservation}
    roles: [fusion]
""",
        encoding="utf-8",
    )
    return path
