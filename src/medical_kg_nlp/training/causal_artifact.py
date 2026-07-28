"""Validation and provenance finalization for completed QLoRA adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from medical_kg_nlp.mining.io import write_json
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.run_output import collect_git_metadata

__all__ = ["finalize_causal_qlora_artifact"]


def finalize_causal_qlora_artifact(
    output_dir: Path,
    *,
    model_id: str,
    model_revision: str,
    parameter_count: int,
    run_spec_path: Path,
    run_spec_sha256: str,
    environment_lock_path: Path,
    environment_lock_sha256: str,
) -> dict[str, Any]:
    """Verify immutable run evidence and attach source-bundle provenance.

    Training writes the adapter before the outer CLI appends run-spec and
    environment evidence. This operation is idempotent so a completed job from
    a source archive can be finalized without retraining or hand-editing JSON.
    """

    manifest_path = output_dir / "run_manifest.json"
    adapter_config_path = output_dir / "final-adapter" / "adapter_config.json"
    manifest = _load_mapping(manifest_path, "QLoRA run manifest")
    if manifest.get("schema_version") != "causal-qlora-artifact.v1":
        raise ValueError("Unsupported QLoRA artifact schema")

    model = _required_mapping(manifest, "model")
    if (
        model.get("model_id") != model_id
        or model.get("revision") != model_revision
        or model.get("parameter_count") != parameter_count
    ):
        raise ValueError("QLoRA artifact model identity does not match the run spec")

    observed_run_spec_sha256 = sha256_file(run_spec_path)
    if (
        observed_run_spec_sha256 != run_spec_sha256
        or _required_mapping(manifest, "run_spec").get("sha256")
        != run_spec_sha256
    ):
        raise ValueError("QLoRA run-spec fingerprint mismatch")

    observed_lock_sha256 = sha256_file(environment_lock_path)
    environment = _required_mapping(manifest, "environment")
    if (
        observed_lock_sha256 != environment_lock_sha256
        or environment.get("lock_sha256") != environment_lock_sha256
    ):
        raise ValueError("QLoRA environment lock fingerprint mismatch")

    if not adapter_config_path.is_file():
        raise ValueError("QLoRA final adapter config is absent")
    adapter_config_sha256 = sha256_file(adapter_config_path)
    if (
        _required_mapping(manifest, "artifacts").get("adapter_config_sha256")
        != adapter_config_sha256
    ):
        raise ValueError("QLoRA adapter config fingerprint mismatch")

    source_control = collect_git_metadata()
    if source_control.get("git_commit") is None:
        raise ValueError(
            "QLoRA artifact cannot be finalized without Git or .source-commit"
        )
    # INVARIANT: finalization changes provenance metadata only. Adapter weights,
    # run controls, model identity, datasets, and metrics remain byte-identical.
    manifest["source_control"] = source_control
    write_json(manifest_path, manifest)
    return {
        "status": "finalized",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "adapter_config_sha256": adapter_config_sha256,
        "source_control": source_control,
    }


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is absent: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"QLoRA manifest {key!r} must be an object")
    return value
