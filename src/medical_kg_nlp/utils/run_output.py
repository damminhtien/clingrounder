from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class HashedRunOutput:
    run_id: str
    run_dir: Path
    manifest_path: Path


def create_hashed_run_dir(
    root: str | Path,
    *,
    label: str,
    inputs: Sequence[str | Path] = (),
) -> HashedRunOutput:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    entropy = uuid.uuid4().hex
    safe_label = _slug(label)
    payload = {
        "timestamp_utc": timestamp,
        "label": label,
        "inputs": [str(item) for item in inputs],
        "entropy": entropy,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    run_id = f"{timestamp}_{safe_label}_{digest}"
    run_dir = _mkdir_unique(Path(root), run_id)
    manifest_path = run_dir / "run_manifest.json"
    manifest = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "timestamp_utc": timestamp,
        "label": label,
        "hash": digest,
        "inputs": [str(item) for item in inputs],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return HashedRunOutput(run_id=run_dir.name, run_dir=run_dir, manifest_path=manifest_path)


def path_in_run(path: str | Path, run_output: HashedRunOutput | None) -> Path:
    output_path = Path(path)
    if run_output is None or output_path.is_absolute():
        return output_path
    if output_path.parts and output_path.parts[0] == "outputs":
        output_path = Path(*output_path.parts[1:]) if len(output_path.parts) > 1 else Path(".")
    return run_output.run_dir / output_path


def _mkdir_unique(root: Path, run_id: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for attempt in range(100):
        candidate = root / run_id if attempt == 0 else root / f"{run_id}_{attempt:02d}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"Could not create a unique run directory under {root}")


def _slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.strip()).strip("-._")
    return (slug or "run")[:48]
