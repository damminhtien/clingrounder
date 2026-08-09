"""Prevent checked-in training specs from drifting from the repository lock."""

from __future__ import annotations

from pathlib import Path

import yaml

from clingrounder.utils.hashing import sha256_file


def test_checked_in_run_specs_pin_the_current_lockfile() -> None:
    """Every reproducible model run must identify this checkout's dependency graph."""

    root = Path(__file__).parents[1]
    specs = []
    for path in sorted((root / "configs").rglob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        environment = payload.get("environment")
        if not isinstance(environment, dict) or "lock_sha256" not in environment:
            continue
        specs.append((path, Path(str(payload.get("run_root", "."))), environment))

    assert specs, "Expected at least one checked-in run spec"

    for path, run_root, environment in specs:
        # INVARIANT: resolve the lock exactly as the runtime resolves each spec,
        # so a stale fingerprint fails in CI before a remote training job starts.
        lock_path = (path.parent / run_root / str(environment["lock_path"])).resolve()
        assert lock_path.is_file(), f"Missing lockfile for {path}"
        assert environment["lock_sha256"] == sha256_file(lock_path), path
