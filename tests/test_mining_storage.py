"""Portable content-addressed storage and hydration contracts."""

from __future__ import annotations

from io import BytesIO

from clingrounder.cli.main import main
from clingrounder.mining.storage import LocalArtifactStore, materialize_stored_object


def test_materialize_stored_object_is_verified_and_idempotent(tmp_path) -> None:
    payload = b"licensed terminology archive"
    store = LocalArtifactStore(tmp_path / "store")
    stored = store.put_stream(BytesIO(payload), metadata={"source": "fixture"})
    output = tmp_path / "inputs" / "source.zip"

    first = materialize_stored_object(
        store,
        stored.sha256,
        output,
        expected_byte_size=len(payload),
    )
    first_mtime = output.stat().st_mtime_ns
    second = materialize_stored_object(
        store,
        stored.sha256,
        output,
        expected_byte_size=len(payload),
    )

    assert output.read_bytes() == payload
    assert first == second == stored
    assert output.stat().st_mtime_ns == first_mtime


def test_artifact_materialize_cli_accepts_a_relocated_store(tmp_path, capsys) -> None:
    payload = b"source bytes copied from another machine"
    store_root = tmp_path / "mounted-cas"
    stored = LocalArtifactStore(store_root).put_stream(BytesIO(payload), metadata={})
    output = tmp_path / "hydrated" / "source.bin"

    status = main(
        [
            "data",
            "artifact",
            "materialize",
            "--store",
            str(store_root),
            "--sha256",
            stored.sha256,
            "--expected-byte-size",
            str(stored.byte_size),
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert output.read_bytes() == payload
    assert stored.sha256 in capsys.readouterr().out
