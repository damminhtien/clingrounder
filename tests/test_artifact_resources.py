"""Contract tests for portable, checksum-verified resource packs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clingrounder.artifacts import (
    ArtifactCache,
    ArtifactCacheError,
    ArtifactDownloadError,
    ArtifactDownloader,
    ArtifactManifest,
    ArtifactManifestError,
    fingerprint_payload,
    payload_size_bytes,
)


def test_manifest_round_trip_and_payload_verification(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "nested").mkdir()
    (source / "nested" / "terms.jsonl").write_text('{"text":"sốt"}\n', encoding="utf-8")
    (source / "profile.yaml").write_text("schema_version: test\n", encoding="utf-8")
    manifest = _manifest(source)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.as_dict(), indent=2), encoding="utf-8")
    loaded = ArtifactManifest.read(manifest_path)

    assert loaded == manifest
    loaded.validate_payload(source)
    assert fingerprint_payload(source) == manifest.sha256
    assert payload_size_bytes(source) == manifest.size_bytes
    assert dict(loaded.metrics) == {"benchmark": "fixture", "entity_f1": 0.8}


def test_cache_is_versioned_and_does_not_overwrite_tampered_entry(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload.txt").write_text("original", encoding="utf-8")
    manifest = _manifest(source, artifact_id="terms", revision="2026.08")
    cache = ArtifactCache(tmp_path / "cache")

    installed = cache.install(source, manifest)
    assert installed == tmp_path / "cache" / "terms" / "2026.08"
    assert cache.resolve(manifest) == installed

    (installed / "payload.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(ArtifactCacheError, match="failed verification"):
        cache.install(source, manifest)
    assert (installed / "payload.txt").read_text(encoding="utf-8") == "tampered"


def test_local_downloader_accepts_file_uri_and_rejects_network(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload.txt").write_text("payload", encoding="utf-8")
    manifest = _manifest(source)
    downloader = ArtifactDownloader()

    installed = downloader.materialize(source.as_uri(), manifest, ArtifactCache(tmp_path / "cache"))
    assert installed.is_dir()
    with pytest.raises(ArtifactDownloadError, match="Unsupported artifact source scheme"):
        downloader.materialize("https://example.invalid/artifact", manifest, ArtifactCache(tmp_path / "other"))


def test_cache_rejects_source_manifest_with_different_identity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload.txt").write_text("payload", encoding="utf-8")
    requested = _manifest(source, artifact_id="requested")
    source_manifest = _manifest(source, artifact_id="different")
    (source / "manifest.json").write_text(
        json.dumps(source_manifest.as_dict()),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactCacheError, match="does not match"):
        ArtifactCache(tmp_path / "cache").install(source, requested)


def test_manifest_rejects_path_traversal_and_unsorted_contents() -> None:
    base = {
        "schema_version": "clingrounder.artifact-manifest.v1",
        "artifact": {
            "id": "x",
            "version": "1",
            "type": "test",
            "license": "MIT",
            "sha256": "0" * 64,
            "size_bytes": 0,
        },
    }
    with pytest.raises(ArtifactManifestError, match="sorted and unique"):
        ArtifactManifest.from_mapping({**base, "contents": ["z", "a"]})
    with pytest.raises(ArtifactManifestError, match="Invalid artifact file name"):
        ArtifactManifest.from_mapping({**base, "contents": ["../payload.txt"]})
    with pytest.raises(ArtifactManifestError, match="unknown=.*extra"):
        ArtifactManifest.from_mapping({**base, "contents": ["payload.txt"], "extra": True})
    with pytest.raises(ArtifactManifestError, match="finite number"):
        ArtifactManifest.from_mapping(
            {**base, "contents": ["payload.txt"], "metrics": {"p95_ms": float("nan")}}
        )


def _manifest(source: Path, *, artifact_id: str = "fixture", revision: str = "1") -> ArtifactManifest:
    return ArtifactManifest(
        artifact_id=artifact_id,
        revision=revision,
        artifact_type="test-pack",
        license="MIT",
        sha256=fingerprint_payload(source),
        size_bytes=payload_size_bytes(source),
        contents=tuple(sorted(path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file())),
        metrics=(("benchmark", "fixture"), ("entity_f1", 0.8)),
    )
