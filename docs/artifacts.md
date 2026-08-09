# Artifact Resources

ClinGrounder keeps models, terminology releases, and small runnable packs explicit and
fingerprinted. The default Python import path never contacts a hosted registry.

## Bundled pack

The wheel contains `vi-clinical-small`, a deliberately small Vietnamese smoke baseline. It is
resolved by a stable artifact ID and revision:

```python
from clingrounder import load_pipeline

with load_pipeline("vi-clinical-small", revision="2026.08", offline=True) as pipeline:
    prediction = pipeline("Bệnh nhân không sốt và có tiền sử tăng huyết áp.")
```

`load_pipeline` verifies the pack manifest before constructing the pipeline. It does not silently
substitute another revision or a newer "latest" artifact.

## Explicit local cache

To materialize a verified bundled pack into a caller-owned cache:

```python
from clingrounder import Pipeline

path = Pipeline.download(
    "vi-clinical-small",
    revision="2026.08",
    cache_dir=".cache/clingrounder/artifacts",
)
print(path)

with load_pipeline(path, offline=True) as pipeline:
    prediction = pipeline("Bệnh nhân không sốt.")
```

The cache layout is:

```text
<cache>/<artifact-id>/<revision>/
```

An existing entry is reused only after checksum and content verification. A mismatched entry is
not overwritten. Installation copies into a temporary sibling and publishes the complete payload
with one atomic rename.

## Custom local artifacts

The low-level API supports local paths and `file://` URIs. A caller supplies a typed manifest and
an explicit cache:

```python
from clingrounder.artifacts import ArtifactCache, ArtifactDownloader, ArtifactManifest

manifest = ArtifactManifest.read("/models/vi-ner/manifest.json")
installed = ArtifactDownloader().materialize(
    "file:///models/vi-ner",
    manifest,
    ArtifactCache(".cache/clingrounder/models"),
)
```

The manifest covers relative payload names, artifact ID, revision, license, byte size, and a
deterministic SHA-256. `manifest.json` is metadata and is excluded from the payload digest to
avoid a circular hash. Absolute paths, traversal components, duplicate names, unknown manifest
fields, and symlinks are rejected.

The core downloader rejects `http://` and `https://` sources by design. Applications that need a
remote registry must add an explicit provider that downloads to a trusted local directory, then
passes the downloaded directory through the same manifest and cache verification. This keeps
network policy, credentials, license acceptance, and retention outside the core package.

## Provenance

Record the artifact ID, revision, manifest SHA-256, source license, and the effective pipeline
configuration in experiment or deployment manifests. The bundled pack is a runnable example, not a
complete ICD-10 or RxNorm release. Full terminology and model artifacts remain separately
acquired, licensed, and pinned resources.
