# Mining Reproducibility

Mining data is split across two storage classes:

```text
Git: source registry, licences, parsers, policies, run configs, dossiers
External/CAS: downloaded archives, parsed corpora, snapshots, model data, indexes
```

Git alone is intentionally insufficient for large or restricted corpora. The portable release
contract connects the checked-in process to the external bytes without recording a developer's
home directory or storage mount.

## Release Contract

The current NER/retrieval release specification is
`configs/mining/releases/open-ner-retrieval-v1.yaml`. It binds:

- `pyproject.toml`, `uv.lock`, and the implementation under `src/medical_kg_nlp`;
- source registry, processing status, source dossiers, and relevant mining policies;
- fused documents, immutable split manifests, and raw-offset NER datasets;
- VietBioNER source-held-out recognition evidence;
- CodiEsp train-only aliases and graph-reranker evidence;
- the official DailyMed-to-RxNorm alias overlay;
- the deterministic full-type NER run specification;
- an optional Linux/GPU checkpoint, which is currently absent and therefore not promoted.

Create the lock after materializing the release:

```bash
uv run medical-kg data release lock \
  --spec configs/mining/releases/open-ner-retrieval-v1.yaml \
  --output data/releases/open-ner-retrieval-v1.lock.json
```

Verify it locally:

```bash
uv run medical-kg data release verify \
  --manifest data/releases/open-ner-retrieval-v1.lock.json \
  --root .
```

The lock is deterministic. Files are SHA-256 hashed directly; directories are hashed from sorted
relative member paths, sizes, and member SHA-256 values. Modification times do not affect the
fingerprint. Symlinks and paths outside the declared release root are rejected.
Exclude patterns are permitted only for implementation artifacts (for example Python bytecode);
datasets, terminology, benchmarks, and model checkpoints are always hashed in full.

## Reproduce On Another Machine

1. Check out the repository revision associated with the archived lock and restore dependencies:

   ```bash
   uv sync --frozen --extra dev --extra data --extra retrieval --extra ml
   ```

2. Mount or restore the content-addressed artifact store. It may live anywhere; set the store URI
   for mining plans that permit an environment override:

   ```bash
   export MEDICAL_KG_ARTIFACT_STORE=file:///mnt/medical-kg/mining-artifacts
   ```

3. Validate source governance before processing any bytes:

   ```bash
   uv run medical-kg data registry validate \
     --registry data/sources/mining_registry.yaml \
     --processing-index data/sources/processing_status.yaml \
     --repository-root .
   ```

4. Choose one of two supported restoration paths:

   - Restore the artifact directories named in the release lock at the same repository-relative
     paths. The repository itself may be located anywhere.
   - Rebuild each source from its checksum-pinned plan and follow the exact `Reproduce` section in
     its dossier under `docs/mining-sources/`. Then run the fusion, snapshot, curation, knowledge,
     and benchmark commands declared there.

5. Place the archived `release.lock.json` under any local path and run `data release verify` with
   the new repository as `--root`. Verification succeeds only when every required byte agrees.

The full-type NER YAML declares `run_root` relative to the YAML file. Dataset, cache, output, and
checkpoint paths are resolved from that root rather than the shell working directory. This means a
worker may invoke the command with an absolute config path from another directory without changing
the run identity. Runtime code uses resolved paths, while `run_manifest.json` stores only
run-root-relative paths and the run-spec SHA-256.

The checked-in GPU run also enables Hugging Face `full_determinism`. Exact checkpoint bytes can
still differ if the CUDA/PyTorch stack differs, so reproducibility has two levels: the release lock
proves identical inputs and policy, while `model.fingerprint` proves identity of a materialized
checkpoint. The manifest records framework, CUDA, and GPU metadata needed to explain a differing
fingerprint.

`outputs/mining/` is Git-ignored. Open-data lock manifests can be checked in under `data/releases/`,
as this release is; restricted-data locks must be archived in the approved experiment store. Do not
commit downloaded corpora merely to make a run portable.

## Promotion Boundaries

Reproducible does not mean suitable for runtime:

| Artifact | Allowed use | Evidence |
| --- | --- | --- |
| VietBioNER reconciled spans | NER training/domain adaptation | Raw offsets, source-held split |
| VietBioNER recognition dictionary | Diagnostic only | Development F1 `0.5221`; procedure precision `0.1216` |
| CodiEsp train aliases | Runtime opt-in | Dev/test retrieval both improve |
| CodiEsp co-occurrence graph | Reranker opt-in | Small positive held-out top-1/MRR delta |
| DailyMed RxNorm aliases | Runtime opt-in | Official checksum-pinned SPL crosswalk |
| Full-type model checkpoint | Not available | Optional lock entry remains absent |

These boundaries are also recorded in `data/sources/processing_status.yaml`. A release verifier
proves content identity; benchmark gates decide whether an artifact may affect NER or retrieval.

## Portability Rules For New Stages

- Persist source URIs, content hashes, version IDs, parser versions, and policy hashes.
- Store paths relative to a plan, snapshot, or release root. Never persist `/Users/...`, `/home/...`,
  drive-letter paths, temporary directories, or mounted-volume paths.
- Keep raw source text immutable. Normalized or translated text is a child document with its own
  identity and offset coordinate system.
- Build derived indexes atomically from canonical JSONL/Parquet sources; do not treat SQLite,
  DuckDB, ANN, or model caches as source of truth.
- Keep train/development/challenge assignments in immutable manifests and group exact/near
  duplicates before assigning splits.
- Pin every model by `model_id`, immutable revision, tokenizer revision, dataset hash, and training
  config. Resolve run paths under a declared root and persist only relative paths. A local directory
  name is not model provenance.
- Record missing optional artifacts explicitly. Never silently substitute another checkpoint,
  terminology release, or alias overlay.
