# Public Data Provenance

This directory contains metadata that is safe to publish without bundling restricted or large
source bytes.

- `local-artifacts.json` fingerprints canonical local-only and manifest-only files selected by the
  public release policy.
- `terminology/` preserves import summaries, source versions, row counts, parser profiles, and
  release comparisons for terminology that is restored outside Git.

Paths inside import manifests describe the expected local layout. Restore authorized source bytes,
run the documented importer, then compare the generated artifact against
`local-artifacts.json` or the relevant mining release lock.

Rebuild the local inventory with:

```bash
uv run medical-kg release inventory \
  --policy configs/repository/public-release.yaml \
  --root . \
  --output data/provenance/local-artifacts.json
```
