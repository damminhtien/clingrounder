# Pipeline Profile Catalog

This catalog separates the supported quickstart from profiles that require local mined
artifacts. Resource readiness is checked by `medical-kg pipeline list-profiles`; it does not
silently rebuild content-addressed indexes.

| ID | File | Maturity | Portability | Support | Resources | Intended use |
| --- | --- | --- | --- | --- | --- | --- |
| `clinical-baseline` | `configs/pipeline/clinical-baseline.yaml` | stable | portable | supported | ready | Default rule-based clinical NLP quickstart |
| `full-terminology` | `configs/pipeline/full_terminology.yaml` | experimental | local | setup_required | ready on this checkout | Full terminology and linking experiments |
| `full-terminology-kg-exact` | `configs/pipeline/full_terminology_kg_exact.yaml` | experimental | local | setup_required | ready on this checkout | Exact graph retrieval diagnostics |
| `general-terminology-vn` | `configs/pipeline/general_terminology_vn.yaml` | experimental | experimental | experimental | ready on this checkout | Vietnamese terminology experiments |
| `mined-vietbioner-silver` | `configs/pipeline/mined_vietbioner_silver.yaml` | experimental | experimental | experimental | ready on this checkout | VietBioNER silver recognition experiments |

## Commands

```bash
medical-kg pipeline list-profiles
medical-kg pipeline list-profiles --check-resources
medical-kg pipeline inspect-config \
  --config configs/pipeline/clinical-baseline.yaml \
  --check-resources
```

Only `clinical-baseline` is intended to work from a clean clone with checked-in resources.
The other profiles retain their fingerprints and paths for reproducibility, but their generated
indexes and mined overlays are local setup requirements and are not part of the portable
quickstart contract.
