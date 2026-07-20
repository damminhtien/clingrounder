# Mining Source Dossiers

This directory records what the repository has actually done with each registered source. The
license registry in `data/sources/mining_registry.yaml` says what is permitted; the machine-readable
status in `data/sources/processing_status.yaml` says what has been executed and how far derived
knowledge may be promoted.

## Status

| Source | State | Strongest allowed use | Dossier |
| --- | --- | --- | --- |
| VietBioNER | curated | training only | [VietBioNER](vietbioner.md) |
| CodiEsp | promoted | runtime opt-in | [CodiEsp](codiesp.md) |
| DailyMed SPL | promoted | runtime opt-in | [DailyMed](dailymed.md) |
| DailyMed-RxNorm | promoted | runtime opt-in | [DailyMed](dailymed.md) |
| PMC OA | proposed | review only | [PMC OA](pmc-oa.md) |
| ClinicalTrials.gov | curated | training only | [ClinicalTrials](clinicaltrials.md) |
| Other registered sources | registered/quarantined | none | [Backlog](backlog.md) |

`runtime opt-in` does not mean enabled by default. It means a versioned artifact passed the
source-specific gates and may be selected explicitly by a pipeline config. `review only` means the
output may prioritize review or diagnostics but may not change runtime NER, linking, or graph facts.

## Required Dossier Sections

Each mined source documents:

1. identity, release, license and immutable acquisition;
2. parser behavior and raw-offset contract;
3. exact observed counts and duplicate handling;
4. source labels versus internal labels;
5. terminology, graph, model or evaluation knowledge extracted;
6. promotion boundary and known failure modes;
7. reproducible commands and artifact roots.

The dossier must describe observed output, not planned capability. A connector without a completed
run remains `registered`; model predictions remain `proposed`; a terminology exact match remains a
review proposal unless a source-specific promotion policy passes.

Validate registry policy and dossier discoverability together:

```bash
uv run medical-kg data registry validate \
  --registry data/sources/mining_registry.yaml \
  --processing-index data/sources/processing_status.yaml
```
