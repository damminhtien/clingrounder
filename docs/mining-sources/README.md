# Mining Source Dossiers

Source dossiers are executable provenance, not narrative status notes. The cross-machine release
workflow and byte-lock contract are defined in `docs/mining-reproducibility.md`; each promoted or
training source must expose a complete `Reproduce` section before it can enter a release spec.

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
| RxNorm Full July 2026 | promoted | runtime opt-in | [RxNorm](rxnorm.md) |
| PMC OA | proposed | review only | [PMC OA](pmc-oa.md) |
| ClinicalTrials.gov | curated | training only | [ClinicalTrials](clinicaltrials.md) |
| Mondo | curated | training only | [Mondo](mondo.md) |
| HPO | curated | training only | [HPO](hpo.md) |
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

Use explicit evidence labels whenever a dossier also records the next tranche:

- **processed**: command completed and the dossier records counts plus artifact fingerprints;
- **planned**: version/checksum/config are pinned, but no output metric may be reported as observed;
- **blocked**: license, credentials, storage, parser, or quality gate prevents execution;
- **excluded**: a source lane was deliberately left out and the clinical/quality reason is stated.

For deep source mining, document each transformation separately rather than saying only "imported":

```text
remote artifact and source checksum
-> CAS SHA-256 and acquisition manifest
-> parser revision and immutable document view
-> duplicate/leakage handling
-> source-label projection
-> terminology or ontology reconciliation
-> graph/model/evaluation artifacts
-> measured quality and promotion boundary
```

Counts must be attached to the stage that produced them. For example, source rows, parsed
documents, entity proposals, accepted aliases and graph edges are different populations and must
not be combined into a single "records" count.

Validate registry policy and dossier discoverability together:

```bash
uv run medical-kg data registry validate \
  --registry data/sources/mining_registry.yaml \
  --processing-index data/sources/processing_status.yaml
```
