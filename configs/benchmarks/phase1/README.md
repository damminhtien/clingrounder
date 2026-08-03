# Phase 1 Benchmark Configurations

These configurations preserve the archived competition experiments without
making them defaults of the clinical NLP toolkit.

## Layout

- `submission/`: deterministic export profiles for historical benchmark runs.
- `pipeline/`: benchmark-specific rule or model pipeline composition.
- `models/`: pinned training, inference, calibration, and parameter-budget specs.
- `mining/`: private Round 2 acquisition plans.
- `experiments/`: campaign records and retired selective-export policies.

The YAML files are public experiment metadata. Restricted source documents,
manual annotations, terminology releases, checkpoints, indexes, and prediction
artifacts stay outside Git. Their checksums and source ownership are recorded in
`data/provenance/local-artifacts.json` and the source dossiers under
`docs/mining-sources/`.

Run the plugin through the task-neutral CLI:

```bash
medical-kg benchmark phase1 --help
```

Exact historical reproduction requires restoring the local artifacts named by
the selected config. The core `medical-kg pipeline` commands never load these
resources implicitly.
