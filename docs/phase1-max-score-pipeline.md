# Phase 1 Max-Score Pipeline

The final Phase 1 composer is intentionally CPU-only:

```text
pinned Rule/XLM-R/Qwen/VietMed outputs
-> exact raw-offset proposal matrix
-> calibrated probability
-> global conflict resolver
-> selective assertion policy
-> candidate metadata hydration and abstention
-> strict validation
-> deterministic ZIP
```

Model inference remains a separate, resumable stage. This avoids loading Qwen or either XLM-R
checkpoint again when only a threshold, assertion rule, or candidate policy changes.

## Run Contract

Run:

```bash
uv run medical-kg benchmark phase1 round2 max-score \
  --config configs/models/phase1-round2-max-score.yaml
```

The config uses `phase1-max-score-run-spec.v1` and pins:

- the authorized-private `documents.jsonl` and original archive fingerprint;
- the verified under-9B budget spec;
- the calibrated proposal verifier;
- every source ZIP with a model-neutral role;
- every terminology JSONL;
- assertion and candidate policies.

All paths resolve below `run_root`. Every file is checked against SHA-256 before parsing.

```yaml
schema_version: phase1-max-score-run-spec.v1
run_root: ../..
documents:
  path: outputs/mining/phase1-round2-hosted-2026-07-27/documents.jsonl
  sha256: <sha256>
  source_archive_sha256: <original-input-archive-sha256>
  expected_count: 100
budget_spec: configs/models/phase1-under9b-max.yaml
verifier:
  path: outputs/phase1/proposal_fusion_20260729/calibrated_genre_f1/verifier.json
  sha256: <sha256>
sources:
  - name: rule
    role: rule
    path: <deterministic-rule-output.zip>
    sha256: <sha256>
  - name: xlmr
    role: token_model
    path: <deterministic-xlmr-output.zip>
    sha256: <sha256>
  - name: qwen
    role: llm
    path: <deterministic-qwen-output.zip>
    sha256: <sha256>
  - name: vietmed
    role: verifier
    path: <deterministic-vietmed-support.zip>
    sha256: <sha256>
dictionaries:
  - path: data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl
    sha256: <sha256>
  - path: data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl
    sha256: <sha256>
  - path: data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl
    sha256: <sha256>
candidate_source_priority: [qwen, rule, xlmr, vietmed]
assertion_regimes: [negation, history]
candidate_policy: rx_unique_keep_icd
output_root: outputs/phase1/max-score
run_label: phase1-under9b-max
```

`verifier` sources are support-only. A VietMed-only proposal is recorded and blocked; VietMed may
increase confidence only when a target-task source proposes the same exact span and type.

## Output

Each content-addressed run writes:

- `output/` and deterministic `output.zip`;
- verified budget manifest;
- proposal matrix and calibrated scores;
- source, assertion, and candidate decision traces;
- counters and run manifest.

The run fails before ZIP creation for an exceeded parameter budget, replaced artifact, invalid raw
offset, wrong code system, code absent from the pinned terminology, or invalid document set.

## Slow Vast Hosts

Reuse the existing `/venv/main` environment and Hugging Face cache. Model and dataset stages should
check pinned fingerprints before download or upload. The max-score composer itself needs no GPU,
network access, or Hugging Face token.
