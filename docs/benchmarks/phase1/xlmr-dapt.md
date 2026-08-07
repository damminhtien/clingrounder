# Joint Vietnamese XLM-R DAPT

This run trains one pinned XLM-R encoder with two objectives:

```text
Vietnamese medical text -> masked-language modeling
pinned terminology aliases -> same-concept synonym contrastive learning
```

Round 2 is an unlabeled, physically separate MLM lane. It cannot provide
entity labels, pseudo-labels, synonym pairs, thresholds, or calibration data.
The run specification and artifact verifier enforce this separation.

## Rebuild Inputs

First reproduce the source outputs documented in:

- [`mining-sources/vietmed-ner.md`](mining-sources/vietmed-ner.md)
- [`mining-sources/vietbioner.md`](mining-sources/vietbioner.md)
- [`mining-sources/phase1-round2.md`](mining-sources/phase1-round2.md)

Build the provenance-separated MLM lanes:

```bash
uv run clingrounder-research model build-dapt-corpus \
  --config configs/benchmarks/phase1/models/xlmr-dapt-corpus-2026-07-29.yaml
```

Build bounded same-concept pairs from the pinned TT06, RxNorm, and Vietnamese
clinical terminology:

```bash
uv run python scripts/build_terminology_synonym_pairs.py \
  --source data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl \
  --source data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl \
  --source data/standards/vn_clinical_lexicon/processed/vn_clinical_lexicon_concepts.jsonl \
  --output outputs/mining/model-datasets/xlmr-terminology-synonym-pairs-2026-07-29/pairs.jsonl \
  --mode canonical_to_alias \
  --max-names-per-concept 8 \
  --max-pairs-per-concept 7 \
  --exclude-abbreviations
```

The checked-in run spec contains the expected hashes for these source files.
Any changed source, derived lane, pair dataset, manifest, config, or lockfile
fails before model loading.

## Inspect And Prefetch

Validate all bytes and print the exact remote commands:

```bash
uv run clingrounder-research model inspect-xlmr-dapt-run \
  --config configs/benchmarks/phase1/models/xlmr-joint-dapt-2026-07-29.yaml
```

Use the machine's existing environment or a maintained Vast PyTorch/Hugging
Face template. Install only missing pinned packages because remote downloads
can dominate this run. Prefetch the immutable model revision before enabling
`local_files_only` training:

```bash
hf download FacebookAI/xlm-roberta-base \
  --revision e73636d4f797dec63c3081bb6ed5c7b0bb3f2089 \
  --cache-dir .cache/clingrounder/model-training
```

The repository commit must be clean. Run from the repository root with the
same `uv.lock` recorded by the run spec.

## Smoke And Train

Run the one-update forward/backward smoke test first:

```bash
uv run clingrounder-research model train-xlmr-dapt-run \
  --config configs/benchmarks/phase1/models/xlmr-joint-dapt-2026-07-29.yaml \
  --max-steps 1 \
  --output-dir outputs/smoke/xlmr-dapt
```

The smoke artifact is explicitly non-promotable. After it passes on Linux,
CUDA, and BF16, start the full resumable run inside `tmux`:

```bash
uv run clingrounder-research model train-xlmr-dapt-run \
  --config configs/benchmarks/phase1/models/xlmr-joint-dapt-2026-07-29.yaml
```

Checkpoints are written under:

```text
outputs/models/xlmr-joint-dapt-2026-07-29/checkpoints/
```

Resume only from a checkpoint produced by the same immutable run:

```bash
uv run clingrounder-research model train-xlmr-dapt-run \
  --config configs/benchmarks/phase1/models/xlmr-joint-dapt-2026-07-29.yaml \
  --resume-from-checkpoint \
  outputs/models/xlmr-joint-dapt-2026-07-29/checkpoints/step-00001000
```

## Verify Returned Artifact

Copy the complete output directory back to the same run-root-relative path,
then run the inspect command again. When an artifact is present, inspection
fails unless all of these match:

- full model-directory fingerprint;
- model ID and immutable revision;
- run-spec ID and SHA-256;
- dependency-lock SHA-256;
- MLM lane hashes and Round 2 policy;
- synonym-pair and terminology-source hashes;
- GPU precision and completed training steps;
- `promotion_eligible=true`, `purpose=training`, and `smoke=false`.

A verified DAPT encoder is still not an automatically promoted NER model.
Fine-tune and evaluate it against the frozen model-development protocol, then
compare exact span/type metrics and source-held slices with the original
XLM-R checkpoint.

## Expected Input Identity

The 2026-07-29 run was prepared with:

| Input | Records |
| --- | ---: |
| VietBioNER unlabeled | 63 |
| VietMed-NER unlabeled | 9,246 |
| Round 2 unlabeled MLM-only | 100 |
| Terminology synonym pairs | 63,247 |

The pair dataset SHA-256 is:

```text
b888f26daa43fa97f7834f1e89217a572b894422a5cea7e1a9a0e0313427d0ec
```

Do not treat these counts alone as identity. The run spec and manifests own
the authoritative per-source hashes.
