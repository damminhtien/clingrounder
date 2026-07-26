# Phase 1 Round 2 Competition Input

## Source Identity And Policy

The caller supplied a local ZIP containing the 100 Phase 1 Round 2 input documents released on
2026-07-22. The archive is private competition input, not an open training corpus. Its immutable
SHA-256 is:

```text
989d82404a9c1f3739e15d68a1e69d0f1f90d35c93c04ab0988e071fc1525545
```

The registry permits only local competition inference and local distribution audit. Redistribution,
hosted processing, pseudo-label training, annotation transfer, and runtime lookup memory are
prohibited. The raw archive and all parsed text remain outside Git in an encrypted local artifact
store.

## Import And Offset Contract

`PlainTextArchiveParser` accepts only regular `.txt` ZIP members, canonicalizes numeric file names,
decodes strict UTF-8, and preserves every decoded character. It rejects path traversal, symlinks,
encrypted members, duplicate canonical IDs, oversized archives, and compression bombs.

The completed import produced:

| Measure | Value |
| --- | ---: |
| source artifacts | 1 |
| parsed documents | 100 |
| unique texts | 100 |
| source ID range | 1-100 |
| total characters | 203,817 |
| mean characters/document | 2,038.17 |
| median characters/document | 1,838 |
| maximum characters/document | 4,481 |
| parser offset mismatches | 0 |

All document records carry archive SHA-256, member name, raw-byte SHA-256, encoding, and newline
mode. A second run reused both cached stages and reproduced the canonical document manifest SHA-256
`60a83690ef97a5dc6201f7877f808f593a6d86914678efeb3437814a0cba005f`.

## Distribution Audit

The audit is benchmark-owned and explicitly `runtime_eligible: false`. It emits aggregate counts,
document IDs, hashes, and similarity scores only. It does not emit source text, entity spans, entity
types, assertions, or candidates.

Observed shape:

| Measure | Value |
| --- | ---: |
| documents with bullet/list formatting | 91 |
| documents containing masked text | 30 |
| clinical-style documents | 37 |
| clinical + question/answer documents | 30 |
| question/answer documents | 22 |
| other mixed style combinations | 11 |

Round 2 reuses substantial wording from the prior 100-note corpus:

| Duplicate evidence | Value |
| --- | ---: |
| exact duplicate documents within Round 2 | 0 |
| documents with an exact prior-corpus line | 98 |
| documents with an exact prior-corpus 8-word shingle | 98 |
| exact-line character fraction | 0.423932 |
| best prior-document Jaccard at least 0.25 | 49 |
| best prior-document Jaccard at least 0.50 | 39 |

This overlap is diagnostic evidence only. It must not be used to copy old annotations into new
documents or to introduce document-specific output rules.

Ten source documents have no exact match against any prior human-gold entity context using a
32-character window and form the priority novelty queue:

```text
1, 24, 40, 48, 76, 79, 81, 83, 84, 94
```

The audit output fingerprints are:

| Artifact | SHA-256 |
| --- | --- |
| `profile.json` | `161074a5c4220ef8309a87da04f4975ec930d9ed52f54ee7bee5fc825b66ce5e` |
| `duplicate_report.json` | `176bfdec087eacb99fb8ae4b21aebc1ed026cefa49993de864c88cb16fff677e` |
| `novelty_queue.jsonl` | `460400c1b49fa3f158bd7ed93dd8577821dbbbaf354f1d62e0a9923854083d3b` |

## Model Supervision Boundary

The model is trained only from the 76 documents allowed by the frozen first-round manual-gold
split. The builder assigns duplicate groups together, then creates 60 training and 16 development
documents with hash salt `42`. The 24-document holdout stays sealed, and Round 2 contributes no
text, labels, aliases, thresholds, or pseudo-labels.

The materialized five-type dataset contains 101 chunks and 2,112 exact raw spans:

| Internal type | Count |
| --- | ---: |
| `DISEASE` | 448 |
| `DRUG` | 177 |
| `SYMPTOM` | 912 |
| `LAB_TEST` | 317 |
| `LAB_RESULT` | 258 |

The span JSONL SHA-256 is
`d87384dfdd8ee93bb26f24da0e96f2497acf6b4a3a00e0f27abfd4a0feb64f30`.
The official 19-entity BTC medication-list example is an executable convention test only:
`included_in_training: false` and `runtime_lookup_memory: false`.

XLM-R cannot express 10 of 2,112 gold boundaries as complete subword boundaries. The checked-in
run masks the crossing tokens from loss; it never widens or rewrites those raw annotations. The
distribution is 8 train and 2 development spans, or `0.4735%` of all labels.

## Executed Artifacts

The CPU smoke exercised tokenization, three optimizer steps, save, reload, overflow inference, and
raw-offset validation with the pinned checkpoint
`FacebookAI/xlm-roberta-base@e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`.
It produced zero useful entities and is explicitly `submission_eligible: false`; it validates the
runtime path, not model quality.

| Artifact | Result |
| --- | --- |
| CPU smoke run | `outputs/models/phase1-five-type-xlmr-base-cpu-smoke_53a056509b6f/` |
| smoke model fingerprint | `e9ebe7bf9ea7b558bede0538a48530d1308b1d77978a26b465cc6b70156d7f62` |
| smoke offset mismatches | `0` |
| clean rule run | `outputs/phase1/round2/20260724T030817Z_round2-rule-frozen-public-43.2014_e1bcf980c2/` |
| clean rule ZIP SHA-256 | `97a471d4d42ebaa12c3db2a1675e98695ca7714792d67c16e5edc7ac2af44a3d` |
| clean rule validation issues | `0` |

The clean rule run contains 100 JSON files and 1,909 entities. Its manifest records commit
`8b6556573c9227909cce17263b864592f3019740`, `git_dirty: false`, source archive SHA, environment
lock SHA, pipeline config, canonical validation dictionaries, and final ZIP SHA. It is the current
structural baseline only: it proves deterministic export and validation, but it is not a viable
submission candidate. Model-only and hybrid artifacts do not exist yet and must not be described as
evaluated.

## Public Rule-Baseline Result

The public grader evaluated the exact clean-rule ZIP on 2026-07-24:

| Measure | Round 2 rule result | Prior Round 1 public baseline | Difference |
| --- | ---: | ---: | ---: |
| primary score | `21.3318` | `43.2014` | `-21.8696` |
| WER | `75.3792` | `50.8167` | `+24.5625` |
| J assertion | `26.5599` | `49.7245` | `-23.1646` |
| J candidates | `14.9439` | `33.8226` | `-18.8787` |
| records scored | `100` | `100` | `0` |

The submitted file SHA-256 was
`97a471d4d42ebaa12c3db2a1675e98695ca7714792d67c16e5edc7ac2af44a3d`, so the score is attached
unambiguously to the artifact described above. The direct comparison to `43.2014` is nevertheless
confounded: that Round 1 artifact used a pipeline/Qwen entity ensemble plus selective assertion and
RxNorm overlays, while this artifact used the current rule pipeline and generic pipeline metadata.
It therefore measures both a weaker artifact composition and Round 2 transfer failure.

The output contained:

| Output evidence | Value |
| --- | ---: |
| entities | `1,909` |
| diagnosis | `601` |
| symptom | `597` |
| lab test | `369` |
| lab result | `163` |
| drug | `179` |
| entities with assertions | `507` |
| entities with candidates | `780` |
| empty documents | `2` |

The metadata policy differed materially from the Round 1 winner. The winning artifact emitted 354
candidate values on 176 medication rows and no diagnosis candidates. The Round 2 rule artifact
emitted 899 candidate values and coded every diagnosis and drug. This is a policy change, not only a
corpus change.

An explicit control run of the current configuration on the unchanged Round 1 input produced 2,028
entities. It shared 1,890 of 2,002 entity identities with the prior rule-only proposal source, but
only 1,508 of 2,809 identities with the `43.2014` public artifact. Consequently, the rule core is
mostly stable against its comparable baseline; the `43.2014` composition was not frozen into the
Round 2 run.

Of the 1,909 Round 2 predictions, 1,387 lie inside lines reused exactly from Round 1 and 522 lie
outside those lines. This is audit evidence only. Whether BTC annotates the complete mixed document
or only labels inherited from source blocks remains unknown because the published specification
does not define region-level annotation eligibility.

Manual inspection of one novelty document found that the rule dictionary missed its central G6PD
deficiency concept while emitting repeated secondary diagnoses. This is concrete evidence of
recognition/domain coverage failure in the mixed education, question-and-answer, and clinical
distribution. It is not evidence that adding more candidates to the same spans will repair WER.

## Current Frozen Baseline And Breakthrough Probe

The best scored Round 2 artifact as of 2026-07-25 is now the immutable input to isolated probes:

| Field | Value |
| --- | --- |
| public score | `23.9854` |
| WER | `72.7063` |
| J assertion | `29.6765` |
| J candidates | `17.2357` |
| entities | `2,037` |
| non-empty candidate rows/values | `177 / 177` |
| artifact | `outputs/phase1/round2/20260725T050933Z_round2-reviewed-recognition-rxnorm-only_ee4c7a81a0/output.zip` |
| ZIP SHA-256 | `f0bad7ce6493fa83bf70ff7ac70446c66fb328bb50730b613a6ea38c59b6d99e` |

The first probe applies only the previously calibrated selective negation and history policy:

| Field | Value |
| --- | --- |
| probe | `A_NEG_HIST` |
| artifact | `outputs/phase1/round2/20260726T105158Z_round2-a-neg-hist_e9df05165b/variants/A_NEG_HIST/output.zip` |
| ZIP SHA-256 | `dd75e6acb56cd2968f6614789b2068f6cf0ccaf9d26721e716fd1b4b9781dc8f` |
| entity additions/removals | `0 / 0` |
| assertion rows changed | `279` |
| selective history/negation emissions | `227 / 88` |
| candidate rows/values before and after | `177 / 177` |
| strict validation issues | `0` |

The entity projection SHA-256 is
`708953f79e1e74b6de1c2495624a955d56ffe0fd88c4270fe4a2895a7b96aea5` for both artifacts.
The run manifest records commit `5ef6739f853f2a18a6274b29884bcc6a134d132f` with
`git_dirty: false`. Promotion still requires a public J assertion gain of at least `1.0` and no
final-score regression.

The probe runner also implements the model-source routing contract:

```text
exact raw quote projection
→ keep every raw occurrence
→ region classification
→ single-source Q&A/educational additions
→ two-source exact agreement in clinical regions
→ non-overlap resolution
→ strict Phase 1 validation
```

New model entities always start with empty assertions and candidates. A proposal source never
supplies trusted offsets: exact quoted text is projected back to the immutable local document, and
every emitted span must satisfy `source_text[start:end] == text`.

### Qwen Reproduction Status

The legacy Round 1 Qwen-derived output was recovered and matches ZIP SHA-256
`c4eddd1bd0162cc52c29132a9b6c51e844cada5c557d4d47b324efae317128e8`. The generator script,
prompt, model identifier, model revision, and decoding parameters were not recovered from the
repository, shell history, or adjacent artifact directories. It is therefore evidence of prior
performance, not a reproducible proposal source. Round 2 Qwen probing remains blocked rather than
substituting an uncalibrated model.

When a complete local source is recovered, pass its Phase 1 directory or ZIP explicitly:

```bash
uv run medical-kg benchmark phase1 round2 probes \
  ... \
  --source qwen=/secure/local/qwen-round2-output.zip
```

The runner fingerprints the source and records its path and SHA-256 in the run manifest. Round 2
text must not be sent to a hosted model.

A metadata-only diagnostic probe preserves every `(text, type, position)` tuple from the rejected
artifact and clears only `assertions` and `candidates`:

| Probe | Value |
| --- | --- |
| artifact | `outputs/phase1/round2/20260724T032159Z_round2-rule-empty-metadata-probe_2496ad5dd6/output.zip` |
| ZIP SHA-256 | `68bcf7e8a3ac3beffc8a5c3557d4af710157d296351d6525ab0fc96e00447d00` |
| entity projection SHA-256 | `1112e583009e8acaa9f79af4eae6546ad2b0d7e6ba999cbf563905b634efd863` |
| validation issues | `0` |

Because its entity projection is identical, that probe cannot improve WER. It is useful only once
to discover whether empty metadata is the safer Round 2 convention. The next quality-bearing
artifact must come from a frozen five-type model or a locally gated hybrid, not from another broad
rule/candidate overlay.

## Linux GPU Handoff

The full run requires Linux, CUDA, at least 16 GiB VRAM, compute capability 8.0 or newer, and BF16.
An A100 or L4 runtime is suitable. Common T4 and P100 sessions do not satisfy the checked contract.
The current Intel macOS machine can inspect the run and perform local inference, but it cannot
produce the required full training result.

The repository tracks `uv.lock`; the run spec verifies lock SHA-256
`8ed9569bb6f221393de6508d2f36729f5b644b390336ab1eaf5bea7b12fcc87a` before importing model
frameworks. Prepare one local transfer archive containing only the first-round model supervision:

```bash
tar -czf outputs/models/phase1-five-type-xlmr-training-inputs.tar.gz \
  -C outputs/mining/model-datasets/phase1-manual-five-type-v1 \
  spans.jsonl manifest.json split_manifest.json
shasum -a 256 outputs/models/phase1-five-type-xlmr-training-inputs.tar.gz
```

The current transfer archive SHA-256 is
`087f88d4886fcf8213d9044eaba38a6a9fdce5378bc95ac0a4b087a5d44cf1a7`.
Its container timestamp may differ when rebuilt; `manifest.json` remains the authoritative
content check for `spans.jsonl`.

On an authorized Colab/Kaggle/private Linux GPU checkout:

```bash
git status --porcelain
mkdir -p outputs/mining/model-datasets/phase1-manual-five-type-v1
tar -xzf /secure/phase1-five-type-xlmr-training-inputs.tar.gz \
  -C outputs/mining/model-datasets/phase1-manual-five-type-v1

uv sync --frozen --extra ml
uv run hf download FacebookAI/xlm-roberta-base \
  --revision e73636d4f797dec63c3081bb6ed5c7b0bb3f2089
uv run medical-kg model inspect-token-classifier-run \
  --config configs/models/phase1-five-type-xlmr-base-2026-07-22.yaml
CUDA_VISIBLE_DEVICES=0 uv run medical-kg model train-token-classifier-run \
  --config configs/models/phase1-five-type-xlmr-base-2026-07-22.yaml
```

The training command uses batch 4, gradient accumulation 4, three epochs, BF16, seed 42, and full
determinism. It records dataset, checkpoint, dependency lock, run-spec, source-control, GPU, model
fingerprint, and metrics in `outputs/models/phase1-five-type-xlmr-base-2026-07-22/run_manifest.json`.

After copying `final-model/` and `run_manifest.json` back to the same repository-relative output
path on the Mac, rerun the inspect command. It verifies the returned model fingerprint, run-spec
SHA, dependency lock SHA, and dataset-manifest SHA without loading Torch. Only a verified full
model may proceed to development threshold calibration, rule/model/hybrid comparison, and local
Round 2 inference.

Run verified development inference and per-type calibration as one bounded command:

```bash
uv run medical-kg benchmark phase1 model-data calibrate \
  --pipeline-config configs/pipeline/phase1-five-type-model-only.yaml \
  --output-dir outputs/models/phase1-five-type-calibration
```

The pipeline profile points to the pinned run spec and returned `final-model/`. The command rejects
CPU smoke, fingerprint drift, run-spec drift, dependency-lock drift, or dataset-manifest drift
before importing model frameworks. It then keeps one model resident, infers exactly the 16
development documents, and writes a content-hashed directory containing:

```text
development_predictions.jsonl
calibration.json
pipeline_calibrated.yaml
run_manifest.json
```

`pipeline_calibrated.yaml` records independent thresholds for all five internal types. The command
does not deserialize holdout labels and does not read Round 2 input.

**Privacy boundary:** never upload the Round 2 ZIP, parsed Round 2 documents, audit windows, or
Round 2 predictions to Colab/Kaggle. The remote job receives only the first-round 76-document
training view, subject to the competition terms. When hosted processing of that view is not
permitted, use a private Linux GPU host instead.

## Promotion Boundary

- Allowed: local inference, aggregate profiling, duplicate diagnostics, and manual prioritization.
- Forbidden: supervised/pseudo-label training on Round 2, copying prior gold by duplicate context,
  document-ID rules, hosted processing, and redistribution.
- The 10-document novelty queue identifies what to inspect first; it contains no annotation proposal.
- Any model used on these documents must have been frozen before reading Round 2 input.

## Reproduce

Install the locked environment, point the plan to an authorized local copy, and use an encrypted
local content-addressed store:

```bash
uv sync --frozen --extra dev --extra data
export PHASE1_ROUND2_ARCHIVE=/secure/input_turn2_vong1.zip
export MEDICAL_KG_ARTIFACT_STORE=file:///secure/medical-kg/mining-artifacts

uv run medical-kg data registry validate \
  --registry data/sources/mining_registry.yaml \
  --processing-index data/sources/processing_status.yaml

uv run medical-kg data run \
  --plan configs/mining/phase1-round2-2026-07-22.yaml
```

Write the deterministic audit reports through the benchmark-owned CLI:

```bash
uv run medical-kg benchmark phase1 round2 audit \
  --documents outputs/mining/phase1-round2-2026-07-22/documents.jsonl \
  --reference-input-dir data/raw/input \
  --reference-gold-dir data/manual_gold \
  --reference-split-manifest data/manual_gold/holdout_manifest.json \
  --output-dir outputs/mining/phase1-round2-2026-07-22/audit
```

The output root is local and ignored by Git:

```text
outputs/mining/phase1-round2-2026-07-22/
```
