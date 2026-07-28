# Round 2 Qwen3-8B QLoRA

## Purpose

Train a repository-owned Vietnamese clinical extraction adapter that can regenerate Round 2
proposals from raw input. This branch is independent of the imported Friend-31 projection; that
projection remains external-teacher evidence and a public-score reference only.

## Frozen Model Budget

```text
base: Qwen/Qwen3-8B
base revision: b968826d9c46dd6066d109eabc6255188de91218
base parameters: 8,190,735,360
adapter parameters: 43,646,976
total parameters: 8,234,382,336
limit: 9,000,000,000
remaining: 765,617,664
```

The adapter stays unmerged so its parameter count, bytes, and provenance can be validated
independently from the base checkpoint.

## Training

Both stages ran on the existing Vast PyTorch environment and cached Qwen checkpoint. No new
instance or environment was created.

| Stage | Records | Steps | Epochs | Train loss | Final eval loss | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| open-corpus curriculum | 980 | 123 | 1 | `0.3350994` | n/a | `836.0432s` |
| Phase 1 specialization | frozen Phase 1 train | 123 | 3 | `0.0555527` | `0.0683995` | `1012.2665s` |

The specialization smoke test completed one optimizer step before the full run. Its eval loss was
`0.148715`. The full run used the source bundle pinned to commit
`2b3a98d4fa312ced50331f971fa6ec43a927e590`.

## Immutable Artifacts

Specialization:

```text
adapter directory:
  ec5a1abae0723c80cd53f24dbf4cd19184d4107d5edc55411ba5e1074384bf05
adapter_model.safetensors:
  b2b2f09686e7b3ca23fafd08bc15ce200535ca3e303c93a70d5603399eef97f1
adapter_config.json:
  b7bb1b51034ebadcdf44b821a4eee370b893c5d9b7db0943152a8711346113e1
run_manifest.json:
  2fed95f6e2e19bfb27cf350fc80bf1a97ab3f3567a44a4d1368fbad3a8c4c73d
```

The verified local backup is:

```text
outputs/models/vast-backups/
  phase1-qwen3-8b-qlora-specialize-2026-07-28/
```

The final adapter and resumable `checkpoint-123` are also stored in the
private Hugging Face repository
`damminhtien/medical-kg-qwen3-8b-phase1-qlora-20260728`. Upload completed
without exposing the access token to the repository or command logs.

Curriculum:

```text
adapter_model.safetensors:
  80a4160b5bfb52b79fdda1ce0658b53b6494cfbd1e17f5b97effc8fa99ecb8ab
run_manifest.json:
  2f4f12f9f513fda07bc9c5c1281564beb06fee62419cb318b93623cb5ec338d8
```

## Round 2 Inference

The pinned run spec is:

```text
configs/models/phase1-qwen3-8b-qlora-inference-2026-07-28.yaml
```

The production run used `recall_only`; the five targeted passes and thinking
adjudication remained disabled. This kept the completed run inside the GPU
budget and prevented an uncalibrated multi-pass union from entering the
submission.

```text
documents: 100/100
raw responses: 100
complete JSON responses: 39
prefix-recovered responses: 61
raw exact-quote entities: 2,267
support-confirmed consensus entities: 766
overlap rejections: 12
strict offset/schema issues: 0
```

Only complete entity objects are recovered from a truncated JSON prefix.
Every recovered quote is projected locally, and every emitted span satisfies
`source[start:end] == text`.

Immutable local inference artifact:

```text
outputs/models/vast-backups/
  phase1-qwen3-8b-qlora-recall1024-round2-2026-07-28/

raw_responses.jsonl:
  4cc16e8200bfdf48d05af1db476b81487e58aa6e9bc38008be878c45186ebed7
trace.jsonl:
  4ef79004d8dc7ebcc0cba29a8500b2d37f9e355d135fca976007bfe9199771e6
portable run archive:
  9b4a0c3a616cd33dc800d62db54834b295c39a10a67a9beedfe3d9ae5975596d
```

## Public Probe

The first probe added 39 QLoRA consensus entities to the frozen `33.0750`
public-score reference and applied selective assertion evidence to eight new
rows:

```text
ZIP SHA-256:
  21ab21f209e46fc8d8c9a912bbcfa083c7587c3a6e22cfacfd1219b9704b7e4b
score: 32.9663
WER: 63.5430
J_assertion: 41.5326
J_candidates: 23.9235
```

Relative to the reference, final score fell `0.1087`, WER worsened `0.1296`,
J_assertion fell `0.2136`, and J_candidates fell `0.0144`. Reject this
variant. Source/support agreement alone did not remove headings such as
`Cận lâm sàng`, modifiers such as `mãn tính`, or underspecified spans such as
`viêm`.

The follow-up `strict` semantic profile excludes short boundary-sensitive
families and keeps only reviewed canonical or context-anchored mentions. Its
entity-only probe adds seven rows and changes no existing assertion or
candidate:

```text
run:
  outputs/phase1/round2/
  20260728T100931Z_round2-qlora-strict-semantics-on-public-best_042b8d96e0/
variant:
  E_QLORA_STRICT_CONSENSUS_ADD
ZIP SHA-256:
  62be7247d56e831a811a323d06a3b564910139fa652fc23cb026ca870c74e691
validation issues: 0
public score: pending
```

## Calibrated Additive Probe

The next gate was trained on the frozen Round 1 pipeline/Qwen proposal matrix,
using only 60 train and 16 development documents. The 24-document holdout
remained unopened. A `0.90` minimum development precision operating point
produced:

```text
development precision: 0.924138
development recall:    0.310905
development F1:        0.465278
feature rows SHA-256:
  ed4b2914b4abed87ef40d4021d513b6a3c71f6e34ce4dbf1441e55a64a06f63e
training dataset SHA-256:
  1265fffb2243bca3ff7328095fedc7d141ed614fc93a4a6bbabc0bccd2a3f258
verifier SHA-256:
  e4d688bc3ac0b76cd502f7db8249296833c55b6f0e9e0d8adf76ae61546bc67b
```

Applied additively to the frozen public-score reference, the verifier rejected
197 below-threshold proposals, three baseline overlaps, and three structural
labels. These include both `Cận lâm sàng` occurrences that escaped the broad
probe. It retained one non-overlapping symptom occurrence,
`tiêu chảy`, with no assertion or candidate change:

```text
run:
  outputs/phase1/round2/
  20260728T111226Z_round2-qlora-calibrated-precision90-source-parit_725cbb879c/
variant:
  E_CALIBRATED_PROPOSAL_ADD
entities: 3385 -> 3386
validation issues: 0
ZIP SHA-256:
  b26cc76f39c6e48ff44c81c3689fe68635782537dcb7b09fd9b30c6bf5ca7ca5
public score: pending
```

This is a controlled one-row probe, not a promoted baseline. Its likely public
delta is small; submit it only when an entity-probe slot is available. The
broad 39-row QLoRA union remains rejected.

No model artifact is promoted from training loss or local agreement alone.
