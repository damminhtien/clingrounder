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

The run consumes the authorized raw Round 2 document manifest and performs recall, five
type-targeted passes, exact-quote projection, pass consensus, and thinking adjudication. Every
output span must satisfy `source[start:end] == text`. New entities initially carry empty assertion
and candidate lists.

Inference status and promotion decisions will be appended after the complete 100-document run,
strict validation, and comparison against both:

- the reproducible rule baseline `f0bad7ce6493...`;
- the external-teacher public-score reference `1b375c092bb5...`.

No model artifact is promoted from training loss alone.
