# Model Card: Vietnamese Clinical NER v1

## Status

**Pending public snapshot.** This document describes the release contract and intended scope; it
does not announce a downloadable model. The checked-in contract remains
`pending_public_snapshot` until a redistributable, human-reviewed, source-held-out dataset passes
the public benchmark audit.

## Model identity

| Field | Value |
| --- | --- |
| Task | Five-type clinical entity extraction |
| Labels | `DISEASE`, `SYMPTOM`, `DRUG`, `LAB_TEST`, `LAB_RESULT` |
| Base model | `FacebookAI/xlm-roberta-base` |
| Base revision | `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` |
| Runtime | Local Python; optional Hugging Face adapter |
| Approval | `unreviewed` |
| Rollback model | None; no public model artifact has been promoted |

The authoritative training specification is
[`configs/training/vi_clinical_ner_v1.yaml`](../configs/training/vi_clinical_ner_v1.yaml). Model
artifacts must also provide a `clingrounder.model-artifact.v1` manifest with model, tokenizer,
runtime, data, configuration, and artifact SHA-256 fingerprints before release.

## Intended use

- Research on Vietnamese and Vietnamese-English clinical entity extraction.
- Local, inspectable proposal generation for ClinGrounder pipelines.
- Reproducible evaluation of raw-text span and type behavior.

The model is an entity proposal component. It does not independently determine assertion status,
terminology codes, relations, diagnosis, or treatment recommendations. Production applications
must retain raw-offset validation and terminology membership checks in the surrounding pipeline.

## Excluded use

- Diagnosis, triage, prescribing, or treatment recommendations.
- Clinical decision support or autonomous charting.
- Claims of clinical safety, regulatory compliance, or generalization to an unreviewed corpus.
- Processing private or restricted text through hosted model services.

## Training data and provenance

No public training snapshot is currently attached to this model card. A future release must record:

1. dataset IDs, licenses, access classes, and immutable snapshot fingerprints;
2. parser/labeler revisions and split policy;
3. template, article, patient, and normalized-text leakage checks;
4. the independent review agreement report and its SHA-256 digest.

Mined silver data, competition artifacts, synthetic data, and private corpora may support research
experiments, but they are not sufficient evidence for this public model release by themselves.

## Evaluation contract

The release evaluation must report exact raw-span/type metrics per label, assertion slices,
terminology linking recall, offset validity, validation errors, initialization time, and memory.
The primary public benchmark is the task-neutral
[`vi_clinical_grounding_v1`](benchmarks/vi_clinical_grounding_v1/methodology.md) contract. Its
checked-in fixture is a synthetic engineering pilot and is explicitly ineligible for clinical
claims. Local exact-span scores and competition scores are diagnostic evidence only.

Inspect the current training gate without installing model dependencies:

```bash
clingrounder-research model inspect-public-training-contract \
  --config configs/training/vi_clinical_ner_v1.yaml
```

## Known limitations

- No public human-reviewed clinical test snapshot is available yet.
- Vietnamese specialty, note-genre, abbreviation, OCR, and code-mixed coverage is not established.
- Token-classification boundaries may require deterministic resolver and raw-offset validation.
- Entity extraction quality does not imply assertion or terminology-linking quality.
- Results from restricted competition experiments must not be presented as this model's public
  performance.

## Promotion and rollback

Promotion requires a `ready` training contract, an approved model artifact manifest, independent
public review evidence, deterministic inference, and protected-metric checks against the rules
baseline. If a future release regresses a protected metric or fails artifact verification, the
previous approved artifact is the rollback target. Until an approved artifact exists, rollback is
not applicable because the research checkpoints are not public runtime dependencies.
