# ClinicalTrials.gov Rare-Disease Interventional Studies

## Source And Acquisition

- Registry source: `clinicaltrials_v2`.
- Transport: ClinicalTrials.gov API v2 study endpoint.
- Discovery query: condition `Rare Diseases`, restricted to `COMPLETED`,
  `ACTIVE_NOT_RECRUITING`, or `RECRUITING` studies.
- Discovery response: 774 matching records; the first API page contained 50 records.
- Discovery response SHA-256:
  `f8e634c7955c34c24a89ea519ffd0950041afb4ecab76caaeea300f11d53d7b0`.
- Accepted tranche: the 28 `INTERVENTIONAL` records on that frozen page.
- Run config: `configs/mining/clinicaltrials-rare-interventional-2026-07-19.yaml`.

The config lists every NCT ID explicitly. Reproduction therefore fetches the selected records by
identifier instead of replaying an API search whose ordering or current study state may change.
Artifacts are immutable and content-addressed. The artifact manifest SHA-256 is
`12f4626021c287e790c2f53f887f4654b6cce3ceadf4d4a009cc6a44b129c5ec`.

This tranche is a coverage sample, not a representative estimate of all 774 records or all rare
diseases. A larger tranche must use pagination checkpoints and preserve each discovery response.

## Parsing And Offset Contract

`clinicaltrials_json` parser revision 2 renders title and summary as context, then renders condition,
intervention, and outcome fields once. Every structured source field is projected during rendering;
the labeler never searches for the same string later. This prevents a repeated condition or drug
name in the summary from receiving the wrong offset.

| Measure | Value |
| --- | ---: |
| artifacts/documents | 28 / 28 |
| rendered characters | 37,281 |
| median document length | 809.5 |
| maximum document length | 5,455 |
| structured annotations | 202 |
| offset/schema issues | 0 |
| unique text hashes | 28 |

The document manifest SHA-256 is
`b5efd533516c0c580e79cfc5a77ef7b54cd84225d94256328c4a379268f901f0`.
`MinedDocument.text` is immutable after rendering and every annotation satisfies
`document.text[start:end] == annotation.text`.

## Source Labels And Internal Types

The source-field labeler emits a silver layer with confidence 1.0 for source projection, not for
clinical truth or terminology normalization.

| Source field | Internal type | Occurrences |
| --- | --- | ---: |
| condition | `DISEASE` | 53 |
| intervention: drug/biological/combination/supplement | `DRUG` | 41 |
| intervention: procedure/device/behavioral/genetic/radiation | `PROCEDURE` | 4 |
| intervention: diagnostic test | `LAB_TEST` | 1 |
| intervention: other | `OTHER` | 4 |
| primary outcome measure | `OTHER` | 31 |
| secondary outcome measure | `OTHER` | 68 |

Outcomes stay `OTHER`: a measure such as overall survival, an ECG safety endpoint, or a quality-of-
life scale is not automatically a lab test, result, symptom, or diagnosis. The inventory contains
201 normalized entries for 202 occurrences. The only type conflict is `placebo`, which occurs once
as `DRUG` and once as source `OTHER`; it remains human-review required.

The annotation manifest SHA-256 is
`f63c098203c1739d51ce39962dfd4121bf615f7086e4e8846381b50110437f91`.

## Relation Knowledge

For each study, the relation labeler connects every registered condition to every registered
intervention in that protocol. It produced 118 source-structured relations with no missing endpoint
or evidence-span issue. The relation type is deliberately neutral:

```text
condition --STUDIES_INTERVENTION--> registered intervention
```

It means only that the registry says the intervention is studied for the condition. It does **not**
mean `TREATS`, clinical efficacy, safety, approval, standard of care, or a positive trial result.
Outcome text is not used to infer an efficacy edge. The relation manifest SHA-256 is
`8cb8a59bfaaabb5ce89bd437bb66560c10e67fffbd6e440801b9194df7df9030`.

The source-only fusion run
`clinicaltrials-rare-interventional-2026-07-19-1e07af376898` retained all 28 documents, 202
annotations, and 118 relations. It found no exact or SimHash-near duplicate group and rejected zero
relations.

An endpoint-only term graph contains 103 term nodes and 118 `STUDIES_INTERVENTION` edges. The SQLite
read benchmark covered 118/118 edges, remained deterministic with eight workers, and measured
2,092 queries/second with p95 latency 7.45 ms. This benchmark proves index consistency and
concurrent-read behavior only; it does not validate clinical relation quality.

## Terminology Crosswalk

The 201-entry inventory was queried against the pinned TT06 ICD-10 and 6 July 2026 RxNorm releases,
including the reviewed DailyMed and CodiEsp alias overlays. Matching is type/code-system filtered,
but all output remains `review_required`.

| Crosswalk status | Entries |
| --- | ---: |
| unique exact concept | 8 |
| exact string, multiple codes | 21 |
| bounded lexical candidates | 48 |
| unmatched | 17 |
| no applicable policy | 107 |

The eight exact-unique mentions are `histoplasmosis`, `systemic lupus erythematosus`,
`osteopetrosis`, `blastomycosis`, `epilepsy`, `uridine`, `ribose`, and
`sodium dichloroacetate`. They are review proposals, not an alias overlay.

The exact-ambiguous rows demonstrate why normalized string equality is insufficient for RxNorm:
ingredient names such as `morphine` and `methoxsalen` also match multiple dose-form/product
concepts. Lexical fallback is especially noisy for English rare-disease names against a Vietnamese
ICD terminology and can return hundreds of weak candidates. FTS rows are useful only as a bounded
review queue; none may be promoted automatically.

The inventory SHA-256 is
`3f6250710867874f64b578f1766f01904be9294a0b4bdb074a20180c18464adf`.
The crosswalk output SHA-256 is
`10348475649897b953f9b8f9f82e57421e2c5c61d96a18878d89a141d893109a`.

## Snapshot And Promotion Boundary

The frozen silver snapshot is
`clinicaltrials-rare-interventional-2026-07-19-silver-v1-feb4268becd8ffdc`, with 27 train and 1
development document. Its manifest SHA-256 is
`f1c056fdb784ae7594f5190088b5e08c1ac38349b914cea59709733051651ad9`.

- Allowed now: relation-model training, graph-feature experiments, rare condition/intervention
  coverage analysis, terminology review queues, and synthetic scenario grounding.
- Blocked now: runtime NER aliases, automatic ICD/RxNorm links, `TREATS` edges, claims of efficacy,
  challenge evaluation, or runtime-default graph evidence.
- Before runtime graph use: review or canonical-link both endpoints, keep the neutral relation type,
  and benchmark the graph feature on a source-held-out retrieval task.
- Before terminology promotion: resolve the 21 exact-ambiguous rows, reject broad FTS matches, and
  demonstrate held-out retrieval gain without type/code-system regressions.

The source is therefore recorded as `curated / training_only`, not `promoted`.

## Reproduce

```bash
export MEDICAL_KG_ARTIFACT_STORE=/Volumes/medical-kg-mining

uv run medical-kg data run \
  --plan configs/mining/clinicaltrials-rare-interventional-2026-07-19.yaml

uv run medical-kg data label propose \
  --documents outputs/mining/clinicaltrials-rare-interventional-2026-07-19/documents.jsonl \
  --adapter medical_kg_nlp.mining.labelers.clinicaltrials:create_clinicaltrials_structured_labeler \
  --adapter-config configs/mining/labelers/clinicaltrials-structured.yaml \
  --output outputs/mining/clinicaltrials-rare-interventional-2026-07-19/structured_annotations.jsonl

uv run medical-kg data relation propose \
  --documents outputs/mining/clinicaltrials-rare-interventional-2026-07-19/documents.jsonl \
  --annotations outputs/mining/clinicaltrials-rare-interventional-2026-07-19/structured_annotations.jsonl \
  --adapter medical_kg_nlp.mining.labelers.clinicaltrials:create_clinicaltrials_structured_relation_labeler \
  --adapter-config configs/mining/labelers/clinicaltrials-relations.yaml \
  --output outputs/mining/clinicaltrials-rare-interventional-2026-07-19/structured_relations.jsonl

uv run medical-kg data dataset inspect \
  --documents outputs/mining/clinicaltrials-rare-interventional-2026-07-19/documents.jsonl \
  --annotations outputs/mining/clinicaltrials-rare-interventional-2026-07-19/structured_annotations.jsonl \
  --output outputs/mining/clinicaltrials-rare-interventional-2026-07-19/source_profile.json \
  --strict

uv run medical-kg data dataset fuse \
  --plan configs/mining/fusion/clinicaltrials-rare-interventional-2026-07-19.yaml
```

Build the inventory/crosswalk with `medical-kg data lexicon build` and
`medical-kg data lexicon crosswalk` using
`configs/mining/crosswalk/clinicaltrials.yaml`. Compile the training-only term graph with
`medical-kg data knowledge compile-graph --accepted-layer silver --relation-endpoints-only`; do not
add `--linked-only` until reviewed canonical links have been materialized.
