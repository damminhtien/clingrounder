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

### TT06 And RxNorm Review Queue

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

### Mondo Condition Canonicalization

The 53 condition entries were queried separately against the pinned 6 July 2026 Mondo release.
This policy accepts only `DISEASE` rows whose source label is `CTGOV_CONDITION`; intervention and
outcome fields cannot acquire Mondo codes through this stage.

| Mondo status | Condition entries |
| --- | ---: |
| unique normalized exact concept | 27 |
| normalized exact, multiple concepts | 3 |
| unmatched | 23 |

The three ambiguous terms are `lung cancer`, `penile cancer`, and `late infantile neuronal ceroid
lipofuscinosis`. Unmatched rows are mostly registry-specific composites or trial eligibility labels,
including `NSCLC Stage IB-IIIA` and `uncommon EGFR mutations`; they are not shortened or guessed.

`attach-exact-links` materialized the 27 exact-unique rows on the corresponding source annotations.
It preserved all 202 annotation IDs, spans, texts, types, assertions, layers, and review statuses,
and did not overwrite existing links. Each attached link records the crosswalk policy, Mondo release,
match mode, and `review_required` status. Its semantic contract is
`exact_terminology_review_evidence_not_clinical_gold`.

The Mondo crosswalk SHA-256 is
`9b36bbc49c3798016f4953d1f6186397d125c61ba9ec537f92c9c0030c16da04`.
The linked-annotation SHA-256 is
`651aec0950b34846ad7d4dbfaf8c78e61553b4c26726f2cf6c28798e2b8d2034`.

## Combined Mondo And Trial Graph

The Mondo hierarchy and neutral trial relations were compiled into one provenance-bearing graph.
`--relation-endpoints-only` excludes 99 unrelated outcome annotations, while
`--canonical-concepts-only` rejects any linked code absent from the pinned terminology. Unlinked
interventions remain term nodes; no drug code is inferred from its surface form.

| Graph measure | Value |
| --- | ---: |
| nodes | 32,173 |
| Mondo concept nodes | 32,097 |
| unlinked source term nodes | 76 |
| `IS_A` edges | 46,447 |
| `STUDIES_INTERVENTION` edges | 118 |
| total edges/evidence rows | 46,565 / 46,565 |
| aliases in SQLite | 112,925 |

The index retained 118/118 trial edges and 46,447/46,447 hierarchy edges. With eight read workers,
the trial-edge benchmark measured 1,523 queries/second at p95 10.81 ms; the hierarchy benchmark
measured 2,991 queries/second at p95 5.01 ms. Both runs were deterministic. These are index coverage
and concurrency measurements, not clinical-quality or reranker-gain measurements.

An actual query resolves `Wilson disease` to `MONDO:0010200`, traverses ancestors such as disorder
of copper metabolism, inborn error of metabolism, hereditary disease, and metabolic disease, and
finds the registered interventions `trientine` and `tetrathiomolybdate` through outgoing
`STUDIES_INTERVENTION` edges. The intervention nodes remain unlinked terms and the edges do not mean
that either intervention treats, is approved for, or was effective against Wilson disease.

This graph can supply auditable ontology ancestors, sibling hard-negative candidates, coverage
targets, and neutral source-evidence features. A source-held-out reranking benchmark is still needed
before any graph feature can be enabled in the default pipeline.

The graph JSONL SHA-256 values are:

- nodes: `4ff58a579d879e365cd611b7d5458e2faca665e6a5adc7a1a8ed1f6452a7e625`;
- edges: `db28c8914b9a06b9717b7925e8da2cf6a0fad0f522a75e05b97ad6f4cac4b608`;
- evidence: `90fe9d3f8c8e65a2ed2f568a1d194e56c7b4a08cacd98be17c33a893971e79e0`.

The SQLite graph input fingerprint is
`f013a9046b4be90abd2cf13d14700754b00ce129befad8a42d1cdd7252468f15`.

## Snapshot And Promotion Boundary

The frozen silver snapshot is
`clinicaltrials-rare-interventional-2026-07-19-silver-v1-feb4268becd8ffdc`, with 27 train and 1
development document. Its manifest SHA-256 is
`f1c056fdb784ae7594f5190088b5e08c1ac38349b914cea59709733051651ad9`.

- Allowed now: relation-model training, graph-feature experiments, rare condition/intervention
  coverage analysis, exact Mondo review evidence, terminology review queues, ontology hard-negative
  generation, and synthetic scenario grounding.
- Blocked now: runtime NER aliases, automatic ICD/RxNorm links, `TREATS` edges, claims of efficacy,
  automatic Mondo prediction gold, challenge evaluation, or runtime-default graph evidence.
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

uv run medical-kg data lexicon crosswalk \
  --inventory outputs/mining/clinicaltrials-rare-interventional-2026-07-19/mention_inventory.jsonl \
  --index outputs/mining/knowledge/mondo-2026-07-06/terminology.sqlite3 \
  --source outputs/mining/knowledge/mondo-2026-07-06/terminology.jsonl \
  --policy configs/mining/crosswalk/clinicaltrials-mondo.yaml \
  --output outputs/mining/clinicaltrials-rare-interventional-2026-07-19/terminology_crosswalk_mondo.jsonl \
  --report-output outputs/mining/clinicaltrials-rare-interventional-2026-07-19/terminology_crosswalk_mondo_report.json \
  --workers 4

uv run medical-kg data lexicon attach-exact-links \
  --annotations outputs/mining/clinicaltrials-rare-interventional-2026-07-19/structured_annotations.jsonl \
  --crosswalk outputs/mining/clinicaltrials-rare-interventional-2026-07-19/terminology_crosswalk_mondo.jsonl \
  --policy configs/mining/linking/clinicaltrials-mondo.yaml \
  --output outputs/mining/clinicaltrials-rare-interventional-2026-07-19/mondo_linked_annotations.jsonl \
  --report-output outputs/mining/clinicaltrials-rare-interventional-2026-07-19/mondo_link_materialization_report.json

uv run medical-kg data knowledge compile-graph \
  --terminology-source outputs/mining/knowledge/mondo-2026-07-06/terminology.jsonl \
  --documents outputs/mining/clinicaltrials-rare-interventional-2026-07-19/documents.jsonl \
  --annotations outputs/mining/clinicaltrials-rare-interventional-2026-07-19/mondo_linked_annotations.jsonl \
  --relations outputs/mining/clinicaltrials-rare-interventional-2026-07-19/structured_relations.jsonl \
  --accepted-layer silver \
  --accepted-review-status proposed \
  --relation-endpoints-only \
  --canonical-concepts-only \
  --nodes-output outputs/mining/knowledge/clinicaltrials-mondo-2026-07-19/nodes.jsonl \
  --edges-output outputs/mining/knowledge/clinicaltrials-mondo-2026-07-19/edges.jsonl \
  --evidence-output outputs/mining/knowledge/clinicaltrials-mondo-2026-07-19/evidence.jsonl \
  --report-output outputs/mining/knowledge/clinicaltrials-mondo-2026-07-19/report.json

uv run medical-kg kg build \
  --nodes outputs/mining/knowledge/clinicaltrials-mondo-2026-07-19/nodes.jsonl \
  --edges outputs/mining/knowledge/clinicaltrials-mondo-2026-07-19/edges.jsonl \
  --evidence outputs/mining/knowledge/clinicaltrials-mondo-2026-07-19/evidence.jsonl \
  --output outputs/mining/knowledge/clinicaltrials-mondo-2026-07-19/graph.sqlite3 \
  --manifest-output outputs/mining/knowledge/clinicaltrials-mondo-2026-07-19/graph_manifest.json

uv run medical-kg kg benchmark-relations \
  --index outputs/mining/knowledge/clinicaltrials-mondo-2026-07-19/graph.sqlite3 \
  --edges outputs/mining/knowledge/clinicaltrials-mondo-2026-07-19/edges.jsonl \
  --relation-type STUDIES_INTERVENTION \
  --workers 8 \
  --repeats 5 \
  --output outputs/mining/knowledge/clinicaltrials-mondo-2026-07-19/studies_intervention_benchmark.json
```

The separate TT06/RxNorm review queue still uses `configs/mining/crosswalk/clinicaltrials.yaml`.
Run the same relation benchmark with `--relation-type IS_A --repeats 3` to reproduce the hierarchy
measurement. First use `medical-kg kg inspect --query "Wilson disease"` to obtain the exact result's
`node_id`. Re-run `kg inspect` with that `--node-id` plus `--ancestors`, then with
`--relation-type STUDIES_INTERVENTION --direction outgoing`; traversal is node-ID based so similarly
named non-human Mondo concepts cannot become the query root accidentally.
