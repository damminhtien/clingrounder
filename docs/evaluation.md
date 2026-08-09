# Evaluation

Evaluation is task-neutral. Core metrics consume `EvaluationDocument`, `EvaluationEntity`, and
`EvaluationRelation`; benchmark plugins translate external schemas through an `EvaluationAdapter`.
The reusable evaluation package never imports a benchmark.

## Evaluation Flow

```text
task records
  -> EvaluationAdapter
  -> neutral documents
  -> exact and overlap matching
  -> entity, assertion, linking, relation, and runtime metrics
  -> structured report
  -> JSON or Markdown renderer
```

Adapters own task labels and serialization conventions. Matchers and metrics own no competition
paths, code aliases, or hidden thresholds.

Benchmark manifests are the source of truth for task taxonomies. A dataset benchmark declares its
supported entity types, assertion labels, and code systems in `dataset_manifest.yaml`; the neutral
loader validates records against those declarations rather than a ClinGrounder-specific label list.
This keeps the evaluator reusable for finding, relation, or domain-specific tasks while preserving
the same raw-offset and code-system invariants.

## Metrics

### Entities

- Exact span and exact type precision, recall, and F1.
- Span-only metrics for separating boundary errors from type errors.
- Overlap matching and boundary error classes: exact, too short, too long, crossing, and spurious.
- Per-type and per-document counts.
- Missing and spurious mention inventories for error analysis.

Raw offsets are validated before matching. An entity whose text does not equal
`source_text[start:end]` is invalid input rather than a low-quality prediction.

### Assertions

Assertions are scored on aligned entities. Reports preserve independent context dimensions where
the schema supports them, including negated, historical, family, possible, planned, conditional,
and resolved states.

Always report both:

- overall accuracy, which includes default/present labels;
- positive-label precision and recall, which exposes systems that always abstain.

### Linking

- Exact code accuracy on aligned, linkable entities.
- Candidate recall at `k`.
- Mean reciprocal rank.
- `linkable_gold_count` counts every gold-coded entity, including entities missed by extraction;
  this is the denominator for Recall@k, Top-1, and MRR.
- `assignment_coverage` is assigned primary-code predictions divided by predicted entities;
  abstentions remain visible instead of being counted as retrieval misses.
- Metrics by entity type, code system, retrieval source, and terminology release.

Code-system validation runs before scoring. A drug-to-ICD or disease-to-RxNorm assignment is a
contract violation, not merely a wrong candidate.

### Relations

- Exact typed edge precision, recall, and F1.
- Endpoint-aligned metrics when entity IDs differ across systems.
- Evidence-span validity and relation-constraint failures.
- Per-relation-type and source-section breakdowns.

### Runtime

`PipelineTrace` records stable stage names, elapsed time, counters, and rule/model provenance.
Runtime reports should include document length and entity count so throughput changes are not
confused with corpus changes.

### Validation gates

Benchmark runners validate predictions after inference and before reporting promotion metrics.
These gates are intentionally separate from F1 and recall:

- raw offsets must match the source text for every returned entity;
- assigned and candidate codes must satisfy type/code-system rules and active terminology
  membership;
- relation endpoints, relation types, and evidence spans must be valid;
- missing predictions and validation failures are counted explicitly.

`validation_error_kinds` in benchmark artifacts is the diagnostic breakdown. A zero-value quality
metric does not excuse a failed invariant, and a perfect synthetic fixture does not establish
clinical validity.

## CLI

```bash
uv run clingrounder evaluate \
  --gold data/samples/gold.jsonl \
  --pred outputs/sample-predictions.jsonl \
  --error-analysis outputs/sample-errors.json
```

Validate first when evaluating externally produced predictions:

```bash
uv run clingrounder validate \
  --profile development \
  --pred outputs/sample-predictions.jsonl \
  --documents data/samples/sample_notes.jsonl \
  --dictionary data/dictionaries/seed_concepts.jsonl
```

## Experiment Discipline

1. Freeze the data split and terminology release before comparing systems.
2. Change one component at a time: extraction, context, retrieval, reranking, or relations.
3. Record config, source fingerprints, model revision, seed, and environment.
4. Compare error counts as well as aggregate metrics.
5. Keep challenge data source-held-out and template-held-out where possible.
6. Group exact, normalized, and near-duplicate documents into the same split.
7. Treat synthetic data as training material, never as blind evaluation.

An experiment can improve one component while degrading another. Promotion reports should state
the primary metric, protected metrics, validation result, and rollback artifact explicitly.

## Error Slices

Recommended slices include:

- entity type and mention length;
- exact dictionary term, alias, abbreviation, or unseen surface;
- clinical, educational, Q&A, medication-list, and noisy/OCR genres;
- negated, historical, family, uncertain, and current contexts;
- repeated mentions and coordinated spans;
- code-system and terminology TTY;
- proposal source and source agreement;
- boundary ownership and overlap structure.

Use slices to choose the next model or data intervention. Do not encode document IDs or fixed
offsets into runtime rules.

## Benchmark Evaluation

Task-specific metrics and historical experiment gates belong to their plugin documentation. The
archived Phase 1 material is under
[`docs/benchmarks/phase1`](benchmarks/phase1/README.md).

## Implementation Map

```text
src/clingrounder/evaluation/
  records.py          neutral records and adapter protocol
  matching.py         exact and overlap alignment
  metrics.py          entity, linking, and relation metrics
  error_analysis.py   structured error inventories
  pipeline_report.py  trace and component reports
```

Useful searches:

```bash
rg "EvaluationAdapter|EvaluationDocument" src/clingrounder/evaluation tests
rg "exact_span|boundary|candidate_recall|relation" src/clingrounder/evaluation tests
```
