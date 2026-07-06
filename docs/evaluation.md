# Evaluation

Evaluation is built around the internal schema so public dataset adapters can normalize labels before
metrics are computed.

## Implemented Metrics

- Exact span/type precision, recall, and F1.
- Relaxed overlap span/type precision, recall, and F1.
- Linking accuracy at 1, recall at 5/10/20, and MRR.
- Assertion/context accuracy.
- Typed relation precision, recall, and F1.
- Phase 1 submission score for the official flat entity JSON format:
  `100 * (0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score)`.
- Error-analysis CSV with document id, error type, text window, gold label, prediction, candidate
  list, and notes.
- Dataset profiling for entity, code, context, relation, section, abbreviation-like mention,
  dictionary coverage, offset issue, and optional unseen-code overlap checks.
- Stage-wise pipeline reports that combine dataset profile, validator issues, end-to-end metrics,
  structured error rows, and per-stage trace counters/timings.

## Recommended Gates

- Run validator checks before evaluation.
- For Phase 1, treat `phase1_score` as the primary loop score and relation F1 as internal
  diagnostics only.
- Validate that `output.zip` contains `output/1.json` through `output/100.json`.
- Track candidate recall at 20 before adding rerankers.
- Treat offset regressions as blocking.
- Compare context-specific errors for negation, family history, historical mentions, and possible
  findings.
- Run data profiling before model or reranker work so long-tail codes, sparse relations, and schema
  risks are visible early.

## Commands

```bash
python scripts/run_pipeline.py \
  --input data/samples/sample_notes.jsonl \
  --output outputs/predictions.jsonl

python scripts/validate_predictions.py \
  --pred outputs/predictions.jsonl \
  --documents data/samples/sample_notes.jsonl \
  --dictionary data/dictionaries/seed_concepts.jsonl

python scripts/evaluate.py \
  --gold data/samples/gold.jsonl \
  --pred outputs/predictions.jsonl

python scripts/profile_data.py \
  --documents data/samples/sample_notes.jsonl \
  --gold data/samples/gold.jsonl \
  --output outputs/profiles/sample_profile.json \
  --markdown outputs/profiles/sample_profile.md

python scripts/evaluate_pipeline_steps.py \
  --documents data/samples/sample_notes.jsonl \
  --gold data/samples/gold.jsonl \
  --dictionary data/dictionaries/seed_concepts.jsonl \
  --run-root outputs/runs \
  --output-dir evaluation/sample

python scripts/build_phase1_submission.py \
  --config configs/phase1_submission.yaml \
  --run-label phase1
```

The profiler output is intended to be a cacheable experiment artifact. Use it to compare train/dev
entity distributions, span lengths, context cue frequency, dictionary coverage, and unseen-code
risks before adding larger models.

The stage-wise report writes:

- `metrics.json` for the full machine-readable report.
- `phase1` inside `metrics.json` for official Phase 1 schema validation, scored flat predictions,
  and Phase 1-specific errors.
- `stage_metrics.csv` for long-format metrics by pipeline stage.
- `errors.csv` and `errors.jsonl` for structured error analysis.
- `profile.json` and `profile.md` for the dataset profile.
- `traces.json` for per-document trace details.
- `summary.md` for a short run summary.

Use `--pred existing_predictions.jsonl` to evaluate a saved prediction file without rerunning the
pipeline. Omit `--pred` to run the pipeline, save `predictions.jsonl`, and collect traces.

Pipeline execution can parallelize across documents:

```bash
python scripts/run_pipeline.py \
  --input data/samples/sample_notes.jsonl \
  --output outputs/predictions.jsonl \
  --workers 4 \
  --parallel-backend process

python scripts/evaluate_pipeline_steps.py \
  --documents data/samples/sample_notes.jsonl \
  --gold data/samples/gold.jsonl \
  --dictionary data/dictionaries/seed_concepts.jsonl \
  --output-dir outputs/evaluation/sample \
  --workers 4 \
  --parallel-backend process
```

Use `process` for CPU-bound batches and `thread` only for low-overhead smoke runs or future I/O-heavy
stages. Output order remains the same as input document order, and workers never write JSONL
directly.

## Phase 1 Submission

Phase 1 is an offline batch submission, not an API. The exported artifact is a ZIP whose root entry
is `output/`, with one JSON file per input TXT file. Each JSON file is a flat list of entities with
only these fields:

- `text`
- `type`
- `assertions`
- `candidates`
- `position`

Allowed Phase 1 types are `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`, `CHẨN_ĐOÁN`, and
`THUỐC`. `position` uses raw-text `[start, end)` offsets, so `raw_text[start:end] == text` must hold.
Only `TRIỆU_CHỨNG`, `CHẨN_ĐOÁN`, and `THUỐC` may emit assertions; lab-test names and lab-test
results must emit `assertions: []`. Only `CHẨN_ĐOÁN` may emit ICD-10 candidates and only `THUỐC`
may emit RxNorm candidates. Other types must emit `candidates: []`. Relations stay internal for
Phase 1 unless the official schema changes.

The checked-in Phase 1 input lives under `data/raw/input` and contains 100 unlabeled TXT files. Since
there is no hidden gold locally, the pre-submit gate for that folder is validation-first: no schema
issues, no offset mismatches, no invalid candidates, and a correct ZIP layout. `phase1_score` is
available for labeled regression data and synthetic samples so the loop engine can still optimize the
official 0.3/0.3/0.4 objective before submitting.

Use `configs/phase1_submission.yaml` as the default source of truth for Phase 1 submission paths,
dictionary, parallelism, and pipeline options. The default config writes to a timestamped hash
directory under `outputs/phase1`, validates the output directory before creating the ZIP, validates
the ZIP payload against source TXT offsets and dictionary candidates, and disables relation
extraction/validation because Phase 1 exports only flat entities. CLI flags override config values
for one-off runs.

## Loop Engineering

Use the loop engine after a stage-wise report exists. It turns metrics and error rows into an
experiment log, a keep/revert/refine decision, prioritized error groups, and a concrete next
experiment.

```bash
python scripts/loop_engineer.py \
  --current-report outputs/evaluation/sample/metrics.json \
  --output-dir outputs/loops/sample \
  --experiment-id BASELINE \
  --module evaluation \
  --hypothesis "Establish a valid end-to-end baseline." \
  --change "Run current pipeline and generate stage-wise metrics."
```

Use `--baseline-report previous/metrics.json` when comparing one meaningful change against a frozen
baseline. Each run writes per-experiment artifacts plus a shared journal. By default the journal is
created as a `journal/` sibling of `--output-dir`; use `--journal-dir` to store it elsewhere.

Per-experiment artifacts:

- `experiment_log.yaml` and `experiment_log.json` for the experiment record.
- `decision.md` for the baseline/keep/revert/refine decision.
- `next_experiment.md` for the highest-priority follow-up.
- `top_error_cases.md` for representative cases from the top error classes.
- `agent_poll.json` for a small polling/status payload with read order, next action, and completion
  markers.
- `agent_compact.md` for token-efficient resume context; agents should read this before the full
  report on long runs.
- `agent_brief.md` for a coding-agent-ready task brief with guardrails, files, commands, and
  acceptance criteria.
- `agent_actions.jsonl` for machine-readable action items.
- `confusion_matrix.csv` for assertion/context confusion.
- `loop_report.json` for the full machine-readable loop report.

Journal artifacts:

- `experiments.jsonl` append-only experiment log.
- `experiment_index.json` latest indexed view by experiment id.
- `experiment_memory.json` buckets for `reuse`, `avoid`, and `refine`.
- `experiment_notebook.md` human-readable experiment notebook.

Implementation boundary:

- `loop_engineer.py` assembles one loop report and keeps backward-compatible public imports.
- `loop_analysis.py` owns metric snapshots, deltas, decisions, error prioritization, and next
  experiment selection.
- `loop_policy.py` owns error policies and agent playbooks.
- `loop_agent.py` owns agent context, actions, polling payloads, and compact briefs.
- `loop_artifacts.py` owns file writing and markdown/CSV/JSONL artifacts.
- `loop_journal.py` owns the append-only experiment log, memory, and notebook outputs.

## Ablation And Bottleneck Workflow

Use `configs/ablations.yaml` to compare small pipeline variants without changing code. The default
variants isolate retrieval sources, candidate reranking, context reasoning, relation extraction, KG
validation, and candidate depth.

```bash
python scripts/run_ablation.py --config configs/ablations.yaml --run-root outputs/runs
```

The command writes:

- `outputs/ablations/predictions/{variant}.jsonl` for validator-compatible predictions.
- `outputs/ablations/traces/{variant}.json` for per-document stage timings.
- `outputs/ablations/summary.csv` for accuracy and top bottleneck comparison.
- `outputs/ablations/stage_timings.csv` for per-stage runtime and counters.
- `outputs/ablations/metrics.json` for the full machine-readable report.

Interpretation loop:

1. Check `validation_issues`; fix schema, offset, dictionary, or KG failures before comparing
   metrics.
2. For Phase 1, compare `phase1_score` first, then inspect span, linking, and context diagnostics;
   relation F1 is internal-only unless the official schema adds relations.
3. Inspect `bottleneck_stage` and `stage_timings.csv` before optimizing. Add Rust/C++ only after a
   stable Python bottleneck is measured and a Python fallback remains.
4. When tuning parameters, add a new variant to `configs/ablations.yaml` instead of editing core
   code for one-off experiments.
