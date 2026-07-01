# Evaluation

Evaluation is built around the internal schema so public dataset adapters can normalize labels before
metrics are computed.

## Implemented Metrics

- Exact span/type precision, recall, and F1.
- Relaxed overlap span/type precision, recall, and F1.
- Linking accuracy at 1, recall at 5/10/20, and MRR.
- Assertion/context accuracy.
- Typed relation precision, recall, and F1.
- Error-analysis CSV with document id, error type, text window, gold label, prediction, candidate
  list, and notes.
- Dataset profiling for entity, code, context, relation, section, abbreviation-like mention,
  dictionary coverage, offset issue, and optional unseen-code overlap checks.
- Stage-wise pipeline reports that combine dataset profile, validator issues, end-to-end metrics,
  structured error rows, and per-stage trace counters/timings.

## Recommended Gates

- Run validator checks before evaluation.
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
  --output-dir outputs/evaluation/sample
```

The profiler output is intended to be a cacheable experiment artifact. Use it to compare train/dev
entity distributions, span lengths, context cue frequency, dictionary coverage, and unseen-code
risks before adding larger models.

The stage-wise report writes:

- `metrics.json` for the full machine-readable report.
- `stage_metrics.csv` for long-format metrics by pipeline stage.
- `errors.csv` and `errors.jsonl` for structured error analysis.
- `profile.json` and `profile.md` for the dataset profile.
- `traces.json` for per-document trace details.
- `summary.md` for a short run summary.

Use `--pred existing_predictions.jsonl` to evaluate a saved prediction file without rerunning the
pipeline. Omit `--pred` to run the pipeline, save `predictions.jsonl`, and collect traces.

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

## Ablation And Bottleneck Workflow

Use `configs/ablations.yaml` to compare small pipeline variants without changing code. The default
variants isolate retrieval sources, candidate reranking, context reasoning, relation extraction, KG
validation, and candidate depth.

```bash
python scripts/run_ablation.py --config configs/ablations.yaml
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
2. Compare `span_exact.f1`, linking recall/MRR, context metrics, and relation F1 across variants.
3. Inspect `bottleneck_stage` and `stage_timings.csv` before optimizing. Add Rust/C++ only after a
   stable Python bottleneck is measured and a Python fallback remains.
4. When tuning parameters, add a new variant to `configs/ablations.yaml` instead of editing core
   code for one-off experiments.
