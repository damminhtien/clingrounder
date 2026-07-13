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

The current competition baseline is entity-only. `assertion_policy: empty` and
`candidate_policy: empty` preserve entity text/type/offsets while deliberately abstaining on the two
high-risk fields. The pipeline still computes assertion and linking traces internally. Python export
APIs remain backward-compatible with `pipeline` defaults; the submission config opts into `empty`
explicitly.

Evaluate reviewed Phase 1 files without converting them to the internal schema:

```bash
python scripts/evaluate_phase1_manual_gold.py \
  --gold-dir data/manual_gold \
  --pred-dir outputs/phase1/<run>/phase1/output \
  --output-dir outputs/evaluation/manual_gold
```

The evaluator locks a deterministic 60/15 train/holdout split, reports entity errors by type, and
separates null accuracy, positive precision/recall, and prediction coverage for assertions and
candidates. Its blocking gate uses holdout Phase 1 score and text score. Missing, spurious, and
boundary thresholds remain diagnostics because the public scorer has shown that a recall-heavy
output can win while exceeding the old local false-positive cap.

Compare two complete entity outputs by selecting a source independently for each Phase 1 type:

```bash
python scripts/ensemble_phase1_outputs.py \
  --primary-dir outputs/phase1/<pipeline-run>/phase1/output \
  --secondary-dir /path/to/qwen-output.zip \
  --gold-dir data/manual_gold \
  --input-dir data/raw/input \
  --output-dir outputs/phase1/<ensemble-run>
```

Without explicit `--source` flags, the command searches source-per-type combinations on the 60-file
train split. It reports the frozen 15-file holdout only after selecting the best train strategy,
writes `strategy_search.json`, validates offsets/schema, and creates `output.zip`.

Analyze entity-only WER, source lineage, and span boundaries:

```bash
uv run python scripts/analyze_phase1_entity_wer.py \
  --gold-dir data/manual_gold \
  --pred outputs/phase1/<final-run>/output.zip \
  --documents data/raw/input \
  --policy data/manual_gold/compiled/phase1_annotation_policy.yaml \
  --stage type_selected=outputs/phase1/<type-run>/output \
  --stage repeat_recovery=outputs/phase1/<repeat-run>/output \
  --final-source-name pipeline_nonoverlap \
  --public-wer 51.6594 \
  --output-dir outputs/evaluation/<wer-analysis-run>
```

The report writes micro and macro-document WER proxies, per-type/source/document metrics,
source-by-type precision diagnostics, leave-one-source-out WER ablations, boundary kinds and
missing/extra fragments, stage deltas, and error rates by compiled knowledge status. WER proxy is
`100 * (1 - local text_score)` and is reported separately from the external public WER. WER uses
the existing Phase 1 text scorer; missing/spurious/boundary taxonomy uses same-type raw-span
overlap so repeated mentions at distant offsets cannot be paired as a fake boundary error.

Run dictionary-constrained candidate overlays without changing entities or assertions:

```bash
python scripts/run_phase1_candidate_ablation.py \
  --base outputs/phase1/<entity-run>/output.zip \
  --dictionary data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl \
  --gold-dir data/manual_gold \
  --input-dir data/raw/input \
  --output-dir outputs/phase1/<candidate-ablation-run>
```

`C1` uses exact unique TT06 ICD aliases, `C2` adds exact unique RxNorm names, and `C3` adds the
longest unique embedded drug alias for spans containing dose/route text. A stage is accepted only
when both holdout Phase 1 score and holdout candidate score increase. Ambiguous aliases, fuzzy
matches, toneless matches, and multi-code output remain disabled.

This is a local pre-submit gate, not a promotion gate. The July 11 C3 run increased holdout score
from `53.3426` to `62.4051` but decreased the public score from `38.7975` to `38.2289` because public
J_candidates fell from `30.0503` to `28.6287`. The reviewed manual gold has candidate labels on
almost every aligned diagnosis/drug and is not representative of hidden candidate prevalence or
code specificity. Keep candidate output empty by default until a narrower public-validated policy
wins; exact uniqueness within the local dictionary alone is insufficient.

## Top 10 Probe Suite

Build isolated entity, assertion, and candidate probes from a frozen Phase 1 artifact:

```bash
uv run python scripts/run_phase1_top10_probes.py \
  --base outputs/phase1/<best-run>/output.zip \
  --source pipeline=outputs/phase1/<pipeline-run>/output.zip \
  --source qwen=/path/to/qwen-output.zip \
  --source codex=/path/to/blind-codex-output.zip \
  --input-dir data/raw/input \
  --gold-dir data/manual_gold \
  --review-manifest data/manual_gold/review_manifest.jsonl \
  --dictionary data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl \
  --output-root outputs/phase1/top10_probes
```

The Top-1 campaign baseline and promotion thresholds are pinned in
`configs/phase1_top1_campaign.yaml`. New runs should pass
`--expected-base-sha256 ab9644f4d635132eee6462929461cf0d9b3bd3b9059e5a598704349410b54644`.
Candidate probes require three exact independent proposal sources by default and are emitted as
`C_*_ONE`, `C_*_THREE`, and `C_*_TEN`; submit the next tier only after the previous tier wins.
Compiled exclusions remain discovery evidence. Runtime exclusion requires a reviewed rule with an
entity type and, where necessary, context constraints in the rule registry.

The default run keeps the deterministic 15-document holdout sealed. Use `--open-holdout` only for
the planned day-6 checkpoint. The runner creates a content-hashed directory and never overwrites an
existing run. Each variant has its own `output/`, `output.zip`, SHA, train-only metrics, decision
trace, strict schema/offset/dictionary validation, and module-isolation report. Every suite run is
also appended to `outputs/loops/journal/phase1_top10_probe_runs.jsonl` and its Markdown index.

Entity execution order is fixed:

```text
lab-result precision/retyping
-> train-compiled strict exclusions
-> reviewed boundary expansion
-> overlap resolution
```

`review` exclusions never run. Numeric or qualitative lab results require a test/vital anchor in
the same clause. Medication dose/strength/route/frequency values are traced as internal medication
attributes and omitted from Phase 1 lab-result output. Boundary discovery uses only train documents,
requires repeated document support, rejects punctuation crossing, and writes draft rules to
`boundary_rule_candidates.yaml`; drafts do not execute. Review and promote general rules by changing
`review_status` in a registry such as `configs/phase1_top10_rule_registry.yaml`, then pass it through
`--rule-registry`.

Candidate probes remain empty unless the exact entity span/type is agreed by at least two of three
independent proposal sources. Candidate rules are compiled only from reviewed train labels, permit
one dictionary-valid code, require TT06 provenance for ICD, and separate RxNorm ingredient/brand
from SCD/SBD. Two-source or derived-source runs are useful for entity review but are explicitly
blocked from candidate promotion.

For full-pipeline candidate experiments, generated and exported sets are deliberately separate:

```text
generated candidates
-> absolute score threshold (global, type, or source calibrated)
-> relative margin from top score
-> at most five qualified candidates
-> Phase 1 export
```

`pipeline_report.py` reports generated count, qualified count, entities without a qualified
candidate, qualification reasons, and source counts for both sets. Legacy candidate JSON without a
`qualified` field defaults to unqualified, so old low-confidence traces cannot silently become a
submission candidate list.

Record every external grader result and keep/reject decision:

```bash
uv run python scripts/record_phase1_public_probe.py \
  --baseline baseline_grader.json \
  --trial trial_grader.json \
  --module entity \
  --probe-name E-OVERLAP \
  --artifact outputs/phase1/top10_probes/<run>/variants/E_OVERLAP/output.zip \
  --policy-diff '{"resolve_overlaps": true}'
```

The public gate requires final score improvement and target-metric improvement. Assertion/candidate
probes require non-target metrics to remain unchanged because entity identity is frozen. Entity
probes allow non-target metrics to improve, but never regress, because changing the matched entity
set also changes the grader denominator even when assertion/candidate fields remain frozen. The gate
appends JSONL and Markdown under `outputs/loops/journal`.

Candidate probe suites accept `--minimum-candidate-proposal-sources` (default `2`). Agreement means
an exact raw span and type emitted independently by at least that many configured proposal sources;
with two sources, this is strict 2-of-2 agreement, and with three sources it is 2-of-3 by default.
A candidate variant with zero emit decisions is never marked `probe_ready`, even if another candidate
field happened to change while materializing the ablation.

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
