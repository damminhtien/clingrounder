# Evaluation

Evaluation is built around the internal schema so public dataset adapters can normalize labels before
metrics are computed.

## Phase 1 Decision Authority

Effective 2026-07-30, local Phase 1 metrics are diagnostic telemetry only. Exact-span F1, local
WER proxies, manual-gold splits, and mined-corpus benchmarks may prioritize experiments, but they
must not promote or reject a model. A keep/reject decision requires a recorded official submission
result.

Final-fit training uses all 100 manually reviewed records plus the owner-authorized, checksum-pinned
private GT source. The old 76/24 and 60/16 splits remain useful for error analysis and training
health, but no document stays excluded from final fit. Structural validation remains blocking:
invalid offsets, schemas, code systems, parameter budgets, or artifact fingerprints must not be
submitted.

The final XLM-R source also includes a bounded renderer-derived Vietnamese Q&A/educational view to
teach genre-specific span boundaries. It copies only raw-offset-valid `train` records whose source
is `synthetic:phase1-region-renderer.v1`; manual records are not duplicated, and Round 2,
quarantine, and Friend31 provenance are rejected. Recreate the ignored training artifact rather
than copying it between machines:

```bash
uv run medical-kg benchmark phase1 model-data build-final-fit-bundle \
  --output-dir outputs/mining/model-datasets/phase1-final-supervision-qa-edu-v1
```

The current deterministic build contains 901 chunks, 19,270 entities, and 54 approved synthetic
records (5.99%). The default Vast token-classifier runner materializes this bundle before training.

### Joint Span OOF Calibration

The joint span/type cross encoder uses a separate document-grouped OOF run before final fitting.
Each fold trains only on the other documents, emits a probability for the candidate's compatible
`EXACT_*` class, and writes `oof_observations.jsonl` without raw note text. Its coverage report
uses unique lattice candidate identities and fails if any candidate or document is unscored.

The calibration artifact is pinned to a **training family** rather than to a fold checkpoint SHA:
the final model must prove the same full supervision data, initializer, label contract, and
optimization recipe. This avoids the invalid claim that a final-fit model itself produced OOF
scores. There is no clinical fallback for missing `qa` or `educational` genre/type support.

Prepare the mixed-genre lattice before OOF fitting. The adapter turns every token chunk into an
immutable child document and groups all chunks plus renderer children of the same source note into
one OOF fold. It validates the token bundle manifest and rejects Round 2 and Friend31 provenance:

```bash
uv run medical-kg benchmark phase1 joint-span prepare-token-bundle \
  --dataset outputs/mining/model-datasets/phase1-final-supervision-qa-edu-v1/spans.jsonl \
  --dataset-manifest outputs/mining/model-datasets/phase1-final-supervision-qa-edu-v1/manifest.json \
  --bundle-build-manifest outputs/mining/model-datasets/phase1-final-supervision-qa-edu-v1/build_manifest.json \
  --model-source qwen=outputs/models/phase1-qwen3-final-token-bundle-source \
  --source-role qwen=llm \
  --model-source xlmr=outputs/models/phase1-xlmr-final-token-bundle-source \
  --source-role xlmr=token_model \
  --output-dir outputs/models/phase1-joint-span-mixed-genre
```

The command accepts no model source only for a rule/structured-medication bootstrap that checks
bundle, offset, and OOF-group contracts. That artifact is not a submission verifier: its candidate
coverage report diagnoses proposal recall, and final training should add independently materialized
Qwen and XLM-R sources over the same child document IDs.

Materialize those sources with the same bundle paths. This avoids mixing spans emitted from parent
notes with chunk-local training coordinates:

```bash
uv run medical-kg benchmark phase1 qwen propose-token-bundle \
  --config configs/benchmarks/phase1/models/phase1-qwen3-8b-qlora-portable-inference-2026-07-31.yaml \
  --dataset outputs/mining/model-datasets/phase1-final-supervision-qa-edu-v1/spans.jsonl \
  --dataset-manifest outputs/mining/model-datasets/phase1-final-supervision-qa-edu-v1/manifest.json \
  --output-dir outputs/models/phase1-qwen3-final-token-bundle-source

uv run medical-kg benchmark phase1 joint-span materialize-token-bundle-source \
  --dataset outputs/mining/model-datasets/phase1-final-supervision-qa-edu-v1/spans.jsonl \
  --dataset-manifest outputs/mining/model-datasets/phase1-final-supervision-qa-edu-v1/manifest.json \
  --model-path outputs/models/phase1-xlmr-dapt/final-model \
  --model-fingerprint <checkpoint-directory-sha256> \
  --model-id FacebookAI/xlm-roberta-base \
  --base-revision <pinned-base-revision> \
  --device cuda \
  --output-dir outputs/models/phase1-xlmr-final-token-bundle-source
```

On Vast, the matching reusable-template runners are
`scripts/benchmarks/phase1/vast/run_qwen_token_bundle.sh` and
`scripts/benchmarks/phase1/vast/run_joint_span_token_bundle_source.sh`. They expect the pinned bundle and model
directory to have been copied to the instance or persistent volume, preserve `HF_HOME`/pip caches,
and resume only complete raw-text-verified Qwen child outputs.

On a cached CUDA template, run:

```bash
DATASET_DIR=outputs/models/phase1-joint-span-mixed-genre \
INITIALIZATION_MODEL=outputs/models/phase1-xlmr-dapt/final-model \
INITIALIZATION_FINGERPRINT=<sha256> \
scripts/benchmarks/phase1/vast/run_joint_span_oof.sh
```

Then materialize the final cross encoder with the identical training family and fit the resolver
calibration from the OOF observations only:

```bash
uv run medical-kg benchmark phase1 joint-span calibrate \
  --observations outputs/models/phase1-joint-span-xlmr-dapt-oof-2026-08-01/oof_observations.jsonl \
  --training-family-fingerprint <manifest-training-family-sha256> \
  --fold-assignment-sha256 <fold-assignments-json-sha256> \
  --output outputs/models/phase1-joint-span-calibration.json \
  --report outputs/models/phase1-joint-span-calibration-report.json
```

### Joint Span Source Collection

The joint span/type verifier is trained on a proposal lattice rather than only gold spans. Qwen
exact-quote inference must therefore run over the governed final-fit corpus and write a source
artifact, not a candidate submission. The command uses raw `[start, end)` projection and retains
separate recall/targeted pass evidence for the learned fusion stage:

```bash
uv run medical-kg benchmark phase1 qwen propose-final-supervision \
  --config configs/benchmarks/phase1/models/phase1-qwen3-8b-qlora-portable-inference-2026-07-31.yaml \
  --output-dir outputs/models/phase1-qwen3-final-supervision-source
```

The owner-authorized archive is supplied through the governed `PHASE1_PART2_ARCHIVE` environment
variable or `--authorized-archive`; it is never copied into a distributable source artifact.
On Vast, use `scripts/benchmarks/phase1/vast/run_qwen_final_supervision.sh`, which reuses the CUDA template and
shared Hugging Face cache. The resulting Qwen rows are proposal evidence only: the learned joint
verifier performs global non-overlap selection, then assertion and linking are recomputed for the
selected raw identities.

Friend31 output is distillation/reference evidence only. It is not a valid runtime source,
submission seed, or reproducible baseline. The machine-readable contract is
`configs/benchmarks/phase1/models/phase1-training-governance-2026-07-30.yaml`.

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
uv run medical-kg pipeline run \
  --input data/samples/sample_notes.jsonl \
  --output outputs/predictions.jsonl

uv run medical-kg validate \
  --profile development \
  --pred outputs/predictions.jsonl \
  --documents data/samples/sample_notes.jsonl \
  --dictionary data/dictionaries/seed_concepts.jsonl

uv run medical-kg evaluate \
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

uv run medical-kg benchmark phase1 submission \
  --input-dir data/raw/input \
  --output-dir outputs/phase1/current/output \
  --zip outputs/phase1/current/output.zip \
  --assertion-policy empty \
  --candidate-policy empty
```

### Full terminology runtime

`configs/benchmarks/phase1/submission/full.yaml` uses the full TT06 ICD-10 and July 6, 2026 RxNorm releases for
normalization. It does not use the seed dictionary as the complete linking database. The smaller
recognition store is intentional: NER trigger coverage and normalization vocabulary are different
precision controls.

Build the full terminology index explicitly, then point the pipeline config at the resulting
immutable SQLite file. `configs/benchmarks/phase1/submission/full-diagnostic.yaml` remains an experiment config for
profiling lexical sources and internal stages; it should not be promoted merely because it returns
more candidates.

```bash
uv run medical-kg terminology build \
  --source data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl \
  --source data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl \
  --cache-dir .cache/medical-kg/terminology

uv run medical-kg pipeline run \
  --input data/samples/sample_notes.jsonl \
  --output outputs/benchmarks/phase1_full/predictions.jsonl \
  --config configs/benchmarks/phase1/submission/full-diagnostic.yaml \
  --run-root outputs/benchmarks/phase1_full \
  --run-label all-lexical
```

The benchmark sums counters across documents, reports initialization separately from processing,
and hashes output ZIP files. Matching ZIP hashes are the required regression check for a pure
runtime optimization. See [ADR 0005](decisions/0005-terminology-retrieval-scaling.md) for the
database, FTS, and ANN decision.

Terminology verification has three deliberately separate roles:

- `tests/benchmarks/phase1/test_btc_phase1_sample.py` checks the organizer's executable span/type/assertion/candidate
  convention only. Its benchmark-local vocabulary and reviewed mappings are not runtime defaults.
- `tests/test_standard_release_contract.py` checks current TT06 and July 6, 2026 RxNorm manifests,
  profile paths, a July-only RxCUI, and typed repository queries against both full releases.
- terminology query-set benchmarks measure retrieval recall/rank/abstention. They must report
  unknown codes and ambiguous exact aliases rather than silently treating either as a top-1 error
  or a unique assignment.

`medical-kg terminology benchmark` reports rank metrics plus a fixed lexical-score abstention
curve. Treat query semantics explicitly:

- CodiEsp codes absent from TT06 are coverage gaps, not retriever errors, and must not calibrate an
  emission threshold.
- DailyMed bare-name queries linked to one or more specific products measure candidate recall, not
  top-1 product assignment. A product-level target requires strength/form evidence.
- BTC examples remain executable benchmark fixtures under `benchmarks/phase1`; they do not provide
  default recognition terms or reviewed linking memory to the reusable pipeline.

The profiler output is intended to be a cacheable experiment artifact. Use it to compare train/dev
entity distributions, span lengths, context cue frequency, dictionary coverage, and unseen-code
risks before adding larger models.

### Proposal calibration

Phase 1 proposal fusion uses a portable sparse logistic verifier rather than treating source
agreement as an emission rule. By default the dataset builder reads only frozen model
train/development documents and verifies that the legacy 24-document holdout is excluded. A
training-governance file may explicitly authorize all 100 reviewed documents for final fitting.
Round 2 input remains excluded. The manifest fingerprints both the source matrix and the exact
serialized feature rows. A positive label means exact raw span and exact Phase 1 type; boundary,
type, boundary+type, and spurious proposals are explicit negatives.

```bash
uv run medical-kg benchmark phase1 proposal-matrix \
  --target-source rule=outputs/phase1/<rule>/output.zip \
  --internal-source xlmr=outputs/models/<xlmr>/predictions.jsonl \
  --target-source qwen=outputs/phase1/<qwen>/output.zip \
  --compatible-source vietmed=outputs/models/<vietmed>/support \
  --output-dir outputs/phase1/proposal-fusion/matrix

uv run medical-kg benchmark phase1 proposal-calibrate \
  --matrix outputs/phase1/proposal-fusion/matrix/proposal_matrix.jsonl \
  --source-role rule=rule \
  --source-role xlmr=token_model \
  --source-role qwen=llm \
  --source-role vietmed=verifier \
  --minimum-development-precision 0.90 \
  --output-dir outputs/phase1/proposal-fusion/calibrated

uv run medical-kg benchmark phase1 proposal-resolve \
  --matrix outputs/phase1/proposal-fusion/matrix/proposal_matrix.jsonl \
  --verifier outputs/phase1/proposal-fusion/calibrated/verifier.json \
  --source-role rule=rule \
  --source-role xlmr=token_model \
  --source-role qwen=llm \
  --source-role vietmed=verifier \
  --output-dir outputs/phase1/proposal-fusion/resolved
```

The final-fit public-probe variant is explicit:

```bash
uv run medical-kg benchmark phase1 proposal-calibrate \
  --matrix outputs/phase1/proposal-fusion/matrix/proposal_matrix.jsonl \
  --source-role rule=rule \
  --source-role xlmr=token_model \
  --source-role qwen=llm \
  --source-role vietmed=verifier \
  --fit-mode full_oof \
  --training-governance \
    configs/benchmarks/phase1/models/phase1-training-governance-2026-07-30.yaml \
  --output-dir outputs/phase1/proposal-fusion/final-fit
```

`full_oof` fits every proposal label supplied by the governed dataset and derives probability
calibration plus thresholds from document-grouped OOF predictions. It is not a local promotion
gate: the report sets `local_metrics_role: diagnostic_only` and `auto_promote: false`.

`--minimum-development-precision` selects the highest-recall threshold meeting that precision
separately for each entity type. Omitting it selects by development F1 and is suitable for source
replacement experiments, not conservative additions to a strong baseline. The verifier records
its feature contract, feature-row SHA-256, thresholds, and calibration objective.

The matrix retains per-source presence, confidence, source-task labels, and support-only status.
The model also uses bounded mention/context hashes, section, genre, question/answer role, span
length, type, source counts per role, verifier support, overlap/containment structure, and
same-span type conflicts. Source names are mapped to portable roles before becoming features.

Genre is calibrated as `clinical`, `qa`, or `educational` while one sparse verifier is shared.
The artifact may store a probability calibrator and per-type operating point for each genre, but
only when train/development support contains enough positive and negative proposals. Otherwise it
records `fallback_reason` and uses the global calibration. The current frozen Phase 1 development
split contains only clinical notes, so Q&A and educational overrides must be learned from a future
labeled corpus rather than copied from hand-written thresholds.

### Boundary verifier

Boundary verification is a separate ranking stage. It retains every original token-model and LLM
quote, then adds overlapping alternatives from the recognition trie, clinical boundary grammar,
structured medication parser, coordination splitting/merging, and bounded token windows. It never
edits normalized text back into raw offsets. Each candidate is labeled `CORRECT`, `TOO_SHORT`,
`TOO_LONG`, or `WRONG_ENTITY`; the binary rank target is exact span plus exact type.

```bash
uv run medical-kg benchmark phase1 boundary-calibrate \
  --matrix outputs/phase1/proposal-fusion/matrix/proposal_matrix.jsonl \
  --proposal-verifier outputs/phase1/proposal-fusion/calibrated/verifier.json \
  --dictionary data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl \
  --source-role rule=rule \
  --source-role xlmr=token_model \
  --source-role qwen=ensemble \
  --source-role vietmed=verifier \
  --output-dir outputs/phase1/proposal-fusion/boundary

uv run medical-kg benchmark phase1 boundary-resolve \
  --matrix outputs/phase1/proposal-fusion/matrix/proposal_matrix.jsonl \
  --proposal-verifier outputs/phase1/proposal-fusion/calibrated/verifier.json \
  --boundary-verifier outputs/phase1/proposal-fusion/boundary/verifier.json \
  --dictionary data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl \
  --source-role rule=rule \
  --source-role xlmr=token_model \
  --source-role qwen=ensemble \
  --source-role vietmed=verifier \
  --output-dir outputs/phase1/proposal-fusion/boundary-resolved
```

The persisted dataset includes a stable cross-encoder representation:
`[GENRE]`, `[SECTION]`, `[QA_ROLE]`, `[TYPE]`, `[LEFT]`, `[ENTITY]`, and `[RIGHT]`.
The default verifier is a fast sparse joint ranker; a pinned Hugging Face sequence classifier can
consume the same rows later without changing candidate generation or evaluation. Thresholds are
selected per type and per genre only with sufficient labeled development support.

Manual-gold scores in `training_report.json` are diagnostic only. They may expose offset bugs,
candidate-family collapse, or relative boundary errors, but they do not authorize promotion to a
Round 2 submission. The report always sets `auto_promote: false`; a frozen public probe is required
because the reviewed corpus has not reproduced the strongest public distribution.

When a frozen proposal verifier supplies the base entity set, the persisted boundary verifier uses
`conservative_replacement`: it preserves every selected entity slot and may only replace its span
inside one unambiguous family. It cannot add or delete entities. Without a proposal verifier, the
artifact is explicitly marked `open_ranker` and should be treated as a broader experimental probe.

Calibrated runtime proposals are represented as an explicit conflict graph. Edges distinguish
same-span type conflicts, boundary overlap, containment, exact duplicates, and drug-versus-lab
number ambiguity. Connected components are suitable for bounded model adjudication. Final
non-overlap selection uses calibrated log-odds with weighted interval scheduling; it does not use
greedy highest-probability suppression or treat rule/model confidence scores as probabilities.

Candidate generation and cross-encoder rows can be much larger than the fitted sparse model.
Interrupted training can resume from the fingerprinted materialized dataset without regenerating
trie, grammar, medication, token-window, and coordination alternatives:

```bash
uv run medical-kg benchmark phase1 boundary-calibrate \
  --dataset-dir outputs/phase1/proposal-fusion/boundary/dataset \
  --output-dir outputs/phase1/proposal-fusion/boundary-resumed
```

Probability calibration has two explicit modes:

- `development` preserves the legacy 60/16 diagnostic and keeps the old holdout sealed;
- `full_oof` fits all governed supervision and evaluates Platt calibration with a nested,
  document-grouped cross-fit before selecting per-type and per-genre thresholds.

The overlap resolver uses calibrated log-odds margin. Source count, model confidence, agreement,
span length, and conflict structure are learned inputs; none are reapplied as hard-coded resolver
bonuses or tie-breaks.

Apply a frozen verifier to an unlabeled Round 2 proposal source with:

```bash
uv run medical-kg benchmark phase1 round2 proposal-verifier \
  --documents outputs/mining/phase1-round2-<run>/documents.jsonl \
  --source-archive-sha256 <source-sha256> \
  --base outputs/phase1/round2/<baseline>/output.zip \
  --expected-base-sha256 <baseline-sha256> \
  --proposal-source outputs/models/<proposal-source> \
  --expected-proposal-source-sha256 <proposal-source-sha256> \
  --verifier outputs/phase1/proposal_calibration/<name>/verifier.json \
  --expected-verifier-sha256 <verifier-sha256>
```

The Round 2 command is additive-only: baseline rows remain unchanged by value, new rows emit empty
assertions/candidates, overlaps are blocked, structural section labels are rejected, and the
directory plus deterministic ZIP pass strict validation before the artifact is reported.

### Proposal-source and disease/symptom evaluation

Score rule, token-model, LLM, and source-task support proposals on the same frozen 60/16 split:

```bash
uv run medical-kg benchmark phase1 proposal-score \
  --target-source rule=outputs/phase1/<rule>/output.zip \
  --internal-source xlmr=outputs/models/<run>/development_predictions.jsonl \
  --target-source qwen=outputs/phase1/<qwen>/output.zip \
  --compatible-source vietmed=outputs/models/<vietmed>/support \
  --output-dir outputs/evaluation/phase1-proposal-sources
```

The scorer never opens the 24-document holdout. A source must cover a complete split or none of it;
partial coverage fails. Ordinary target sources receive exact span+type and relaxed same-type
overlap metrics. Compatible source-task labels are scored separately:

- duplicate compatibility rows are grouped into one raw span;
- `DISEASESYMTOM -> {CHẨN_ĐOÁN, TRIỆU_CHỨNG}` measures compatible-span support;
- compatible support is not reported as target-label precision and cannot enter output directly.

Train the dedicated target-task type verifier with:

```bash
uv run medical-kg benchmark phase1 type-verifier \
  --matrix outputs/phase1/<run>/proposals/proposal_matrix.jsonl \
  --representation-source outputs/models/<vietmed>/support \
  --output-dir outputs/models/phase1-disease-symptom-verifier
```

Its contract is:

```text
mention + local context + section + question/answer role + optional model evidence
→ DISEASE | SYMPTOM | NONE
```

`NONE` is the default. There is no disease fallback. VietMed `DISEASESYMTOM` is an optional
representation feature only; labels are always derived from reviewed Phase 1 train data. The
dataset includes other entity types, spurious proposals, and boundary conflicts as hard negatives.

Phase 1 model promotion never reads baseline numbers from Python defaults. The holdout gate loads
`data/manual_gold/model_holdout_baseline.json` and verifies its frozen-split SHA, model-split SHA,
corpus fingerprint, and holdout ID fingerprint before comparison. A corpus or split change therefore
fails closed until a new baseline artifact is produced and reviewed.

Threshold calibration ranks trials by the official Phase 1 score, with text score and error counts
used only as deterministic tie-breakers. Because the current development split has only 16
documents, every per-type search also reports support, five-repeat duplicate-group cross-validation,
and a deterministic 200-sample bootstrap 95% interval. These fields are diagnostics; they do not
open holdout labels or bypass the fingerprinted promotion gate.

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
uv run medical-kg pipeline run \
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

The stable Phase 1 CLI takes explicit input, output, dictionary, parallelism, and export-policy
arguments. It validates the directory before creating a deterministic ZIP, validates the ZIP again
against raw TXT offsets and dictionary candidates, and disables relation extraction because Phase 1
exports only flat entities. Historical `configs/phase1_*.yaml` files remain experiment records; they
are not an implicit second configuration source for the benchmark command.

The conservative competition baseline uses `--assertion-policy empty` and
`--candidate-policy empty`, preserving entity text/type/offsets while abstaining on the two high-risk
fields. The stable CLI also supports `pipeline`. Selective calibration remains task experiment code
under `benchmarks/phase1` and is not part of the reusable command surface.

Compile reviewed mappings and calibrate emit versus abstain with document-fold cross-validation:

```bash
uv run python scripts/benchmarks/phase1/build_phase1_reviewed_candidates.py --split train
uv run python scripts/benchmarks/phase1/calibrate_phase1_candidates.py \
  --pred outputs/phase1/<full-run>/phase1/full_internal_predictions.jsonl \
  --reviewed-map data/manual_gold/reviewed_candidate_map.jsonl \
  --dictionary data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl \
  --dictionary data/standards/rxnorm/processed/rxnorm_full_07062026_concepts.jsonl \
  --output-dir outputs/phase1/<full-run>/calibration
```

The whitelist compiler requires a single reviewed code, exact unique type-compatible dictionary
match, TT06 provenance for ICD, and an allowed RxNorm term type. Calibration cross-fits the
whitelist per document fold and compares actual one-code Jaccard against empty-output Jaccard.
`candidate_calibration.json` reports source and policy groups; local recommendation is still only a
pre-submit gate because public candidate prevalence has differed from manual gold.
Calibrated probabilities must be copied explicitly into
`pipeline.link_emit_probabilities_by_source`. Missing source entries resolve to probability zero;
the runtime never substitutes a retrieval score or a hard-coded exact-match confidence.
The full config also persists internal prediction and trace JSONL files inside the hashed run, so
every calibration result can be tied to exact candidate provenance and stage counters.

When terminology files are supplied, calibration also reports exact-span candidate coverage by
gold nullness and terminology granularity: RxNorm ingredient, brand, SCD/SBD, and ICD parent/leaf.
Each bucket includes support, prediction coverage, exact-set accuracy, top-1 accuracy, and mean
Jaccard. These are distribution diagnostics, not automatic promotion gates; unknown codes remain
visible as `terminology_unknown` instead of being guessed from code shape.

Candidate calibration also writes a config-ready `expected_jaccard_policy` block. It estimates null
gold prevalence by code system and candidate correctness by primary source and rank. The
benchmark-only selective export API can then choose an empty set or a ranked prefix of one to five
codes instead of requiring exactly one eligible code. This policy is intentionally absent from the
stable submission CLI because the available calibration set is not blind:

```yaml
selective:
  candidates:
    selection_policy: expected_jaccard
    max_candidates: 5
    minimum_expected_jaccard_gain: 0.05
    empty_probabilities:
      ICD-10: 0.40
      RxNorm: 0.55
    rank_probabilities:
      ICD-10:
        dictionary_exact: [0.90, 0.30, 0.10]
      RxNorm:
        dictionary_exact: [0.85, 0.25]
```

The selector computes expected set Jaccard for each contiguous prefix under the calibrated marginal
model and compares it with the calibrated empty-set score. Missing rank calibration terminates the
prefix rather than falling back to retrieval score. Do not copy values from the example above;
generate them on a blind policy-training split and keep the policy holdout unopened until the
candidate regime is frozen.

Evaluate reviewed Phase 1 files without converting them to the internal schema:

```bash
python scripts/benchmarks/phase1/evaluate_phase1_manual_gold.py \
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
python scripts/benchmarks/phase1/ensemble_phase1_outputs.py \
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
uv run python scripts/benchmarks/phase1/analyze_phase1_entity_wer.py \
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

Freeze the complete reviewed corpus before an entity ablation campaign:

```bash
uv run python scripts/benchmarks/phase1/freeze_phase1_holdout.py
```

`data/manual_gold/holdout_manifest.json` fingerprints every numeric gold JSON and raw TXT. Running
the command again verifies the frozen corpus and fails if either labels or source text drift. Use
`--replace` only after an intentional corpus revision. The current 100-document corpus freezes 76
train and 24 holdout documents under the SHA-256/mod-5 policy.

This split is a **legacy regression holdout**, not a blind policy holdout: its labels and aggregate
results have already informed rule development. A valid blind policy holdout must come from newly
annotated documents whose labels are inaccessible to rule/alias authors until the candidate,
assertion, and entity policies are frozen. Do not relabel a subset of these 100 reviewed files as
blind. Candidate calibration defaults to the legacy train split for development, and its result is
only a safety signal until evaluated once on genuinely unseen policy data. Report candidate metrics
by null/non-null, ICD parent/leaf, and RxNorm ingredient/brand/SCD/SBD buckets on that corpus.

For a new task, seal the raw-document split before annotation or rule development:

```bash
uv run python scripts/benchmarks/phase1/manage_policy_holdout.py create \
  --documents data/new_task/input \
  --corpus-id new-task-v1 \
  --output data/new_task/policy_holdout.sealed.json
```

The sealed manifest contains source hashes and split IDs but no label hashes. After the policy is
frozen, put exactly the holdout labels in a separate directory and create a new opened record:

```bash
uv run python scripts/benchmarks/phase1/manage_policy_holdout.py open \
  --documents data/new_task/input \
  --manifest data/new_task/policy_holdout.sealed.json \
  --holdout-gold-dir data/new_task/holdout_gold \
  --output outputs/new_task/policy_holdout.opened.json
```

Opening refuses an all-corpus gold directory and never mutates the sealed manifest. This is an
audit protocol, not an access-control boundary; repository permissions or an external reviewer
must still prevent rule authors from reading holdout labels early.

Run lab-result, medication-span, symptom-boundary, and diagnosis-boundary families independently:

```bash
uv run python scripts/benchmarks/phase1/run_phase1_entity_ablations.py \
  --base outputs/phase1/<public-best>/output.zip \
  --expected-base-sha256 <sha256> \
  --stage pipeline=outputs/phase1/<pipeline>/output.zip \
  --stage qwen_nonoverlap=outputs/phase1/<ensemble>/output.zip \
  --public-wer <public-wer>
```

The runner learns boundary rules from train only, fixes all variants before opening holdout, writes
strict-validated deterministic ZIP files, and emits a full WER/source/boundary report per variant.
It never combines winning families. `keep_candidate` means both all-corpus and holdout WER improve;
public promotion still requires an isolated external probe.

Run dictionary-constrained candidate overlays without changing entities or assertions:

```bash
python scripts/benchmarks/phase1/run_phase1_candidate_ablation.py \
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
uv run python scripts/benchmarks/phase1/run_phase1_top10_probes.py \
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
`configs/benchmarks/phase1/experiments/top1-campaign.yaml`. New runs should pass
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
`review_status` in a registry such as `data/manual_gold/derived/top10-rule-registry.yaml`, then pass it through
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
candidate, qualification reasons, and source counts for both sets. Internal candidate JSON must
provide `retrieval_score`, `emit_probability`, `source`, `evidence_sources`, `qualified`, and a
qualification reason. Legacy candidate JSON is rejected.

Record every external grader result and keep/reject decision:

```bash
uv run python scripts/benchmarks/phase1/record_phase1_public_probe.py \
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
