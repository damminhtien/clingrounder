# Round 2 Inferred Golden

## Scope

Round 2 has no organizer-provided labels in the repository. The generated
`gold_strict` and `gold_review` directories are therefore inferred local
supervision, not official ground truth.

The public BTC medication-list example is treated as an executable
specification for:

- raw `[start, end)` offsets;
- medication spans that include contiguous strength, form, release, route,
  and frequency;
- a medication boundary that stops before `điều trị ...`;
- separately labeled indication symptoms;
- `isHistorical` on pre-admission medications;
- ICD-10 candidates only for `CHẨN_ĐOÁN` and RxNorm candidates only for
  `THUỐC`.

The exact example remains frozen in:

```text
tests/fixtures/phase1/btc_medication_list_crlf.txt
tests/fixtures/phase1/btc_medication_list_expected.json
tests/test_btc_phase1_sample.py
```

## Compilation Policy

`gold_strict` accepts an entity only when:

1. at least two named proposal sources agree on exact raw span and type; and
2. that span has no unresolved type or boundary conflict.

The only boundary override is a medication full span that
`MedicationMentionParser` can reproduce from an overlapping drug-name span.
The full span must itself be proposed, and the overlapping evidence must
cover at least two sources.

Candidates require one dictionary-valid code supported by at least two
sources. Ambiguous, conflicting, wrong-system, and unknown codes become
empty lists. Assertions are recomputed by the calibrated selective negation
and history policy. Lab test and lab result assertions remain empty.

`gold_review` is the validated union of all source proposals. Its
`review_queue.jsonl` records source-only entities, type conflicts, boundary
conflicts, and short medication spans superseded by the BTC full-span rule.
It is not training eligible until reviewed.

## Reproduce

Run after every proposal source covers all 100 documents:

```bash
uv run medical-kg benchmark phase1 round2 golden \
  --documents outputs/mining/phase1-round2-hosted-2026-07-27/documents.jsonl \
  --source-archive-sha256 \
    989d82404a9c1f3739e15d68a1e69d0f1f90d35c93c04ab0988e071fc1525545 \
  --source \
    baseline=outputs/phase1/round2/20260727T041845Z_round2-qwen-rx-unique-keep-icd_c5c0b16cdf/variants/C_RX_UNIQUE_KEEP_ICD/output \
  --source \
    friend31=outputs/phase1/round2/20260727T104010Z_round2-friend31-import_09e719190c/variants/E_FRIEND31_FULL_KNOWN/output \
  --source \
    qwen=outputs/models/phase1-qwen3-vietmed-consensus-e975f4a-r4/consensus
```

The command creates a content-hashed run under
`outputs/phase1/round2_golden/` and records:

```text
gold_strict/
gold_review/
gold_strict.zip
gold_review.zip
review_queue.jsonl
review_groups.jsonl
decisions.jsonl
invalid_proposals.jsonl
candidate_rejections.jsonl
policy.json
summary.json
summary.md
manifest.json
run_manifest.json
```

The manifest pins proposal artifacts, terminology files, BTC fixture hashes,
Git state, environment lock, source archive hash, and output ZIP hashes.

## Usage Restrictions

- Use `gold_strict` only as weak supervision or diagnostics.
- Do not report it as organizer gold.
- Do not use either layer as a blind challenge/evaluation set.
- Do not train from `gold_review` until its queue is resolved.
- Keep the Round 2 raw text and derived labels in authorized-private storage.

## Public Probe

The Min-2 `gold_strict.zip` was submitted once to measure how far consensus
weak supervision is from the hidden organizer labels:

```text
submitted: 2026-07-28 11:34
ZIP SHA-256: f2e01df4271d8fc1c0df9bb6b55f6c2e3861d6112ae89c39db9afa95e5276f90
final: 26.0282
WER: 70.9973
J_assertion: 32.5410
J_candidates: 18.9127
records: 100
```

Decision: reject as a submission artifact. Min-2 agreement is too sparse and
has substantially lower entity recall than the scored Friend-31 projection.
Keep it only for weak supervision and disagreement analysis.
