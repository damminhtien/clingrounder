# Manual Phase 1 Gold

This directory contains manually reviewed Phase 1 gold files for the 100 raw notes in
`data/raw/input`.

Gold files use the official Phase 1 flat schema:

- `text`
- `type`
- `assertions`
- `candidates`
- `position`

Review rules:

- Read the full raw note before accepting any entity.
- Old model outputs may be used only as draft hints, never copied as gold without raw-text review.
- Preserve raw `[start, end)` offsets exactly: `raw_text[start:end] == text`.
- Use only the 5 Phase 1 types: `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`,
  `CHẨN_ĐOÁN`, `THUỐC`.
- Use assertions only for `TRIỆU_CHỨNG`, `CHẨN_ĐOÁN`, and `THUỐC`.
- Use ICD-10 candidates only for `CHẨN_ĐOÁN` and RxNorm candidates only for `THUỐC`.
- Leave `candidates: []` when no exact code can be justified. A non-empty code must exist in the
  locked TT06 or RxNorm release; rebuild the reviewed candidate dictionary after changing it.

Progress is tracked in `review_manifest.jsonl`. A `review_candidates` item defaults to
`scope: entity`; use `candidate_mapping` or `annotation_note` when the note explains an accepted
label rather than rejecting an entity proposal. The synchronizer realigns stale review positions
to exact raw surfaces and nulls evidence that cannot be bound unambiguously.

Validation during review:

```bash
uv run python scripts/sync_manual_gold_manifest.py
uv run python scripts/build_manual_gold_candidate_dictionary.py
uv run python scripts/validate_manual_gold.py --allow-incomplete
```

`reviewed_candidate_concepts.jsonl` is a compact normalization/validation resource. It includes
only codes used by reviewed labels and is deliberately not the NER recognition dictionary.

Audit against conventions demonstrated by official BTC samples:

```bash
uv run python scripts/audit_manual_gold_convention.py --strict
```

The audit is intentionally non-mutating. `blocking` findings must be resolved before completion;
`review` findings require an explicit concept-level decision in `convention_decisions.jsonl`.
Document ids and absolute offsets are forbidden in that decision file.

Final validation, after all 100 files exist:

```bash
uv run python scripts/build_manual_gold_candidate_dictionary.py
uv run python scripts/validate_manual_gold.py
uv run python scripts/audit_manual_gold_convention.py --strict
```

Compile reviewed labels, guideline notes, and rejected mentions into concept-level knowledge:

```bash
uv run python scripts/build_phase1_annotation_knowledge.py
```

This writes `compiled/phase1_annotation_policy.yaml`, `annotation_knowledge.json`,
`policy_conflicts.csv`, `conflict_summary.json`, and `report.md`. Strict aliases require support
from at least two reviewed documents and cannot have a type or accepted-vs-rejected conflict.
Document identifiers are retained only as audit provenance; the runtime policy contains no
document-specific output rules.

The runtime build uses the train split by default. Audit all 100 reviewed documents separately so
holdout problems cannot be hidden while holdout labels remain excluded from runtime rule discovery:

```bash
uv run python scripts/build_phase1_annotation_knowledge.py \
  --split all \
  --output-dir outputs/evaluation/manual_gold_annotation_all
```

The command fails when any conflict remains. Use `--allow-unresolved-conflicts` only while producing
an exploratory queue for review, never for the final compiled policy.
