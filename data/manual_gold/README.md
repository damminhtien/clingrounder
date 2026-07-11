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
- Leave `candidates: []` when the correct code is not available in the validation dictionary.

Progress is tracked in `review_manifest.jsonl`.

Validation during review:

```bash
uv run python scripts/validate_manual_gold.py --allow-incomplete
```

Final validation, after all 100 files exist:

```bash
uv run python scripts/validate_manual_gold.py
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
