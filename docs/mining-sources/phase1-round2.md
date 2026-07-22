# Phase 1 Round 2 Competition Input

## Source Identity And Policy

The caller supplied a local ZIP containing the 100 Phase 1 Round 2 input documents released on
2026-07-22. The archive is private competition input, not an open training corpus. Its immutable
SHA-256 is:

```text
989d82404a9c1f3739e15d68a1e69d0f1f90d35c93c04ab0988e071fc1525545
```

The registry permits only local competition inference and local distribution audit. Redistribution,
hosted processing, pseudo-label training, annotation transfer, and runtime lookup memory are
prohibited. The raw archive and all parsed text remain outside Git in an encrypted local artifact
store.

## Import And Offset Contract

`PlainTextArchiveParser` accepts only regular `.txt` ZIP members, canonicalizes numeric file names,
decodes strict UTF-8, and preserves every decoded character. It rejects path traversal, symlinks,
encrypted members, duplicate canonical IDs, oversized archives, and compression bombs.

The completed import produced:

| Measure | Value |
| --- | ---: |
| source artifacts | 1 |
| parsed documents | 100 |
| unique texts | 100 |
| source ID range | 1-100 |
| total characters | 203,817 |
| mean characters/document | 2,038.17 |
| median characters/document | 1,838 |
| maximum characters/document | 4,481 |
| parser offset mismatches | 0 |

All document records carry archive SHA-256, member name, raw-byte SHA-256, encoding, and newline
mode. A second run reused both cached stages and reproduced the canonical document manifest SHA-256
`60a83690ef97a5dc6201f7877f808f593a6d86914678efeb3437814a0cba005f`.

## Distribution Audit

The audit is benchmark-owned and explicitly `runtime_eligible: false`. It emits aggregate counts,
document IDs, hashes, and similarity scores only. It does not emit source text, entity spans, entity
types, assertions, or candidates.

Observed shape:

| Measure | Value |
| --- | ---: |
| documents with bullet/list formatting | 91 |
| documents containing masked text | 30 |
| clinical-style documents | 37 |
| clinical + question/answer documents | 30 |
| question/answer documents | 22 |
| other mixed style combinations | 11 |

Round 2 reuses substantial wording from the prior 100-note corpus:

| Duplicate evidence | Value |
| --- | ---: |
| exact duplicate documents within Round 2 | 0 |
| documents with an exact prior-corpus line | 98 |
| documents with an exact prior-corpus 8-word shingle | 98 |
| exact-line character fraction | 0.423932 |
| best prior-document Jaccard at least 0.25 | 49 |
| best prior-document Jaccard at least 0.50 | 39 |

This overlap is diagnostic evidence only. It must not be used to copy old annotations into new
documents or to introduce document-specific output rules.

Ten source documents have no exact match against any prior human-gold entity context using a
32-character window and form the priority novelty queue:

```text
1, 24, 40, 48, 76, 79, 81, 83, 84, 94
```

The audit output fingerprints are:

| Artifact | SHA-256 |
| --- | --- |
| `profile.json` | `161074a5c4220ef8309a87da04f4975ec930d9ed52f54ee7bee5fc825b66ce5e` |
| `duplicate_report.json` | `176bfdec087eacb99fb8ae4b21aebc1ed026cefa49993de864c88cb16fff677e` |
| `novelty_queue.jsonl` | `460400c1b49fa3f158bd7ed93dd8577821dbbbaf354f1d62e0a9923854083d3b` |

## Promotion Boundary

- Allowed: local inference, aggregate profiling, duplicate diagnostics, and manual prioritization.
- Forbidden: supervised/pseudo-label training on Round 2, copying prior gold by duplicate context,
  document-ID rules, hosted processing, and redistribution.
- The 10-document novelty queue identifies what to inspect first; it contains no annotation proposal.
- Any model used on these documents must have been frozen before reading Round 2 input.

## Reproduce

Install the locked environment, point the plan to an authorized local copy, and use an encrypted
local content-addressed store:

```bash
uv sync --frozen --extra dev --extra data
export PHASE1_ROUND2_ARCHIVE=/secure/input_turn2_vong1.zip
export MEDICAL_KG_ARTIFACT_STORE=file:///secure/medical-kg/mining-artifacts

uv run medical-kg data registry validate \
  --registry data/sources/mining_registry.yaml \
  --processing-index data/sources/processing_status.yaml

uv run medical-kg data run \
  --plan configs/mining/phase1-round2-2026-07-22.yaml
```

Until the benchmark CLI exposes the audit command, reproduce the deterministic reports through its
public Python API:

```bash
uv run python - <<'PY'
from medical_kg_nlp.benchmarks.phase1.round2 import (
    build_phase1_round2_audit,
    write_phase1_round2_audit,
)
from medical_kg_nlp.mining.io import load_documents

documents_path = "outputs/mining/phase1-round2-2026-07-22/documents.jsonl"
audit = build_phase1_round2_audit(
    load_documents(documents_path),
    reference_input_dir="data/raw/input",
    reference_gold_dir="data/manual_gold",
    reference_split_manifest="data/manual_gold/holdout_manifest.json",
)
write_phase1_round2_audit(
    audit,
    "outputs/mining/phase1-round2-2026-07-22/audit",
    documents_manifest_path=documents_path,
)
PY
```

The output root is local and ignored by Git:

```text
outputs/mining/phase1-round2-2026-07-22/
```
