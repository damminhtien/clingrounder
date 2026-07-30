# Phase 1 Part 2 Authorized Training Source

## Effective Policy

On 2026-07-30 the data owner authorized this private, checksum-pinned bundle for supervised local
and hosted training. This policy supersedes the earlier quarantine restriction. The historical
aggregate audit remains in `phase1-part2-quarantine.md` because it records useful distribution and
offset evidence.

The source may be used for:

- supervised NER, assertion, and candidate training;
- distillation and proposal-verifier training;
- local diagnostic evaluation;
- temporary owner-authorized hosted GPU processing.

It may not be used as document-specific runtime memory, copied directly into a current submission,
or redistributed. Final predictions must still be generated from raw input by repository-owned
code and pinned model artifacts.

## Immutable Inputs

| Artifact | SHA-256 | Records |
| --- | --- | ---: |
| outer archive | `46da2a7718078b95024e97feb66d49e44917d7b00b3981bad3eaaae13adc418e` | 2 nested ZIPs |
| `input.zip` | `ecb0bb792ad8649b06dcfd10847a1d33c963a9332ac4690c56b0004643eceb5c` | 100 TXT |
| `gt.zip` | `fbd75944ff485dcdb0257a33867ef0fc78ae29cd7dfa5b5bdd0ecd18baed28b3` | 100 JSON |

## Offset Contract

`gt.zip` offsets match text after exactly one transformation:

```text
CRLF -> LF
```

They do not match the original decoded CRLF bytes, and NFC/NFKC normalization changes valid
boundaries. The importer must therefore:

1. preserve the original source artifact by hash;
2. create an immutable LF child document with `parent_document_id`;
3. apply annotations only to that child document;
4. enforce `child_text[start:end] == entity.text` for every row.

## Training And Decision Contract

The effective machine-readable policy is
`configs/models/phase1-training-governance-2026-07-30.yaml`.

- Final fit uses all 100 manually reviewed Round 1 records and all 100 records from this source.
- A local metric is diagnostic only and cannot promote or reject a model.
- Promotion and rejection require a recorded official submission score.
- Friend31 is teacher/reference evidence only and is prohibited from final runtime composition.
