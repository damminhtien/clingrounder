# VietBioNER

## Source And Acquisition

VietBioNER is pinned to Git commit `19ba70a5947d1be72906d407c860b1666b9337e9` under CC BY 4.0.
`configs/mining/vietbioner.yaml` downloads the immutable archive and the BRAT parser preserves each
annotator copy as a separate document. The corpus is Vietnamese biomedical literature, not clinical
notes.

## Parsing And Human Labels

| Measure | Value |
| --- | ---: |
| parsed annotator documents | 70 |
| unique texts after exact reconciliation | 63 |
| source-human annotations | 3,574 |
| discontinuous annotations | 248 |
| offset issues | 0 |

Source labels are `Symptom_and_Disease`, `DiagnosticProcedure`, `DateTime`, `Location`, and
`Organisation`. The import maps `Symptom_and_Disease` to broad internal `FINDING`; it does not guess
whether a mention is a symptom or diagnosis.

Seven exact-text document pairs came from independent annotators. Their exact annotation
micro-Jaccard is 0.647. Reconciliation keeps 3,109 consensus/training annotations and routes 164
disagreement hypotheses to review instead of taking an unsafe union.

## Extracted Knowledge

- Mention inventory: 768 entries, including 137 multi-document and 115 duplicate-consensus-backed
  entries.
- TT06/RxNorm exact crosswalk: 8 unique exact entries covering 22 occurrences; all remain
  `review_required` because the source's broad finding label cannot prove internal type.
- Train-only recognition compiler: 100 code-free concepts.
- Held-out result on 12 source documents: precision 0.610, recall 0.613, F1 0.612, with 161 false
  positives; 111 false positives are overlapping boundary errors.

## Promotion Boundary

VietBioNER is training/review knowledge only. Its 100 recognition concepts are not enabled in the
runtime recognizer. It is useful for Vietnamese biomedical vocabulary and cross-lingual NER, but it
does not establish clinical-note distribution, assertion labels, or diagnosis-vs-symptom semantics.

Primary artifacts are under `outputs/mining/vietbioner-19ba70a/`; the reconciled files are under its
`reconciled/` directory. Reproduction commands remain in `docs/data-mining.md`.
