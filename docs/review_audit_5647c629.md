# Review Audit: 5647c629

This audit maps the static review of commit `5647c629` to the remediation now committed on top of
that snapshot. It distinguishes correctness fixes from statistical limitations that code alone
cannot remove.

## Finding Status

| # | Review finding | Status | Current evidence |
| ---: | --- | --- | --- |
| 1 | Full-terminology candidates could not pass the linking gate | Fixed in the retrieval contract | SQLite FTS returns bounded lexical scores, fusion preserves the primary score, and linking stores retrieval score separately from calibrated emit probability. Approximate sources remain opt-in until their query-set calibration passes. |
| 2 | NER-pinned codes could survive linker abstention | Fixed | Rule NER emits unassigned spans; every linking pass clears prior code state; export accepts only qualified, type-compatible candidates. |
| 3 | The full profile omitted full ICD-10 VN | Fixed | `configs/phase1_full.yaml` loads full TT06 ICD-10 VN and RxNorm July 6, 2026 as normalization sources. Release tests query both repositories directly. |
| 4 | Calibrated config paths were not portable | Fixed | `ResolvedPipelineConfig` resolves paths relative to the profile file, and the calibrated artifact is reloaded from another directory in tests. |
| 5 | Hybrid NER made dictionary output authoritative | Fixed for arbitration | Exact agreement merges evidence and overlap conflicts use weighted interval scheduling. Stronger model proposals can replace dictionary spans. The small source priors are deterministic defaults, not learned probabilities. |
| 6 | Reviewed memory was default, weakly keyed, and terminal | Fixed | Reusable profiles default to no memory. Opt-in rows require mention plus entity type, reviewed status, source hashes, compatible terminology releases, and matching medication structure. Runtime source naming is stable. |
| 7 | Selective export existed but was unreachable from the stable CLI | Resolved by isolation | Selective policies and legacy configs are explicitly benchmark experiments. The stable CLI exposes only supported `empty` and `pipeline` modes. |
| 8 | CI skipped release-critical contracts | Fixed | Pull requests run fast checks plus Python 3.11 release/integration contracts. Nightly remains the broader lane. |
| 9 | `make phase1-submit` used the old command hierarchy | Fixed | The Make target uses `benchmark phase1 submission`, covered by a developer-surface test. |
| 10 | Calibration used 16 documents and hard-coded gate numbers | Partly fixed; sample-size risk remains | The gate loads a fingerprinted baseline artifact. Threshold selection uses official Phase 1 score and reports support, duplicate-group repeated CV, and deterministic bootstrap intervals. A new blind corpus is still required. |
| 11 | Normalization stage implied downstream behavior it did not provide | Fixed by choosing the diagnostic design | The trace stage is named `lookup_normalization_diagnostics`; counters state that downstream spans remain in source coordinates and normalized text is not passed downstream. |
| 12 | Cross-type aliases were silently dropped | Fixed without guessing | Rule NER retains `AmbiguousEntityProposal` records. Rule-only output abstains; hybrid NER may use exact-span proposal evidence only when an independent model selects a supported type. |

## Terminology Test Boundary

The organizer's medication-list example is an executable benchmark fixture only. It verifies
offset, full medication span, assertion, and expected RxCUI convention, but it is not loaded by
default profiles and does not establish current terminology coverage.

Current release verification is separate:

- TT06 ICD-10 VN manifest and direct typed code/query checks.
- RxNorm full July 6, 2026 manifest and direct typed query checks, including a July-only concept.
- Query-set ranking/abstention metrics for full terminology retrieval.
- Explicit reporting of unknown codes and ambiguous exact aliases.

## Residual Risks

- Hybrid source priors need held-out model/rule disagreement data before they can be called
  calibrated.
- The 16-document development split produces useful diagnostics, not a reliable estimate of
  generalization. A source-held-out blind set remains necessary.
- Full-store approximate retrieval should be promoted source by source only after Recall@k, MRR,
  abstention, latency, and memory gates pass.
- The mapped normalization text remains diagnostic. Any future normalized-text NER path must map
  every span back to raw offsets before it can replace the source-text path.

## Verification

On 25 July 2026:

- Ruff: passed.
- mypy: passed across 270 source files.
- Fast pytest suite: `677 passed, 20 deselected`.
- Release/integration suite: `17 passed` in the sandbox-compatible lanes.
- Process-pool integration: passed outside the sandbox; the sandbox itself blocks the macOS
  semaphore capability check.
- Latest remote CI and Nightly runs on `ec9a9c5` were successful. Commits after that SHA were
  verified locally and had not yet been pushed when this audit was written.
