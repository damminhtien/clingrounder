# Round 2 Full-OOF Boundary Probe Rejection, 2026-07-31

## Official Result

The following deterministic ZIP is rejected as a submission artifact:

```text
ZIP SHA-256: 47d659cc2e0479a25835ed24aa36ee1d4aeb1733708dcf0baa9a458b2d653072
score: 25.3752
WER: 68.5056
J_assertion: 36.0658
J_candidates: 12.7677
```

The public-score reference remains unchanged:

```text
ZIP SHA-256: 1b375c092bb5affcbbd830661870bf39c736eed31d4c6dcb6eec2c522e7c558f
score: 33.0750
WER: 63.4134
J_assertion: 41.7462
J_candidates: 23.9379
```

## Scope Of The Rejection

This was not an isolated public boundary test against the reference artifact. The repository-owned
full-OOF proposal resolver generated 2,703 entities before and after boundary repair, while the
public reference has 3,385. The submitted ZIP therefore changed proposal selection, assertion
projection, and candidate hydration relative to the reference before the boundary stage ran.

The conservative overlay itself changed exactly one identity locally:

```text
document: 22
type: KẾT_QUẢ_XÉT_NGHIỆM
before: [2479, 2483) "tăng"
after:  [2479, 2497) "tăng bilirubin máu"
source support: 3
probability: 0.9431
margin over base: 0.0387
```

Two other variants were rejected because they had only one supporting source. The replacement
preserved raw offsets and did not copy candidate or assertion metadata from the old identity.

Consequently, the official result rejects the **full-OOF max-score composition**. It does not
establish that the one conservative boundary repair caused the regression, and it must not be
promoted as a standalone boundary result.

## Score Decomposition

Relative to the public reference, the composite artifact lost `7.6998` final points:

| Component | Metric regression | Weighted final loss |
| --- | ---: | ---: |
| Entity text/WER | `+5.0922` WER | `1.5277` |
| Assertion | `-5.6804` Jaccard | `1.7041` |
| Candidate | `-11.1702` Jaccard | `4.4681` |

Candidate loss accounts for about 58% of the score regression. The full-OOF resolver also changed
the selected entity population materially, so candidate and assertion changes are secondary
effects of identity drift rather than evidence for either metadata module in isolation.

## Locked Decision

- Do not submit `phase1-round2-boundary-conservative-probe.yaml` again.
- Do not use its full-OOF thresholds or output as a public baseline.
- Keep the repository-owned control artifact only for diagnostics; it has no official score and
  cannot isolate the boundary delta against the external public reference.
- A future boundary probe must start from a reproducible, officially scored base and change only
  a frozen subset of entity identities. Candidate and assertion projections must remain bytewise
  identical for every unmodified identity.
- Preserve `1b375c...` only as an external public-score reference, not as a runtime dependency or
  source for regenerated submissions.
