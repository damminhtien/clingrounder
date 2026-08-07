# Benchmark results

The current checked-in fixture is a three-document synthetic pilot. It is intentionally too small
to support a clinical performance claim. The following snapshot was generated on 2026-08-07 from
commit `12ab64cac0a9e83792c19adab0f5c914f2dc4542` on macOS with Python 3.14:

| System | Entity exact F1 | Assertion macro-F1 | Recall@5 | Top-1 | Relation F1 | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact dictionary | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A | 17.97 |
| Lexical | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A | 15.52 |
| Hybrid | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A | 14.78 |
| Full deterministic | 1.0000 | 1.0000 | 1.0000 | 1.0000 | N/A | 16.12 |

Relation F1 is `N/A` because this fixture has zero gold relations. Runtime values are illustrative
for this machine and should be regenerated rather than used as a cross-machine claim. The
reproducibility command writes the authoritative `summary.json` and fingerprints.
