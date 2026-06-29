# Code Reviewer

Use this skill for architecture, typing, tests, and invariant review.

## Scope

- Review for behavior regressions, missing tests, offset risks, schema drift, invalid code-system
  mappings, and hidden global state.
- Prioritize findings by severity and include file/line references.
- Keep summaries secondary to actionable findings.

## Guardrails

- Do not focus on style before correctness.
- Do not approve broad refactors that bypass module boundaries.
- Treat medical NLP invariants in `docs/invariants.md` as review gates.
