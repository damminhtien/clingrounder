# Experiment Runner

Use this skill for reproducible runs, configs, metrics collection, and ablation reports.

## Scope

- Run pipeline commands with explicit config/input/output paths.
- Save metrics and error analysis artifacts.
- Compare retrieval, context, relation, and KG ablations.
- Keep reports reproducible from checked-in commands and configs.

## Guardrails

- Do not mix benchmark output with ad hoc logs.
- Validate predictions before scoring.
- Record known data limitations and dictionary coverage limits.
