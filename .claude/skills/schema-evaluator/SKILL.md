# Schema Evaluator

Use this skill for schema, JSON validation, metrics, and export-format work.

## Scope

- Maintain dataclass/enumeration schemas under `src/clingrounder/schema/`.
- Validate prediction JSONL with structured issues.
- Maintain span, linking, context, relation, and end-to-end metrics.
- Keep adapters converting external data into the internal schema.

## Guardrails

- Do not train models.
- Do not change entity or relation enum values without updating tests and docs.
- Preserve backwards-compatible JSON fields unless the task explicitly changes the output format.
- Run schema and evaluator tests before handoff.
