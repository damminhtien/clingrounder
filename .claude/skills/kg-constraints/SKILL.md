# KG Constraints

Use this skill for ontology compatibility, relation constraints, and graph-backed validation.

## Scope

- Maintain entity type to code-system compatibility.
- Maintain relation type compatibility.
- Add KG evidence as a reranker feature, not an unconstrained generator.
- Keep graph storage lightweight until traversal requirements are proven.

## Guardrails

- Invalid code-system outputs must be cleared or rejected.
- Invalid relation endpoints must be rejected.
- KG validation must not invent codes or relations absent from upstream candidates.
