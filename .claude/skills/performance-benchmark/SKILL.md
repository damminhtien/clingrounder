# Performance Benchmark

Use this skill for profiling, benchmarking, and deciding whether native extensions are justified.

## Scope

- Profile runtime and memory.
- Identify candidate generation, phrase matching, fuzzy search, offset mapping, and merge bottlenecks.
- Recommend Rust/C++ only after measured evidence.
- Keep benchmark commands reproducible.

## Guardrails

- Do not optimize before correctness tests pass.
- Do not introduce native extensions without a Python fallback or clear build instructions.
- Keep raw benchmark logs when exact output matters.
