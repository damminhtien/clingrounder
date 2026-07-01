from __future__ import annotations

from dataclasses import dataclass


CORE_SCORE_METRICS = (
    "span_exact_f1",
    "linking_accuracy_at_1",
    "context_macro_f1",
    "relation_f1",
)


@dataclass(frozen=True)
class ErrorPolicy:
    module: str
    impact: float
    fixability: float
    cost: float
    recommendation: str
    success_metric: str


@dataclass(frozen=True)
class AgentPlaybook:
    focus_files: tuple[str, ...]
    commands: tuple[str, ...]
    guardrails: tuple[str, ...]


AGENT_PLAYBOOKS: dict[str, AgentPlaybook] = {
    "schema": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/schema/validator.py",
            "src/medical_kg_nlp/schema/annotation.py",
            "tests/test_prediction_validator.py",
        ),
        commands=("uv run pytest tests/test_prediction_validator.py -q", "uv run mypy src"),
        guardrails=(
            "Keep exported prediction fields backward-compatible.",
            "Schema or enum changes require focused tests and docs.",
        ),
    ),
    "preprocessing": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/preprocessing/offset_mapping.py",
            "src/medical_kg_nlp/preprocessing/sentence_splitter.py",
            "tests/test_offset_mapping.py",
        ),
        commands=(
            "uv run pytest tests/test_offset_mapping.py -q",
            "uv run pytest tests/test_pipeline_tracing.py -q",
        ),
        guardrails=(
            "Never destroy or rewrite original character offsets.",
            "Normalized text is lookup-only unless an explicit offset map is used.",
        ),
    ),
    "entity_extraction": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/ner/rule_ner.py",
            "src/medical_kg_nlp/dictionaries/dictionary_store.py",
            "tests/test_pipeline_smoke.py",
            "tests/test_dictionary.py",
        ),
        commands=(
            "uv run pytest tests/test_pipeline_smoke.py tests/test_dictionary.py -q",
            "uv run pytest tests/test_offset_mapping.py -q",
        ),
        guardrails=(
            "Every emitted span must validate against the original source text.",
            "Avoid broad NER refactors unless the top error cases prove they are needed.",
        ),
    ),
    "candidate_generation": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/retrieval/candidate_generator.py",
            "src/medical_kg_nlp/dictionaries/dictionary_store.py",
            "src/medical_kg_nlp/linking/linker.py",
            "tests/test_candidate_generation.py",
        ),
        commands=(
            "uv run pytest tests/test_candidate_generation.py tests/test_dictionary.py -q",
            "uv run pytest tests/test_prediction_validator.py -q",
        ),
        guardrails=(
            "Candidate generation must filter by entity type before final linking.",
            "Never emit candidate codes outside the loaded dictionary.",
        ),
    ),
    "normalization": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/linking/linker.py",
            "src/medical_kg_nlp/retrieval/candidate_generator.py",
            "src/medical_kg_nlp/dictionaries/dictionary_store.py",
            "tests/test_candidate_generation.py",
        ),
        commands=(
            "uv run pytest tests/test_candidate_generation.py tests/test_dictionary.py -q",
            "uv run pytest tests/test_prediction_validator.py -q",
        ),
        guardrails=(
            "Never map DRUG entities to ICD-10 or DISEASE entities to RxNorm.",
            "Candidate recall@20 must be fixed before reranker tuning can help.",
        ),
    ),
    "context": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/context/rules.py",
            "src/medical_kg_nlp/context/assertion.py",
            "tests/test_context_rules.py",
            "tests/test_context_metrics.py",
        ),
        commands=(
            "uv run pytest tests/test_context_rules.py tests/test_context_metrics.py -q",
            "uv run pytest tests/test_pipeline_smoke.py -q",
        ),
        guardrails=(
            "Negated diseases must not become confirmed patient conditions.",
            "Family-history diseases must not become patient-present diseases.",
        ),
    ),
    "relation_extraction": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/relations/rule_relations.py",
            "src/medical_kg_nlp/relations/candidate_pairs.py",
            "src/medical_kg_nlp/kg/constraints.py",
            "tests/test_kg_constraints.py",
        ),
        commands=("uv run pytest tests/test_kg_constraints.py -q", "uv run pytest tests/test_pipeline_smoke.py -q"),
        guardrails=(
            "Relation endpoint types must satisfy KG constraints.",
            "Do not keep impossible relations for score-chasing.",
        ),
    ),
    "kg_validation": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/kg/constraints.py",
            "src/medical_kg_nlp/kg/validator.py",
            "tests/test_kg_constraints.py",
            "tests/test_prediction_validator.py",
        ),
        commands=(
            "uv run pytest tests/test_kg_constraints.py tests/test_prediction_validator.py -q",
            "uv run mypy src",
        ),
        guardrails=(
            "Ontology/KG violations are blocking, not cosmetic.",
            "Invalid code systems should be rejected or reset before export.",
        ),
    ),
    "evaluation": AgentPlaybook(
        focus_files=(
            "src/medical_kg_nlp/evaluation/pipeline_report.py",
            "src/medical_kg_nlp/evaluation/loop_engineer.py",
            "src/medical_kg_nlp/evaluation/loop_analysis.py",
            "src/medical_kg_nlp/evaluation/loop_artifacts.py",
            "src/medical_kg_nlp/evaluation/loop_agent.py",
            "src/medical_kg_nlp/evaluation/loop_journal.py",
            "src/medical_kg_nlp/evaluation/loop_policy.py",
            "tests/test_pipeline_report.py",
            "tests/test_loop_engineer.py",
        ),
        commands=(
            "uv run pytest tests/test_pipeline_report.py tests/test_loop_engineer.py -q",
            "uv run ruff check .",
        ),
        guardrails=(
            "Do not hide validation failures behind aggregate metrics.",
            "Keep reports machine-readable and stable for agents.",
        ),
    ),
}


DEFAULT_AGENT_PLAYBOOK = AgentPlaybook(
    focus_files=("docs/evaluation.md", "src/medical_kg_nlp/evaluation/pipeline_report.py"),
    commands=("uv run pytest tests -q",),
    guardrails=(
        "Make one meaningful change per experiment.",
        "Add focused tests before relying on a metric improvement.",
    ),
)


ERROR_POLICIES: dict[str, ErrorPolicy] = {
    "schema": ErrorPolicy("schema", 1.0, 0.95, 1.0, "Fix schema parsing/export before comparing model metrics.", "validation_issue_count"),
    "offset": ErrorPolicy("preprocessing", 1.0, 0.9, 1.0, "Fix offset preservation or span trimming regression.", "validation_issue_count"),
    "invalid_code_system": ErrorPolicy("kg_validation", 1.0, 0.9, 1.0, "Tighten entity type to code-system constraints.", "validation_issue_count"),
    "unknown_dictionary_code": ErrorPolicy("normalization", 1.0, 0.85, 1.0, "Force linked codes and candidates to come from the loaded dictionary.", "validation_issue_count"),
    "invalid_candidate_code_system": ErrorPolicy("candidate_generation", 0.95, 0.9, 1.0, "Filter candidates by entity type before ranking.", "candidate_missing_gold"),
    "candidate_missing_gold": ErrorPolicy("normalization", 0.95, 0.75, 1.4, "Improve dictionary aliases or retrieval sources so gold codes enter top-k.", "linking_recall_at_20"),
    "candidate_empty": ErrorPolicy("normalization", 0.95, 0.8, 1.2, "Add exact, abbreviation, fuzzy, or n-gram coverage for empty candidate lists.", "linking_recall_at_20"),
    "linking_wrong_top1": ErrorPolicy("normalization", 0.9, 0.65, 1.6, "Tune reranking, context features, or score blending after candidate recall is healthy.", "linking_accuracy_at_1"),
    "linking_unlinked": ErrorPolicy("normalization", 0.9, 0.75, 1.2, "Inspect assignment thresholds and dictionary coverage for unlinked gold entities.", "linking_accuracy_at_1"),
    "severe_context_error": ErrorPolicy("context", 0.95, 0.8, 1.2, "Prioritize negation, family-history, and uncertainty rules before larger models.", "context_macro_f1"),
    "context_confusion": ErrorPolicy("context", 0.8, 0.75, 1.3, "Add cue-specific regression cases and section/sentence scoped rules.", "context_macro_f1"),
    "span_boundary": ErrorPolicy("entity_extraction", 0.9, 0.65, 1.4, "Tune tokenizer, dictionary span selection, or postprocess merge/split rules.", "span_exact_f1"),
    "missing_entity": ErrorPolicy("entity_extraction", 0.9, 0.7, 1.3, "Increase span recall with aliases, abbreviations, or recall-oriented NER rules.", "span_exact.recall"),
    "spurious_entity": ErrorPolicy("entity_extraction", 0.8, 0.75, 1.2, "Add blocklist, section prior, or confidence threshold for false positive mentions.", "span_exact.precision"),
    "type_confusion": ErrorPolicy("entity_extraction", 0.85, 0.75, 1.2, "Add type-specific aliases or postprocess type disambiguation.", "span_exact_f1"),
    "invalid_relation": ErrorPolicy("kg_validation", 0.9, 0.9, 1.0, "Apply relation endpoint/type constraints before export.", "validation_issue_count"),
    "missing_relation": ErrorPolicy("relation_extraction", 0.75, 0.65, 1.5, "Increase candidate pair recall or add relation rules for frequent missing types.", "relation_f1"),
    "spurious_relation": ErrorPolicy("relation_extraction", 0.75, 0.75, 1.2, "Tighten relation constraints, direction checks, or distance thresholds.", "relation_f1"),
    "relation_type_confusion": ErrorPolicy("relation_extraction", 0.75, 0.65, 1.4, "Add type-specific relation features and endpoint constraints.", "relation_f1"),
}


DEFAULT_ERROR_POLICY = ErrorPolicy(
    module="pipeline",
    impact=0.6,
    fixability=0.6,
    cost=1.5,
    recommendation="Inspect examples and add the smallest targeted regression test.",
    success_metric="loop_score",
)
