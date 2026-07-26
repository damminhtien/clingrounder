"""Parse arguments and lazily dispatch one consolidated CLI command."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import Any

from medical_kg_nlp.cli.parser import build_parser

__all__ = ["main"]

_HANDLERS = {
    "pipeline_run": ("medical_kg_nlp.cli.commands.pipeline", "run_pipeline"),
    "terminology_build": ("medical_kg_nlp.cli.commands.terminology", "build_index"),
    "terminology_query_set": (
        "medical_kg_nlp.cli.commands.terminology",
        "build_query_set",
    ),
    "terminology_inspect": ("medical_kg_nlp.cli.commands.terminology", "inspect_index"),
    "terminology_benchmark": (
        "medical_kg_nlp.cli.commands.terminology",
        "benchmark_index",
    ),
    "kg_build": ("medical_kg_nlp.cli.commands.kg", "build_graph_index"),
    "kg_benchmark_aliases": (
        "medical_kg_nlp.cli.commands.kg",
        "benchmark_graph_aliases_command",
    ),
    "kg_benchmark_relations": (
        "medical_kg_nlp.cli.commands.kg",
        "benchmark_graph_relations_command",
    ),
    "kg_benchmark_reranker": (
        "medical_kg_nlp.cli.commands.kg",
        "benchmark_graph_reranker_command",
    ),
    "kg_inspect": ("medical_kg_nlp.cli.commands.kg", "inspect_graph_index"),
    "evaluate": ("medical_kg_nlp.cli.commands.evaluate", "evaluate"),
    "validate": ("medical_kg_nlp.cli.commands.validate", "validate"),
    "benchmark_phase1_submission": (
        "medical_kg_nlp.cli.commands.phase1",
        "run_phase1_submission",
    ),
    "benchmark_phase1_round2_audit": (
        "medical_kg_nlp.cli.commands.phase1",
        "audit_phase1_round2",
    ),
    "benchmark_phase1_round2_probes": (
        "medical_kg_nlp.cli.commands.phase1",
        "run_phase1_round2_probe_suite",
    ),
    "benchmark_phase1_model_data_build": (
        "medical_kg_nlp.cli.commands.phase1",
        "build_phase1_model_data",
    ),
    "benchmark_phase1_model_data_calibrate": (
        "medical_kg_nlp.cli.commands.phase1",
        "calibrate_phase1_model_data",
    ),
    "benchmark_phase1_model_data_compare": (
        "medical_kg_nlp.cli.commands.phase1",
        "compare_phase1_model_variants",
    ),
    "model_validate_token_dataset": (
        "medical_kg_nlp.cli.commands.model",
        "validate_token_dataset",
    ),
    "model_train_token_classifier": (
        "medical_kg_nlp.cli.commands.model",
        "train_token_classifier",
    ),
    "model_inspect_token_classifier_run": (
        "medical_kg_nlp.cli.commands.model",
        "inspect_token_classifier_run",
    ),
    "model_train_token_classifier_run": (
        "medical_kg_nlp.cli.commands.model",
        "train_token_classifier_run",
    ),
    "data_registry_validate": ("medical_kg_nlp.cli.commands.data", "validate_registry"),
    "data_artifact_materialize": (
        "medical_kg_nlp.cli.commands.data",
        "materialize_artifact",
    ),
    "data_source_sync": ("medical_kg_nlp.cli.commands.data", "sync_registered_source"),
    "data_dataset_build": ("medical_kg_nlp.cli.commands.data", "build_dataset"),
    "data_dataset_inspect": ("medical_kg_nlp.cli.commands.data", "inspect_dataset"),
    "data_dataset_attach_block_evidence": (
        "medical_kg_nlp.cli.commands.data",
        "attach_dataset_block_evidence",
    ),
    "data_dataset_reconcile_duplicates": (
        "medical_kg_nlp.cli.commands.data",
        "reconcile_duplicates",
    ),
    "data_dataset_fuse": (
        "medical_kg_nlp.cli.commands.data",
        "fuse_datasets",
    ),
    "data_dataset_harmonize": (
        "medical_kg_nlp.cli.commands.data",
        "harmonize_dataset",
    ),
    "data_dataset_curate_annotations": (
        "medical_kg_nlp.cli.commands.data",
        "curate_annotation_dataset",
    ),
    "data_dataset_export_spans": (
        "medical_kg_nlp.cli.commands.data",
        "export_span_training_dataset",
    ),
    "data_lexicon_build": ("medical_kg_nlp.cli.commands.data", "build_lexicon"),
    "data_lexicon_crosswalk": (
        "medical_kg_nlp.cli.commands.data",
        "crosswalk_lexicon",
    ),
    "data_lexicon_propose_linked_aliases": (
        "medical_kg_nlp.cli.commands.data",
        "propose_linked_aliases",
    ),
    "data_lexicon_propose_dailymed_product_aliases": (
        "medical_kg_nlp.cli.commands.data",
        "propose_dailymed_product_aliases",
    ),
    "data_lexicon_attach_exact_links": (
        "medical_kg_nlp.cli.commands.data",
        "attach_exact_crosswalk_links",
    ),
    "data_mapping_compile_dailymed_rxnorm": (
        "medical_kg_nlp.cli.commands.data",
        "compile_dailymed_rxnorm",
    ),
    "data_mapping_compile_rxnorm_ndc": (
        "medical_kg_nlp.cli.commands.data",
        "compile_rxnorm_ndc",
    ),
    "data_mapping_link_dailymed_products": (
        "medical_kg_nlp.cli.commands.data",
        "link_dailymed_products",
    ),
    "data_mapping_audit_dailymed_rxnorm": (
        "medical_kg_nlp.cli.commands.data",
        "audit_dailymed_rxnorm",
    ),
    "data_ontology_compile_obo": (
        "medical_kg_nlp.cli.commands.data",
        "compile_obo_ontology",
    ),
    "data_ontology_compile_hpo_associations": (
        "medical_kg_nlp.cli.commands.data",
        "compile_hpo_association_knowledge",
    ),
    "data_knowledge_mine_abbreviations": (
        "medical_kg_nlp.cli.commands.data",
        "mine_abbreviation_knowledge",
    ),
    "data_knowledge_compile_aliases": (
        "medical_kg_nlp.cli.commands.data",
        "compile_alias_knowledge",
    ),
    "data_knowledge_compile_recognition": (
        "medical_kg_nlp.cli.commands.data",
        "compile_recognition_knowledge_artifact",
    ),
    "data_knowledge_compile_graph": (
        "medical_kg_nlp.cli.commands.data",
        "compile_graph_knowledge",
    ),
    "data_knowledge_benchmark_recognition": (
        "medical_kg_nlp.cli.commands.data",
        "benchmark_recognition_knowledge",
    ),
    "data_label_propose": ("medical_kg_nlp.cli.commands.data", "propose_labels"),
    "data_relation_propose": (
        "medical_kg_nlp.cli.commands.data",
        "propose_relations",
    ),
    "data_relation_mine_cooccurrence": (
        "medical_kg_nlp.cli.commands.data",
        "mine_cooccurrence",
    ),
    "data_review_export": ("medical_kg_nlp.cli.commands.data", "export_review"),
    "data_review_import": ("medical_kg_nlp.cli.commands.data", "import_review"),
    "data_review_quality": ("medical_kg_nlp.cli.commands.data", "review_quality"),
    "data_coverage_report": ("medical_kg_nlp.cli.commands.data", "report_coverage"),
    "data_snapshot_freeze": ("medical_kg_nlp.cli.commands.data", "freeze_snapshot"),
    "data_release_lock": (
        "medical_kg_nlp.cli.commands.data",
        "lock_mining_release",
    ),
    "data_release_verify": (
        "medical_kg_nlp.cli.commands.data",
        "verify_mining_release",
    ),
    "data_run": ("medical_kg_nlp.cli.commands.data", "run_plan"),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return a process exit status."""

    parser = build_parser()
    args = parser.parse_args(argv)
    handler_name = getattr(args, "handler", None)
    if not isinstance(handler_name, str):
        parser.print_help()
        return 2
    module_name, function_name = _HANDLERS[handler_name]
    module = importlib.import_module(module_name)
    handler = getattr(module, function_name)
    return int(_typed_handler(handler)(args))


def _typed_handler(value: object) -> Callable[[Any], int]:
    if not callable(value):
        raise TypeError("CLI handler is not callable")
    return value
