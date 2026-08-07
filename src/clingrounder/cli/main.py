"""Parse arguments and lazily dispatch one consolidated CLI command."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
import sys
from typing import Any

from clingrounder.cli.parser import CliScope, build_parser

__all__ = ["benchmark_main", "main", "operational_main", "research_main"]

_HANDLERS = {
    "pipeline_run": ("clingrounder.cli.commands.pipeline", "run_pipeline"),
    "pipeline_inspect_config": (
        "clingrounder.cli.commands.pipeline",
        "inspect_pipeline_config",
    ),
    "pipeline_list_profiles": (
        "clingrounder.cli.commands.pipeline",
        "list_pipeline_profiles",
    ),
    "terminology_build": ("clingrounder.cli.commands.terminology", "build_index"),
    "terminology_query_set": (
        "clingrounder.cli.commands.terminology",
        "build_query_set",
    ),
    "terminology_inspect": ("clingrounder.cli.commands.terminology", "inspect_index"),
    "terminology_benchmark": (
        "clingrounder.cli.commands.terminology",
        "benchmark_index",
    ),
    "kg_build": ("clingrounder.cli.commands.kg", "build_graph_index"),
    "kg_benchmark_aliases": (
        "clingrounder.cli.commands.kg",
        "benchmark_graph_aliases_command",
    ),
    "kg_benchmark_relations": (
        "clingrounder.cli.commands.kg",
        "benchmark_graph_relations_command",
    ),
    "kg_benchmark_reranker": (
        "clingrounder.cli.commands.kg",
        "benchmark_graph_reranker_command",
    ),
    "kg_inspect": ("clingrounder.cli.commands.kg", "inspect_graph_index"),
    "evaluate": ("clingrounder.cli.commands.evaluate", "evaluate"),
    "validate": ("clingrounder.cli.commands.validate", "validate"),
    "benchmark_list": (
        "clingrounder.cli.commands.benchmark",
        "list_benchmarks",
    ),
    "benchmark_runtime_run": (
        "clingrounder.cli.commands.benchmark",
        "run_runtime_benchmark",
    ),
    "benchmark_compare": (
        "clingrounder.cli.commands.benchmark",
        "compare_runtime_benchmark",
    ),
    "release_audit": (
        "clingrounder.cli.commands.release",
        "audit_release",
    ),
    "release_inventory": (
        "clingrounder.cli.commands.release",
        "inventory_local_artifacts",
    ),
    "model_validate_token_dataset": (
        "clingrounder.cli.commands.model",
        "validate_token_dataset",
    ),
    "model_inspect_inference_budget": (
        "clingrounder.cli.commands.model",
        "inspect_inference_budget",
    ),
    "model_build_dapt_corpus": (
        "clingrounder.cli.commands.model",
        "build_dapt_corpus_run",
    ),
    "model_inspect_xlmr_dapt_run": (
        "clingrounder.cli.commands.model",
        "inspect_xlmr_dapt_run",
    ),
    "model_train_xlmr_dapt_run": (
        "clingrounder.cli.commands.model",
        "train_xlmr_dapt_run",
    ),
    "model_train_token_classifier": (
        "clingrounder.cli.commands.model",
        "train_token_classifier",
    ),
    "model_inspect_token_classifier_run": (
        "clingrounder.cli.commands.model",
        "inspect_token_classifier_run",
    ),
    "model_train_token_classifier_run": (
        "clingrounder.cli.commands.model",
        "train_token_classifier_run",
    ),
    "model_inspect_causal_qlora_run": (
        "clingrounder.cli.commands.model",
        "inspect_causal_qlora_run",
    ),
    "model_finalize_causal_qlora_run": (
        "clingrounder.cli.commands.model",
        "finalize_causal_qlora_run",
    ),
    "model_train_causal_qlora_run": (
        "clingrounder.cli.commands.model",
        "train_causal_qlora_run",
    ),
    "data_registry_validate": ("clingrounder.cli.commands.data", "validate_registry"),
    "data_artifact_materialize": (
        "clingrounder.cli.commands.data",
        "materialize_artifact",
    ),
    "data_source_sync": ("clingrounder.cli.commands.data", "sync_registered_source"),
    "data_dataset_build": ("clingrounder.cli.commands.data", "build_dataset"),
    "data_dataset_inspect": ("clingrounder.cli.commands.data", "inspect_dataset"),
    "data_dataset_attach_block_evidence": (
        "clingrounder.cli.commands.data",
        "attach_dataset_block_evidence",
    ),
    "data_dataset_reconcile_duplicates": (
        "clingrounder.cli.commands.data",
        "reconcile_duplicates",
    ),
    "data_dataset_fuse": (
        "clingrounder.cli.commands.data",
        "fuse_datasets",
    ),
    "data_dataset_harmonize": (
        "clingrounder.cli.commands.data",
        "harmonize_dataset",
    ),
    "data_dataset_curate_annotations": (
        "clingrounder.cli.commands.data",
        "curate_annotation_dataset",
    ),
    "data_dataset_export_spans": (
        "clingrounder.cli.commands.data",
        "export_span_training_dataset",
    ),
    "data_dataset_build_exact_quote_curriculum": (
        "clingrounder.cli.commands.data",
        "build_exact_quote_curriculum_dataset",
    ),
    "data_dataset_freeze_source_splits": (
        "clingrounder.cli.commands.data",
        "freeze_dataset_source_splits",
    ),
    "data_lexicon_build": ("clingrounder.cli.commands.data", "build_lexicon"),
    "data_lexicon_crosswalk": (
        "clingrounder.cli.commands.data",
        "crosswalk_lexicon",
    ),
    "data_lexicon_propose_linked_aliases": (
        "clingrounder.cli.commands.data",
        "propose_linked_aliases",
    ),
    "data_lexicon_propose_dailymed_product_aliases": (
        "clingrounder.cli.commands.data",
        "propose_dailymed_product_aliases",
    ),
    "data_lexicon_attach_exact_links": (
        "clingrounder.cli.commands.data",
        "attach_exact_crosswalk_links",
    ),
    "data_mapping_compile_dailymed_rxnorm": (
        "clingrounder.cli.commands.data",
        "compile_dailymed_rxnorm",
    ),
    "data_mapping_compile_rxnorm_ndc": (
        "clingrounder.cli.commands.data",
        "compile_rxnorm_ndc",
    ),
    "data_mapping_link_dailymed_products": (
        "clingrounder.cli.commands.data",
        "link_dailymed_products",
    ),
    "data_mapping_audit_dailymed_rxnorm": (
        "clingrounder.cli.commands.data",
        "audit_dailymed_rxnorm",
    ),
    "data_ontology_compile_obo": (
        "clingrounder.cli.commands.data",
        "compile_obo_ontology",
    ),
    "data_ontology_compile_hpo_associations": (
        "clingrounder.cli.commands.data",
        "compile_hpo_association_knowledge",
    ),
    "data_knowledge_mine_abbreviations": (
        "clingrounder.cli.commands.data",
        "mine_abbreviation_knowledge",
    ),
    "data_knowledge_compile_aliases": (
        "clingrounder.cli.commands.data",
        "compile_alias_knowledge",
    ),
    "data_knowledge_compile_recognition": (
        "clingrounder.cli.commands.data",
        "compile_recognition_knowledge_artifact",
    ),
    "data_knowledge_compile_graph": (
        "clingrounder.cli.commands.data",
        "compile_graph_knowledge",
    ),
    "data_knowledge_benchmark_recognition": (
        "clingrounder.cli.commands.data",
        "benchmark_recognition_knowledge",
    ),
    "data_label_propose": ("clingrounder.cli.commands.data", "propose_labels"),
    "data_relation_propose": (
        "clingrounder.cli.commands.data",
        "propose_relations",
    ),
    "data_relation_mine_cooccurrence": (
        "clingrounder.cli.commands.data",
        "mine_cooccurrence",
    ),
    "data_review_export": ("clingrounder.cli.commands.data", "export_review"),
    "data_review_import": ("clingrounder.cli.commands.data", "import_review"),
    "data_review_quality": ("clingrounder.cli.commands.data", "review_quality"),
    "data_coverage_report": ("clingrounder.cli.commands.data", "report_coverage"),
    "data_snapshot_freeze": ("clingrounder.cli.commands.data", "freeze_snapshot"),
    "data_release_lock": (
        "clingrounder.cli.commands.data",
        "lock_mining_release",
    ),
    "data_release_verify": (
        "clingrounder.cli.commands.data",
        "verify_mining_release",
    ),
    "data_run": ("clingrounder.cli.commands.data", "run_plan"),
}


def main(
    argv: Sequence[str] | None = None,
    *,
    scope: CliScope | None = None,
    prog: str = "clingrounder",
) -> int:
    """Run one command and return a process exit status."""

    parser = build_parser(scope, prog=prog)
    args = parser.parse_args(argv)
    handler_name = getattr(args, "handler", None)
    if not isinstance(handler_name, str):
        parser.print_help()
        return 2
    target = _HANDLERS.get(handler_name)
    if target is None:
        from clingrounder.benchmarks.registry import resolve_benchmark_handler

        benchmark_target = resolve_benchmark_handler(handler_name)
        if benchmark_target is None:
            raise KeyError(f"Unknown CLI handler: {handler_name}")
        module_name, function_name = benchmark_target.module, benchmark_target.function
    else:
        module_name, function_name = target
    module = importlib.import_module(module_name)
    handler = getattr(module, function_name)
    return int(_typed_handler(handler)(args))


def operational_main() -> int:
    """Run the stable operational command set installed as ``clingrounder``."""

    return main(scope="operational", prog="clingrounder")


def research_main() -> int:
    """Run data-mining and local-model research commands."""

    return main(scope="research", prog="clingrounder-research")


def benchmark_main() -> int:
    """Run optional benchmark plugins without loading them in operational CLI startup."""

    # The shared benchmark parser retains the explicit ``benchmark`` namespace when used from
    # Python. The installed entrypoint hides that implementation detail for a short command.
    return main(
        ["benchmark", *sys.argv[1:]],
        scope="benchmark",
        prog="clingrounder-benchmark",
    )


def _typed_handler(value: object) -> Callable[[Any], int]:
    if not callable(value):
        raise TypeError("CLI handler is not callable")
    return value
