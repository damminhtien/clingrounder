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
    "terminology_inspect": ("medical_kg_nlp.cli.commands.terminology", "inspect_index"),
    "evaluate": ("medical_kg_nlp.cli.commands.evaluate", "evaluate"),
    "validate": ("medical_kg_nlp.cli.commands.validate", "validate"),
    "benchmark_phase1": ("medical_kg_nlp.cli.commands.phase1", "run_phase1"),
    "data_registry_validate": ("medical_kg_nlp.cli.commands.data", "validate_registry"),
    "data_source_sync": ("medical_kg_nlp.cli.commands.data", "sync_registered_source"),
    "data_dataset_build": ("medical_kg_nlp.cli.commands.data", "build_dataset"),
    "data_label_propose": ("medical_kg_nlp.cli.commands.data", "propose_labels"),
    "data_review_export": ("medical_kg_nlp.cli.commands.data", "export_review"),
    "data_review_import": ("medical_kg_nlp.cli.commands.data", "import_review"),
    "data_coverage_report": ("medical_kg_nlp.cli.commands.data", "report_coverage"),
    "data_snapshot_freeze": ("medical_kg_nlp.cli.commands.data", "freeze_snapshot"),
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
