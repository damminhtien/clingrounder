"""Optional plugin descriptor for the archived Phase 1 benchmark."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass

from medical_kg_nlp.benchmarks.plugins import BenchmarkHandler

__all__ = ["PHASE1_PLUGIN", "Phase1BenchmarkPlugin"]

_COMMAND_MODULE = "medical_kg_nlp.benchmarks.phase1.commands"
_HANDLER_FUNCTIONS = {
    "benchmark_phase1_submission": "run_phase1_submission",
    "benchmark_phase1_round2_audit": "audit_phase1_round2",
    "benchmark_phase1_round2_probes": "run_phase1_round2_probe_suite",
    "benchmark_phase1_round2_proposal_verifier": (
        "run_phase1_round2_proposal_verifier_command"
    ),
    "benchmark_phase1_round2_max_score": "run_phase1_round2_max_score",
    "benchmark_phase1_round2_golden": "build_phase1_round2_golden_command",
    "benchmark_phase1_proposal_calibrate": "calibrate_phase1_proposals",
    "benchmark_phase1_proposal_matrix": "build_phase1_proposal_matrix_command",
    "benchmark_phase1_proposal_resolve": "resolve_phase1_proposals",
    "benchmark_phase1_boundary_calibrate": "calibrate_phase1_boundaries",
    "benchmark_phase1_boundary_resolve": "resolve_phase1_boundaries",
    "benchmark_phase1_proposal_score": "score_phase1_proposal_sources",
    "benchmark_phase1_type_verifier": "train_phase1_type_verifier",
    "benchmark_phase1_model_data_build": "build_phase1_model_data",
    "benchmark_phase1_model_data_build_final_fit": "build_phase1_final_token_data",
    "benchmark_phase1_model_data_build_final_fit_bundle": (
        "build_phase1_final_token_training_bundle_command"
    ),
    "benchmark_phase1_model_data_augment_regions": "augment_phase1_model_regions",
    "benchmark_phase1_model_data_augment_user_synthetic": (
        "augment_phase1_model_user_synthetic"
    ),
    "benchmark_phase1_model_data_calibrate": "calibrate_phase1_model_data",
    "benchmark_phase1_model_data_compare": "compare_phase1_model_variants",
    "benchmark_phase1_qwen_data_build": "build_phase1_qwen_data",
    "benchmark_phase1_qwen_inspect": "inspect_phase1_qwen_run",
    "benchmark_phase1_qwen_propose": "propose_phase1_qwen_entities",
    "benchmark_phase1_qwen_final_supervision_propose": (
        "propose_phase1_qwen_final_supervision_entities"
    ),
    "benchmark_phase1_qwen_token_bundle_propose": (
        "propose_phase1_qwen_token_bundle_entities"
    ),
    "benchmark_phase1_joint_span_prepare_final_fit": (
        "prepare_phase1_joint_span_final_fit_command"
    ),
    "benchmark_phase1_joint_span_prepare_token_bundle": (
        "prepare_phase1_joint_span_token_bundle_command"
    ),
    "benchmark_phase1_joint_span_materialize_token_source": (
        "materialize_phase1_joint_span_token_source_command"
    ),
    "benchmark_phase1_joint_span_materialize_token_bundle_source": (
        "materialize_phase1_joint_span_token_bundle_source_command"
    ),
    "benchmark_phase1_joint_span_run": "run_phase1_joint_span_command",
    "benchmark_phase1_joint_span_train": "train_phase1_joint_span_verifier_command",
    "benchmark_phase1_joint_span_train_oof": (
        "run_phase1_joint_span_transformer_oof_command"
    ),
    "benchmark_phase1_joint_span_calibrate": "calibrate_phase1_joint_span_command",
    "benchmark_phase1_qwen_vietnamese_support": "propose_phase1_vietnamese_support",
}


@dataclass(frozen=True, slots=True)
class Phase1BenchmarkPlugin:
    """Preserve reproducibility without making competition code part of the core API."""

    name: str = "phase1"
    summary: str = "Archived Vietnamese medical extraction challenge benchmark"

    def register_cli(
        self,
        parsers: argparse._SubParsersAction[argparse.ArgumentParser],
    ) -> None:
        from medical_kg_nlp.benchmarks.phase1.cli import register_phase1_cli

        register_phase1_cli(parsers)

    def handlers(self) -> Mapping[str, BenchmarkHandler]:
        return {
            name: BenchmarkHandler(module=_COMMAND_MODULE, function=function)
            for name, function in _HANDLER_FUNCTIONS.items()
        }


PHASE1_PLUGIN = Phase1BenchmarkPlugin()
