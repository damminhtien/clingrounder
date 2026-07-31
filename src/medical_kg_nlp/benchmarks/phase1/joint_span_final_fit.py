"""Reproducibly prepare final-fit joint span/type supervision from independent sources."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from medical_kg_nlp.benchmarks.phase1.final_supervision import (
    Phase1FinalSupervisionCorpus,
)
from medical_kg_nlp.benchmarks.phase1.joint_span_dataset import (
    build_phase1_joint_span_dataset,
    write_phase1_joint_span_dataset,
)
from medical_kg_nlp.benchmarks.phase1.joint_span_sources import (
    build_phase1_joint_span_proposal_matrix,
    build_phase1_rule_source_rows,
    load_phase1_joint_span_source_rows,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.mining.io import write_json
from medical_kg_nlp.utils.hashing import sha256_directory, sha256_file

__all__ = ["prepare_phase1_joint_span_final_fit"]


def prepare_phase1_joint_span_final_fit(
    corpus: Phase1FinalSupervisionCorpus,
    dictionary: DictionaryStore,
    *,
    model_sources: Mapping[str, tuple[Path, ProposalSourceRole | str]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build one hash-pinned final-fit verifier dataset from governed source evidence.

    RuleNER is always present as a deterministic lexical source. ``model_sources`` must provide
    at least one independently materialized source, such as Qwen exact quotes or an XLM-R
    projection. This function cannot read Round 2, friend outputs, or raw model checkpoints.
    """

    if not model_sources:
        raise ValueError("Joint span final fit requires at least one model proposal source")
    if "rule" in model_sources:
        raise ValueError("The rule source is constructed internally and cannot be overridden")
    if any(not name.strip() for name in model_sources):
        raise ValueError("Joint span source names must be non-empty")

    source_roles: dict[str, ProposalSourceRole] = {"rule": ProposalSourceRole.RULE}
    source_rows = {
        "rule": build_phase1_rule_source_rows(corpus.reviewed, dictionary),
    }
    source_descriptors: dict[str, dict[str, str]] = {}
    for name, (path, role) in sorted(model_sources.items()):
        source_path = Path(path)
        if not source_path.is_dir():
            raise FileNotFoundError(source_path)
        parsed_role = ProposalSourceRole(role)
        if parsed_role is ProposalSourceRole.VERIFIER:
            raise ValueError("Verifier-only evidence cannot be a final-fit lattice source")
        source_roles[name] = parsed_role
        source_rows[name] = load_phase1_joint_span_source_rows(source_path, corpus.reviewed)
        # INVARIANT: source contents, not directory names, identify the supervised lattice.
        source_descriptors[name] = {
            "path": str(source_path),
            "sha256": sha256_directory(source_path),
            "role": parsed_role.value,
        }

    matrix = build_phase1_joint_span_proposal_matrix(
        corpus.reviewed,
        source_rows,
        source_roles=source_roles,
    )
    dataset = build_phase1_joint_span_dataset(
        corpus.reviewed,
        matrix,
        source_roles=source_roles,
        source_dataset_by_document=corpus.source_by_document,
        dictionary_matcher=None,
    )
    output = Path(output_dir)
    paths = write_phase1_joint_span_dataset(dataset, output)
    preparation_manifest = {
        "schema_version": "phase1-joint-span-final-fit.v1",
        "purpose": "final_fit_all_authorized_supervision",
        "promotion": "official_submission_metrics_only",
        "final_supervision": corpus.manifest,
        "sources": {
            "rule": {
                "role": ProposalSourceRole.RULE.value,
                "dictionary_entry_count": len(dictionary.entries),
            },
            **source_descriptors,
        },
        "dataset": {
            **paths,
            "manifest_sha256": sha256_file(output / "manifest.json"),
        },
        "policy": {
            "round2_included": False,
            "friend31_included": False,
            "selection": "deferred_to_joint_span_verifier",
            "raw_offsets": "immutable",
        },
    }
    write_json(output / "preparation_manifest.json", preparation_manifest)
    return preparation_manifest
