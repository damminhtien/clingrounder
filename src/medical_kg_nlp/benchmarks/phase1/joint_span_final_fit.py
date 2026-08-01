"""Reproducibly prepare final-fit joint span/type supervision from independent sources."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from medical_kg_nlp.benchmarks.phase1.final_supervision import (
    Phase1FinalSupervisionCorpus,
)
from medical_kg_nlp.benchmarks.phase1.joint_span_preparation import (
    prepare_phase1_joint_span_supervision,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore

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

    return prepare_phase1_joint_span_supervision(
        corpus.reviewed,
        dictionary,
        source_dataset_by_document=corpus.source_by_document,
        oof_group_by_document={
            document_id: f"phase1-origin:{document_id}"
            for document_id in corpus.reviewed.source_texts
        },
        supervision_manifest=corpus.manifest,
        model_sources=model_sources,
        output_dir=output_dir,
        purpose="final_fit_all_authorized_supervision",
        require_model_source=True,
    )
