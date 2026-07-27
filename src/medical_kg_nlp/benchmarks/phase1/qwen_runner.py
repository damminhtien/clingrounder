"""Reproducible Qwen proposal inference over authorized Phase 1 documents."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.adapters.generative import TransformersCausalLMRuntime
from medical_kg_nlp.benchmarks.phase1.phase1_ensemble import (
    load_phase1_output_source,
)
from medical_kg_nlp.benchmarks.phase1.qwen_proposals import (
    Phase1AdjudicationCandidate,
    Phase1QwenAdapter,
    apply_phase1_adjudication,
    select_qwen_confirmed_proposals,
)
from medical_kg_nlp.benchmarks.phase1.qwen_run_spec import Phase1QwenRunSpec
from medical_kg_nlp.benchmarks.phase1.round2 import load_phase1_round2_documents
from medical_kg_nlp.mining.io import load_documents, write_json, write_jsonl
from medical_kg_nlp.ner.proposal import EntityProposal
from medical_kg_nlp.ner.span_resolver import EvidenceWeightedSpanResolver
from medical_kg_nlp.ontology.phase1 import PHASE1_ALLOWED_TYPES
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.schema.types import EntityType
from medical_kg_nlp.utils.hashing import sha256_file

__all__ = ["Phase1QwenProposalRunConfig", "run_phase1_qwen_proposals"]

_PHASE1_LABEL_TO_TYPE = {
    "TRIỆU_CHỨNG": EntityType.SYMPTOM,
    "TÊN_XÉT_NGHIỆM": EntityType.LAB_TEST,
    "KẾT_QUẢ_XÉT_NGHIỆM": EntityType.LAB_RESULT,
    "CHẨN_ĐOÁN": EntityType.DISEASE,
    "THUỐC": EntityType.DRUG,
}
_ENTITY_TYPE_TO_PHASE1_LABEL = {
    value: key for key, value in _PHASE1_LABEL_TO_TYPE.items()
}
_TARGETED_PASSES = (
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
)


@dataclass(frozen=True, slots=True)
class Phase1QwenProposalRunConfig:
    """Private input, support evidence, and output paths for one proposal run."""

    documents_path: Path
    expected_source_archive_sha256: str
    output_dir: Path
    support_sources: tuple[tuple[str, Path], ...] = ()
    expected_document_count: int = 100
    run_adjudication: bool = True

    def __post_init__(self) -> None:
        if not self.documents_path.is_file():
            raise ValueError("Qwen proposal documents manifest does not exist")
        names = [name for name, _ in self.support_sources]
        if len(names) != len(set(names)) or any(not name.strip() for name in names):
            raise ValueError("Qwen support source names must be non-empty and unique")
        if any(name.startswith("qwen.") for name in names):
            raise ValueError("External support source names cannot use the qwen.* namespace")


def run_phase1_qwen_proposals(
    run_spec: Phase1QwenRunSpec,
    config: Phase1QwenProposalRunConfig,
) -> dict[str, Any]:
    """Run recall, type-targeted, consensus, and optional adjudication passes."""

    run_spec.verify_dataset_inputs()
    documents = load_phase1_round2_documents(
        load_documents(config.documents_path),
        expected_archive_sha256=config.expected_source_archive_sha256,
        expected_count=config.expected_document_count,
    )
    support_by_name = {
        name: load_phase1_output_source(path)
        for name, path in config.support_sources
    }
    expected_ids = {document.document_id for document in documents}
    for name, rows_by_doc in support_by_name.items():
        if set(rows_by_doc) != expected_ids:
            raise ValueError(f"Support source {name!r} does not cover the private corpus")

    runtime = TransformersCausalLMRuntime(
        model_id=run_spec.model.model_id,
        revision=run_spec.model.revision,
        device=run_spec.device,
        dtype=run_spec.dtype,  # type: ignore[arg-type]
        local_files_only=run_spec.local_files_only,
    )
    adapter = Phase1QwenAdapter(
        runtime,
        max_window_characters=run_spec.max_window_characters,
        window_overlap_characters=run_spec.window_overlap_characters,
        structured_retries=run_spec.structured_retries,
    )
    output = config.output_dir
    consensus_dir = output / "consensus"
    adjudicated_dir = output / "adjudicated"
    consensus_dir.mkdir(parents=True, exist_ok=True)
    if config.run_adjudication:
        adjudicated_dir.mkdir(parents=True, exist_ok=True)

    trace_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for document in documents:
        proposal_sources, pass_trace, pass_raw = _run_document_passes(
            adapter,
            run_spec,
            document,
        )
        for name, rows_by_doc in support_by_name.items():
            proposal_sources[name] = _rows_to_proposals(
                rows_by_doc[document.document_id],
                document.text,
                source=name,
            )
        thresholds = {
            _PHASE1_LABEL_TO_TYPE[label]: threshold
            for label, threshold in run_spec.thresholds.items()
        }
        confirmed = select_qwen_confirmed_proposals(
            proposal_sources,
            thresholds=thresholds,
        )
        resolved = EvidenceWeightedSpanResolver().resolve(confirmed)
        consensus_rows = _proposals_to_rows(resolved.selected, document.text)
        _write_document_rows(
            consensus_dir / f"{document.document_id}.json",
            consensus_rows,
        )
        counters["consensus.entities"] += len(consensus_rows)
        counters["consensus.overlap_rejected"] += len(confirmed) - len(resolved.selected)

        adjudicated_rows: list[dict[str, Any]] = []
        adjudication_hash: str | None = None
        if config.run_adjudication:
            candidates = _adjudication_candidates(proposal_sources, document.text)
            decisions, adjudication_hash = adapter.adjudicate(
                document.text,
                candidates,
                generation=run_spec.adjudication_generation,
            )
            adjudicated = apply_phase1_adjudication(
                document.text,
                candidates,
                decisions,
                minimum_confidence=min(run_spec.thresholds.values()),
            )
            adjudicated_resolved = EvidenceWeightedSpanResolver().resolve(adjudicated)
            adjudicated_rows = _proposals_to_rows(
                adjudicated_resolved.selected,
                document.text,
            )
            _write_document_rows(
                adjudicated_dir / f"{document.document_id}.json",
                adjudicated_rows,
            )
            counters["adjudicated.entities"] += len(adjudicated_rows)
        trace_rows.append(
            {
                "document_id": document.document_id,
                "text_sha256": _text_sha256(document.text),
                "passes": pass_trace,
                "support_sources": sorted(support_by_name),
                "proposal_source_counts": {
                    name: len(rows) for name, rows in sorted(proposal_sources.items())
                },
                "consensus_entity_count": len(consensus_rows),
                "adjudicated_entity_count": len(adjudicated_rows),
                "adjudication_response_sha256": adjudication_hash,
            }
        )
        raw_rows.extend(
            {"document_id": document.document_id, **row} for row in pass_raw
        )

    trace_sha256 = write_jsonl(output / "trace.jsonl", trace_rows)
    raw_sha256 = write_jsonl(output / "raw_responses.jsonl", raw_rows)
    manifest = {
        "schema_version": "phase1-qwen-proposal-run.v1",
        "run_spec": run_spec.to_dict(),
        "inputs": {
            "documents": {
                "path": str(config.documents_path),
                "sha256": sha256_file(config.documents_path),
                "source_archive_sha256": config.expected_source_archive_sha256,
            },
            "support_sources": [
                {"name": name, "path": str(path), "sha256": sha256_file(path)}
                for name, path in config.support_sources
                if path.is_file()
            ],
        },
        "outputs": {
            "consensus_dir": str(consensus_dir),
            "adjudicated_dir": (
                str(adjudicated_dir) if config.run_adjudication else None
            ),
            "trace_sha256": trace_sha256,
            "raw_responses_sha256": raw_sha256,
        },
        "counters": dict(sorted(counters.items())),
        "policy": {
            "xlmr_direct_output": False,
            "qwen_confirmation_required": True,
            "new_entity_assertions": [],
            "new_entity_candidates": [],
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _run_document_passes(
    adapter: Phase1QwenAdapter,
    run_spec: Phase1QwenRunSpec,
    document: ClinicalDocument,
) -> tuple[
    dict[str, tuple[EntityProposal, ...]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    all_types = tuple(_TARGETED_PASSES)
    recall = adapter.extract(
        document.text,
        pass_id="recall",
        target_types=all_types,  # type: ignore[arg-type]
        generation=run_spec.recall_generation,
    )
    results = [recall]
    proposal_sources: dict[str, list[EntityProposal]] = {
        proposal.source: [] for proposal in recall.proposals
    }
    for proposal in recall.proposals:
        proposal_sources[proposal.source].append(proposal)
    for label in _TARGETED_PASSES:
        result = adapter.extract(
            document.text,
            pass_id=f"targeted.{label}",
            target_types=(label,),  # type: ignore[arg-type]
            generation=run_spec.targeted_generation,
        )
        results.append(result)
        for proposal in result.proposals:
            proposal_sources.setdefault(proposal.source, []).append(proposal)
    trace = [
        {
            "pass_id": result.pass_id,
            "prompt_hash": result.prompt_hash,
            "proposal_count": len(result.proposals),
            "rejected": list(result.rejected),
            "response_sha256": list(result.response_sha256),
        }
        for result in results
    ]
    raw = [
        {
            "pass_id": result.pass_id,
            "window_index": window_index,
            "response_sha256": result.response_sha256[window_index],
            "response": response,
        }
        for result in results
        for window_index, response in enumerate(result.raw_responses)
    ]
    return (
        {
            name: tuple(rows)
            for name, rows in sorted(proposal_sources.items())
        },
        trace,
        raw,
    )


def _rows_to_proposals(
    rows: Sequence[Mapping[str, Any]],
    source_text: str,
    *,
    source: str,
) -> tuple[EntityProposal, ...]:
    proposals: list[EntityProposal] = []
    for index, row in enumerate(rows):
        entity_type = _PHASE1_LABEL_TO_TYPE.get(str(row.get("type")))
        position = row.get("position")
        if (
            entity_type is None
            or not isinstance(position, list)
            or len(position) != 2
        ):
            raise ValueError(f"Invalid support entity from {source}: {row}")
        start, end = int(position[0]), int(position[1])
        if source_text[start:end] != row.get("text"):
            raise ValueError(f"Support entity from {source} violates raw offsets")
        proposals.append(
            EntityProposal(
                span=(start, end),
                candidate_types=(entity_type,),
                source=source,
                score=float(row.get("confidence", 1.0)),
                evidence_ids=(f"{source}.{index}",),
            )
        )
    return tuple(proposals)


def _adjudication_candidates(
    proposal_sources: Mapping[str, Sequence[EntityProposal]],
    source_text: str,
) -> tuple[Phase1AdjudicationCandidate, ...]:
    grouped: dict[
        tuple[int, int, EntityType],
        list[EntityProposal],
    ] = defaultdict(list)
    for proposals in proposal_sources.values():
        for proposal in proposals:
            if proposal.entity_type is not None:
                grouped[(*proposal.span, proposal.entity_type)].append(proposal)
    candidates: list[Phase1AdjudicationCandidate] = []
    for index, ((start, end, entity_type), evidence) in enumerate(
        sorted(grouped.items(), key=lambda item: item[0])
    ):
        candidates.append(
            Phase1AdjudicationCandidate(
                proposal_id=f"p{index:04d}",
                text=source_text[start:end],
                entity_type=_ENTITY_TYPE_TO_PHASE1_LABEL[entity_type],  # type: ignore[arg-type]
                span=(start, end),
                sources=tuple(sorted({proposal.source for proposal in evidence})),
                confidence=max(proposal.score for proposal in evidence),
            )
        )
    return tuple(candidates)


def _proposals_to_rows(
    proposals: Sequence[EntityProposal],
    source_text: str,
) -> list[dict[str, Any]]:
    rows = []
    for proposal in proposals:
        entity_type = proposal.entity_type
        if entity_type is None:
            continue
        start, end = proposal.span
        text = source_text[start:end]
        if not text:
            raise ValueError("Selected Qwen proposal has an empty raw span")
        rows.append(
            {
                "text": text,
                "type": _ENTITY_TYPE_TO_PHASE1_LABEL[entity_type],
                "assertions": [],
                "candidates": [],
                "position": [start, end],
            }
        )
    return rows


def _write_document_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    invalid_types = {str(row["type"]) for row in rows} - PHASE1_ALLOWED_TYPES
    if invalid_types:
        raise ValueError(f"Qwen output contains invalid Phase 1 types: {invalid_types}")
    path.write_text(
        json.dumps(list(rows), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _text_sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
