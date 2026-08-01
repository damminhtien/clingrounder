"""Build deterministic joint span/type supervision from governed text and proposal sources."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.benchmarks.phase1.joint_span import (
    Phase1JointSpanCandidate,
    Phase1JointSpanLabel,
    generate_phase1_joint_span_lattice,
    label_phase1_joint_span_candidate,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole
from medical_kg_nlp.benchmarks.phase1.reviewed_corpus import Phase1ReviewedCorpus
from medical_kg_nlp.benchmarks.phase1.split_contract import phase1_document_sort_key
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.mining.io import write_json, write_jsonl
from medical_kg_nlp.ner.dictionary_matcher import DictionaryMatcher
from medical_kg_nlp.ner.rule_ner import RuleBasedNER
from medical_kg_nlp.ontology.phase1 import PHASE1_TYPE_BY_ENTITY_TYPE

__all__ = [
    "Phase1JointSpanDataset",
    "Phase1JointSpanExample",
    "build_phase1_joint_span_dataset",
    "build_phase1_rule_proposal_rows",
    "write_phase1_joint_span_dataset",
]

_DATASET_SCHEMA = "phase1-joint-span-dataset.v1"
_MAX_NEGATIVES_PER_FAMILY = 5


@dataclass(frozen=True, slots=True)
class Phase1JointSpanExample:
    """One exact source substring, multi-class target, and model-ready cross-encoder text."""

    candidate: Phase1JointSpanCandidate
    label: Phase1JointSpanLabel
    split: str
    source_dataset: str

    def __post_init__(self) -> None:
        if self.split not in {"train", "development", "holdout"}:
            raise ValueError("Joint span example split is invalid")
        if not self.source_dataset.strip():
            raise ValueError("Joint span example source dataset is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.candidate.to_dict(),
            "label": self.label.value,
            "split": self.split,
            "source_dataset": self.source_dataset,
        }


@dataclass(frozen=True, slots=True)
class Phase1JointSpanDataset:
    """Materialized examples and immutable manifest for a transformer verifier run."""

    examples: tuple[Phase1JointSpanExample, ...]
    manifest: Mapping[str, Any]


def build_phase1_rule_proposal_rows(
    corpus: Phase1ReviewedCorpus,
    dictionary: DictionaryStore,
) -> tuple[dict[str, tuple[dict[str, Any], ...]], dict[str, ProposalSourceRole]]:
    """Preserve independent RuleNER extractor proposals before its legacy span resolver runs.

    MODEL: ambiguous disease/symptom proposals are emitted once per candidate type. The joint
    verifier learns whether either exact type is correct instead of applying a disease fallback.
    """

    ner = RuleBasedNER(dictionary, disease_symptom_fallback="abstain")
    rows_by_document: dict[str, tuple[dict[str, Any], ...]] = {}
    source_roles: dict[str, ProposalSourceRole] = {}
    for document_id in sorted(corpus.source_texts, key=phase1_document_sort_key):
        source_text = corpus.source_texts[document_id]
        trace = ner.extract_with_trace(source_text).trace
        rows: list[dict[str, Any]] = []
        for proposal_index, proposal in enumerate(trace.proposals):
            start, end = proposal.span
            source_name = f"rule:{proposal.source}"
            source_roles[source_name] = ProposalSourceRole.RULE
            for entity_type in proposal.candidate_types:
                phase1_type = PHASE1_TYPE_BY_ENTITY_TYPE.get(entity_type)
                if phase1_type is None:
                    continue
                rows.append(
                    {
                        "document_id": document_id,
                        "proposal_id": (
                            f"{document_id}:rule:{proposal_index}:{start}:{end}:"
                            f"{phase1_type}"
                        ),
                        "text": source_text[start:end],
                        "type": phase1_type,
                        "position": [start, end],
                        "sources": [source_name],
                        "source_count": 1,
                        "all_source_agreement": False,
                        "status": "source_only",
                        "source_evidence": {
                            source_name: {
                                "confidence": proposal.score,
                                "source_labels": [entity_type.value],
                                "support_only": False,
                            }
                        },
                    }
                )
        rows_by_document[document_id] = tuple(sorted(rows, key=_proposal_row_sort_key))
    if not source_roles:
        raise ValueError("Rule proposal construction produced no source roles")
    return rows_by_document, dict(sorted(source_roles.items()))


def build_phase1_joint_span_dataset(
    corpus: Phase1ReviewedCorpus,
    proposal_rows_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    source_roles: Mapping[str, ProposalSourceRole | str],
    source_dataset_by_document: Mapping[str, str],
    dictionary_matcher: DictionaryMatcher | None = None,
) -> Phase1JointSpanDataset:
    """Build balanced joint examples without adding gold-only candidates to the lattice.

    A document may have zero exact candidates. That is intentional telemetry: candidate coverage
    describes proposal-source recall and cannot be repaired by seeding labels into model inputs.
    """

    if set(corpus.source_texts) != set(source_dataset_by_document):
        raise ValueError("Joint span source provenance must cover every corpus document")
    unknown_rows = set(proposal_rows_by_document) - set(corpus.source_texts)
    if unknown_rows:
        raise ValueError("Joint span proposal rows reference unknown corpus documents")
    examples: list[Phase1JointSpanExample] = []
    gold_total = exact_covered = 0
    for document_id in sorted(corpus.source_texts, key=phase1_document_sort_key):
        source_text = corpus.source_texts[document_id]
        candidates = generate_phase1_joint_span_lattice(
            document_id,
            source_text,
            proposal_rows_by_document.get(document_id, ()),
            source_roles=source_roles,
            dictionary_matcher=dictionary_matcher,
        )
        gold_rows = corpus.gold_rows[document_id]
        labels = [
            (candidate, label_phase1_joint_span_candidate(candidate, gold_rows))
            for candidate in candidates
        ]
        # INVARIANT: coverage is recall over unique gold identities, not the number of exact
        # lattice rows. Multiple generators can emit the same raw span/type candidate and must
        # not inflate the OOF denominator or numerator.
        gold_identities = {_row_identity(row) for row in gold_rows}
        exact_candidate_identities = {
            _candidate_identity(candidate)
            for candidate, label in labels
            if label.value.startswith("EXACT_")
        }
        gold_total += len(gold_identities)
        exact_covered += len(gold_identities & exact_candidate_identities)
        for candidate, label in _sample_lattice_family_examples(labels):
            examples.append(
                Phase1JointSpanExample(
                    candidate=candidate,
                    label=label,
                    split=corpus.split_by_document[document_id],
                    source_dataset=source_dataset_by_document[document_id],
                )
            )
    examples.sort(key=_example_sort_key)
    serialized = "".join(
        json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for example in examples
    )
    manifest = {
        "schema_version": _DATASET_SCHEMA,
        "example_count": len(examples),
        "examples_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "source_document_count": len(corpus.source_texts),
        "source_dataset_counts": dict(
            sorted(Counter(source_dataset_by_document.values()).items())
        ),
        "split_counts": dict(sorted(Counter(example.split for example in examples).items())),
        "label_counts": dict(sorted(Counter(example.label.value for example in examples).items())),
        "candidate_coverage": {
            "covered_gold": exact_covered,
            "gold": gold_total,
            "recall": exact_covered / gold_total if gold_total else 0.0,
        },
        "source_roles": {
            source: ProposalSourceRole(role).value
            for source, role in sorted(source_roles.items())
        },
        "round2_included": False,
        "friend31_included": False,
    }
    return Phase1JointSpanDataset(tuple(examples), manifest)


def _row_identity(row: Mapping[str, Any]) -> tuple[int, int, str]:
    """Return the only identity an exact-span/type lattice candidate can cover."""

    position = row.get("position")
    entity_type = row.get("type")
    if (
        not isinstance(position, Sequence)
        or isinstance(position, str)
        or len(position) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in position)
        or not isinstance(entity_type, str)
    ):
        raise ValueError("Joint span gold rows require a raw position and Phase 1 type")
    return position[0], position[1], entity_type


def _candidate_identity(candidate: Phase1JointSpanCandidate) -> tuple[int, int, str]:
    """Project one proposal onto the raw identity used by exact-label coverage."""

    start, end = candidate.variant.position
    return start, end, candidate.variant.entity_type


def write_phase1_joint_span_dataset(
    dataset: Phase1JointSpanDataset,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write model examples and a hash-pinned manifest for local or Vast training."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    example_path = output / "examples.jsonl"
    manifest_path = output / "manifest.json"
    sha256 = write_jsonl(example_path, (example.to_dict() for example in dataset.examples))
    if sha256 != dataset.manifest.get("examples_sha256"):
        raise RuntimeError("Joint span dataset serialization changed its precomputed SHA-256")
    write_json(manifest_path, dict(dataset.manifest))
    return {
        "examples_path": str(example_path),
        "manifest_path": str(manifest_path),
        "examples_sha256": sha256,
    }


def _sample_lattice_family_examples(
    labeled: Sequence[tuple[Phase1JointSpanCandidate, Phase1JointSpanLabel]],
) -> tuple[tuple[Phase1JointSpanCandidate, Phase1JointSpanLabel], ...]:
    """Keep all exact candidates and diverse bounded hard negatives within each family."""

    by_family: dict[str, list[tuple[Phase1JointSpanCandidate, Phase1JointSpanLabel]]] = defaultdict(list)
    for item in labeled:
        by_family[item[0].variant.family_id].append(item)
    sampled: list[tuple[Phase1JointSpanCandidate, Phase1JointSpanLabel]] = []
    for family_id in sorted(by_family):
        values = by_family[family_id]
        exact = [item for item in values if item[1].value.startswith("EXACT_")]
        negatives = sorted(
            (item for item in values if not item[1].value.startswith("EXACT_")),
            key=lambda item: (
                item[1].value,
                -len(item[0].variant.sources),
                item[0].variant.position,
                item[0].variant.variant_id,
            ),
        )
        sampled.extend(exact)
        chosen: list[tuple[Phase1JointSpanCandidate, Phase1JointSpanLabel]] = []
        seen_labels: set[Phase1JointSpanLabel] = set()
        for item in negatives:
            if item[1] in seen_labels or len(chosen) >= _MAX_NEGATIVES_PER_FAMILY:
                continue
            chosen.append(item)
            seen_labels.add(item[1])
        for item in negatives:
            if len(chosen) >= _MAX_NEGATIVES_PER_FAMILY:
                break
            if item not in chosen:
                chosen.append(item)
        sampled.extend(chosen)
    return tuple(sorted(sampled, key=lambda item: _candidate_order(item[0])))


def _example_sort_key(example: Phase1JointSpanExample) -> tuple[Any, ...]:
    return (*_candidate_order(example.candidate), example.label.value)


def _candidate_order(candidate: Phase1JointSpanCandidate) -> tuple[Any, ...]:
    return (
        phase1_document_sort_key(candidate.variant.document_id),
        candidate.variant.position,
        candidate.variant.entity_type,
        candidate.variant.variant_id,
    )


def _proposal_row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    position = row["position"]
    assert isinstance(position, list)
    return (int(position[0]), int(position[1]), str(row["type"]), str(row["proposal_id"]))
