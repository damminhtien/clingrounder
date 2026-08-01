"""Pinned submission runner for learned Phase 1 joint span/type composition."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from medical_kg_nlp.adapters.generative.budget_spec import (
    load_inference_budget_spec,
    verify_inference_budget_spec,
)
from medical_kg_nlp.adapters.huggingface import HuggingFaceModelConfig
from medical_kg_nlp.benchmarks.phase1.joint_span import Phase1JointSpanSelectionPolicy
from medical_kg_nlp.benchmarks.phase1.joint_span_calibration import (
    load_phase1_joint_span_calibration,
)
from medical_kg_nlp.benchmarks.phase1.joint_span_pipeline import (
    Phase1JointSpanPipeline,
    Phase1JointSpanResult,
)
from medical_kg_nlp.benchmarks.phase1.joint_span_sources import (
    build_phase1_medication_parser_source_rows,
    build_phase1_rule_source_rows,
    load_phase1_joint_span_source_rows,
)
from medical_kg_nlp.benchmarks.phase1.joint_span_verifier import (
    HuggingFacePhase1JointSpanVerifier,
    calibrate_phase1_joint_span_verifier,
)
from medical_kg_nlp.benchmarks.phase1.max_score_pipeline import CandidateMetadataPolicy
from medical_kg_nlp.benchmarks.phase1.phase1 import (
    validate_phase1_submission_documents,
    validate_phase1_submission_zip,
    zip_phase1_output_dir,
)
from medical_kg_nlp.benchmarks.phase1.phase1_selective_overlays import AssertionRegime
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole
from medical_kg_nlp.benchmarks.phase1.reviewed_corpus import Phase1ReviewedCorpus
from medical_kg_nlp.benchmarks.phase1.round2 import load_phase1_round2_documents
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.mining.io import load_documents, write_json, write_jsonl, write_text
from medical_kg_nlp.utils.hashing import sha256_directory, sha256_file
from medical_kg_nlp.utils.io import read_yaml
from medical_kg_nlp.utils.run_output import create_hashed_run_dir

__all__ = [
    "Phase1JointSpanDirectoryArtifact",
    "Phase1JointSpanModelSourceSpec",
    "Phase1JointSpanRunSpec",
    "load_phase1_joint_span_run_spec",
    "run_phase1_joint_span",
]

_SPEC_SCHEMA = "phase1-joint-span-run-spec.v4"
_TRAINING_MANIFEST_SCHEMA = "phase1-joint-span-verifier-training.v2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class Phase1JointSpanFileArtifact:
    """One immutable file input verified before it is parsed."""

    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.sha256) is None:
            raise ValueError("Pinned file SHA-256 must be lowercase hexadecimal")

    def verify(self, *, name: str) -> None:
        if not self.path.is_file():
            raise ValueError(f"Pinned {name} is absent: {self.path}")
        observed = sha256_file(self.path)
        if observed != self.sha256:
            raise ValueError(
                f"Pinned {name} SHA-256 mismatch: expected={self.sha256}, observed={observed}"
            )


@dataclass(frozen=True, slots=True)
class Phase1JointSpanDirectoryArtifact:
    """One immutable model or source directory verified by content, not its path."""

    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.sha256) is None:
            raise ValueError("Pinned directory SHA-256 must be lowercase hexadecimal")

    def verify(self, *, name: str) -> None:
        if not self.path.is_dir():
            raise ValueError(f"Pinned {name} is absent: {self.path}")
        observed = sha256_directory(self.path)
        if observed != self.sha256:
            raise ValueError(
                f"Pinned {name} SHA-256 mismatch: expected={self.sha256}, observed={observed}"
            )


@dataclass(frozen=True, slots=True)
class Phase1JointSpanModelSourceSpec:
    """One pre-materialized independent model source with an explicit resolver role."""

    name: str
    role: ProposalSourceRole
    artifact: Phase1JointSpanDirectoryArtifact

    def __post_init__(self) -> None:
        if not self.name.strip() or self.name in {"rule", "medication_parser"}:
            raise ValueError("Joint span model source name is reserved or empty")
        if self.role is ProposalSourceRole.VERIFIER:
            raise ValueError("Verifier-only evidence cannot introduce joint span candidates")


@dataclass(frozen=True, slots=True)
class Phase1JointSpanRunSpec:
    """All explicit inputs required to regenerate one official 100-document submission."""

    config_path: Path
    run_root: Path
    documents: Phase1JointSpanFileArtifact
    source_archive_sha256: str
    expected_count: int
    budget_spec_path: Path
    verifier: Phase1JointSpanDirectoryArtifact
    verifier_model_id: str
    verifier_device: str
    verifier_batch_size: int
    verifier_max_length: int
    verifier_training_manifest: Phase1JointSpanFileArtifact
    calibration: Phase1JointSpanFileArtifact
    model_sources: tuple[Phase1JointSpanModelSourceSpec, ...]
    dictionaries: tuple[Phase1JointSpanFileArtifact, ...]
    candidate_source_priority: tuple[str, ...]
    assertion_regimes: tuple[AssertionRegime, ...]
    candidate_policy: CandidateMetadataPolicy
    output_root: Path
    run_label: str

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.source_archive_sha256) is None:
            raise ValueError("Source archive SHA-256 must be lowercase hexadecimal")
        if self.expected_count < 1:
            raise ValueError("Expected document count must be positive")
        if not self.verifier_model_id.strip() or not self.verifier_device.strip():
            raise ValueError("Joint verifier model identity and device must be non-empty")
        if self.verifier_batch_size < 1 or self.verifier_max_length < 32:
            raise ValueError("Joint verifier batch size or max length is invalid")
        source_names = tuple(source.name for source in self.model_sources)
        if not source_names or len(source_names) != len(set(source_names)):
            raise ValueError("Joint span model sources must be non-empty and unique")
        all_source_names = {"rule", "medication_parser", *source_names}
        if set(self.candidate_source_priority) != all_source_names:
            raise ValueError("Candidate priority must name generated and model sources exactly once")
        if len(self.candidate_source_priority) != len(all_source_names):
            raise ValueError("Candidate source priority contains duplicates")
        if not self.dictionaries:
            raise ValueError("Joint span submission requires pinned terminology dictionaries")
        if self.candidate_policy not in {"keep", "rx_unique_keep_icd"}:
            raise ValueError("Joint span candidate policy is unsupported")
        if not self.run_label.strip():
            raise ValueError("Joint span run label must be non-empty")


def load_phase1_joint_span_run_spec(path: str | Path) -> Phase1JointSpanRunSpec:
    """Load a path-portable joint-span submission configuration without implicit artifacts."""

    config_path = Path(path).resolve()
    raw = read_yaml(config_path)
    if raw.get("schema_version") != _SPEC_SCHEMA:
        raise ValueError("Unsupported Phase 1 joint span run spec")
    run_root = _resolve(config_path.parent, _required_string(raw, "run_root"))
    _require_below_root(config_path, run_root)
    documents_raw = _mapping(raw.get("documents"), "documents")
    verifier_raw = _mapping(raw.get("verifier"), "verifier")
    calibration_raw = _mapping(raw.get("calibration"), "calibration")
    source_rows = _list_of_mappings(raw.get("model_sources"), "model_sources")
    dictionary_rows = _list_of_mappings(raw.get("dictionaries"), "dictionaries")
    output_root = _resolve(run_root, str(raw.get("output_root", "outputs/phase1/joint-span")))
    budget_spec_path = _resolve(run_root, _required_string(raw, "budget_spec"))
    _require_below_root(output_root, run_root)
    _require_below_root(budget_spec_path, run_root)
    return Phase1JointSpanRunSpec(
        config_path=config_path,
        run_root=run_root,
        documents=_pinned_file(documents_raw, run_root),
        source_archive_sha256=_required_string(documents_raw, "source_archive_sha256"),
        expected_count=_required_int(documents_raw, "expected_count", default=100),
        budget_spec_path=budget_spec_path,
        verifier=_pinned_directory(verifier_raw, run_root),
        verifier_model_id=_required_string(verifier_raw, "model_id"),
        verifier_device=str(verifier_raw.get("device", "cpu")).strip(),
        verifier_batch_size=_required_int(verifier_raw, "batch_size", default=16),
        verifier_max_length=_required_int(verifier_raw, "max_length", default=384),
        verifier_training_manifest=_pinned_file(
            _mapping(verifier_raw.get("training_manifest"), "verifier.training_manifest"),
            run_root,
        ),
        calibration=_pinned_file(calibration_raw, run_root),
        model_sources=tuple(
            Phase1JointSpanModelSourceSpec(
                name=_required_string(row, "name"),
                role=ProposalSourceRole(_required_string(row, "role")),
                artifact=_pinned_directory(row, run_root),
            )
            for row in source_rows
        ),
        dictionaries=tuple(_pinned_file(row, run_root) for row in dictionary_rows),
        candidate_source_priority=_string_tuple(
            raw.get("candidate_source_priority"), "candidate_source_priority"
        ),
        assertion_regimes=tuple(
            cast(AssertionRegime, value)
            for value in _string_tuple(
                raw.get("assertion_regimes", ["negation", "history"]),
                "assertion_regimes",
            )
        ),
        candidate_policy=cast(
            CandidateMetadataPolicy,
            str(raw.get("candidate_policy", "rx_unique_keep_icd")),
        ),
        output_root=output_root,
        run_label=str(raw.get("run_label", "phase1-joint-span")).strip(),
    )


def run_phase1_joint_span(spec: Phase1JointSpanRunSpec) -> dict[str, Any]:
    """Regenerate, validate, and ZIP one learned joint-span Phase 1 submission."""

    spec.documents.verify(name="documents")
    spec.verifier.verify(name="joint span verifier")
    spec.verifier_training_manifest.verify(name="joint span verifier training manifest")
    spec.calibration.verify(name="joint span calibration")
    for source in spec.model_sources:
        source.artifact.verify(name=f"model source {source.name}")
    for index, artifact in enumerate(spec.dictionaries):
        artifact.verify(name=f"dictionary {index}")
    budget_manifest = verify_inference_budget_spec(load_inference_budget_spec(spec.budget_spec_path))
    documents = load_phase1_round2_documents(
        load_documents(spec.documents.path),
        expected_archive_sha256=spec.source_archive_sha256,
        expected_count=spec.expected_count,
    )
    corpus = _proposal_corpus(documents)
    dictionary = DictionaryStore(
        [
            entry
            for artifact in spec.dictionaries
            for entry in DictionaryStore.load_entries_jsonl(artifact.path)
        ]
    )
    source_roles = {
        "rule": ProposalSourceRole.RULE,
        "medication_parser": ProposalSourceRole.RULE,
        **{source.name: source.role for source in spec.model_sources},
    }
    proposal_sources = {
        "rule": build_phase1_rule_source_rows(corpus, dictionary),
        "medication_parser": build_phase1_medication_parser_source_rows(corpus, dictionary),
        **{
            source.name: load_phase1_joint_span_source_rows(source.artifact.path, corpus)
            for source in spec.model_sources
        },
    }
    calibration = load_phase1_joint_span_calibration(spec.calibration.path)
    training_family_fingerprint = _verified_training_family_fingerprint(
        spec.verifier_training_manifest.path,
        verifier_fingerprint=spec.verifier.sha256,
    )
    if calibration.training_family_fingerprint != training_family_fingerprint:
        raise ValueError(
            "Joint span calibration and final verifier do not share a training family"
        )
    base_verifier = HuggingFacePhase1JointSpanVerifier(
        HuggingFaceModelConfig(
            model_id=str(spec.verifier.path),
            revision=spec.verifier.sha256,
            device=spec.verifier_device,
            batch_size=spec.verifier_batch_size,
            max_length=spec.verifier_max_length,
        )
    )
    verifier = calibrate_phase1_joint_span_verifier(base_verifier, calibration)
    result = Phase1JointSpanPipeline(
        verifier=verifier,
        selection_policy=calibration.selection_policy,
        source_roles=source_roles,
        budget_manifest=budget_manifest,
        dictionary=dictionary,
        candidate_source_priority=spec.candidate_source_priority,
        assertion_regimes=spec.assertion_regimes,
        candidate_policy=spec.candidate_policy,
    ).run(documents, proposal_sources)
    run = create_hashed_run_dir(
        spec.output_root,
        label=spec.run_label,
        inputs=(
            spec.config_path,
            spec.documents.path,
            spec.budget_spec_path,
            spec.verifier.path,
            spec.verifier_training_manifest.path,
            spec.calibration.path,
            *(source.artifact.path for source in spec.model_sources),
            *(artifact.path for artifact in spec.dictionaries),
        ),
        resolved_config={
            "source_archive_sha256": spec.source_archive_sha256,
            "expected_count": spec.expected_count,
            "verifier": {
                "model_id": spec.verifier_model_id,
                "sha256": spec.verifier.sha256,
                "training_manifest_sha256": spec.verifier_training_manifest.sha256,
                "training_family_fingerprint": training_family_fingerprint,
                "device": spec.verifier_device,
                "batch_size": spec.verifier_batch_size,
                "max_length": spec.verifier_max_length,
            },
            "calibration": {
                "sha256": spec.calibration.sha256,
                "training_family_fingerprint": calibration.training_family_fingerprint,
                "oof_observations_sha256": calibration.oof_observations_sha256,
                "fold_assignment_sha256": calibration.fold_assignment_sha256,
            },
            "model_sources": [
                {"name": source.name, "role": source.role.value, "sha256": source.artifact.sha256}
                for source in spec.model_sources
            ],
            "selection_policy": _selection_policy_payload(calibration.selection_policy),
            "candidate_source_priority": list(spec.candidate_source_priority),
            "assertion_regimes": list(spec.assertion_regimes),
            "candidate_policy": spec.candidate_policy,
        },
    )
    artifact_report = _write_result(
        result,
        run.run_dir,
        documents=documents,
        dictionary=dictionary,
        expected_count=spec.expected_count,
    )
    run_manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    run_manifest["pipeline"] = artifact_report
    write_json(run.manifest_path, run_manifest)
    return {
        "schema_version": "phase1-joint-span-run.v1",
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "run_manifest": str(run.manifest_path),
        **artifact_report,
    }


def _verified_training_family_fingerprint(
    path: Path,
    *,
    verifier_fingerprint: str,
) -> str:
    """Read a pinned final-fit manifest and prove it owns the selected model directory.

    INVARIANT: OOF calibration is tied to the training family, not a fold checkpoint. The final
    run still verifies the exact model directory before accepting the family relationship.
    """

    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "verifier training manifest")
    if payload.get("schema_version") != _TRAINING_MANIFEST_SCHEMA:
        raise ValueError("Unsupported joint span verifier training manifest schema")
    model = _mapping(payload.get("model"), "joint span verifier training model")
    if model.get("fingerprint") != verifier_fingerprint:
        raise ValueError("Joint span verifier training manifest does not own the pinned model")
    family = payload.get("training_family_fingerprint")
    if not isinstance(family, str) or _SHA256_RE.fullmatch(family) is None:
        raise ValueError("Joint span verifier training family fingerprint is invalid")
    return family


def _proposal_corpus(documents: Sequence[Any]) -> Phase1ReviewedCorpus:
    """Adapt raw competition inputs for deterministic rule/model proposal generation only."""

    source_texts = {str(document.document_id): str(document.text) for document in documents}
    if len(source_texts) != len(documents):
        raise ValueError("Joint span submission inputs contain duplicate document IDs")
    return Phase1ReviewedCorpus(
        source_texts=source_texts,
        gold_rows={document_id: () for document_id in source_texts},
        split_by_document={document_id: "submission" for document_id in source_texts},
    )


def _write_result(
    result: Phase1JointSpanResult,
    run_dir: Path,
    *,
    documents: Sequence[Any],
    dictionary: DictionaryStore,
    expected_count: int,
) -> dict[str, Any]:
    """Write auditable evidence and reject any directory or ZIP schema/offset violation."""

    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    for document_id, rows in sorted(result.rows_by_document.items(), key=_document_sort_key):
        write_text(
            output_dir / f"{document_id}.json",
            json.dumps(list(rows), ensure_ascii=False, indent=2) + "\n",
        )
    issues = validate_phase1_submission_documents(documents, output_dir, dictionary=dictionary)
    if issues:
        raise ValueError(f"Joint span directory validation failed: {issues[0].to_json()}")
    zip_path = run_dir / "output.zip"
    zip_phase1_output_dir(output_dir, zip_path)
    zip_issues = validate_phase1_submission_zip(
        zip_path,
        documents=documents,
        dictionary=dictionary,
        expected_count=expected_count,
    )
    if zip_issues:
        raise ValueError(f"Joint span ZIP validation failed: {zip_issues[0].to_json()}")
    write_json(run_dir / "budget_manifest.json", result.budget_manifest)
    write_json(run_dir / "proposal_summary.json", _mapping(result.proposal_matrix.get("summary"), "summary"))
    write_jsonl(run_dir / "proposal_matrix.jsonl", _rows(result.proposal_matrix.get("matrix"), "matrix"))
    write_jsonl(run_dir / "joint_scores.jsonl", (item.to_dict() for item in result.joint_scores))
    write_jsonl(run_dir / "source_decisions.jsonl", result.source_decisions)
    write_jsonl(run_dir / "assertion_decisions.jsonl", result.assertion_decisions)
    write_jsonl(run_dir / "candidate_decisions.jsonl", result.candidate_decisions)
    write_json(run_dir / "counters.json", dict(result.counters))
    return {
        "output_dir": str(output_dir),
        "zip_path": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "document_count": len(result.rows_by_document),
        "entity_count": sum(len(rows) for rows in result.rows_by_document.values()),
        "validation_issue_count": 0,
        "counters": dict(result.counters),
        "budget": {
            "total_parameters": result.budget_manifest["total_parameters"],
            "maximum_parameters": result.budget_manifest["maximum_parameters"],
        },
    }


def _pinned_file(raw: Mapping[str, Any], run_root: Path) -> Phase1JointSpanFileArtifact:
    path = _resolve(run_root, _required_string(raw, "path"))
    _require_below_root(path, run_root)
    return Phase1JointSpanFileArtifact(path=path, sha256=_required_string(raw, "sha256"))


def _pinned_directory(
    raw: Mapping[str, Any], run_root: Path
) -> Phase1JointSpanDirectoryArtifact:
    path = _resolve(run_root, _required_string(raw, "path"))
    _require_below_root(path, run_root)
    return Phase1JointSpanDirectoryArtifact(path=path, sha256=_required_string(raw, "sha256"))


def _selection_policy_payload(policy: Phase1JointSpanSelectionPolicy) -> dict[str, Any]:
    return {
        "genre_type_thresholds": [
            {"genre": genre, "type": entity_type, "threshold": threshold}
            for genre, entity_type, threshold in policy.genre_type_thresholds
        ],
        "false_positive_cost": policy.false_positive_cost,
    }


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _require_below_root(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Joint span path escapes run_root: {path}") from error


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _list_of_mappings(value: object, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return [_mapping(item, field) for item in value]


def _required_string(raw: Mapping[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _required_int(raw: Mapping[str, Any], field: str, *, default: int) -> int:
    value = raw.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    values = tuple(str(item).strip() for item in value)
    if not values or any(not item for item in values):
        raise ValueError(f"{field} must contain non-empty strings")
    return values


def _rows(value: object, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, Mapping) for row in value):
        raise ValueError(f"{field} must be a list of mappings")
    return [dict(row) for row in value]


def _document_sort_key(item: tuple[str, object]) -> tuple[int, int | str]:
    document_id = item[0]
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)
