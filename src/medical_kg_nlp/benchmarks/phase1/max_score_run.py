"""Pinned artifact runner for the calibrated Phase 1 max-score pipeline."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from medical_kg_nlp.adapters.generative.budget_spec import (
    load_inference_budget_spec,
    verify_inference_budget_spec,
)
from medical_kg_nlp.benchmarks.phase1.max_score_pipeline import (
    CandidateMetadataPolicy,
    Phase1MaxScorePipeline,
    Phase1MaxScoreResult,
)
from medical_kg_nlp.benchmarks.phase1.phase1 import (
    validate_phase1_submission_documents,
    validate_phase1_submission_zip,
    zip_phase1_output_dir,
)
from medical_kg_nlp.benchmarks.phase1.phase1_ensemble import (
    load_phase1_output_source,
)
from medical_kg_nlp.benchmarks.phase1.phase1_selective_overlays import (
    AssertionRegime,
)
from medical_kg_nlp.benchmarks.phase1.proposal_calibration import (
    Phase1ProposalVerifier,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole
from medical_kg_nlp.benchmarks.phase1.round2 import (
    load_phase1_round2_documents,
)
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.mining.io import (
    load_documents,
    write_json,
    write_jsonl,
    write_text,
)
from medical_kg_nlp.ontology.phase1 import PHASE1_ALLOWED_TYPES
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.io import read_yaml
from medical_kg_nlp.utils.run_output import create_hashed_run_dir

__all__ = [
    "Phase1MaxScoreRunSpec",
    "PinnedPhase1Artifact",
    "load_phase1_max_score_run_spec",
    "run_phase1_max_score",
]

_SPEC_SCHEMA = "phase1-max-score-run-spec.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class PinnedPhase1Artifact:
    """One immutable file input whose bytes must match before use."""

    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.sha256) is None:
            raise ValueError("Pinned artifact SHA-256 must be lowercase hexadecimal")

    def verify(self, *, name: str) -> None:
        """Fail closed before parsing stale, partial, or replaced bytes."""

        if not self.path.is_file():
            raise ValueError(f"Pinned {name} is absent: {self.path}")
        actual = sha256_file(self.path)
        if actual != self.sha256:
            raise ValueError(
                f"Pinned {name} SHA-256 mismatch: expected {self.sha256}, got {actual}"
            )


@dataclass(frozen=True, slots=True)
class Phase1ProposalSourceSpec:
    """Pinned proposal artifact and its model-neutral calibration role."""

    name: str
    role: ProposalSourceRole
    artifact: PinnedPhase1Artifact

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Proposal source name must be non-empty")


@dataclass(frozen=True, slots=True)
class Phase1MaxScoreRunSpec:
    """Portable composition config with no implicit model or terminology paths."""

    config_path: Path
    run_root: Path
    documents: PinnedPhase1Artifact
    source_archive_sha256: str
    expected_count: int
    budget_spec_path: Path
    verifier: PinnedPhase1Artifact
    proposal_thresholds: tuple[tuple[str, float], ...]
    sources: tuple[Phase1ProposalSourceSpec, ...]
    dictionaries: tuple[PinnedPhase1Artifact, ...]
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
        names = tuple(source.name for source in self.sources)
        if len(names) < 2 or len(names) != len(set(names)):
            raise ValueError("Max-score run requires unique names for at least two sources")
        if set(names) != set(self.candidate_source_priority):
            raise ValueError("Candidate priority must name every proposal source")
        if not self.dictionaries:
            raise ValueError("Max-score run requires pinned terminology sources")
        threshold_types = {entity_type for entity_type, _ in self.proposal_thresholds}
        if threshold_types and threshold_types != set(PHASE1_ALLOWED_TYPES):
            raise ValueError(
                "Proposal threshold overrides must define every Phase 1 entity type"
            )
        if len(threshold_types) != len(self.proposal_thresholds) or any(
            not 0.0 <= threshold <= 1.0
            for _, threshold in self.proposal_thresholds
        ):
            raise ValueError("Proposal threshold overrides are invalid")
        if not self.run_label.strip():
            raise ValueError("Max-score run label must be non-empty")


def load_phase1_max_score_run_spec(
    path: str | Path,
) -> Phase1MaxScoreRunSpec:
    """Load and validate a path-portable max-score run specification."""

    config_path = Path(path).resolve()
    raw = read_yaml(config_path)
    if raw.get("schema_version") != _SPEC_SCHEMA:
        raise ValueError("Unsupported Phase 1 max-score run spec")
    run_root = _resolve(config_path.parent, _required_string(raw, "run_root"))
    _require_below_root(config_path, run_root)
    documents_raw = _mapping(raw.get("documents"), "documents")
    verifier_raw = _mapping(raw.get("verifier"), "verifier")
    source_rows = _list_of_mappings(raw.get("sources"), "sources")
    dictionary_rows = _list_of_mappings(raw.get("dictionaries"), "dictionaries")
    output_root = _resolve(run_root, str(raw.get("output_root", "outputs/phase1/max-score")))
    budget_spec_path = _resolve(
        run_root,
        _required_string(raw, "budget_spec"),
    )
    _require_below_root(output_root, run_root)
    _require_below_root(budget_spec_path, run_root)
    return Phase1MaxScoreRunSpec(
        config_path=config_path,
        run_root=run_root,
        documents=_pinned_artifact(documents_raw, run_root),
        source_archive_sha256=_required_string(
            documents_raw,
            "source_archive_sha256",
        ),
        expected_count=int(documents_raw.get("expected_count", 100)),
        budget_spec_path=budget_spec_path,
        verifier=_pinned_artifact(verifier_raw, run_root),
        proposal_thresholds=_threshold_overrides(raw.get("proposal_thresholds")),
        sources=tuple(
            Phase1ProposalSourceSpec(
                name=_required_string(row, "name"),
                role=ProposalSourceRole(_required_string(row, "role")),
                artifact=_pinned_artifact(row, run_root),
            )
            for row in source_rows
        ),
        dictionaries=tuple(
            _pinned_artifact(row, run_root) for row in dictionary_rows
        ),
        candidate_source_priority=_string_tuple(
            raw.get("candidate_source_priority"),
            "candidate_source_priority",
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
        run_label=str(raw.get("run_label", "phase1-under9b-max")).strip(),
    )


def run_phase1_max_score(spec: Phase1MaxScoreRunSpec) -> dict[str, Any]:
    """Verify all artifacts, compose the final output, and write a strict ZIP."""

    spec.documents.verify(name="documents")
    spec.verifier.verify(name="proposal verifier")
    for source in spec.sources:
        source.artifact.verify(name=f"proposal source {source.name}")
    for index, artifact in enumerate(spec.dictionaries):
        artifact.verify(name=f"dictionary {index}")

    budget_manifest = verify_inference_budget_spec(
        load_inference_budget_spec(spec.budget_spec_path)
    )
    verifier_payload = json.loads(spec.verifier.path.read_text(encoding="utf-8"))
    if not isinstance(verifier_payload, Mapping):
        raise ValueError("Proposal verifier artifact must be a JSON object")
    verifier = Phase1ProposalVerifier.from_dict(verifier_payload)
    if spec.proposal_thresholds:
        # MODEL: the learned probabilities stay immutable while an aggregate public-density
        # operating point may replace the development thresholds for every entity type.
        overrides = dict(spec.proposal_thresholds)
        verifier = replace(
            verifier,
            thresholds=spec.proposal_thresholds,
            genre_thresholds=tuple(
                (genre, entity_type, overrides[entity_type])
                for genre, entity_type, _ in verifier.genre_thresholds
            ),
        )
    documents = load_phase1_round2_documents(
        load_documents(spec.documents.path),
        expected_archive_sha256=spec.source_archive_sha256,
        expected_count=spec.expected_count,
    )
    dictionary = DictionaryStore(
        [
            entry
            for artifact in spec.dictionaries
            for entry in DictionaryStore.load_entries_jsonl(artifact.path)
        ]
    )
    sources = {
        source.name: load_phase1_output_source(source.artifact.path)
        for source in spec.sources
    }
    pipeline = Phase1MaxScorePipeline(
        verifier=verifier,
        source_roles={source.name: source.role for source in spec.sources},
        budget_manifest=budget_manifest,
        dictionary=dictionary,
        candidate_source_priority=spec.candidate_source_priority,
        assertion_regimes=spec.assertion_regimes,
        candidate_policy=spec.candidate_policy,
    )
    result = pipeline.run(documents, sources)

    run = create_hashed_run_dir(
        spec.output_root,
        label=spec.run_label,
        inputs=(
            spec.config_path,
            spec.documents.path,
            spec.budget_spec_path,
            spec.verifier.path,
            *(source.artifact.path for source in spec.sources),
            *(artifact.path for artifact in spec.dictionaries),
        ),
        resolved_config={
            "source_archive_sha256": spec.source_archive_sha256,
            "expected_count": spec.expected_count,
            "sources": [
                {"name": source.name, "role": source.role.value}
                for source in spec.sources
            ],
            "proposal_thresholds": dict(spec.proposal_thresholds),
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
        "schema_version": "phase1-max-score-run.v1",
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "run_manifest": str(run.manifest_path),
        **artifact_report,
    }


def _write_result(
    result: Phase1MaxScoreResult,
    run_dir: Path,
    *,
    documents: list[Any],
    dictionary: DictionaryStore,
    expected_count: int,
) -> dict[str, Any]:
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    for document_id in sorted(
        result.rows_by_document,
        key=_document_sort_key,
    ):
        write_text(
            output_dir / f"{document_id}.json",
            json.dumps(
                list(result.rows_by_document[document_id]),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )
    issues = validate_phase1_submission_documents(
        documents,
        output_dir,
        dictionary=dictionary,
    )
    if issues:
        raise ValueError(
            f"Max-score directory validation failed: {issues[0].to_json()}"
        )

    zip_path = run_dir / "output.zip"
    zip_phase1_output_dir(output_dir, zip_path)
    zip_issues = validate_phase1_submission_zip(
        zip_path,
        documents=documents,
        dictionary=dictionary,
        expected_count=expected_count,
    )
    if zip_issues:
        raise ValueError(
            f"Max-score ZIP validation failed: {zip_issues[0].to_json()}"
        )
    write_json(run_dir / "budget_manifest.json", result.budget_manifest)
    write_json(run_dir / "proposal_summary.json", result.proposal_matrix["summary"])
    write_jsonl(run_dir / "proposal_matrix.jsonl", result.proposal_matrix["matrix"])
    write_jsonl(
        run_dir / "proposal_scores.jsonl",
        (item.to_dict() for item in result.proposal_scores),
    )
    write_jsonl(run_dir / "source_decisions.jsonl", result.source_decisions)
    write_jsonl(run_dir / "assertion_decisions.jsonl", result.assertion_decisions)
    write_jsonl(run_dir / "candidate_decisions.jsonl", result.candidate_decisions)
    write_json(run_dir / "counters.json", result.counters)
    return {
        "output_dir": str(output_dir),
        "zip_path": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "document_count": len(result.rows_by_document),
        "entity_count": sum(
            len(rows) for rows in result.rows_by_document.values()
        ),
        "validation_issue_count": 0,
        "counters": dict(result.counters),
        "budget": {
            "total_parameters": result.budget_manifest["total_parameters"],
            "maximum_parameters": result.budget_manifest["maximum_parameters"],
        },
    }


def _pinned_artifact(
    raw: Mapping[str, Any],
    run_root: Path,
) -> PinnedPhase1Artifact:
    path = _resolve(run_root, _required_string(raw, "path"))
    _require_below_root(path, run_root)
    return PinnedPhase1Artifact(
        path=path,
        sha256=_required_string(raw, "sha256"),
    )


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _require_below_root(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Max-score path escapes run_root: {path}") from error


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


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    result = tuple(str(item).strip() for item in value)
    if not result or any(not item for item in result):
        raise ValueError(f"{field} must contain non-empty strings")
    return result


def _threshold_overrides(value: object) -> tuple[tuple[str, float], ...]:
    if value is None:
        return ()
    raw = _mapping(value, "proposal_thresholds")
    thresholds: list[tuple[str, float]] = []
    for entity_type, threshold in raw.items():
        if (
            not isinstance(entity_type, str)
            or not isinstance(threshold, int | float)
            or isinstance(threshold, bool)
        ):
            raise ValueError(
                "Proposal threshold overrides must map type strings to numbers"
            )
        thresholds.append((entity_type, float(threshold)))
    return tuple(sorted(thresholds))


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)
