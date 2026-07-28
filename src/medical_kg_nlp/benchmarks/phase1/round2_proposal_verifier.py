"""Apply a calibrated proposal verifier as an isolated Round 2 entity probe."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medical_kg_nlp.benchmarks.phase1.phase1 import (
    validate_phase1_submission_documents,
    validate_phase1_submission_zip,
    zip_phase1_output_dir,
)
from medical_kg_nlp.benchmarks.phase1.phase1_ensemble import (
    load_phase1_output_source,
)
from medical_kg_nlp.benchmarks.phase1.phase1_proposals import (
    build_phase1_proposal_matrix,
    write_phase1_proposal_matrix,
)
from medical_kg_nlp.benchmarks.phase1.phase1_selective_overlays import (
    validate_probe_isolation,
)
from medical_kg_nlp.benchmarks.phase1.proposal_calibration import (
    Phase1ProposalVerifier,
    ScoredPhase1Proposal,
    score_phase1_proposal_rows,
)
from medical_kg_nlp.benchmarks.phase1.proposal_features import ProposalSourceRole
from medical_kg_nlp.benchmarks.phase1.proposal_features import (
    is_phase1_heading_only_proposal,
)
from medical_kg_nlp.benchmarks.phase1.round2 import load_phase1_round2_documents
from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.mining.io import load_documents, write_json, write_jsonl
from medical_kg_nlp.schema.document import ClinicalDocument
from medical_kg_nlp.utils.hashing import sha256_file
from medical_kg_nlp.utils.run_output import create_hashed_run_dir

__all__ = [
    "Phase1Round2ProposalVerifierConfig",
    "apply_verified_proposal_additions",
    "run_phase1_round2_proposal_verifier",
]


@dataclass(frozen=True, slots=True)
class Phase1Round2ProposalVerifierConfig:
    """Pinned inputs for one additive-only calibrated entity probe."""

    documents_path: Path
    expected_source_archive_sha256: str
    base: Path
    expected_base_sha256: str
    proposal_source: Path
    expected_proposal_source_sha256: str
    verifier_path: Path
    expected_verifier_sha256: str
    dictionary_paths: tuple[Path, ...]
    output_root: Path = Path("outputs/phase1/round2")
    run_label: str = "round2-proposal-verifier"
    expected_count: int = 100

    def __post_init__(self) -> None:
        for name, value in (
            ("base", self.expected_base_sha256),
            ("proposal source", self.expected_proposal_source_sha256),
            ("verifier", self.expected_verifier_sha256),
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"Expected {name} SHA-256 must be lowercase hexadecimal")
        if not self.dictionary_paths:
            raise ValueError("Round 2 proposal verifier requires validation dictionaries")
        if self.expected_count < 1:
            raise ValueError("Expected Round 2 document count must be positive")


def apply_verified_proposal_additions(
    base_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    scored: Sequence[ScoredPhase1Proposal],
    *,
    source_text_by_document: Mapping[str, str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], Counter[str]]:
    """Add selected non-overlapping proposals without mutating baseline rows."""

    output = {
        document_id: [dict(row) for row in rows]
        for document_id, rows in base_by_document.items()
    }
    decisions: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for item in scored:
        row = item.row
        document_id = str(row.get("document_id", ""))
        if document_id not in output:
            raise ValueError(f"Verified proposal references unknown document {document_id!r}")
        action = "blocked_below_threshold"
        if item.rejection_reason == "structural_heading":
            action = "blocked_structural_heading"
        elif item.selected:
            if (
                source_text_by_document is not None
                and is_phase1_heading_only_proposal(
                    row,
                    source_text_by_document[document_id],
                )
            ):
                action = "blocked_structural_heading"
            elif any(_rows_overlap(row, existing) for existing in output[document_id]):
                action = "blocked_baseline_overlap"
            else:
                output[document_id].append(_entity_only_row(row))
                output[document_id].sort(key=_row_sort_key)
                action = "added"
        elif item.selected_before_overlap:
            action = "blocked_proposal_overlap"
        counters[action] += 1
        decisions.append(
            {
                "document_id": document_id,
                "proposal_id": row.get("proposal_id"),
                "text": row.get("text"),
                "type": row.get("type"),
                "position": row.get("position"),
                "sources": row.get("sources"),
                "status": row.get("status"),
                "probability": item.probability,
                "threshold": item.threshold,
                "action": action,
                "reason": item.rejection_reason,
            }
        )
    counters["output_entity_total"] = sum(len(rows) for rows in output.values())
    return output, decisions, counters


def run_phase1_round2_proposal_verifier(
    config: Phase1Round2ProposalVerifierConfig,
) -> dict[str, Any]:
    """Build and strict-validate one content-addressed calibrated entity probe."""

    _verify_hash(config.base, config.expected_base_sha256, name="baseline")
    _verify_hash(
        config.proposal_source,
        config.expected_proposal_source_sha256,
        name="proposal source",
    )
    _verify_hash(config.verifier_path, config.expected_verifier_sha256, name="verifier")
    documents = load_phase1_round2_documents(
        load_documents(config.documents_path),
        expected_archive_sha256=config.expected_source_archive_sha256,
        expected_count=config.expected_count,
    )
    source_text_by_document = {
        document.document_id: document.text for document in documents
    }
    base = load_phase1_output_source(config.base)
    proposal_source = load_phase1_output_source(config.proposal_source)
    expected_ids = set(source_text_by_document)
    if set(base) != expected_ids or set(proposal_source) != expected_ids:
        raise ValueError("Baseline/proposal document ids do not match Round 2 input")
    dictionary = _load_dictionary(config.dictionary_paths)
    _validate_artifact(
        config.base,
        documents=documents,
        dictionary=dictionary,
        expected_count=config.expected_count,
    )

    matrix = build_phase1_proposal_matrix(
        {"baseline": base, "candidate": proposal_source},
        source_text_by_document,
        source_metadata={
            "baseline": {
                "role": ProposalSourceRole.ENSEMBLE.value,
                "path": str(config.base),
                "sha256": config.expected_base_sha256,
            },
            "candidate": {
                "role": ProposalSourceRole.LLM.value,
                "path": str(config.proposal_source),
                "sha256": config.expected_proposal_source_sha256,
            },
        },
    )
    candidate_only = [
        row for row in matrix["matrix"] if row.get("sources") == ["candidate"]
    ]
    verifier_payload = json.loads(config.verifier_path.read_text(encoding="utf-8"))
    if not isinstance(verifier_payload, Mapping):
        raise ValueError("Proposal verifier artifact must be a JSON object")
    verifier = Phase1ProposalVerifier.from_dict(verifier_payload)
    scored = score_phase1_proposal_rows(
        candidate_only,
        source_text_by_document,
        verifier,
        source_roles={
            "baseline": ProposalSourceRole.ENSEMBLE,
            "candidate": ProposalSourceRole.LLM,
        },
    )
    output, decisions, counters = apply_verified_proposal_additions(
        base,
        scored,
        source_text_by_document=source_text_by_document,
    )

    run = create_hashed_run_dir(
        config.output_root,
        label=config.run_label,
        inputs=(
            config.documents_path,
            config.base,
            config.proposal_source,
            config.verifier_path,
            *config.dictionary_paths,
        ),
        resolved_config={
            "expected_source_archive_sha256": config.expected_source_archive_sha256,
            "expected_base_sha256": config.expected_base_sha256,
            "expected_proposal_source_sha256": config.expected_proposal_source_sha256,
            "expected_verifier_sha256": config.expected_verifier_sha256,
            "expected_count": config.expected_count,
            "application_policy": "additive_nonoverlapping",
            "new_entity_assertions": [],
            "new_entity_candidates": [],
        },
    )
    write_phase1_proposal_matrix(matrix, run.run_dir / "proposal_matrix")
    variant_dir = run.run_dir / "variants" / "E_CALIBRATED_PROPOSAL_ADD"
    output_dir = variant_dir / "output"
    _write_rows(output, output_dir)
    _validate_directory(output_dir, documents, dictionary)
    isolation_issues = validate_probe_isolation(base, output, module="entity")
    if isolation_issues:
        raise ValueError(f"Calibrated entity probe isolation failed: {isolation_issues[:5]}")
    zip_path = variant_dir / "output.zip"
    zip_phase1_output_dir(output_dir, zip_path)
    zip_issues = validate_phase1_submission_zip(
        zip_path,
        documents=documents,
        dictionary=dictionary,
        expected_count=config.expected_count,
    )
    if zip_issues:
        raise ValueError(
            "Calibrated entity probe ZIP validation failed: "
            f"{[issue.to_json() for issue in zip_issues[:5]]}"
        )
    write_jsonl(variant_dir / "decisions.jsonl", decisions)
    write_jsonl(
        variant_dir / "scores.jsonl",
        (item.to_dict() for item in scored),
    )
    report = {
        "schema_version": "phase1-round2-proposal-verifier-run.v1",
        "name": "E_CALIBRATED_PROPOSAL_ADD",
        "module": "entity",
        "baseline_entity_count": sum(len(rows) for rows in base.values()),
        "output_entity_count": sum(len(rows) for rows in output.values()),
        "proposal_matrix_group_count": len(matrix["matrix"]),
        "candidate_only_proposal_count": len(candidate_only),
        "counters": dict(sorted(counters.items())),
        "validation_issue_count": 0,
        "isolation_issues": [],
        "zip": str(zip_path),
        "zip_sha256": sha256_file(zip_path),
        "verifier": {
            "path": str(config.verifier_path),
            "sha256": config.expected_verifier_sha256,
            "training_dataset_sha256": verifier.training_dataset_sha256,
            "thresholds": verifier.threshold_by_type,
            "minimum_development_precision": (
                verifier.minimum_development_precision
            ),
        },
        "public_status": "pending",
    }
    write_json(variant_dir / "variant_report.json", report)
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    manifest["proposal_verifier_probe"] = report
    write_json(run.manifest_path, manifest)
    return {
        "run_id": run.run_id,
        "run_dir": str(run.run_dir),
        "run_manifest": str(run.manifest_path),
        "variant": report,
    }


def _entity_only_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "text": row.get("text"),
        "type": row.get("type"),
        "assertions": [],
        "candidates": [],
        "position": list(row.get("position", [])),
    }


def _rows_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_position = _position(left)
    right_position = _position(right)
    return min(left_position[1], right_position[1]) > max(
        left_position[0],
        right_position[0],
    )


def _position(row: Mapping[str, Any]) -> tuple[int, int]:
    value = row.get("position")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise ValueError("Phase 1 row has an invalid position")
    return value[0], value[1]


def _row_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str, str]:
    start, end = _position(row)
    return start, end, str(row.get("type", "")), str(row.get("text", ""))


def _verify_hash(path: Path, expected: str, *, name: str) -> None:
    observed = _path_sha256(path)
    if observed != expected:
        raise ValueError(
            f"Frozen {name} SHA-256 mismatch: expected {expected}, observed {observed}"
        )


def _path_sha256(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
    return digest.hexdigest()


def _load_dictionary(paths: Sequence[Path]) -> DictionaryStore:
    entries = []
    for path in paths:
        entries.extend(DictionaryStore.load_entries_jsonl(path))
    return DictionaryStore(entries)


def _validate_artifact(
    path: Path,
    *,
    documents: Sequence[ClinicalDocument],
    dictionary: DictionaryStore,
    expected_count: int,
) -> None:
    if path.is_file() and path.suffix.lower() == ".zip":
        issues = validate_phase1_submission_zip(
            path,
            documents=documents,
            dictionary=dictionary,
            expected_count=expected_count,
        )
    elif path.is_dir():
        issues = validate_phase1_submission_documents(
            documents,
            path,
            dictionary=dictionary,
        )
    else:
        raise ValueError(f"Unsupported Phase 1 artifact: {path}")
    if issues:
        raise ValueError(
            f"Frozen Round 2 baseline failed validation: "
            f"{[issue.to_json() for issue in issues[:5]]}"
        )


def _validate_directory(
    path: Path,
    documents: Sequence[ClinicalDocument],
    dictionary: DictionaryStore,
) -> None:
    issues = validate_phase1_submission_documents(
        documents,
        path,
        dictionary=dictionary,
    )
    if issues:
        raise ValueError(
            f"Calibrated proposal directory failed validation: "
            f"{[issue.to_json() for issue in issues[:5]]}"
        )


def _write_rows(
    rows_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for document_id in sorted(rows_by_document, key=_document_sort_key):
        rows = rows_by_document[document_id]
        (output_dir / f"{document_id}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)
