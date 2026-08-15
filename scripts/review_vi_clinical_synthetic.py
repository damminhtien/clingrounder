#!/usr/bin/env python3
"""Run an independent technical review of the synthetic Vietnamese benchmark.

This command is intentionally separate from the human review-pack workflow.  It checks the
published synthetic contract from the outside: raw offsets, schema-level invariants, terminology
membership for the fixture concepts, relation endpoints, and the expected semantics of each test
template.  It does not promote labels to clinical evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = ["review_synthetic_snapshot"]


_EXPECTED_DATASET_ID = "vi-clinical-grounding-synthetic-v1"
_EXPECTED_TYPES = frozenset({"DISEASE", "SYMPTOM", "DRUG", "LAB_TEST", "LAB_RESULT"})
_EXPECTED_ASSERTIONS = frozenset({"PRESENT", "NEGATED", "HISTORICAL", "FAMILY", "POSSIBLE"})
_EXPECTED_CODE_SYSTEMS = frozenset({"ICD-10", "LOCAL", "NONE", "RxNorm"})

# This is a small, explicit fixture contract, not a substitute for a clinical terminology
# release.  Keeping it here makes the review independent from the generator's CONCEPTS tuple.
_KNOWN_CODES: Mapping[tuple[str, str], tuple[str, str]] = {
    ("LOCAL", "SYMPTOM_FEVER"): ("SYMPTOM", "sốt"),
    ("LOCAL", "SYMPTOM_COUGH"): ("SYMPTOM", "ho"),
    ("LOCAL", "SYMPTOM_DYSPNEA"): ("SYMPTOM", "khó thở"),
    ("LOCAL", "SYMPTOM_CHEST_PAIN"): ("SYMPTOM", "đau ngực"),
    ("LOCAL", "SYMPTOM_HEADACHE"): ("SYMPTOM", "đau đầu"),
    ("LOCAL", "SYMPTOM_NAUSEA_VOMITING"): ("SYMPTOM", "buồn nôn và nôn"),
    ("ICD-10", "I10"): ("DISEASE", "tăng huyết áp"),
    ("ICD-10", "E11"): ("DISEASE", "đái tháo đường type 2"),
    ("ICD-10", "J18.9"): ("DISEASE", "viêm phổi"),
    ("ICD-10", "J45"): ("DISEASE", "hen phế quản"),
    ("ICD-10", "C34"): ("DISEASE", "ung thư phổi"),
    ("ICD-10", "I21.9"): ("DISEASE", "nhồi máu cơ tim cấp"),
    ("RxNorm", "6809"): ("DRUG", "metformin"),
    ("RxNorm", "435"): ("DRUG", "salbutamol"),
    ("RxNorm", "1191"): ("DRUG", "aspirin"),
    ("RxNorm", "723"): ("DRUG", "amoxicillin"),
    ("RxNorm", "83367"): ("DRUG", "atorvastatin"),
    ("RxNorm", "7646"): ("DRUG", "omeprazole"),
    ("LOCAL", "GLUCOSE"): ("LAB_TEST", "đường huyết"),
    ("LOCAL", "CREATININE"): ("LAB_TEST", "creatinin"),
    ("LOCAL", "HBA1C"): ("LAB_TEST", "HbA1c"),
    ("LOCAL", "ECG"): ("LAB_TEST", "điện tâm đồ"),
}

_TEMPLATE_CONTRACT: Mapping[str, tuple[tuple[tuple[str, str], ...], tuple[str, ...], tuple[str, ...]]] = {
    "test.question": (
        (("SYMPTOM", "PRESENT"), ("SYMPTOM", "PRESENT")),
        ("hỏi:", "đáp:"),
        (),
    ),
    "test.repeated": (
        (("SYMPTOM", "NEGATED"), ("SYMPTOM", "PRESENT")),
        ("ban đầu không", "xuất hiện"),
        (),
    ),
    "test.mixed": (
        (("DISEASE", "PRESENT"), ("DRUG", "PRESENT"), ("SYMPTOM", "PRESENT")),
        ("clinical note:", "symptom:"),
        (),
    ),
    "test.possible": (
        (("DISEASE", "POSSIBLE"), ("SYMPTOM", "PRESENT")),
        ("có thể mắc",),
        (),
    ),
    "test.family": (
        (("SYMPTOM", "PRESENT"), ("DISEASE", "FAMILY")),
        ("chị gái", "tiền sử"),
        (),
    ),
    "test.medication": (
        (("DRUG", "PRESENT"), ("SYMPTOM", "PRESENT")),
        ("danh sách thuốc", "chỉ định"),
        (),
    ),
    "test.lab": (
        (("LAB_TEST", "PRESENT"), ("LAB_RESULT", "PRESENT")),
        ("bảng xét nghiệm",),
        ("HAS_VALUE",),
    ),
}


def review_synthetic_snapshot(
    benchmark_dir: str | Path,
    *,
    split: str = "test",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Review one synthetic split and optionally write a PHI-safe JSON report.

    The report deliberately records document IDs and failure codes only.  It never copies raw
    text or mention strings, and its ``eligible_for_clinical_claim`` field is always false.
    """

    root = Path(benchmark_dir).expanduser().resolve()
    manifest_path = root / "dataset_manifest.yaml"
    manifest = _load_mapping(manifest_path)
    dataset = _require_mapping(manifest, "dataset")
    if dataset.get("id") != _EXPECTED_DATASET_ID:
        raise ValueError(f"unexpected synthetic dataset id: {dataset.get('id')!r}")
    dataset_status = str(dataset.get("status", "")).casefold()
    if dataset.get("synthetic") is not True and not dataset_status.startswith("synthetic"):
        raise ValueError("technical synthetic review requires dataset.synthetic=true")

    split_payload = _require_mapping(_require_mapping(manifest, "splits"), split)
    split_path = (root / str(split_payload["path"])).resolve()
    if root not in split_path.parents:
        raise ValueError("split path escapes benchmark directory")
    rows = _load_jsonl(split_path)

    failures: list[dict[str, Any]] = []
    template_counts: Counter[str] = Counter()
    assertion_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    document_ids: set[str] = set()
    for row in rows:
        document_id = str(row.get("document_id", ""))
        errors = _review_document(row, manifest)
        if document_id in document_ids:
            errors.append("duplicate_document_id")
        document_ids.add(document_id)
        metadata = row.get("metadata")
        if isinstance(metadata, Mapping):
            template_counts[str(metadata.get("template_group", ""))] += 1
        entities = row.get("entities")
        if isinstance(entities, list):
            for entity in entities:
                if isinstance(entity, Mapping):
                    type_counts[str(entity.get("type", ""))] += 1
                    assertion_counts[str(entity.get("assertion", ""))] += 1
        relations = row.get("relations")
        if isinstance(relations, list):
            for relation in relations:
                if isinstance(relation, Mapping):
                    relation_counts[str(relation.get("type", ""))] += 1
        if errors:
            failures.append({"document_id": document_id, "checks": sorted(set(errors))})

    expected_templates = set(_TEMPLATE_CONTRACT)
    checks = {
        "dataset_is_synthetic": dataset.get("synthetic") is True
        or dataset_status.startswith("synthetic"),
        "documents_are_unique": len(document_ids) == len(rows),
        "all_documents_pass_contract": not failures,
        "all_expected_templates_present": set(template_counts) == expected_templates,
        "split_count_matches_manifest": len(rows) == split_payload.get("documents"),
        "manifest_entity_taxonomy_matches": set(manifest.get("entities", [])) == _EXPECTED_TYPES,
        "manifest_assertion_taxonomy_matches": set(manifest.get("assertions", []))
        == _EXPECTED_ASSERTIONS,
        "manifest_code_systems_match": set(manifest.get("code_systems", []))
        == _EXPECTED_CODE_SYSTEMS,
    }
    report: dict[str, Any] = {
        "schema_version": "clingrounder.synthetic-technical-review.v1",
        "reviewer": "codex",
        "review_kind": "template_and_invariant_review",
        "human_clinical_review": False,
        "eligible_for_engineering_use": not failures and all(checks.values()),
        "eligible_for_clinical_claim": False,
        "dataset": {
            "id": str(dataset.get("id", "")),
            "version": str(dataset.get("version", "")),
            "status": str(dataset.get("status", "")),
        },
        "split": split,
        "source_manifest_sha256": _sha256(manifest_path),
        "source_split_sha256": _sha256(split_path),
        "documents": {
            "reviewed": len(rows),
            "passed": len(rows) - len(failures),
            "failed": len(failures),
            "by_template": dict(sorted(template_counts.items())),
        },
        "counts": {
            "entities_by_type": dict(sorted(type_counts.items())),
            "assertions": dict(sorted(assertion_counts.items())),
            "relations": dict(sorted(relation_counts.items())),
        },
        "checks": checks,
        "failures": failures,
        "status": "technical_review_pass" if not failures and all(checks.values()) else "technical_review_failed",
        "limitations": [
            "This is a Codex technical consistency review, not human clinical annotation.",
            "The synthetic source is not clinical evidence and remains ineligible for clinical claims.",
            "Semantic checks are limited to the declared synthetic template contract.",
        ],
    }
    if output_path is not None:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def _review_document(row: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[str]:
    """Validate one document without emitting its text in the report."""

    errors: list[str] = []
    text = row.get("text")
    if not isinstance(text, str) or not text:
        return ["missing_text"]
    metadata = row.get("metadata")
    template = metadata.get("template_group") if isinstance(metadata, Mapping) else None
    contract = _TEMPLATE_CONTRACT.get(str(template))
    if contract is None:
        errors.append("unknown_template_group")
    entities = row.get("entities")
    if not isinstance(entities, list):
        return errors + ["entities_not_list"]
    seen_entity_ids: set[str] = set()
    observed_contract: list[tuple[str, str]] = []
    for entity in entities:
        if not isinstance(entity, Mapping):
            errors.append("entity_not_object")
            continue
        entity_id = str(entity.get("id", ""))
        if not entity_id or entity_id in seen_entity_ids:
            errors.append("duplicate_or_empty_entity_id")
        seen_entity_ids.add(entity_id)
        span = entity.get("span")
        entity_text = entity.get("text")
        if (
            not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in span)
            or not isinstance(entity_text, str)
        ):
            errors.append("invalid_span_shape")
            continue
        start, end = span
        if not 0 <= start < end <= len(text) or text[start:end] != entity_text:
            errors.append("offset_text_mismatch")
        entity_type = str(entity.get("type", ""))
        assertion = str(entity.get("assertion", ""))
        if entity_type not in _EXPECTED_TYPES:
            errors.append("unknown_entity_type")
        if assertion not in _EXPECTED_ASSERTIONS:
            errors.append("unknown_assertion")
        observed_contract.append((entity_type, assertion))
        errors.extend(_check_code(entity, entity_type, entity_text))

    if contract is not None:
        expected_entities, cues, relation_types = contract
        if tuple(observed_contract) != expected_entities:
            errors.append("template_entity_contract_mismatch")
        lowered = text.casefold()
        if any(cue.casefold() not in lowered for cue in cues):
            errors.append("template_context_cue_missing")
        relations = row.get("relations")
        observed_relations = tuple(
            str(relation.get("type", ""))
            for relation in relations
            if isinstance(relation, Mapping)
        ) if isinstance(relations, list) else ()
        if observed_relations != relation_types:
            errors.append("template_relation_contract_mismatch")
    entity_types = {
        str(entity.get("id", "")): str(entity.get("type", ""))
        for entity in entities
        if isinstance(entity, Mapping)
    }
    errors.extend(_check_relations(row, entity_types))
    return errors


def _check_code(entity: Mapping[str, Any], entity_type: str, entity_text: str) -> list[str]:
    """Check fixture terminology membership and type/text ownership."""

    code_system = str(entity.get("code_system", ""))
    code = entity.get("code")
    if code_system == "NONE":
        if entity_type != "LAB_RESULT":
            return ["none_code_system_wrong_entity_type"]
        return [] if code is None else ["none_code_system_has_code"]
    if not isinstance(code, str) or not code.strip():
        return ["missing_assigned_code"]
    concept = _KNOWN_CODES.get((code_system, code))
    if concept is None:
        return ["unknown_fixture_code"]
    expected_type, expected_text = concept
    errors: list[str] = []
    if expected_type != entity_type:
        errors.append("code_type_mismatch")
    if expected_text != entity_text:
        errors.append("code_text_mismatch")
    return errors


def _check_relations(row: Mapping[str, Any], entity_types: Mapping[str, str]) -> list[str]:
    relations = row.get("relations")
    if not isinstance(relations, list):
        return ["relations_not_list"]
    errors: list[str] = []
    seen: set[str] = set()
    for relation in relations:
        if not isinstance(relation, Mapping):
            errors.append("relation_not_object")
            continue
        relation_id = str(relation.get("id", ""))
        if not relation_id or relation_id in seen:
            errors.append("duplicate_or_empty_relation_id")
        seen.add(relation_id)
        head = str(relation.get("head", ""))
        tail = str(relation.get("tail", ""))
        if head not in entity_types or tail not in entity_types:
            errors.append("relation_endpoint_missing")
        if head == tail:
            errors.append("relation_self_loop")
        if str(relation.get("type", "")) != "HAS_VALUE":
            errors.append("unknown_relation_type")
        elif entity_types.get(head) != "LAB_TEST" or entity_types.get(tail) != "LAB_RESULT":
            errors.append("has_value_endpoint_type_mismatch")
    return errors


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping: {path}")
    return payload


def _require_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Read one mapping field and fail with a useful contract error."""

    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping field {key!r}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = review_synthetic_snapshot(args.benchmark, split=args.split, output_path=args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "technical_review_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
