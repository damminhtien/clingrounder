from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from medical_kg_nlp.utils.io import read_jsonl, read_source_text
from medical_kg_nlp.utils.text import normalize_for_match


PHASE1_ENTITY_TYPES = (
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
)
_STRICT_NEGATIVE_CATEGORIES = frozenset(
    {
        "administrative",
        "anatomy_only",
        "coreference",
        "generic_or_vague",
        "normal_state",
        "procedure_or_device",
        "risk_factor_or_lifestyle",
        "test_target_only",
        "unspecified_medication",
    }
)
_UNSTABLE_CATEGORIES = frozenset(
    {"nested_or_duplicate", "noisy_text", "planned_or_future", "unstable_policy", "uncertain"}
)
_GENERIC_CONTEXT_ALIASES = frozenset(
    {
        "alt",
        "glucose",
        "huyết áp",
        "inr",
        "kali",
        "mri",
        "mạch",
        "ra máu",
        "sưng",
        "tăng",
        "x-quang",
        "yếu",
        "đau",
    }
)
_CONFLICT_FIELDS = (
    "conflict_id",
    "conflict_type",
    "severity",
    "normalized_text",
    "entity_types",
    "document_ids",
    "accepted_count",
    "rejected_count",
    "details",
    "recommended_action",
)


def compile_annotation_knowledge(
    *,
    gold_dir: str | Path,
    manifest_path: str | Path,
    strict_document_support: int = 2,
    document_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    gold_root = Path(gold_dir)
    manifest_file = Path(manifest_path)
    all_manifest_rows = read_jsonl(manifest_file)
    selected_document_ids = (
        {str(document_id) for document_id in document_ids} if document_ids is not None else None
    )
    manifest_rows = [
        row
        for row in all_manifest_rows
        if selected_document_ids is None or str(row.get("document_id", "")) in selected_document_ids
    ]
    conflicts: list[dict[str, Any]] = []
    accepted_mentions: list[dict[str, Any]] = []
    rejected_mentions: list[dict[str, Any]] = []
    guideline_rows: list[dict[str, Any]] = []
    seen_documents: set[str] = set()
    loaded_gold_paths: list[Path] = []

    for manifest in manifest_rows:
        document_id = str(manifest.get("document_id", "")).strip()
        if not document_id:
            _add_conflict(
                conflicts,
                conflict_type="manifest_missing_document_id",
                severity="blocking",
                details="Manifest row has no document_id.",
                recommended_action="repair_manifest_row",
            )
            continue
        if document_id in seen_documents:
            _add_conflict(
                conflicts,
                conflict_type="manifest_duplicate_document",
                severity="blocking",
                document_ids=[document_id],
                details="document_id appears more than once in review_manifest.jsonl.",
                recommended_action="deduplicate_manifest",
            )
            continue
        seen_documents.add(document_id)
        gold_path = _manifest_path(manifest.get("gold_file"), gold_root / f"{document_id}.json")
        source_path = _manifest_path(manifest.get("source_file"), Path("data/raw/input") / f"{document_id}.txt")
        if not gold_path.exists():
            _add_missing_file_conflict(conflicts, document_id, "gold", gold_path)
            continue
        if not source_path.exists():
            _add_missing_file_conflict(conflicts, document_id, "source", source_path)
            continue
        loaded_gold_paths.append(gold_path)
        source_text = read_source_text(source_path)
        gold_rows = _load_gold_rows(gold_path, document_id, conflicts)
        expected_count = manifest.get("entity_count")
        if isinstance(expected_count, int) and expected_count != len(gold_rows):
            _add_conflict(
                conflicts,
                conflict_type="manifest_entity_count_mismatch",
                severity="high",
                document_ids=[document_id],
                accepted_count=len(gold_rows),
                details=f"Manifest says {expected_count}; gold file contains {len(gold_rows)} entities.",
                recommended_action="review_manifest_and_gold",
            )
        _compile_gold_document(
            document_id,
            source_text,
            gold_rows,
            accepted_mentions,
            conflicts,
        )
        _compile_review_candidates(
            document_id,
            source_text,
            manifest.get("review_candidates"),
            rejected_mentions,
            conflicts,
        )
        _compile_guidelines(document_id, manifest.get("guideline_notes"), guideline_rows)

    positive_groups = _group_positive_mentions(accepted_mentions)
    negative_groups = _group_negative_mentions(rejected_mentions)
    _add_knowledge_conflicts(positive_groups, negative_groups, conflicts)
    conflicts = _finalize_conflicts(conflicts)
    unstable_mentions = {
        str(row["normalized_text"])
        for row in conflicts
        if row.get("normalized_text")
        and row["conflict_type"] in {"positive_negative_same_mention", "positive_type_disagreement"}
    }
    policy = _build_policy(
        positive_groups,
        negative_groups,
        unstable_mentions,
        strict_document_support=strict_document_support,
    )
    guidelines = _group_guidelines(guideline_rows)
    summary = _build_summary(
        manifest_rows=manifest_rows,
        loaded_gold_paths=loaded_gold_paths,
        accepted_mentions=accepted_mentions,
        rejected_mentions=rejected_mentions,
        guideline_rows=guideline_rows,
        policy=policy,
        conflicts=conflicts,
    )
    return {
        "schema_version": "phase1-annotation-knowledge.v1",
        "inputs": {
            "gold_dir": str(gold_root),
            "review_manifest": str(manifest_file),
            "review_manifest_sha256": _sha256_file(manifest_file),
            "gold_files_sha256": _sha256_paths(loaded_gold_paths),
            "selected_document_ids": sorted(seen_documents, key=_document_sort_key),
            "excluded_manifest_document_count": len(all_manifest_rows) - len(manifest_rows),
        },
        "compiler_config": {"strict_document_support": strict_document_support},
        "summary": summary,
        "policy": policy,
        "knowledge": {
            "accepted_mentions": _positive_group_rows(positive_groups),
            "rejected_mentions": _negative_group_rows(negative_groups),
            "guidelines": guidelines,
        },
        "conflicts": conflicts,
    }


def write_annotation_knowledge(report: Mapping[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "annotation_knowledge.json", report)
    _write_yaml(output / "phase1_annotation_policy.yaml", _mapping(report.get("policy")))
    conflicts = _dict_list(report.get("conflicts"))
    _write_conflicts_csv(output / "policy_conflicts.csv", conflicts)
    _write_json(output / "conflict_summary.json", _conflict_summary(conflicts))
    (output / "report.md").write_text(render_annotation_knowledge_markdown(report), encoding="utf-8")


def render_annotation_knowledge_markdown(report: Mapping[str, Any]) -> str:
    summary = _mapping(report.get("summary"))
    policy = _mapping(report.get("policy"))
    aliases = _mapping(policy.get("aliases"))
    strict_aliases = _mapping(aliases.get("strict"))
    context_aliases = _mapping(aliases.get("context_required"))
    conflicts = _dict_list(report.get("conflicts"))
    lines = [
        "# Phase 1 Annotation Knowledge Report",
        "",
        "## Summary",
        "",
        f"- Reviewed documents: {summary.get('reviewed_document_count', 0)}",
        f"- Accepted entities: {summary.get('accepted_entity_count', 0)}",
        f"- Review/rejected mentions: {summary.get('rejected_mention_count', 0)}",
        f"- Guideline notes: {summary.get('guideline_note_count', 0)}",
        f"- Strict runtime aliases: {summary.get('strict_alias_count', 0)}",
        f"- Context-required aliases: {summary.get('context_required_alias_count', 0)}",
        f"- Strict exclusions: {summary.get('strict_exclusion_count', 0)}",
        f"- Conflicts: {summary.get('conflict_count', 0)}",
        "",
        "Runtime policy contains concept-level rules only; document identifiers remain audit provenance and are not runtime selectors.",
        "",
        "## Strict Aliases",
        "",
    ]
    for entity_type in PHASE1_ENTITY_TYPES:
        lines.append(
            f"- `{entity_type}`: {len(_string_list(strict_aliases.get(entity_type)))} strict, "
            f"{len(_string_list(context_aliases.get(entity_type)))} context-required"
        )
    lines.extend(
        [
            "",
            "## Conflict Summary",
            "",
            "| Type | Severity | Count |",
            "| --- | --- | ---: |",
        ]
    )
    conflict_counts = Counter((str(row.get("conflict_type")), str(row.get("severity"))) for row in conflicts)
    for (conflict_type, severity), count in sorted(conflict_counts.items()):
        lines.append(f"| `{conflict_type}` | {severity} | {count} |")
    lines.extend(
        [
            "",
            "## Highest-Priority Conflicts",
            "",
            "| Severity | Type | Mention | Documents | Action |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    severity_rank = {"blocking": 0, "high": 1, "medium": 2, "warning": 3}
    for row in sorted(
        conflicts,
        key=lambda item: (
            severity_rank.get(str(item.get("severity")), 9),
            str(item.get("conflict_type")),
            str(item.get("normalized_text")),
        ),
    )[:50]:
        lines.append(
            "| {severity} | `{kind}` | {mention} | {documents} | `{action}` |".format(
                severity=row.get("severity", ""),
                kind=row.get("conflict_type", ""),
                mention=str(row.get("normalized_text", "")).replace("|", "\\|"),
                documents=str(row.get("document_ids", "")).replace("|", "\\|"),
                action=row.get("recommended_action", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _compile_gold_document(
    document_id: str,
    source_text: str,
    gold_rows: list[dict[str, Any]],
    accepted_mentions: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
) -> None:
    spans: list[tuple[int, int, str, str]] = []
    for index, row in enumerate(gold_rows):
        text = row.get("text")
        entity_type = row.get("type")
        position = row.get("position")
        if entity_type not in PHASE1_ENTITY_TYPES:
            _add_conflict(
                conflicts,
                conflict_type="gold_invalid_entity_type",
                severity="blocking",
                document_ids=[document_id],
                entity_types=[str(entity_type)],
                details=f"Gold row {index} has unsupported type {entity_type!r}.",
                recommended_action="repair_gold_type",
            )
            continue
        if not isinstance(text, str) or not text or not _valid_position(position):
            _add_conflict(
                conflicts,
                conflict_type="gold_invalid_span_schema",
                severity="blocking",
                document_ids=[document_id],
                entity_types=[str(entity_type)],
                details=f"Gold row {index} has invalid text or position.",
                recommended_action="repair_gold_span",
            )
            continue
        assert isinstance(position, list)
        start, end = position
        if start < 0 or end <= start or end > len(source_text) or source_text[start:end] != text:
            _add_conflict(
                conflicts,
                conflict_type="gold_offset_mismatch",
                severity="blocking",
                normalized_text=normalize_for_match(text),
                document_ids=[document_id],
                entity_types=[str(entity_type)],
                accepted_count=1,
                details=f"Gold row {index} does not satisfy source_text[start:end] == text.",
                recommended_action="repair_gold_offset",
            )
            continue
        normalized = normalize_for_match(text)
        if not normalized:
            continue
        accepted_mentions.append(
            {
                "document_id": document_id,
                "text": text,
                "normalized_text": normalized,
                "type": entity_type,
                "position": [start, end],
            }
        )
        spans.append((start, end, str(entity_type), text))
    _add_span_conflicts(document_id, spans, conflicts)


def _compile_review_candidates(
    document_id: str,
    source_text: str,
    value: Any,
    rejected_mentions: list[dict[str, Any]],
    conflicts: list[dict[str, Any]],
) -> None:
    candidates = value if isinstance(value, list) else []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            continue
        text = str(candidate.get("text", "")).strip()
        reason = str(candidate.get("reason", "")).strip()
        normalized = normalize_for_match(text)
        category = _classify_review_reason(reason)
        position = candidate.get("position")
        valid_position = position if _valid_position(position) and isinstance(position, list) else None
        if position is not None and not _valid_position(position):
            _add_conflict(
                conflicts,
                conflict_type="review_invalid_position",
                severity="medium",
                normalized_text=normalized,
                document_ids=[document_id],
                rejected_count=1,
                details=f"Review candidate {index} has invalid position schema.",
                recommended_action="repair_review_candidate_position",
            )
        elif valid_position is not None:
            start, end = valid_position
            if start < 0 or end <= start or end > len(source_text) or source_text[start:end] != text:
                _add_conflict(
                    conflicts,
                    conflict_type="review_offset_mismatch",
                    severity="medium",
                    normalized_text=normalized,
                    document_ids=[document_id],
                    rejected_count=1,
                    details=f"Review candidate {index} does not match its raw-text position.",
                    recommended_action="repair_or_null_review_position",
                )
        rejected_mentions.append(
            {
                "document_id": document_id,
                "text": text,
                "normalized_text": normalized,
                "position": list(valid_position) if valid_position is not None else None,
                "reason": reason,
                "reason_category": category,
                "policy_stability": "unstable" if category in _UNSTABLE_CATEGORIES else "stable",
            }
        )
        if category in _UNSTABLE_CATEGORIES:
            _add_conflict(
                conflicts,
                conflict_type="unstable_policy_evidence",
                severity="warning",
                normalized_text=normalized,
                document_ids=[document_id],
                rejected_count=1,
                details=reason,
                recommended_action="keep_out_of_automatic_runtime_policy",
            )


def _compile_guidelines(document_id: str, value: Any, guideline_rows: list[dict[str, Any]]) -> None:
    notes = value if isinstance(value, list) else []
    for note in notes:
        text = str(note).strip()
        if not text:
            continue
        guideline_rows.append(
            {
                "document_id": document_id,
                "text": text,
                "categories": _classify_guideline(text),
            }
        )


def _group_positive_mentions(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["normalized_text"]), str(row["type"]))
        group = groups.setdefault(
            key,
            {"normalized_text": key[0], "type": key[1], "surfaces": Counter(), "documents": set(), "count": 0},
        )
        group["surfaces"][str(row["text"])] += 1
        group["documents"].add(str(row["document_id"]))
        group["count"] += 1
    return groups


def _group_negative_mentions(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        normalized = str(row.get("normalized_text", ""))
        if not normalized:
            continue
        group = groups.setdefault(
            normalized,
            {
                "normalized_text": normalized,
                "surfaces": Counter(),
                "documents": set(),
                "categories": Counter(),
                "reasons": Counter(),
                "count": 0,
            },
        )
        group["surfaces"][str(row.get("text", ""))] += 1
        group["documents"].add(str(row.get("document_id", "")))
        group["categories"][str(row.get("reason_category", "other"))] += 1
        group["reasons"][str(row.get("reason", ""))] += 1
        group["count"] += 1
    return groups


def _add_knowledge_conflicts(
    positive_groups: Mapping[tuple[str, str], Mapping[str, Any]],
    negative_groups: Mapping[str, Mapping[str, Any]],
    conflicts: list[dict[str, Any]],
) -> None:
    types_by_text: dict[str, set[str]] = defaultdict(set)
    positive_count_by_text: Counter[str] = Counter()
    documents_by_text: dict[str, set[str]] = defaultdict(set)
    for (normalized, entity_type), group in positive_groups.items():
        types_by_text[normalized].add(entity_type)
        positive_count_by_text[normalized] += int(group["count"])
        documents_by_text[normalized].update(_string_set(group["documents"]))
    for normalized, entity_types in types_by_text.items():
        if len(entity_types) <= 1:
            continue
        _add_conflict(
            conflicts,
            conflict_type="positive_type_disagreement",
            severity="high",
            normalized_text=normalized,
            entity_types=sorted(entity_types),
            document_ids=sorted(documents_by_text[normalized], key=_document_sort_key),
            accepted_count=positive_count_by_text[normalized],
            details="The same normalized mention is accepted with multiple Phase 1 types.",
            recommended_action="require_context_or_resolve_type_policy",
        )
    for normalized in sorted(set(types_by_text) & set(negative_groups)):
        negative = negative_groups[normalized]
        documents = documents_by_text[normalized] | _string_set(negative["documents"])
        _add_conflict(
            conflicts,
            conflict_type="positive_negative_same_mention",
            severity="high",
            normalized_text=normalized,
            entity_types=sorted(types_by_text[normalized]),
            document_ids=sorted(documents, key=_document_sort_key),
            accepted_count=positive_count_by_text[normalized],
            rejected_count=int(negative["count"]),
            details="Mention is accepted in gold and also appears in review_candidates.",
            recommended_action="require_context_specific_rule",
        )


def _build_policy(
    positive_groups: Mapping[tuple[str, str], Mapping[str, Any]],
    negative_groups: Mapping[str, Mapping[str, Any]],
    unstable_mentions: set[str],
    *,
    strict_document_support: int,
) -> dict[str, Any]:
    strict_aliases: dict[str, list[str]] = {entity_type: [] for entity_type in PHASE1_ENTITY_TYPES}
    context_aliases: dict[str, list[str]] = {entity_type: [] for entity_type in PHASE1_ENTITY_TYPES}
    reviewed_aliases: dict[str, list[str]] = {entity_type: [] for entity_type in PHASE1_ENTITY_TYPES}
    types_by_text: dict[str, set[str]] = defaultdict(set)
    for normalized, entity_type in positive_groups:
        types_by_text[normalized].add(entity_type)
    for (normalized, entity_type), group in sorted(positive_groups.items()):
        if normalized in unstable_mentions or len(types_by_text[normalized]) > 1:
            continue
        document_support = len(_string_set(group["documents"]))
        occurrence_count = int(group["count"])
        if document_support >= strict_document_support:
            target = context_aliases if _alias_requires_context(normalized, entity_type) else strict_aliases
            target[entity_type].append(normalized)
        elif occurrence_count >= 2:
            reviewed_aliases[entity_type].append(normalized)

    strict_exclusions: dict[str, list[str]] = defaultdict(list)
    review_exclusions: dict[str, list[str]] = defaultdict(list)
    for normalized, group in sorted(negative_groups.items()):
        categories = set(_counter(group["categories"]))
        if normalized in types_by_text:
            continue
        strict_categories = categories & _STRICT_NEGATIVE_CATEGORIES
        if strict_categories:
            for category in sorted(strict_categories):
                strict_exclusions[category].append(normalized)
        else:
            for category in sorted(categories or {"other"}):
                review_exclusions[category].append(normalized)

    return {
        "schema_version": "phase1-annotation-policy.v1",
        "runtime_constraints": {
            "document_specific_rules": False,
            "preserve_raw_offsets": True,
            "emit_every_occurrence": True,
            "allow_overlapping_same_concept": False,
            "assertion_policy": "empty",
            "candidate_policy": "empty",
        },
        "aliases": {
            "strict": {key: sorted(values) for key, values in strict_aliases.items()},
            "context_required": {key: sorted(values) for key, values in context_aliases.items()},
            "reviewed": {key: sorted(values) for key, values in reviewed_aliases.items()},
        },
        "exclusions": {
            "strict": {key: sorted(set(values)) for key, values in sorted(strict_exclusions.items())},
            "review": {key: sorted(set(values)) for key, values in sorted(review_exclusions.items())},
        },
        "unstable_mentions": sorted(unstable_mentions),
        "boundary_rules": _boundary_rules(),
    }


def _boundary_rules() -> dict[str, Any]:
    return {
        "global": {
            "exclude_leading_context": ["được chẩn đoán", "tiền sử", "phủ nhận", "không có"],
            "stop_at": [";", "\n"],
            "split_coordinated_mentions": True,
            "prefer_non_overlapping_concepts": True,
        },
        "CHẨN_ĐOÁN": {
            "include_attached_modifiers": ["subtype", "site", "laterality", "severity", "chronicity"],
            "keep_lexical_modifiers": ["không đặc hiệu", "không ổn định", "không biến chứng"],
            "exclude": ["generic lesion", "coreference", "procedure wrapper"],
        },
        "THUỐC": {
            "include_attached_modifiers": ["strength", "dose form", "route", "frequency"],
            "stop_before": ["indication", "treatment verb", "unrelated temporal clause"],
        },
        "TRIỆU_CHỨNG": {
            "include_attached_modifiers": ["body site", "laterality", "severity"],
            "positive_inability_phrases": True,
            "exclude": ["anatomy only", "normal state", "generic symptom group"],
        },
        "TÊN_XÉT_NGHIỆM": {
            "include_attached_modifiers": ["modality", "body site", "contrast", "diagnostic target"],
            "exclude": ["treatment procedure", "generic evaluation", "result value"],
        },
        "KẾT_QUẢ_XÉT_NGHIỆM": {
            "include_attached_modifiers": ["unit", "qualifier", "descriptive finding"],
            "require_test_or_vital_anchor": True,
            "exclude": ["test name", "date", "dose", "age", "administrative number"],
        },
    }


def _positive_group_rows(groups: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (_, _), group in sorted(groups.items()):
        rows.append(
            {
                "normalized_text": group["normalized_text"],
                "type": group["type"],
                "occurrence_count": int(group["count"]),
                "document_support": len(_string_set(group["documents"])),
                "surface_forms": _counter_rows(group["surfaces"]),
                "supporting_documents": sorted(_string_set(group["documents"]), key=_document_sort_key),
            }
        )
    return rows


def _negative_group_rows(groups: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, group in sorted(groups.items()):
        rows.append(
            {
                "normalized_text": group["normalized_text"],
                "occurrence_count": int(group["count"]),
                "document_support": len(_string_set(group["documents"])),
                "surface_forms": _counter_rows(group["surfaces"]),
                "reason_categories": _counter_rows(group["categories"]),
                "reasons": _counter_rows(group["reasons"]),
                "supporting_documents": sorted(_string_set(group["documents"]), key=_document_sort_key),
            }
        )
    return rows


def _group_guidelines(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        text = str(row["text"])
        group = grouped.setdefault(text, {"text": text, "documents": set(), "categories": Counter()})
        group["documents"].add(str(row["document_id"]))
        for category in _string_list(row.get("categories")):
            group["categories"][category] += 1
    return [
        {
            "text": group["text"],
            "document_support": len(_string_set(group["documents"])),
            "categories": sorted(_counter(group["categories"])),
            "supporting_documents": sorted(_string_set(group["documents"]), key=_document_sort_key),
        }
        for _, group in sorted(grouped.items())
    ]


def _build_summary(
    *,
    manifest_rows: list[dict[str, Any]],
    loaded_gold_paths: list[Path],
    accepted_mentions: list[dict[str, Any]],
    rejected_mentions: list[dict[str, Any]],
    guideline_rows: list[dict[str, Any]],
    policy: Mapping[str, Any],
    conflicts: list[dict[str, Any]],
) -> dict[str, Any]:
    aliases = _mapping(policy.get("aliases"))
    strict_aliases = _mapping(aliases.get("strict"))
    context_aliases = _mapping(aliases.get("context_required"))
    exclusions = _mapping(policy.get("exclusions"))
    strict_exclusions = _mapping(exclusions.get("strict"))
    return {
        "manifest_document_count": len(manifest_rows),
        "reviewed_document_count": len(loaded_gold_paths),
        "accepted_entity_count": len(accepted_mentions),
        "accepted_entity_count_by_type": dict(sorted(Counter(row["type"] for row in accepted_mentions).items())),
        "rejected_mention_count": len(rejected_mentions),
        "guideline_note_count": len(guideline_rows),
        "strict_alias_count": sum(len(_string_list(value)) for value in strict_aliases.values()),
        "context_required_alias_count": sum(
            len(_string_list(value)) for value in context_aliases.values()
        ),
        "reviewed_alias_count": sum(
            len(_string_list(value)) for value in _mapping(aliases.get("reviewed")).values()
        ),
        "strict_exclusion_count": sum(len(_string_list(value)) for value in strict_exclusions.values()),
        "unstable_mention_count": len(_string_list(policy.get("unstable_mentions"))),
        "conflict_count": len(conflicts),
        "conflict_count_by_type": dict(sorted(Counter(row["conflict_type"] for row in conflicts).items())),
        "conflict_count_by_severity": dict(sorted(Counter(row["severity"] for row in conflicts).items())),
    }


def _add_span_conflicts(
    document_id: str,
    spans: list[tuple[int, int, str, str]],
    conflicts: list[dict[str, Any]],
) -> None:
    for index, left in enumerate(sorted(spans)):
        for right in sorted(spans)[index + 1 :]:
            if right[0] >= left[1]:
                break
            if left[:2] == right[:2] and left[2:] == right[2:]:
                conflict_type = "gold_duplicate_entity"
                severity = "high"
                action = "deduplicate_gold_entity"
            elif left[:2] == right[:2] and left[2] != right[2]:
                conflict_type = "gold_same_span_type_disagreement"
                severity = "blocking"
                action = "resolve_gold_type"
            else:
                conflict_type = "gold_overlapping_entities"
                severity = "medium"
                action = "review_non_overlapping_boundary_policy"
            _add_conflict(
                conflicts,
                conflict_type=conflict_type,
                severity=severity,
                normalized_text=normalize_for_match(left[3]),
                entity_types=sorted({left[2], right[2]}),
                document_ids=[document_id],
                accepted_count=2,
                details=f"Overlapping spans {left[:2]} {left[3]!r} and {right[:2]} {right[3]!r}.",
                recommended_action=action,
            )


def _classify_review_reason(reason: str) -> str:
    normalized = normalize_for_match(reason)
    rules = (
        ("procedure_or_device", ("procedure", "intervention", "surgical", "thủ thuật", "phẫu thuật", "can thiệp", "device", "stent", "catheter", "shunt")),
        ("unstable_policy", ("review-only", "review only", "until policy", "not yet stable", "guideline", "cần review", "review separately")),
        ("nested_or_duplicate", ("nested", "duplicate", "too short", "full phrase", "lặp", "boundary")),
        ("risk_factor_or_lifestyle", ("risk factor", "lifestyle", "substance", "smoking", "hút thuốc", "rượu", "caffeine", "diet")),
        ("coreference", ("coreferential", "reference phrase", "tổn thương này", "this lesion")),
        ("anatomy_only", ("anatomy", "anatomical", "giải phẫu")),
        ("normal_state", ("normal state", "trạng thái bình thường", "cảm thấy khỏe")),
        ("test_target_only", ("test target", "organism mention", "mục tiêu xét nghiệm")),
        ("planned_or_future", ("planned", "future test", "kế hoạch", "dự kiến")),
        ("noisy_text", ("noisy", "typo", "translation", "lỗi dịch", "unclear abbreviation", "không rõ")),
        ("unspecified_medication", ("without a specific drug", "not a specific drug", "không có tên thuốc", "unspecified drug")),
        ("administrative", ("logistics", "administrative", "hành chính")),
        ("generic_or_vague", ("generic", "vague", "not a specific", "too broad", "chung chung", "mơ hồ")),
        ("uncertain", ("uncertain", "not clearly", "could be", "may be", "không chắc")),
    )
    for category, keywords in rules:
        if any(keyword in normalized for keyword in keywords):
            return category
    return "other"


def _alias_requires_context(normalized: str, entity_type: str) -> bool:
    if entity_type == "KẾT_QUẢ_XÉT_NGHIỆM":
        return True
    if normalized in _GENERIC_CONTEXT_ALIASES:
        return True
    if entity_type == "TÊN_XÉT_NGHIỆM" and len(normalized.split()) == 1 and len(normalized) <= 5:
        return True
    return not any(character.isalpha() for character in normalized)


def _classify_guideline(note: str) -> list[str]:
    normalized = normalize_for_match(note)
    categories: list[str] = []
    checks = (
        ("boundary", ("span", "boundary", "split", "full phrase", "nested")),
        ("exclusion", ("do not annotate", "exclude", "remain review-only", "không lấy")),
        ("occurrence", ("occurrence", "repeated", "lặp")),
        ("assertion", ("ishistorical", "isnegated", "isfamily", "negation", "historical")),
        ("candidate", ("candidate", "icd", "rxnorm", "normalize")),
        ("lab_pairing", ("test", "result", "xét nghiệm", "kết quả")),
        ("type", tuple(normalize_for_match(value) for value in PHASE1_ENTITY_TYPES)),
        ("offset", ("offset", "raw text")),
    )
    for category, keywords in checks:
        if any(keyword in normalized for keyword in keywords):
            categories.append(category)
    return categories or ["general"]


def _load_gold_rows(path: Path, document_id: str, conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        _add_conflict(
            conflicts,
            conflict_type="gold_invalid_json",
            severity="blocking",
            document_ids=[document_id],
            details=str(error),
            recommended_action="repair_gold_json",
        )
        return []
    if not isinstance(payload, list):
        _add_conflict(
            conflicts,
            conflict_type="gold_invalid_root",
            severity="blocking",
            document_ids=[document_id],
            details="Gold file root must be a JSON list.",
            recommended_action="repair_gold_json",
        )
        return []
    return [row for row in payload if isinstance(row, dict)]


def _add_missing_file_conflict(conflicts: list[dict[str, Any]], document_id: str, role: str, path: Path) -> None:
    _add_conflict(
        conflicts,
        conflict_type=f"manifest_missing_{role}_file",
        severity="blocking",
        document_ids=[document_id],
        details=f"Missing {role} file: {path}",
        recommended_action=f"restore_{role}_file",
    )


def _add_conflict(
    conflicts: list[dict[str, Any]],
    *,
    conflict_type: str,
    severity: str,
    normalized_text: str = "",
    entity_types: Iterable[str] = (),
    document_ids: Iterable[str] = (),
    accepted_count: int = 0,
    rejected_count: int = 0,
    details: str,
    recommended_action: str,
) -> None:
    conflicts.append(
        {
            "conflict_type": conflict_type,
            "severity": severity,
            "normalized_text": normalized_text,
            "entity_types": sorted(set(entity_types)),
            "document_ids": sorted(set(document_ids), key=_document_sort_key),
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "details": details,
            "recommended_action": recommended_action,
        }
    )


def _finalize_conflicts(conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for conflict in conflicts:
        payload = json.dumps(conflict, ensure_ascii=False, sort_keys=True)
        conflict_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        row = {"conflict_id": conflict_id, **conflict}
        unique[conflict_id] = row
    return sorted(
        unique.values(),
        key=lambda row: (
            str(row["conflict_type"]),
            str(row["normalized_text"]),
            str(row["document_ids"]),
        ),
    )


def _conflict_summary(conflicts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "conflict_count": len(conflicts),
        "by_type": dict(sorted(Counter(row["conflict_type"] for row in conflicts).items())),
        "by_severity": dict(sorted(Counter(row["severity"] for row in conflicts).items())),
        "blocking": [row for row in conflicts if row["severity"] == "blocking"],
    }


def _write_conflicts_csv(path: Path, conflicts: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CONFLICT_FIELDS)
        writer.writeheader()
        for row in conflicts:
            writer.writerow(
                {
                    field: json.dumps(row.get(field), ensure_ascii=False)
                    if field in {"entity_types", "document_ids"}
                    else row.get(field, "")
                    for field in _CONFLICT_FIELDS
                }
            )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, allow_unicode=True, sort_keys=False)


def _counter_rows(value: Any) -> list[dict[str, Any]]:
    counter = _counter(value)
    return [{"value": key, "count": count} for key, count in counter.most_common()]


def _counter(value: Any) -> Counter[str]:
    return value if isinstance(value, Counter) else Counter()


def _manifest_path(value: Any, fallback: Path) -> Path:
    if isinstance(value, str) and value.strip():
        return Path(value)
    return fallback


def _valid_position(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(isinstance(item, int) for item in value)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_paths(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=str):
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return []


def _string_set(value: Any) -> set[str]:
    if isinstance(value, set):
        return {str(item) for item in value}
    return set(_string_list(value))


def _document_sort_key(document_id: str) -> tuple[int, int | str]:
    return (0, int(document_id)) if document_id.isdigit() else (1, document_id)
