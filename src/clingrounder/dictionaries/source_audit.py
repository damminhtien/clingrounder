from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from clingrounder.dictionaries.rxnorm_sources import (
    RXNORM_DEFAULT_TTYS,
    RXNORM_FULL_FALLBACK_TTYS,
    profile_rxnorm_release,
)
from clingrounder.utils.io import read_jsonl, read_yaml, write_jsonl
from clingrounder.utils.text import normalize_for_match

_SOURCE_REQUIRED_FIELDS = ("id", "name", "category", "access", "url", "license", "use")
_SOURCE_VERSION_FIELDS = ("version", "release_date", "issued_date", "effective_date")
_TEXT_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹ]+(?:[-_/][0-9A-Za-zÀ-ỹ]+)*")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "bệnh",
        "bệnh nhân",
        "by",
        "các",
        "cho",
        "chứng",
        "có",
        "của",
        "đã",
        "được",
        "for",
        "hiện",
        "hiện tại",
        "in",
        "is",
        "kết",
        "kết quả",
        "khi",
        "không",
        "là",
        "một",
        "nhân",
        "nhập",
        "nhập viện",
        "ngày",
        "người",
        "of",
        "on",
        "phải",
        "quả",
        "sử",
        "tại",
        "the",
        "thấy",
        "thuật",
        "tiền",
        "tiền sử",
        "to",
        "trước",
        "trong",
        "triệu",
        "triệu chứng",
        "tính",
        "và",
        "viện",
        "với",
    }
)
_LAB_METABOLITE_RXNORM_ALIASES = frozenset(
    normalize_for_match(alias)
    for alias in (
        "alanine",
        "aspartate",
        "bilirubin",
        "cholesterol",
        "creatinine",
        "glucose",
        "lactate",
        "lipase",
        "potassium",
        "sodium",
        "succinate",
        "urea",
    )
)
_FALSE_POSITIVE_ACTIONS = {
    "too_short": "review_block_alias_or_require_type_context",
    "ascii_single_token_icd_alias": "review_block_alias_or_require_disease_context",
    "lab_or_metabolite_rxnorm_alias_requires_drug_context": "block_unless_explicit_drug_context",
}
_FALSE_POSITIVE_SEVERITIES = {
    "too_short": "medium",
    "ascii_single_token_icd_alias": "medium",
    "lab_or_metabolite_rxnorm_alias_requires_drug_context": "high",
}
_MEDICAL_CUE_TOKENS = frozenset(
    {
        "áp",
        "bụng",
        "chóng",
        "chụp",
        "đau",
        "điện",
        "động",
        "đường",
        "gan",
        "giảm",
        "hct",
        "huyết",
        "kali",
        "men",
        "mạch",
        "máu",
        "ngực",
        "nhiễm",
        "phẫu",
        "phổi",
        "sốt",
        "suy",
        "thận",
        "tim",
        "tăng",
        "thuốc",
        "thủ",
        "tiểu",
        "trị",
        "viêm",
        "xét",
    }
)


def build_source_audit_report(
    *,
    registry_path: str | Path,
    standard_versions_path: str | Path | None = None,
    dictionary_paths: Sequence[str | Path] = (),
    local_files: Sequence[Mapping[str, Any]] = (),
    rxnorm_release_paths: Sequence[str | Path | Mapping[str, Any]] = (),
    input_dir: str | Path | None = None,
    unknown_top_k: int = 100,
) -> dict[str, Any]:
    registry = read_yaml(registry_path)
    source_ids = _registry_source_ids(registry)
    source_issues = _source_registry_issues(registry)
    file_manifest = [_local_file_manifest(item) for item in _registry_local_files(registry, local_files)]
    dictionary_profiles = [
        profile_dictionary(path, known_source_ids=source_ids, input_dir=input_dir, unknown_top_k=unknown_top_k)
        for path in dictionary_paths
    ]
    rxnorm_profiles = []
    for release in rxnorm_release_paths:
        if isinstance(release, Mapping):
            content = str(release.get("content") or "prescribable").lower()
            if content not in {"full", "prescribable"}:
                raise ValueError(f"Unsupported RxNorm release content: {content!r}")
            profile = profile_rxnorm_release(
                release["path"],
                archive_member_root=str(release.get("archive_member_root") or "") or None,
                allowed_ttys=RXNORM_FULL_FALLBACK_TTYS if content == "full" else RXNORM_DEFAULT_TTYS,
            )
            profile["content"] = content
            rxnorm_profiles.append(profile)
        else:
            rxnorm_profiles.append(profile_rxnorm_release(release))
    standard_versions = _standard_versions_summary(standard_versions_path) if standard_versions_path else {}
    manual_review_queue = _manual_review_queue(dictionary_profiles)
    false_positive_blocklist = false_positive_blocklist_candidates(dictionary_profiles)
    summary = _audit_summary(
        source_issues,
        file_manifest,
        dictionary_profiles,
        rxnorm_profiles,
        manual_review_queue,
        false_positive_blocklist,
    )
    summary["registry_resource_count"] = len(source_ids)
    return {
        "schema_version": "medical-source-audit.v1",
        "registry": {
            "path": str(registry_path),
            "resource_count": len(source_ids),
            "source_ids": sorted(source_ids),
            "resources": _registry_resource_details(registry),
            "issues": source_issues,
        },
        "standard_versions": standard_versions,
        "files": file_manifest,
        "dictionaries": dictionary_profiles,
        "rxnorm_release_profiles": rxnorm_profiles,
        "manual_review_queue": manual_review_queue,
        "false_positive_blocklist": false_positive_blocklist,
        "summary": summary,
    }


def profile_dictionary(
    path: str | Path,
    *,
    known_source_ids: set[str] | None = None,
    input_dir: str | Path | None = None,
    unknown_top_k: int = 100,
) -> dict[str, Any]:
    rows = read_jsonl(path)
    concept_ids = [str(row.get("concept_id", "")) for row in rows]
    code_keys = [(str(row.get("code_system", "")), str(row.get("code", ""))) for row in rows]
    alias_entries: dict[str, list[dict[str, str]]] = defaultdict(list)
    broad_aliases: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()
    missing_source_rows = 0
    unknown_sources: Counter[str] = Counter()
    blocked_alias_count = 0
    for row in rows:
        row_sources = _row_source_ids(row)
        if not row_sources:
            missing_source_rows += 1
        for source_id in row_sources:
            source_counts[source_id] += 1
            if known_source_ids is not None and source_id not in known_source_ids:
                unknown_sources[source_id] += 1
        blocked_alias_count += len(_string_values(row.get("blocked_aliases")))
        for alias in _row_aliases(row):
            normalized = normalize_for_match(alias)
            if not normalized:
                continue
            alias_entries[normalized].append(
                {
                    "alias": alias,
                    "concept_id": str(row.get("concept_id", "")),
                    "code": str(row.get("code", "")),
                    "code_system": str(row.get("code_system", "")),
                    "semantic_type": str(row.get("semantic_type", "")),
                }
            )
            broad_reason = _broad_alias_reason(row, alias, normalized)
            if broad_reason is not None:
                broad_aliases.append(
                    {
                        "concept_id": str(row.get("concept_id", "")),
                        "alias": alias,
                        "normalized_alias": normalized,
                        "reason": broad_reason,
                    }
                )
    ambiguous_aliases = _ambiguous_aliases(alias_entries)
    unknown_mentions = (
        unknown_mention_candidates(input_dir, rows, top_k=unknown_top_k) if input_dir is not None else []
    )
    return {
        "path": str(path),
        "row_count": len(rows),
        "by_code_system": _count_rows(rows, "code_system"),
        "by_semantic_type": _count_rows(rows, "semantic_type"),
        "icd10_hierarchy": _icd10_hierarchy_summary(rows),
        "rxnorm_enrichment": _rxnorm_enrichment_summary(rows),
        "by_source": dict(sorted(source_counts.items())),
        "missing_source_rows": missing_source_rows,
        "unknown_source_ids": dict(sorted(unknown_sources.items())),
        "unique_concept_ids": len(set(concept_ids)),
        "duplicate_concept_ids": _duplicates(concept_ids),
        "unique_codes": len(set(code_keys)),
        "duplicate_codes": [
            {"code_system": code_system, "code": code, "count": count}
            for (code_system, code), count in _duplicate_counter(code_keys).items()
            if code_system and code
        ],
        "alias_count": sum(len(entries) for entries in alias_entries.values()),
        "unique_alias_count": len(alias_entries),
        "ambiguous_aliases": ambiguous_aliases[:100],
        "ambiguous_alias_count": len(ambiguous_aliases),
        "blocked_alias_count": blocked_alias_count,
        "broad_aliases": broad_aliases[:100],
        "broad_alias_count": len(broad_aliases),
        "unknown_mention_candidates": unknown_mentions,
    }


def write_source_audit_report(report: Mapping[str, Any], output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "source_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (path / "dictionary_coverage.md").write_text(render_dictionary_coverage_markdown(report), encoding="utf-8")
    write_jsonl(path / "manual_review_queue.jsonl", [dict(row) for row in report.get("manual_review_queue", [])])
    write_jsonl(
        path / "false_positive_blocklist.jsonl",
        [dict(row) for row in report.get("false_positive_blocklist", [])],
    )
    (path / "false_positive_blocklist.md").write_text(render_false_positive_blocklist_markdown(report), encoding="utf-8")


def render_dictionary_coverage_markdown(report: Mapping[str, Any]) -> str:
    summary = _mapping(report.get("summary"))
    lines = [
        "# Medical Source Audit",
        "",
        f"- Registry resources: {summary.get('registry_resource_count', 0)}",
        f"- Registry issues: {summary.get('registry_issue_count', 0)}",
        f"- Missing required local files: {summary.get('missing_required_file_count', 0)}",
        f"- Dictionary profiles: {summary.get('dictionary_count', 0)}",
        f"- Manual review items: {summary.get('manual_review_item_count', 0)}",
        f"- False-positive blocklist candidates: {summary.get('false_positive_blocklist_count', 0)}",
        "",
        "## Source Registry",
        "",
    ]
    registry = _mapping(report.get("registry"))
    for resource in _dict_list(registry.get("resources")):
        version = resource.get("version") or resource.get("release_date") or resource.get("issued_date") or "unversioned"
        lines.append(
            f"- `{resource.get('id')}`: {resource.get('name')} | access={resource.get('access')} | "
            f"version={version} | license={resource.get('license')} | use={resource.get('use')}"
        )
    lines.extend(["", "## Files", ""])
    for item in _dict_list(report.get("files")):
        status = "ok" if item.get("exists") else "missing"
        required = "required" if item.get("required") else "optional"
        lines.append(f"- `{item.get('source_id')}` `{item.get('path')}`: {status}, {required}")
    lines.extend(["", "## Dictionaries", ""])
    for profile in _dict_list(report.get("dictionaries")):
        lines.extend(
            [
                f"### `{profile.get('path')}`",
                "",
                f"- Rows: {profile.get('row_count', 0)}",
                f"- Code systems: {profile.get('by_code_system', {})}",
                f"- Semantic types: {profile.get('by_semantic_type', {})}",
                f"- ICD hierarchy: {profile.get('icd10_hierarchy', {})}",
                f"- RxNorm enrichment: {profile.get('rxnorm_enrichment', {})}",
                f"- Sources: {profile.get('by_source', {})}",
                f"- Ambiguous aliases: {profile.get('ambiguous_alias_count', 0)}",
                f"- Broad/blocked review aliases: {profile.get('broad_alias_count', 0)}",
                f"- Missing source rows: {profile.get('missing_source_rows', 0)}",
                "",
            ]
        )
    rxnorm_profiles = _dict_list(report.get("rxnorm_release_profiles"))
    if rxnorm_profiles:
        lines.extend(["## RxNorm Releases", ""])
        for profile in rxnorm_profiles:
            conso = _mapping(profile.get("rxnconso"))
            rel = _mapping(profile.get("rxnrel"))
            sat = _mapping(profile.get("rxnsat"))
            lines.extend(
                [
                    f"### `{profile.get('path')}`",
                    "",
                    f"- Required files: {profile.get('required_files', {})}",
                    f"- RXNCONSO active concepts: {conso.get('active_concepts', 0)}",
                    f"- RXNCONSO accepted concepts: {conso.get('accepted_concepts', 0)}",
                    f"- RXNREL active rows: {rel.get('active_rows', 0)}",
                    f"- RXNSAT active rows: {sat.get('active_rows', 0)}",
                    "",
                ]
            )
    lines.extend(["## Top Manual Review Items", ""])
    for item in _dict_list(report.get("manual_review_queue"))[:30]:
        lines.append(
            f"- {item.get('severity')} `{item.get('issue_type')}` "
            f"{item.get('concept_id', '')} {item.get('alias', item.get('term', ''))}: {item.get('notes', '')}"
        )
    return "\n".join(lines) + "\n"


def render_false_positive_blocklist_markdown(report: Mapping[str, Any]) -> str:
    rows = _dict_list(report.get("false_positive_blocklist"))
    reason_counts = Counter(str(row.get("reason", "<missing>")) for row in rows)
    lines = [
        "# False-Positive Blocklist Candidates",
        "",
        f"- Total candidates: {len(rows)}",
        f"- By reason: {dict(sorted(reason_counts.items()))}",
        "",
        "These rows are review candidates. They are not automatically applied to the runtime dictionary.",
        "",
        "## Top Candidates",
        "",
    ]
    for row in rows[:100]:
        lines.append(
            f"- {row.get('severity')} `{row.get('reason')}` `{row.get('alias')}` -> "
            f"{row.get('concept_id')} action={row.get('recommended_action')}"
        )
    return "\n".join(lines) + "\n"


def false_positive_blocklist_candidates(dictionary_profiles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    dictionaries_by_key: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for profile in dictionary_profiles:
        dictionary_path = str(profile.get("path", ""))
        for alias in _dict_list(profile.get("broad_aliases")):
            reason = str(alias.get("reason", ""))
            key = (
                str(alias.get("concept_id", "")),
                str(alias.get("normalized_alias", "")),
                reason,
            )
            dictionaries_by_key[key].add(dictionary_path)
            grouped.setdefault(
                key,
                {
                    "severity": _FALSE_POSITIVE_SEVERITIES.get(reason, "medium"),
                    "issue_type": "false_positive_alias_candidate",
                    "dictionary": dictionary_path,
                    "concept_id": str(alias.get("concept_id", "")),
                    "alias": str(alias.get("alias", "")),
                    "normalized_alias": str(alias.get("normalized_alias", "")),
                    "reason": reason,
                    "recommended_action": _FALSE_POSITIVE_ACTIONS.get(reason, "review"),
                    "notes": _false_positive_notes(reason),
                },
            )
    rows = []
    for key, row in grouped.items():
        dictionaries = sorted(dictionaries_by_key[key])
        row = dict(row)
        row["dictionary"] = dictionaries[0] if dictionaries else row.get("dictionary", "")
        row["dictionaries"] = dictionaries
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            0 if row["severity"] == "high" else 1,
            str(row["reason"]),
            str(row["normalized_alias"]),
            str(row["concept_id"]),
        ),
    )


def file_fingerprints(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    exists = input_path.exists()
    result: dict[str, Any] = {
        "path": str(input_path),
        "exists": exists,
    }
    if not exists:
        return result
    md5 = hashlib.md5()  # noqa: S324 - checksum for reproducibility, not security.
    sha256 = hashlib.sha256()
    with input_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    result.update(
        {
            "size_bytes": input_path.stat().st_size,
            "md5": md5.hexdigest(),
            "sha256": sha256.hexdigest(),
        }
    )
    return result


def unknown_mention_candidates(
    input_dir: str | Path,
    dictionary_rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 100,
    min_count: int = 3,
) -> list[dict[str, Any]]:
    alias_keys = {normalize_for_match(alias) for row in dictionary_rows for alias in _row_aliases(row)}
    counts: Counter[str] = Counter()
    input_path = Path(input_dir)
    for path in sorted(input_path.glob("*.txt"), key=lambda item: _path_sort_key(item)):
        tokens = [token.casefold() for token in _TEXT_TOKEN_RE.findall(path.read_text(encoding="utf-8"))]
        for n in (2, 3):
            for index in range(0, max(0, len(tokens) - n + 1)):
                gram_tokens = tokens[index : index + n]
                if all(token in _STOPWORDS or token.isdigit() for token in gram_tokens):
                    continue
                if not any(token in _MEDICAL_CUE_TOKENS for token in gram_tokens):
                    continue
                term = " ".join(gram_tokens)
                normalized = normalize_for_match(term)
                if len(normalized) < 4 or normalized in alias_keys:
                    continue
                counts[term] += 1
    return [
        {"term": term, "count": count, "normalized": normalize_for_match(term)}
        for term, count in counts.most_common()
        if count >= min_count
    ][:top_k]


def _source_registry_issues(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    resources = registry.get("resources")
    if not isinstance(resources, list):
        return [{"severity": "high", "issue_type": "registry_schema", "notes": "resources must be a list"}]
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, resource in enumerate(resources, start=1):
        if not isinstance(resource, dict):
            issues.append({"severity": "high", "issue_type": "registry_schema", "path": f"resources[{index}]"})
            continue
        source_id = str(resource.get("id", "")).strip()
        if not source_id:
            issues.append({"severity": "high", "issue_type": "missing_source_id", "path": f"resources[{index}]"})
            continue
        if source_id in seen:
            issues.append({"severity": "high", "issue_type": "duplicate_source_id", "source_id": source_id})
        seen.add(source_id)
        for key in _SOURCE_REQUIRED_FIELDS:
            if not str(resource.get(key, "")).strip():
                issues.append({"severity": "high", "issue_type": "missing_registry_field", "source_id": source_id, "field": key})
        if not any(str(resource.get(key, "")).strip() for key in _SOURCE_VERSION_FIELDS):
            issues.append(
                {
                    "severity": "medium",
                    "issue_type": "missing_version_or_release_date",
                    "source_id": source_id,
                    "notes": "Add version/release_date or issued/effective dates for reproducibility.",
                }
            )
    return issues


def _registry_source_ids(registry: Mapping[str, Any]) -> set[str]:
    resources = registry.get("resources")
    if not isinstance(resources, list):
        return set()
    return {
        str(resource.get("id", "")).strip()
        for resource in resources
        if isinstance(resource, dict) and str(resource.get("id", "")).strip()
    }


def _registry_resource_details(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    resources = registry.get("resources")
    if not isinstance(resources, list):
        return []
    details: list[dict[str, Any]] = []
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        source_id = str(resource.get("id", "")).strip()
        if not source_id:
            continue
        detail: dict[str, Any] = {}
        for key in (
            "id",
            "name",
            "category",
            "access",
            "url",
            "version",
            "release_date",
            "issued_date",
            "effective_date",
            "license",
            "use",
            "download_urls",
            "notes",
        ):
            if key in resource:
                detail[key] = resource[key]
        local_files = resource.get("local_files")
        if isinstance(local_files, list):
            detail["local_files"] = [
                {
                    "role": str(item.get("role", "unspecified")),
                    "path": str(item.get("path", "")),
                    "required": bool(item.get("required", True)),
                }
                for item in local_files
                if isinstance(item, dict) and item.get("path")
            ]
        details.append(detail)
    return sorted(details, key=lambda item: str(item.get("id", "")))


def _registry_local_files(
    registry: Mapping[str, Any],
    extra_files: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    resources = registry.get("resources")
    if isinstance(resources, list):
        for resource in resources:
            if not isinstance(resource, dict):
                continue
            source_id = str(resource.get("id", "")).strip()
            local_files = resource.get("local_files", [])
            if not isinstance(local_files, list):
                continue
            for item in local_files:
                if isinstance(item, str):
                    rows.append({"source_id": source_id, "path": item, "role": "unspecified", "required": True})
                elif isinstance(item, dict):
                    row = dict(item)
                    row.setdefault("source_id", source_id)
                    row.setdefault("required", True)
                    rows.append(row)
    rows.extend(dict(item) for item in extra_files)
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row.get("source_id", "")), str(row.get("role", "")), str(row.get("path", "")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _local_file_manifest(item: Mapping[str, Any]) -> dict[str, Any]:
    path = str(item.get("path", "")).strip()
    result = {
        "source_id": str(item.get("source_id", "")).strip(),
        "role": str(item.get("role", "")).strip() or "unspecified",
        "required": bool(item.get("required", True)),
        **file_fingerprints(path),
    }
    return result


def _standard_versions_summary(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"path": str(path), "payload": payload}


def _manual_review_queue(dictionary_profiles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in dictionary_profiles:
        dictionary_path = str(profile.get("path", ""))
        for alias in _dict_list(profile.get("ambiguous_aliases"))[:50]:
            rows.append(
                {
                    "severity": "high",
                    "issue_type": "ambiguous_alias",
                    "dictionary": dictionary_path,
                    "alias": alias.get("alias"),
                    "normalized_alias": alias.get("normalized_alias"),
                    "concepts": alias.get("concepts"),
                    "notes": "Alias maps to multiple concepts; decide blocklist, type-specific rule, or preferred concept.",
                }
            )
        for alias in _dict_list(profile.get("broad_aliases"))[:50]:
            rows.append(
                {
                    "severity": "medium",
                    "issue_type": "broad_or_lab_like_alias",
                    "dictionary": dictionary_path,
                    **alias,
                    "notes": "Review as possible false-positive alias or blocked_alias candidate.",
                }
            )
        for mention in _dict_list(profile.get("unknown_mention_candidates"))[:50]:
            rows.append(
                {
                    "severity": "low",
                    "issue_type": "unknown_mention_candidate",
                    "dictionary": dictionary_path,
                    **mention,
                    "notes": "Frequent input n-gram not covered by dictionary aliases; review before adding.",
                }
            )
    return rows


def _audit_summary(
    source_issues: Sequence[Mapping[str, Any]],
    file_manifest: Sequence[Mapping[str, Any]],
    dictionary_profiles: Sequence[Mapping[str, Any]],
    rxnorm_profiles: Sequence[Mapping[str, Any]],
    manual_review_queue: Sequence[Mapping[str, Any]],
    false_positive_blocklist: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    missing_required = [item for item in file_manifest if item.get("required") and not item.get("exists")]
    return {
        "registry_issue_count": len(source_issues),
        "registry_resource_count": len({str(item.get("source_id", "")) for item in file_manifest if item.get("source_id")}),
        "file_count": len(file_manifest),
        "missing_required_file_count": len(missing_required),
        "dictionary_count": len(dictionary_profiles),
        "rxnorm_release_profile_count": len(rxnorm_profiles),
        "manual_review_item_count": len(manual_review_queue),
        "false_positive_blocklist_count": len(false_positive_blocklist),
    }


def _row_aliases(row: Mapping[str, Any]) -> list[str]:
    aliases: list[str] = []
    for key in (
        "canonical_name",
        "official_name_vi",
        "official_name_en",
        "aliases",
        "synonyms",
        "abbreviations",
        "ingredient",
        "brand_name",
        "generic_name",
    ):
        aliases.extend(_string_values(row.get(key)))
    blocked = {normalize_for_match(alias) for alias in _string_values(row.get("blocked_aliases"))}
    return [alias for alias in _unique_strings(aliases) if normalize_for_match(alias) not in blocked]


def _row_source_ids(row: Mapping[str, Any]) -> set[str]:
    source_ids: set[str] = set()
    raw_source_ids = row.get("source_ids")
    if isinstance(raw_source_ids, str):
        source_ids.add(raw_source_ids)
    elif isinstance(raw_source_ids, list | tuple | set):
        source_ids.update(str(source_id) for source_id in raw_source_ids if str(source_id).strip())
    source = row.get("source")
    if isinstance(source, str) and source.strip():
        source_ids.add(source.strip())
    return source_ids


def _broad_alias_reason(row: Mapping[str, Any], alias: str, normalized_alias: str) -> str | None:
    code_system = str(row.get("code_system", ""))
    semantic_type = str(row.get("semantic_type", ""))
    if len(normalized_alias) <= 2:
        return "too_short"
    if code_system == "ICD-10" and semantic_type == "DISEASE" and alias.isascii() and " " not in alias.strip():
        return "ascii_single_token_icd_alias"
    if code_system == "RxNorm" and normalized_alias in _LAB_METABOLITE_RXNORM_ALIASES:
        return "lab_or_metabolite_rxnorm_alias_requires_drug_context"
    return None


def _false_positive_notes(reason: str) -> str:
    if reason == "lab_or_metabolite_rxnorm_alias_requires_drug_context":
        return "Treat as a drug only near explicit medication/dose cues; otherwise prefer lab/metabolite context."
    if reason == "ascii_single_token_icd_alias":
        return "Single-token English ICD aliases frequently create false positives in clinical prose."
    if reason == "too_short":
        return "Very short aliases should require type/context-specific evidence before matching."
    return "Review before adding to blocked_aliases or context-gated matching."


def _ambiguous_aliases(alias_entries: Mapping[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for normalized, entries in alias_entries.items():
        concept_ids = {entry["concept_id"] for entry in entries}
        code_systems = {entry["code_system"] for entry in entries}
        if len(concept_ids) <= 1:
            continue
        rows.append(
            {
                "normalized_alias": normalized,
                "alias": entries[0]["alias"],
                "concept_count": len(concept_ids),
                "code_systems": sorted(code_systems),
                "concepts": sorted(entries, key=lambda item: item["concept_id"]),
            }
        )
    return sorted(rows, key=lambda item: (-int(item["concept_count"]), str(item["normalized_alias"])))


def _count_rows(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(row.get(key, "<missing>")) for row in rows)
    return dict(sorted(counter.items()))


def _icd10_hierarchy_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    icd_rows = [row for row in rows if row.get("code_system") == "ICD-10"]
    by_chapter = Counter(str(row.get("icd10_chapter") or "<missing>") for row in icd_rows)
    return {
        "rows": len(icd_rows),
        "with_parent_code": sum(1 for row in icd_rows if row.get("parent_code")),
        "with_block": sum(1 for row in icd_rows if row.get("icd10_block")),
        "with_chapter": sum(1 for row in icd_rows if row.get("icd10_chapter")),
        "by_chapter": dict(sorted(by_chapter.items())),
    }


def _rxnorm_enrichment_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rxnorm_rows = [row for row in rows if row.get("code_system") == "RxNorm"]
    return {
        "rows": len(rxnorm_rows),
        "with_ingredient": sum(1 for row in rxnorm_rows if row.get("ingredient") or row.get("ingredients")),
        "with_brand_name": sum(1 for row in rxnorm_rows if row.get("brand_name") or row.get("brand_names")),
        "with_dose_form": sum(1 for row in rxnorm_rows if row.get("dose_form") or row.get("dose_forms")),
        "with_strength": sum(1 for row in rxnorm_rows if row.get("strength") or row.get("strengths")),
        "with_status": sum(1 for row in rxnorm_rows if row.get("rxnorm_status")),
        "inactive_or_obsolete": sum(1 for row in rxnorm_rows if row.get("rxnorm_status") == "inactive"),
    }


def _duplicates(values: Sequence[str]) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in _duplicate_counter(values).items() if value]


def _duplicate_counter(values: Iterable[Any]) -> Counter[Any]:
    counter = Counter(values)
    return Counter({key: value for key, value in counter.items() if value > 1})


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def _unique_strings(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).split()).strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _path_sort_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (10**12, path.name)
