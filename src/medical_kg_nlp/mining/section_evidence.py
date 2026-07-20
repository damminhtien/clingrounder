"""Attach source-block and section evidence tiers without changing annotation spans."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from medical_kg_nlp.mining.records import AnnotationProposal, MinedDocument

__all__ = [
    "BlockEvidencePolicy",
    "BlockEvidenceResult",
    "BlockEvidenceRule",
    "attach_block_evidence",
    "load_block_evidence_policy",
]


@dataclass(frozen=True)
class BlockEvidenceRule:
    """One ordered source-structure rule loaded from a versioned policy."""

    rule_id: str
    evidence_tier: str
    block_kinds: tuple[str, ...] = ()
    section_type_patterns: tuple[str, ...] = ()
    section_path_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.rule_id.strip() or not self.evidence_tier.strip():
            raise ValueError("Block evidence rules require non-empty IDs and tiers")
        if not (self.block_kinds or self.section_type_patterns or self.section_path_patterns):
            raise ValueError(f"Block evidence rule {self.rule_id!r} has no selector")
        for pattern in (*self.section_type_patterns, *self.section_path_patterns):
            re.compile(pattern, flags=re.IGNORECASE)


@dataclass(frozen=True)
class BlockEvidencePolicy:
    """How one parser's source blocks map to task-neutral evidence tiers."""

    policy_id: str
    source_block_format: str
    rules: tuple[BlockEvidenceRule, ...]
    default_tier: str
    uncontained_tier: str = "uncontained"

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.source_block_format.strip():
            raise ValueError("Block evidence policy identity must be non-empty")
        if not self.rules or not self.default_tier.strip() or not self.uncontained_tier.strip():
            raise ValueError("Block evidence policy requires rules and fallback tiers")
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("Block evidence rule IDs must be unique")


@dataclass(frozen=True)
class BlockEvidenceResult:
    """Unmodified source annotations enriched with auditable block metadata."""

    annotations: tuple[AnnotationProposal, ...]
    report: dict[str, Any]


@dataclass(frozen=True)
class _SourceBlock:
    kind: str
    span: tuple[int, int]
    section_path: tuple[str, ...]
    section_type: str
    text_sha256: str


def load_block_evidence_policy(path: str | Path) -> BlockEvidencePolicy:
    """Load ordered regex selectors from YAML instead of embedding source headings in code."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Block evidence policy must be an object")
    if raw.get("schema_version") != "medical-block-evidence-policy.v1":
        raise ValueError("Unsupported block evidence policy schema version")
    raw_rules = raw.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("Block evidence policy requires a non-empty rules list")
    rules: list[BlockEvidenceRule] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, Mapping):
            raise ValueError("Block evidence rules must be objects")
        rules.append(
            BlockEvidenceRule(
                rule_id=str(raw_rule["rule_id"]),
                evidence_tier=str(raw_rule["evidence_tier"]),
                block_kinds=_strings(raw_rule.get("block_kinds"), "block_kinds"),
                section_type_patterns=_strings(
                    raw_rule.get("section_type_patterns"), "section_type_patterns"
                ),
                section_path_patterns=_strings(
                    raw_rule.get("section_path_patterns"), "section_path_patterns"
                ),
            )
        )
    return BlockEvidencePolicy(
        policy_id=str(raw["policy_id"]),
        source_block_format=str(raw["source_block_format"]),
        rules=tuple(rules),
        default_tier=str(raw.get("default_tier", "other_context")),
        uncontained_tier=str(raw.get("uncontained_tier", "uncontained")),
    )


def attach_block_evidence(
    documents: Sequence[MinedDocument],
    annotations: Sequence[AnnotationProposal],
    policy: BlockEvidencePolicy,
) -> BlockEvidenceResult:
    """Attach one block/tier to each annotation while preserving its identity and content."""

    documents_by_id = {document.document_id: document for document in documents}
    if len(documents_by_id) != len(documents):
        raise ValueError("Cannot attach block evidence with duplicate document IDs")
    blocks_by_document: dict[str, tuple[_SourceBlock, ...]] = {}
    block_tiers: Counter[str] = Counter()
    for document in documents:
        if document.metadata.get("source_block_format") != policy.source_block_format:
            raise ValueError(
                f"Document {document.document_id!r} does not use block format "
                f"{policy.source_block_format!r}"
            )
        blocks = _load_source_blocks(document)
        blocks_by_document[document.document_id] = blocks
        for block in blocks:
            tier, _ = _classify_block(block, policy)
            block_tiers[tier] += 1

    output: list[AnnotationProposal] = []
    tier_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    tier_entity_counts: Counter[str] = Counter()
    for annotation in sorted(annotations, key=lambda item: item.annotation_id):
        matched_document = documents_by_id.get(annotation.document_id)
        if matched_document is None:
            raise ValueError(f"Annotation {annotation.annotation_id!r} references unknown document")
        start, end = annotation.span
        if matched_document.text[start:end] != annotation.text:
            raise ValueError(f"Annotation {annotation.annotation_id!r} has an offset mismatch")
        matched_block = next(
            (
                candidate
                for candidate in blocks_by_document[annotation.document_id]
                if candidate.span[0] <= start and end <= candidate.span[1]
            ),
            None,
        )
        if matched_block is None:
            tier = policy.uncontained_tier
            rule_id = "uncontained"
            metadata = {
                **annotation.metadata,
                "evidence_tier": tier,
                "evidence_rule_id": rule_id,
            }
        else:
            tier, rule_id = _classify_block(matched_block, policy)
            metadata = {
                **annotation.metadata,
                "evidence_tier": tier,
                "evidence_rule_id": rule_id,
                "source_block_kind": matched_block.kind,
                "source_block_span": json.dumps(matched_block.span, separators=(",", ":")),
                "source_section_path": json.dumps(
                    matched_block.section_path,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "source_section_type": matched_block.section_type,
            }
        tier_counts[tier] += 1
        rule_counts[rule_id] += 1
        tier_entity_counts[f"{tier}:{annotation.entity_type}"] += 1
        # INVARIANT: section evidence is metadata only. IDs, spans, text, labels, assertions,
        # concepts, confidence, layer, and review status remain byte-for-byte equivalent.
        output.append(replace(annotation, metadata=metadata))

    report = {
        "schema_version": "medical-block-evidence.v1",
        "policy_id": policy.policy_id,
        "document_count": len(documents),
        "annotation_count": len(annotations),
        "source_block_count": sum(len(blocks) for blocks in blocks_by_document.values()),
        "block_tier_counts": dict(sorted(block_tiers.items())),
        "annotation_tier_counts": dict(sorted(tier_counts.items())),
        "rule_counts": dict(sorted(rule_counts.items())),
        "tier_entity_type_counts": dict(sorted(tier_entity_counts.items())),
        "span_content_mutation_count": 0,
    }
    return BlockEvidenceResult(annotations=tuple(output), report=report)


def _load_source_blocks(document: MinedDocument) -> tuple[_SourceBlock, ...]:
    raw_value = document.metadata.get("source_blocks")
    if raw_value is None:
        raise ValueError(f"Document {document.document_id!r} has no source_blocks metadata")
    raw_blocks = json.loads(raw_value)
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ValueError(f"Document {document.document_id!r} has invalid source_blocks")
    blocks: list[_SourceBlock] = []
    previous_end = -1
    for raw in raw_blocks:
        if not isinstance(raw, Mapping):
            raise ValueError("Source blocks must be objects")
        raw_span = raw.get("span")
        if not isinstance(raw_span, list) or len(raw_span) != 2:
            raise ValueError("Source block span must be a two-integer list")
        span = (int(raw_span[0]), int(raw_span[1]))
        if span[0] < 0 or span[1] <= span[0] or span[0] < previous_end:
            raise ValueError("Source blocks must be ordered, non-overlapping, and non-empty")
        text = document.text[span[0] : span[1]]
        expected_hash = str(raw.get("text_sha256", ""))
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != expected_hash:
            raise ValueError("Source block text hash does not match the immutable document")
        raw_path = raw.get("section_path", [])
        if not isinstance(raw_path, list):
            raise ValueError("Source block section_path must be a list")
        blocks.append(
            _SourceBlock(
                kind=str(raw.get("kind", "")),
                span=span,
                section_path=tuple(str(value) for value in raw_path),
                section_type=str(raw.get("section_type", "")),
                text_sha256=expected_hash,
            )
        )
        previous_end = span[1]
    return tuple(blocks)


def _classify_block(
    block: _SourceBlock,
    policy: BlockEvidencePolicy,
) -> tuple[str, str]:
    path = " > ".join(block.section_path)
    for rule in policy.rules:
        if rule.block_kinds and block.kind not in rule.block_kinds:
            continue
        if rule.section_type_patterns and not any(
            re.search(pattern, block.section_type, flags=re.IGNORECASE)
            for pattern in rule.section_type_patterns
        ):
            continue
        if rule.section_path_patterns and not any(
            re.search(pattern, path, flags=re.IGNORECASE)
            for pattern in rule.section_path_patterns
        ):
            continue
        return rule.evidence_tier, rule.rule_id
    return policy.default_tier, "default"


def _strings(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    output = tuple(str(item) for item in value)
    if any(not item.strip() for item in output):
        raise ValueError(f"{field_name} values must be non-empty")
    return output
