"""Compile terminology and mined relations into a deduplicated knowledge graph."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.dictionaries.synonym_table import ConceptEntry
from medical_kg_nlp.kg.knowledge_schema import (
    KnowledgeEdge,
    KnowledgeEvidence,
    KnowledgeNode,
    KnowledgeNodeKind,
)
from medical_kg_nlp.mining.io import write_json, write_jsonl
from medical_kg_nlp.mining.records import (
    AnnotationLayer,
    AnnotationProposal,
    ConceptLink,
    MinedDocument,
    RelationProposal,
    ReviewStatus,
)
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.hashing import sha256_file, sha256_text
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = ["GraphCompilationConfig", "compile_knowledge_graph"]


@dataclass(frozen=True)
class GraphCompilationConfig:
    """Fail-closed quality filters for graph promotion."""

    accepted_layers: tuple[AnnotationLayer, ...] = (
        AnnotationLayer.SILVER,
        AnnotationLayer.GOLD,
        AnnotationLayer.CHALLENGE,
    )
    accepted_review_statuses: tuple[ReviewStatus, ...] = (
        ReviewStatus.PROPOSED,
        ReviewStatus.ACCEPTED,
    )
    include_entity_types: tuple[str, ...] = ()
    include_unlinked_terms: bool = True
    include_structured_terminology_relations: bool = True
    relation_endpoints_only: bool = False
    require_canonical_concepts: bool = False
    preferred_code_systems_by_entity_type: tuple[
        tuple[str, tuple[str, ...]], ...
    ] = ()

    def __post_init__(self) -> None:
        entity_types = [
            entity_type for entity_type, _ in self.preferred_code_systems_by_entity_type
        ]
        if len(entity_types) != len(set(entity_types)):
            raise ValueError("Preferred graph code-system entity types must be unique")
        if any(
            not entity_type.strip() or not code_systems
            for entity_type, code_systems in self.preferred_code_systems_by_entity_type
        ):
            raise ValueError("Preferred graph code-system mappings must be explicit")
        for entity_type, code_systems in self.preferred_code_systems_by_entity_type:
            # INVARIANT: endpoint selection is a typed graph policy, not a free-form
            # string convention that may silently stop matching source annotations.
            EntityType(entity_type)
            parsed_systems = tuple(CodeSystem(value) for value in code_systems)
            if len(parsed_systems) != len(set(parsed_systems)):
                raise ValueError(
                    f"Preferred graph code systems for {entity_type} must be unique"
                )


@dataclass
class _NodeAccumulator:
    node_id: str
    kind: KnowledgeNodeKind
    entity_type: str
    code_system: str | None = None
    code: str | None = None
    labels: dict[str, int] = field(default_factory=dict)
    aliases: set[str] = field(default_factory=set)
    versions: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    occurrence_count: int = 0
    documents: set[str] = field(default_factory=set)

    def add_label(self, label: str, *, priority: int) -> None:
        cleaned = label.strip()
        if not cleaned:
            return
        previous = self.labels.get(cleaned)
        self.labels[cleaned] = priority if previous is None else min(previous, priority)
        self.aliases.add(cleaned)

    def to_record(self) -> KnowledgeNode:
        if not self.labels:
            raise ValueError(f"Knowledge node {self.node_id!r} has no display label")
        label = min(
            self.labels,
            key=lambda value: (self.labels[value], normalize_for_match(value), value),
        )
        aliases = tuple(
            sorted(
                (value for value in self.aliases if value != label),
                key=lambda value: (normalize_for_match(value), value),
            )
        )
        return KnowledgeNode(
            node_id=self.node_id,
            kind=self.kind,
            label=label,
            normalized_label=normalize_for_match(label),
            entity_type=self.entity_type,
            code_system=self.code_system,
            code=self.code,
            aliases=aliases,
            terminology_versions=tuple(sorted(self.versions)),
            sources=tuple(sorted(self.sources)),
            occurrence_count=self.occurrence_count,
            document_count=len(self.documents),
        )


@dataclass
class _EdgeAccumulator:
    edge_id: str
    head_node_id: str
    tail_node_id: str
    relation_type: str
    support_count: int = 0
    documents: set[str] = field(default_factory=set)
    confidence_sum: float = 0.0
    confidence_min: float = 1.0
    confidence_max: float = 0.0
    sources: set[str] = field(default_factory=set)
    layers: set[str] = field(default_factory=set)

    def add(
        self,
        *,
        confidence: float,
        source: str,
        layer: str,
        document_id: str | None,
    ) -> None:
        self.support_count += 1
        self.confidence_sum += confidence
        self.confidence_min = min(self.confidence_min, confidence)
        self.confidence_max = max(self.confidence_max, confidence)
        self.sources.add(source)
        self.layers.add(layer)
        if document_id is not None:
            self.documents.add(document_id)

    def to_record(self) -> KnowledgeEdge:
        return KnowledgeEdge(
            edge_id=self.edge_id,
            head_node_id=self.head_node_id,
            tail_node_id=self.tail_node_id,
            relation_type=self.relation_type,
            support_count=self.support_count,
            document_count=len(self.documents),
            confidence_mean=self.confidence_sum / self.support_count,
            confidence_min=self.confidence_min,
            confidence_max=self.confidence_max,
            sources=tuple(sorted(self.sources)),
            layers=tuple(sorted(self.layers)),
        )


def compile_knowledge_graph(
    *,
    terminology_paths: Sequence[str | Path],
    alias_overlay_paths: Sequence[str | Path],
    documents: Sequence[MinedDocument],
    annotations: Sequence[AnnotationProposal],
    relations: Sequence[RelationProposal],
    config: GraphCompilationConfig,
    nodes_output: str | Path,
    edges_output: str | Path,
    evidence_output: str | Path,
    report_output: str | Path,
    documents_path: str | Path | None = None,
    annotations_path: str | Path | None = None,
    relations_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compile canonical hierarchy plus corpus relations into immutable JSONL."""

    nodes: dict[str, _NodeAccumulator] = {}
    edges: dict[str, _EdgeAccumulator] = {}
    evidence: list[KnowledgeEvidence] = []
    counters: Counter[str] = Counter()

    entries = _load_terminology(terminology_paths)
    concept_ids: dict[tuple[str, str], str] = {}
    concept_entry_ids: dict[tuple[str, str], str] = {}
    concept_nodes_by_entry_id: dict[str, str] = {}
    for entry, source_path in entries:
        if entry.code is None:
            counters["code_free_terminology_skipped"] += 1
            continue
        node_id = concept_node_id(entry.code_system.value, entry.code)
        concept_ids[(entry.code_system.value, entry.code)] = node_id
        concept_entry_ids[(entry.code_system.value, entry.concept_id)] = node_id
        previous_entry_node = concept_nodes_by_entry_id.get(entry.concept_id)
        if previous_entry_node is not None and previous_entry_node != node_id:
            raise ValueError(f"Terminology concept ID {entry.concept_id!r} is not globally unique")
        concept_nodes_by_entry_id[entry.concept_id] = node_id
        node = nodes.setdefault(
            node_id,
            _NodeAccumulator(
                node_id=node_id,
                kind=KnowledgeNodeKind.CONCEPT,
                entity_type=entry.semantic_type.value,
                code_system=entry.code_system.value,
                code=entry.code,
            ),
        )
        _merge_node_type(node, entry.semantic_type.value)
        node.add_label(entry.official_name_vi or entry.canonical_name, priority=0)
        node.add_label(entry.canonical_name, priority=1)
        if entry.official_name_en:
            node.add_label(entry.official_name_en, priority=2)
        for alias in entry.all_names:
            node.add_label(alias, priority=4)
        node.sources.add(f"terminology:{source_path.name}")

    for entry, source_path in entries:
        if entry.code is None:
            continue
        child_id = concept_ids[(entry.code_system.value, entry.code)]
        for parent in entry.parents:
            parent_id = concept_entry_ids.get((entry.code_system.value, parent)) or concept_ids.get(
                (entry.code_system.value, parent)
            )
            if parent_id is None or parent_id == child_id:
                counters["unknown_or_self_parent_skipped"] += 1
                continue
            edge = _edge_accumulator(edges, child_id, parent_id, "IS_A")
            source = f"terminology:{source_path.name}"
            edge.add(confidence=1.0, source=source, layer="canonical", document_id=None)
            evidence.append(
                _evidence(
                    edge.edge_id,
                    source_record_id=entry.concept_id,
                    source_record_kind="terminology_parent",
                    source=source,
                )
            )
            counters["hierarchy_observation_count"] += 1

    _apply_alias_overlays(
        alias_overlay_paths,
        nodes,
        concept_ids,
        concept_nodes_by_entry_id,
        counters,
    )
    if config.include_structured_terminology_relations:
        _apply_structured_terminology_relations(
            entries,
            concept_ids,
            nodes,
            edges,
            evidence,
            counters,
        )

    documents_by_id = _unique_documents(documents)
    all_annotations_by_id, annotations_by_id = _valid_annotations(
        annotations,
        documents_by_id,
        config,
        nodes,
        counters,
        relation_endpoint_ids=(
            {
                annotation_id
                for relation in relations
                for annotation_id in (
                    relation.head_annotation_id,
                    relation.tail_annotation_id,
                )
            }
            if config.relation_endpoints_only
            else None
        ),
    )
    annotation_nodes: dict[str, str] = {}
    for annotation in annotations_by_id.values():
        selected_concept = _selected_annotation_concept(annotation, config)
        annotation_node_id = _annotation_node_id(annotation, config, selected_concept)
        if annotation_node_id is None:
            counters["unlinked_annotation_skipped"] += 1
            continue
        annotation_nodes[annotation.annotation_id] = annotation_node_id
        node = _annotation_node(nodes, annotation, annotation_node_id, selected_concept)
        node.occurrence_count += 1
        node.documents.add(annotation.document_id)
        node.sources.add(f"annotation:{annotation.labeler_id}")
        node.add_label(annotation.text, priority=3)
        if selected_concept is not None:
            node.versions.add(selected_concept.terminology_version)
        else:
            for concept in annotation.concepts:
                node.versions.add(concept.terminology_version)
        if len(annotation.concepts) > 1:
            for concept in annotation.concepts:
                concept_id = concept_node_id(concept.code_system, concept.code)
                if concept_id == annotation_node_id:
                    continue
                if config.require_canonical_concepts and concept_id not in nodes:
                    counters["annotation_noncanonical_link_skipped"] += 1
                    continue
                concept_node = nodes.setdefault(
                    concept_id,
                    _NodeAccumulator(
                        node_id=concept_id,
                        kind=KnowledgeNodeKind.CONCEPT,
                        entity_type=annotation.entity_type,
                        code_system=concept.code_system,
                        code=concept.code,
                    ),
                )
                _merge_annotation_type(
                    concept_node,
                    annotation.entity_type,
                    concept.code_system,
                )
                concept_node.add_label(annotation.text, priority=3)
                concept_node.versions.add(concept.terminology_version)
                concept_node.sources.add(f"annotation-link:{annotation.labeler_id}")
                concept_node.occurrence_count += 1
                concept_node.documents.add(annotation.document_id)
                mapping_edge = _edge_accumulator(
                    edges,
                    annotation_node_id,
                    concept_id,
                    "MAPS_TO",
                )
                source = f"annotation-link:{annotation.labeler_id}"
                mapping_edge.add(
                    confidence=concept.confidence,
                    source=source,
                    layer=annotation.layer.value,
                    document_id=annotation.document_id,
                )
                evidence.append(
                    _evidence(
                        mapping_edge.edge_id,
                        source_record_id=f"{annotation.annotation_id}:{concept.code_system}:{concept.code}",
                        source_record_kind="annotation_concept_link",
                        source=source,
                        document_id=annotation.document_id,
                        source_artifact_id=documents_by_id[
                            annotation.document_id
                        ].source_artifact_id,
                    )
                )
                counters["mapping_observation_count"] += 1

    relation_ids: set[str] = set()
    for relation in relations:
        if relation.relation_id in relation_ids:
            raise ValueError(f"Duplicate relation ID {relation.relation_id!r}")
        relation_ids.add(relation.relation_id)
        document = documents_by_id.get(relation.document_id)
        if document is None:
            raise ValueError(f"Unknown relation document {relation.document_id!r}")
        relation.validate(document, all_annotations_by_id)
        if not _accepted_relation(relation, config):
            counters["relation_quality_filter_skipped"] += 1
            continue
        head_id = annotation_nodes.get(relation.head_annotation_id)
        tail_id = annotation_nodes.get(relation.tail_annotation_id)
        if head_id is None or tail_id is None:
            counters["relation_filtered_endpoint_skipped"] += 1
            continue
        if head_id == tail_id:
            counters["relation_self_edge_skipped"] += 1
            continue
        edge = _edge_accumulator(edges, head_id, tail_id, relation.relation_type)
        source = f"relation:{relation.labeler_id or relation.label_source}"
        edge.add(
            confidence=relation.confidence,
            source=source,
            layer=relation.layer.value,
            document_id=relation.document_id,
        )
        evidence.append(
            _evidence(
                edge.edge_id,
                source_record_id=relation.relation_id,
                source_record_kind="mined_relation",
                source=source,
                document_id=relation.document_id,
                source_artifact_id=document.source_artifact_id,
                evidence_span=relation.evidence_span,
                head_annotation_id=relation.head_annotation_id,
                tail_annotation_id=relation.tail_annotation_id,
            )
        )
        counters["relation_observation_count"] += 1

    node_records = [nodes[node_id].to_record() for node_id in sorted(nodes)]
    edge_records = [edges[edge_id].to_record() for edge_id in sorted(edges)]
    evidence.sort(key=lambda item: item.evidence_id)
    nodes_sha = write_jsonl(nodes_output, (record.to_dict() for record in node_records))
    edges_sha = write_jsonl(edges_output, (record.to_dict() for record in edge_records))
    evidence_sha = write_jsonl(evidence_output, (record.to_dict() for record in evidence))
    report: dict[str, Any] = {
        "schema_version": "mined-knowledge-graph.v1",
        "config": {
            "accepted_layers": [value.value for value in config.accepted_layers],
            "accepted_review_statuses": [
                value.value for value in config.accepted_review_statuses
            ],
            "include_entity_types": list(config.include_entity_types),
            "include_unlinked_terms": config.include_unlinked_terms,
            "include_structured_terminology_relations": (
                config.include_structured_terminology_relations
            ),
            "relation_endpoints_only": config.relation_endpoints_only,
            "require_canonical_concepts": config.require_canonical_concepts,
            "preferred_code_systems_by_entity_type": {
                entity_type: list(code_systems)
                for entity_type, code_systems in config.preferred_code_systems_by_entity_type
            },
        },
        "input_counts": {
            "terminology_concepts": len(entries),
            "documents": len(documents),
            "annotations": len(annotations),
            "relations": len(relations),
        },
        "output_counts": {
            "nodes": len(node_records),
            "concept_nodes": sum(
                record.kind == KnowledgeNodeKind.CONCEPT for record in node_records
            ),
            "term_nodes": sum(record.kind == KnowledgeNodeKind.TERM for record in node_records),
            "edges": len(edge_records),
            "evidence": len(evidence),
        },
        "node_entity_type_counts": dict(
            sorted(Counter(record.entity_type for record in node_records).items())
        ),
        "node_code_system_counts": dict(
            sorted(
                Counter(
                    record.code_system or "NONE"
                    for record in node_records
                ).items()
            )
        ),
        "edge_relation_type_counts": dict(
            sorted(Counter(record.relation_type for record in edge_records).items())
        ),
        "decision_counts": dict(sorted(counters.items())),
        "inputs": {
            "terminology": [_fingerprinted_path(path) for path in terminology_paths],
            "alias_overlays": [_fingerprinted_path(path) for path in alias_overlay_paths],
            "documents": _optional_fingerprinted_path(documents_path),
            "annotations": _optional_fingerprinted_path(annotations_path),
            "relations": _optional_fingerprinted_path(relations_path),
        },
        "outputs": {
            "nodes": {"path": str(Path(nodes_output)), "sha256": nodes_sha},
            "edges": {"path": str(Path(edges_output)), "sha256": edges_sha},
            "evidence": {"path": str(Path(evidence_output)), "sha256": evidence_sha},
        },
    }
    write_json(report_output, report)
    return report


def concept_node_id(code_system: str, code: str) -> str:
    """Return a stable opaque identity for a terminology concept."""

    identity = "\x1f".join((code_system, code))
    return f"concept:{sha256_text(identity)[:24]}"


def term_node_id(entity_type: str, text: str) -> str:
    """Return a stable identity for an uncoded normalized surface."""

    normalized = normalize_for_match(text)
    if not normalized:
        raise ValueError("Term node text normalizes to empty")
    identity = "\x1f".join((entity_type, normalized))
    return f"term:{sha256_text(identity)[:24]}"


def _load_terminology(
    paths: Sequence[str | Path],
) -> list[tuple[ConceptEntry, Path]]:
    output: list[tuple[ConceptEntry, Path]] = []
    for raw_path in paths:
        path = Path(raw_path)
        output.extend((entry, path) for entry in DictionaryStore.load_entries_jsonl(path))
    return output


def _apply_alias_overlays(
    paths: Sequence[str | Path],
    nodes: dict[str, _NodeAccumulator],
    concept_ids: Mapping[tuple[str, str], str],
    concept_entry_ids: Mapping[str, str],
    counters: Counter[str],
) -> None:
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                alias = _required_string(raw, "alias", path, line_number)
                target_concept_id = _optional_non_empty_string(raw.get("target_concept_id"))
                code_system = _optional_non_empty_string(raw.get("code_system"))
                code = _optional_non_empty_string(raw.get("code"))
                if target_concept_id is not None:
                    node_id = concept_entry_ids.get(target_concept_id)
                elif code_system is not None and code is not None:
                    node_id = concept_ids.get((code_system, code))
                else:
                    raise ValueError(
                        f"{path}:{line_number}: alias requires target_concept_id or code system/code"
                    )
                if node_id is None:
                    # INVARIANT: heterogeneous overlays may mention local concepts, but graph
                    # compilation never invents a node absent from its canonical sources.
                    counters["unknown_alias_target_skipped"] += 1
                    continue
                nodes[node_id].add_label(alias, priority=4)
                nodes[node_id].sources.add(f"alias-overlay:{path.name}")
                counters["alias_overlay_observation_count"] += 1


def _apply_structured_terminology_relations(
    entries: Sequence[tuple[ConceptEntry, Path]],
    concept_ids: Mapping[tuple[str, str], str],
    nodes: dict[str, _NodeAccumulator],
    edges: dict[str, _EdgeAccumulator],
    evidence: list[KnowledgeEvidence],
    counters: Counter[str],
) -> None:
    """Promote exact RxNorm structured attributes into auditable graph edges.

    RxNorm releases expose ingredient, dosage-form, and strength attributes as flat
    concept fields rather than an edge table. We only promote an ingredient when the
    normalized value resolves to one unique IN/PIN/MIN node; ambiguous values remain
    metadata, not guessed graph edges.
    """

    ingredient_targets: dict[str, set[str]] = {}
    for entry, _ in entries:
        if (
            entry.code_system != CodeSystem.RXNORM
            or entry.code is None
            or entry.rxnorm_tty not in {"IN", "PIN", "MIN"}
        ):
            continue
        node_id = concept_ids.get((entry.code_system.value, entry.code))
        if node_id is None:
            continue
        for name in (entry.canonical_name, entry.ingredient):
            if name:
                ingredient_targets.setdefault(normalize_for_match(name), set()).add(node_id)

    for entry, source_path in entries:
        if entry.code_system != CodeSystem.RXNORM or entry.code is None:
            continue
        head_id = concept_ids.get((entry.code_system.value, entry.code))
        if head_id is None:
            continue
        source = f"terminology:{source_path.name}"
        if entry.ingredient and entry.rxnorm_tty not in {"IN", "PIN", "MIN"}:
            target_ids = ingredient_targets.get(normalize_for_match(entry.ingredient), set())
            if len(target_ids) == 1:
                tail_id = next(iter(target_ids))
                if tail_id != head_id:
                    _add_structured_edge(
                        edges,
                        evidence,
                        head_id=head_id,
                        tail_id=tail_id,
                        relation_type="HAS_ACTIVE_INGREDIENT",
                        source=source,
                        source_record_id=f"{entry.concept_id}:ingredient:{source_path.name}",
                    )
                    counters["structured_ingredient_observation_count"] += 1
            elif not target_ids:
                counters["structured_ingredient_unresolved_skipped"] += 1
            else:
                counters["structured_ingredient_ambiguous_skipped"] += 1

        if entry.dose_form:
            tail_id = _structured_term_node(
                nodes,
                entity_type="DOSAGE_FORM",
                text=entry.dose_form,
                source=source,
            )
            _add_structured_edge(
                edges,
                evidence,
                head_id=head_id,
                tail_id=tail_id,
                relation_type="HAS_DOSAGE_FORM",
                source=source,
                source_record_id=f"{entry.concept_id}:dose_form:{source_path.name}",
            )
            counters["structured_dose_form_observation_count"] += 1

        if entry.strength:
            strength = _clean_strength(entry.strength)
            if strength:
                tail_id = _structured_term_node(
                    nodes,
                    entity_type="STRENGTH",
                    text=strength,
                    source=source,
                )
                _add_structured_edge(
                    edges,
                    evidence,
                    head_id=head_id,
                    tail_id=tail_id,
                    relation_type="HAS_STRENGTH",
                    source=source,
                    source_record_id=f"{entry.concept_id}:strength:{source_path.name}",
                )
                counters["structured_strength_observation_count"] += 1


def _structured_term_node(
    nodes: dict[str, _NodeAccumulator],
    *,
    entity_type: str,
    text: str,
    source: str,
) -> str:
    node_id = term_node_id(entity_type, text)
    node = nodes.setdefault(
        node_id,
        _NodeAccumulator(
            node_id=node_id,
            kind=KnowledgeNodeKind.TERM,
            entity_type=entity_type,
        ),
    )
    node.add_label(text, priority=0)
    node.sources.add(source)
    return node_id


def _add_structured_edge(
    edges: dict[str, _EdgeAccumulator],
    evidence: list[KnowledgeEvidence],
    *,
    head_id: str,
    tail_id: str,
    relation_type: str,
    source: str,
    source_record_id: str,
) -> None:
    edge = _edge_accumulator(edges, head_id, tail_id, relation_type)
    edge.add(confidence=1.0, source=source, layer="canonical", document_id=None)
    evidence.append(
        _evidence(
            edge.edge_id,
            source_record_id=source_record_id,
            source_record_kind="terminology_structured_attribute",
            source=source,
        )
    )


def _clean_strength(value: str) -> str:
    prefix = "RXN_AVAILABLE_STRENGTH="
    return value.removeprefix(prefix).strip()


def _unique_documents(documents: Sequence[MinedDocument]) -> dict[str, MinedDocument]:
    output: dict[str, MinedDocument] = {}
    for document in documents:
        if document.document_id in output:
            raise ValueError(f"Duplicate document ID {document.document_id!r}")
        output[document.document_id] = document
    return output


def _valid_annotations(
    annotations: Sequence[AnnotationProposal],
    documents: Mapping[str, MinedDocument],
    config: GraphCompilationConfig,
    nodes: Mapping[str, _NodeAccumulator],
    counters: Counter[str],
    relation_endpoint_ids: set[str] | None,
) -> tuple[dict[str, AnnotationProposal], dict[str, AnnotationProposal]]:
    all_annotations: dict[str, AnnotationProposal] = {}
    accepted: dict[str, AnnotationProposal] = {}
    include_types = set(config.include_entity_types)
    for annotation in annotations:
        if annotation.annotation_id in all_annotations:
            raise ValueError(f"Duplicate annotation ID {annotation.annotation_id!r}")
        document = documents.get(annotation.document_id)
        if document is None:
            raise ValueError(f"Unknown annotation document {annotation.document_id!r}")
        annotation.validate_offsets(document)
        all_annotations[annotation.annotation_id] = annotation
        if (
            relation_endpoint_ids is not None
            and annotation.annotation_id not in relation_endpoint_ids
        ):
            # INVARIANT: endpoint-only experiments validate every source row but cannot
            # leak labels or aliases from unrelated held-out annotations into the graph.
            counters["annotation_non_endpoint_skipped"] += 1
            continue
        if annotation.layer not in config.accepted_layers:
            counters["annotation_layer_filter_skipped"] += 1
            continue
        if annotation.review_status not in config.accepted_review_statuses:
            counters["annotation_review_filter_skipped"] += 1
            continue
        if include_types and annotation.entity_type not in include_types:
            counters["annotation_type_filter_skipped"] += 1
            continue
        selected_concept = _selected_annotation_concept(annotation, config)
        if len(annotation.concepts) > 1:
            counter = (
                "multi_concept_annotation_preferred"
                if selected_concept is not None
                else "multi_concept_annotation_as_term"
            )
            counters[counter] += 1
        if selected_concept is not None:
            concept = selected_concept
            node_id = concept_node_id(concept.code_system, concept.code)
            existing = nodes.get(node_id)
            if existing is None and config.require_canonical_concepts:
                # INVARIANT: a mined source can propose a code, but a runtime graph cannot
                # create it unless the loaded canonical terminology contains that code.
                counters["annotation_unknown_canonical_concept_skipped"] += 1
                continue
            if existing is not None:
                _merge_annotation_type(
                    existing,
                    annotation.entity_type,
                    concept.code_system,
                )
            counters[f"selected_annotation_concept:{concept.code_system}"] += 1
        accepted[annotation.annotation_id] = annotation
    return all_annotations, accepted


def _accepted_relation(relation: RelationProposal, config: GraphCompilationConfig) -> bool:
    return (
        relation.layer in config.accepted_layers
        and relation.review_status in config.accepted_review_statuses
    )


def _annotation_node_id(
    annotation: AnnotationProposal,
    config: GraphCompilationConfig,
    selected_concept: ConceptLink | None,
) -> str | None:
    if selected_concept is not None:
        return concept_node_id(selected_concept.code_system, selected_concept.code)
    if not config.include_unlinked_terms:
        return None
    return term_node_id(annotation.entity_type, annotation.text)


def _annotation_node(
    nodes: dict[str, _NodeAccumulator],
    annotation: AnnotationProposal,
    node_id: str,
    selected_concept: ConceptLink | None,
) -> _NodeAccumulator:
    if selected_concept is not None:
        concept = selected_concept
        node = nodes.setdefault(
            node_id,
            _NodeAccumulator(
                node_id=node_id,
                kind=KnowledgeNodeKind.CONCEPT,
                entity_type=annotation.entity_type,
                code_system=concept.code_system,
                code=concept.code,
            ),
        )
    else:
        node = nodes.setdefault(
            node_id,
            _NodeAccumulator(
                node_id=node_id,
                kind=KnowledgeNodeKind.TERM,
                entity_type=annotation.entity_type,
            ),
        )
    if selected_concept is not None:
        _merge_annotation_type(node, annotation.entity_type, concept.code_system)
    else:
        _merge_node_type(node, annotation.entity_type)
    return node


def _selected_annotation_concept(
    annotation: AnnotationProposal,
    config: GraphCompilationConfig,
) -> ConceptLink | None:
    preferred = dict(config.preferred_code_systems_by_entity_type).get(
        annotation.entity_type
    )
    if preferred is None:
        return annotation.concepts[0] if len(annotation.concepts) == 1 else None
    matching = tuple(
        concept for concept in annotation.concepts if concept.code_system in preferred
    )
    return matching[0] if len(matching) == 1 else None


def _merge_annotation_type(
    node: _NodeAccumulator,
    annotation_entity_type: str,
    code_system: str,
) -> None:
    if node.entity_type == annotation_entity_type:
        return
    # INVARIANT: HPO canonical concepts remain FINDING nodes even when a source schema calls
    # their concrete textual occurrences SYMPTOM. No other cross-type merge is implicit.
    if (
        code_system == CodeSystem.HPO.value
        and node.entity_type == "FINDING"
        and annotation_entity_type == "SYMPTOM"
    ):
        return
    _merge_node_type(node, annotation_entity_type)


def _merge_node_type(node: _NodeAccumulator, entity_type: str) -> None:
    if node.entity_type != entity_type:
        raise ValueError(
            f"Node {node.node_id!r} has conflicting entity types "
            f"{node.entity_type!r} and {entity_type!r}"
        )


def _edge_accumulator(
    edges: dict[str, _EdgeAccumulator],
    head_node_id: str,
    tail_node_id: str,
    relation_type: str,
) -> _EdgeAccumulator:
    identity = "\x1f".join((head_node_id, relation_type, tail_node_id))
    edge_id = f"edge:{sha256_text(identity)[:24]}"
    return edges.setdefault(
        edge_id,
        _EdgeAccumulator(
            edge_id=edge_id,
            head_node_id=head_node_id,
            tail_node_id=tail_node_id,
            relation_type=relation_type,
        ),
    )


def _evidence(
    edge_id: str,
    *,
    source_record_id: str,
    source_record_kind: str,
    source: str,
    document_id: str | None = None,
    source_artifact_id: str | None = None,
    evidence_span: tuple[int, int] | None = None,
    head_annotation_id: str | None = None,
    tail_annotation_id: str | None = None,
) -> KnowledgeEvidence:
    identity = "\x1f".join((edge_id, source_record_kind, source_record_id))
    return KnowledgeEvidence(
        evidence_id=f"evidence:{sha256_text(identity)[:24]}",
        edge_id=edge_id,
        source_record_id=source_record_id,
        source_record_kind=source_record_kind,
        source=source,
        document_id=document_id,
        source_artifact_id=source_artifact_id,
        evidence_span=evidence_span,
        head_annotation_id=head_annotation_id,
        tail_annotation_id=tail_annotation_id,
    )


def _required_string(
    raw: Mapping[str, object],
    field_name: str,
    path: Path,
    line_number: int,
) -> str:
    value = raw.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_number}: {field_name} must be a non-empty string")
    return value.strip()


def _optional_non_empty_string(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _fingerprinted_path(path: str | Path) -> dict[str, str]:
    return {"path": str(Path(path)), "sha256": sha256_file(path)}


def _optional_fingerprinted_path(path: str | Path | None) -> dict[str, str] | None:
    return None if path is None else _fingerprinted_path(path)
