"""Compile one pinned OBO Graph JSON release into terminology and KG artifacts.

The compiler keeps the rich source record separate from the runtime terminology row. This avoids
discarding synonym scopes, xrefs, definitions, deprecation replacements, and source properties
just because the current dictionary loader does not consume them.
"""

from __future__ import annotations

import importlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from medical_kg_nlp.kg.knowledge_schema import (
    KnowledgeEdge,
    KnowledgeEvidence,
    KnowledgeNode,
    KnowledgeNodeKind,
)
from medical_kg_nlp.kg.constraints import code_system_valid_for_entity_type
from medical_kg_nlp.mining.graph_knowledge import concept_node_id
from medical_kg_nlp.mining.io import write_json, write_jsonl
from medical_kg_nlp.schema.types import CodeSystem, EntityType
from medical_kg_nlp.utils.hashing import sha256_file, sha256_text
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = ["OBOGraphCompilationConfig", "compile_obo_graph_release"]

_OBO_BASE = "http://purl.obolibrary.org/obo/"
_REPLACED_BY = f"{_OBO_BASE}IAO_0100001"
_SYNONYM_SCOPES = {
    "hasExactSynonym": "EXACT",
    "hasBroadSynonym": "BROAD",
    "hasNarrowSynonym": "NARROW",
    "hasRelatedSynonym": "RELATED",
}


@dataclass(frozen=True)
class OBOGraphCompilationConfig:
    """Identity and typing contract for one OBO Graph namespace."""

    source_id: str
    source_version: str
    iri_prefix: str
    code_system: CodeSystem
    entity_type: EntityType

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.source_version.strip():
            raise ValueError("Ontology source_id and source_version must be non-empty")
        if not self.iri_prefix.strip() or ":" in self.iri_prefix:
            raise ValueError("Ontology iri_prefix must be a plain OBO prefix such as MONDO or HP")
        if not code_system_valid_for_entity_type(self.entity_type, self.code_system):
            # INVARIANT: an ontology import cannot bypass the same code/type safety enforced for
            # pipeline entities and retrieval candidates.
            raise ValueError(
                f"Code system {self.code_system.value} is invalid for {self.entity_type.value}"
            )


def compile_obo_graph_release(
    *,
    input_path: str | Path,
    output_dir: str | Path,
    config: OBOGraphCompilationConfig,
) -> dict[str, Any]:
    """Stream a release into rich concepts, query terminology, and canonical hierarchy.

    The JSON file is scanned twice. The first pass retains only active IDs. The second materializes
    source records after hierarchy parents are known. This keeps memory proportional to the target
    namespace rather than the complete imported ontology graph.
    """

    source_path = Path(input_path)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()

    active_ids, namespace_ids = _collect_namespace_ids(source_path, config, counters)
    parent_ids, edge_records, evidence_records = _compile_hierarchy(
        source_path,
        config,
        active_ids=active_ids,
        namespace_ids=namespace_ids,
        counters=counters,
    )

    concepts_path = target / "concepts.jsonl"
    terminology_path = target / "terminology.jsonl"
    nodes_path = target / "nodes.jsonl"
    edges_path = target / "edges.jsonl"
    evidence_path = target / "evidence.jsonl"
    report_path = target / "report.json"

    concepts_sha = write_jsonl(
        concepts_path,
        _iter_concept_records(source_path, config, parent_ids, counters),
    )
    terminology_sha = write_jsonl(
        terminology_path,
        _iter_terminology_records(concepts_path),
    )
    nodes_sha = write_jsonl(nodes_path, _iter_node_records(concepts_path))
    edges_sha = write_jsonl(edges_path, (edge.to_dict() for edge in edge_records))
    evidence_sha = write_jsonl(
        evidence_path,
        (evidence.to_dict() for evidence in evidence_records),
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "id": config.source_id,
            "version": config.source_version,
            "input": str(source_path),
            "input_sha256": sha256_file(source_path),
            "iri_prefix": config.iri_prefix,
            "code_system": config.code_system.value,
            "entity_type": config.entity_type.value,
        },
        "counts": dict(sorted(counters.items())),
        "outputs": {
            "concepts": {"path": str(concepts_path), "sha256": concepts_sha},
            "terminology": {"path": str(terminology_path), "sha256": terminology_sha},
            "nodes": {"path": str(nodes_path), "sha256": nodes_sha},
            "edges": {"path": str(edges_path), "sha256": edges_sha},
            "evidence": {"path": str(evidence_path), "sha256": evidence_sha},
        },
        "promotion": {
            "runtime_default": False,
            "reason": (
                "Ontology labels are source-language terminology. Runtime recognition requires "
                "a separate reviewed alias policy and held-out benchmark."
            ),
        },
    }
    write_json(report_path, report)
    return report


def _collect_namespace_ids(
    path: Path,
    config: OBOGraphCompilationConfig,
    counters: Counter[str],
) -> tuple[set[str], set[str]]:
    active: set[str] = set()
    namespace: set[str] = set()
    for raw in _iter_obo_items(path, "nodes"):
        counters["all_graph_node_count"] += 1
        curie = _target_curie(raw.get("id"), config.iri_prefix)
        if curie is None:
            continue
        counters["namespace_node_count"] += 1
        if curie in namespace:
            raise ValueError(f"Duplicate ontology concept ID {curie!r}")
        namespace.add(curie)
        if _is_deprecated(raw):
            counters["deprecated_node_count"] += 1
            continue
        if not _clean_string(raw.get("lbl")):
            counters["active_missing_label_count"] += 1
            continue
        active.add(curie)
    counters["active_node_count"] = len(active)
    return active, namespace


def _compile_hierarchy(
    path: Path,
    config: OBOGraphCompilationConfig,
    *,
    active_ids: set[str],
    namespace_ids: set[str],
    counters: Counter[str],
) -> tuple[dict[str, set[str]], tuple[KnowledgeEdge, ...], tuple[KnowledgeEvidence, ...]]:
    parents: dict[str, set[str]] = defaultdict(set)
    edges: dict[tuple[str, str], KnowledgeEdge] = {}
    evidence: dict[tuple[str, str], KnowledgeEvidence] = {}
    source = f"ontology:{config.source_id}:{config.source_version}"
    for raw in _iter_obo_items(path, "edges"):
        counters["all_graph_edge_count"] += 1
        if raw.get("pred") != "is_a":
            counters["non_hierarchy_edge_count"] += 1
            continue
        child = _target_curie(raw.get("sub"), config.iri_prefix)
        parent = _target_curie(raw.get("obj"), config.iri_prefix)
        if child is None or parent is None:
            counters["external_hierarchy_edge_count"] += 1
            continue
        if child not in active_ids or parent not in active_ids:
            if child in namespace_ids and parent in namespace_ids:
                counters["inactive_hierarchy_edge_count"] += 1
            else:
                counters["dangling_hierarchy_edge_count"] += 1
            continue
        if child == parent:
            counters["self_hierarchy_edge_count"] += 1
            continue
        key = (child, parent)
        if key in edges:
            counters["duplicate_hierarchy_edge_count"] += 1
            continue
        parents[child].add(parent)
        child_code = _curie_code(child)
        parent_code = _curie_code(parent)
        head_id = concept_node_id(config.code_system.value, child_code)
        tail_id = concept_node_id(config.code_system.value, parent_code)
        edge_id = _edge_id(head_id, "IS_A", tail_id)
        edges[key] = KnowledgeEdge(
            edge_id=edge_id,
            head_node_id=head_id,
            tail_node_id=tail_id,
            relation_type="IS_A",
            support_count=1,
            document_count=0,
            confidence_mean=1.0,
            confidence_min=1.0,
            confidence_max=1.0,
            sources=(source,),
            layers=("canonical",),
        )
        source_record_id = f"{child}|is_a|{parent}"
        evidence[key] = KnowledgeEvidence(
            evidence_id=_evidence_id(edge_id, source_record_id),
            edge_id=edge_id,
            source_record_id=source_record_id,
            source_record_kind="ontology_parent",
            source=source,
        )
    counters["hierarchy_edge_count"] = len(edges)
    ordered_keys = sorted(edges)
    return parents, tuple(edges[key] for key in ordered_keys), tuple(evidence[key] for key in ordered_keys)


def _iter_concept_records(
    path: Path,
    config: OBOGraphCompilationConfig,
    parent_ids: Mapping[str, set[str]],
    counters: Counter[str],
) -> Iterator[dict[str, Any]]:
    xref_namespaces: Counter[str] = Counter()
    synonym_scopes: Counter[str] = Counter()
    for raw in _iter_obo_items(path, "nodes"):
        curie = _target_curie(raw.get("id"), config.iri_prefix)
        if curie is None:
            continue
        label = _clean_string(raw.get("lbl"))
        meta = _mapping(raw.get("meta"))
        deprecated = bool(meta.get("deprecated", False))
        synonyms = _synonym_records(meta.get("synonyms"))
        for synonym in synonyms:
            synonym_scopes[str(synonym["scope"])] += 1
        xrefs = _xref_values(meta.get("xrefs"))
        for xref in xrefs:
            xref_namespaces[_curie_namespace(xref)] += 1
        replacements = _replacement_ids(meta.get("basicPropertyValues"))
        if replacements:
            counters["replacement_node_count"] += 1
        definition, definition_xrefs = _definition(meta.get("definition"))
        aliases = _unique_strings(
            str(synonym["text"])
            for synonym in synonyms
            if str(synonym["text"]) != label
        )
        parents = tuple(sorted(parent_ids.get(curie, ())))
        properties = _property_records(meta.get("basicPropertyValues"))
        counters["concept_record_count"] += 1
        yield {
            "concept_id": curie,
            "code": _curie_code(curie),
            "code_system": config.code_system.value,
            "canonical_name": label,
            "semantic_type": config.entity_type.value,
            "aliases": list(aliases),
            "synonyms": synonyms,
            "definition": definition,
            "definition_xrefs": list(definition_xrefs),
            "xrefs": list(xrefs),
            "parents": list(parents),
            "parent_codes": [_curie_code(parent) for parent in parents],
            "subsets": list(_string_values(meta.get("subsets"))),
            "properties": properties,
            "deprecated": deprecated,
            "replacement_ids": list(replacements),
            "terminology_eligible": bool(label and not deprecated),
            "source": config.source_id,
            "source_version": config.source_version,
        }
    counters["xref_count"] = sum(xref_namespaces.values())
    counters["synonym_count"] = sum(synonym_scopes.values())
    for namespace, count in sorted(xref_namespaces.items()):
        counters[f"xref_namespace:{namespace}"] = count
    for scope, count in sorted(synonym_scopes.items()):
        counters[f"synonym_scope:{scope}"] = count


def _iter_terminology_records(path: Path) -> Iterator[dict[str, Any]]:
    for raw in _iter_jsonl(path):
        if not raw.get("terminology_eligible"):
            continue
        yield {
            "concept_id": raw["concept_id"],
            "code": raw["code"],
            "code_system": raw["code_system"],
            "canonical_name": raw["canonical_name"],
            "official_name_en": raw["canonical_name"],
            "semantic_type": raw["semantic_type"],
            "aliases": raw["aliases"],
            "parents": raw["parent_codes"],
            "source": f"{raw['source']}:{raw['source_version']}",
        }


def _iter_node_records(path: Path) -> Iterator[dict[str, Any]]:
    for raw in _iter_jsonl(path):
        if not raw.get("terminology_eligible"):
            continue
        label = str(raw["canonical_name"])
        node = KnowledgeNode(
            node_id=concept_node_id(str(raw["code_system"]), str(raw["code"])),
            kind=KnowledgeNodeKind.CONCEPT,
            label=label,
            normalized_label=normalize_for_match(label),
            entity_type=str(raw["semantic_type"]),
            code_system=str(raw["code_system"]),
            code=str(raw["code"]),
            aliases=tuple(str(value) for value in raw.get("aliases", [])),
            terminology_versions=(str(raw["source_version"]),),
            sources=(f"ontology:{raw['source']}:{raw['source_version']}",),
        )
        yield node.to_dict()


def _iter_obo_items(path: Path, array_name: str) -> Iterator[Mapping[str, Any]]:
    """Stream one OBO Graph array without loading imported ontologies into memory."""

    try:
        ijson = importlib.import_module("ijson")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Streaming ontology compilation requires the 'data' extra: "
            "uv sync --extra data"
        ) from error
    # SCALING: Mondo JSON includes imported classes and exceeds 100 MB. ijson keeps peak memory
    # bounded while allowing independent node and edge passes over the immutable source object.
    with path.open("rb") as handle:
        items = ijson.items(handle, f"graphs.item.{array_name}.item")
        for raw in items:
            if not isinstance(raw, Mapping):
                raise ValueError(f"OBO Graph {array_name} entries must be objects")
            yield cast(Mapping[str, Any], raw)


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield cast(Mapping[str, Any], raw)


def _target_curie(value: object, prefix: str) -> str | None:
    text = _clean_string(value)
    if text is None:
        return None
    iri_prefix = f"{_OBO_BASE}{prefix}_"
    if text.startswith(iri_prefix):
        return f"{prefix}:{text.removeprefix(iri_prefix)}"
    if text.startswith(f"{prefix}:"):
        return text
    return None


def _curie_code(curie: str) -> str:
    return curie.partition(":")[2]


def _curie_namespace(value: str) -> str:
    namespace, separator, _ = value.partition(":")
    return namespace if separator else "URI_OR_UNSCOPED"


def _is_deprecated(raw: Mapping[str, Any]) -> bool:
    return bool(_mapping(raw.get("meta")).get("deprecated", False))


def _synonym_records(value: object) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return output
    for raw in value:
        row = _mapping(raw)
        text = _clean_string(row.get("val"))
        if text is None:
            continue
        predicate = _clean_string(row.get("pred")) or "unknown"
        output.append(
            {
                "text": text,
                "scope": _SYNONYM_SCOPES.get(predicate, "UNKNOWN"),
                "predicate": predicate,
                "xrefs": list(_string_values(row.get("xrefs"))),
            }
        )
    return sorted(output, key=lambda row: (normalize_for_match(str(row["text"])), str(row["text"])))


def _xref_values(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return _unique_strings(
        text
        for raw in value
        if (text := _clean_string(_mapping(raw).get("val"))) is not None
    )


def _replacement_ids(value: object) -> tuple[str, ...]:
    return _unique_strings(
        str(row["value"])
        for row in _property_records(value)
        if row["predicate"] == _REPLACED_BY
    )


def _property_records(value: object) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return output
    for raw in value:
        row = _mapping(raw)
        predicate = _clean_string(row.get("pred"))
        property_value = _clean_string(row.get("val"))
        if predicate is None or property_value is None:
            continue
        output.append(
            {
                "predicate": predicate,
                "value": property_value,
                "xrefs": list(_string_values(row.get("xrefs"))),
            }
        )
    return sorted(output, key=lambda row: (str(row["predicate"]), str(row["value"])))


def _definition(value: object) -> tuple[str | None, tuple[str, ...]]:
    row = _mapping(value)
    return _clean_string(row.get("val")), _string_values(row.get("xrefs"))


def _mapping(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _string_values(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return _unique_strings(text for item in value if (text := _clean_string(item)) is not None)


def _unique_strings(values: Iterator[str]) -> tuple[str, ...]:
    unique = {value.strip() for value in values if value.strip()}
    return tuple(sorted(unique, key=lambda value: (normalize_for_match(value), value)))


def _clean_string(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _edge_id(head_node_id: str, relation_type: str, tail_node_id: str) -> str:
    identity = "\x1f".join((head_node_id, relation_type, tail_node_id))
    return f"edge:{sha256_text(identity)[:24]}"


def _evidence_id(edge_id: str, source_record_id: str) -> str:
    identity = "\x1f".join((edge_id, "ontology_parent", source_record_id))
    return f"evidence:{sha256_text(identity)[:24]}"
