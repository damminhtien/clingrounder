"""Compile HPO disease-phenotype and disease-gene tables without losing evidence fields."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, TextIO, cast

from medical_kg_nlp.kg.knowledge_schema import (
    KnowledgeEdge,
    KnowledgeEvidence,
    KnowledgeNode,
    KnowledgeNodeKind,
)
from medical_kg_nlp.mining.graph_knowledge import concept_node_id
from medical_kg_nlp.mining.io import write_json, write_jsonl
from medical_kg_nlp.utils.hashing import sha256_file, sha256_text
from medical_kg_nlp.utils.text import normalize_for_match

__all__ = ["compile_hpo_associations"]

_HPOA_FIELDS = (
    "database_id",
    "disease_name",
    "qualifier",
    "hpo_id",
    "reference",
    "evidence",
    "onset",
    "frequency",
    "sex",
    "modifier",
    "aspect",
    "biocuration",
)
_GENE_FIELDS = ("ncbi_gene_id", "gene_symbol", "association_type", "disease_id", "source")


def compile_hpo_associations(
    *,
    hpoa_path: str | Path,
    genes_path: str | Path,
    hpo_concepts_path: str | Path,
    output_dir: str | Path,
    source_version: str,
) -> dict[str, Any]:
    """Compile source rows, graph records, and a quality report for one HPO release.

    Edge aggregation uses a temporary SQLite table. The release contains hundreds of thousands of
    observations; disk-backed aggregation avoids retaining one Python object per edge while still
    preserving every raw source row in JSONL.
    """

    if not source_version.strip():
        raise ValueError("HPO source_version must be non-empty")
    hpoa = Path(hpoa_path)
    genes = Path(genes_path)
    hpo_concepts = Path(hpo_concepts_path)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    disease_labels: dict[str, Counter[str]] = defaultdict(Counter)
    gene_labels: dict[str, Counter[str]] = defaultdict(Counter)
    hpo_nodes = _load_hpo_nodes(hpo_concepts)

    associations_path = target / "phenotype_associations.jsonl"
    gene_associations_path = target / "gene_associations.jsonl"
    nodes_path = target / "nodes.jsonl"
    edges_path = target / "edges.jsonl"
    evidence_path = target / "evidence.jsonl"
    report_path = target / "report.json"

    descriptor, aggregate_name = tempfile.mkstemp(
        prefix=".hpo-associations.", suffix=".sqlite3", dir=target
    )
    os.close(descriptor)
    connection = sqlite3.connect(aggregate_name)
    try:
        _initialize_aggregate(connection)
        associations_sha = write_jsonl(
            associations_path,
            _iter_hpoa_records(
                hpoa,
                known_hpo_ids=set(hpo_nodes),
                source_version=source_version,
                connection=connection,
                disease_labels=disease_labels,
                counters=counters,
            ),
        )
        gene_associations_sha = write_jsonl(
            gene_associations_path,
            _iter_gene_records(
                genes,
                source_version=source_version,
                connection=connection,
                disease_labels=disease_labels,
                gene_labels=gene_labels,
                counters=counters,
            ),
        )
        connection.commit()
        nodes_sha = write_jsonl(
            nodes_path,
            _iter_graph_nodes(
                hpo_nodes,
                disease_labels=disease_labels,
                gene_labels=gene_labels,
                source_version=source_version,
            ),
        )
        edges_sha = write_jsonl(edges_path, _iter_graph_edges(connection, source_version))
        evidence_sha = write_jsonl(
            evidence_path,
            _iter_graph_evidence(
                associations_path,
                gene_associations_path,
                source_version=source_version,
            ),
        )
        counters["graph_node_count"] = len(hpo_nodes) + len(disease_labels) + len(gene_labels)
        counters["graph_edge_count"] = _edge_count(connection)
    finally:
        connection.close()
        Path(aggregate_name).unlink(missing_ok=True)

    report: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "id": "hpo",
            "version": source_version,
            "hpoa": {"path": str(hpoa), "sha256": sha256_file(hpoa)},
            "genes": {"path": str(genes), "sha256": sha256_file(genes)},
            "hpo_concepts": {
                "path": str(hpo_concepts),
                "sha256": sha256_file(hpo_concepts),
            },
        },
        "counts": dict(sorted(counters.items())),
        "outputs": {
            "phenotype_associations": {
                "path": str(associations_path),
                "sha256": associations_sha,
            },
            "gene_associations": {
                "path": str(gene_associations_path),
                "sha256": gene_associations_sha,
            },
            "nodes": {"path": str(nodes_path), "sha256": nodes_sha},
            "edges": {"path": str(edges_path), "sha256": edges_sha},
            "evidence": {"path": str(evidence_path), "sha256": evidence_sha},
        },
        "semantics": {
            "HAS_PHENOTYPE": "Positive source assertion only.",
            "NOT_HAS_PHENOTYPE": "Explicit HPOA NOT qualifier; never a positive phenotype edge.",
            "ASSOCIATED_GENE": (
                "Source-listed disease-gene association; association_type remains in the raw row."
            ),
        },
        "promotion": {
            "runtime_default": False,
            "reason": (
                "Association evidence is suitable for coverage and graph-feature experiments, "
                "not direct clinical assertion or candidate generation."
            ),
        },
    }
    write_json(report_path, report)
    return report


def _iter_hpoa_records(
    path: Path,
    *,
    known_hpo_ids: set[str],
    source_version: str,
    connection: sqlite3.Connection,
    disease_labels: dict[str, Counter[str]],
    counters: Counter[str],
) -> Iterator[dict[str, Any]]:
    for row_number, raw in enumerate(_iter_tsv(path, _HPOA_FIELDS, comments=True), start=1):
        row = {field: raw.get(field, "").strip() for field in _HPOA_FIELDS}
        disease_id = row["database_id"]
        hpo_id = row["hpo_id"]
        qualifier = row["qualifier"]
        if not _valid_curie(disease_id) or not _valid_curie(hpo_id):
            counters["invalid_phenotype_row_count"] += 1
            continue
        association_id = _source_record_id("hpoa", row_number, row)
        duplicate = _record_signature(connection, "hpoa", row)
        counters["phenotype_association_count"] += 1
        counters[f"disease_namespace:{_curie_parts(disease_id)[0]}"] += 1
        counters[f"evidence_code:{row['evidence'] or 'EMPTY'}"] += 1
        counters[f"aspect:{row['aspect'] or 'EMPTY'}"] += 1
        if row["onset"]:
            counters["onset_qualified_count"] += 1
        if row["frequency"]:
            counters["frequency_qualified_count"] += 1
        if duplicate:
            counters["duplicate_phenotype_row_count"] += 1
        if row["disease_name"]:
            disease_labels[disease_id][row["disease_name"]] += 1

        graph_eligible = True
        if hpo_id not in known_hpo_ids:
            graph_eligible = False
            counters["unknown_hpo_id_count"] += 1
        if qualifier == "NOT":
            relation_type = "NOT_HAS_PHENOTYPE"
            counters["negated_phenotype_association_count"] += 1
        elif not qualifier:
            relation_type = "HAS_PHENOTYPE"
            counters["positive_phenotype_association_count"] += 1
        else:
            relation_type = ""
            graph_eligible = False
            counters[f"unsupported_qualifier:{qualifier}"] += 1
        if graph_eligible:
            _record_edge(connection, disease_id, relation_type, hpo_id)
        yield {
            "association_id": association_id,
            **row,
            "relation_type": relation_type or None,
            "graph_eligible": graph_eligible,
            "source": "hpo",
            "source_version": source_version,
        }


def _iter_gene_records(
    path: Path,
    *,
    source_version: str,
    connection: sqlite3.Connection,
    disease_labels: dict[str, Counter[str]],
    gene_labels: dict[str, Counter[str]],
    counters: Counter[str],
) -> Iterator[dict[str, Any]]:
    for row_number, raw in enumerate(_iter_tsv(path, _GENE_FIELDS), start=1):
        row = {field: raw.get(field, "").strip() for field in _GENE_FIELDS}
        gene_id = row["ncbi_gene_id"]
        disease_id = row["disease_id"]
        if not _valid_curie(gene_id) or not _valid_curie(disease_id):
            counters["invalid_gene_row_count"] += 1
            continue
        association_id = _source_record_id("gene", row_number, row)
        duplicate = _record_signature(connection, "gene", row)
        counters["gene_association_count"] += 1
        counters[f"gene_association_type:{row['association_type'] or 'EMPTY'}"] += 1
        counters[f"gene_disease_namespace:{_curie_parts(disease_id)[0]}"] += 1
        if duplicate:
            counters["duplicate_gene_row_count"] += 1
        if row["gene_symbol"]:
            gene_labels[gene_id][row["gene_symbol"]] += 1
        disease_labels.setdefault(disease_id, Counter())
        _record_edge(connection, disease_id, "ASSOCIATED_GENE", gene_id)
        yield {
            "association_id": association_id,
            **row,
            "relation_type": "ASSOCIATED_GENE",
            "graph_eligible": True,
            "source": "hpo",
            "source_version": source_version,
        }


def _load_hpo_nodes(path: Path) -> dict[str, KnowledgeNode]:
    nodes: dict[str, KnowledgeNode] = {}
    for raw in _iter_jsonl(path):
        if not raw.get("terminology_eligible"):
            continue
        concept_id = str(raw["concept_id"])
        if not concept_id.startswith("HP:"):
            raise ValueError(f"Expected HP concept, found {concept_id!r}")
        code = str(raw["code"])
        label = str(raw["canonical_name"])
        nodes[concept_id] = KnowledgeNode(
            node_id=concept_node_id("HPO", code),
            kind=KnowledgeNodeKind.CONCEPT,
            label=label,
            normalized_label=normalize_for_match(label),
            entity_type="FINDING",
            code_system="HPO",
            code=code,
            aliases=tuple(str(value) for value in raw.get("aliases", [])),
            terminology_versions=(str(raw["source_version"]),),
            sources=(f"ontology:hpo:{raw['source_version']}",),
        )
    return nodes


def _iter_graph_nodes(
    hpo_nodes: Mapping[str, KnowledgeNode],
    *,
    disease_labels: Mapping[str, Counter[str]],
    gene_labels: Mapping[str, Counter[str]],
    source_version: str,
) -> Iterator[dict[str, Any]]:
    yield from (hpo_nodes[curie].to_dict() for curie in sorted(hpo_nodes))
    source = f"ontology-association:hpo:{source_version}"
    for curie in sorted(disease_labels):
        namespace, code = _curie_parts(curie)
        label = _preferred_label(disease_labels[curie], fallback=curie)
        yield KnowledgeNode(
            node_id=concept_node_id(namespace, code),
            kind=KnowledgeNodeKind.CONCEPT,
            label=label,
            normalized_label=normalize_for_match(label),
            entity_type="DISEASE",
            code_system=namespace,
            code=code,
            sources=(source,),
        ).to_dict()
    for curie in sorted(gene_labels):
        namespace, code = _curie_parts(curie)
        label = _preferred_label(gene_labels[curie], fallback=curie)
        yield KnowledgeNode(
            node_id=concept_node_id(namespace, code),
            kind=KnowledgeNodeKind.CONCEPT,
            label=label,
            normalized_label=normalize_for_match(label),
            entity_type="GENE",
            code_system=namespace,
            code=code,
            sources=(source,),
        ).to_dict()


def _iter_graph_edges(
    connection: sqlite3.Connection,
    source_version: str,
) -> Iterator[dict[str, Any]]:
    source = f"ontology-association:hpo:{source_version}"
    rows = connection.execute(
        "SELECT head_curie, relation_type, tail_curie, support_count "
        "FROM edge_aggregate ORDER BY head_curie, relation_type, tail_curie"
    )
    for head_curie, relation_type, tail_curie, support_count in rows:
        head_namespace, head_code = _curie_parts(str(head_curie))
        tail_namespace, tail_code = _curie_parts(str(tail_curie))
        head_id = concept_node_id(head_namespace, head_code)
        tail_id = concept_node_id("HPO" if tail_namespace == "HP" else tail_namespace, tail_code)
        yield KnowledgeEdge(
            edge_id=_edge_id(head_id, str(relation_type), tail_id),
            head_node_id=head_id,
            tail_node_id=tail_id,
            relation_type=str(relation_type),
            support_count=int(support_count),
            document_count=0,
            confidence_mean=1.0,
            confidence_min=1.0,
            confidence_max=1.0,
            sources=(source,),
            layers=("canonical",),
        ).to_dict()


def _iter_graph_evidence(
    associations_path: Path,
    gene_associations_path: Path,
    *,
    source_version: str,
) -> Iterator[dict[str, Any]]:
    source = f"ontology-association:hpo:{source_version}"
    for path, kind in (
        (associations_path, "hpoa"),
        (gene_associations_path, "disease_gene"),
    ):
        for raw in _iter_jsonl(path):
            if not raw.get("graph_eligible"):
                continue
            head_curie = str(raw["database_id"] if kind == "hpoa" else raw["disease_id"])
            tail_curie = str(raw["hpo_id"] if kind == "hpoa" else raw["ncbi_gene_id"])
            relation_type = str(raw["relation_type"])
            head_namespace, head_code = _curie_parts(head_curie)
            tail_namespace, tail_code = _curie_parts(tail_curie)
            head_id = concept_node_id(head_namespace, head_code)
            tail_id = concept_node_id("HPO" if tail_namespace == "HP" else tail_namespace, tail_code)
            edge_id = _edge_id(head_id, relation_type, tail_id)
            source_record_id = str(raw["association_id"])
            yield KnowledgeEvidence(
                evidence_id=_evidence_id(edge_id, kind, source_record_id),
                edge_id=edge_id,
                source_record_id=source_record_id,
                source_record_kind=kind,
                source=source,
            ).to_dict()


def _initialize_aggregate(connection: sqlite3.Connection) -> None:
    # SCALING: SQLite performs bounded-memory deduplication and edge counting for 285k+ HPOA rows.
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE edge_aggregate (
            head_curie TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            tail_curie TEXT NOT NULL,
            support_count INTEGER NOT NULL,
            PRIMARY KEY (head_curie, relation_type, tail_curie)
        ) WITHOUT ROWID;
        CREATE TABLE row_signatures (
            source_kind TEXT NOT NULL,
            signature TEXT NOT NULL,
            PRIMARY KEY (source_kind, signature)
        ) WITHOUT ROWID;
        """
    )


def _record_edge(
    connection: sqlite3.Connection,
    head_curie: str,
    relation_type: str,
    tail_curie: str,
) -> None:
    connection.execute(
        "INSERT INTO edge_aggregate(head_curie, relation_type, tail_curie, support_count) "
        "VALUES (?, ?, ?, 1) ON CONFLICT(head_curie, relation_type, tail_curie) "
        "DO UPDATE SET support_count = support_count + 1",
        (head_curie, relation_type, tail_curie),
    )


def _record_signature(
    connection: sqlite3.Connection,
    source_kind: str,
    row: Mapping[str, str],
) -> bool:
    signature = sha256_text(json.dumps(row, sort_keys=True, ensure_ascii=False))
    cursor = connection.execute(
        "INSERT OR IGNORE INTO row_signatures(source_kind, signature) VALUES (?, ?)",
        (source_kind, signature),
    )
    return cursor.rowcount == 0


def _edge_count(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT COUNT(*) FROM edge_aggregate").fetchone()
    return 0 if row is None else int(row[0])


def _iter_tsv(
    path: Path,
    expected_fields: tuple[str, ...],
    *,
    comments: bool = False,
) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        source: TextIO | Iterator[str]
        source = (line for line in handle if not line.startswith("#")) if comments else handle
        reader = csv.DictReader(source, delimiter="\t")
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(
                f"Unexpected columns in {path}: {reader.fieldnames!r}; expected {expected_fields!r}"
            )
        for row in reader:
            yield {str(key): "" if value is None else str(value) for key, value in row.items()}


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield cast(Mapping[str, Any], raw)


def _source_record_id(kind: str, row_number: int, row: Mapping[str, str]) -> str:
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False)
    return f"{kind}:{row_number}:{sha256_text(payload)[:20]}"


def _valid_curie(value: str) -> bool:
    namespace, separator, code = value.partition(":")
    return bool(separator and namespace.strip() and code.strip())


def _curie_parts(value: str) -> tuple[str, str]:
    namespace, separator, code = value.partition(":")
    if not separator or not namespace or not code:
        raise ValueError(f"Expected CURIE, found {value!r}")
    return namespace, code


def _preferred_label(labels: Counter[str], *, fallback: str) -> str:
    if not labels:
        return fallback
    return min(labels, key=lambda label: (-labels[label], normalize_for_match(label), label))


def _edge_id(head_node_id: str, relation_type: str, tail_node_id: str) -> str:
    identity = "\x1f".join((head_node_id, relation_type, tail_node_id))
    return f"edge:{sha256_text(identity)[:24]}"


def _evidence_id(edge_id: str, kind: str, source_record_id: str) -> str:
    identity = "\x1f".join((edge_id, kind, source_record_id))
    return f"evidence:{sha256_text(identity)[:24]}"
