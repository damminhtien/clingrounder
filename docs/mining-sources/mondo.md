# Mondo Disease Ontology

## Identity And Acquisition

- Source: Mondo Disease Ontology.
- Release: `v2026-07-06`.
- Registry access: `open`, CC BY 4.0, attribution required.
- Acquisition plan: `configs/mining/mondo-2026-07-06.yaml`.
- Release JSON: 107,273,669 bytes, SHA-256
  `80b8658b4ec7da7699f7f8f6460425396e42f1bb7fbead837469ebf1907f7c30`.
- Upstream source manifest: 865 bytes, SHA-256
  `3bc9cf0369b257803bd79b34cfab98346c385569aaaf9af6b742f0af39ec0e98`.

The source manifest is retained because Mondo integrates independently versioned inputs. This
release records, among others, DOID `2026-05-30`, ICD-10-CM `2024ab`, ICD-11 Foundation
`2025-01-26`, NCIt `26.02d`, OMIM `2026-06-03`, and ORDO `4.7`. Those versions are provenance, not
permission to treat every Mondo xref as an exact replacement code.

Both artifacts live in the content-addressed mining store. `outputs/mining/mondo-2026-07-06` is an
acquisition manifest only; ontology-derived artifacts live under
`outputs/mining/knowledge/mondo-2026-07-06`.

## Processing

`clingrounder-research data ontology compile-obo` streams the OBO Graph JSON with `ijson`. It scans the target
namespace separately from imported classes, then materializes:

| Artifact | Purpose | Rows |
| --- | --- | ---: |
| `concepts.jsonl` | rich source record, including deprecated concepts | 36,072 |
| `terminology.jsonl` | active query terminology | 32,097 |
| `nodes.jsonl` | canonical MONDO graph nodes | 32,097 |
| `edges.jsonl` | active MONDO-to-MONDO `IS_A` edges | 46,447 |
| `evidence.jsonl` | one source pointer per hierarchy edge | 46,447 |

The rich concept record retains the label, definition and definition xrefs, synonym text/scope,
xrefs, subsets, basic properties, parent IDs, deprecation flag, and replacement IDs. This is
deliberately richer than `DictionaryStore`; current runtime fields do not determine what source
evidence is preserved.

Deprecated concepts remain in `concepts.jsonl` for migration audit, but cannot enter terminology or
graph nodes. Imported classes and cross-namespace hierarchy edges are counted but not silently
converted into MONDO concepts. The raw source remains immutable, so a future inter-ontology graph
can recover them without downloading a different release.

## Observed Coverage And Quality

| Measure | Count |
| --- | ---: |
| all graph nodes, including imports | 63,848 |
| MONDO concepts | 36,072 |
| active MONDO concepts | 32,097 |
| deprecated concepts | 3,975 |
| concepts with replacements | 2,228 |
| synonyms | 95,023 |
| exact / broad / narrow / related synonyms | 72,073 / 1,478 / 2,535 / 18,937 |
| xrefs | 147,587 |
| internal `IS_A` edges | 46,447 |
| hierarchy edges outside the target namespace | 34,825 |
| non-`IS_A` OBO Graph edges excluded from this hierarchy | 32,413 |

High-volume xref namespaces include UMLS and MEDGEN (21,502 each), GARD (15,947), DOID (12,091),
Orphanet (10,491), OMIM (10,176), SCTID (9,157), MESH (8,211), and NCIt (7,434). Xrefs are coverage
and review evidence. The compiler does not assume that a same-row xref is automatically equivalent
under the repository's target coding convention.

No duplicate concept ID was accepted. A duplicate ID is a hard compiler error; duplicate or self
hierarchy edges are counted and skipped. The complete report is
`outputs/mining/knowledge/mondo-2026-07-06/report.json`.

## Database And Retrieval

Two derived indexes were built successfully:

- SQLite FTS5 terminology: 32,097 concepts, 113,886 aliases, 90 MB.
- SQLite knowledge graph: 32,097 nodes, 46,447 edges/evidence rows, 112,849 normalized aliases,
  92 MB.

An exact/FTS query for `systemic lupus erythematosus` returns `MONDO:0007915` first while retaining
more specific SLE concepts below it. The `IS_A` index consistency benchmark covered 46,447/46,447
edges, was deterministic with eight workers, and measured 3,192 queries/second with p95 4.66 ms.
This checks storage and concurrent lookup, not clinical correctness or reranker gain.

## How The Source Is Used

- **Terminology:** English disease labels and scoped synonyms are queryable in an opt-in MONDO
  index. They are not copied into the Vietnamese recognition dictionary.
- **Ontology/KG:** active `IS_A` edges support ancestor distance, sibling discovery, rare-disease
  coverage planning, and offline graph-feature experiments.
- **Crosswalk review:** OMIM, Orphanet, UMLS, ICD, SNOMED and other xrefs form a bounded review queue;
  none are exported as a candidate solely because an xref exists.
- **Model data:** parent/sibling groups can create ontology-aware hard negatives and source-held-out
  rare-disease slices. Mondo labels are not clinical-note spans.
- **Deduplication:** MONDO CURIE is the canonical source identity; deprecated CURIEs are redirected
  only after explicit replacement review.

### PMC Case Evidence Integration

Exact case-specific disease mentions from the PMC rare-case tranche produced 41 unique Mondo
mappings across 124 occurrences. The compiler selects Mondo as the canonical `DISEASE` endpoint
while preserving an annotation's original ICD-10 link for audit. The resulting disease-symptom
edges are neutral source-block co-occurrences, not Mondo ontology axioms or causal facts. They stay
outside runtime defaults until evaluated on independent human-linked data; see
`docs/mining-sources/pmc-oa.md` for counts, hashes and reproduction commands.

## Promotion Boundary

The source is `curated / training_only`. The terminology and graph indexes are reproducible but are
not runtime defaults. Promotion requires:

1. reviewed Vietnamese aliases or model queries rather than uncalibrated English NER aliases;
2. a typed MONDO-to-target-code crosswalk with ambiguous and deprecated mappings rejected;
3. a source-held-out retrieval/reranker benchmark showing gain without invalid code emission;
4. explicit config selection so Phase 1 ICD/RxNorm behavior cannot change implicitly.

## Reproduce

```bash
export CLINGROUNDER_ARTIFACT_STORE=/Volumes/clingrounder-mining

uv run clingrounder-research data run --plan configs/mining/mondo-2026-07-06.yaml

uv run clingrounder-research data ontology compile-obo \
  --input "$MONDO_JSON_OBJECT" \
  --output-dir outputs/mining/knowledge/mondo-2026-07-06 \
  --source-id mondo --source-version 2026-07-06 \
  --iri-prefix MONDO --code-system MONDO --entity-type DISEASE

uv run clingrounder terminology build \
  --source outputs/mining/knowledge/mondo-2026-07-06/terminology.jsonl \
  --output outputs/mining/knowledge/mondo-2026-07-06/terminology.sqlite3 \
  --manifest-output outputs/mining/knowledge/mondo-2026-07-06/terminology_manifest.json

uv run clingrounder kg build \
  --nodes outputs/mining/knowledge/mondo-2026-07-06/nodes.jsonl \
  --edges outputs/mining/knowledge/mondo-2026-07-06/edges.jsonl \
  --evidence outputs/mining/knowledge/mondo-2026-07-06/evidence.jsonl \
  --output outputs/mining/knowledge/mondo-2026-07-06/graph.sqlite3 \
  --manifest-output outputs/mining/knowledge/mondo-2026-07-06/graph_manifest.json
```

`$MONDO_JSON_OBJECT` is resolved from `artifacts.jsonl`; do not replace it with an unverified latest
URL.
