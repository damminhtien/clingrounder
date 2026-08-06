# Human Phenotype Ontology

## Identity And Acquisition

- Source: Human Phenotype Ontology (HPO) ontology and annotation releases.
- Release: `v2026-06-23`.
- Registry access: `open_with_terms`; retain source attribution and license evidence.
- Acquisition plan: `configs/mining/hpo-2026-06-23.yaml`.
- `hp.json`: 23,019,454 bytes, SHA-256
  `3b646565695329aa399e937883c68d5d424d0df5eaab2f22baa0e08d44fdbe87`.
- `phenotype.hpoa`: 35,672,303 bytes, SHA-256
  `89004f85b253f980ffe84218d2c080665cbf67a57bbb322111d6a2db5eb31dff`.
- `genes_to_disease.txt`: 1,477,762 bytes, SHA-256
  `a247027ae9944e34545e0a91060243ff6c118681c06379b9721af1ee4f39286a`.

The three immutable objects are processed separately because ontology hierarchy, phenotype
assertions, and disease-gene associations have different semantics and quality gates.

## Ontology Processing

The same streaming OBO Graph compiler used for Mondo targets only `HP:` classes and emits rich
concepts, terminology, KG nodes, hierarchy edges, evidence, and a report under
`outputs/mining/knowledge/hpo-2026-06-23/ontology`.

| Measure | Count |
| --- | ---: |
| HPO concepts | 20,413 |
| active concepts | 19,836 |
| deprecated concepts | 577 |
| concepts with replacements | 496 |
| synonyms | 26,237 |
| exact / broad / narrow / related synonyms | 23,642 / 561 / 545 / 1,489 |
| xrefs | 18,063 |
| active `IS_A` edges | 24,378 |

The rich rows preserve definitions, scoped synonyms, xrefs, parents, properties, deprecation, and
replacement IDs. Active concepts are typed as `FINDING` with code system `HPO`; HPO is permitted for
`FINDING` and `SYMPTOM`, but not for `DISEASE`, `DRUG`, or lab-result output.

The FTS5 terminology index contains 19,836 concepts and 44,812 aliases (34 MB). Querying `seizure`
returns `HP:0001250` first and preserves more specific seizure phenotypes below it. The hierarchy
graph index contains 19,836 nodes and 24,378 edges (41 MB). Its eight-worker consistency benchmark
covered all edges at 3,163 queries/second with p95 4.31 ms.

## Disease-Phenotype Processing

`medical-kg-research data ontology compile-hpo-associations` parses every HPOA field:

```text
database_id, disease_name, qualifier, hpo_id, reference, evidence,
onset, frequency, sex, modifier, aspect, biocuration
```

Every source row is written to `phenotype_associations.jsonl`. The graph uses only known active HPO
IDs and recognized qualifiers:

- empty qualifier -> `HAS_PHENOTYPE`;
- `NOT` -> `NOT_HAS_PHENOTYPE`;
- any other qualifier -> preserved in source rows but blocked from graph promotion.

This is an invariant: the 727 explicit `NOT` rows never become positive phenotype evidence.

| Measure | Count |
| --- | ---: |
| phenotype association rows | 285,598 |
| positive rows | 284,871 |
| explicit `NOT` rows | 727 |
| exact duplicate source rows | 32 |
| unknown HPO IDs | 0 |
| rows with frequency | 221,921 |
| rows with onset | 3,052 |
| OMIM / ORPHA / DECIPHER observations | 169,427 / 115,875 / 296 |
| TAS / PCS / IEA evidence observations | 136,275 / 118,774 / 30,549 |

Exact duplicate rows are detected in disk-backed SQLite aggregation and reported, but raw
provenance is retained. Multiple references for the same disease-phenotype pair aggregate into one
edge with a support count and separate evidence rows.

## Disease-Gene Processing

All five source columns are retained: `ncbi_gene_id`, `gene_symbol`, `association_type`,
`disease_id`, and `source`.

| Association type | Rows |
| --- | ---: |
| `MENDELIAN` | 7,075 |
| `POLYGENIC` | 581 |
| `UNKNOWN` | 8,288 |
| **Total** | **15,944** |

The graph relation is deliberately `ASSOCIATED_GENE` for every row. `association_type` remains on
the source record; the compiler does not relabel `UNKNOWN` or `POLYGENIC` evidence as causal.

## Association Graph And Performance

The association artifacts under `outputs/mining/knowledge/hpo-2026-06-23/associations` contain:

- 40,197 nodes: active HPO phenotypes plus referenced diseases and genes;
- 300,937 deduplicated edges;
- 301,542 evidence rows;
- 285,598 raw phenotype rows and 15,944 raw gene rows.

The SQLite graph index is 250 MB and validated every edge/evidence endpoint. A `HAS_PHENOTYPE`
consistency benchmark covered 284,267/284,267 positive edges and remained deterministic with eight
workers. It reached only 213 queries/second with p95 109.38 ms because many disease nodes have high
phenotype fan-out. This is a measured bottleneck: the full association graph is not suitable for an
unbounded online reranker query. Future runtime experiments must preselect diseases, cap neighbors,
or materialize compact phenotype signatures.

### PMC Case Evidence Integration

The PMC rare-case tranche queries case-specific `SYMPTOM` proposals against the pinned HPO index.
Twenty-nine normalized mentions matched one unique HPO concept, covering 103 occurrences. Those
links are appended to review proposals without replacing existing LOCAL links. After assertion and
source-block gates, ten disease-symptom observations produced nine neutral `CO_OCCURS_WITH` pairs
in the combined Mondo/HPO graph.

This integration does not add `HAS_PHENOTYPE` edges to HPO. Eight pairs have only one-document
support, and all endpoints originate from local bronze proposals rather than source human labels.
The graph is therefore useful for review and future held-out feature tests, not as a new canonical
phenotype-association release. Full counts, hashes and commands are in
`docs/mining-sources/pmc-oa.md`.

## How The Source Is Used

- **Terminology:** English phenotype labels and synonyms support opt-in retrieval and annotation
  review; they are not Vietnamese NER aliases.
- **KG:** hierarchy, positive phenotype, explicit negative phenotype, and neutral disease-gene
  relations support offline graph evidence and explanations.
- **Model/challenge data:** HPO provides rare phenotype coverage targets, disease-held-out groups,
  hard negatives from siblings, and minimal pairs involving absent versus present phenotypes.
- **Reasoning features:** phenotype overlap, information-content weighting, and ancestor closure can
  become bounded reranker features after a source-held-out benchmark.
- **Deduplication:** source rows are fingerprinted; graph edges aggregate support while evidence rows
  remain one-to-one with eligible observations.

## Promotion Boundary

HPO is `curated / training_only`. It is blocked from runtime defaults because:

1. labels are English ontology language, not validated Vietnamese clinical mentions;
2. disease IDs span OMIM, ORPHA and DECIPHER rather than the current ICD/RxNorm output convention;
3. source association means phenotype knowledge, not a patient-specific assertion;
4. the full association query benchmark is too slow for unbounded online use;
5. association-type and negative-phenotype semantics need task-specific feature evaluation.

## Reproduce

```bash
export MEDICAL_KG_ARTIFACT_STORE=/Volumes/medical-kg-mining
uv run medical-kg-research data run --plan configs/mining/hpo-2026-06-23.yaml

uv run medical-kg-research data ontology compile-obo \
  --input "$HPO_JSON_OBJECT" \
  --output-dir outputs/mining/knowledge/hpo-2026-06-23/ontology \
  --source-id hpo --source-version 2026-06-23 \
  --iri-prefix HP --code-system HPO --entity-type FINDING

uv run medical-kg-research data ontology compile-hpo-associations \
  --hpoa "$HPOA_OBJECT" --genes "$HPO_GENE_OBJECT" \
  --hpo-concepts outputs/mining/knowledge/hpo-2026-06-23/ontology/concepts.jsonl \
  --source-version 2026-06-23 \
  --output-dir outputs/mining/knowledge/hpo-2026-06-23/associations
```

Object paths must be resolved from the acquisition manifest and checked against the hashes above.
