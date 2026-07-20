# Registered And Quarantined Sources

These sources are known to the registry but have not completed a reproducible mining run. They must
not be counted in corpus size, terminology coverage, or model-training claims.

| Source ID | Current blocker | Next executable gate |
| --- | --- | --- |
| `vn_moh_guidelines` | no pinned document list or source fingerprint | select a bounded release, verify reuse terms, parse sections |
| `vietmed_ner` | annotation-data license is not explicit | obtain license evidence; keep quarantined until then |
| `loinc_2_82` | account/license acceptance and archive absent | import locally, validate release hash, build lab terminology only |
| `mondo` | no release pinned | pin JSON/OBO release and benchmark ontology edge import |
| `hpo` | no release pinned | pin ontology plus disease-phenotype annotations and provenance |
| `biored` | dataset license review incomplete | verify redistribution and annotation use before local import |
| `synthea` | generator commit and scenario config absent | pin commit, seed, FHIR export and deterministic patient groups |
| `mimic_iv_note` | credentialed DUA archive unavailable | import only on encrypted local storage; never use hosted labeling |
| `n2c2_i2b2` | dataset-specific approval/archive unavailable | create a private local-only plan after access is granted |

Registration is intentionally cheap; promotion is intentionally expensive. No source in this table
currently contributes runtime aliases, recognition terms, graph edges, training spans, or benchmark
claims.
