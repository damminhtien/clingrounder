# ClinGrounder v1 scope

ClinGrounder is a local Python toolkit for grounding Vietnamese and mixed Vietnamese-English
clinical text to source-backed entities, clinical context, and terminology candidates.

## Supported

- Languages: Vietnamese, mixed Vietnamese-English
- Entities: `DISEASE`, `SYMPTOM`, `DRUG`, `LAB_TEST`, `LAB_RESULT`
- Assertions: `PRESENT`, `NEGATED`, `HISTORICAL`, `FAMILY`, `POSSIBLE`
- Terminologies: ICD-10, RxNorm, plus local identifiers for non-coded observations
- Deployment: local Python with deterministic rule-based execution
- Output: raw-text spans, assertion status, candidate concepts, and validation evidence

## Experimental

- Dense retrieval
- Knowledge-graph reranking
- Transformer adapters
- Relation extraction
- Data mining and synthetic-data generation

Experimental modules are opt-in and do not change the deterministic baseline implicitly.

## Out of scope

- Clinical decision support
- Diagnosis or treatment recommendations
- Full EHR interoperability
- Regulatory compliance or medical-device certification
- Automatic ingestion of private clinical data

## Intended users

Clinical NLP researchers and application developers who need inspectable spans, terminology
retrieval, and reproducible local experiments. ClinGrounder is research software and must not be
used as the sole basis for clinical decisions.

## Product promise

The stable v1 contract is raw-offset ownership, typed annotations, explicit terminology
membership, deterministic local execution, and reproducible configuration/resource fingerprints.
Model-backed extraction, graph reasoning, and mining are research extensions with separate
benchmarks and provenance.
