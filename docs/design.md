# Technical Design

## Problem Decomposition

The task is treated as end-to-end clinical information extraction and normalization, not simple NER.
The prototype decomposes it into document loading, offset-safe preprocessing, section detection,
sentence splitting, entity extraction, assertion classification, candidate generation, candidate
reranking, normalization assignment, code-system validation, relation extraction, KG consistency
checking, JSON export, prediction validation, and evaluation.

## Architecture

```text
ClinicalDocument
  -> collapse_whitespace_preserve_offsets
  -> split_sections
  -> split_sentences
  -> RuleBasedNER
  -> AssertionClassifier
  -> CandidateGenerator
  -> HeuristicReranker
  -> EntityLinker.apply_candidates
  -> KGValidator.validate_entities
  -> RuleRelationExtractor
  -> KGValidator.validate_relations
  -> ClinicalPrediction
```

Each module is independently replaceable. Rule components provide deterministic baselines, while
transformer NER, dense retrieval, learned reranking, learned context classification, and learned
relation extraction remain extension points. `PipelineTrace` records timings and counters for each
stage so ablations can identify whether an accuracy or runtime change came from retrieval,
reranking, context, validation, or relation extraction.

## Schema

Internal schemas live under `src/medical_kg_nlp/schema/`:

- `EntityType`: disease, symptom, drug, lab test, lab result, procedure, patient info, anatomy, finding, other.
- `AssertionStatus`: present, negated, historical, family, possible, conditional, planned, resolved, unknown.
- `CodeSystem`: ICD-10, RxNorm, UMLS, SNOMED, LOCAL, NONE.
- `RelationType`: treats, has symptom, suggests, has test, has value, has dose, route, frequency, causal, associative, ontology edges, unknown.
- `EntityAnnotation` preserves exact character offsets and candidate lists.
- `ClinicalPrediction` exports the final JSON object with `document_id`, `text_hash`, `entities`, `relations`, and metadata.

## Module Interfaces

- `DatasetAdapter.load_documents(path)` returns `list[ClinicalDocument]`.
- `RuleBasedNER.extract(text)` returns offset-valid entities.
- `AssertionClassifier.classify(entity, sentence)` returns an assertion label.
- `CandidateGenerator.generate(mention, entity_type, context_window)` returns dictionary-constrained candidates.
- `HeuristicReranker.rerank(candidates, context_window)` orders candidates.
- `EntityLinker.apply_candidates(entity, candidates)` writes top code and candidate list.
- `RuleRelationExtractor.extract(entities, sentences)` returns typed relations.
- `KGValidator` filters or clears invalid outputs.
- `evaluate_predictions(gold, pred)` returns span, linking, context, and relation metrics.

## Data Flow

The sample note is stored in `data/samples/sample_notes.jsonl`. The runner loads it into `ClinicalDocument`, extracts spans from the original text, links only against `data/dictionaries/seed_concepts.jsonl`, validates ontology constraints, and writes JSONL predictions. Candidate generation combines exact, abbreviation, fuzzy/token-overlap, character n-gram, and BM25 retrieval before type and code-system filtering.

No normalization step is allowed to alter final offsets. Matching normalization is used only for lookup keys.

## Evaluation Plan

Implemented metrics:

- Exact span/type precision, recall, F1.
- Relaxed overlap span/type precision, recall, F1.
- Linking accuracy@1, recall@5/10/20, and MRR.
- Context accuracy, macro-F1, and confusion matrix.
- Typed relation precision, recall, F1.
- Error analysis CSV with document id, error type, text window, gold, prediction, candidate list, and notes.

Future public-dataset adapters should convert all gold labels to the same internal schema before metrics are computed.

## Risks

- Seed dictionaries are incomplete and must be replaced or augmented with official ICD-10/RxNorm/UMLS resources.
- Public clinical datasets have licensing constraints, so adapters need local file paths.
- Rule-based context can over-scope cues in long sentences.
- Vietnamese abbreviation and synonym coverage is limited.
- The final competition may score shorter or longer entity boundaries differently.

## Future Adaptation Strategy

1. Add a schema adapter for the official prediction format.
2. Load full ICD-10 and RxNorm releases into `ConceptEntry`.
3. Expand Vietnamese-English synonym coverage and abbreviation ambiguity handling.
4. Train or plug in transformer NER while preserving original character offsets.
5. Add learned context and relation classifiers behind existing interfaces.
6. Use candidate-generation recall@20 as the linking gate before training rerankers.
7. Add KG scoring as a candidate reranker feature, not as an unconstrained generator.
