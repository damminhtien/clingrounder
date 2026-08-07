# Relation Extraction Baseline

The default relation extractor is a conservative, data-driven baseline. It is not a general
ontology reasoner and it does not infer medical knowledge from sentence proximity alone.

## Sources

- `structural_medication_attribute`: same-clause drug and medication attributes.
- `structural_lab_result`: same-clause laboratory test and result anchors.
- `sentence_cooccurrence_proposal`: explicit lexical cue plus same-clause entity pair.
- `terminology_ontology_backed`: reviewed code-to-code records in
  `data/relations/known_treats.jsonl`.
- `model_backed`: reserved for a future calibrated model adapter; no model relation source is
  emitted by the baseline today.

Every baseline relation carries `RelationEvidence` with a source, rule ID, evidence span,
support score, and provenance. Support scores are heuristic strengths unless a calibrated model
is explicitly named in the provenance.

## Abstention policy

The baseline blocks negated, family, possible, conditional, planned, resolved, and unknown
entities. Historical entities may participate when the relation itself is historical or
structural. Semantic relations require an explicit cue in the same clause. Newline, list-item,
sentence punctuation, and semicolon boundaries block nearest-entity linking.

Ontology-backed validation fails closed when endpoint membership or an `IS_A` hierarchy cannot be
verified. Unknown ontology membership is reported as `unknown_ontology_membership`; it is not
treated as evidence that a relation is valid.

## Evaluation slices

Use `clingrounder.evaluation.relation_slice_counts` to profile relation type, token distance,
same-clause versus same-sentence scope, assertion status, evidence source, terminology-backed
versus heuristic evidence, and medication/lab structural slices.
