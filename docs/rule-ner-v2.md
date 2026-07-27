# Rule NER V2

Rule NER V2 is the deterministic entity-extraction baseline. It is proposal-first:

```text
independent foundation proposals
-> dependent structure proposals
-> contextual type resolution
-> one global span resolver
-> raw-offset entities and trace
```

No extractor receives a mutable list of accepted spans. Dictionary, medication, laboratory, and
boundary sources emit evidence independently; `EvidenceWeightedSpanResolver` decides overlap once
after the complete proposal set is available.

## Module Map

| Module | Responsibility |
| --- | --- |
| `ner/proposal.py` | Immutable proposal, decision, and trace records |
| `ner/contracts.py` | Proposal extractor contract and shared read-only context |
| `ner/rule_engine.py` | Composition and final raw-offset materialization |
| `ner/extractors/dictionary.py` | Exact/toneless recognition evidence |
| `ner/type_resolver.py` | Type choice from observed evidence and bounded context |
| `ner/extractors/structured_lab.py` | Section-scoped unknown test/result pairs |
| `ner/lab_observation_extractor.py` | Values adjacent to known test anchors |
| `ner/medication_mention_parser.py` | Structured strength/form/route/frequency spans |
| `ner/medication_list_parser.py` | List bounds and indication scopes |
| `ner/extractors/boundary.py` | Allow-listed symptom/diagnosis composition |
| `ner/span_resolver.py` | Global non-overlap optimization |

## Safety Contracts

- `source_text[start:end] == entity.text` for every selected entity.
- Normalization is lookup-only.
- Context can choose only among types supplied by proposals; it cannot invent a type.
- Neutral section headings terminate symptom/diagnosis scope.
- A medication list item is an upper bound. Only parsed SIG components can extend a drug span.
- Bare numbers and qualitative lab words need a structural section or a test anchor.
- Mined recognition concepts are code-free. Linking remains a separate typed terminology query.

## Measured Stages

The table uses the 100 reviewed local files and the frozen 24-document holdout. `WER proxy` is the
entity text/type metric used by the repository, not a public Round 2 score.

| Stage | All WER | Holdout WER | Missing | Spurious | Boundary |
| --- | ---: | ---: | ---: | ---: | ---: |
| Proposal core | 49.9618 | 49.1857 | 974 | 215 | 522 |
| Reviewed recognition | 46.4023 | 47.5463 | 885 | 219 | 490 |
| Structured lab grammar | 43.8113 | 43.5998 | 792 | 235 | 494 |
| Clinical boundary + atomic terms | 42.3793 | 42.0951 | 792 | 222 | 414 |
| Contextual reviewed type evidence | 41.9568 | 42.0951 | 784 | 214 | 414 |
| Parsed medication boundaries | 39.9225 | 39.7141 | 783 | 213 | 356 |
| Qualitative result before test | **39.7885** | **39.5727** | **780** | 214 | **356** |

The largest gain after proposal composition came from correcting medication-list ownership. The
old adjudicator treated every bullet containing a drug as a complete medication span. The current
contract keeps full BTC-style SIG spans while avoiding narrative suffixes.

## Reproduce

Run from the repository root at the exact source-control revision:

```bash
uv sync --frozen --extra dev

uv run python scripts/mine_phase1_recognition_knowledge.py

uv run medical-kg benchmark phase1 submission \
  --input-dir data/raw/input \
  --output-dir outputs/phase1/rule_ner_v2_lab_qualifier_before_test/output \
  --zip outputs/phase1/rule_ner_v2_lab_qualifier_before_test/output.zip \
  --dictionary data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl \
  --pipeline-config configs/pipeline/rule_ner_mined_recognition.yaml \
  --assertion-policy empty \
  --candidate-policy empty \
  --parallel-backend serial \
  --workers 1

uv run python scripts/analyze_phase1_entity_wer.py \
  --pred outputs/phase1/rule_ner_v2_lab_qualifier_before_test/output.zip \
  --stage baseline=outputs/phase1/rule_ner_v2_container_constraints/output.zip \
  --final-source-name rule_ner_v2 \
  --output-dir outputs/evaluation/rule_ner_v2_lab_qualifier_before_test
```

The recognition mining command must reproduce content-addressed run
`phase1-recognition-577236248a8e` at this revision. Its holdout gate reports +10 exact true
positives, -1 false positive, and +0.01254 exact F1 before the profile consumes it.

## Next Rule Work

Continue only as isolated proposal families:

1. Context-gated short symptom anchors, starting with explicit symptom-list structure.
2. Lab-test precision gates for bare imaging abbreviations and vital names.
3. Test/result phrase composition where a known anchor exists.
4. Test modality/anatomy boundary composition.

Do not add document IDs, absolute offsets, complete gold phrases tied to one note, or global
blacklists derived from holdout errors. Keep a family only when all-split WER improves and frozen
holdout does not regress.
