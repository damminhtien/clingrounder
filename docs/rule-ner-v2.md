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
| `ner/document_structure.py` | Raw-offset lines, sections, list boundaries, and coarse genre |
| `ner/extractors/dictionary.py` | Exact/toneless recognition evidence |
| `ner/extractors/contextual_alias.py` | Train-compiled short aliases gated by structural context |
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
| Qualitative result before test | 39.7885 | 39.5727 | 780 | 214 | 356 |
| Context-gated short aliases | 39.7719 | 39.4697 | 753 | 233 | 382 |
| Structured symptom completion | **39.3638** | **39.2544** | **752** | 232 | 366 |

The largest gain after proposal composition came from correcting medication-list ownership. The
old adjudicator treated every bullet containing a drug as a complete medication span. The current
contract keeps full BTC-style SIG spans while avoiding narrative suffixes.

The contextual-alias compiler reviewed 20 policy aliases but emitted only three symptom rules
supported by the frozen train inventory: `đau`, `ra máu`, and `yếu`. Numeric aliases remain owned
by the lab grammar, while broad lab-test aliases remain owned by existing sources. Short aliases
alone traded 27 fewer missing entities for more boundary and spurious errors. Completing only a
parsed medication indication or a list item that starts with the reviewed alias recovered 16
boundary errors overall. On the final stage, 18 of 21 matched structured completions had exact
boundaries.

## Reproduce

Run from the repository root at the exact source-control revision:

```bash
uv sync --frozen --extra dev

uv run python scripts/benchmarks/phase1/mine_phase1_recognition_knowledge.py

uv run medical-kg benchmark phase1 submission \
  --input-dir data/raw/input \
  --output-dir outputs/phase1/rule_ner_v2_structured_symptom_boundary/output \
  --zip outputs/phase1/rule_ner_v2_structured_symptom_boundary/output.zip \
  --dictionary data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl \
  --pipeline-config configs/benchmarks/phase1/pipeline/rule_ner_mined_recognition.yaml \
  --assertion-policy empty \
  --candidate-policy empty \
  --parallel-backend serial \
  --workers 1

uv run python scripts/benchmarks/phase1/analyze_phase1_entity_wer.py \
  --pred outputs/phase1/rule_ner_v2_structured_symptom_boundary/output.zip \
  --stage contextual_alias=outputs/phase1/rule_ner_v2_contextual_alias/output.zip \
  --final-source-name structured_symptom_boundary \
  --output-dir outputs/evaluation/rule_ner_v2_structured_symptom_boundary
```

The recognition mining command must reproduce content-addressed run
`phase1-recognition-a147402078e0` at this revision. Its recognition-dictionary holdout gate reports
+10 exact true positives, -1 false positive, and +0.01204 exact F1 before the profile consumes it.
The same run writes `contextual_alias_rules.yaml`, `contextual_alias_decisions.jsonl`, and
`contextual_alias_report.json`; the profile loads the rules only because the parent mining run
passed its frozen holdout gate.

## Next Rule Work

Continue only as isolated proposal families:

1. Lab-test precision gates for bare imaging abbreviations and vital names.
2. Test/result phrase composition where a known anchor exists.
3. Test modality/anatomy boundary composition.
4. Train-only source/type calibration for global proposal utility; proposal scores currently
   remain emission priors rather than learned probabilities.

Do not add document IDs, absolute offsets, complete gold phrases tied to one note, or global
blacklists derived from holdout errors. Keep a family only when all-split WER improves and frozen
holdout does not regress.
