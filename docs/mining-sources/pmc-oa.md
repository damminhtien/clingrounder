# PMC Open Access Rare Cases

## Source And Acquisition

- Registry source: `pmc_oa`.
- Transport: Europe PMC REST 6.9 JATS `fullTextXML` endpoint.
- Selection query: open-access full-text case reports from 2024-2026 whose title/publication type
  includes case reports and whose title includes rare, unusual, or novel.
- Discovery response SHA-256:
  `849b1e1b42a02ad2df25e8e657f5b11b88557f3d69acef4cbf2153076e264773`.
- Accepted tranche: first 50 discovery records whose per-article license is exactly `CC BY`.
- Run config: `configs/mining/pmc-rare-cases-ccby-2026-07-19.yaml`.

The earlier ten-article smoke slice mixed CC BY, CC BY-NC, and CC BY-NC-ND. It remains useful for
parser tests, but it is not the source of this redistributable bronze snapshot. The current config
lists every PMCID and license metadata explicitly, so reruns do not silently change selection.

## Parsing And Deduplication

The JATS parser renders article text once and stores it as immutable `MinedDocument.text`. All later
annotations use raw `[start,end)` offsets against that text. The completed run contains:

| Measure | Value |
| --- | ---: |
| artifacts/documents | 50 / 50 |
| source characters | 846,855 |
| median document length | 16,515.5 |
| maximum document length | 45,844 |
| exact duplicate documents | 0 |
| normalized/SimHash-near groups | 0 |
| unique text hashes | 50 |
| schema/offset issues | 0 |

All documents are English `case_report_article` records, parsed by `jats_xml`, with attribution
redistribution. Raw artifacts are content-addressed under the configured artifact store.

The source-only fusion run `pmc-rare-cases-ccby-2026-07-19-e58fb0e2c28b` produced 50 singleton
groups. No article was collapsed and no near-duplicate split barrier was required at the configured
SimHash threshold. The fusion plan remains checked in so a larger tranche is audited the same way.

## Knowledge Extraction

The local full-terminology pipeline generated a bronze proposal layer, not source gold:

| Entity type | Proposal occurrences |
| --- | ---: |
| DISEASE | 573 |
| LAB_TEST | 738 |
| LAB_RESULT | 405 |
| SYMPTOM | 283 |
| DRUG | 56 |
| medication attributes | 47 |

There are 2,102 proposals across all 50 documents, with 1,547 concept links. Context proposals
include 119 negated, 45 historical, 14 possible, 4 family and 1 planned occurrence. The mention
inventory has 430 normalized entries: 150 multi-document, 47 repeated in one document, and 233
singletons.

These counts measure proposal coverage only. The local pipeline is the label source, so the records
cannot independently validate that pipeline and cannot become gold merely because a concept link is
present. In particular, numeric lab results and broad English disease terms require review.

The frozen 42-document train split has 381 inventory entries, including 125 multi-document entries.
The fail-closed recognition compiler excludes lab results and medication attributes, then requires
at least two occurrences in two train documents. Eighty-five entries passed those row gates, but all
85 already existed with the same type in the baseline recognition dictionaries. The compiler
therefore emitted **zero new recognition concepts**. This negative result blocks dictionary growth
from this tranche; it also shows that more articles alone would mostly repeat existing English
clinical vocabulary.

The deterministic review export contains one queue record per article and has SHA-256
`994b603e844abbf682c62f558756f514dea2da85d21628a35bbdf83b625f5b06`.

## Promotion Boundary

- Allowed now: review prioritization, rare-case vocabulary analysis, weak-label experiments on the
  train split, and synthetic scenario grounding.
- Blocked now: runtime dictionary promotion, canonical graph facts, challenge evaluation, or claims
  of NER precision/recall.
- A term must occur in multiple train documents, pass type/context review, and win a held-out
  recognition or retrieval benchmark before an opt-in artifact can be created.
- Relations require reviewed endpoints and explicit relation evidence. Same-article occurrence is
  not a causal or treatment relation.

The frozen bronze snapshot is
`pmc-rare-cases-ccby-2026-07-19-bronze-v1-c6c8a342c33eca51`; it contains 42 train and 8 development
documents. Manifest SHA-256:
`6153c36e17d95e964c0ffcb50473ba0e1a5d6e6e2eb374f841d02941215302df`.

## Reproduce

```bash
export MEDICAL_KG_ARTIFACT_STORE=/Volumes/medical-kg-mining
uv run medical-kg data run \
  --plan configs/mining/pmc-rare-cases-ccby-2026-07-19.yaml

uv run medical-kg data label propose \
  --documents outputs/mining/pmc-rare-cases-ccby-2026-07-19/documents.jsonl \
  --adapter medical_kg_nlp.mining.labelers.pipeline:create_local_pipeline_labeler \
  --adapter-config configs/mining/labelers/local_pipeline_full_terminology.yaml \
  --output outputs/mining/pmc-rare-cases-ccby-2026-07-19/pipeline_proposals.jsonl

uv run medical-kg data dataset inspect \
  --documents outputs/mining/pmc-rare-cases-ccby-2026-07-19/documents.jsonl \
  --annotations outputs/mining/pmc-rare-cases-ccby-2026-07-19/pipeline_proposals.jsonl \
  --output outputs/mining/pmc-rare-cases-ccby-2026-07-19/source_profile.json \
  --strict

uv run medical-kg data dataset fuse \
  --plan configs/mining/fusion/pmc-rare-cases-ccby-2026-07-19.yaml

uv run medical-kg data knowledge compile-recognition \
  --inventory outputs/mining/pmc-rare-cases-ccby-2026-07-19/train_mention_inventory.jsonl \
  --policy configs/mining/recognition/pmc-rare-cases-ccby-2026-07-19.yaml \
  --baseline-dictionary data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl \
  --baseline-dictionary data/standards/vn_clinical_lexicon/processed/vn_clinical_lexicon_concepts.jsonl \
  --output outputs/mining/pmc-rare-cases-ccby-2026-07-19/train_recognition_concepts.jsonl \
  --decisions-output outputs/mining/pmc-rare-cases-ccby-2026-07-19/train_recognition_decisions.jsonl \
  --report-output outputs/mining/pmc-rare-cases-ccby-2026-07-19/train_recognition_report.json
```
