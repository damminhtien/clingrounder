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

## Parsing, Sections And Deduplication

JATS parser revision 2 renders article text once and stores it as immutable
`MinedDocument.text`. It additionally records every article-title, section-title and paragraph
block with its exact raw span, text SHA-256, JATS `sec-type` and hierarchical section path. These
blocks are metadata only: revision 2 produced zero text drift against revision 1 for all 50
articles. All annotations continue to use raw `[start,end)` offsets against the unchanged text.

The completed run contains:

| Measure | Value |
| --- | ---: |
| artifacts/documents | 50 / 50 |
| source characters | 846,855 |
| median document length | 16,515.5 |
| maximum document length | 45,844 |
| exact duplicate documents | 0 |
| normalized/SimHash-near groups | 0 |
| unique text hashes | 50 |
| rendered source blocks | 2,403 |
| block text/hash mismatches | 0 |
| revision-1/revision-2 text drift | 0 / 50 |
| schema/offset issues | 0 |

All documents are English `case_report_article` records, parsed by `jats_xml`, with attribution
redistribution. Raw artifacts are content-addressed under the configured artifact store.

The source-only fusion run `pmc-rare-cases-ccby-2026-07-19-e58fb0e2c28b` produced 50 singleton
groups. No article was collapsed and no near-duplicate split barrier was required at the configured
SimHash threshold. The fusion plan remains checked in so a larger tranche is audited the same way.

### Section Evidence

`configs/mining/sections/pmc-case-evidence.yaml` classifies blocks from source structure and
versioned heading patterns. It does not label medical entities. The stage attaches the matching
block span/path and one evidence tier to an existing proposal without changing its ID, type, text,
span, assertion, candidate or confidence.

| Evidence tier | Blocks | Pipeline proposals | Intended use |
| --- | ---: | ---: | --- |
| `case_specific` | 635 | 970 | strict case review/training view |
| `article_summary` | 218 | 174 | abstract/title context, review separately |
| `literature_context` | 778 | 815 | background/discussion, not patient evidence |
| `non_clinical` | 591 | 50 | references, footnotes, acknowledgements and metadata |
| `other_article_context` | 181 | 93 | unresolved section context |

All 2,102 proposals were contained in one verified source block. The strict curation policy
`configs/mining/curation/pmc-case-specific-pipeline.yaml` retains only the 970 `case_specific`
proposals and rejects the other 1,132. Section headings vary between publishers, so this is a
source-structure heuristic, not human case/non-case gold.

## Knowledge Extraction

The local full-terminology pipeline generated a bronze proposal layer, not source gold. Counts
below are for the full article text before the strict section view:

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

The case-specific view contains 312 normalized mention entries from 970 occurrences. Its entity
counts are 428 `LAB_TEST`, 190 `DISEASE`, 156 `LAB_RESULT`, 125 `SYMPTOM`, 32 `DRUG` and 39
medication attributes. This view reduces contamination from literature discussion but does not
establish label correctness because the same local pipeline produced the proposals.

### Mondo/HPO Crosswalk Evidence

Only case-specific `DISEASE` and `SYMPTOM` mentions were queried against a combined, release-pinned
Mondo 2026-07-06 and HPO 2026-06-23 SQLite index. The policy is
`configs/mining/crosswalk/pmc-case-mondo-hpo.yaml`; it enforces `DISEASE -> MONDO` and
`SYMPTOM -> HPO/FINDING` before lookup.

| Exact crosswalk outcome | Terms | Occurrences |
| --- | ---: | ---: |
| one exact concept | 70 | 227 |
| multiple exact codes | 5 | 8 |
| no exact match | 17 | 80 |
| skipped non-disease/symptom types | 220 | 655 |

Of the 70 unique exact terms, 41 are Mondo disease mappings (124 occurrences) and 29 are HPO
phenotype mappings (103 occurrences). Every row has `automatic_promotion_allowed: false` and
`promotion_status: review_required`. Exact terminology identity does not prove that the pipeline
found the right span/type or that a mention is asserted for the case patient. No crosswalk result
has been written into the runtime dictionary or canonical graph.

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

The original full-article bronze snapshot remains reproducible, but it should not be used for new
model work because it mixes patient case and literature context. The replacement strict snapshot
is `pmc-rare-cases-ccby-2026-07-19-case-evidence-bronze-v2-97eec3c89687cd6e`; it contains 39
article-grouped train and 11 development documents, 970 annotations and zero relations. Manifest
SHA-256: `98224d74d7097f225632e97c9119f80d601c20cd982a5d9a67aafd1e8725a224`.

## Reproduce

```bash
export MEDICAL_KG_ARTIFACT_STORE=/Volumes/medical-kg-mining
uv run medical-kg data run \
  --plan configs/mining/pmc-rare-cases-ccby-2026-07-19.yaml

uv run medical-kg data label propose \
  --documents outputs/mining/pmc-rare-cases-ccby-2026-07-19/documents_v2.jsonl \
  --adapter medical_kg_nlp.mining.labelers.pipeline:create_local_pipeline_labeler \
  --adapter-config configs/mining/labelers/local_pipeline_full_terminology.yaml \
  --output outputs/mining/pmc-rare-cases-ccby-2026-07-19/pipeline_proposals_v2.jsonl

uv run medical-kg data dataset attach-block-evidence \
  --documents outputs/mining/pmc-rare-cases-ccby-2026-07-19/documents_v2.jsonl \
  --annotations outputs/mining/pmc-rare-cases-ccby-2026-07-19/pipeline_proposals_v2.jsonl \
  --policy configs/mining/sections/pmc-case-evidence.yaml \
  --output outputs/mining/pmc-rare-cases-ccby-2026-07-19/pipeline_proposals_v2_sections.jsonl \
  --report-output outputs/mining/pmc-rare-cases-ccby-2026-07-19/section_evidence_report.json

uv run medical-kg data dataset curate-annotations \
  --annotations outputs/mining/pmc-rare-cases-ccby-2026-07-19/pipeline_proposals_v2_sections.jsonl \
  --policy configs/mining/curation/pmc-case-specific-pipeline.yaml \
  --accepted-output outputs/mining/pmc-rare-cases-ccby-2026-07-19/case_specific_proposals.jsonl \
  --rejected-output outputs/mining/pmc-rare-cases-ccby-2026-07-19/non_case_proposals.jsonl \
  --report-output outputs/mining/pmc-rare-cases-ccby-2026-07-19/case_specific_curation_report.json

uv run medical-kg data lexicon build \
  --documents outputs/mining/pmc-rare-cases-ccby-2026-07-19/documents_v2.jsonl \
  --annotations outputs/mining/pmc-rare-cases-ccby-2026-07-19/case_specific_proposals.jsonl \
  --output outputs/mining/pmc-rare-cases-ccby-2026-07-19/case_specific_mention_inventory.jsonl \
  --conflicts-output outputs/mining/pmc-rare-cases-ccby-2026-07-19/case_specific_mention_conflicts.jsonl \
  --report-output outputs/mining/pmc-rare-cases-ccby-2026-07-19/case_specific_mention_report.json

uv run medical-kg terminology build \
  --source outputs/mining/knowledge/mondo-2026-07-06/terminology.jsonl \
  --source outputs/mining/knowledge/hpo-2026-06-23/ontology/terminology.jsonl \
  --output outputs/mining/pmc-rare-cases-ccby-2026-07-19/mondo_hpo_terminology.sqlite3

uv run medical-kg data lexicon crosswalk \
  --inventory outputs/mining/pmc-rare-cases-ccby-2026-07-19/case_specific_mention_inventory.jsonl \
  --index outputs/mining/pmc-rare-cases-ccby-2026-07-19/mondo_hpo_terminology.sqlite3 \
  --source outputs/mining/knowledge/mondo-2026-07-06/terminology.jsonl \
  --source outputs/mining/knowledge/hpo-2026-06-23/ontology/terminology.jsonl \
  --policy configs/mining/crosswalk/pmc-case-mondo-hpo.yaml \
  --output outputs/mining/pmc-rare-cases-ccby-2026-07-19/mondo_hpo_crosswalk.jsonl \
  --report-output outputs/mining/pmc-rare-cases-ccby-2026-07-19/mondo_hpo_crosswalk_report.json

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
