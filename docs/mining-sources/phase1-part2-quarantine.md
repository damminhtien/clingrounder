# Phase 1 Part 2 Bundle: Quarantine Audit

## Policy Boundary

The caller described this local bundle as data previously leaked by the competition organizer.
No organizer authorization, license, or public-release notice accompanies the archive. It is
therefore registered as `quarantine`, with no promotion boundary.

Allowed use is limited to:

- verifying immutable checksums;
- reviewing the license and competition rules;
- aggregate, read-only analysis that does not expose document text, spans, labels, or IDs.

The following uses are prohibited until the organizer explicitly authorizes them:

- supervised or pseudo-label training;
- copying annotations into a current submission;
- creating aliases, thresholds, mention memories, or document-specific rules;
- public-grader probes derived from these labels;
- hosted processing or redistribution.

This dossier records aggregate evidence so the team does not repeatedly rediscover or accidentally
operationalize it. No raw document, annotation row, or derived runtime artifact is committed.

## Immutable Identity

The supplied outer archive contains two nested ZIP files, each with 100 data members:

| Artifact | SHA-256 | Bytes | Members |
| --- | --- | ---: | ---: |
| outer bundle | `46da2a7718078b95024e97feb66d49e44917d7b00b3981bad3eaaae13adc418e` | 791,634 | 2 nested ZIPs |
| `input.zip` | `ecb0bb792ad8649b06dcfd10847a1d33c963a9332ac4690c56b0004643eceb5c` | 454,754 | 100 TXT |
| `gt.zip` | `fbd75944ff485dcdb0257a33867ef0fc78ae29cd7dfa5b5bdd0ecd18baed28b3` | 353,019 | 100 JSON |

The 100 input texts have:

- zero ID-aligned exact matches against the current Round 2 corpus;
- zero cross-ID exact text-hash matches against that corpus;
- zero shared unique text hashes.

This is not direct hidden gold for the current Round 2 documents. It is still unauthorized
competition annotation data and remains quarantined.

## Offset Coordinate Evidence

The nested TXT files contain 7,413 CRLF sequences. Across 15,444 annotation rows:

| Coordinate view | Exact `text == source[start:end]` |
| --- | ---: |
| strict UTF-8 decoded text with CRLF preserved | 6 / 15,444 |
| text after only `CRLF -> LF` | 15,444 / 15,444 |
| LF text plus NFC normalization | 15,035 / 15,444 |
| LF text plus NFKC normalization | 14,627 / 15,444 |

The annotation coordinate source was therefore an LF-normalized text view, not the decoded archive
bytes. Unicode normalization was not part of that coordinate transformation.

This does not justify silently changing the repository's raw-offset invariant. If this source is
ever authorized, the parser must materialize an immutable LF child document with a parent hash and
an explicit offset map. The current Round 2 imported documents already contain LF only, so this
finding does not change their exported offsets.

## Aggregate Label Shape

The bundle contains 15,444 entity rows:

| Type | Count | Share |
| --- | ---: | ---: |
| `TRIỆU_CHỨNG` | 10,395 | 67.31% |
| `CHẨN_ĐOÁN` | 2,566 | 16.61% |
| `TÊN_XÉT_NGHIỆM` | 1,384 | 8.96% |
| `KẾT_QUẢ_XÉT_NGHIỆM` | 637 | 4.12% |
| `THUỐC` | 462 | 2.99% |

Observed annotation density is 15.57 entities per 1,000 characters. Per-document entity counts have
median 152, 90th percentile 211, minimum 65, and maximum 322. The current question/answer marker
classifies none of these documents as question/answer, so this density must not be treated as a
target for the current mixed Round 2 distribution.

Span lengths show that the convention is not limited to dictionary-sized mentions:

| Type | Median length | 90th percentile |
| --- | ---: | ---: |
| diagnosis | 16 | 32 |
| symptom | 12 | 27 |
| test name | 18 | 36 |
| test result | 12 | 77 |
| medication | 14 | 28 |

The long tail for test results is important: many are descriptive imaging or laboratory findings,
not isolated numbers or one-token qualitative values. A numeric-only lab-result extractor cannot
represent this convention.

Other aggregate boundary evidence:

- 556 test-name spans include a modality prefix;
- 339 symptom spans include a prefix such as a sensation or episode marker;
- 127 diagnosis spans include severity or subtype material;
- 58.23% of medication spans contain a strength, route, or frequency-like attribute;
- only 10 result spans are exactly a standalone qualitative token.

## Occurrence, Duplicate, And Overlap Behavior

For 13,260 normalized mention/document groups, 11,339 groups (85.51%) annotate every exact raw
occurrence. The remaining 1,921 groups annotate only a subset. The likely policy is therefore
"recover repeated eligible occurrences", not "emit every string match globally".

The supplied labels also contain 37 duplicate identity rows and 46 overlapping span pairs. These
small inconsistencies are evidence that the archive is noisy annotation output, not a clean
validator contract. They do not justify deliberately emitting duplicates or overlaps.

## Assertion Convention

Treating missing assertion fields as empty for aggregate analysis gives:

| Assertion set | Rows |
| --- | ---: |
| empty | 13,793 |
| `isHistorical` | 1,080 |
| `isNegated` | 562 |
| `isFamily` | 4 |
| `isNegated` + `isHistorical` | 5 |

The default is abstention: 89.31% of rows carry no assertion. Family assertion is extremely rare,
while multi-label output appears only when independent historical and negation evidence coexist.

However, 330 non-empty assertions occur on test-name or test-result rows. This conflicts with the
published Phase 1 contract and the repository validator, which allow assertions only for symptoms,
diagnoses, and medications. The archive also omits assertion fields on many rows. These
contradictions mean the labels cannot replace the official schema. Selective `A_NEG_HIST` remains
the defensible submission policy; lab assertions require organizer clarification or an isolated,
legitimate public example.

## Candidate Convention

Candidate output is sparse:

| Eligible type | Coded rows | Total rows | Coverage |
| --- | ---: | ---: | ---: |
| diagnosis | 427 | 2,566 | 16.64% |
| medication | 140 | 462 | 30.30% |

There are 567 coded rows and 575 code values. Of these rows, 559 contain exactly one code and only
eight contain two. No row contains more than two.

Across 368 normalized mention/type groups with a code:

- 356 groups have one consistent mapping;
- 12 groups have conflicting mappings;
- mapping consistency is 96.74%.

Using the full pinned terminology, only 28.3% of known mappings are exact dictionary-alias matches.
The rest require context or semantic linking. Known RxNorm mappings are dominated by ingredient and
brand concepts (`IN=56`, `BN=25`), with only 12 clinical/branded clinical drug mappings
(`SCD=11`, `SBD=1`). Full medication spans therefore do not imply that a strength-specific RxCUI
should always be emitted.

The operational hypothesis to test on authorized data is:

```text
at most one reviewed code
-> ingredient/brand first for drugs
-> strength-specific product only with complete evidence
-> abstain on ambiguity
```

This is consistent with recent public candidate probes, which penalized broad candidate lists.

## Schema Divergence

Running the strict Phase 1 validator after the LF coordinate transformation reports 2,501 issues:

- 2,161 schema issues, mainly omitted fields;
- 330 assertions on types that the public contract excludes;
- 10 candidate codes unknown even after loading the seed plus full terminology.

The archive is best interpreted as an intermediate or noisy annotation export. The repository must
not relax required fields, type constraints, dictionary membership, or ZIP validation based on
this bundle.

## Independently Testable Improvements

The aggregate audit supports hypotheses, not direct implementation from leaked labels. The
following work can be developed and evaluated using authorized manual gold, public organizer
examples, and source-held-out open data:

1. **Proposal-conditioned result-span classifier.** Generate candidate clauses around a lab or
   imaging anchor, then classify and trim the descriptive finding. Keep the existing numeric
   extractor as one proposal source rather than the whole result model.
2. **Test-name boundary model.** Learn modality, contrast, anatomy, and specimen boundaries from
   authorized examples. Dictionary matching supplies anchors; a span model decides the full raw
   boundary.
3. **Context-gated occurrence recovery.** Re-run accepted concept spans across the document, but
   require compatible section, clause, and type evidence for each occurrence.
4. **Medication full-span parsing.** Continue using structured medication components and raw offset
   projection. Candidate selection remains separate from span expansion.
5. **Sparse candidate reranking.** Retrieve broadly for traceability, calibrate a single-code emit
   probability, and abstain unless the top candidate is both type-compatible and clearly separated
   from alternatives.
6. **Selective assertion scope.** Keep default empty output; extend history and negation only with
   explicit section/local evidence and termination rules.
7. **Explicit newline adapters.** Add a source-specific CRLF-to-LF child-document adapter and
   round-trip offset tests before ingesting any authorized archive that defines offsets on LF text.

These changes must be ablated independently. Entity experiments must keep assertions and candidates
fixed; assertion and candidate experiments must keep the entity projection fixed.

## Authorization Path

If the organizer later publishes or explicitly authorizes this bundle:

1. pin the release notice and terms alongside these checksums;
2. replace `quarantine` with the exact permitted access class and uses;
3. implement the LF child-document parser and verify all 15,444 offsets;
4. normalize missing external fields into a separate adapter without weakening the submission
   validator;
5. remove duplicate rows, report overlap policy, and create source/template-held-out splits;
6. use it only as an auxiliary corpus, with current competition documents held out by exact and
   near-duplicate groups.

Until those steps are complete, this source contributes no training data, runtime knowledge, or
competition output.

## Reproduce Identity Checks

Only checksum verification is intentionally executable while the source remains quarantined:

```bash
export PHASE1_PART2_ARCHIVE=/secure/input_part2.zip
shasum -a 256 "$PHASE1_PART2_ARCHIVE"
unzip -p "$PHASE1_PART2_ARCHIVE" input.zip | shasum -a 256
unzip -p "$PHASE1_PART2_ARCHIVE" gt.zip | shasum -a 256
```

Expected hashes are listed under **Immutable Identity**. No importer, model job, or derived
annotation artifact is exposed while authorization remains unresolved.
