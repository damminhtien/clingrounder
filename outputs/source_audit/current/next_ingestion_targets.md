# Next Ingestion Targets

## Confirmed Local State

- Source registry validates with `registry_issue_count = 0`.
- Required local raw/processed files are present; `missing_required_file_count = 0`.
- TT06 ICD extract has 12,159 concepts and full local checksums in `source_manifest.json`.
- RxNorm prescribable June 2026 has `RXNCONSO.RRF`, `RXNREL.RRF`, and `RXNSAT.RRF`.
- RxNorm prescribable profile: 40,667 accepted concepts, 613,324 active relation rows, 467,328 active attribute rows.
- RxNorm July 2026 full bundle is local and profiles both `prescribe/rrf` and root `rrf` with zero malformed rows.
- July candidate layers contain 40,675 Prescribable concepts and 73,912 Full fallback concepts.
- Runtime-controlled dictionary remains separate from full standards:
  - Full TT06: `data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl`
  - Full RxNorm prescribable: `data/standards/rxnorm/processed/rxnorm_prescribable_06012026_concepts.jsonl`
  - Runtime controlled: `data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl`

## Open Gaps

- The exact June full fallback archive remains unavailable, so the June baseline cannot reproduce a same-date Full fallback layer. July Full is available but must not be substituted silently.
- Seed ICD rows still lack chapter/block metadata; TT06-derived controlled rows now carry it.
- VN alias mining has candidates but no curation decision yet.
- Unknown phrase mining still includes section/procedure phrases; use it as a review queue, not an automatic dictionary patch.
- Procedure terminology is only mined as candidates, not modeled as a first-class runtime semantic type yet.

## Highest ROI Next Steps

1. Run a controlled June-versus-July RxNorm ablation, then promote July only if linking and false-positive metrics improve without regressions.
2. Review `outputs/source_audit/alias_mining/alias_candidates.md` and promote only high-confidence aliases with regression tests.
3. Add a curated procedure/local terminology pack for frequent terms such as `phẫu thuật`, `thủ thuật`, `chụp CT`, `chụp X-quang`, and `xét nghiệm`.
4. Backfill ICD chapter/block metadata into seed rows or prefer controlled rows when building future runtime dictionaries.
5. Run source ablations after each curated batch to separate gains from TT06, RxNorm, VN aliases, and procedure/lab lexicon.
