# Medical Source Audit

- Registry resources: 18
- Registry issues: 0
- Missing required local files: 0
- Dictionary profiles: 2
- Manual review items: 223
- False-positive blocklist candidates: 101

## Source Registry

- `cdc_icd10cm_2026`: CDC ICD-10-CM FY 2026 files | access=open | version=FY 2026 April 1 update | license=us_government_public_resource | use=icd10cm_reference_only_not_phase1_primary
- `codiesp_zenodo`: CodiEsp corpus | access=open | version=Zenodo record 3837305 | license=CC-BY-4.0 | use=icd_coding_cases_and_spans
- `context_algorithm`: ConText assertion algorithm | access=paper_reference | version=publication 2009 | license=cite_paper | use=negation_temporality_experiencer_design
- `i2b2_2010`: 2010 i2b2/VA concepts, assertions, and relations challenge | access=dua_required | version=2010 challenge | license=data_use_agreement | use=assertion_and_relation_taxonomy
- `icd10_vn_tt06_2026`: ICD-10 tiếng Việt theo TT 06/2026/TT-BYT | access=open_official_pdf | version=TT 06/2026/TT-BYT | license=vietnam_government_document | use=phase1_icd10_vietnamese_primary
- `icd_kcb_vn`: Vietnamese ICD-10 lookup | access=open_manual_review | version=manual-review snapshot 2026-07-04 | license=verify_before_bulk_use | use=vietnamese_icd_labels_tt06_lookup
- `medlineplus_xml`: MedlinePlus Health Topic XML | access=open_with_attribution | version=local curation snapshot 2026-07-04 | license=attribution_required | use=aliases_public_health_topics
- `mimic_iv_note`: MIMIC-IV-Note | access=credentialed | version=credentialed local copy required | license=physionet_credentialed_health_data | use=local_private_evaluation_only
- `nbme_kaggle`: NBME Score Clinical Patient Notes | access=platform_terms | version=competition snapshot | license=kaggle_competition_terms | use=local_span_extraction_benchmark
- `negex`: NegEx | access=paper_reference | version=publication 2001 | license=cite_paper | use=negation_cue_design
- `rxnav_rest_2026_07_04`: RxNav REST API lookups on 2026-07-04 | access=open_with_terms | version=manual lookup snapshot 2026-07-04 | license=nlm_rxnorm_terms | use=phase1_drug_alias_rxcui_curation
- `rxnorm_current`: RxNorm current release and RxNav API | access=open_with_terms | version=current pointer resolved by locked source_versions.json | license=nlm_rxnorm_terms | use=rxnorm_generic_current_pointer
- `rxnorm_full_2026_06_01`: RxNorm Full Monthly Release June 1 2026 | access=open_with_terms_or_umls | version=2026-06-01 | license=nlm_rxnorm_terms | use=phase1_rxnorm_fallback_drug_codes
- `rxnorm_prescribable_2026_06_01`: RxNorm Current Prescribable Content Monthly Release June 1 2026 | access=open_with_terms | version=2026-06-01 | license=nlm_rxnorm_terms | use=phase1_rxnorm_primary_drug_codes
- `seed`: Local seed curation | access=local | version=repo-local | license=project | use=runtime_seed
- `synthea`: Synthea synthetic patient generator | access=open | version=upstream-unpinned | license=apache-2.0 | use=synthetic_patient_records
- `vn_clinical_lexicon_reviewed_2026_07_05`: Reviewed Vietnamese clinical LOCAL lexicon | access=local_reviewed | version=reviewed snapshot 2026-07-05 | license=project | use=reviewed_vietnamese_symptom_lab_procedure_aliases
- `who_icd10_2019`: WHO ICD-10 2019 ClaML and browser | access=open | version=2019 | license=who_terms_apply | use=icd10_hierarchy_reference_not_vietnamese_primary

## Files

- `seed` `data/dictionaries/seed_concepts.jsonl`: ok, required
- `seed` `data/dictionaries/vietnamese_medical_alias.jsonl`: ok, required
- `vn_clinical_lexicon_reviewed_2026_07_05` `data/standards/vn_clinical_lexicon/raw/reviewed_terms.tsv`: ok, required
- `vn_clinical_lexicon_reviewed_2026_07_05` `data/standards/vn_clinical_lexicon/processed/vn_clinical_lexicon_concepts.jsonl`: ok, required
- `vn_clinical_lexicon_reviewed_2026_07_05` `data/standards/vn_clinical_lexicon/processed/vn_clinical_lexicon_import_manifest.json`: ok, required
- `icd10_vn_tt06_2026` `data/standards/icd10_vn/raw/06-byt-kem.pdf`: ok, required
- `icd10_vn_tt06_2026` `data/standards/icd10_vn/processed/06-byt-kem.tsv`: ok, required
- `icd10_vn_tt06_2026` `data/standards/icd10_vn/processed/tt06_icd10_extract.jsonl`: ok, required
- `icd10_vn_tt06_2026` `data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl`: ok, required
- `rxnorm_prescribable_2026_06_01` `data/standards/rxnorm/raw/RxNorm_full_prescribe_06012026.zip`: ok, required
- `rxnorm_prescribable_2026_06_01` `data/standards/rxnorm/processed/rxnorm_prescribable_06012026_concepts.jsonl`: ok, required
- `rxnorm_prescribable_2026_06_01` `data/standards/rxnorm/processed/rxnorm_prescribable_06012026_import_manifest.json`: ok, required
- `rxnorm_full_2026_06_01` `data/standards/rxnorm/raw/RxNorm_full_06012026.zip`: missing, optional
- `icd10_vn_tt06_2026` `data/standards/icd10_vn/processed/tt06_icd10_extract.jsonl`: ok, required
- `icd10_vn_tt06_2026` `data/standards/icd10_vn/processed/tt06_icd10_concepts.jsonl`: ok, required
- `rxnorm_prescribable_2026_06_01` `data/standards/rxnorm/processed/rxnorm_prescribable_06012026_concepts.jsonl`: ok, required

## Dictionaries

### `data/dictionaries/seed_concepts.jsonl`

- Rows: 203
- Code systems: {'ICD-10': 100, 'LOCAL': 31, 'RxNorm': 72}
- Semantic types: {'DISEASE': 100, 'DRUG': 72, 'LAB_TEST': 14, 'SYMPTOM': 17}
- ICD hierarchy: {'rows': 100, 'with_parent_code': 100, 'with_block': 0, 'with_chapter': 0, 'by_chapter': {'<missing>': 100}}
- RxNorm enrichment: {'rows': 72, 'with_ingredient': 72, 'with_brand_name': 65, 'with_dose_form': 72, 'with_strength': 0, 'with_status': 0, 'inactive_or_obsolete': 0}
- Sources: {'cdc_icd10cm_2026': 94, 'icd_kcb_vn': 38, 'medlineplus_xml': 51, 'rxnav_rest_2026_07_04': 54, 'rxnorm_current': 70, 'seed': 39, 'who_icd10_2019': 76}
- Ambiguous aliases: 0
- Broad/blocked review aliases: 90
- Missing source rows: 0

### `data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl`

- Rows: 316
- Code systems: {'ICD-10': 146, 'LOCAL': 51, 'RxNorm': 119}
- Semantic types: {'DISEASE': 146, 'DRUG': 119, 'LAB_TEST': 21, 'SYMPTOM': 30}
- ICD hierarchy: {'rows': 146, 'with_parent_code': 131, 'with_block': 130, 'with_chapter': 146, 'by_chapter': {'I': 10, 'II': 13, 'III': 5, 'IV': 11, 'IX': 32, 'V': 11, 'VI': 7, 'VII': 5, 'X': 10, 'XI': 17, 'XII': 2, 'XIII': 5, 'XIV': 9, 'XIX': 2, 'XVII': 4, 'XVIII': 3}}
- RxNorm enrichment: {'rows': 119, 'with_ingredient': 119, 'with_brand_name': 108, 'with_dose_form': 113, 'with_strength': 16, 'with_status': 116, 'inactive_or_obsolete': 1}
- Sources: {'cdc_icd10cm_2026': 94, 'icd10_vn_tt06_2026': 112, 'icd_kcb_vn': 41, 'medlineplus_xml': 51, 'rxnav_rest_2026_07_04': 54, 'rxnorm_current': 70, 'rxnorm_prescribable_2026_06_01': 116, 'seed': 39, 'vn_clinical_lexicon_reviewed_2026_07_05': 21, 'who_icd10_2019': 76}
- Ambiguous aliases: 23
- Broad/blocked review aliases: 103
- Missing source rows: 0

## RxNorm Releases

### `data/standards/rxnorm/raw/RxNorm_full_prescribe_06012026.zip`

- Required files: {'RXNCONSO.RRF': True, 'RXNREL.RRF': True, 'RXNSAT.RRF': True}
- RXNCONSO active concepts: 81397
- RXNCONSO accepted concepts: 40667
- RXNREL active rows: 613324
- RXNSAT active rows: 467328

## Top Manual Review Items

- medium `broad_or_lab_like_alias` ICD10:E11 NIDDM: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:E11 T2DM: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:E11 DM2: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:J18.9 pneumonia: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:J18.9 PNA: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:J45 Asthma: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:I10 hypertension: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:I10 HTN: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:I10 THA: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:I21.9 MI: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:I21.9 AMI: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:I21.9 NMCT: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` LOCAL:SYMPTOM_COUGH Ho: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:J44.9 COPD: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:J44.9 BPTNMT: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:N18.9 CKD: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:N18.9 BTM: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:N17.9 AKI: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:N17.9 ARF: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:I25.10 CAD: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:I25.10 CHD: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:I50.9 HF: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:I50.9 CHF: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:K21.9 GERD: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` LOCAL:SYMPTOM_CHEST_PAIN CP: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` LOCAL:SYMPTOM_HEADACHE HA: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` LOCAL:SYMPTOM_NAUSEA_VOMITING ói: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` LOCAL:TEST_CREATININE Cr: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` LOCAL:TEST_GLUCOSE BG: Review as possible false-positive alias or blocked_alias candidate.
- medium `broad_or_lab_like_alias` ICD10:I48.91 afib: Review as possible false-positive alias or blocked_alias candidate.
