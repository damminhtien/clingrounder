# Manual Gold Candidate Audit

Date: 2026-07-07

Dictionary used for validation:

- `data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl`

Rules:

- Do not assign candidates outside the loaded controlled dictionary.
- Keep drug mentions on RxNorm and disease mentions on ICD-10.
- Preserve original spans and offsets; this pass only changes candidate lists.

## Files 1-7

- `1.json`: filled `viêm tuyến mồ hôi` with ICD-10 `L73.2` after adding the TT06/ICD concept alias.
- `2.json`: no candidate changes. Existing `K70.3` and `K76.82` remain as controlled-dictionary candidates.
- `3.json`: filled both `nhịp tim chậm tương đối` spans with `R00.1`; filled both `nhồi máu cơ tim vùng dưới cũ` spans with `I25.2`.
- `4.json`: no candidate changes. Kept `NSAID` and `NSAIDs` empty because the controlled RxNorm dictionary has no safe drug-class concept and ICD poisoning/adverse-effect codes are not valid drug candidates.
- `5.json`: filled `tắc nghẽn đường mật` with `K83.1`.
- `6.json`: no candidate changes. Existing `I65.29` remains for carotid artery occlusion/stenosis wording.
- `7.json`: filled `hội chứng nghiện rượu` with `F10.2`, `Ảo giác do rượu` with `F10.5`, and negated `loạn thần` with `F29`.
- `8.json`: filled every `nốt tuyến giáp...` occurrence with `E04.1`; did not add neoplasm or malignancy candidates because the text does not confirm those diagnoses.

Controlled dictionary additions in this pass:

- `L73.2` Hidradenitis suppurativa / Viêm tuyến mồ hôi mủ
- `R00.1` Bradycardia, unspecified / Nhịp tim chậm, không xác định
- `I25.2` Old myocardial infarction / Nhồi máu cơ tim cũ
- `K83.1` Obstruction of bile duct / Tắc nghẽn ống mật
- `F10.2` Alcohol dependence syndrome
- `F10.5` Alcohol-related psychotic disorder
- `F29` Unspecified nonorganic psychosis
- `E04.1` Nontoxic single thyroid nodule / Bướu giáp đơn nhân không độc
