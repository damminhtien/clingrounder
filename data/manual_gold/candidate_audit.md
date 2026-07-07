# Manual Gold Candidate Audit

Date: 2026-07-07

Dictionary used for validation:

- `data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl`

Rules:

- Do not assign candidates outside the loaded controlled dictionary.
- Keep drug mentions on RxNorm and disease mentions on ICD-10.
- Preserve original spans and offsets; this pass only changes candidate lists.

## Files 1-14

- `1.json`: filled `viêm tuyến mồ hôi` with ICD-10 `L73.2` after adding the TT06/ICD concept alias.
- `2.json`: no candidate changes. Existing `K70.3` and `K76.82` remain as controlled-dictionary candidates.
- `3.json`: filled both `nhịp tim chậm tương đối` spans with `R00.1`; filled both `nhồi máu cơ tim vùng dưới cũ` spans with `I25.2`.
- `4.json`: no candidate changes. Kept `NSAID` and `NSAIDs` empty because the controlled RxNorm dictionary has no safe drug-class concept and ICD poisoning/adverse-effect codes are not valid drug candidates.
- `5.json`: filled `tắc nghẽn đường mật` with `K83.1`.
- `6.json`: no candidate changes. Existing `I65.29` remains for carotid artery occlusion/stenosis wording.
- `7.json`: filled `hội chứng nghiện rượu` with `F10.2`, `Ảo giác do rượu` with `F10.5`, and negated `loạn thần` with `F29`.
- `8.json`: filled every `nốt tuyến giáp...` occurrence with `E04.1`; did not add neoplasm or malignancy candidates because the text does not confirm those diagnoses.
- `9.json`: used existing controlled candidates `D25.9` for non-specific uterine leiomyoma/fibroid mentions and RxNorm `90176` for bare `iron`.
- `10.json`: filled malignant rectal tumor context with `C20`; filled rectal biopsy adenoma context with `D12.8`. Kept all tests, lab results, imaging findings, and symptoms candidate-free.
- `11.json`: filled historical chronic diagnoses with controlled ICD-10 candidates `I10`, `M10.9`, `N18.9`, `C18.9`, `E66.9`, `K74.6`, `K76.6`, `R18`, and `J90`. Kept `liệu pháp lợi tiểu` as `THUỐC` with `[]` because no safe RxNorm active ingredient/class candidate is available in the controlled dictionary.
- `12.json`: used small high-confidence candidates `D86.8`, `J40`, `J18.9`, `J81`, RxNorm `82122`, and RxNorm `161`. Kept symptoms, tests, and lab-result values candidate-free; did not annotate `Lấy mẫu` separately from `cấy máu`.
- `13.json`: used compact candidates `J70.3`, `M35.8`, `E66.9`, `I10`, `D84.8`, `L03.9`, `B00.9`, `B01.9`, `B02.9`, RxNorm `10831`, and RxNorm `3640`. Kept oxygen therapy and corticosteroid class mentions as `THUỐC` with `[]`; split glued `doxycyclinebactrim` into adjacent drug spans.
- `14.json`: used compact cardiovascular/allergy candidates `J30.2`, `J30.1`, `I25.9`, and `I25.1`. Kept stress test, perfusion scan, coronary angiography, and generic abnormal result candidate-free; did not annotate planned CABG/procedure wording.

Controlled dictionary additions in this pass:

- `L73.2` Hidradenitis suppurativa / Viêm tuyến mồ hôi mủ
- `R00.1` Bradycardia, unspecified / Nhịp tim chậm, không xác định
- `I25.2` Old myocardial infarction / Nhồi máu cơ tim cũ
- `K83.1` Obstruction of bile duct / Tắc nghẽn ống mật
- `F10.2` Alcohol dependence syndrome
- `F10.5` Alcohol-related psychotic disorder
- `F29` Unspecified nonorganic psychosis
- `E04.1` Nontoxic single thyroid nodule / Bướu giáp đơn nhân không độc
- `C20` Malignant neoplasm of rectum / U ác tính ở trực tràng
- `D12.8` Benign neoplasm of rectum / U lành ở trực tràng
- `M10.9` Gout, unspecified / Bệnh gút không đặc hiệu
- `K74.6` Other and unspecified cirrhosis of liver / Xơ gan khác và không xác định
- `K76.6` Portal hypertension / Tăng áp lực tĩnh mạch cửa
- `R18` Ascites / Chứng cổ trướng
- `D86.8` Sarcoidosis of other and combined sites / Bệnh u hạt vị trí khác và/hoặc vị trí kết hợp
- `J70.3` Chronic drug-induced interstitial lung disorders / Rối loạn phổi mô kẽ mạn tính do thuốc
- `D84.8` Other specified immunodeficiencies / Suy giảm miễn dịch xác định khác
- `L03.9` Cellulitis, unspecified / Viêm mô tế bào không đặc hiệu
- `M35.8` Other specified systemic involvement of connective tissue
- `J30.1` Allergic rhinitis due to pollen / Viêm mũi dị ứng do phấn hoa
- `J30.2` Other seasonal allergic rhinitis / Viêm mũi dị ứng theo mùa khác
- `I25.9` Chronic ischaemic heart disease, unspecified / Bệnh tim thiếu máu cục bộ mạn tính, không xác định
