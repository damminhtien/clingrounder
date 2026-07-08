# Manual Gold Candidate Audit

Date: 2026-07-07

Dictionary used for validation:

- `data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl`

Rules:

- Do not assign candidates outside the loaded controlled dictionary.
- Keep drug mentions on RxNorm and disease mentions on ICD-10.
- Preserve original spans and offsets; this pass only changes candidate lists.

## Files 1-17, 21, 100

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
- `15.json`: used existing controlled `I62.9` for nontraumatic intracranial hemorrhage and added RxNorm `221099` for `yếu tố IX đậm đặc`. Kept `Điều trị chống đông` as `THUỐC` with `[]` because no specific anticoagulant ingredient is named.
- `16.json`: used compact ICD candidates `C18.9`, `J44.9`, `E14.9`, `K70.3`, `I95.9`, `J18.9`, `I48.9`, `I47.1`, `J98.1`, and `J90`. Kept symptoms, physical findings, tests, and result values candidate-free; excluded procedure/treatment-method spans such as IV fluids and nebulization.
- `17.json`: used compact candidates `K21.9`, RxNorm `161`, RxNorm `5640`, `L02.9`, `M00.9`, `L03.9`, and `B99`. Kept symptoms, tests, lab/imaging result values, and generic `kháng sinh tĩnh mạch` candidate-free; excluded procedure, admin, and travel context spans.
- `21.json`: used compact ICD candidates `I10`, `I50.9`, `N18.9`, `I71.6`, and `Z72.0`. Kept stent graft/device wording and planned phase-2 surgery wording review-only because Phase 1 has no procedure/device type.
- `100.json`: used compact candidates `E83.5`, `C18.9`, `E21.0`, `I70.9`, `I20.8`, `G40.9`, `I63.9`, RxNorm `4917`, `1719290`, and `313002`. Kept labs, symptoms, and imaging-test names candidate-free; did not annotate generic outpatient test wording, vague acute process wording, fall event, or standalone `Truyền dịch`.

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
- `221099` RxNorm coagulation factor IX, human
- `E83.5` Disorders of calcium metabolism / Rối loạn chuyển hóa calci
- `I70.9` Generalized and unspecified atherosclerosis / Xơ vữa động mạch không xác định
- `I20.8` Other forms of angina pectoris / Cơn đau thắt ngực thể khác
- `G40.9` Epilepsy, unspecified / Bệnh động kinh, không xác định
- `I63.9` Cerebral infarction, unspecified / Nhồi máu não, không xác định
- `1719290` RxNorm 2 ML furosemide 10 MG/ML Injection
- `313002` RxNorm sodium chloride 9 MG/ML Injectable Solution
- `B99` Other and unspecified infectious diseases / Bệnh truyền nhiễm khác và/hoặc không xác định
- `L02.9` Cutaneous abscess, furuncle and carbuncle, unspecified / Áp xe da, nhọt và cụm nhọt không đặc hiệu
- `M00.9` Pyogenic arthritis, unspecified / Viêm khớp mủ không đặc hiệu
- `E14.9` Unspecified diabetes mellitus, without complications / Đái tháo đường không xác định, không kèm biến chứng
- `I48.9` Atrial fibrillation and atrial flutter, unspecified / Rung nhĩ và/hoặc cuồng nhĩ, không xác định
- `J98.1` Pulmonary collapse / Xẹp phổi
- `I71.6` Thoracoabdominal aortic aneurysm, without mention of rupture / Phình động mạch chủ ngực - bụng, không vỡ
- `Z72.0` Tobacco use / Sử dụng thuốc lá
