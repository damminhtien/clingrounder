# Manual Gold Candidate Audit

Date: 2026-07-07

Dictionary used for validation:

- `data/standards/phase1_seed_tt06_rxnorm_controlled_concepts.jsonl`

Rules:

- Do not assign candidates outside the loaded controlled dictionary.
- Keep drug mentions on RxNorm and disease mentions on ICD-10.
- Preserve original spans and offsets; this pass only changes candidate lists.

## Files 1-27, 31-34, 41-42, 96-100

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
- `18.json`: used compact ICD candidates `I80.2`, `I26.9`, and `I71.0`. Kept chest-pain symptoms and CT angiography test names candidate-free; excluded patient sex, CT indication-only pulmonary embolism wording, generic result lead-in, and section/admin headings.
- `19.json`: used compact candidates `C34.9`, `C79.3`, `I82.2`, `I10`, `I34.1`, and RxNorm `214182`. Kept symptom mentions candidate-free, including negated neurologic symptoms; did not mark Vicodin as negated when the text only says it did not relieve pain.
- `20.json`: used compact candidates for historical cardiometabolic disease, hip/lower-extremity injury, cardiopulmonary imaging diagnoses, valve insufficiency, metoprolol, and desmopressin. Corrected supplied desmopressin candidate `3264` to controlled/local RxNorm `3251` because `3264` is dexamethasone.
- `21.json`: used compact ICD candidates `I10`, `I50.9`, `N18.9`, `I71.6`, and `Z72.0`. Kept stent graft/device wording and planned phase-2 surgery wording review-only because Phase 1 has no procedure/device type.
- `22.json`: used compact candidates `I99`, RxNorm `212033`, and `I62.0`. Corrected supplied aspirin candidate `211033` to controlled/local RxNorm `212033` for `aspirin 325 MG Oral Tablet`; kept CT test names and negated neurologic symptom candidate-free.
- `23.json`: used compact ICD candidates `I25.1`, `I10`, `E78.5`, `E14.9`, `S06.6`, `S06.3`, `G93.0`, `S06.5`, and `S06.4`. Kept raw typo span `ăng huyết áp` unchanged for offset safety; symptoms and CT/test names remain candidate-free.
- `24.json`: used compact candidates `C50.9`, RxNorm `82122`, RxNorm `2231`, and RxNorm `11124`. Kept breast surgery, lymph-node surgery, JP drain, and aspiration procedure review-only because Phase 1 has no procedure/device type.
- `25.json`: used compact ICD candidates `I71.0`, `G82.2`, and `I77.0`. Kept endovascular intervention and vascular access wording review-only because Phase 1 has no procedure type.
- `26.json`: used compact ICD candidate `E66.9` for historical obesity. Kept weight-change symptoms candidate-free and excluded diet/surgery context.
- `27.json`: used compact candidates `Q96.9`, `I10`, `C18.9`, `I26.9`, RxNorm `11289`, RxNorm `214199`, and RxNorm `866508`. Corrected supplied Coumadin candidate `202421` to controlled RxNorm `11289`; kept procedure and generic GI-evaluation wording review-only.
- `31.json`: used compact ICD candidate `O48` for `Thai 41 tuần`. Kept obstetric symptoms candidate-free and excluded labor-induction wording.
- `32.json`: used compact candidates `C92.1`, `I10`, `E11.9`, `I48.9`, `N18.9`, `J96.9`, `A41.0`, `I82.9`, `I26.9`, RxNorm `435`, `5032`, and `313988`. Kept leading `ho` history artifacts out of symptom gold.
- `33.json`: used compact candidates `E66.9`, `E11.9`, `G47.3`, `I50.9`, `R09.0`, RxNorm `4603`, `1808`, `1807513`, `1665515`, `212033`, `1437702`, and `1743704`. Kept oxygen mask, EMS logistics, and diuresis-volume wording review-only.
- `34.json`: used compact ICD candidates `K51.9` and `K70.9`. Kept opioid analgesic class mentions as `THUỐC` with `[]`, and excluded surgery/procedure plus LLQ/anatomic location spans.
- `41.json`: used compact GI candidates `A08.4`, `K58.9`, `K26.9`, `K63.3`, `K20`, `K22.1`, and RxNorm `7646`. Did not split `cryptosporidium` or `h. pylori` into diagnoses because they appear only as test target/result context.
- `42.json`: used compact candidates `I51.9`, `N28.9`, and RxNorm `6813`. Kept current symptoms candidate-free, marked explicit `phủ nhận` symptoms negated, and did not treat `không thể tỉnh táo đủ lâu` as negation.
- `96.json`: used compact candidates `M86.6`, `N31.9`, `G82.2`, `L89.3`, `N39.0`, RxNorm `11124`/`74169` for glued `vancozosyn`, RxNorm `10831` for `bactrim`, `I95.9`, `A41.9`, `M86.9`, `R91.8`, RxNorm `313002`, `20481`, and `11124`. Corrected supplied Bactrim candidate `151399` to controlled RxNorm `10831`; kept devices, procedures, generic antibiotics, normal exam phrases, and standalone temperature review-only.
- `97.json`: used compact candidates `M86.9`, `N31.9`, `G82.2`, `R68.0`, `I95.9`, `N39.0`, `J18.9`, and `R00.1`. Kept labs/tests candidate-free and excluded catheter/device and vital-sign wording.
- `98.json`: used compact candidates `I10`, `M89.50`, and `C90.0`. Kept CT/IgA test and result mentions candidate-free; excluded stem-cell mobilization chemotherapy as treatment/procedure context.
- `99.json`: used existing controlled `F41.9` for historical unspecified anxiety disorder. Kept repeated current symptoms candidate-free and excluded prostate surgery wording.
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
- `I80.2` Phlebitis and thrombophlebitis of other deep vessels of lower extremities / Viêm tĩnh mạch và/hoặc viêm [tắc] tĩnh mạch huyết khối của tĩnh mạch sâu khác ở chi dưới
- `I26.9` Pulmonary embolism without mention of acute cor pulmonale / Thuyên tắc mạch phổi không có tâm phế cấp tính
- `I71.0` Dissection of aorta [any part] / Tách thành động mạch chủ [bất kỳ đoạn nào]
- `C34.9` Bronchus or lung, unspecified / U ác tính ở phế quản hoặc phổi, không xác định
- `C79.3` Secondary malignant neoplasm of brain and cerebral meninges / U ác tính thứ phát ở não và/hoặc màng não
- `I82.2` Embolism and thrombosis of vena cava / Thuyên tắc và/hoặc huyết khối tĩnh mạch chủ
- `I99` Other and unspecified disorders of circulatory system / Rối loạn hệ tuần hoàn khác và/hoặc không xác định
- `I62.0` Nontraumatic subdural haemorrhage / Xuất huyết dưới màng cứng không do chấn thương
- `212033` RxNorm aspirin 325 MG Oral Tablet
- `R26.9` Unspecified abnormalities of gait and mobility / Rối loạn dáng đi không xác định
- `I31.3` Pericardial effusion (noninflammatory) / Tràn dịch màng ngoài tim (không do viêm)
- `I35.1` Aortic (valve) insufficiency / Hở van động mạch chủ
- `I36.1` Nonrheumatic tricuspid (valve) insufficiency / Hở van ba lá không do bệnh thấp
- `J43.9` Emphysema, unspecified / Khí phế thũng không xác định
- `K44.9` Diaphragmatic hernia without obstruction or gangrene / Thoát vị cơ hoành
- `G93.0` Cerebral cysts / Bệnh u nang não
- `S06.3` Focal brain injury / Tổn thương não khu trú
- `S06.4` Epidural haemorrhage / Xuất huyết ngoài màng cứng
- `S06.5` Traumatic subdural haemorrhage / Xuất huyết dưới màng cứng do chấn thương
- `S06.6` Traumatic subarachnoid haemorrhage / Xuất huyết dưới màng nhện do chấn thương
- `S09.9` Unspecified injury of head / Tổn thương không xác định ở đầu
- `S81.8` Open wound of other parts of lower leg / Vết thương hở ở phần khác của chi dưới
- `S89.9` Unspecified injury of lower leg / Tổn thương không xác định ở cẳng chân
- `A41.0` Sepsis due to Staphylococcus aureus / Nhiễm trùng hệ thống do tụ cầu vàng
- `C50.9` Breast, unspecified / U ác tính ở vú, không xác định
- `E11.9` Type 2 diabetes mellitus, without complications / Bệnh đái tháo đường típ 2, không kèm biến chứng
- `I82.9` Embolism and thrombosis of unspecified vein / Thuyên tắc và/hoặc huyết khối, không xác định tĩnh mạch
- `J96.9` Respiratory failure, unspecified / Suy hô hấp, không xác định
- `M89.50` Osteolysis, multiple sites / Bệnh tiêu xương, nhiều vị trí
- `313988` RxNorm furosemide 40 MG Oral Tablet
- `K26.9` Duodenal ulcer, unspecified / Loét tá tràng, không xác định
- `K58.9` Irritable bowel syndrome, unspecified
- `K63.3` Ulcer of intestine / Loét ruột
- `G82.2` Paraplegia, unspecified / Hội chứng liệt nửa người dưới thắt lưng, không xác định
- `I77.0` Arteriovenous fistula, acquired / Rò động tĩnh mạch, mắc phải
- `N28.9` Disorder of kidney and ureter, unspecified / Rối loạn thận và niệu quản, không xác định
- `N39.0` Urinary tract infection, site not specified / Nhiễm khuẩn đường tiết niệu, vị trí không xác định
- `O48` Prolonged pregnancy / Thai kỳ quá ngày sinh
- `R68.0` Hypothermia, not associated with low environmental temperature / Hạ thân nhiệt, không liên quan đến nhiệt độ môi trường thấp
- `866508` RxNorm 5 ML metoprolol tartrate 1 MG/ML Injection
- `G47.3` Sleep apnoea / Ngưng thở khi ngủ
- `M86.6` Other chronic osteomyelitis / Viêm xương tủy mạn tính khác
- `L89.3` Stage IV decubitus ulcer / Loét do tì đè giai đoạn IV
- `R09.0` Asphyxia / Ngạt thở
- `R91.8` Other nonspecific abnormal finding of lung field / Phát hiện bất thường không đặc hiệu khác của trường phổi
- `74169` RxNorm piperacillin / tazobactam
- `1437702` RxNorm albuterol 0.833 MG/ML / ipratropium bromide 0.167 MG/ML Inhalation Solution
- `1665515` RxNorm 150 ML levofloxacin 5 MG/ML Injection
- `1743704` RxNorm methylprednisolone 125 MG Injection
- `1807513` RxNorm vancomycin 1000 MG Injection
