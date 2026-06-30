Công thức tổng quát:

[
\boxed{
\text{Distribution Profiling}
+
\text{Dictionary/Code Prior}
+
\text{Section/Context Prior}
+
\text{Ontology Constraint}
+
\text{Metric-aware Thresholding}
+
\text{Error Loop}
}
]

Bài này bản chất không phải NER đơn thuần mà là clinical IE end-to-end gồm NER, entity linking, ICD/RxNorm coding, context reasoning, relation extraction và ontology reasoning.  Vì vậy “hack” mạnh nhất nằm ở **normalization, context rule, schema validity, và ontology validation**, không nằm ở model lớn.

---

# 1. Hack số 1: đo distribution ngay khi có train/dev

Khi dataset release, việc đầu tiên không phải train model. Việc đầu tiên là viết `profile_data.py`.

Cần thống kê:

```text
1. Entity type distribution
2. Code frequency distribution
3. Mention → code frequency
4. Section header distribution
5. Assertion/context distribution
6. Relation type distribution
7. Note length distribution
8. Abbreviation frequency
9. No-diacritic / typo rate
10. Train-dev code overlap
11. Entity boundary pattern
12. Null / missing field convention
```

Ví dụ output cần có:

```text
Top disease codes:
E11: 9.8%
I10: 8.7%
J18.9: 5.4%
J45: 4.1%
N18.3: 2.9%

Top drug mentions:
metformin
insulin
salbutamol
paracetamol
amoxicillin

Top context markers:
không ghi nhận
tiền sử
nghi
theo dõi
mẹ/bố/cha
```

Sau đó tạo các prior:

[
P(\text{code} \mid \text{mention})
]

[
P(\text{assertion} \mid \text{section})
]

[
P(\text{relation} \mid \text{entity_type}_1, \text{entity_type}_2)
]

Đây là vũ khí cực mạnh vì nhiều benchmark y khoa có **label repetition cao**. Một mention như “THA” gần như luôn map về tăng huyết áp, trừ vài case hiếm.

---

# 2. Hack số 2: xây “mention-code memory” từ train set

Tạo bảng:

```text
normalized_mention -> most_common_code
```

Ví dụ:

```text
"đái tháo đường type 2" -> E11
"tiểu đường type 2" -> E11
"dtđ type 2" -> E11
"tha" -> I10
"copd" -> J44
"hen phế quản" -> J45
"metformin" -> RxNorm/metformin_code
```

Nếu train/dev distribution giống private test, bảng này ăn rất nhiều điểm.

Pseudo-policy:

```python
if normalized_mention in train_mention_code_table:
    return most_frequent_code(normalized_mention)
else:
    return hybrid_retrieval_linker(mention)
```

Nhưng phải normalize mạnh:

```text
lowercase
remove accents
normalize whitespace
remove punctuation
standardize hyphen
standardize roman numerals: II -> 2
normalize type/typ/tuýp
normalize Vietnamese-English variants
```

Ví dụ:

```text
"ĐTĐ typ II"
"DTĐ type 2"
"dai thao duong type ii"
"tiểu đường type 2"
```

đều phải về cùng key:

```text
dai thao duong type 2
```

Đây là distribution hack hợp lệ và rất hiệu quả.

---

# 3. Hack số 3: ưu tiên normalization hơn model lớn

Trong bài này, nếu span đúng nhưng code sai thì vẫn mất rất nhiều điểm. Nếu gold code không nằm trong top-k candidates thì reranker không cứu được; tài liệu loop engineering cũng nhấn mạnh candidate recall@20 là metric sống còn, mục tiêu nên là trên 95%. 

Do đó, thay vì dồn GPU train NER lớn, ta nên dồn công vào:

```text
ICD dictionary
RxNorm/drug dictionary
Vietnamese synonym table
abbreviation table
no-diacritic table
typo table
brand-name/generic-name table
BM25 index
char n-gram TF-IDF
dense retrieval optional
cross-encoder rerank optional
```

Pipeline nên là:

[
\text{mention}
\rightarrow
\text{exact}
\rightarrow
\text{alias}
\rightarrow
\text{abbreviation}
\rightarrow
\text{fuzzy}
\rightarrow
\text{BM25}
\rightarrow
\text{dense}
\rightarrow
\text{rerank}
\rightarrow
\text{ontology filter}
]

Score:

[
s(c)
====

\alpha s_{exact}
+
\beta s_{prior}
+
\gamma s_{bm25}
+
\delta s_{char}
+
\eta s_{dense}
+
\lambda s_{context}
+
\mu s_{ontology}
]

Với cá nhân ít GPU, **exact/alias/abbreviation/BM25/char n-gram** có ROI cao hơn fine-tune model lớn.

---

# 4. Hack số 4: tận dụng section header như feature mạnh

Bệnh án thường có cấu trúc. Nếu dataset do người ra đề thiết kế, họ rất có thể giữ các section kiểu:

```text
Lý do vào viện
Bệnh sử
Tiền sử bản thân
Tiền sử gia đình
Khám lâm sàng
Cận lâm sàng
Chẩn đoán
Điều trị
Đơn thuốc
Kế hoạch
```

Section prior:

```text
"Tiền sử"              -> HISTORICAL
"Tiền sử gia đình"     -> FAMILY
"Chẩn đoán"            -> PRESENT / CONFIRMED
"Đơn thuốc"            -> DRUG / CURRENT_MED
"Kế hoạch"             -> PLANNED
"Cận lâm sàng"         -> LAB_TEST / LAB_RESULT
```

Rule:

[
P(\text{FAMILY} \mid \text{section = family history}) \approx 1
]

[
P(\text{HISTORICAL} \mid \text{section = past medical history}) \uparrow
]

Rất nhiều đội sẽ chỉ feed raw text vào model. Ta phải biến section thành feature rõ ràng.

---

# 5. Hack số 5: rule phủ định/nghi ngờ/tiền sử ăn điểm rẻ

Các lỗi clinical nguy hiểm nhất là negation, family, historical; tài liệu cũng nêu rõ nhóm này cần đo riêng và là nhóm lỗi quan trọng. 

Ta nên xây rule engine trước model.

## Negation markers

```text
không
chưa
không ghi nhận
không phát hiện
không thấy
không có bằng chứng
loại trừ
âm tính với
denies
no evidence of
negative for
```

## Uncertainty markers

```text
nghi
theo dõi
chưa loại trừ
khả năng
có thể
gợi ý
?
rule out
suspected
possible
likely
```

## Historical markers

```text
tiền sử
đã từng
trước đây
năm 2020
cũ
sau điều trị
đã khỏi
```

## Family markers

```text
bố
mẹ
cha
anh
chị
em
con
gia đình
family history
```

Critical pair:

```text
Không ghi nhận viêm phổi       -> NEGATED
Không loại trừ viêm phổi       -> POSSIBLE
Theo dõi viêm phổi             -> POSSIBLE
Tiền sử viêm phổi              -> HISTORICAL
Mẹ bệnh nhân viêm phổi         -> FAMILY
```

Đây là “hack” rẻ nhưng rất mạnh.

---

# 6. Hack số 6: conservative relation strategy

Relation extraction thường khó. Nếu metric relation phạt FP nặng, đừng generate quá nhiều relation. Nên dùng **high-precision relation rules** trước.

Ví dụ type constraints:

```text
DRUG -> TREATS -> DISEASE/SYMPTOM
DRUG -> HAS_DOSAGE -> DOSAGE
DRUG -> HAS_FREQUENCY -> FREQUENCY
LAB_TEST -> TEST_VALUE -> LAB_VALUE
TEST -> REVEALS -> DISEASE
DISEASE -> HAS_SYMPTOM -> SYMPTOM
```

Không cho:

```text
LAB_RESULT -> TREATS -> DISEASE
DISEASE -> HAS_DOSAGE -> DRUG
SYMPTOM -> REVEALS -> TEST
```

Tài liệu trước cũng khuyến nghị relation dùng candidate pairs, entity-marker transformer và KG/type constraints. 

Chiến lược:

```text
Nếu confidence cao -> xuất relation
Nếu không chắc -> bỏ
Nếu relation có trong ontology -> tăng confidence
Nếu relation vi phạm type -> remove
```

Trong nhiều scoring F1, relation FP rất độc. High precision có thể tốt hơn high recall, nhất là nếu relation chỉ chiếm 10–20% tổng score.

---

# 7. Hack số 7: ontology validator như “last-mile score booster”

Ontology không chỉ để reasoning. Nó là bộ lọc sửa lỗi.

Hard constraints:

```text
DRUG không được map ICD-10
DISEASE không được map RxNorm
LAB_TEST không được TREATS disease
NEGATED disease không được dùng làm confirmed diagnosis
FAMILY disease không được gán PRESENT cho bệnh nhân
```

Ví dụ lỗi phải chặn:

```json
{
  "text": "metformin",
  "type": "DRUG",
  "code_system": "ICD-10",
  "code": "E11"
}
```

Đúng hơn:

```json
{
  "text": "metformin",
  "type": "DRUG",
  "code_system": "RxNorm",
  "code": "...",
  "assertion": "PRESENT"
}
```

và relation:

```json
{
  "head": "metformin",
  "tail": "đái tháo đường type 2",
  "type": "TREATS",
  "evidence": "ontology"
}
```

Trong tài liệu trước cũng nhấn mạnh ontology validation để loại lỗi vô lý như drug bị map sang ICD-10. 

---

# 8. Hack số 8: schema và offset phải 100% sạch

Đây là điểm nhiều đội tự giết mình.

Nếu submission sai JSON, sai field name, sai type, sai span offset, hoặc code không nằm trong vocabulary, model tốt cũng vô nghĩa. Tài liệu loop engineering đặt schema metrics như `schema_valid_rate`, `json_parse_success_rate`, `invalid_code_rate`, `invalid_relation_rate`, `ontology_violation_rate`. 

Phải có:

```text
validate_submission.py
validate_offsets.py
validate_codes.py
validate_relations.py
validate_ontology.py
```

Invariant bắt buộc:

```python
assert text[start:end] == predicted_text
assert entity.type in allowed_entity_types
assert entity.code in allowed_codes[entity.code_system]
assert relation.head in entity_ids
assert relation.tail in entity_ids
assert relation.type in allowed_relation_types
```

Đây là hack “không sexy” nhưng cực đáng tiền.

---

# 9. Hack số 9: tune threshold theo metric, không theo cảm giác

Mỗi module cần threshold riêng.

Ví dụ entity extraction:

```text
SYMPTOM: threshold thấp hơn để tăng recall
DISEASE: threshold trung bình
DRUG: threshold thấp nếu dictionary match
LAB_VALUE: threshold cao hơn để tránh số rác
```

Relation:

```text
TREATS: chỉ xuất nếu confidence cao hoặc ontology có edge
HAS_DOSAGE: xuất nếu gần drug trong cùng câu
TEST_VALUE: xuất nếu gần lab test trong window nhỏ
HAS_SYMPTOM: xuất nếu disease + symptom trong cùng sentence/section
```

Objective:

[
\theta^*
========

\arg\max_{\theta}
Score_{local}(\theta)
]

Không dùng một threshold chung cho tất cả. Đó là lãng phí distribution.

---

# 10. Hack số 10: synthetic stress set theo đúng bẫy của người ra đề

Trước khi có test, tự tạo bộ stress test:

```text
1. Negation
2. Family history
3. Historical
4. Possible/uncertain
5. Drug dosage/frequency
6. Lab value/unit
7. No-diacritic
8. Abbreviation
9. Typo
10. Long note
11. Nested entity
12. Relation direction
```

Ví dụ file `stress_context.jsonl`:

```text
Không ghi nhận hen phế quản.
Không loại trừ hen phế quản.
Theo dõi hen phế quản.
Tiền sử hen phế quản.
Mẹ bệnh nhân bị hen phế quản.
Bệnh nhân sẽ được nội soi ngày mai.
```

Mỗi lần sửa code phải chạy stress set. Đây là cách tránh regression khi loop nhanh.

---

# 11. Hack số 11: exploit train-test overlap hợp lệ

Trong các cuộc thi NLP y khoa, train/test thường overlap mạnh ở:

```text
common diseases
common drugs
common abbreviations
section templates
hospital writing style
ICD code subset
relation schema
```

Ta nên đo:

[
Overlap_{mention}
=================

\frac{|\text{mentions}*{dev} \cap \text{mentions}*{train}|}
{|\text{mentions}_{dev}|}
]

[
Overlap_{code}
==============

\frac{|\text{codes}*{dev} \cap \text{codes}*{train}|}
{|\text{codes}_{dev}|}
]

Nếu overlap cao:

```text
mention-code memory rất mạnh
```

Nếu overlap thấp:

```text
retrieval + synonym + ontology mạnh hơn
```

Policy:

```python
if mention_seen_in_train:
    use_memorized_mapping()
elif alias_seen:
    use_alias_mapping()
else:
    use_retrieval_reranker()
```

Đây là một trong những cách “win chặt” nhất.

---

# 12. Hack số 12: parent-code fallback cho ICD

Nếu không chắc full ICD code, fallback về parent/common code.

Ví dụ:

```text
Gold có thể là J18.9
Model phân vân J18 / J18.0 / J18.9
```

Nếu metric có partial credit theo hierarchy, parent fallback có lợi. Nếu metric exact-only, chọn most frequent child trong train.

Policy:

[
c^*
===

\arg\max_c
\left[
s(c)
+
\lambda \log P(c)
+
\mu \log P(c \mid parent)
\right]
]

Ví dụ:

```text
mention = "viêm phổi"
candidates = J18, J18.0, J18.9
choose = most frequent train code for "viêm phổi"
```

Nếu train cho thấy 80% “viêm phổi” là `J18.9`, chọn `J18.9`.

---

# 13. Hack số 13: no-diacritic + typo augmentation

Dữ liệu bệnh án tiếng Việt rất dễ có:

```text
dai thao duong
tang huyet ap
viem phoi
kho tho
metfomin
salbutamon
```

Tạo augmentation dictionary:

```text
đái tháo đường -> dai thao duong
tăng huyết áp -> tang huyet ap
viêm phổi -> viem phoi
khó thở -> kho tho
metformin -> metfomin, metformine
salbutamol -> salbutamon, salbutamol khí dung
```

Không nhất thiết train model. Chỉ cần retriever robust:

```text
char 3-gram TF-IDF
edit distance
accent-insensitive matching
token sort ratio
```

Char n-gram thường cực mạnh với typo/no-diacritic.

---

# 14. Hack số 14: default policy khi không chắc

Cần thiết kế fallback rõ ràng.

Ví dụ:

## Entity type fallback

```text
Nếu mention nằm trong drug dictionary -> DRUG
Nếu mention nằm trong ICD dictionary -> DISEASE
Nếu có số + unit sau lab keyword -> LAB_VALUE
Nếu nằm trong symptom lexicon -> SYMPTOM
```

## Assertion fallback

```text
section = family history -> FAMILY
section = past medical history -> HISTORICAL
negation marker near entity -> NEGATED
uncertainty marker near entity -> POSSIBLE
else -> PRESENT
```

## Relation fallback

```text
drug + disease same sentence + drug treats disease in ontology -> TREATS
lab_test + lab_value within 8 tokens -> TEST_VALUE
disease + symptom same sentence + ontology has relation -> HAS_SYMPTOM
else -> no relation
```

Trong clinical IE, default tốt hơn model mơ hồ.

---

# 15. Hack số 15: public leaderboard shadow validation

Không tối ưu mù public LB. Nhưng vẫn phải khai thác tín hiệu public hợp lý.

Cách làm:

```text
1. Freeze local dev.
2. Submit baseline.
3. Submit từng thay đổi nhỏ.
4. Ghi local delta và public delta.
5. Ước lượng public-private mismatch.
```

Bảng theo dõi:

```text
exp_id | local_delta | public_delta | decision
N004   | +0.012      | +0.010       | keep
C006   | +0.004      | +0.020       | inspect; maybe public has many uncertainty cases
R003   | +0.008      | -0.015       | risky; relation overfit
```

Tài liệu loop engineering cũng khuyến nghị không tối ưu mù leaderboard, mà phải tách train/dev/local_holdout/public_lb_shadow và dùng public LB ít để xác nhận trend. 

---

# 16. Thứ tự ưu tiên để win với ít hardware

Nếu làm cá nhân, tôi sẽ ưu tiên như sau:

## Priority 0 — Submission không chết

```text
schema validator
offset validator
code validator
relation validator
```

## Priority 1 — Dictionary/memory layer

```text
train mention-code memory
ICD aliases
drug aliases
abbreviations
Vietnamese synonyms
no-diacritic keys
```

## Priority 2 — Context rules

```text
NEGATED
FAMILY
HISTORICAL
POSSIBLE
PLANNED
```

## Priority 3 — Hybrid normalization

```text
exact
fuzzy
BM25
char n-gram
dense optional
reranker optional
```

## Priority 4 — Simple NER + postprocess

```text
PhoBERT / XLM-R / domain model
BIOES
sliding window
span merge
threshold tuning
```

## Priority 5 — Relation high precision

```text
type constraints
distance constraints
ontology relation
few classifier/rules
```

## Priority 6 — Ensemble nhẹ

```text
rule + model
dictionary + model
2–3 seed ensemble nếu còn thời gian
```

---

# 17. Cấu trúc hệ thống nên build

```text
air-med/
  data_profile/
    profile_entities.py
    profile_codes.py
    profile_sections.py
    profile_relations.py

  dictionaries/
    icd_aliases.csv
    drug_aliases.csv
    abbreviations.csv
    vietnamese_synonyms.csv
    typo_variants.csv

  src/
    preprocessing/
      normalize_text.py
      offset_mapper.py
      section_splitter.py

    entity/
      dictionary_matcher.py
      ner_model.py
      span_merger.py

    linking/
      mention_memory.py
      bm25_linker.py
      char_ngram_linker.py
      reranker.py
      linker_ensemble.py

    context/
      context_rules.py
      context_classifier.py

    relation/
      relation_rules.py
      relation_classifier.py
      relation_constraints.py

    ontology/
      validator.py
      kg.py

    scoring/
      local_score.py
      error_analysis.py
      threshold_tuning.py

    submission/
      exporter.py
      validator.py
```

---

# 18. “Win chặt” strategy trong 7 ngày đầu khi có data

## Ngày 1

```text
profile data
schema validator
offset validator
baseline dictionary matcher
local scorer
```

## Ngày 2

```text
mention-code memory
abbreviation table
no-diacritic normalization
exact/fuzzy linker
```

## Ngày 3

```text
context rule engine
section splitter
family/history/negation tests
```

## Ngày 4

```text
BM25 + char n-gram retrieval
candidate recall@20 report
ontology code validator
```

## Ngày 5

```text
small NER model
BIOES
span postprocess
boundary error analysis
```

## Ngày 6

```text
relation high-precision rules
threshold tuning
public LB sanity check
```

## Ngày 7

```text
error taxonomy
fix top 3 errors
freeze local holdout
prepare robust submission
```

Đây là đường cá nhân thực dụng nhất.

---

# 19. Kết luận chiến thuật

Muốn “hack distribution và win chặt”, ta không nên hỏi:

```text
Model nào mạnh nhất?
```

Mà phải hỏi:

```text
Dataset lặp lại gì?
Metric thưởng/phạt gì?
Gold code nằm trong vocabulary nào?
Mention nào xuất hiện nhiều?
Section nào quyết định context?
Rule nào sửa được nhiều lỗi nhất?
Threshold nào tối đa score?
Ontology constraint nào chặn lỗi ngu?
```

Công thức tôi chọn:

[
\boxed{
\text{Memorize what is stable}
+
\text{Retrieve what is unseen}
+
\text{Rule what is clinical}
+
\text{Constrain what is impossible}
+
\text{Tune what affects metric}
}
]

Nếu làm đúng, ta có thể thắng nhiều đội dùng model lớn hơn, vì bài này nhiều khả năng là bài **distribution-aware engineering**, không phải brute-force GPU.
