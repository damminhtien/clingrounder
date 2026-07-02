## Kết luận nhanh

Đề này **không phải bài chatbot y tế** và cũng **không chỉ là NER**. Đây là bài:

[
\boxed{
\text{Clinical Information Extraction}
+
\text{Entity Normalization}
+
\text{Assertion Detection}
+
\text{Ontology-constrained Validation}
}
]

Các phân tích trước của bạn cũng đã chốt đúng bản chất: bài này kết hợp Clinical NLP, NER, Entity Linking, ICD/RxNorm Coding, Context Reasoning, Relation Extraction và Ontology/KG Reasoning; không nên xem là NER đơn thuần. 

Nhưng với **schema Phase 1 đang đưa ra**, phần cần ưu tiên nhất là:

[
\boxed{
\text{span đúng}
+
\text{type đúng}
+
\text{assertion đúng}
+
\text{ICD/RxNorm candidate đúng}
+
\text{JSON/offset tuyệt đối sạch}
}
]

---

# 1. Phân tích timeline và hình thức nộp

## Câu đề

> Phase 1 — Vòng 1 - Sơ loại — 02/07/2026 → 30/07/2026 — Tệp ZIP — GPU

## Ý nghĩa kỹ thuật

Phase 1 là **offline batch inference**. Bạn nhận `test.zip`, sinh output `.json`, rồi nộp lại ZIP. Đây chưa phải bài serving API.

Điểm mấu chốt:

| Chi tiết           | Hàm ý                                                                   |
| ------------------ | ----------------------------------------------------------------------- |
| `Tệp ZIP`          | Cần exporter chuẩn: `output/1.json`, `output/2.json`, …                 |
| `GPU`              | Có thể dùng GPU, nhưng không bắt buộc nếu pipeline dictionary/rule mạnh |
| `100 bản ghi test` | Quy mô nhỏ, rất phù hợp với hybrid system + LLM hỗ trợ + validator      |
| Không có train     | Không thể dựa hoàn toàn vào supervised fine-tuning                      |

Với Phase 1, chiến lược đúng không phải “train model thật lớn”, mà là **xây hệ thống chắc, đúng schema, có dictionary, có rule, có validation**. Tài liệu loop engineering của bạn cũng nhấn mạnh phải chia pipeline thành module và đo riêng từng tầng thay vì chỉ thử model lớn. 

---

# 2. Phân tích câu tổng quan đầu tiên

## Câu đề

> Bài toán yêu cầu xây dựng hệ thống AI xử lý văn bản y khoa tự do - ghi chú bác sĩ, giấy xuất viện, kết quả xét nghiệm, hồ sơ EHR - để phát hiện và chuẩn hóa các khái niệm y tế xuất hiện trong văn bản.

## Tách nghĩa

Câu này chứa 3 nhiệm vụ chính:

[
\text{free-form clinical text}
\rightarrow
\text{detect medical concepts}
\rightarrow
\text{normalize concepts}
]

## Điểm mấu chốt

### 1. “Văn bản y khoa tự do”

Không phải dữ liệu dạng bảng. Input có thể lộn xộn:

```text
BN nam 70t, ho đờm xanh 1 tuần, TS dùng Chlorpheniramine...
```

Có thể có:

| Hiện tượng                    | Ví dụ                        | Rủi ro                           |
| ----------------------------- | ---------------------------- | -------------------------------- |
| Viết tắt                      | BN, TS, HA, ĐTĐ, THA         | Model bỏ sót hoặc map sai        |
| Không dấu                     | tang huyet ap                | Dictionary thường không match    |
| Sai chính tả                  | lazer, metfomin              | Exact match chết                 |
| Nhiều khái niệm trong một câu | ho, sốt, đau thượng vị, GERD | Cần multi-span extraction        |
| Văn bản dài                   | EHR / xuất viện              | Cần sliding window hoặc chunking |

### 2. “Phát hiện khái niệm”

Đây là **span extraction**:

[
x = \text{text}
]

[
E = {(start_i, end_i, text_i, type_i)}_{i=1}^{n}
]

Tức là phải tìm chính xác cụm từ trong input.

### 3. “Chuẩn hóa khái niệm”

Không đủ khi chỉ extract `"trào ngược dạ dày"`. Phải map sang mã chuẩn:

[
\text{"trào ngược dạ dày - thực quản"}
\rightarrow
{K21.0, K21.9}
]

[
\text{"Chlorpheniramine"}
\rightarrow
{360047}
]

Đây là **entity linking / concept normalization**, thường là phần ăn điểm lớn nhất.

---

# 3. Phân tích câu “xác định loại khái niệm”

## Câu đề

> Hệ thống cần xác định loại khái niệm: triệu chứng, kết quả xét nghiệm, bệnh, thuốc, thông tin bệnh nhân.

## Điểm mấu chốt

Trong output chính thức, bảng nhãn thực tế là:

```text
TRIỆU_CHỨNG
TÊN_XÉT_NGHIỆM
KẾT_QUẢ_XÉT_NGHIỆM
CHẨN_ĐOÁN
THUỐC
```

Có một điểm cần chú ý: phần mô tả tổng quan có nhắc “thông tin bệnh nhân”, nhưng bảng nhãn output **không có nhãn THÔNG_TIN_BỆNH_NHÂN**. Vì vậy Phase 1 có khả năng **không yêu cầu extract tên, tuổi, địa chỉ, SĐT**.

Đừng tự thêm type ngoài schema. Nếu thêm:

```json
"type": "THÔNG_TIN_BỆNH_NHÂN"
```

thì có nguy cơ sai schema.

## Quy tắc triển khai

Chỉ dùng đúng 5 type:

[
T =
{
\text{TRIỆU_CHỨNG},
\text{TÊN_XÉT_NGHIỆM},
\text{KẾT_QUẢ_XÉT_NGHIỆM},
\text{CHẨN_ĐOÁN},
\text{THUỐC}
}
]

---

# 4. Phân tích câu “ánh xạ bệnh với ICD-10 và thuốc với RxNorm”

## Câu đề

> Ánh xạ bệnh với chuẩn ICD-10 và thuốc với chuẩn RxNorm.

## Ý nghĩa

Chỉ có 2 loại cần `candidates`:

| Type        | Candidate system |
| ----------- | ---------------- |
| `CHẨN_ĐOÁN` | ICD-10           |
| `THUỐC`     | RxNorm           |
| Loại khác   | `[]`             |

## Điểm mấu chốt

Đây không phải classification đơn giản. Đây là bài:

[
m = \text{mention}
]

[
C = \text{ICD/RxNorm candidate set}
]

[
c^* = \arg\max_{c \in C} s(m, c, context)
]

Ví dụ:

```text
"trào ngược dạ dày - thực quản"
```

có thể map:

```json
"candidates": ["K21.0", "K21.9"]
```

Vì đề cho phép **list candidates**, không nhất thiết chỉ 1 mã. Đây là điểm quan trọng. Nếu không chắc mã chính xác, có thể trả về top-k hợp lý, nhưng phải kiểm tra metric có phạt nhiều candidate sai không.

## Rủi ro lớn

Sai code system là lỗi nặng:

```json
{
  "text": "metformin",
  "type": "THUỐC",
  "candidates": ["E11"]
}
```

Sai vì `E11` là ICD, không phải RxNorm.

Cần validator:

```python
if entity.type == "THUỐC":
    assert all(code in rxnorm_codes for code in entity.candidates)

if entity.type == "CHẨN_ĐOÁN":
    assert all(code in icd10_codes for code in entity.candidates)
```

Tài liệu chiến thuật của bạn cũng nhấn mạnh normalization nên ưu tiên hơn model lớn: exact/alias/abbreviation/BM25/char n-gram có ROI cao hơn fine-tune nặng khi làm cá nhân ít GPU. 

---

# 5. Phân tích câu “suy luận mối liên hệ ngữ cảnh”

## Câu đề

> Đồng thời suy luận mối liên hệ ngữ cảnh: phủ định, người nhà, tiền sử.

## Ý nghĩa

Đây chính là trường:

```json
"assertions": []
```

Có 3 assertion:

```text
isNegated
isFamily
isHistorical
```

## Cực kỳ quan trọng

Entity bị phủ định **vẫn phải extract**.

Ví dụ:

```text
Không ho.
```

Không được bỏ `"ho"`. Đúng là:

```json
{
  "text": "ho",
  "type": "TRIỆU_CHỨNG",
  "assertions": ["isNegated"],
  "candidates": [],
  "position": [...]
}
```

## Các case dễ sai

| Câu                              | Entity       | Assertion đúng                                    |
| -------------------------------- | ------------ | ------------------------------------------------- |
| `không ho`                       | ho           | `isNegated`                                       |
| `chưa ghi nhận viêm phổi`        | viêm phổi    | `isNegated`                                       |
| `tiền sử hen phế quản`           | hen phế quản | `isHistorical`                                    |
| `bố bệnh nhân đau bụng tương tự` | đau bụng     | `isFamily`                                        |
| `mẹ bị ung thư vú`               | ung thư vú   | `isFamily`                                        |
| `không có tiền sử hen phế quản`  | hen phế quản | `isNegated`, `isHistorical` có thể cùng xuất hiện |

Đề nói một concept có thể có **0, 1, 2 hoặc cả 3 assertion**. Do đó hệ thống phải multi-label, không phải single-label.

[
a_i \in {0,1}^3
]

với:

[
a_i =
[
isNegated,
isFamily,
isHistorical
]
]

---

# 6. Phân tích câu “quan hệ giữa các khái niệm”

## Câu đề

> ... cũng như quan hệ giữa các khái niệm.

## Mâu thuẫn / điểm cần cảnh giác

Mô tả nói có “quan hệ giữa các khái niệm”, nhưng schema output lại không có field kiểu:

```json
"relations": []
```

Output chỉ có:

```json
text
position
type
assertions
candidates
```

Vì vậy có 2 khả năng:

### Khả năng A — Phase 1 không chấm relation explicit

Nếu leaderboard chỉ chấm theo schema mục 3.2, thì relation chỉ nên dùng **nội bộ** để hỗ trợ:

| Relation nội bộ     | Dùng để                           |
| ------------------- | --------------------------------- |
| Drug treats disease | tăng confidence RxNorm/ICD        |
| Lab test has value  | ghép tên xét nghiệm với kết quả   |
| Disease has symptom | hỗ trợ type/assertion             |
| Section context     | hỗ trợ `isHistorical`, `isFamily` |

Không nên output thêm field ngoài schema.

### Khả năng B — Đề thiếu schema relation

Nếu sau này BTC cập nhật schema, mới thêm module relation.

## Chiến lược an toàn

Phase 1 nên implement relation như **internal reasoning layer**, không đưa vào output trừ khi format chính thức yêu cầu.

---

# 7. Phân tích phần input

## Câu đề

> Input là một đoạn văn bản y khoa dạng tự do.

## Điểm mấu chốt

Mỗi file `.txt` là một raw string. Việc quan trọng nhất là **không phá offset**.

Sai lầm thường gặp:

| Preprocessing                            | Hậu quả      |
| ---------------------------------------- | ------------ |
| lower-case rồi tính offset trên text mới | sai position |
| remove dấu                               | sai position |
| chuẩn hóa khoảng trắng                   | sai position |
| xóa newline                              | sai position |
| replace ký tự Unicode                    | sai position |

## Quy tắc bắt buộc

Bạn có thể tạo `normalized_text` để match dictionary, nhưng output phải tính position trên `original_text`.

Cần invariant:

```python
assert original_text[start:end] == entity.text
```

---

# 8. Phân tích field `position`

## Câu đề

> position: List 2 phần tử [start, end] — vị trí ký tự bắt đầu/kết thúc của cụm trong input.

## Điểm mấu chốt

Đề nói hơi mơ hồ: “start/end” và “index từ 0 đến n-1”. Nhưng ví dụ `"ho đờm xanh"` có `position: [42, 53]`, nếu tính theo Python thì:

```python
text[42:53] == "ho đờm xanh"
```

Tức là rất có khả năng format là:

[
[start, end)
]

với `end` là vị trí ngay sau ký tự cuối.

## Nhưng có vấn đề

Trong ví dụ dài, một số position được đưa ra có vẻ **không khớp tuyệt đối** nếu tính trực tiếp trên chuỗi đã paste. Vì vậy không nên hardcode theo ví dụ. Cách đúng là:

```python
start = original_text.index(entity_text)
end = start + len(entity_text)
assert original_text[start:end] == entity_text
```

Nếu có nhiều mention giống nhau, phải chọn đúng occurrence theo context.

## Validator bắt buộc

```python
def validate_offset(text: str, item: dict) -> None:
    start, end = item["position"]
    assert 0 <= start < end <= len(text)
    assert text[start:end] == item["text"]
```

Đây là một trong các lỗi “tự sát” nếu không kiểm.

---

# 9. Phân tích field `text`

## Câu đề

> text: Cụm từ trong input mà hệ thống xác định là khái niệm y tế.

## Điểm mấu chốt

`text` phải là exact substring từ input, không phải dạng chuẩn hóa.

Sai:

```json
{
  "text": "trào ngược dạ dày thực quản"
}
```

nếu input là:

```text
bệnh trào ngược dạ dày - thực quản
```

Đúng:

```json
{
  "text": "bệnh trào ngược dạ dày - thực quản"
}
```

Hoặc tùy gold annotation, có thể là:

```json
{
  "text": "trào ngược dạ dày - thực quản"
}
```

## Boundary là điểm rất quan trọng

Ví dụ:

```text
đái tháo đường type 2 không kiểm soát
```

Có thể có các boundary:

| Span                                    | Chất lượng                       |
| --------------------------------------- | -------------------------------- |
| `đái tháo đường`                        | thiếu thông tin                  |
| `đái tháo đường type 2`                 | tốt                              |
| `đái tháo đường type 2 không kiểm soát` | có thể đúng hơn nếu ICD chi tiết |

Boundary sai kéo theo code sai.

---

# 10. Phân tích field `type`

## Câu đề

> type: Loại khái niệm y tế.

## Điểm mấu chốt

Không chỉ cần detect entity, còn phải phân loại đúng.

Ví dụ:

```text
WBC: 14,43
```

Phải tách:

```json
{
  "text": "WBC",
  "type": "TÊN_XÉT_NGHIỆM"
}
```

```json
{
  "text": "14,43",
  "type": "KẾT_QUẢ_XÉT_NGHIỆM"
}
```

Không được gộp thành:

```json
{
  "text": "WBC:14,43",
  "type": "KẾT_QUẢ_XÉT_NGHIỆM"
}
```

trừ khi gold annotation làm vậy. Nhưng theo ví dụ, đề muốn tách test name và result.

## Các pattern quan trọng

| Pattern                                         | Type                 |
| ----------------------------------------------- | -------------------- |
| `ho`, `sốt`, `đau thượng vị`                    | `TRIỆU_CHỨNG`        |
| `WBC`, `NEUT%`, `HbA1c`, `CRP`                  | `TÊN_XÉT_NGHIỆM`     |
| `14,43`, `76,4%`, `8.5 mmol/L`                  | `KẾT_QUẢ_XÉT_NGHIỆM` |
| `viêm phổi`, `GERD`, `tăng huyết áp`            | `CHẨN_ĐOÁN`          |
| `Metformin 500mg`, `Chlorpheniramine 0.4 MG/ML` | `THUỐC`              |

---

# 11. Phân tích field `assertions`

## Câu đề

> assertions: List các chuỗi thể hiện mối liên hệ ngữ cảnh, tối đa 3 phần tử, áp dụng cho CHẨN_ĐOÁN, THUỐC, TRIỆU_CHỨNG.

## Điểm mấu chốt

Với `TÊN_XÉT_NGHIỆM` và `KẾT_QUẢ_XÉT_NGHIỆM`, có thể để:

```json
"assertions": []
```

Mặc dù ví dụ có object không chứa `assertions`, cách an toàn hơn là **luôn output đủ field**:

```json
{
  "text": "WBC",
  "type": "TÊN_XÉT_NGHIỆM",
  "assertions": [],
  "candidates": [],
  "position": [282, 285]
}
```

Lý do: schema ổn định hơn, validator dễ hơn, tránh lỗi field missing nếu hệ chấm nghiêm.

## Rule engine nên có

```text
isNegated:
  không, chưa, không ghi nhận, không phát hiện, không thấy, âm tính

isHistorical:
  tiền sử, đã từng, trước đây, bệnh cũ, sau điều trị

isFamily:
  bố, mẹ, cha, anh, chị, em, con, gia đình, họ hàng
```

Các tài liệu trước của bạn cũng nhấn mạnh negation/family/history là nhóm lỗi clinical nghiêm trọng và cần đo riêng. 

---

# 12. Phân tích field `candidates`

## Câu đề

> candidates: List mã chuẩn y tế dự đoán, chỉ áp dụng cho CHẨN_ĐOÁN và THUỐC.

## Điểm mấu chốt

Đây là phần **normalization/linking**.

Nên thiết kế theo pipeline:

```text
mention
→ normalize mention
→ exact match
→ alias dictionary
→ abbreviation dictionary
→ fuzzy match
→ BM25 / char n-gram
→ optional dense retrieval
→ rerank
→ ontology validator
→ candidates
```

## Vì sao candidate generation quan trọng?

Nếu gold code không nằm trong top candidates, reranker không thể cứu.

Metric nội bộ cần đo:

[
Recall@k = \frac{#{\text{gold code nằm trong top-k}}}{#{\text{mentions cần code}}}
]

Mục tiêu thực dụng:

[
\boxed{Recall@20 > 95%}
]

Tài liệu loop engineering của bạn cũng đặt `candidate_recall@20`, `linking_accuracy@1`, `wrong_code_rate`, `wrong_code_system_rate` là metric trọng yếu cho normalization. 

---

# 13. Phân tích phần “không cung cấp train”

## Câu đề

> Lưu ý quan trọng: Đề bài không cung cấp tập train. Thí sinh cần sử dụng các giải pháp ngoài lời giải chính để tạo thêm dữ liệu phục vụ huấn luyện mô hình.

## Ý nghĩa thật

Đây là tín hiệu cực lớn: BTC không kỳ vọng thí sinh chỉ supervised learning từ train set. Họ chấp nhận hoặc khuyến khích:

| Nguồn                        | Dùng để                           |
| ---------------------------- | --------------------------------- |
| Synthetic data bằng LLM      | train NER/context classifier      |
| Public medical datasets      | học format clinical IE            |
| Pretrained biomedical model  | feature extractor                 |
| Annotation thủ công seed nhỏ | validation local                  |
| Corpus y khoa tiếng Việt     | synonym, abbreviation, dictionary |

## Nhưng với 100 test files

Bạn có thể thắng bằng engineering nhiều hơn training:

[
\boxed{
\text{Dictionary}
+
\text{Rule}
+
\text{Retriever}
+
\text{LLM-assisted weak labeling}
+
\text{Validator}
}
]

Không nhất thiết phải fine-tune model lớn ngay.

---

# 14. Các điểm mâu thuẫn / ambiguity trong đề

## 14.1. Có “thông tin bệnh nhân” nhưng không có label output

Mô tả nói có thông tin bệnh nhân, nhưng bảng label không có. Kết luận: **không output thông tin bệnh nhân trừ khi BTC cập nhật schema**.

## 14.2. Có “quan hệ giữa các khái niệm” nhưng không có field `relations`

Kết luận: dùng relation làm internal reasoning, chưa output relation.

## 14.3. Ví dụ JSON không nhất quán field

Một số object thiếu `assertions` hoặc `candidates`. Nhưng schema mô tả có đủ field. Kết luận: **luôn output đủ field**.

Chuẩn an toàn:

```json
{
  "text": "...",
  "position": [0, 10],
  "type": "TRIỆU_CHỨNG",
  "assertions": [],
  "candidates": []
}
```

## 14.4. Position trong ví dụ có thể không đáng tin tuyệt đối

Kết luận: tự tính offset từ raw input và validate bằng slicing.

---

# 15. Mô hình hóa bài toán chính xác

Với mỗi input text (x), cần sinh danh sách entity:

[
Y =
[
e_1, e_2, ..., e_n
]
]

Mỗi entity:

[
e_i =
(
t_i,
s_i,
r_i,
y_i,
a_i,
C_i
)
]

Trong đó:

| Ký hiệu                  | Nghĩa                 |
| ------------------------ | --------------------- |
| (t_i)                    | text span             |
| (s_i = [start_i, end_i)) | character offset      |
| (r_i)                    | entity type           |
| (a_i)                    | assertion set         |
| (C_i)                    | ICD/RxNorm candidates |

Ràng buộc:

[
x[start_i:end_i] = t_i
]

[
r_i \in {
TRIỆU_CHỨNG,
TÊN_XÉT_NGHIỆM,
KẾT_QUẢ_XÉT_NGHIỆM,
CHẨN_ĐOÁN,
THUỐC
}
]

[
a_i \subseteq
{
isNegated,
isFamily,
isHistorical
}
]

[
C_i =
\begin{cases}
ICD10Codes, & r_i = CHẨN_ĐOÁN \
RxNormCodes, & r_i = THUỐC \
\emptyset, & otherwise
\end{cases}
]

---

# 16. Điểm mấu chốt theo thứ tự ưu tiên

## Priority 0 — Output không được chết schema

Phải có:

```text
validate_json.py
validate_schema.py
validate_offset.py
validate_code.py
validate_type.py
```

Đây là nền. Sai schema là mất điểm ngay.

## Priority 1 — Offset và boundary

Validator bắt buộc:

```python
assert text[start:end] == item["text"]
```

Sai offset là lỗi rất độc vì dù entity đúng, scorer có thể tính sai toàn bộ.

## Priority 2 — Dictionary + normalization

Đây là phần nên đầu tư mạnh nhất:

```text
ICD-10 dictionary
RxNorm dictionary
Vietnamese synonym table
abbreviation table
no-diacritic normalization
typo fuzzy matching
BM25 / char n-gram retriever
```

## Priority 3 — Assertion rules

Các rule `isNegated`, `isFamily`, `isHistorical` có ROI rất cao.

Ví dụ:

```text
Không ghi nhận viêm phổi        → isNegated
Không loại trừ viêm phổi        → không nên isNegated, có thể uncertain nhưng schema không có uncertain
Tiền sử hen phế quản            → isHistorical
Mẹ bệnh nhân bị ung thư vú      → isFamily
```

Vì schema không có `isUncertain`, các case “nghi”, “theo dõi”, “không loại trừ” cần xử lý cẩn thận: không gán `isNegated`.

## Priority 4 — Lab parser

Kết quả xét nghiệm thường có pattern rõ:

```text
WBC:14,43
NEUT%:76,4
HbA1c 8.5%
CRP 12 mg/L
```

Nên viết parser riêng:

[
\text{LAB_NAME} [:\s] \text{LAB_VALUE}
]

## Priority 5 — Drug parser

Thuốc thường có:

```text
drug name + dose + unit + route/frequency
```

Ví dụ:

```text
Chlorpheniramine 0.4 MG/ML
Capsaicin 0.38 MG/ML
Metformin 500mg
```

Nên extract full drug mention gồm dose nếu input có.

---

# 17. Checklist đọc đề để triển khai

## Output contract

```text
[ ] Mỗi input .txt sinh đúng 1 output .json
[ ] JSON là list
[ ] Mỗi item là dict
[ ] Có text
[ ] Có position
[ ] Có type
[ ] Có assertions
[ ] Có candidates
[ ] position là [start, end)
[ ] text[start:end] == item["text"]
[ ] type đúng 5 nhãn tiếng Việt
[ ] assertions chỉ gồm isNegated/isFamily/isHistorical
[ ] candidates ICD chỉ cho CHẨN_ĐOÁN
[ ] candidates RxNorm chỉ cho THUỐC
[ ] các type khác candidates = []
```

## System modules

```text
[ ] raw text reader
[ ] offset-preserving preprocessor
[ ] dictionary matcher
[ ] NER/span extractor
[ ] lab parser
[ ] drug parser
[ ] assertion detector
[ ] ICD linker
[ ] RxNorm linker
[ ] ontology/code validator
[ ] JSON exporter
[ ] submission zip builder
```

## Local evaluation dù không có train

Tạo seed validation thủ công khoảng 30–50 đoạn:

```text
[ ] 10 đoạn triệu chứng/chẩn đoán
[ ] 10 đoạn thuốc
[ ] 10 đoạn xét nghiệm
[ ] 10 đoạn negation/history/family
[ ] 10 đoạn typo/no-diacritic/abbreviation
```

---

# 18. Câu chốt chiến thuật

Đề này nên được hiểu như sau:

[
\boxed{
\text{Đọc bệnh án}
\rightarrow
\text{tìm đúng span}
\rightarrow
\text{gán đúng type}
\rightarrow
\text{hiểu context}
\rightarrow
\text{map disease/drug về code chuẩn}
\rightarrow
\text{xuất JSON sạch}
}
]

Không nên over-engineer ngay vào “ontology reasoning” phức tạp. Với Phase 1, ontology nên dùng như **bộ lọc và bộ tăng điểm cuối pipeline**:

```text
THUỐC không được map ICD
CHẨN_ĐOÁN không được map RxNorm
Family disease không phải bệnh hiện tại của bệnh nhân
Negated symptom vẫn là entity nhưng có isNegated
Lab value phải gần lab name
Drug dose phải đi cùng drug mention
```

Công thức thắng thực dụng:

[
\boxed{
\text{High-recall span extraction}
+
\text{Hybrid ICD/RxNorm normalization}
+
\text{Rule-based assertion}
+
\text{Offset/schema validator}
+
\text{Ontology constraint}
}
]

Với cá nhân ít hardware, đây là bài **distribution-aware engineering**, không phải brute-force GPU.
