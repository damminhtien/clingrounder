Dưới đây là phân tích **từng ý trong đề bài Vòng 1**, theo hướng đọc để triển khai hệ thống thi thật.

# 0. Bản chất bài toán

Đề này không phải chatbot y tế và cũng không chỉ là NER. Bài toán đúng hơn là:

[
\boxed{
\text{Clinical Information Extraction}
+
\text{Entity Normalization}
+
\text{Assertion Detection}
+
\text{JSON/Offset Validation}
}
]

Với schema Phase 1 hiện tại, trọng tâm thực chiến là:

[
\boxed{
\text{text đúng}
+
\text{type đúng}
+
\text{assertions đúng}
+
\text{candidates đúng}
+
\text{position đúng}
}
]

Tài liệu phân tích trước cũng đã chốt đúng hướng: với Phase 1, cần ưu tiên span, type, assertion, ICD/RxNorm candidate và JSON/offset sạch hơn là xây một hệ thống ontology quá phức tạp ngay từ đầu. 

---

# 1. “Vòng 1 - Sơ loại: 02/07/2026 - 30/07/2026”

Ý này nói về **giai đoạn thi offline**.

Bạn không cần deploy API ở vòng này. Bạn nhận dữ liệu test, chạy inference cục bộ, rồi nộp file ZIP chứa JSON dự đoán. Đây là **offline batch inference**, chưa phải bài serving hay real-time system. 

Hàm ý kỹ thuật:

| Ý trong đề  | Cần hiểu                                                     |
| ----------- | ------------------------------------------------------------ |
| Vòng 1      | Chỉ cần sinh output đúng format                              |
| Tệp ZIP     | Cần script export và zip chuẩn                               |
| GPU         | Có thể dùng model, nhưng không bắt buộc                      |
| 100 bản ghi | Quy mô nhỏ, rất hợp với hybrid rule + dictionary + validator |
| 5 lần/ngày  | Phải có local validation, không được nộp mò                  |

Chiến lược đúng:

[
\boxed{
\text{làm hệ thống chắc, đúng schema, đúng offset trước}
}
]

Không nên bắt đầu bằng train model lớn.

---

# 2. “Nộp kết quả dự đoán dưới dạng file JSON”

Đây là **output contract**. Hệ thống của bạn phải biến mỗi input text thành một danh sách entity JSON.

Ví dụ một entity:

```json
{
  "text": "amlodipine 10 mg po daily",
  "type": "THUỐC",
  "candidates": ["308135"],
  "assertions": ["isHistorical"],
  "position": [58, 83]
}
```

Mỗi item trong JSON là một **khái niệm y tế được phát hiện**.

Tức là bài yêu cầu:

[
x = \text{raw clinical text}
]

[
Y = [e_1, e_2, ..., e_n]
]

với mỗi entity:

[
e_i =
(
text_i,
type_i,
candidates_i,
assertions_i,
position_i
)
]

Điểm chết người: nếu JSON sai format, hệ thống có thể bị chấm lỗi ngay dù model tốt.

---

# 3. “output.zip có cấu trúc output/1.json ... 100.json”

Cấu trúc bắt buộc:

```text
output/
    ├── 1.json
    ├── 2.json
    ├── ...
    └── 100.json
```

Ý nghĩa kỹ thuật:

| Thành phần   | Cần làm                                    |
| ------------ | ------------------------------------------ |
| `output/`    | Folder gốc sau khi giải nén                |
| `1.json`     | Dự đoán cho bản ghi 1                      |
| `2.json`     | Dự đoán cho bản ghi 2                      |
| `100.json`   | Dự đoán cho bản ghi 100                    |
| `output.zip` | Zip đúng folder, không zip sai cấp thư mục |

Nhiều đội chết vì zip sai cấp, ví dụ:

```text
wrong_output.zip/
    output/
        output/
            1.json
```

hoặc:

```text
wrong_output.zip/
    1.json
    2.json
```

Bạn cần script validate:

```python
assert zip_contains("output/")
assert files == ["1.json", "2.json", ..., "100.json"]
```

Trong tài liệu trước, validator được xem là bắt buộc vì schema/JSON/offset sai sẽ làm hệ thống mất điểm dù prediction tốt. 

---

# 4. “Top ~15 đội phải gửi source code”

Ý này là chống hard-code output.

BTC sẽ yêu cầu các đội top gửi:

```text
source code
data processing
training code
inference code
data sử dụng
model weights
README cài đặt
```

Hàm ý cực quan trọng: pipeline của bạn phải **reproducible**.

Không được làm kiểu notebook rời rạc, chạy tay, sửa file thủ công. Cần cấu trúc repo tối thiểu:

```text
air-med/
  src/
    preprocessing/
    entity/
    linking/
    context/
    submission/
  data/
    dictionaries/
    raw/
  models/
  configs/
  scripts/
    run_inference.py
    build_submission.py
    validate_submission.py
  README.md
  requirements.txt
```

README phải có lệnh kiểu:

```bash
pip install -r requirements.txt
python scripts/run_inference.py --input data/test --output output
python scripts/build_submission.py --output output --zip output.zip
python scripts/validate_submission.py --input data/test --output output
```

Nếu BTC không dựng lại được code, bạn có nguy cơ bị loại.

---

# 5. Input là gì?

Ví dụ input:

```text
'Danh sách thuốc trước nhập viện chính xác và đầy đủ. 
1. amlodipine 10 mg po daily 
2. aspirin 81 mg po daily 
...
11. clonazepam 1.5 mg po qhs điều trị lo âu mất ngủ'
```

Đây là **văn bản y khoa tự do**.

Đặc điểm input:

| Đặc điểm                  | Ví dụ                        | Vấn đề                           |
| ------------------------- | ---------------------------- | -------------------------------- |
| Có thuốc                  | `amlodipine 10 mg po daily`  | Cần extract full medication span |
| Có liều                   | `10 mg`, `81 mg`             | Ảnh hưởng RxNorm candidate       |
| Có đường dùng             | `po`                         | Không nhất thiết vào RxNorm code |
| Có tần suất               | `daily`, `q6h`, `qid`, `qhs` | Thuộc medication instruction     |
| Có triệu chứng/lý do dùng | `ho`, `đau nhức`, `táo bón`  | Type là `TRIỆU_CHỨNG`            |
| Có ngữ cảnh               | `trước nhập viện`            | Assertion `isHistorical`         |

Nhiệm vụ không phải dịch câu. Nhiệm vụ là bóc ra các span có ý nghĩa y tế.

---

# 6. Output là danh sách entity, không phải object tổng

Output mẫu là:

```json
[
  {
    "text": "amlodipine 10 mg po daily",
    "type": "THUỐC",
    "candidates": ["308135"],
    "assertions": ["isHistorical"],
    "position": [58, 83]
  },
  {
    "text": "ho",
    "type": "TRIỆU_CHỨNG",
    "assertions": [],
    "position": [196, 198]
  }
]
```

Nên hiểu output là:

[
\text{List[Entity]}
]

Không phải:

```json
{
  "entities": [...]
}
```

Không phải:

```json
{
  "result": [...]
}
```

Không phải:

```json
{
  "relations": [...]
}
```

Phase 1 hiện tại là **flat entity list**, không có field relation explicit. Tài liệu Vòng 1 cũng chỉ ra nếu schema không có relation thì không nên output relation; relation chỉ nên dùng nội bộ để hỗ trợ suy luận. 

---

# 7. Field `text`

`text` là exact span trong input.

Ví dụ:

```json
"text": "amlodipine 10 mg po daily"
```

Không được normalize text khi output.

Sai:

```json
"text": "Amlodipine 10mg oral once daily"
```

Đúng:

```json
"text": "amlodipine 10 mg po daily"
```

Vì metric `text_score` dùng WER, bạn có thể được partial credit nếu gần đúng, nhưng tốt nhất vẫn là:

[
\boxed{text = original_text[start:end]}
]

Tài liệu trước nhấn mạnh không normalize text trong output và luôn tính trên raw input file để tránh sai offset. 

---

# 8. Field `type`

Các type xuất hiện trong đề/ví dụ gồm:

```text
TRIỆU_CHỨNG
TÊN_XÉT_NGHIỆM
KẾT_QUẢ_XÉT_NGHIỆM
CHẨN_ĐOÁN
THUỐC
```

Không tự thêm type khác.

Đặc biệt, phần mô tả tổng quan có thể nói “thông tin bệnh nhân”, nhưng schema output Phase 1 không có nhãn `THÔNG_TIN_BỆNH_NHÂN`. Vì vậy không nên output type này nếu BTC không cập nhật schema. 

Sai type bị phạt rất nặng. Nếu text đúng nhưng type sai, đề nói entity sẽ bị tính 2 lần: một false positive và một false negative, và các metric đều 0 cho phần đó. 

Ví dụ:

```text
Gold: "ho" - TRIỆU_CHỨNG
Pred: "ho" - CHẨN_ĐOÁN
```

Đây không phải lỗi nhỏ. Đây là lỗi nặng.

---

# 9. Field `candidates`

`candidates` là danh sách mã chuẩn.

Chỉ nên dùng cho:

| Type                 | Candidate      |
| -------------------- | -------------- |
| `THUỐC`              | RxNorm         |
| `CHẨN_ĐOÁN`          | ICD-10         |
| `TRIỆU_CHỨNG`        | thường để `[]` |
| `TÊN_XÉT_NGHIỆM`     | thường để `[]` |
| `KẾT_QUẢ_XÉT_NGHIỆM` | thường để `[]` |

Ví dụ thuốc:

```json
{
  "text": "aspirin 81 mg po daily",
  "type": "THUỐC",
  "candidates": ["243670"]
}
```

Ví dụ triệu chứng:

```json
{
  "text": "ho",
  "type": "TRIỆU_CHỨNG",
  "candidates": []
}
```

Điểm cực quan trọng: metric candidates dùng Jaccard. Nếu gold là:

```json
["308135"]
```

Bạn dự đoán:

```json
["308135"]
```

thì:

[
J = 1
]

Nhưng nếu bạn dự đoán:

```json
["308135", "123", "456"]
```

thì:

[
J = \frac{1}{3}
]

Nên không được spam nhiều candidate. Candidate nên là **small set, high precision**. Tài liệu Vòng 1 cũng nhấn mạnh không nên trả top-10 candidate bừa vì Jaccard sẽ phạt phần union lớn. 

---

# 10. Field `assertions`

`assertions` là danh sách các nhãn ngữ cảnh.

Các assertion chính:

```text
isNegated
isFamily
isHistorical
```

Ý nghĩa:

| Assertion      | Nghĩa                           | Ví dụ                                           |
| -------------- | ------------------------------- | ----------------------------------------------- |
| `isNegated`    | Phủ định                        | `không ho`, `không ghi nhận viêm phổi`          |
| `isFamily`     | Người nhà, không phải bệnh nhân | `mẹ bị ung thư vú`                              |
| `isHistorical` | Tiền sử/quá khứ                 | `tiền sử hen phế quản`, `thuốc trước nhập viện` |

Một entity có thể có nhiều assertion cùng lúc:

```json
"assertions": ["isNegated", "isHistorical"]
```

Ví dụ:

```text
Không có tiền sử hen phế quản.
```

Có thể hiểu là:

```json
{
  "text": "hen phế quản",
  "type": "CHẨN_ĐOÁN",
  "assertions": ["isNegated", "isHistorical"]
}
```

Trong ví dụ đề bài, câu mở đầu là:

```text
Danh sách thuốc trước nhập viện...
```

nên tất cả thuốc trong danh sách được gắn:

```json
"assertions": ["isHistorical"]
```

Tài liệu trước cũng nhấn mạnh assertion phải là multi-label, không phải single-label. 

---

# 11. Field `position`

`position` là vị trí ký tự của span trong input:

```json
"position": [58, 83]
```

Nên hiểu theo quy ước Python:

[
[start, end)
]

tức là:

```python
text[start:end] == entity["text"]
```

Validator bắt buộc:

```python
def validate_offset(raw_text: str, item: dict) -> None:
    start, end = item["position"]
    assert 0 <= start < end <= len(raw_text)
    assert raw_text[start:end] == item["text"]
```

Không được tính position sau khi đã lowercase, remove dấu, normalize space hoặc xóa newline.

Sai lầm phổ biến:

```python
clean_text = normalize(raw_text)
start = clean_text.find(entity)
```

Sai vì offset này không còn khớp raw input.

Cách đúng:

```python
normalized_text = normalize(raw_text)  # chỉ dùng để match
# nhưng position phải map ngược về raw_text
```

Offset sai là lỗi rất độc vì dù entity đúng, scorer có thể không match được. 

---

# 12. Metric `text_score`

Đề dùng Word Error Rate trên trường `text`.

[
text_score =
\frac{
\sum_{i \in test} (1 - WER(i))
}{
len(test)
}
]

Ý nghĩa:

| Dự đoán text              | Hậu quả                    |
| ------------------------- | -------------------------- |
| Exact span đúng           | tốt nhất                   |
| Gần đúng                  | có thể được partial credit |
| Boundary thiếu/thừa nhiều | WER giảm                   |
| Sai type dù text đúng     | vẫn bị phạt nặng           |

Ví dụ:

Gold:

```text
acetaminophen 325-650 mg po q6h:prn
```

Pred:

```text
acetaminophen
```

Có thể vẫn có một phần overlap, nhưng mất thông tin dose/frequency, đồng thời candidate RxNorm có thể sai.

Vì vậy với thuốc, nên extract full medication phrase.

---

# 13. Metric `assertions_score`

Đề dùng Jaccard similarity:

[
J(A,B) = \frac{|A \cap B|}{|A \cup B|}
]

Trường hợp:

| Gold                            | Pred               | Jaccard |
| ------------------------------- | ------------------ | ------: |
| `[]`                            | `[]`               |       1 |
| `["isHistorical"]`              | `[]`               |       0 |
| `[]`                            | `["isHistorical"]` |       0 |
| `["isNegated", "isHistorical"]` | `["isNegated"]`    |     1/2 |
| `["isFamily"]`                  | `["isFamily"]`     |       1 |

Hàm ý chiến thuật:

[
\boxed{
\text{Đừng gắn assertion nếu không chắc}
}
]

Đặc biệt các câu:

```text
không loại trừ viêm phổi
nghi viêm phổi
theo dõi viêm phổi
```

không nên gắn `isNegated`, vì đây là uncertainty, trong schema hiện tại không có `isUncertain`. Tài liệu phân tích trước cũng nhấn mạnh case “không loại trừ” không phải negated. 

---

# 14. Metric `candidates_score`

Đây là phần có trọng số lớn nhất:

[
final_score =
0.3 \cdot text
+
0.3 \cdot assertions
+
0.4 \cdot candidates
]

Nhưng candidates chỉ có ý nghĩa nếu entity/type đã đúng.

Candidate score cũng dùng Jaccard. Do đó:

| Chiến lược                                  | Rủi ro                        |
| ------------------------------------------- | ----------------------------- |
| Output đúng 1 candidate chắc                | tốt                           |
| Output nhiều candidate để “cover”           | bị phạt union lớn             |
| Output sai code system                      | rất xấu                       |
| Không output candidate cho thuốc/chẩn đoán  | mất nhiều điểm                |
| Output candidate cho triệu chứng/xét nghiệm | có thể bị phạt nếu gold empty |

Tài liệu Vòng 1 kết luận ưu tiên thực tế là `Candidates > Text ≈ Assertions`, nhưng phải hiểu candidate phụ thuộc vào span/type đúng trước đó. 

---

# 15. Công thức final score

[
final_score
===========

0.3 \cdot text_score
+
0.3 \cdot assertions_score
+
0.4 \cdot candidates_score
]

Hàm ý:

[
\boxed{
\text{ICD/RxNorm linking là phần tăng điểm lớn nhất}
}
]

Nhưng thứ tự triển khai không phải bắt đầu từ candidate. Thứ tự đúng là:

[
\boxed{
\text{schema/offset}
\rightarrow
\text{span + type}
\rightarrow
\text{candidate}
\rightarrow
\text{assertion}
}
]

Vì nếu span/type sai, candidate đúng cũng không cứu được.

---

# 16. Lưu ý “sai loại khái niệm bị tính 2 lần”

Đây là câu cực quan trọng.

Ví dụ:

Gold:

```json
{
  "text": "mất ngủ",
  "type": "TRIỆU_CHỨNG"
}
```

Pred:

```json
{
  "text": "mất ngủ",
  "type": "CHẨN_ĐOÁN"
}
```

Hệ chấm coi như:

```text
1 missing gold TRIỆU_CHỨNG
1 spurious pred CHẨN_ĐOÁN
```

Và mỗi lần đều 0 cho text/assertion/candidate.

Kết luận:

[
\boxed{
\text{type classifier phải conservative}
}
]

Với ambiguous case, nên dùng rule/type dictionary:

| Nếu mention nằm trong drug dictionary | `THUỐC` |
| Nếu mention là lab acronym/value pattern | `TÊN_XÉT_NGHIỆM` / `KẾT_QUẢ_XÉT_NGHIỆM` |
| Nếu mention là symptom lexicon | `TRIỆU_CHỨNG` |
| Nếu mention có ICD mapping bệnh | `CHẨN_ĐOÁN` |

---

# 17. “Hạ tầng chấm: GPU”

Điều này không có nghĩa bạn bắt buộc phải dùng GPU.

Phase 1 chỉ nộp file ZIP. GPU là phía hệ thống chấm hoặc điều kiện cho phép. Với 100 bản ghi, pipeline rule/dictionary/retrieval có thể chạy CPU. Tài liệu phân tích trước cũng nêu với 100 test files, hướng win không nhất thiết là train lớn mà là dictionary + rule + retriever + validator. 

GPU chỉ cần nếu bạn dùng:

```text
NER transformer
embedding model
cross-encoder reranker
LLM local
```

Nhưng với cá nhân ít hardware, ROI cao hơn nằm ở:

```text
RxNorm dictionary
ICD-10 dictionary
Vietnamese aliases
abbreviation table
char n-gram fuzzy matching
assertion rules
offset validator
```

---

# 18. “Giới hạn nộp bài: 5 lần/ngày”

Nghĩa là public leaderboard không thể là evaluation chính.

Bạn cần local validation tự tạo:

```text
seed_valid/
  symptoms.json
  drugs.json
  labs.json
  diagnosis.json
  assertions.json
```

Tạo 30–50 câu thủ công để test:

```text
Không ho.
Không ghi nhận viêm phổi.
Không loại trừ viêm phổi.
Tiền sử tăng huyết áp.
Mẹ bị ung thư vú.
Metformin 500mg uống ngày 2 lần.
WBC: 14,43.
HbA1c 8.5%.
```

Mỗi lần sửa pipeline phải chạy local validator trước khi nộp.

---

# 19. “Thời gian chờ: 600 giây”

Một lần submission mất khoảng 10 phút chờ.

Hàm ý:

| Nếu không có local scorer | Bạn mất thời gian |
| ------------------------- | ----------------- |
| Nộp thử format sai        | mất lượt          |
| Nộp thử offset sai        | mất lượt          |
| Nộp threshold mò          | dễ overfit public |

Cần có:

```text
validate_submission.py
validate_offsets.py
validate_schema.py
validate_candidates.py
diff_predictions.py
```

---

# 20. Ví dụ input-output nói gì về annotation style?

Từ ví dụ, ta rút ra nhiều rule vàng.

## 20.1. Thuốc được extract full phrase

Gold:

```text
amlodipine 10 mg po daily
aspirin 81 mg po daily
metoprolol succinate xl 50 mg po daily
guaifenesin ml po q6h:prn
```

Không chỉ extract:

```text
amlodipine
aspirin
metoprolol
guaifenesin
```

Vì candidate RxNorm phụ thuộc dose/form.

## 20.2. Triệu chứng sau “điều trị” cũng được extract

Ví dụ:

```text
guaifenesin ... điều trị ho
```

Output có:

```json
{
  "text": "ho",
  "type": "TRIỆU_CHỨNG"
}
```

Tức là symptom/reason vẫn là entity riêng.

## 20.3. Thuốc trước nhập viện là historical

Vì câu đầu:

```text
Danh sách thuốc trước nhập viện...
```

nên tất cả thuốc:

```json
"assertions": ["isHistorical"]
```

Đây là section-level assertion. Không chỉ nhìn quanh từng entity, phải nhìn header/ngữ cảnh toàn đoạn.

## 20.4. Triệu chứng không tự động historical

Trong ví dụ, thuốc có `isHistorical`, nhưng triệu chứng như `ho`, `đau nhức`, `táo bón`, `lo âu`, `mất ngủ` có assertions `[]`.

Tức là “thuốc trước nhập viện” không nhất thiết làm symptom/reason thành historical.

---

# 21. Các mâu thuẫn/ambiguity cần cảnh giác

## 21.1. Có nói “quan hệ giữa các khái niệm” nhưng output không có `relations`

Kết luận: hiện tại không output relation. Dùng relation nội bộ nếu cần.

Ví dụ nội bộ:

```text
clonazepam -> treats -> lo âu
clonazepam -> treats -> mất ngủ
```

Nhưng JSON final không có relation field.

## 21.2. Có nói “thông tin bệnh nhân” nhưng type không có

Kết luận: không output tuổi/giới/tên nếu schema không yêu cầu.

## 21.3. Ví dụ có thể thiếu field ở một số item

Cách an toàn: luôn output đủ 5 field:

```json
{
  "text": "...",
  "type": "...",
  "candidates": [],
  "assertions": [],
  "position": [0, 10]
}
```

Tài liệu phân tích đề cũng khuyến nghị luôn output đủ field để schema ổn định hơn và validator dễ hơn. 

---

# 22. Mô hình hóa chuẩn để code

Với mỗi input (x):

[
Y = [e_1, e_2, ..., e_n]
]

Mỗi entity:

[
e_i =
(
t_i,
s_i,
r_i,
a_i,
C_i
)
]

Trong đó:

| Ký hiệu                  | Nghĩa            |
| ------------------------ | ---------------- |
| (t_i)                    | text span        |
| (s_i = [start_i, end_i)) | character offset |
| (r_i)                    | type             |
| (a_i)                    | assertion set    |
| (C_i)                    | candidate set    |

Ràng buộc:

[
x[start_i:end_i] = t_i
]

[
r_i \in
{
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

Tài liệu phân tích bài toán cũng mô hình hóa entity theo text span, character offset, type, assertion set và ICD/RxNorm candidates với ràng buộc `x[start:end] = text`. 

---

# 23. Pipeline nên xây

```text
Raw input text
    ↓
Read original text, preserve offset
    ↓
Section/context detector
    ↓
Dictionary + regex span candidates
    ↓
NER/span extractor
    ↓
Span merge + type classifier
    ↓
Assertion rule engine
    ↓
Candidate linker
        - RxNorm for THUỐC
        - ICD-10 for CHẨN_ĐOÁN
    ↓
Ontology/code validator
    ↓
JSON schema validator
    ↓
output/1.json ... output/100.json
    ↓
output.zip
```

Trong đó module quan trọng nhất:

```text
linking/
  rxnorm_linker.py
  icd10_linker.py
  alias_dictionary.py
  abbreviation_dictionary.py
  char_ngram_matcher.py
  bm25_linker.py
```

Tài liệu Vòng 1 cũng đề xuất pipeline preserve offset → section detector → dictionary/regex → NER/LLM-assisted extractor → assertion rule → candidate linker → validator → output.zip. 

---

# 24. Thứ tự ưu tiên triển khai

## Priority 0 — Không chết format

```text
validate_json.py
validate_schema.py
validate_offset.py
validate_type.py
build_submission.py
```

## Priority 1 — Extract span + offset đúng

```text
drug parser
symptom dictionary
diagnosis dictionary
lab parser
offset mapper
```

## Priority 2 — Type đúng

```text
THUỐC vs CHẨN_ĐOÁN vs TRIỆU_CHỨNG
TÊN_XÉT_NGHIỆM vs KẾT_QUẢ_XÉT_NGHIỆM
```

## Priority 3 — Candidate linking

```text
RxNorm exact/alias/fuzzy
ICD-10 exact/alias/fuzzy
no-diacritic matching
abbreviation expansion
```

## Priority 4 — Assertion rules

```text
isHistorical
isNegated
isFamily
```

## Priority 5 — Tuning theo metric

```text
entity threshold
candidate threshold
assertion threshold
do-not-spam-candidates rule
```

---

# 25. Kết luận chiến thuật

Đề này nên được hiểu ngắn gọn là:

[
\boxed{
\text{Đọc text y khoa}
\rightarrow
\text{tìm span y tế}
\rightarrow
\text{gán type}
\rightarrow
\text{gán assertion}
\rightarrow
\text{map ICD/RxNorm nếu cần}
\rightarrow
\text{xuất JSON đúng offset}
}
]

Công thức thắng thực dụng:

[
\boxed{
\text{Exact Offset}
+
\text{High-Precision Type}
+
\text{Small Candidate Set}
+
\text{Strong ICD/RxNorm Dictionary}
+
\text{Context Rule Engine}
+
\text{Aggressive Validator}
}
]

Không bắt đầu bằng “model nào mạnh nhất”. Bắt đầu bằng câu hỏi:

```text
Output có đúng format không?
Offset có slicing được không?
Type có đúng enum không?
Thuốc/chẩn đoán có candidate đúng hệ không?
Assertion có bị spam không?
Candidate có bị spam không?
```

Với cá nhân ít hardware, đây là bài **engineering-aware + metric-aware**, không phải bài brute-force GPU.
