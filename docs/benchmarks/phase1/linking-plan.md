Build mạnh nhất theo kiểu **hybrid linker**, không phải chỉ một dictionary đơn giản.

Mục tiêu của module này:

[
m = \text{mention trong bệnh án}
]

[
C_m = {c_1, c_2, ..., c_k}
]

[
c^* = \arg\max_{c \in C_m} s(m, c, context)
]

Với Phase 1, không nên trả nhiều candidate bừa vì candidate score dùng Jaccard; candidate nên là **small set, high precision**. Phân tích Phase 1 trước đó cũng đã chốt candidates có trọng số cao nhất và không nên spam top-10 mã.

---

# 1. Kiến trúc tổng thể

Pipeline nên là:

```text
mention
  ↓
normalize mention
  ↓
exact match
  ↓
alias / synonym / abbreviation
  ↓
fuzzy match
  ↓
BM25 / char n-gram TF-IDF
  ↓
dense retrieval, optional
  ↓
rerank
  ↓
ontology/type validator
  ↓
small candidate set
```

Công thức score:

[
s(c)
====

\alpha s_{\text{exact}}
+
\beta s_{\text{alias}}
+
\gamma s_{\text{bm25}}
+
\delta s_{\text{char}}
+
\eta s_{\text{dense}}
+
\lambda s_{\text{context}}
+
\mu s_{\text{ontology}}
]

Đây đúng với hướng đã phân tích: ưu tiên normalization, dùng exact/alias/abbreviation/BM25/char n-gram trước, dense/reranker để tăng thêm nếu còn thời gian.

---

# 2. Nguồn dữ liệu gốc

## 2.1. RxNorm

Dùng **RxNorm Current Prescribable Content** trước, vì nó sạch hơn cho bài thuốc. NLM mô tả subset này gồm thuốc đang kê đơn được, nhiều OTC, chỉ chứa active RxNorm normalized names/codes/attributes/relationships, không có obsolete/suppressed data và không cần login UMLS. ([Thư viện Quốc gia về Y tế][1])

Các file chính:

```text
RXNCONSO.RRF  # concept names, RxCUI, term type
RXNREL.RRF    # relationships
RXNSAT.RRF    # attributes, NDC, UNII...
```

NLM cũng nói RxNorm files là pipe-delimited RRF, UTF-8, và current prescribable release ngày 01/06/2026 hiện có link tải trực tiếp; hôm nay 05/07/2026 nên release tháng 07 nhiều khả năng chưa ra vì monthly release theo first Monday. ([Thư viện Quốc gia về Y tế][2])

Trong `RXNCONSO`, ưu tiên các `TTY` sau:

| TTY         | Ý nghĩa                | Dùng thế nào                                     |
| ----------- | ---------------------- | ------------------------------------------------ |
| `IN`        | Ingredient             | map thuốc chỉ có tên hoạt chất                   |
| `PIN`       | Precise Ingredient     | hoạt chất cụ thể hơn                             |
| `MIN`       | Multiple Ingredients   | thuốc phối hợp                                   |
| `SCD`       | Semantic Clinical Drug | **rất quan trọng**: ingredient + strength + form |
| `SBD`       | Semantic Branded Drug  | branded drug                                     |
| `SCDF`      | Clinical Drug Form     | form không strength                              |
| `SBDF`      | Branded Drug Form      | branded form                                     |
| `GPCK/BPCK` | Pack                   | thuốc dạng gói/bộ                                |

Với đề này, nếu mention là:

```text
amlodipine 10 mg po daily
```

thì không nên map về ingredient `amlodipine` chung nếu có thể map ra **clinical drug** chứa `10 MG`. Phase 1 ví dụ đã cho thấy RxNorm có thể chấm theo drug + strength/form, không chỉ ingredient.

---

## 2.2. ICD-10 tiếng Việt

Schema tối thiểu cho ICD:

```text
code
title_vi
title_en
chapter
block
parent_code
is_leaf
aliases
normalized_aliases
```

Ví dụ:

```json
{
  "code": "E11",
  "title_vi": "Đái tháo đường không phụ thuộc insulin",
  "title_en": "Type 2 diabetes mellitus",
  "parent_code": "E10-E14",
  "aliases": [
    "đái tháo đường type 2",
    "tiểu đường type 2",
    "dtđ type 2",
    "dm2",
    "t2dm"
  ]
}
```

---

# 3. Data model nên dùng

Dùng chung một entity dictionary format cho cả ICD và RxNorm:

```python
from dataclasses import dataclass
from typing import Literal

CodeSystem = Literal["ICD10", "RXNORM"]

@dataclass(frozen=True)
class CodeEntry:
    code: str
    system: CodeSystem
    canonical_name: str
    normalized_name: str
    aliases: tuple[str, ...]
    normalized_aliases: tuple[str, ...]
    semantic_type: str
    tty: str | None = None          # RxNorm only
    parent_code: str | None = None  # ICD only
    priority: float = 0.0
```

Index cần build:

```text
exact_index: normalized_alias -> list[CodeEntry]
code_index: code -> CodeEntry
bm25_index: alias/document text -> CodeEntry
char_ngram_index: alias/document text -> CodeEntry
dense_index: embedding -> CodeEntry, optional
```

---

# 4. Normalizer là phần quan trọng nhất

Không dùng raw string để match. Phải có `normalize_key()` thật mạnh.

```python
import re
import unicodedata

ROMAN_MAP = {
    " i ": " 1 ",
    " ii ": " 2 ",
    " iii ": " 3 ",
    " iv ": " 4 ",
}

def remove_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

def normalize_key(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[,;:/()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Chuẩn hóa tiếng Việt y khoa hay gặp
    repl = {
        "tuýp": "type",
        "typ": "type",
        "típ": "type",
        "đtđ": "dai thao duong",
        "dtđ": "dai thao duong",
        "td2": "type 2",
        "tha": "tang huyet ap",
        "happ": "huyet ap",
        "bn": "benh nhan",
    }
    no_acc = remove_accents(text)
    no_acc = f" {no_acc} "

    for k, v in repl.items():
        no_acc = no_acc.replace(f" {remove_accents(k)} ", f" {v} ")

    for k, v in ROMAN_MAP.items():
        no_acc = no_acc.replace(k, v)

    no_acc = re.sub(r"[^a-z0-9.%+-]+", " ", no_acc)
    no_acc = re.sub(r"\s+", " ", no_acc).strip()
    return no_acc
```

Ví dụ các cụm sau phải về gần cùng key:

```text
ĐTĐ typ II
DTĐ type 2
dai thao duong type ii
tiểu đường tuýp 2
đái tháo đường type 2
```

về:

```text
dai thao duong type 2
```

Điểm này rất quan trọng vì trước đó ta đã xác định mention-code memory và normalization mạnh là distribution hack hợp lệ, đặc biệt với các mention lặp lại như THA, ĐTĐ, COPD.

---

# 5. ICD-10 linker

## 5.1. ICD alias table

Không chỉ dùng tên chính thức. Phải tự xây alias:

```csv
code,alias,source,priority
I10,tăng huyết áp,manual,1.0
I10,THA,abbreviation,0.98
I10,cao huyết áp,synonym,0.95
E11,đái tháo đường type 2,manual,1.0
E11,tiểu đường type 2,synonym,0.98
E11,ĐTĐ type 2,abbreviation,0.98
J18.9,viêm phổi,popular,0.90
J45,hen phế quản,manual,1.0
K21.9,trào ngược dạ dày thực quản,manual,1.0
```

Nguồn alias:

```text
1. ICD title_vi chính thức
2. ICD title_en
3. tên bệnh phổ thông
4. viết tắt y khoa Việt Nam
5. không dấu
6. biến thể type/tuýp/typ
7. biến thể cấp/mạn
8. alias LLM generate nhưng phải review
9. alias từ train/test nếu có annotation/public feedback
```

## 5.2. ICD scoring

```python
def score_icd_candidate(
    mention_key: str,
    candidate: CodeEntry,
    bm25_score: float,
    char_score: float,
    context: str,
) -> float:
    exact = 1.0 if mention_key in candidate.normalized_aliases else 0.0
    alias = max(token_jaccard(mention_key, a) for a in candidate.normalized_aliases)
    prior = candidate.priority

    specificity = 0.0
    if "type 2" in mention_key and candidate.code.startswith("E11"):
        specificity += 0.2
    if "type 1" in mention_key and candidate.code.startswith("E10"):
        specificity += 0.2
    if "tang huyet ap" in mention_key and candidate.code == "I10":
        specificity += 0.2

    return (
        5.0 * exact
        + 2.0 * alias
        + 1.0 * bm25_score
        + 1.2 * char_score
        + 0.8 * prior
        + specificity
    )
```

## 5.3. ICD fallback policy

Với ICD, nhiều mention mơ hồ:

```text
viêm phổi
đái tháo đường
suy tim
bệnh thận mạn
```

Policy:

```text
Nếu mention cụ thể rõ → chọn child code cụ thể.
Nếu mention chung và train/public prior có mã thường gặp → chọn most frequent child.
Nếu không có prior → chọn parent/common code an toàn.
Nếu metric exact-only → ưu tiên code thường gặp nhất.
Nếu metric partial hierarchy → parent fallback có thể tốt hơn.
```

Ví dụ:

```text
"viêm phổi" → J18.9 thường an toàn hơn J18 nếu gold hay dùng unspecified.
"tăng huyết áp" → I10.
"đái tháo đường type 2" → E11 hoặc E11.9 tùy dictionary/metric.
```

---

# 6. RxNorm linker

RxNorm khó hơn ICD vì thuốc có **ingredient + strength + form + brand**.

## 6.1. Parse medication mention trước

Ví dụ:

```text
amlodipine 10 mg po daily
metformin 500mg bid
paracetamol 500 mg
salbutamol khí dung
chlorpheniramine 0.4 MG/ML
```

Parser nên tách:

```python
@dataclass(frozen=True)
class DrugMention:
    raw: str
    name: str
    name_key: str
    strength_value: float | None
    strength_unit: str | None
    denominator_value: float | None
    denominator_unit: str | None
    form: str | None
    route: str | None
    frequency: str | None
```

Regex cơ bản:

```python
DRUG_STRENGTH_RE = re.compile(
    r"(?P<name>[a-zA-ZÀ-ỹ0-9\- ]+?)\s*"
    r"(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>mg|mcg|g|ml|iu|meq|%)"
    r"(?:\s*/\s*(?P<den_value>\d+(?:[.,]\d+)?)?\s*(?P<den_unit>ml|l|g))?",
    re.IGNORECASE,
)
```

Chuẩn hóa unit:

```python
UNIT_MAP = {
    "mg": "MG",
    "mcg": "MCG",
    "μg": "MCG",
    "ug": "MCG",
    "g": "G",
    "ml": "ML",
    "iu": "UNT",
    "meq": "MEQ",
}
```

Route/frequency thường **không quyết định RxCUI clinical drug**, nhưng giúp nhận diện boundary:

```text
po, uống, oral
iv, tiêm tĩnh mạch
bid, ngày 2 lần
tid, ngày 3 lần
qhs, trước ngủ
prn, khi cần
```

Tài liệu Phase 1 trước đó cũng nhấn mạnh cần parse full medication phrase, nhưng khi map RxNorm thì strength/form quan trọng hơn route/frequency.

---

## 6.2. RxNorm alias table

Từ `RXNCONSO.RRF`, tạo các alias:

```text
RXCUI
STR
TTY
SAB
SUPPRESS
CVF
```

Chỉ lấy:

```text
SAB = RXNORM
SUPPRESS != O/E/Y nếu dùng full release
hoặc dùng Current Prescribable Content cho sạch
```

Priority:

```python
TTY_PRIORITY = {
    "SCD": 1.00,
    "SBD": 0.98,
    "IN": 0.75,
    "PIN": 0.78,
    "MIN": 0.80,
    "SCDF": 0.70,
    "SBDF": 0.68,
    "GPCK": 0.60,
    "BPCK": 0.60,
}
```

Nếu mention có strength:

```text
metformin 500mg
```

thì boost `SCD/SBD`.

Nếu mention chỉ có name:

```text
metformin
```

thì boost `IN/PIN`.

---

## 6.3. RxNorm scoring

```python
def score_rx_candidate(
    mention: DrugMention,
    candidate: CodeEntry,
    bm25_score: float,
    char_score: float,
) -> float:
    candidate_key = candidate.normalized_name
    exact_name = 1.0 if mention.name_key in candidate.normalized_aliases else 0.0
    name_sim = max(token_jaccard(mention.name_key, a) for a in candidate.normalized_aliases)

    tty_boost = {
        "SCD": 1.0,
        "SBD": 0.95,
        "IN": 0.65,
        "PIN": 0.65,
        "MIN": 0.70,
        "SCDF": 0.55,
        "SBDF": 0.50,
    }.get(candidate.tty or "", 0.3)

    strength_boost = 0.0
    if mention.strength_value is not None:
        if contains_strength(candidate.canonical_name, mention.strength_value, mention.strength_unit):
            strength_boost = 1.5
        else:
            strength_boost = -1.0

    return (
        4.0 * exact_name
        + 2.0 * name_sim
        + 1.2 * bm25_score
        + 1.2 * char_score
        + 1.0 * tty_boost
        + strength_boost
    )
```

Critical rule:

```text
Nếu mention có strength mà candidate không chứa strength → giảm điểm.
Nếu mention không có strength → không ép SCD.
Nếu mention là brand name → cho SBD cơ hội cao.
Nếu mention là generic → SCD/IN ưu tiên.
```

---

# 7. Candidate generation nhiều tầng

## Tier 1 — exact normalized match

```python
cands = exact_index.get(normalize_key(mention), [])
```

Ưu tiên cực cao, thường output 1 candidate.

## Tier 2 — alias / abbreviation

```python
alias_key = alias_table.get(normalize_key(mention))
cands += exact_index.get(alias_key, [])
```

Ví dụ:

```text
THA → tăng huyết áp → I10
ĐTĐ type 2 → đái tháo đường type 2 → E11
COPD → J44
GERD → K21
```

## Tier 3 — fuzzy / char n-gram

Cực mạnh cho typo và không dấu:

```text
metfomin → metformin
salbutamon → salbutamol
tang huyet ap → tăng huyết áp
viem phoi → viêm phổi
```

Nên dùng char 3–5 gram TF-IDF thay vì chỉ edit distance.

## Tier 4 — BM25

Index mỗi code như một document:

```text
canonical_name + aliases + english_name + semantic_type + parent_name
```

BM25 tốt với query ngắn, alias nhiều.

## Tier 5 — dense retrieval, optional

Dùng nếu còn thời gian:

```text
SapBERT / BioSyn / multilingual sentence embedding
```

Nhưng với dữ liệu tiếng Việt, dense retrieval chỉ nên là **bổ sung**, không thay thế exact/fuzzy/BM25. Dictionary + char n-gram thường có ROI cao hơn trong giai đoạn đầu.

---

# 8. Reranker

Reranker nhận top 20–50 candidate:

```text
[mention] + [context window] + [candidate canonical name] + [aliases]
```

Input logic:

```text
mention: "đái tháo đường type 2"
context: "tiền sử đái tháo đường type 2, đang dùng metformin"
candidate: "E11 - Type 2 diabetes mellitus - Đái tháo đường không phụ thuộc insulin"
```

Nếu dùng LLM, bắt buộc khóa output:

```json
{
  "allowed_candidates": ["E11", "E11.9", "E10", "R73"],
  "answer_must_be_one_of_allowed_candidates": true
}
```

Không cho LLM sinh mã ngoài danh sách. Phân tích trước cũng đã chỉ rõ LLM nên dùng để rerank top-k có schema ràng buộc, không sinh ICD/RxNorm tự do.

---

# 9. Output candidate policy

Vì Jaccard phạt candidate thừa, policy nên rất chặt:

```python
def select_output_candidates(ranked: list[tuple[str, float]]) -> list[str]:
    if not ranked:
        return []

    top_code, top_score = ranked[0]

    if top_score >= 0.90:
        return [top_code]

    if len(ranked) >= 2:
        second_code, second_score = ranked[1]
        if top_score >= 0.75 and second_score >= 0.72 and abs(top_score - second_score) <= 0.05:
            return [top_code, second_code]

    if top_score >= 0.65:
        return [top_code]

    return []
```

Sau đó tune threshold theo local validation.

---

# 10. Validators bắt buộc

```python
def validate_candidates(entity_type: str, candidates: list[str]) -> None:
    if entity_type == "THUỐC":
        assert all(c in RXNORM_CODE_SET for c in candidates)
    elif entity_type == "CHẨN_ĐOÁN":
        assert all(c in ICD10_CODE_SET for c in candidates)
    else:
        assert candidates == []
```

Hard constraints:

```text
THUỐC       → chỉ RxNorm
CHẨN_ĐOÁN   → chỉ ICD-10
TRIỆU_CHỨNG → candidates = []
TÊN_XÉT_NGHIỆM → candidates = []
KẾT_QUẢ_XÉT_NGHIỆM → candidates = []
```

Đây là điều đã phân tích rất rõ: sai code system như `metformin -> E11` là lỗi nặng, cần validator chặn.

---

# 11. Repository layout

```text
air-med-linker/
  data/
    raw/
      rxnorm/
        RXNCONSO.RRF
        RXNREL.RRF
        RXNSAT.RRF
      icd10_vn/
        icd10_qd4469.xlsx

    processed/
      rxnorm_entries.parquet
      icd10_entries.parquet
      aliases.parquet

  resources/
    abbreviations_vi.csv
    disease_synonyms_vi.csv
    drug_aliases_vi.csv
    typo_variants.csv
    brand_generic.csv

  src/
    normalize/
      text_normalizer.py
      unit_normalizer.py

    ingest/
      build_rxnorm.py
      build_icd10.py
      build_aliases.py

    index/
      exact_index.py
      bm25_index.py
      char_ngram_index.py
      dense_index.py

    linker/
      base.py
      icd10_linker.py
      rxnorm_linker.py
      ensemble_linker.py
      reranker.py

    validate/
      code_validator.py
      schema_validator.py

  tests/
    test_normalize_key.py
    test_icd10_linker.py
    test_rxnorm_linker.py
    test_candidate_validator.py
```

---

# 12. Test cases nên viết ngay

```python
def test_icd_tha():
    assert link_icd("THA")[0].code == "I10"

def test_icd_diabetes_type_2():
    assert link_icd("ĐTĐ type II")[0].code.startswith("E11")

def test_icd_no_diacritic():
    assert link_icd("dai thao duong type 2")[0].code.startswith("E11")

def test_rxnorm_metformin_ingredient():
    result = link_rx("metformin")
    assert result
    assert result[0].system == "RXNORM"

def test_rxnorm_metformin_strength():
    result = link_rx("metformin 500mg")
    assert result
    assert result[0].system == "RXNORM"
    assert result[0].tty in {"SCD", "SBD", "IN", "PIN"}

def test_drug_cannot_return_icd():
    candidates = link_rx("metformin 500mg")
    assert all(c.system == "RXNORM" for c in candidates)

def test_disease_cannot_return_rxnorm():
    candidates = link_icd("đái tháo đường type 2")
    assert all(c.system == "ICD10" for c in candidates)
```

---

# 13. Metrics nội bộ

Phải đo riêng linker:

```text
candidate_recall@1
candidate_recall@5
candidate_recall@20
linking_accuracy@1
mean_reciprocal_rank
wrong_code_system_rate
empty_candidate_rate
over_candidate_rate
```

Mục tiêu:

[
Recall@20 > 95%
]

vì nếu gold code không nằm trong top-20 thì reranker không cứu được. Đây cũng là metric đã được xác định là sống còn trong phân tích loop engineering.

---

# 14. Thứ tự build thực dụng

## Ngày 1: dictionary sạch

```text
[ ] Parse ICD-10 VN
[ ] Parse RxNorm RXNCONSO
[ ] Build CodeEntry
[ ] Build exact_index
[ ] Build normalize_key
[ ] Build validator
```

## Ngày 2: alias mạnh

```text
[ ] disease_synonyms_vi.csv
[ ] abbreviations_vi.csv
[ ] drug_aliases_vi.csv
[ ] no-diacritic aliases
[ ] type/tuýp/typ normalization
[ ] common typo variants
```

## Ngày 3: retriever

```text
[ ] BM25 index
[ ] char n-gram TF-IDF
[ ] ensemble score
[ ] threshold tuning
```

## Ngày 4: medication parser

```text
[ ] drug name extraction
[ ] strength parser
[ ] unit normalization
[ ] SCD/SBD boost
[ ] route/frequency ignored for code but kept for span
```

## Ngày 5: reranker + error loop

```text
[ ] candidate_recall@20 report
[ ] wrong_code analysis
[ ] empty_candidate analysis
[ ] over-candidate analysis
[ ] tune output policy
```

---

# 15. Công thức chốt

Một ICD/RxNorm linker mạnh không phải là:

```text
string match tên bệnh/thuốc
```

Mà là:

[
\boxed{
\text{Official vocab}
+
\text{Vietnamese aliases}
+
\text{abbreviation expansion}
+
\text{no-diacritic/typo robustness}
+
\text{BM25/char n-gram retrieval}
+
\text{drug strength parser}
+
\text{small-set candidate policy}
+
\text{code-system validator}
}
]

Với bài này, tôi sẽ ưu tiên:

```text
1. normalize_key thật tốt
2. ICD/RxNorm alias table thật dày
3. char n-gram + BM25
4. medication strength parser
5. candidate validator
6. threshold tuning theo Jaccard
```

Đây là phần có ROI cao hơn fine-tune model lớn, đặc biệt khi thi cá nhân và Phase 1 chỉ cần batch inference trên 100 records.

[1]: https://www.nlm.nih.gov/research/umls/rxnorm/docs/prescribe.html "RxNorm Current Prescribable Content"
[2]: https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html "RxNorm Files"
