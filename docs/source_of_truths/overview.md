Phân tích các câu trong đề bài, chỉ ra các điểm mấu chốt:
Bài 2 - Ontological Reasoning in Medical Knowledge Retrieval
Phase 1 kết thúc trong

28d : 13h : 09m : 42s

Đã đăng ký
Tổng quan
Lịch trình
Lộ trình

3 Phase
1
Phase 1
LIVE
Vòng 1 - Sơ loại
02/07/2026 → 30/07/2026

Tệp ZIP
GPU
2
Phase 2
Vòng 2 - Sơ khảo
17/08/2026 → 19/08/2026

API endpoint
GPU
3
Phase 3
Vòng 3 - Chung kết
09/09/2026 → 10/09/2026

API endpoint
GPU
Bài toán yêu cầu xây dựng hệ thống AI xử lý văn bản y khoa tự do - ghi chú bác sĩ, giấy xuất viện, kết quả xét nghiệm, hồ sơ EHR - để phát hiện và chuẩn hóa các khái niệm y tế xuất hiện trong văn bản. Hệ thống cần xác định loại khái niệm (triệu chứng, kết quả xét nghiệm, bệnh, thuốc, thông tin bệnh nhân), ánh xạ bệnh với chuẩn ICD-10 và thuốc với chuẩn RxNorm, đồng thời suy luận mối liên hệ ngữ cảnh (phủ định, người nhà, tiền sử) cũng như quan hệ giữa các khái niệm. Đây là bài toán nền tảng cho chuyển đổi số y tế, giúp dữ liệu lâm sàng phi cấu trúc có thể liên thông và khai thác trên quy mô lớn cho chẩn đoán, nghiên cứu dịch tễ và các ứng dụng AI y khoa.

1. Tổng quan
Bài toán tập trung vào việc sử dụng các giải pháp NLP, LLM hoặc kết hợp agents để xây dựng một hệ thống AI có khả năng thực hiện đồng thời:

Xác định và chuẩn hóa khái niệm y tế chuyên môn trong văn bản.
Suy luận ontology (Ontological Reasoning) trên dữ liệu y khoa dạng văn bản tự do (free-form clinical text) — xác định quan hệ giữa các khái niệm y tế trong một ngữ cảnh nhất định.
Hệ thống AI được cung cấp hai cơ sở tri thức y khoa: ICD (cho bệnh) và RxNorm (cho thuốc). Nhiệm vụ của hệ thống:

Phát hiện các khái niệm y tế xuất hiện trong văn bản.
Xác định loại khái niệm (triệu chứng, tên xét nghiệm, kết quả xét nghiệm, chẩn đoán, thuốc).
Ánh xạ các khái niệm cần tri thức (chẩn đoán, thuốc) với chuẩn ICD/RxNorm tương ứng và trả về danh sách mã định danh phù hợp nhất.
Xác định mối liên hệ ngữ cảnh giữa các khái niệm trong đoạn văn (phủ định, người nhà, tiền sử).
Bài toán xử lý hai nhóm giải pháp chính:

Xác định và chuẩn hóa khái niệm y tế từ văn bản tự do.
Suy luận mối liên hệ của các khái niệm đã xác định.
2. Bối cảnh
Trong lĩnh vực y tế, dữ liệu lâm sàng và hồ sơ bệnh án thường được ghi nhận dưới nhiều định dạng và cách diễn đạt khác nhau, phụ thuộc vào cơ sở khám chữa bệnh, chuyên khoa, ngôn ngữ chuyên môn và thói quen nhập liệu của nhân viên y tế.

Để đảm bảo khả năng liên thông, thống nhất và khai thác dữ liệu trên quy mô lớn, nhiều hệ thống chuẩn y khoa đã được xây dựng như ICD, SNOMED CT, RxNorm, LOINC, UMLS,… Các chuẩn này đóng vai trò như "ngôn ngữ chung" giúp đồng bộ dữ liệu giữa các bệnh viện, hệ thống bảo hiểm, nền tảng nghiên cứu và các ứng dụng AI trong y tế.

Tuy nhiên, trong thực tế vận hành, phần lớn dữ liệu y khoa vẫn tồn tại dưới dạng văn bản tự do — ghi chú bác sĩ, mô tả triệu chứng, kết luận chẩn đoán, báo cáo cận lâm sàng — nơi cùng một khái niệm có thể được diễn đạt theo nhiều cách khác nhau, sử dụng từ viết tắt, thuật ngữ địa phương hoặc chứa lỗi chính tả và cấu trúc không chuẩn hóa.

Việc chuẩn hóa các khái niệm y tế từ văn bản tự do đòi hỏi mô hình phải hiểu ngữ cảnh chuyên môn sâu, xử lý hiện tượng đa nghĩa, đồng nghĩa và các biến thể diễn đạt phức tạp. Đây là một hướng nghiên cứu và ứng dụng quan trọng, đóng vai trò nền tảng cho quá trình chuyển đổi số và phát triển AI trong chăm sóc sức khỏe.

3. Mô tả bài toán
3.1 Input
Input là một đoạn văn bản y khoa dạng tự do (free-form text). Văn bản có thể là:

Kết quả khám lâm sàng
Giấy xuất viện
Ghi chú của bác sĩ
Kết quả chẩn đoán hình ảnh
Kết quả xét nghiệm
Hồ sơ sức khỏe điện tử (EHR)
Các ghi chú lâm sàng khác
Dữ liệu đầu vào có thể chứa: thuật ngữ y khoa, viết tắt, thông tin bệnh nhân và nhiều loại khái niệm y tế xuất hiện đồng thời trong cùng một văn bản (mọi văn bản đều chứa nhiều hơn 1 khái niệm).

Ví dụ input:

"Bệnh nhân bị bệnh 1 tuần nay, ho đờm xanh, tức ngực, đau thượng vị, ợ hơi, được chẩn đoán mắc bệnh trào ngược dạ dày - thực quản."

3.2 Output
Output là danh sách các khái niệm y tế được phát hiện trong văn bản. Mỗi khái niệm gồm các trường sau:

Trường	Mô tả
text	Cụm từ trong input mà hệ thống xác định là khái niệm y tế
position	List 2 phần tử [start, end] — vị trí ký tự bắt đầu/kết thúc của cụm trong input (index từ 0 đến n-1, n = độ dài input theo ký tự)
type	Loại khái niệm y tế (xem bảng nhãn dưới)
assertions	List các chuỗi thể hiện mối liên hệ ngữ cảnh (tối đa 3 phần tử, áp dụng cho CHẨN_ĐOÁN, THUỐC, TRIỆU_CHỨNG)
candidates	List mã chuẩn y tế dự đoán (chỉ áp dụng cho CHẨN_ĐOÁN và THUỐC)
Bảng nhãn loại khái niệm (type):

Nhãn	Ý nghĩa
TRIỆU_CHỨNG	Tên triệu chứng bệnh nhân mắc phải
TÊN_XÉT_NGHIỆM	Tên xét nghiệm bệnh nhân thực hiện
KẾT_QUẢ_XÉT_NGHIỆM	Kết quả xét nghiệm bệnh nhân thực hiện, bao gồm giá trị và đơn vị
CHẨN_ĐOÁN	Tên chẩn đoán của bác sĩ về bệnh mà bệnh nhân mắc phải
THUỐC	Tên thuốc bệnh nhân điều trị
Bảng giá trị assertions:

Giá trị	Ý nghĩa	Ví dụ
isNegated	Khái niệm bị phủ định trong văn bản	"không ho"
isFamily	Khái niệm liên quan đến tình trạng của người nhà, họ hàng	"bố bệnh nhân xuất hiện trường hợp đau bụng tương tự"
isHistorical	Khái niệm liên quan đến tiền sử bệnh nhân	"có tiền sử hen suyễn"
Một khái niệm có thể có 0, 1, 2 hoặc cả 3 giá trị assertion cùng lúc. Nếu không có liên hệ đặc biệt, assertions là list rỗng [].

Cấu trúc candidates: list các string mã chuẩn y tế.

Với CHẨN_ĐOÁN: list mã ICD-10.
Với THUỐC: list mã RxNorm.
Với các loại khái niệm khác: list rỗng [].
3.3 Ví dụ đầy đủ
Input:

"Bệnh nhân nam 70 tuổi bị bệnh 1 tuần nay, ho đờm xanh, tức ngực, đau thượng vị, ợ hơi, được chẩn đoán mắc bệnh trào ngược dạ dày - thực quản. Bệnh nhân có tiền sử sử dụng Chlorpheniramine 0.4 MG/ML, Capsaicin 0.38 MG/ML, đã tiến hành tổng phân tích tế bào máu bằng máy lazer (tbm): WBC:14,43; NEUT% (Tỷ lệ % bạch cầu trung tính):76,4; LYPH% (Tỷ lệ bạch cầu lympho):12,8;"

Output các khái niệm tương ứng:

CHẨN_ĐOÁN: "bệnh trào ngược dạ dày - thực quản" → mã ICD: K21.0, K21.9
TRIỆU_CHỨNG: "ho đờm xanh", "tức ngực", "đau thượng vị", "ợ hơi"
TÊN_XÉT_NGHIỆM: "WBC", "NEUT% (Tỷ lệ % bạch cầu trung tính)", "LYPH% (Tỷ lệ bạch cầu lympho)"
KẾT_QUẢ_XÉT_NGHIỆM: "14,43", "76,4", "12,8"
THUỐC:
"Chlorpheniramine 0.4 MG/ML" → RxNorm: 360047, assertion: isHistorical
"Capsaicin 0.38 MG/ML" → RxNorm: 1660761, assertion: isHistorical
Format JSON chuẩn của output:

[
  {
    "text": "bệnh trào ngược dạ dày - thực quản",
    "type": "CHẨN_ĐOÁN",
    "assertions": [],
    "candidates": ["K21.0", "K21.9"],
    "position": [98, 132]
  },
  {
    "text": "ho đờm xanh",
    "type": "TRIỆU_CHỨNG",
    "assertions": [],
    "position": [42, 53]
  },
  {
    "text": "Chlorpheniramine 0.4 MG/ML",
    "type": "THUỐC",
    "candidates": ["360047"],
    "assertions": ["isHistorical"],
    "position": [167, 193]
  },
  {
    "text": "WBC",
    "type": "TÊN_XÉT_NGHIỆM",
    "position": [260, 263]
  },
  {
    "text": "14,43",
    "type": "KẾT_QUẢ_XÉT_NGHIỆM",
    "position": [264, 269]
  }
]
Lưu ý: Các giá trị liên quan đến thông tin cá nhân (tên, tuổi, địa chỉ, SĐT) trong dữ liệu đều là giá trị synthetic, không phải thông tin người thật.

4. Dữ liệu bài toán
4.1 Cơ sở dữ liệu chuẩn y tế
ICD-10 cho các loại bệnh (áp dụng cho khái niệm CHẨN_ĐOÁN).
RxNorm cho các loại thuốc (áp dụng cho khái niệm THUỐC).
4.2 Bộ dữ liệu được cung cấp
Thí sinh được cung cấp tập test gồm 100 bản ghi, lưu trong file test.zip. Sau khi giải nén:

test/
└── input/
    ├── 1.txt       # Văn bản đầu vào của bản ghi 1
    ├── 2.txt       # Văn bản đầu vào của bản ghi 2
    ├── ...
    └── 100.txt
Các file .txt là văn bản free-form text làm input.
Mọi văn bản đều chứa nhiều hơn 1 khái niệm y tế.
Với mỗi file .txt, thí sinh cần trả về một file .json tương ứng chứa list các dictionary mô tả khái niệm y tế phát hiện được (format như mục 3.2).

4.3 Dữ liệu huấn luyện
Lưu ý quan trọng: Đề bài không cung cấp tập train. Thí sinh cần sử dụng các giải pháp ngoài lời giải chính để tạo thêm dữ liệu phục vụ huấn luyện mô hình.

Các hướng có thể cân nhắc:

Tạo synthetic data bằng LLM lớn.
Sử dụng pretrained model trên dữ liệu y khoa công khai.
Khai thác các bộ dữ liệu y khoa mã nguồn mở (i2b2, MIMIC, n2c2, BioBERT corpora,...).
Annotation thủ công một tập nhỏ làm seed.
Data augmentation từ corpus y khoa tiếng Việt.
Chúc thí sinh thi tốt!