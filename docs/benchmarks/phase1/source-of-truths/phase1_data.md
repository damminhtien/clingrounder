# Phân tích dữ liệu `input.zip`
## 1. Cấu trúc
- ZIP có thư mục `input/` và 100 file `.txt`, đánh số `1.txt` đến `100.txt`.
- Không có nhãn gold trong gói này; đây là dữ liệu inference/test cần sinh `output/*.json`.

## 2. Thống kê độ dài
- Số file: 100
- Độ dài ký tự: min=136, median=1222.5, mean=1323.4, max=4428
- Số token thô: min=32, median=255.0, mean=279.4, max=1019
- Số dòng: min=3, median=29.5, mean=29.6, max=140

Các file dài nhất:
- `3.txt`: 4428 ký tự, 1019 token, 140 dòng
- `41.txt`: 3347 ký tự, 697 token, 56 dòng
- `20.txt`: 3007 ký tự, 621 token, 61 dòng
- `47.txt`: 2984 ký tự, 635 token, 33 dòng
- `54.txt`: 2956 ký tự, 658 token, 73 dòng
- `1.txt`: 2942 ký tự, 597 token, 54 dòng
- `58.txt`: 2931 ký tự, 607 token, 76 dòng
- `23.txt`: 2878 ký tự, 613 token, 32 dòng
- `48.txt`: 2762 ký tự, 595 token, 52 dòng
- `13.txt`: 2572 ký tự, 528 token, 43 dòng

Các file ngắn nhất:
- `55.txt`: 136 ký tự, 32 token, 7 dòng
- `31.txt`: 196 ký tự, 43 token, 3 dòng
- `90.txt`: 235 ký tự, 51 token, 8 dòng
- `15.txt`: 281 ký tự, 61 token, 8 dòng
- `68.txt`: 297 ký tự, 61 token, 11 dòng
- `62.txt`: 334 ký tự, 68 token, 13 dòng
- `99.txt`: 341 ký tự, 70 token, 12 dòng
- `79.txt`: 344 ký tự, 72 token, 8 dòng
- `29.txt`: 373 ký tự, 85 token, 12 dòng
- `57.txt`: 373 ký tự, 79 token, 9 dòng

## 3. Cấu trúc section
Top heading/section sau chuẩn hóa:
- 74: đánh giá tại bệnh viện
- 52: tiền sử bệnh hiện tại
- 48: tiền sử bệnh
- 39: tiền sử bệnh nội khoa
- 35: bệnh sử hiện tại
- 33: triệu chứng hiện tại
- 30: các sự kiện trước khi nhập viện
- 27: đặc điểm triệu chứng
- 22: diễn biến bệnh
- 18: các bệnh lý mãn tính
- 16: kết quả xét nghiệm
- 16: các thủ thuật đã thực hiện
- 15: kết quả chẩn đoán hình ảnh
- 15: các phát hiện chẩn đoán khác
- 15: tiền sử phẫu thuật / thủ thuật
- 14: các bệnh lý mạn tính
- 11: thời điểm khởi phát triệu chứng
- 11: tình trạng ngay trước khi nhập viện
- 10: các triệu chứng hiện tại
- 9: triệu chứng khi nhập viện

Phân bố section đánh số:
- ('1', '2', '3'): 73 file
- ('1', '2'): 15 file
- ('2', '3'): 7 file
- (không đánh số): 2 file
- ('2',): 2 file
- ('1', '3'): 1 file

## 4. Marker ngữ cảnh
- `neg`: xuất hiện trong 86/100 file, tổng 478 marker thô
- `historical`: xuất hiện trong 95/100 file, tổng 490 marker thô
- `family`: xuất hiện trong 6/100 file, tổng 10 marker thô
- `uncertain`: xuất hiện trong 30/100 file, tổng 52 marker thô

Lưu ý: marker thô có false positive; ví dụ `không đặc hiệu` không nên coi là phủ định entity, và `con trai phát hiện bệnh nhân` không đồng nghĩa `isFamily` cho bệnh của người nhà.

## 5. Từ khóa y khoa hay gặp theo lexicon thô
Triệu chứng:
- nôn: 86 lần / 23 file
- khó thở: 83 lần / 33 file
- yếu: 71 lần / 27 file
- sốt: 47 lần / 20 file
- đau bụng: 47 lần / 20 file
- đau ngực: 44 lần / 20 file
- buồn nôn: 41 lần / 21 file
- ho: 41 lần / 16 file
- phù: 34 lần / 14 file
- mệt mỏi: 31 lần / 14 file
- tiêu chảy: 21 lần / 8 file
- đau đầu: 21 lần / 7 file
- ngất: 20 lần / 6 file
- đánh trống ngực: 20 lần / 7 file
- chóng mặt: 18 lần / 10 file

Chẩn đoán/bệnh:
- tăng huyết áp: 30 lần / 24 file
- ung thư: 28 lần / 15 file
- hẹp: 22 lần / 9 file
- đái tháo đường: 21 lần / 18 file
- nhiễm trùng: 17 lần / 12 file
- nhiễm khuẩn: 12 lần / 6 file
- rung nhĩ: 12 lần / 8 file
- sỏi: 12 lần / 5 file
- tắc nghẽn: 11 lần / 7 file
- bệnh tim mạch: 10 lần / 9 file
- nhồi máu: 10 lần / 6 file
- viêm phổi: 10 lần / 6 file
- thiếu máu: 9 lần / 6 file
- suy tim: 8 lần / 7 file
- tăng lipid máu: 8 lần / 7 file

Thuốc:
- tylenol: 10 lần / 5 file
- omeprazole: 9 lần / 2 file
- lasix: 8 lần / 5 file
- metoprolol: 7 lần / 5 file
- nitroglycerin: 7 lần / 3 file
- prednisone: 6 lần / 1 file
- vancomycin: 6 lần / 5 file
- aspirin: 5 lần / 5 file
- albuterol: 3 lần / 3 file
- doxycycline: 3 lần / 2 file
- clopidogrel: 2 lần / 1 file
- furosemide: 2 lần / 2 file
- heparin: 2 lần / 2 file
- insulin: 2 lần / 1 file
- levofloxacin: 2 lần / 2 file

Xét nghiệm:
- creatinine: 17 lần / 8 file
- kali: 15 lần / 5 file
- bạch cầu: 11 lần / 10 file
- bilirubin: 9 lần / 3 file
- hct: 7 lần / 2 file
- k: 7 lần / 4 file
- troponin: 7 lần / 6 file
- inr: 6 lần / 3 file
- alt: 5 lần / 4 file
- ast: 4 lần / 3 file
- bun: 4 lần / 4 file
- glucose: 4 lần / 3 file
- ure: 4 lần / 4 file
- đường huyết: 4 lần / 4 file
- lactate: 3 lần / 3 file

## 6. Vấn đề chất lượng dữ liệu
- Double-space groups: 2015 nhóm, xuất hiện trong 86 file.
- Thiếu khoảng trắng sau dấu chấm: 21 lần, trong 7 file.
- Token bất thường `w`: [2].
- Một số câu có dấu hiệu dịch máy/ẩn danh và lỗi nối từ, cần xử lý bằng parser bảo toàn offset.

## 7. Khuyến nghị pipeline
1. Reader giữ nguyên raw text; mọi normalization chỉ dùng để match, không dùng để xuất offset.
2. Section detector: ưu tiên `Tiền sử`, `Bệnh sử hiện tại`, `Đánh giá tại bệnh viện`, `Kết quả xét nghiệm`, `Thuốc trước khi nhập viện`.
3. Entity extraction hybrid: dictionary/regex cho thuốc, xét nghiệm, triệu chứng phổ biến; model/LLM chỉ hỗ trợ tăng recall.
4. Assertion rule engine: phủ định, tiền sử, người nhà; cần blacklist `không đặc hiệu` và xử lý `không loại trừ` là uncertain, không phải negated.
5. Candidate linking: `CHẨN_ĐOÁN` → ICD-10; `THUỐC` → RxNorm; type khác candidates=[]; không spam candidate vì metric dùng Jaccard.
6. Validator bắt buộc: schema, allowed type/assertion, candidates theo type, và `raw_text[start:end] == text`.
