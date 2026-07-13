# Phase 1 Annotation Knowledge Report

## Summary

- Reviewed documents: 76
- Accepted entities: 2113
- Review/rejected mentions: 223
- Guideline notes: 324
- Strict runtime aliases: 103
- Context-required aliases: 20
- Strict exclusions: 106
- Conflicts: 87
- Accepted conflict decisions: 15

Runtime policy contains concept-level rules only; document identifiers remain audit provenance and are not runtime selectors.

## Strict Aliases

- `TRIỆU_CHỨNG`: 40 strict, 3 context-required
- `TÊN_XÉT_NGHIỆM`: 23 strict, 7 context-required
- `KẾT_QUẢ_XÉT_NGHIỆM`: 0 strict, 10 context-required
- `CHẨN_ĐOÁN`: 32 strict, 0 context-required
- `THUỐC`: 8 strict, 0 context-required

## Conflict Summary

| Type | Severity | Count |
| --- | --- | ---: |
| `gold_overlapping_entities` | medium | 1 |
| `review_offset_mismatch` | medium | 51 |
| `unstable_policy_evidence` | warning | 35 |

## Highest-Priority Conflicts

| Severity | Type | Mention | Documents | Action |
| --- | --- | --- | --- | --- |
| medium | `gold_overlapping_entities` | viêm gan virus c | ['88'] | `review_non_overlapping_boundary_policy` |
| medium | `review_offset_mismatch` | bị ngã trong bồn tắm | ['100'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | can thiệp x-quang | ['9'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | có thể có một tổn thương t2 | ['10'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | cảm thấy khỏe | ['3'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | du lịch | ['17'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | du thuyền | ['17'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | dòng picc đã đặt | ['66'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | dẫn lưu dịch | ['17'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | ho | ['7'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | huyết áp 159/72 | ['3'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | hút 0.5cc dịch mủ | ['17'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | hút thuốc | ['3'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | họng | ['38'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | khoa cấp cứu | ['17'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | khám bác sĩ chăm sóc chính | ['17'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | không có triệu chứng trước đó | ['3'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | không thấy giảm đau | ['19'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | không tuân thủ điều trị bằng thuốc | ['3'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | mạch 83 | ['3'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | một số vùng không thể đánh giá tốt | ['10'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | nghi ngờ | ['12'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | nhịp thở 20 | ['3'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | nội soi mật tụy ngược dòng | ['5'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | phòng cấp cứu | ['19'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | phòng cấp cứu khám và điều trị | ['12'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | phẫu thuật mở cắt nối trực tràng/đại tràng sigma | ['16'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | phẫu thuật nội soi cắt bỏ tuyến tiền liệt bên trái | ['99'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | phẫu thuật sửa van tĩnh mạch chủ ngực bụng | ['21'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | phổi rõ tiếng khi nghe | ['17'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | quá trình bệnh lý tim phổi cấp tính | ['66'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | quá trình cấp tính | ['100'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | spo2 độ bão hòa oxy 94-95 ra | ['3'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | stent graft động mạch chủ ngực | ['21'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | thăm khám chẩn đoán và điều trị | ['4'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | tim nhịp đều | ['17'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | truyền dịch | ['100'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | truyền dịch tĩnh mạch 750cc | ['16'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | tổn thương cấp tính | ['100'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | tổn thương này | ['8'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | xét nghiệm ngoại trú | ['100'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | xông khí dung | ['16'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | đi lại | ['19'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | điều trị bảo tồn | ['9'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | điều trị ngoại khoa | ['9'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | điều trị nội khoa | ['9'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | đặt catheter động mạch | ['5'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | đặt nội khí quản | ['5'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | đặt stent | ['38'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | đặt stent đường mật | ['5'] | `repair_or_null_review_position` |
