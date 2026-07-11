# Phase 1 Annotation Knowledge Report

## Summary

- Reviewed documents: 75
- Accepted entities: 2244
- Review/rejected mentions: 277
- Guideline notes: 351
- Strict runtime aliases: 104
- Context-required aliases: 24
- Strict exclusions: 135
- Conflicts: 123

Runtime policy contains concept-level rules only; document identifiers remain audit provenance and are not runtime selectors.

## Strict Aliases

- `TRIỆU_CHỨNG`: 42 strict, 4 context-required
- `TÊN_XÉT_NGHIỆM`: 22 strict, 9 context-required
- `KẾT_QUẢ_XÉT_NGHIỆM`: 0 strict, 11 context-required
- `CHẨN_ĐOÁN`: 32 strict, 0 context-required
- `THUỐC`: 8 strict, 0 context-required

## Conflict Summary

| Type | Severity | Count |
| --- | --- | ---: |
| `positive_negative_same_mention` | high | 15 |
| `positive_type_disagreement` | high | 5 |
| `review_offset_mismatch` | medium | 68 |
| `unstable_policy_evidence` | warning | 35 |

## Highest-Priority Conflicts

| Severity | Type | Mention | Documents | Action |
| --- | --- | --- | --- | --- |
| high | `positive_negative_same_mention` | 1 | ['66'] | `require_context_specific_rule` |
| high | `positive_negative_same_mention` | 14279 | ['51'] | `require_context_specific_rule` |
| high | `positive_negative_same_mention` | biến đổi cấp tính | ['35'] | `require_context_specific_rule` |
| high | `positive_negative_same_mention` | bình thường | ['1', '3', '17', '35', '37', '48', '56'] | `require_context_specific_rule` |
| high | `positive_negative_same_mention` | bất thường | ['8', '14', '56', '91', '92', '95'] | `require_context_specific_rule` |
| high | `positive_negative_same_mention` | chưa phát hiện bất thường | ['11', '28'] | `require_context_specific_rule` |
| high | `positive_negative_same_mention` | dấu hiệu sinh tồn | ['54', '91', '97'] | `require_context_specific_rule` |
| high | `positive_negative_same_mention` | ho | ['5', '7', '10', '12', '27', '32', '38', '40', '98'] | `require_context_specific_rule` |
| high | `positive_negative_same_mention` | hẹp gây hạn chế dòng chảy | ['43'] | `require_context_specific_rule` |
| high | `positive_negative_same_mention` | kháng sinh | ['58', '96'] | `require_context_specific_rule` |
| high | `positive_negative_same_mention` | khó thở | ['1', '8', '14', '27', '28', '32', '33', '38', '42', '43', '44', '47', '50', '53', '56', '91', '93', '95', '96', '100'] | `require_context_specific_rule` |
| high | `positive_negative_same_mention` | ngã | ['35', '48'] | `require_context_specific_rule` |
| high | `positive_negative_same_mention` | thuốc giảm đau opioid | ['34'] | `require_context_specific_rule` |
| high | `positive_negative_same_mention` | tăng gánh nhẹ tuần hoàn phổi | ['37'] | `require_context_specific_rule` |
| high | `positive_negative_same_mention` | đái tháo đườngđái tháo đường | ['35'] | `require_context_specific_rule` |
| high | `positive_type_disagreement` | huyết khối | ['16', '32'] | `require_context_or_resolve_type_policy` |
| high | `positive_type_disagreement` | hạ huyết áp | ['5', '58', '96', '97'] | `require_context_or_resolve_type_policy` |
| high | `positive_type_disagreement` | hạ huyết áp không đặc hiệu | ['16', '46', '58'] | `require_context_or_resolve_type_policy` |
| high | `positive_type_disagreement` | tim to | ['3', '20', '37'] | `require_context_or_resolve_type_policy` |
| high | `positive_type_disagreement` | tăng men gan | ['4', '35'] | `require_context_or_resolve_type_policy` |
| medium | `review_offset_mismatch` | bất thường | ['92'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | bị ngã trong bồn tắm | ['100'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | can thiệp x-quang | ['9'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | chưa phát hiện bất thường | ['11'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | chọc dò dịch ổ bụng 7l | ['11'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | chọc dò màng phổi 3l4 | ['11'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | chống đông máu | ['65'] | `repair_or_null_review_position` |
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
| medium | `review_offset_mismatch` | khoa cấp cứu | ['13'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | khoa cấp cứu | ['17'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | khám bác sĩ chăm sóc chính | ['17'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | khám tại phòng khám | ['13'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | không có triệu chứng trước đó | ['3'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | không liên quan đến gắng sức | ['14'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | không thấy giảm đau | ['19'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | không tuân thủ điều trị bằng thuốc | ['3'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | mông bên phải | ['13'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | mạch 83 | ['3'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | một số vùng không thể đánh giá tốt | ['10'] | `repair_or_null_review_position` |
| medium | `review_offset_mismatch` | nghi ngờ | ['12'] | `repair_or_null_review_position` |
