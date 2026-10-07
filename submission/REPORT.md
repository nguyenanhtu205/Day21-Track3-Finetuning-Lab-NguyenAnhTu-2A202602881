# Báo cáo Lab 21 — Fine-tuning LLMs bằng LoRA

**Họ tên:** Nguyễn Anh Tú<br>
**MSSV:** 2A202602881<br>
**Ngày:** 07/10/2026<br>
**Tier:** T4 · **Base model:** `unsloth/Qwen3.5-4B` · **GPU:** Tesla T4 14.6 GB (fp16)

## 1. Lựa chọn thí nghiệm

Tôi dùng corpus mặc định 250 ticket CSKH tiếng Việt, với đầu ra JSON gồm bốn trường
`intent`, `urgency`, `product`, `sentiment`. Dataset này phù hợp vì từng trường có nhãn
khách quan, nhờ đó tách được lỗi hiểu ticket khỏi lỗi định dạng. Base model
`unsloth/Qwen3.5-4B` phù hợp T4; T4 không có bfloat16 phần cứng nên toàn bộ train dùng
fp16 và gradient scaling. Split cố định seed 42 là 225 train / 25 validation.

Đo độ dài trên đủ 250 mẫu cho p95=98 và gợi ý `max_length=256`. Run T4 giữ
`max_length=1024`, là cấu hình tier để có biên an toàn cho corpus dài hơn; với corpus
này đó là cấp phát dư, không phải con số được đoán. Tôi ghi rõ độ lệch này thay vì nói
1024 là p95.

## 2. Bằng chứng loss mask (NB1)

Chat template **giữ `<think>`**: `reasoning preserved — safe to train on traces`.
Tôi dùng `MASK_MODE=assistant-only` với labels pre-tokenized, do đó NB3 và NB4 dùng
đúng mask đã chứng minh thay vì dựa vào `assistant_only_loss` của thư viện.

| Kiểm tra | Kết quả |
|---|---:|
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi bị loại khỏi loss | `true` |
| Token supervised / tổng token | 39 / 94 |
| `supervised_fraction` | 0.4149 |

Đoạn loss giải mã được là `</think>` rồi JSON đáp án và EOS, ví dụ
`{"intent":"doi_tra", "urgency":"trung_binh", ...}`. Tỷ lệ 41.49% xác nhận
prompt không bị tính loss, khác hẳn chế độ `everything` là 100%.

## 3. Baseline đóng băng trước train (NB2)

Tập đánh giá đủ gồm 50 target và 15 regression mẫu; `EVAL_LIMIT` không được đặt.
Baseline (b) được chạy trước train với prompt tối ưu nguyên bản (SHA
`719e74d3b6232053`). Vì (b) tốt hơn (a), đây là đối thủ thật của LoRA.

| Run | Target | Regression | Format | Latency ms/mẫu |
|---|---:|---:|---:|---:|
| (a) Base + naive prompt | 0.0000 | 0.7911 | 0.0000 | 3284.0 |
| (b) Base + optimized prompt | 0.7650 | 0.7911 | 1.0000 | 1050.6 |
| (c) LoRA fine-tune | 0.9700 | 0.6778 | 1.0000 | 1381.9 |

Prompt (b) thắng (a) ở target, format và latency; nó không bị làm yếu. Fine-tune tăng
target 0.2050 nhưng chậm hơn (b) 331.3 ms/mẫu và giảm regression 0.1133.

## 4. Cấu hình đúng và đối chứng (NB3–NB4)

`correct` gắn LoRA vào 12 linear module của text decoder với `r=16`, `alpha=32=2r`,
LR `1e-4`, batch hiệu dụng 16 và 30 optimizer step. Model có 24 linear-attention và
8 full-attention layer. Cả bốn run dùng đúng 30 step.

| Run | Vị trí | r | Trainable params | LR | Train loss | Target NB5 | Train s | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| correct | text-linear | 16 | 32,464,896 | 1e-4 | 0.6271 | 0.9700 | 393.7 | 8.78 |
| attn_only | q,v | 283 | 32,456,704 | 1e-4 | 0.5379 | 0.9700 | 267.9 | 8.79 |
| wrong_lr | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | 0.0000 | 400.2 | 8.78 |
| qlora | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.9400 | 467.2 | 3.86 |

### 4.1 Vị trí gắn adapter so với rank

`attn_only` dùng `r=283`, `alpha=566`, nhằm khớp ngân sách: 32,456,704 so với
32,464,896 tham số, chênh 0.025%. Trên target, nó hòa `correct` ở 0.9700. Vì vậy dữ
liệu này không ủng hộ kết luận rằng chỉ thay placement tất yếu làm giảm chất lượng; rank
lớn ở đây là kiểm soát công bằng chứ không phải biến chất lượng. Tuy nhiên loss không
được dùng để kết luận: `attn_only` có loss thấp hơn 0.5379 nhưng không thắng target.

### 4.2 Learning rate sai

`wrong_lr` chỉ đổi LR từ `1e-4` xuống thang full fine-tune `1e-5`. Loss cuối 1.5702 cao
hơn mạnh so với 0.6271 của `correct`, target rơi về 0.0000 và format cũng 0.0000. Log
cho thấy loss chỉ giảm chậm từ khoảng 2.163 xuống 1.119, khác với cấu hình đúng giảm
rất nhanh. Nếu chỉ nhìn một đoạn loss mà không biết LR, có thể tưởng train chưa đủ lâu;
eval frozen chứng minh update LoRA đã quá nhỏ.

### 4.3 QLoRA

QLoRA giảm VRAM từ 8.78 xuống 3.86 GB: tiết kiệm 4.92 GB, tương đương **56.0%**. Đổi
lại nó train chậm hơn (467.2 s), loss cao hơn (0.7058), và target giảm từ 0.9700 xuống
0.9400. Tiết kiệm này hữu ích khi VRAM là giới hạn cứng; nhưng vì 16-bit LoRA đã vừa
T4 và thắng target, số đo ủng hộ dùng 16-bit LoRA làm mặc định cho họ model này.

## 5. Phán quyết (NB5)

**Kết quả regression gate: FAILED.**<br>
`target Δ=+0.2050`; `regression Δ=-0.1133`; `valid_trace_rate=0.0000`.

LoRA học triage rất rõ: target tăng 0.2050 so với baseline tốt và format giữ 1.0. Nhưng
đó không đủ để deploy. Regression giảm 0.1133 từ 0.7911 xuống 0.6778, vượt xa tolerance
0.020; latency cũng cao hơn baseline (b). Kết luận nhân quả là train set chỉ gồm JSON
triage đã chuyên biệt hóa hành vi model và gây catastrophic forgetting với câu hỏi ngoài
miền. Tôi không nới gate hay sửa eval sau khi thấy kết quả. Thử nghiệm tiếp theo là trộn
1–5% replay data phổ thông, giữ base/mask/prompt/LR/step budget và đánh giá lại cùng tập
đã đóng băng.

## 6. Định tính

`qualitative_comparison.json` cho thấy fine-tune thắng baseline (b) trên 33/50 ticket
và hòa ở 17/50; không có ticket nào (b) có field score cao hơn fine-tune. Để không
cherry-pick, hai hàng cuối là các ca fine-tune **thua nhãn vàng** ở urgency (đồng thời
hòa baseline), thay vì bịa một ca thua baseline không tồn tại.

| # | Ticket rút gọn | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|
| 1 | Chuột không dây, “trả lại”, gấp | Nhầm `hoan_tien`, 0.75 | Đúng 4/4, 1.00 | FT thắng: phân biệt đổi/trả và hoàn tiền. |
| 2 | Máy xay, “muốn đổi”, 3 ngày | Nhầm intent + urgency, 0.50 | Đúng 4/4, 1.00 | FT thắng rõ rệt. |
| 3 | Nồi chiên thiếu phụ kiện, không vội | Nhầm intent + urgency, 0.50 | Sai urgency, 0.75 | FT thắng nhưng thiên về urgency trung bình. |
| 4 | Đèn LED giao chậm, “khi nào tiện” | Nhầm intent + urgency, 0.50 | Sai urgency, 0.75 | FT cải thiện intent, còn lỗi hiệu chỉnh urgency. |
| 5 | Bình giữ nhiệt, chưa thấy tiền, khi nào tiện | Sai urgency, 0.75 | Sai urgency, 0.75 | ❌ FT thua nhãn vàng; hòa (b). |
| 6 | Áo khoác bị lỗi, khi nào tiện | Sai urgency, 0.75 | Sai urgency, 0.75 | ❌ FT thua nhãn vàng; hòa (b). |

Lỗi lặp lại là các cụm urgency mềm như “khi nào tiện”, “không vội”: cả hai model thiên
về `trung_binh` thay vì `thap`.

## 7. Kết luận và điều tôi học được

Tôi không nên deploy adapter này như thay thế chung cho base model đã được prompt tốt.
Target 0.9700 rất cao, nhưng regression gate cho thấy chi phí chuyên biệt hóa: giảm
0.1133, lớn hơn năm lần ngưỡng 0.020. Nếu endpoint chỉ nhận ticket đã được route vào
triage, adapter có thể đáng dùng sau khi cân nhắc latency; nếu endpoint nhận cả câu hỏi
phổ thông, verdict FAILED phải được tôn trọng. Đòn bẩy mạnh nhất trong thí nghiệm này là
learning rate: sai một bậc thập phân đưa target về 0. Trong khi đó `attn_only` có loss
thấp nhất nhưng chỉ hòa target, chứng minh train loss là proxy không đủ. QLoRA giảm 56%
VRAM nhưng chậm hơn và mất 0.03 target, nên là trade-off triển khai, không phải mặc định
chất lượng. Bước cải thiện đúng là replay data 1–5%, không phải làm yếu baseline hay
nới gate.

**Ba điều tôi học được:**

1. Mask phải được decode và assert; loss curve đẹp không chứng minh prompt đã bị che đúng.
2. Prompt engineering có thể vừa chính xác vừa nhanh hơn, nên baseline phải là prompt mạnh.
3. Accuracy target không đủ cho quyết định deploy; regression gate phát hiện quên kiến thức.

**Nếu có thêm 2 giờ:** Tôi sẽ chạy contrast replay 1–5% với cùng eval frozen.

## Phụ lục — thưởng

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
