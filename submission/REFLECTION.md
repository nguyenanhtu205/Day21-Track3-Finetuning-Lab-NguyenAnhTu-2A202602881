# Reflection — Lab 21

**1. Điều gì làm bạn ngạc nhiên nhất?**

LoRA đạt target 0.970 nhưng vẫn FAILED: regression giảm 0.113. Target accuracy không
đủ để kết luận model tốt hơn cho triển khai.

**2. Bạn mất nhiều thời gian nhất ở đâu?**

NB4 mất nhiều thời gian nhất vì nạp model và train ba contrast độc lập. QLoRA còn chậm
hơn 16-bit LoRA dù tiết kiệm nhiều VRAM.

**3. Niềm tin nào đã thay đổi?**

Tôi không còn tin loss thấp hơn là đủ để chọn adapter. `attn_only` có loss tốt nhất
nhưng chỉ hòa target với cấu hình correct.

**4. Bạn dùng AI assistant vào việc gì?**

Tôi dùng AI assistant để đọc log, đối chiếu artefact và cấu trúc report. AI không thay
thế GPU; tất cả số liệu và phán quyết ở đây đến từ artefact Colab thực.

**5. Bước đầu tiên cho khách hàng thật?**

Tôi sẽ định nghĩa và freeze eval/regression gate trước khi train, sau đó decode labels
để chứng minh loss mask đúng.
