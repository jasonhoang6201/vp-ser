---
name: feedback-eta
description: Mỗi khi chạy job (Colab, train, eval), báo kèm thời gian dự kiến xong
metadata:
  type: feedback
---

Khi bắt đầu chạy một job, hoặc khi báo trạng thái, luôn kèm dự kiến bao lâu xong: giờ local (UTC+7)
và cách ước tính (ví dụ đo được bao nhiêu giây / batch). Khi có số đo thật thì cập nhật lại ETA.

**Why:** 2026-11-01 người dùng nói: "khi run cái nào thì nhớ báo thêm dự kiến chạy bao lâu xong".
**How to apply:** ghi ETA vào cả câu trả lời lẫn bảng tiến độ trong `notebooks/PROGRESS.md`.
Liên quan: [[reference-colab-progress]].
