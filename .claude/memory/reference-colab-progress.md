---
name: reference-colab-progress
description: notebooks/PROGRESS.md là source of truth cho thực nghiệm Colab (hạ tầng, quy ước, tiến độ, kết quả)
metadata:
  type: reference
---

Thực nghiệm chạy qua Colab CLI cài global (`colab` trong `~/.local/bin`, bản 0.8.0; bản trong `venv` đã gỡ 2026-11-21
theo yêu cầu người dùng), session `eaasr`. Trạng thái, cách chạy, kết quả và sự cố
ghi ở `notebooks/PROGRESS.md`; đọc file đó trước khi tiếp tục sau khi context bị compact.

**Why:** 2026-11-01 người dùng yêu cầu: "nếu thấy thông tin dài thì tạo 1 note markdown để ghi thông tin quan trọng
và progress chạy... file markdown đó là source of truth để reference".
**How to apply:** cập nhật PROGRESS.md mỗi khi có kết quả, lỗi, quyết định; khi resume, đọc nó thay vì đoán.
Liên quan: [[project-final-project]].
