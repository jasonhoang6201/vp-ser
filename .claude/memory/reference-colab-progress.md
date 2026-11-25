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

**Cập nhật 2026-11-25:** để nộp bài, người dùng cho xóa PROGRESS.md, `runs/`, code MELD, baseline 11-14, cascade 12,
90_compare, run cũ và hạ tầng resume (xem [[project-colab-durable]]). Bản cuối còn trong git (`git show ea5f57b:notebooks/PROGRESS.md`).
Cách chạy lại và cấu hình từng dòng của report giờ nằm ở `notebooks/README.md`; `tools/compare_results.py` in lại mọi bảng.
