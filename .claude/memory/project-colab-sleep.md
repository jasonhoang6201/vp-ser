---
name: project-colab-sleep
description: Mac sleep (gập nắp) làm Colab thu hồi session CLI; phải giữ máy thức khi chạy job
metadata:
  type: project
---

Gập nắp Mac (clamshell sleep) lúc đang chạy job Colab CLI thì VM bị thu hồi, mất mọi thứ trên VM
(2026-11-01 21:23, notebook 21). Session rảnh không ai poll cũng bị tắt (17:20 → 20:54 cùng ngày).

**Why:** Colab cần client (CLI trên máy local) giữ kết nối; máy sleep thì session bị coi là bỏ.

**How to apply:** trước job dài, nhắc người dùng cắm sạc, không gập nắp, chạy `caffeinate -i`; luôn tải
kết quả về từng phần ngay khi có (xem [[reference-colab-progress]]).
