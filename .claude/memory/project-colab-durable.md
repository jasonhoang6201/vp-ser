---
name: project-colab-durable
description: Colab upload lớn làm chết VM; dữ liệu IEMOCAP và trạng thái chạy để ở repo HF private, chuỗi resume được
metadata:
  type: project
---

`colab upload` file lớn (1.4 GB, kể cả chia 20 MB) làm VM Colab chết hai lần (2026-11-02). Dữ liệu IEMOCAP chính thức
nằm ở repo HF **private** `jasonhoang6201/iemocap-16k`; trạng thái chạy (cache, results, checkpoint SenseVoice-FT) đẩy lên
repo private `jasonhoang6201/iemocap-runs`. Chuỗi IEMOCAP: `caffeinate -i zsh notebooks/tools/run_iemocap_official.sh`,
chạy lại là resume (bỏ bước đã xong, bỏ seed đã có npz).

**Why:** người dùng có lúc phải offline giữa chừng, VM bị thu hồi ([[project-colab-sleep]]); license IEMOCAP cấm phát tán
nên repo HF không bao giờ được để public.

**How to apply:** file lớn đưa lên VM qua HF (VM tải rất nhanh), không qua `colab upload`; job dài thì dùng mẫu script resume này.
