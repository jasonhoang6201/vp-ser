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

Cập nhật 2026-11-15: script hiện tại là `tools/run_iemocap_s42.sh`, `run_iemocap_whisper_asr.sh`, `run_meld_s42.sh`
(chain_lib nhận `DS`, `SESSION`; mỗi đợt một repo state riêng). **VM Colab bị thu hồi sau khoảng 24 giờ kể từ lúc tạo**
(VM `eaasr` tạo 14/11 19:32, mất 15/11 khoảng 19:35, giữa một bước). Khi lên chuỗi dài, tính trước mốc 24 giờ; chuỗi
chạy lại sẽ tự tạo VM mới và restore, chỉ mất bước đang chạy dở.

**Why:** người dùng có lúc phải offline giữa chừng, VM bị thu hồi ([[project-colab-sleep]]); license IEMOCAP cấm phát tán
nên repo HF không bao giờ được để public.

**How to apply:** file lớn đưa lên VM qua HF (VM tải rất nhanh), không qua `colab upload`; job dài thì dùng mẫu script resume này.
