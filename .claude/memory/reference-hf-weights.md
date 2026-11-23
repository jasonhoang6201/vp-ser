---
name: reference-hf-weights
description: Weight hệ đề xuất 5 fold ở repo HF private master-122025/eaasr-iemocap; report phải dùng số của chính lần chạy có weight
metadata:
  type: reference
---

Repo HF **private** `master-122025/eaasr-iemocap` (org của người dùng), mỗi fold `fold_s{1..5}/asr` (Whisper
large-v3-turbo FT, fp16, processor, metrics) và `fold_s{k}/ser` (`model.safetensors` của `TextAudio` trong
`src/21_combined.py`, tokenizer, `config.json` ghi cách dựng feature audio, `metrics_test.json`); `demo/` chứa clip
demo. Train lại ngày 2026-11-22/23, seed 42. UA hệ đầy đủ từng fold (S5 S1 S2 S3 S4): 81.31 78.89 84.79 78.55 82.33,
trung bình 81.17; WER Whisper-FT 7.69 12.73 8.27 14.37 4.72, trung bình 9.56 (S1, S3 cao do vài câu chỉ có tiếng
cười/thở bị lặp từ).

**Người dùng chốt (2026-11-23):** weight là bằng chứng nên report và slide phải dùng số của lần chạy này. Kết quả
lần này đã thay vào tên gốc (`whisper_turbo_finetune*`, `combined_wasr_ctx_whisper_turbo*`, `asr_whisper_ft*.csv`),
số cũ còn trong git. Ablation đọc transcript Whisper (bỏ audio, bỏ ngữ cảnh, WavLM) chạy lại bằng
`tools/run_iemocap_wasr_rerun.sh`; bỏ text, text đúng, SV-FT giữ nguyên. Code train là bản gốc (chỉ phần lưu weight
đã gỡ). Token HF fine-grained cần quyền write cho org. Đã thay số mới vào main.tex, slide, flow.md, outline
(2026-11-23); kết luận đổi: lỗi ASR tốn 0.72 UA khi có audio (không còn "gần như 0"), SV-FT 81.29 > Whisper 81.17.

Demo `notebooks/src/30_demo.py` (fold S5), `CLIP_SET` chọn `demo/clips` (6 clip: 4 đúng, 2 sai giải thích được) hoặc
`demo/candidates` (39 CREMA-D + 7 clip thử, đúng 23/39; joy→anger 7/10, sadness→neutral 5/9). Chi tiết ở
`notebooks/demo/SOURCES.md`.

Colab chỉ cho 2 VM cùng lúc; fold độc lập nên chạy song song 2 hàng. Liên quan: [[project-colab-durable]],
[[project-ablation-plan]].
