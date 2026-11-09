---
name: feedback-no-local-models
description: Không chạy model (TTS, Whisper, ...) trên laptop; chạy trên Colab, hoặc chỉ việc nhỏ ở local
metadata:
  type: feedback
---

Model (TTS, ASR như Whisper, mạng nơ-ron) luôn chạy trên Colab, không chạy trên laptop. Ở local chỉ làm việc nhẹ:
viết file, render canvas bằng Chrome headless, ghép ffmpeg.

**Why:** người dùng nói (2026-11-02, lúc làm clip MFCC) "đảm bảo nhỏ, hoặc chạy colab, k chạy model dưới laptop".
**How to apply:** kiểm lời đọc bằng Whisper thì upload script lên session Colab; xem [[project-mfcc-clips]].
