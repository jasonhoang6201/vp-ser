---
name: feedback-session-split
description: Khi chạy song song nhiều session Claude, việc làm clip (narration, scene, TTS, render) thuộc session khác; session còn lại chỉ lo training và research
metadata:
  type: feedback
---

2026-11-03 người dùng chạy hai session Claude cùng lúc. Một session làm clip lộ trình (`video-pipeline/`, Colab session `tts`),
session kia chỉ lo training (Colab session `eaasr`, `notebooks/`) và research.

**Why:** hai session cùng chạy `run_tts.sh` trên một session Colab `tts` và cùng sửa `video-pipeline/README.md` thì trùng việc, giẫm chân nhau.
**How to apply:** trước khi đụng vào clip, kiểm `ListAgents` xem có peer session nào đang làm không; nếu có thì không chạy TTS/render,
không sửa README clip, không tạo lại session `tts`. Liên quan [[project-mfcc-clips]], [[reference-colab-progress]].
