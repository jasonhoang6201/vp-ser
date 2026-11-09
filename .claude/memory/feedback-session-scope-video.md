---
name: feedback-session-scope-video
description: Session làm clip chỉ lo generate video và dạy học; không đụng training, research, model của đề tài
metadata:
  type: feedback
---

Trong session làm clip (video-pipeline/), chỉ lo việc generate video và giảng dạy cho người dùng học. Không quan tâm, không chạm vào việc training và research làm model (notebook thực nghiệm, phiên Colab `eaasr`, HF iemocap-runs).

**Why:** người dùng chạy song song một session khác lo training/research; hai việc tách bạch để không tranh tài nguyên hay làm lẫn.

**How to apply:** chỉ dùng phiên Colab `tts` cho lồng tiếng; không kill, không exec vào phiên khác; không đề xuất thí nghiệm hay sửa notebook. Số liệu thực nghiệm chỉ được nhắc khi đã có sẵn trên trang demo. Liên quan [[project-mfcc-clips]].
