---
name: feedback-no-forward-refs
description: Khi dạy kiến thức nền, không chèn liên hệ tới kiến thức chưa học (SER, ASR, CTC, model, đề tài)
metadata:
  type: feedback
---

Khi giải thích một khái niệm nền (sóng, f0, họa âm, spectrogram, Mel...), chỉ dùng những gì người dùng
đã học. Không thêm mục "liên hệ đề tài", không nhắc SER/ASR/CTC/Whisper/model để "nối" sang phần sau.

**Why:** 2026-10-30, khi đang học cái lược (họa âm), người dùng nói: "khoan hay chèn những kiến thức tôi
chưa cần biết vào như ser... hiện tại nhắc những kiến thức sau làm loãng và tôi đang đọc k hiểu".

**How to apply:** trả lời đúng câu hỏi, ví dụ đời thường, ngắn. Phần liên hệ đề tài để dành tới khi học
đến phần đó thì nhắc lại. Áp dụng cho cả chat lẫn trang HTML demo mới.
Liên quan: [[feedback-walkthrough]], [[user-profile]].
