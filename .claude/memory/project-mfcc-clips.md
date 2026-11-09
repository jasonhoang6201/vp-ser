---
name: project-mfcc-clips
description: Người dùng muốn clip video có giọng đọc cho mọi trang lộ trình từ MFCC (05) đến Đánh giá (24); pipeline ở video-pipeline/, mp4 ở video/<trang demo>/ (cả hai ở gốc project), làm theo từng trang
metadata:
  type: project
---

2026-11-02 người dùng yêu cầu làm clip cho từng phần MFCC: light theme, không màu mè, animation diagram, giải thích chi tiết
(painpoint lúc đó, ra đời giải quyết gì, tốt cho gì, ảnh hưởng, cách chạy, vì sao có phương pháp đó), mục tiêu là biết thuật ngữ
và biết nó làm gì (trình độ thạc sĩ). Giọng: VieNeu-TTS v3 Turbo qua Colab CLI.

Cùng ngày người dùng nói rõ: phải có full clip để học, gồm cả GMM, HMM, VITS, Tacotron..., tức là mọi trang `demo/06-*.html`
đến `demo/24-*.html` (mỗi section của trang là một clip, như MFCC). Làm theo từng trang, batch TTS cả trang trong một session Colab,
người dùng compact giữa các trang.

Pipeline và bảng trạng thái các clip: `video-pipeline/README.md` (đọc khi resume).

2026-11-03: đủ 99 clip trang 05 đến 24, đã lồng tiếng, render ra `video/<trang demo>/*.mp4` và kiểm lời bằng Whisper small trên Colab.
Kinh nghiệm: `colab exec` đôi khi treo dù VM đã chạy xong (mất kết nối), nên `run_tts.sh` giới hạn bước tổng hợp 15 phút;
VieNeu thỉnh thoảng đọc lặp đuôi câu, quét log Whisper tìm cụm lặp rồi lồng tiếng lại riêng clip đó.

**Why:** người dùng cần bộ clip đầy đủ để tự học cả lộ trình; tiết kiệm context nên mỗi trang là một lượt.
**How to apply:** clip mới theo đúng quy trình trong README; cập nhật bảng trạng thái ở đó; đi theo [[feedback-problem-first]];
không chạy model ở laptop ([[feedback-no-local-models]]).
