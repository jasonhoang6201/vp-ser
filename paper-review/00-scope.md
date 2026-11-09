# Phạm vi

- Chủ đề: Emotion-Aware Speech Recognition
- Câu hỏi: Joint ASR + SER. Một model (hoặc một hệ thống train chung) vừa chép lời vừa nhận diện
  cảm xúc. Học chung hai task giúp nhau hay cản nhau, và cách xử lý (multi-task CTC, Whisper/SenseVoice
  sinh token cảm xúc, two-stage training, loss weighting).
- Năm: 2024–2026. Paper trước 2024 chỉ lấy khi là paper nền tảng mà các paper 2024–2026 dựa vào,
  gắn cờ `[nền tảng]`, tối đa 1–2 paper.
- Mục đích: report bài cuối kì môn Voice Processing (thạc sĩ).
- Ngôn ngữ đầu ra (dàn ý, report, chữ trong hình): **tiếng Anh**. Trang HTML để học và giải thích:
  tiếng Việt.

## Tiêu chí bổ sung của người dùng: scope và kết quả có đúng, có thực tế không

Áp dụng cho mọi paper, kể cả venue hạng cao. Mỗi summary ở P2 có mục "Độ tin cậy của kết quả":

- Dữ liệu test: giọng diễn xuất (IEMOCAP, RAVDESS...) hay tự nhiên (MSP-Podcast...). Claim có vượt
  quá dữ liệu đã test không.
- So sánh công bằng: baseline chạy lại cùng split hay lấy số từ paper khác.
- Speaker-independent: người nói ở test có xuất hiện ở train không (IEMOCAP: leave-one-session-out?).
- Mức cải thiện so với dao động: số lần chạy, độ lệch chuẩn, kiểm định thống kê.
- Claim trong abstract khớp với bảng kết quả không (ví dụ "cải thiện cả ASR và SER" nhưng WER tăng).
- Metric: UA/WA/macro-F1, WER/CER, có báo đủ cả hai phía ASR và SER không.

## Venue

| Venue | Loại | Hạng | Nguồn tra hạng | Ghi chú |
|---|---|---|---|---|
| Interspeech | conference | ICORE2026 A | portal.core.edu.au | venue chính của ngành speech |
| ICASSP | conference | ICORE2026 "Multiconference" (không xếp hạng riêng); CORE2023 B | portal.core.edu.au/conf-ranks/2042 | giữ lại theo quyết định của người dùng (2026-10-26) |
| IEEE SLT | workshop | không có trong ICORE | portal.core.edu.au | giữ lại theo quyết định của người dùng |
| ACL | conference | ICORE2026 A* | portal.core.edu.au | ghi rõ main / Findings |
| EMNLP | conference | ICORE2026 A* | portal.core.edu.au | ghi rõ main / Findings |
| NAACL, EACL | conference | ICORE2026 A | portal.core.edu.au | |
| ICLR, NeurIPS, AAAI | conference | ICORE2026 A* | portal.core.edu.au | |
| ACM Multimedia | conference | ICORE2026 A* | portal.core.edu.au | |
| ICME | conference | ICORE2026 A | portal.core.edu.au | |
| ACII | conference | ICORE2026 B | portal.core.edu.au | venue chính của affective computing; chỉ lấy nếu paper rất sát chủ đề |
| IEEE/ACM TASLP | journal | Q1 (Signal Processing; Acoustics and Ultrasonics; Speech and Hearing), SJR 2024 1.061 | resurchify, scijournal (SCImago chặn truy cập tự động) [cần kiểm tra trên scimagojr.com] | |
| IEEE Trans. Affective Computing | journal | Q1 (Human-Computer Interaction; Software), SJR 2024 2.136 | resurchify, scijournal [cần kiểm tra trên scimagojr.com] | |
| Speech Communication | journal | Q1 (Linguistics and Language), Q2 ở các category khác | resurchify [cần kiểm tra] | lấy nếu thiếu paper |
| arXiv | preprint | - | - | gắn `[arXiv]` sau khi kiểm chưa được nhận ở venue nào |
