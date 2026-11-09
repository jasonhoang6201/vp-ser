# Danh sách paper (n = 13)

Mọi paper đều đã có PDF (không có `[TẢI TAY]`). Venue và hạng đã tra trên portal.core.edu.au (ICORE2026) và trang arXiv (mục Comments).

| # | id | Title | Venue + năm | Hạng | Cờ | Hướng | Vì sao chọn | PDF |
|---|---|---|---|---|---|---|---|---|
| 1 | shen-2024-ent | Emotion Neural Transducer for Fine-Grained SER | ICASSP 2024 | CORE2023 B | [hạng B] | Joint ASR+SER trong transducer | Một model ra cả transcript lẫn cảm xúc, báo cả WER và WA | pdf/shen-2024-ent.pdf |
| 2 | lee-2026-adaltm | AdaLTM: Adaptive Layer-wise Task Vector Merging | Interspeech 2026 | A | | Model merging thay multi-task | Đối mặt trực tiếp với xung đột ASR-SER | pdf/lee-2026-adaltm.pdf |
| 3 | an-2024-sensevoice | FunAudioLLM (SenseVoice) | arXiv 2024 | - | [arXiv] | Model hợp nhất ASR+SER+LID+AED | Gần sản phẩm nhất, mở mã nguồn | pdf/an-2024-sensevoice.pdf |
| 4 | chen-2025-descriptors | LLM-Empowered Fine-Grained Speech Descriptors | Interspeech 2025 | A | | LLM sinh transcript + mô tả + nhãn | Có kiểm định thống kê, LOSO | pdf/chen-2025-descriptors.pdf |
| 5 | bansal-2023-joint | Efficacy and Noise-Robustness of Jointly Learned SER and ASR | Interspeech 2023 | A | [nền tảng] | Multi-task cổ điển + nhiễu | Baseline joint dễ tái hiện (SpeechBrain) | pdf/bansal-2023-joint.pdf |
| 6 | hsu-2026-cgate | Is Text All You Need? (C-Gate) | NeurIPS 2026 | A* | | Speech-LLM một decoder cho ASR+SER | Venue cao, mới nhất | pdf/hsu-2026-cgate.pdf |
| 7 | si-2025-harmonifuse | HarmoniFuse | arXiv 2025 (nộp ICASSP 2026, chưa xác nhận) | - | [arXiv] | Layer fusion theo task | Đúng bài toán xung đột trong speech LM | pdf/si-2025-harmonifuse.pdf |
| 8 | geng-2025-osum | OSUM | arXiv 2025 | - | [arXiv] | Speech-LLM, train ASR+X | Ý tưởng bắt xuất transcript khi làm SER | pdf/geng-2025-osum.pdf |
| 9 | wang-2025-mtldff | SER with Multi-Task Learning and Dynamic Feature Fusion | Interspeech 2025 | A | | MTL 4 task trên giọng tự nhiên | Bằng chứng ASR là task phụ có ích nhất | pdf/wang-2025-mtldff.pdf |
| 10 | zhang-2025-beyond | Beyond Classification: Speech Emotion Reasoning | IJCNLP-AACL 2025 | B | [hạng B] | AudioLLM giải thích cảm xúc | Đại diện hướng audio-LLM | pdf/zhang-2025-beyond.pdf |
| 11 | li-2024-werfusion | SER with ASR Transcripts: WER and Fusion | IEEE SLT 2024 | không xếp hạng | [workshop] | Cascade: WER ảnh hưởng SER | Related work về lỗi ASR | pdf/li-2024-werfusion.pdf |
| 12 | li-2025-r3 | Revise, Reason, and Recognize | ICASSP 2025 | CORE2023 B | [hạng B] | LLM sửa lỗi ASR rồi đoán cảm xúc | Related work, có đo độ nhạy prompt | pdf/li-2025-r3.pdf |
| 13 | wu-2025-affectiveasr | ASR for Affective Speech | IEEE ASRU 2025 | C | [hạng C] | ASR bền với giọng cảm xúc | Đại diện nhánh "cảm xúc làm ASR sai" | pdf/wu-2025-affectiveasr.pdf |

## Đề xuất 5 paper
1. shen-2024-ent: đúng trọng tâm joint, đo cả hai phía, có ablation, có code.
2. lee-2026-adaltm: venue A, nói thẳng xung đột ASR-SER; hợp để luyện đọc phản biện (claim mạnh hơn bảng số).
3. an-2024-sensevoice: model hợp nhất gần sản phẩm, dùng được cho demo.
4. chen-2025-descriptors: hướng LLM, có kiểm định thống kê.
5. bansal-2023-joint `[nền tảng]`: baseline multi-task đơn giản, leave-one-speaker-out, có test nhiễu.

## Paper bỏ ra và lý do
- Speech Emotion Recognition with ASR Integration (Li, arXiv 2601.17901, luận án PhD): rất hữu ích để đọc tổng quan, nhưng là luận án dài, không phải một paper; dùng làm tài liệu tham khảo.
- MTLSER (Expert Systems with Applications 2025): sau paywall ScienceDirect, chưa tải được; có thể thêm nếu bạn tải qua tài khoản trường.
- Speech recognition and SER approach for VLSP 2025 (aclanthology 2025.vlsp-1.2): tiếng Việt, hợp nếu làm thực nghiệm tiếng Việt, nhưng VLSP là shared task workshop không xếp hạng.
- Joint MoE Speech Enhancement + Robust ER (TASLP, arXiv 2509.08470): joint với speech enhancement, không phải ASR.
- Zgórzyński et al. (Interspeech 2025): multi-task nhưng là classification + regression cảm xúc, không có ASR.
- Gao et al. Two-stage finetuning (Interspeech 2023): trước 2024; đã có Bansal làm paper nền tảng.

## Ghi chú tìm kiếm
- Semantic Scholar trả lỗi 429 trong suốt phiên; dùng arXiv API, DBLP (cũng bị 429) và WebSearch thay thế.
- SCImago chặn truy cập tự động; hạng journal lấy qua resurchify/scijournal, cần kiểm lại nếu trích journal.
