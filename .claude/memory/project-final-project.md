---
name: project-final-project
description: Đề tài cuối kì là Emotion-Aware Speech Recognition; trạng thái và các hướng đề xuất
metadata:
  type: project
---

Đề tài: "Emotion-Aware Speech Recognition" (bài cuối kì Voice Processing, thạc sĩ).

Trạng thái 2026-10-24: đã research tổng quan, tạo `overview.html`, `slides.html` (root project)
và `.claude/research/papers.md`. Chưa chọn hướng cụ thể.

Các hướng đã đề xuất (chi tiết trong overview.html, mục "Hướng bạn có thể làm"):
H1 đo bias WER theo cảm xúc; H2 multi-task CTC + emotion head; H3 cascade Whisper + emotion2vec
+ gated fusion; H4 fine-tune Whisper sinh token cảm xúc; H5 LLM post-ASR. Gợi ý chính: H1 + H2/H4,
có thể làm trên tiếng Việt (VLSP 2025 ASR-SER).

Cập nhật 2026-10-26: chạy skill research-paper, thư mục `paper-review/` (chỉ một chủ đề nên không có folder con theo chủ đề).
Chốt: nhánh joint ASR + SER, paper 2024–2026, report viết tiếng Anh, giữ ICASSP/SLT.
Người dùng yêu cầu kiểm kỹ scope/kết quả paper có thực tế không. Đang ở DỪNG 1 (chọn paper).
Ràng buộc quan trọng nhất: phải reproduce được (train + ra kết quả) trên Kaggle T4x2 hoặc Colab Pro;
ưu tiên paper có code công khai và model nhỏ (ENT, SenseVoice-Small fine-tune).

Cập nhật 2026-11-01: bắt đầu thực nghiệm trên Colab Pro (có A100). Notebook ở `notebooks/` (xem README ở đó),
trạng thái chi tiết ghi trong `paper-review/progress.md`. Dataset trước mắt: MELD; IEMOCAP
bản HF dùng được nếu spot-check ổn; người dùng đi xin license IEMOCAP chính thức. Bước kế: người dùng chạy
baseline, gửi kết quả, rồi thiết kế model kết hợp dựa trên những gì baseline cho thấy.

**Why:** cần một hướng khả thi trong phạm vi bài cuối kì nhưng vẫn có điểm mới.
**How to apply:** khi người dùng chọn hướng, cập nhật file này với hướng đã chọn, dataset, deadline (nếu được nói).
Liên quan: [[reference-research-kb]].
