---
name: project-ablation-plan
description: Kế hoạch ablation study cho hệ cuối (2026-11-14): giữ thí nghiệm nào, cắt gì, cái nào đã có số, cái nào chưa chạy
metadata:
  type: project
---

Brainstorm 2026-11-14. Người dùng muốn chỉ giữ ablation **thay đổi trực tiếp kết quả đầu ra**. Prosody bị bỏ
vì dài mà không tăng điểm rõ. Người dùng **chưa chọn** trong danh sách bên dưới.

**Hệ cuối** (`asr_ctx_whisper`): transcript SenseVoice-FT + 3 câu ngữ cảnh → RoBERTa-large, cộng encoder Whisper
large-v3 làm nhánh audio, gộp hai nhánh bằng gated fusion. UA CV 5 fold 81.67 ± 3.03 (3 seed).

**Nguyên tắc:** mỗi thí nghiệm bỏ hoặc thay đúng một module, mọi thí nghiệm chạy cùng điều kiện 5 fold × 3 seed.
Báo mean ± std, số fold thắng trên 5, McNemar.

**Có cần module không:**
1. Bỏ audio (SV-FT + ngữ cảnh, chỉ text): có 77.44, nhưng fold 1-4 mới 1 seed. Cần thêm seed 1, 2, khoảng 1.7 giờ.
2. Bỏ ngữ cảnh (SV-FT + Whisper audio): chưa chạy, khoảng 3 giờ. Hiện chỉ có cặp WavLM (79.01 so với 74.61).
3. Bỏ text (chỉ encoder Whisper): chưa chạy, cần sửa `src/21_combined.py`, khoảng 1-2 giờ.

**Chọn module có đúng không:**
4. Thay Whisper encoder bằng WavLM: đã đủ, 79.15 ± 3.49, Whisper hơn +2.5, thắng 5/5 fold.
5. Thay transcript SV-FT bằng transcript Whisper (một backbone làm cả ASR lẫn encoder): chưa chạy.
   `cache/iemocap/asr_whisper-large-v3-turbo.csv` mới có 5531 câu 4 lớp. Ngữ cảnh cần cả 10039 câu, nên phải
   transcribe thêm khoảng 4500 câu (khoảng 15 phút), rồi chạy khoảng 3 giờ. Lưu ý: encoder là large-v3, còn ASR là
   large-v3-turbo; nếu làm theo hướng một backbone thì dùng cùng checkpoint. Đây là thí nghiệm trả lời câu hỏi
   "sao không dùng Whisper làm transcript luôn".
6. Thay transcript bằng text đúng (cận trên): bản chỉ text đã có (79.02 so với 77.44, lỗi ASR mất khoảng 1.6);
   bản có audio chưa chạy.

**WER:**
7. So sánh ASR: Whisper turbo 13.2, SV zero-shot 11.2, SV-FT 9.7 (trung bình 5 fold). Đã đủ.

**Cắt hoặc để phụ lục:** prosody, attention pooling, emotion2vec (rò rỉ IEMOCAP), so sánh gated với concat.
MTL-CTC, ENT, Qwen2-Audio và cascade đưa vào bảng baseline, không tính là ablation.

**Gợi ý đã đưa:** bản tối thiểu (khoảng 5 giờ) gồm 1 (thêm seed), 2, 4, 6 (bản chỉ text), 7.
Nếu còn compute unit thì thêm 3 và 5; nên ưu tiên 5.

Lý do chọn SenseVoice-FT làm ASR lúc đầu: WER thấp hơn Whisper turbo; fine-tune rẻ (khoảng 234M tham số, khoảng
38 phút mỗi fold); hướng joint ASR + SER ban đầu. Ở WER khoảng 10-13 thì lỗi ASR chỉ làm mất khoảng 1.6 UA, nên
chưa chắc SV-FT tốt hơn Whisper xét theo SER.

**Cập nhật 2026-11-14 (sau trao đổi):** người dùng muốn trình bày theo 4 bước:
(1) WER zero-shot SenseVoice so với Whisper; (2) WER sau fine-tune (Whisper fine-tune chưa chạy, nếu thiếu thì so sánh
không công bằng); (3) chọn thành phần hệ đề xuất; (4) ablation từng module trên hệ tốt nhất.
Người dùng thêm ablation "ASR không fine-tune", nên nguồn transcript thành bảng 2 × 2 (SenseVoice/Whisper ×
zero-shot/fine-tune), giữ nguyên ngữ cảnh + Whisper encoder + gated fusion. Mới có ô SV-FT (81.67). Hai ô zero-shot cần
transcribe đủ 10039 câu (hiện chỉ có 5531 câu test có nhãn). Gợi ý: chạy hai ô zero-shot trước, rồi mới quyết định có
fine-tune Whisper không. "Model tốt nhất" tách theo vai trò: ASR là SV-FT, encoder audio là Whisper large-v3.
Trong report gọi hệ `combined` là "hệ đề xuất"; "cascade" trong repo là baseline.

**Cập nhật tiếp 2026-11-14:** người dùng cho rằng không cần ablation "ASR không fine-tune", vì so sánh text đúng với
transcript SV-FT (79.02 so với 77.44, chỉ text) đã đo được ảnh hưởng của WER. Đồng ý: khoảng −1.6 là trần cho mọi cải
thiện ASR (đi từ WER 9.7 về 0), còn zero-shot so với fine-tune chỉ chênh khoảng 1.5 WER, nên chênh UA dự kiến nằm
trong nhiễu seed (khoảng 1-1.5 UA). Vì vậy bỏ bảng 2 × 2 nguồn transcript. Điều kiện kèm theo: chạy thêm seed 1, 2 cho
`asr_ctx` và `gt_ctx` ở fold 1-4, để con số −1.6 đủ chắc. Trong report trình bày nó như cận trên. Câu hỏi có fine-tune
Whisper cho bước 2 (WER) hay không vẫn chưa chốt.

**Chốt về seed (2026-11-14):** người dùng chọn chạy 1 seed cho mọi ablation. Trong report không nhắc seed, chỉ ghi là
đã chạy thực nghiệm. Dùng **seed 0** (không phải 42), vì các run có sẵn đều có seed 0, nên tái sử dụng được và không
phải chạy lại. Bảng dùng số seed 0, không dùng ensemble 3 seed: full Whisper 81.06, bỏ audio 77.44, WavLM 79.01,
text đúng + ngữ cảnh 79.02, transcript + WavLM không ngữ cảnh 74.61. Còn phải chạy: bỏ ngữ cảnh (`asr_whisper`,
5 fold, khoảng 1 giờ) và bỏ text (chỉ Whisper encoder, cần sửa code, khoảng 40 phút).

**Chốt cuối (2026-11-14 tối):** người dùng có thêm CU nên cho chạy lại **toàn bộ** với seed 42 (đè quyết định seed 0 ở
trên), kể cả fine-tune SenseVoice, cộng thêm fine-tune Whisper large-v3-turbo; kết quả cũ bị ghi đè. Trình bày theo
4 bước: WER zero-shot → WER fine-tune (SV và Whisper) → chọn encoder audio (Whisper so với WavLM) → ablation (bỏ audio,
bỏ ngữ cảnh, bỏ text, text đúng). Trạng thái chạy ở mục "Đợt chạy lại seed 42" trong `notebooks/PROGRESS.md`.
MELD tạm gác; người dùng hỏi IEMOCAP còn được quan tâm không: đã trả lời là vẫn là chuẩn so sánh, xu hướng challenge
chuyển sang MSP-Podcast (Odyssey 2024, Interspeech 2025), chưa thấy challenge SER 2026.

**Đổi ASR sang Whisper (2026-11-15):** kết quả seed 42 cho WER fine-tune trung bình 5 fold là Whisper turbo FT 9.18,
SV-FT 9.73. Người dùng chốt: làm theo đúng thứ tự, bước nào chọn được model tốt hơn thì các bước sau dùng model đó.
Vì vậy ASR của hệ đề xuất là **Whisper large-v3-turbo FT**, và các bước chọn encoder và ablation chạy lại trên
transcript Whisper. Không dùng lập luận "giữ SenseVoice vì cận trên chỉ 0.54 UA". Kết quả lưu ở `combined_wasr_*`.
`e2v_only` và `gt_ctx` không đọc transcript ASR nên dùng lại. Các kết quả `combined_asr_*` (transcript SV) giữ lại
để tham khảo. Hình kiến trúc và dàn ý phải đổi khối ASR từ SenseVoice-Small sang Whisper large-v3-turbo.

**Chốt report (2026-11-15 tối):** report **chỉ viết về IEMOCAP, không nhắc MELD** (kể cả pilot gradient MELD trong
dàn ý cũ). MELD chỉ là thí nghiệm riêng để người dùng nắm khi vấn đáp: lỗi ASR mất ~0.5 UA ở WER ~10% (IEMOCAP) nhưng
~14 wF1 ở WER ~27% (MELD); Whisper FT bị lặp từ trên MELD nên MELD chọn SenseVoice. Hình kiến trúc: khối ASR là
Whisper large-v3-turbo (đã sửa và kiểm PNG). Không có template hay giới hạn trang; người dùng giao "do best" cho số
bảng và tiêu đề.

**Kết quả cuối (2026-11-16 00:11, transcript Whisper-FT):** WER Whisper FT 8.32 (chạy lại); đầy đủ 81.17 UA;
WavLM 80.24; bỏ audio 77.54; bỏ ngữ cảnh 78.85; bỏ text 72.41; text đúng chỉ text 78.57 (lỗi ASR mất 1.03).
Hệ đầy đủ dùng transcript SV-FT được 81.44 (+0.27, trong nhiễu fold), giữ Whisper theo thứ tự bước; chưa hỏi lại
người dùng. Dàn ý `paper-review/03-outline.md` draft 3 đã điền số.

**Why:** report cần chứng minh từng module đóng góp gì mà không làm loãng bằng các thí nghiệm phụ.
**How to apply:** khi người dùng chốt danh sách, cập nhật file này và `notebooks/PROGRESS.md`, rồi lên chuỗi chạy
(báo ETA). Khi trao đổi, gọi thí nghiệm bằng tên module (ví dụ "bỏ ngữ cảnh"), không dùng nhãn chữ A/B/C.
Liên quan: [[project-final-project]], [[reference-colab-progress]].
