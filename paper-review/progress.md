# Progress: Emotion-Aware Speech Recognition

| Bước | Trạng thái | Ngày |
|---|---|---|
| P0 Chốt phạm vi | xong | 2026-10-26 |
| P1 Tìm paper | xong (13 paper, đủ PDF) | 2026-10-26 |
| P2 Summary | xong, `01-papers.html` | 2026-10-26 |
| DỪNG 1 | đang chờ người dùng chọn paper | |

Bước kế: người dùng tick chọn khoảng 5 paper trên `01-papers.html`, rồi làm P3 (roadmap) cho từng paper đã chọn.

## Thực nghiệm Colab (2026-11-01)

Notebook ở `notebooks/` (root project), nguồn `notebooks/src/*.py`, code đánh giá chung `notebooks/eaasr/common.py`.
Quyết định của người dùng: test trước trên MELD; IEMOCAP bản HF `AbstractTTS/IEMOCAP` được dùng nếu kiểm tra
ngẫu nhiên thấy ổn (notebook 01), song song điền form USC; có A100 (Colab Pro) nên thử cả audio-LLM;
model kết hợp không bắt buộc theo paper, rút ra từ baseline rồi test giả thuyết.
Baseline: SenseVoice-Small (10), multi-task CTC kiểu Bansal (11), cascade Whisper + emotion2vec + RoBERTa
fusion kiểu li-2024 (12), ENT code gốc (13), Qwen2-Audio zero-shot (14). Chưa chạy, chờ người dùng gửi kết quả.
Phát hiện khi đọc code ENT: không có dev, báo max qua các epoch trên test (chọn model trên test).

## Quyết định của người dùng

- P0 (2026-10-26): nhánh joint ASR + SER; output tiếng Anh; giữ ICASSP/SLT; năm 2024–2026.
  Yêu cầu thêm: xem kĩ scope và kết quả của paper có đúng, có thực tế không (xem tiêu chí trong `00-scope.md`).
  Ở P2, phần này nằm trong mục "AI đề xuất đọc sâu" của mỗi paper, bắt đầu bằng "Độ tin cậy:".
- 2026-10-26: ràng buộc mới, quan trọng nhất: phải reproduce được (train model, ra kết quả) trên
  Kaggle T4x2 hoặc Colab Pro. Đánh giá khả thi: ENT (có code, wav2vec2-base đóng băng) và
  SenseVoice-Small fine-tune (FunASR, jsonl có sẵn trường emo_target + target) là khả thi;
  Bansal không có code chính thức (tự dựng được); AdaLTM, chen-2025, C-Gate, OSUM, HarmoniFuse quá nặng
  hoặc cần dữ liệu khó lấy.
- Người dùng chưa có nền tảng speech: giải thích từng câu hỏi một, đi chậm.
- 2026-10-28: đang học nền tảng trước khi chốt DỪNG 1. Đã giải thích ASR, WER, SER, WA/UA, Mel, MFCC (trong chat).
  Người dùng chưa hiểu spectrogram nên đã làm `demo/01-spectrogram.html` (8 bước, có tương tác). Kế tiếp: Mel, rồi HMM.
- 2026-10-29: đã hiểu sample rate (16 kHz), Hz, Nyquist, chu kỳ từ dây thanh (f0 = trầm/cao, hình dạng chu kỳ = chữ),
  âm hữu thanh và vô thanh. Từng nhầm: chia 16000/4 thay vì 16000/400.
- 2026-10-30: đã hiểu FFT cả câu mất thứ tự (ví dụ ly sinh tố), spectrogram giữ thời gian nhờ cắt frame,
  đánh đổi độ dài frame (biết "lúc nào" và biết "cao bao nhiêu"). Từng nhầm: tưởng overlap lớn cứu được frame dài,
  tưởng cắt frame là để lấy pha. Sẵn sàng sang Mel.
  Chưa rõ phân biệt f0 và formant trên spectrogram nên đã làm `demo/02-f0-formant.html` (lược × khuôn, 5 mục).
  Vướng tiếp: "rung 300 lần/giây sao lại có 600 Hz", tưởng rung mạnh = tần số cao rồi tắt dần = tần số thấp.
  Đã làm `demo/03-cong-dan.html`: cộng dần họa âm để dựng lại luồng hơi dây thanh, thử sóng lẻ 450 Hz.
  Người dùng không thích chữ "phụt", dùng "luồng hơi".
- 2026-10-31: đã hiểu quy tắc bội số (f ÷ f0 nguyên), FFT không biết trước f0, âm không lặp lại thì có mọi tần số.
  Từng nhầm: tưởng ước số (100 trong giọng 300) cũng có vạch; tưởng 750 là bội của 300. Kế tiếp: Mel.
  Đã chốt lộ trình Mel trước rồi MFCC. Đã hiểu: mỗi frame FFT thành một cột (dọc là tần số, màu là cường độ),
  frame chồng nhau 15 ms nhưng cột vẽ rộng 10 ms đặt sát nhau, 1 giây khoảng 100 cột.
  Từng nhầm: tưởng FFT chỉ ra "tần số cơ bản" (thật ra ra mọi tần số 0–8000 Hz).
  Đã hiểu "phổ" (spectrum) và nguồn gốc chữ 譜. Đã làm `demo/04-mel.html` (4 mục: nghe cặp âm cách 100 Hz,
  thang Mel, Mel filterbank trên một frame, Mel spectrogram). Đang chờ người dùng đọc trang này.
  Đã hiểu: trục tần số chia đều 257 bin là thước đo (số lẻ như 238 Hz vẫn đo được), tai so theo tỉ lệ,
  Mel để gọn, đỡ nhiễu, giữ chi tiết vùng thấp. Từng nhầm: tưởng "không đều" là do giọng phát tần số lẻ.
  Từng nhầm: tưởng Mel giữ nguyên lượng thông tin, hoặc Mel bỏ vùng không nghe được (thật ra 257 thành 80 số,
  vùng thấp gần giữ nguyên, vùng cao gộp khoảng 8 bin thành 1, không bỏ vùng nào).
  Từng nhầm: tưởng từ Mel dựng lại đúng được 257 bin (thật ra lossy, chỉ đoán gần đúng, nhờ họa âm ở âm hữu thanh).
  Đã hiểu thang Mel (đơn vị mel thay Hz, bộ lọc Mel đóng vai "ngăn" thay bin). Từng nhầm: coi 1000 Hz là ranh giới
  cứng, coi vùng dưới 1000 Hz là "vùng của chữ". Kế tiếp: mục 3 (filterbank), mục 4, rồi MFCC.
  Chưa hiểu bộ lọc tam giác khi chỉ có chữ, nên đã thêm mục 3 mới vào `04-mel.html` (ví dụ nhỏ: tổng có trọng số,
  chữ nhật so với tam giác khi họa âm dịch từng bin). Mục cũ thành 4 và 5.
  Đã hiểu bộ lọc tam giác (1 bộ lọc Mel trùm nhiều mel, ra 1 số; gộp thay vì bỏ bớt bin để giữ năng lượng, ổn định,
  đổi mượt). Từng nhầm: "1 filter = 1 mel". Xong Mel spectrogram. Kế tiếp: MFCC.
  Đã làm `demo/05-mfcc.html` (5 mục: log nén khoảng giá trị, log đổi nhân thành cộng, DCT cộng dần sóng cosin,
  giữ 13 hệ số so 2 âm × 2 giọng, ma trận MFCC và dựng lại từ 13 hệ số). Đang chờ người dùng đọc.
  Từng nhầm: tưởng Mel "ra" dB (thật ra Mel chia trục tần số, dB là giá trị mỗi ô).
  Người dùng góp ý trang 05 sai thứ tự (vào thẳng log, DCT mà chưa biết MFCC giải quyết gì). Đã thêm mục 1 (vấn đề:
  40 số Mel trộn lược và khuôn, cùng "a" khác giọng lệch 59, khác chữ lệch 92) và mục 2 (3 bước, mỗi bước gỡ vướng gì,
  theo nguyên lý gì). Các mục cũ thành 3 đến 7. Cách dạy về sau: vấn đề → mục tiêu → thành phần và nguyên lý → chi tiết.
- 2026-11-02: người dùng xin sinh một lần toàn bộ lộ trình. Đã làm `demo/00-lo-trinh.html` (bản đồ 3 cột) và 06 đến 17:
  GMM, HMM, hệ ASR cổ điển (lexicon, LM, WER), DNN-HMM, CTC, attention/Transformer/Conformer, wav2vec 2.0/HuBERT/Whisper,
  TTS và vocoder (pha, Griffin-Lim, HiFi-GAN), Tacotron 2, FastSpeech 2, VITS, XTTS ("XTS" người dùng nói là XTTS).
  CSS/JS chung ở `demo/assets/learn.css`, `learn.js` (trang 01 đến 05 vẫn inline, chưa có thanh điều hướng).
  Đã chạy thử JS bằng stub DOM và chụp headless Chrome vài trang; chưa bấm thử tương tác trong trình duyệt thật.
  Người dùng vẫn đang ở trang 05 MFCC; còn nợ câu hỏi "30 000 thì bao nhiêu dB".
  Người dùng hỏi đã đủ kiến thức tới model SER chưa: chưa. Đã làm nhánh SER 18–24 (SER cơ bản, frame → câu,
  BERT/RoBERTa, fusion + ngữ cảnh, MTL/SenseVoice/ENT, audio-LLM, bẫy đánh giá), mỗi trang có ô "Gắn với thực nghiệm"
  lấy số từ notebooks/PROGRESS.md. Lộ trình 00 thành 4 cột. Thứ tự đề xuất cho đề tài: 01–05, 10–12, 18–24 (06–09 nền, 13–17 tùy chọn).
  Người dùng xin clip video cho từng phần MFCC (giọng VieNeu-TTS qua Colab CLI, làm từng clip, compact giữa các clip).
  Pipeline ở `video-pipeline/` tại gốc project (README có quy trình và trạng thái), mp4 ở `video/<trang demo>/`. Clip 1 (vấn đề) xong. Kế tiếp: clip 2 (log).
