# Clip lộ trình (từ MFCC trở đi)

Mỗi section của các trang `demo/05-mfcc.html` đến `demo/24-danh-gia.html` là một clip (1920×1080, 30 fps, nền sáng,
có phụ đề và giọng đọc). Clip MFCC đặt tên `<số>-<tên>`, clip của trang khác đặt tên `<trang>-<số>-<tên>`.
Làm theo từng trang: viết cả trang, đọc lời một lượt trên Colab, render.

Thư mục này chỉ chứa phần dựng clip. File mp4 nằm ở `../video/<trang demo>/`, ví dụ `../video/06-gmm/gmm-1-van-de.mp4`.
Trong file này, `demo/` là `paper-review/demo/` ở gốc project. Thêm trang mới thì khai báo tiền tố clip trong `PAGES` ở `tools/render.mjs`.

### MFCC (`demo/05-mfcc.html`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| 01-van-de | vấn đề: 40 số Mel trộn lược và khuôn, tương quan, quá nhiều số; mục tiêu và 3 bước | xong |
| 02-log | log: dB, khoảng giá trị, tai nghe theo tỉ lệ, nhân thành cộng, đồng hình, vì sao log sau Mel, floor | xong |
| 03-dct | DCT: sóng cosin, hệ số, vì sao không FFT, dồn năng lượng, khử tương quan (≈ PCA), lịch sử | xong |
| 04-giu-13 | giữ 13 hệ số, kiểm khoảng cách, c0, c1, liftering, cepstrum, đỉnh f0 | xong |
| 05-ca-cau | ma trận MFCC, delta, delta-delta, CMN, CMVN, công thức đầy đủ | xong |
| 06-ngay-nay | DNN, fbank thay MFCC, vì sao DCT có hại, ngày nay dùng gì | xong |

### GMM (`demo/06-gmm.html`, dữ liệu và EM dùng chung: `lib/gmm.js`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| gmm-1-van-de | đám mây điểm, khoảng cách tới trung bình, template, VQ, mục tiêu: địa hình mật độ | xong |
| gmm-2-thanh-phan | đồi Gauss, trộn, EM, likelihood, mỗi âm một GMM, vì sao Gauss (CLT), vì sao trộn, lịch sử | xong |
| gmm-3-gauss | histogram, công thức, diện tích 1, log-likelihood, MLE, covariance, diagonal | xong |
| gmm-4-em | một đồi không đủ, gà và trứng, bước E, bước M, log-likelihood không giảm, chọn K, đỉnh gần, variance floor | xong |
| gmm-5-cham-diem | so likelihood, ranh giới cong, sát ranh giới, rất xa, Bayes, ứng dụng, giới hạn → HMM | xong |

### HMM (`demo/07-hmm.html`, dữ liệu và thuật toán dùng chung: `lib/hmm.js`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| hmm-1-van-de | chấm từng frame nhảy lung tung, ba vấn đề (nhảy, thứ tự, độ dài), DTW, mục tiêu | xong |
| hmm-2-thanh-phan | trạng thái, chuyển, phát (GMM), GMM-HMM, ba thuật toán, giả định Markov, lịch sử | xong |
| hmm-3-tu-lap | vòng tự lặp, lấy mẫu độ dài, 1/(1 − p), phân bố hình học, trái sang phải, 3 trạng thái mỗi âm | xong |
| hmm-4-viterbi | trellis, 741 cách chia, giữ đường tốt nhất, công thức, lần ngược, nhiễu, quy hoạch động, forced alignment | xong |
| hmm-5-forward | Forward (tổng thay max), so các từ, Baum-Welch (forward-backward, hậu nghiệm, bước M), ghép âm vị thành câu | xong |

### Hệ ASR cổ điển (`demo/08-he-asr-co-dien.html`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| asr-1-van-de | bốn vấn đề: âm ít từ nhiều, nghe giống nghĩa khác, đồng cấu âm, cần một con số; mục tiêu | xong |
| asr-2-cong-thuc | decoder, mô hình âm học, từ điển, LM, WER; W* = argmax P(X\|W)P(W); vì sao tách đôi; kênh nhiễu | xong |
| asr-3-triphone | từ điển nối âm vị, monophone, triphone, 64 000, cây quyết định, senone, lịch sử | xong |
| asr-4-lm | n-gram, bigram, làm trơn, trọng số LM (0 / 1 / 3), tập dev, beam search | xong |
| asr-5-wer | so từng vị trí sai, S/D/I, ví dụ 50%, Levenshtein, vượt 100%, âm tiết và CER, hệ 1990–2010 | xong |

### DNN-HMM (`demo/09-dnn-hmm.html`, mạng minh họa dùng chung: `lib/nn.js`)

Số độ chính xác trong lời đọc lấy từ Chrome (`node tools/nncheck.mjs <H> <bước,...>`), không lấy từ Node: hai bên làm
tròn tanh/exp hơi khác, 3000 bước momentum khuếch đại khác biệt đó.

| Clip | Nội dung | Trạng thái |
|---|---|---|
| dnn-1-van-de | giữ hệ, thay GMM; generative/discriminative; 440 chiều; buộc MFCC; ranh giới rối; mục tiêu | xong |
| dnn-2-thanh-phan | đường đi hệ lai, 5 thành phần, hybrid, Bourlard-Morgan, 2010–2012 | xong |
| dnn-3-noron | tổng có trọng số, so bộ lọc Mel, tanh/ReLU, đường thẳng z = 0, xoay và dịch, vì sao cần hàm uốn | xong |
| dnn-4-nhieu-lop | dữ liệu 3 vùng, softmax, cross-entropy, gradient descent, backprop, H = 0/2/8, mạng thật | xong |
| dnn-5-ghep | nhãn từ forced alignment, P(s\|x) và p(x\|s), Bayes, prior, scaled likelihood, giảm 20–30%, giới hạn | xong |

### CTC (`demo/10-ctc.html`, rút gọn, đếm đường, đầu ra "mai": `lib/ctc.js`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| ctc-1-van-de | hệ lai nhiều thành phần, cặp (âm thanh, câu chữ), độ dài lệch, không biết ranh giới, end-to-end | xong |
| ctc-2-thanh-phan | đường đi, trống, rút gọn, loss, giải mã, đơn vị đầu ra, Graves 2006, Deep Speech | xong |
| ctc-3-blank | gộp lặp rồi xóa trống, hello, chữ kéo dài, trống ngăn chữ trùng, khoảng lặng, spike | xong |
| ctc-4-loss | 15 chuỗi cho "ba", tổng như Forward, số chuỗi tăng nhanh, lưới − b − a −, nhảy qua trống, "aa", loss | xong |
| ctc-5-giai-ma | đầu ra 30 frame, greedy, "nmai", "maaim", độc lập có điều kiện, beam search + LM, sang attention | xong |

### Attention, Transformer, Conformer (`demo/11-attention.html`, key, ma trận "xin chào", đầu minh họa: `lib/attn.js`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| att-1-van-de | CTC đoán độc lập, không biết chữ đã viết, LM ngoài, RNN chậm; mục tiêu; ba mốc 2015/2017/2020 | xong |
| att-2-thanh-phan | encoder, decoder tự hồi quy, attention, self-attention; Bahdanau, Chorowski, LAS | xong |
| att-3-qkv | key/value/query, tích vô hướng, softmax, đầu ra, xoay query, độ dài query, √d | xong |
| att-4-decoder | "xin chào" từng ký tự, ma trận đường chéo, <eos>, teacher forcing, beam, bỏ qua/lặp, CTC/attention | xong |
| att-5-transformer | RNN tuần tự, self-attention song song, multi-head, khối Transformer, vị trí, chi phí bình phương, Conformer, RNN-T | xong |

### wav2vec 2.0, HuBERT, Whisper (`demo/12-ssl-whisper.html`, 60 frame "a s i" và k-means: `lib/ssl.js`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| ssl-1-van-de | chép lời đắt, tiếng Việt ít dữ liệu, âm thanh không nhãn nhiều; tiền huấn luyện rồi tinh chỉnh | xong |
| ssl-2-hai-huong | SSL che rồi đoán, giám sát yếu, BERT, đơn vị rời rạc, ba cách lấy mã | xong |
| ssl-3-wav2vec | CNN 20 ms, lượng tử hóa, che một nửa, Transformer, chọn 1/101, 53 000 giờ + 10 phút: 4,8/8,2 | xong |
| ssl-4-hubert | k-means trên "a s i" (K = 3/5/8), hai vòng 100/500 cụm, Hidden-unit BERT, WavLM | xong |
| ssl-5-whisper | encoder-decoder, 680 000 giờ, log-Mel 80 kênh 30 giây, token việc, lọc nhãn, hallucination, so sánh | xong |

### TTS và vocoder (`demo/13-tts-vocoder.html`, âm "a" giả lập, pha, Griffin-Lim: `lib/tts.js`)

Số pha và Griffin-Lim lấy từ Chrome: `node tools/chromeval.mjs gmm,dsp,tts "<biểu thức>"`.

| Clip | Nội dung | Trạng thái |
|---|---|---|
| tts-1-van-de | đi ngược nhận dạng, ghép đoạn, HMM, một chữ nhiều cách đọc, sóng dài, chữ ≠ cách đọc | xong |
| tts-2-khung | xử lý văn bản (chuẩn hóa, G2P, thanh), mô hình âm học, vocoder, vì sao qua Mel, cái giá | xong |
| tts-3-pha | độ lớn và pha, dựng lại với pha thật / 0 / ngẫu nhiên, overlap-add | xong |
| tts-4-griffin-lim | ba bước lặp, sai lệch 61% → 23% → 9%, Mel còn tệ hơn, Tacotron bản đầu | xong |
| tts-5-vocoder | WaveNet (giãn nở), WaveGlow, HiFi-GAN (generator, discriminator, chu kỳ 2–11) | xong |
| tts-6-do | MOS, CER qua ASR, speaker embedding, tóm tắt trang | xong |

### Tacotron 2 (`demo/14-tacotron2.html`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| taco-1-van-de | chuỗi mô hình rời, đặc trưng chuyên gia, giọng phẳng (trung bình), mục tiêu một mạng, mượn encoder-decoder | xong |
| taco-2-khoi | encoder (conv + BiLSTM), attention theo vị trí, decoder tự hồi quy, pre-net, post-net, WaveNet, MOS 4,53 | xong |
| taco-3-buoc | một bước decoder: pre-net, attention, trung bình có trọng số, LSTM, dừng; teacher forcing, exposure bias, dropout pre-net | xong |
| taco-4-attention | "xin chào": attention tốt, lặp chữ, bỏ chữ, không dừng; sinh tuần tự chậm; tóm tắt trang | xong |

### FastSpeech 2 (`demo/15-fastspeech2.html`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| fs-1-van-de | Tacotron chậm, hay hỏng, khó điều khiển; tách độ dài ra; FastSpeech 2019/2021; giống HMM | xong |
| fs-2-cach-lam | encoder/decoder Transformer, variance adaptor, dự đoán độ dài, length regulator, f0 và năng lượng, MFA, tốc độ | xong |
| fs-3-do-dai | "xin chào" 6 âm vị, chép vector theo độ dài, 71 frame, không attention, tốc độ × 0,7, nhấn một âm | xong |
| fs-4-cao-do | đường f0 và năng lượng, thanh điệu, đổi thanh, f0 phẳng, dịch cao độ, huấn luyện/dùng, đoán trung bình, cần vocoder | xong |

### VITS (`demo/16-vits.html`, MAS và độ dài ngẫu nhiên: `lib/vits.js`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| vits-1-van-de | hai giai đoạn lệch nhau, Mel là nút thắt, cần MFA, đọc y hệt; mục tiêu; tên đầy đủ của VITS | xong |
| vits-2-manh | VAE, prior + flow, MAS, độ dài ngẫu nhiên, GAN; MOS 4,43 / 4,46 | xong |
| vits-3-vae | autoencoder, z là phân phối, lúc học (KL), lúc dùng (flow ngược), vì sao cần flow, một chữ nhiều cách đọc | xong |
| vits-4-mas | lưới điểm 6 × 56, ràng buộc đơn điệu, công thức Q, độ dài 7/10/6/7/14/12, nhiễu 2 → lệch 14, học dần, Viterbi | xong |
| vits-5-ngau-nhien | log-normal minh họa, mức 0 / 0,8 / 1,6, ba tham số noise_scale, tóm tắt trang, khi nào dùng VITS | xong |

### XTTS (`demo/17-xtts.html`, lượng tử hóa và GPT 8 mã: `lib/xtts.js`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| xtts-1-van-de | mỗi giọng một mô hình, giọng lạ, đa ngôn ngữ, dữ liệu nhiễu; GPT viết tiếp; zero-shot; Tortoise, VALL-E | xong |
| xtts-2-khoi | VQ-VAE, Perceiver Resampler, GPT-2, decoder HiFi-GAN; học cùng người; 16 ngôn ngữ + Hindi | xong |
| xtts-3-vq | 160 frame 2 chiều, K = 2 / 8 / 64 (sai lệch 1,49 / 0,23 / 0,015), đánh đổi, 325 vs 256 000 bit/giây | xong |
| xtts-4-nhiet-do | phân phối sau mã 0 (0,54 / 0,25), nhiệt độ 0,2 / 5, sinh ở 0,1 / 3 / 0,8 (8 / 35 / 25 kiểu cặp), top-k, lỗi tự hồi quy | xong |
| xtts-5-nhan-ban | bốn bước nhân bản, phụ thuộc mẫu, accent, đạo đức, bảng bốn thế hệ TTS | xong |

### SER cơ bản (`demo/18-ser-co-ban.html`, số câu MELD, bảng nhầm lẫn, mặt phẳng valence-arousal: `lib/ser.js`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| ser-1-van-de | nói gì / cảm xúc gì, "Thật à?" ba cách nói, bốn vấn đề, mục tiêu trang | xong |
| ser-2-manh | kiểu nhãn, dấu hiệu, dữ liệu, thước đo; ghép lại | xong |
| ser-3-nhan | mặt phẳng valence × arousal, bảng ngữ điệu, arousal tốt, valence cần từ ngữ, tham số điều khiển trang 15 | xong |
| ser-4-du-lieu | IEMOCAP 5 phiên, 4 lớp 5 531 câu, chia theo phiên; MELD 7 lớp, test 2 610 câu; bỏ câu hòa phiếu | xong |
| ser-5-do | bảng nhầm lẫn, WA/UA 50 → 48,9/28,6 → 48,1/14,3, macro/weighted-F1, số liệu trang 18, class weight | xong |

### Từ frame đến câu (`demo/19-frame-den-cau.html`, bằng chứng giận, pooling, trọng số tầng: `lib/pool.js`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| pool-1-van-de | 150 vector một câu, cảm xúc cục bộ, 12/24 tầng, encoder to; mục tiêu | xong |
| pool-2-cong-cu | đường đi, pooling, trộn tầng, probe/fine-tune, emotion2vec | xong |
| pool-3-pooling | 75 frame, mean 0,25 / 0,43, attention 0,75 / 0,91, đoạn 6 frame, statistics pooling | xong |
| pool-4-tang | công thức SUPERB, trọng số ASR / người nói / SER, tầng cuối, LayerNorm | xong |
| pool-5-probe | probe, fine-tune, vài tầng trên, LoRA 16 384 vs 1 048 576 | xong |
| pool-6-e2v | thầy-trò EMA, che, loss frame + câu, khác wav2vec/HuBERT, probe tuyến tính, emotion2vec+, tóm tắt trang | xong |

### SER từ chữ (`demo/20-text-bert.html`, bộ đoán từ khóa và WER: `lib/bert.js`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| bert-1-van-de | giọng giận ≈ vui, tra từ khóa, phủ định / mỉa mai, dữ liệu ít, chữ ASR có lỗi; mục tiêu | xong |
| bert-2-cach-lam | tokenizer mảnh từ, Transformer encoder, MLM 15%, token đầu + fine-tune; BERT, RoBERTa | xong |
| bert-3-loi-asr | ba ví dụ (WER 20 / 20 / 17%: không đổi / giận → trung tính / mất "not"), cách giảm hại, tóm tắt trang | xong |

### Ghép âm thanh và chữ (`demo/21-fusion.html`, ba tình huống, ghép muộn / cổng, ngữ cảnh: `lib/fus.js`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| fus-1-van-de | hai vector ghép thế nào, bảng bốn tình huống (mỉa mai, ASR sai, giận ≈ vui, "Thật à?"), mục tiêu | xong |
| fus-2-kieu | ghép sớm, ghép muộn, cổng, cross-attention, ngữ cảnh hội thoại | xong |
| fus-3-cong | mỉa mai: muộn sai, cổng 0,8 đúng; ASR sai: hòa 40/40, cổng 55%; đồng ý; cổng cần dữ liệu; g trung bình | xong |
| fus-4-cross | "great" làm query trên 40 frame, cùng công thức trang 11, hai chiều, chồng lớp, cái giá | xong |
| fus-5-ngu-canh | "Thật à?" với bốn câu trước, nối vào RoBERTa, thứ tự câu, rò rỉ nhãn, tóm tắt trang | xong |

### Học ASR và SER cùng lúc (`demo/22-multitask.html`, hai loss bậc hai, PCGrad: `lib/mtl.js`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| mtl-1-van-de | hai hệ rời vs một mô hình, ba vấn đề nối tiếp, hy vọng, nguy cơ, mục tiêu | xong |
| mtl-2-thanh-phan | encoder chung + CTC + pooling, loss α, xung đột gradient, cách giải quyết, token chỉ việc, ENT | xong |
| mtl-3-gradient | mặt phẳng tham số, cosine 0,45 → −1, dừng 1,59 / 1,00, α = 1: SER 4,45 → 6,74, PCGrad, mô hình thật ≈ vuông góc | xong |
| mtl-4-sensevoice | 4 token LID/SER/AED/ITN, CTC song song, so Whisper, một token cảm xúc, file fine-tune | xong |
| mtl-5-ent | RNN-T, lưới căn, ưu điểm, ENT joint cảm xúc theo bước, tóm tắt trang | xong |

### Audio-LLM (`demo/23-audio-llm.html`, 8 câu trả lời và luật đọc, số token: `lib/llm.js`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| llm-1-van-de | hỏi bằng lời, ba vấn đề (đầu ra mỗi việc, không lý do, không hiểu thế giới), LLM chưa nghe được âm thanh, mục tiêu | xong |
| llm-2-cach-lam | encoder Whisper, bộ chiếu, Qwen-7B (8,2 tỉ); tiền huấn luyện, SFT, DPO | xong |
| llm-3-token | 25 token/giây, 4 giây ≈ 140, 30 giây ≈ 790, bình phương ≈ 32 lần, tự hồi quy, 16 GB | xong |
| llm-4-doc | 8 câu trả lời, luật đọc 1 → 3 → 5 → 6/8, hai câu không cứu được, câu dẫn khi chép lời, rủi ro, tóm tắt trang | xong |

### Đánh giá cho đúng (`demo/24-danh-gia.html`, mô phỏng chọn epoch, khoảng tin cậy, WER: `lib/evalx.js`)

| Clip | Nội dung | Trạng thái |
|---|---|---|
| eval-1-van-de | một mô hình nhiều con số, sáu sai lầm, mục tiêu | xong |
| eval-2-quy-tac | sáu quy tắc: tách tập, theo người nói, nhiều seed, đủ thước đo, rò rỉ, báo cách làm | xong |
| eval-3-chon-epoch | UA thật 30, một lần chạy max 31,6; trung bình 30,0 / 30,0 / 31,8; 30 epoch 32,4; lệch 2,5 → 33,8 | xong |
| eval-4-tin-cay | công thức khoảng tin cậy, 1 200 câu ±2,5, n = 300 / 5 000, UA, seed, McNemar, bootstrap | xong |
| eval-5-ro-ri | rò rỉ, hai dấu hiệu, không chắc thì ghi chú, mô hình lớn | xong |
| eval-6-wer | "Mr. Smith…": WER 71% → 0% sau chuẩn hóa, Whisper normalizer, quy tắc, tóm tắt trang, hết lộ trình | xong |

## Quy trình một clip

1. Viết `narration/<clip>.json`: mỗi đoạn có `id`, `sub` (phụ đề, viết thuật ngữ đúng) và `say` (câu cho TTS đọc,
   thuật ngữ viết theo cách đọc: "em ép xi xi", "ép ép ti", "héc", số viết bằng chữ).
   Gọi khái niệm bằng thuật ngữ, không bằng hình ảnh ví von ("Gaussian", không "quả đồi"; "tham số điều khiển", không
   "núm vặn"); ví von chỉ nằm trong một câu riêng mở đầu "Ví von:". Dùng "vấn đề" hoặc "hạn chế", không dùng "vướng".
2. Viết `scenes/<clip>.html` (khung chung: `lib/ui.js`): mỗi shot vẽ theo thời gian tuyệt đối, neo vào đoạn lời bằng `S(id)`, `D(id)`.
   Xem bố cục trước khi có tiếng: `node tools/render.mjs <clip> --fake-dur --stills id1,id2`.
3. Lồng tiếng trên Colab CPU (session `tts`, VieNeu-TTS v3 Turbo, giọng Ngọc Huyền):
   `zsh tools/run_tts.sh <clip>`. Ra `audio/<clip>/*.wav` và `durations.json`. Nhiều clip thì chạy lần lượt trong cùng
   session (lần sau cài lại rất nhanh). Kiểm lời đọc bằng Whisper: upload `tools/asr_check.py` rồi chạy với tên các clip.
   Tạo session nếu chưa có: `venv/bin/colab new -s tts` (ở thư mục gốc project, không thêm `--gpu`).
4. Render: `node tools/render.mjs <clip>` ra `../video/<trang demo>/<clip>.mp4`. Xem thử theo thời gian thực: mở `scenes/<clip>.html`, bấm Space.

Notebook `tts_vieneu.ipynb` chạy được trực tiếp trên Colab (upload file json vào `/content/tts/`).
Mẫu các giọng khác ở `audio/voices/`; đổi `VOICE` trong notebook nếu muốn giọng khác.
