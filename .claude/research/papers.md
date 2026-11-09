# Emotion-Aware Speech Recognition: kho tài liệu tham khảo

Cập nhật: 2026-10-24. Là ghi chú research cho bài cuối kì môn Voice Processing (thạc sĩ).
Trang overview dạng HTML: `overview.html`, bản slide: `slides.html` (cả hai ở root project).

Quy ước: [đã kiểm tra] = đã đọc abstract/trang chính thức trong lúc research.
[cần kiểm tra] = thông tin từ nguồn thứ cấp hoặc chưa đọc full paper, nên mở paper trước khi trích dẫn.

---

## 1. Định nghĩa nhanh

"Emotion-aware ASR" không phải một task chuẩn hóa duy nhất. Trong tài liệu nó bao gồm 3 nhóm:

1. **ASR bền với giọng có cảm xúc**: giọng tức giận, vui, buồn làm WER tăng; tìm cách giảm.
2. **ASR + SER cùng lúc (joint / multi-task)**: một model vừa chép lời vừa nhận diện cảm xúc,
   hai task hỗ trợ nhau.
3. **SER dựa trên output của ASR**: dùng transcript (có lỗi) + âm học để đoán cảm xúc,
   nghiên cứu ảnh hưởng của WER lên SER.

Gần đây thêm nhóm 4: **Audio-LLM** sinh ra transcript + nhãn cảm xúc + lời giải thích (reasoning).

---

## 2. Danh sách paper theo nhóm

### 2.1 Phân tích tác động qua lại ASR <-> cảm xúc

| Paper | Venue | Ý chính | Link |
|---|---|---|---|
| ASR and Emotional Speech: A Word-Level Investigation of the Mutual Impact of Speech and Emotion Recognition (Li, Zhao, Klejch, Bell, Lai) | Interspeech 2023 | Chạy 4 ASR (Kaldi, wav2vec2, Conformer, Whisper) trên IEMOCAP, MOSI, MELD; phân tích lỗi theo cảm xúc ở mức từ; rồi cho text-SER chạy trên transcript có WER tăng dần. [đã kiểm tra] | https://arxiv.org/abs/2305.16065 |
| Speech Emotion Recognition with ASR Transcripts: A Comprehensive Study on Word Error Rate and Fusion Techniques (Li, Bell, Lai) | SLT 2024 | Benchmark 11 ASR x 3 corpus (IEMOCAP, CMU-MOSI, MSP-Podcast) x 6 kiểu fusion. Kết luận: SER chịu được WER thấp, transcript ~10% WER cho kết quả gần với text chuẩn (nhất là Whisper). Đề xuất framework ASR error correction + modality-gated fusion. [đã kiểm tra] WER IEMOCAP (Table 3, 5.500 câu 4 lớp, không fine-tune ASR trên IEMOCAP): transcript tốt nhất trong 11 ASR 12.31, sau framework sửa lỗi **10.12** (thấp nhất đã xác minh trong tài liệu, 2026-11-04). Chuẩn hóa: lowercase + bỏ dấu câu, không bỏ từ đệm. | https://arxiv.org/abs/2406.08353 |
| Speech Emotion Recognition with ASR Integration (Yuanchao Li, PhD thesis, Edinburgh) | arXiv 2601.17901, 2026 | Tổng hợp cả hướng: speech foundation model mạnh ở ASR nhưng bỏ sót cue paralinguistic, có bias theo cảm xúc; lỗi ASR có tính hệ thống theo loại từ và cảm xúc; đề xuất incongruity-aware fusion, joint training, LLM error correction. **Nên đọc đầu tiên như một survey.** [đã kiểm tra abstract] | https://arxiv.org/abs/2601.17901 |
| Foundation Model Assisted Automatic Speech Emotion Recognition: Transcribing, Annotating, and Augmenting | arXiv 2309.08108 | Dùng foundation model để chép lời, gán nhãn, tăng cường dữ liệu cho SER. [cần kiểm tra] | https://arxiv.org/abs/2309.08108 |

### 2.2 Joint / multi-task ASR + SER

| Paper | Venue | Ý chính | Link |
|---|---|---|---|
| End-to-End Speech Emotion Recognition Combined with Acoustic-to-Word ASR Model (Feng et al., Kyoto) | Interspeech 2020 | Một trong những paper sớm ghép ASR end-to-end (A2W) vào SER. [đã kiểm tra] | https://www.isca-archive.org/interspeech_2020/feng20_interspeech.html |
| Speech Emotion Recognition with Multi-Task Learning (Cai et al.) | Interspeech 2021 | wav2vec 2.0 + 2 head: CTC (ASR) và emotion classifier; ASR làm task phụ giúp SER đạt SOTA IEMOCAP thời điểm đó. [cần kiểm tra tác giả] | https://paperswithcode.com/paper/speech-emotion-recognition-with-multi-task |
| Fusing ASR Outputs in Joint Training for Speech Emotion Recognition (Li, Bell, Lai) | ICASSP 2022 | Joint ASR-SER, fuse hidden state của ASR + text output bằng hierarchical co-attention. ~63% trên IEMOCAP. [đã kiểm tra] | https://arxiv.org/abs/2110.15684 |
| MMER: Multimodal Multi-task Learning for Speech Emotion Recognition | Interspeech 2023 | wav2vec2 + CTC phụ trợ + text; lý do: CTC ép encoder học căn chỉnh âm-chữ. [đã kiểm tra] | https://arxiv.org/abs/2203.16794 |
| Two-stage finetuning: pretrain SSL với ASR rồi mới finetune SER (Gao et al.) | Interspeech 2023 | Chỉ ra multi-task thường bị **gradient conflict**; giải bằng 2 giai đoạn. UA ~76.1% IEMOCAP. [đã kiểm tra] | https://www.isca-archive.org/interspeech_2023/gao23d_interspeech.pdf |
| MF-AED-AEC: SER by Multimodal Fusion, ASR Error Detection and ASR Error Correction (He, Shi, Li, Toda) | ICASSP 2024 | Chưa tìm thấy code chính thức (kiểm 2026-11-04). HuBERT-base + BERT-base (cả hai fine-tune), text từ Whisper-medium.en (WER 20.48 trên IEMOCAP). **Không sửa transcript lúc inference**: AED (gán nhãn KEEP/DELETE/CHANGE cho từng token, nhãn lấy từ căn LCS giữa ASR và transcript đúng) và AEC (decoder transformer 1 layer sinh từ đúng cho các token CHANGE) chỉ là loss phụ lúc train, bỏ đi khi inference. Fusion: cross-attention 2 chiều + khối tách thông tin chung/riêng (MIR), không ngữ cảnh. Loss = emo + 0.1·(3·AED + AEC). IEMOCAP 4 lớp, 5-fold leave-one-session-out, chỉ báo UAR, không nói dev set, 1 lần chạy. UAR: HuBERT 69.8; BERT text đúng 67.7 / ASR 66.4; HuBERT+BERT text đúng 75.5 / ASR 75.2; đề xuất 79.3; bỏ AED 77.8, bỏ AEC 78.2, bỏ MF 77.4. [đã đọc full text HTML v2] | https://arxiv.org/abs/2401.13260 |
| CARE: Leveraging Content and Acoustic Representations for SER | arXiv 2409.05566 (2024) | Có baseline WavLM-base + RoBERTa trên IEMOCAP 4 lớp: 73.02, nhưng báo weighted F1 và dùng text đúng, nên không so trực tiếp với UA. [cần kiểm tra] | https://arxiv.org/abs/2409.05566 |
| On the Efficacy and Noise-Robustness of Jointly Learned Speech Emotion and Automatic Speech Recognition | Interspeech 2023 | MTL trên IEMOCAP, thử nhiễu (noise, babble, music). Joint training giúp cả WER lẫn SER (số liệu ~-10.7% WER tương đối, +2.3% SER theo nguồn thứ cấp). [cần kiểm tra số] | https://arxiv.org/abs/2305.12540 |
| A Tiny Whisper-SER: Unifying ASR and Multi-label SER | (venue chưa rõ) | Fine-tune Whisper cho cả ASR và multi-label SER; train thẳng làm tệ cả hai, cần loss weighting 2 giai đoạn. [cần kiểm tra] | https://www.researchgate.net/publication/384833411 |
| Extending Whisper for emotion prediction using word-level pseudo labels (Kwok et al.) | ICASSP 2025 | Mở rộng Whisper dự đoán cảm xúc ở mức từ. [cần kiểm tra] | (tìm trên IEEE Xplore / arXiv) |
| AESR: Speech Recognition With Speech Emotion Recognizing Learning | Springer 2024/25 | MTL ASR-SER, MELD + SIMS, giảm WER/CER. [cần kiểm tra] | https://link.springer.com/chapter/10.1007/978-981-96-1045-7_8 |
| Multi-task learning for SER in naturalistic conditions (Zgórzyński et al.) | Interspeech 2025 | Hệ thống challenge: Whisper encoder + WavLM + RoBERTa, joint classification + regression. | https://www.isca-archive.org/interspeech_2025/zgorzynski25_interspeech.pdf |

### 2.3 Unified speech foundation model (1 model, nhiều task, có emotion)

| Model | Ý chính | Link |
|---|---|---|
| SenseVoice (FunAudioLLM, Alibaba, 2024) | Một model làm ASR + LID + SER + AED. SenseVoice-Small: encoder-only, non-autoregressive, thêm các embedding "câu hỏi" (LID, SER, AED, ITN) vào đầu chuỗi input, train bằng CTC. Nhanh ~15x Whisper-Large (theo README). ~234M params. **Gần nhất với khái niệm "emotion-aware ASR" ở dạng sản phẩm.** | https://arxiv.org/abs/2407.04051 , https://github.com/FunAudioLLM/SenseVoice |
| emotion2vec (Ma et al., Findings ACL 2024) | Self-supervised representation cho cảm xúc (online distillation, loss utterance + frame). Linear probe trên feature đóng băng đã mạnh. Có bản emotion2vec+ sau đó. | https://arxiv.org/abs/2312.15185 |
| EmoBox (Interspeech 2024) | Toolkit + benchmark SER: 32 dataset, 14 ngôn ngữ, 10 pretrained model. Dùng để so sánh công bằng. IEMOCAP: 5531 câu, 5 fold theo split gốc (2 người nói mỗi fold, tức theo session); feature layer cuối của encoder + head nhỏ; validation = 20% tập train; không nói rõ lấy trung bình fold hay chọn cấu hình theo tập nào. UA IEMOCAP: Whisper large v3 encoder 73.54, WavLM large 69.47 (Table 4). [đã đọc paper 2026-11-04] | https://arxiv.org/abs/2406.07162 |
| PL-Distill / AwCKA (Yang et al., ICASSP 2026) | Distill Qwen2-Audio 8.4B sang student 1.1B (Whisper large v3 encoder đóng băng + Qwen2-0.5B, LoRA). IEMOCAP UA 83.91 / WA 82.12. Theo split EmoBox; "selecting the best-performing model" không nói theo dev hay test; 1 lần chạy; chỉ audio. Đứng đầu leaderboard sota2.com (trang này chỉ gom bảng của PL-Distill và RE-LLM, không ghi protocol). [đã đọc paper 2026-11-04] | https://arxiv.org/abs/2602.01547 |
| MLL: Energy-Adaptive Mixup + FLAM (Wang et al., arXiv 2512.04551v2) | Chỉ audio, fine-tune WavLM-large. IEMOCAP 5531 câu, 5-fold theo session: WA 78.47 / UA 79.14. Không nói dev, không seed/std, không tìm thấy code. Hàng baseline trùng đúng số của Kang. Table 3: EAM so với LAM chỉ +0.08 UA (78.02 → 78.10), FLAM so với mean pool +0.15, focal +0.46, SupCon +0.73, đổi HuBERT sang WavLM +1.10. Đứng đầu bảng sota2 "Speaker-independent 5-fold". [đã đọc full text HTML, 2026-11-05] | https://arxiv.org/abs/2512.04551 |
| Label-Adaptive Mixup (Kang, Zhang, Jiang, ICASSP 2023) | Fine-tune HuBERT-large (22 layer), mixup gán nhãn theo độ dài câu, center loss. LOSO 5-fold WA 75.37 / UA 76.04. Code chọn epoch theo validation WA (10% tách từ train), nhưng paper nói rõ kiến trúc và hyperparameter chọn theo kết quả trên session 5, tức test của một fold. Không công bố file chia fold. [đã đọc full text HTML + code, 2026-11-05] | https://arxiv.org/abs/2305.06273 , https://github.com/leitro/LabelAdaptiveMixup-SER |
| Crab (Ueda, Lima, Costa, arXiv 2603.23673v2) | WavLM-large + RoBERTa-large + cross-modal transformer, contrastive ở 5 layer (MLCS), LR encoder 1e-6, LR head 1e-5. IEMOCAP 5531 câu: WAR 73.48 / UAR 75.85; speech 65.18, text 71.98. **Không ghi cách chia IEMOCAP**, repo chỉ có recipe MELD và MSP, không có code IEMOCAP. Ablation loss chỉ làm trên MELD: CE 52.53, MLS+CE 55.39, MLCS 55.65, bỏ LR phân tầng 51.82. Dùng text nên không phải speech-only dù nằm ở bảng SER của sota2. [đã đọc full text HTML + code, 2026-11-05] | https://arxiv.org/abs/2603.23673 , https://github.com/AI-Unicamp/Crab |
| VIB-Emo (Chen et al., arXiv 2505.23236) | HuBERT + LLaMA, joint SER + SED + ASR, LOSO 5-fold theo tiền xử lý EmoBox. IEMOCAP UA 77.6, WER 11.1 (chưa kiểm normalizer). Là entry hiếm trong top vừa làm ASR vừa làm SER. Pretrain trên GigaSpeech-m 670k câu của SpeechCraft. [đọc phần protocol và Table 2, 2026-11-05] | https://arxiv.org/abs/2505.23236 |
| SED rectification, Sys. 10 (Chen et al., arXiv 2606.14086) | Xây trên VIB-Emo, thêm chọn dữ liệu theo confidence và RL sửa nhãn mô tả prosody. IEMOCAP 80.98, không ghi tên metric (paper gốc [12] dùng UA, sota2 ghi Accuracy). So baseline của chính họ: 78.08 → 80.98. Thêm SED: 73.65 → 78.08. M và tỉ lệ 90% được so ngay trên bảng test. Table 2 đặt cạnh các SLM mà không ghi là đã fine-tune trên IEMOCAP. Đứng đầu bảng sota2 "IEMOCAP (test)". [đã đọc full text HTML, 2026-11-05] | https://arxiv.org/abs/2606.14086 |

### 2.4 LLM: sửa lỗi ASR + nhận diện cảm xúc

| Paper | Ý chính | Link |
|---|---|---|
| GenSEC challenge (SLT 2024): LLM-based Generative Error Correction cho ASR, speaker tagging, **emotion recognition** | Cho LLM đọc N-best của ASR rồi sửa lỗi / đoán cảm xúc. [cần kiểm tra arXiv id] | https://arxiv.org/abs/2409.09785 |
| LLM prompting cho post-ASR emotion prediction (Stepachev, Chen, Haddow) | GenSEC Task 3, IEMOCAP 4 lớp, **chỉ text** (11 ASR), gpt-4o prompting không fine-tune, ngữ cảnh = các câu trước (CW). Train UA (gpt-4o): CW 0 0.449, 2 0.510, 4 0.545, 8 0.555, 16 0.575, 32 0.574 (CI bootstrap ±0.019); lợi lớn nhất ở CW 0→4, sau đó bão hòa. Chọn transcript theo least_punc tốt nhất, theo WER tệ (0.492). Test chính thức: baseline 0.551, best **0.751** (CW 29, fuse 3 ASR). Chọn cấu hình trên train, 1 lần chạy. [đã đọc full text HTML, 2026-11-04] | https://arxiv.org/abs/2410.03312 |
| Revise, Reason, and Recognize (R3) | LLM: sửa lỗi ASR -> reasoning -> nhận diện cảm xúc, prompt chuyên cho emotion. | https://arxiv.org/abs/2409.15551 |
| Improving Speech-based Emotion Recognition with Contextual Utterance Analysis and LLMs (Zhang, Poellabauer) | GenSEC Task 3, IEMOCAP 4 lớp, **chỉ text**, GPT-3.5/GPT-4 prompting, transcript "ensemble" (GPT chọn bản mạch lạc nhất trong 11 ASR), ngữ cảnh = các câu trước trong cùng script. Train UA (GPT-3.5): CL 3 0.510, 5 0.529, 10 0.548, 15 0.562 (không có CL 0). Test: 0.694 (GPT-3.5, CL 10), best **0.752** (GPT-4, CL 10). Không dev, 1 lần chạy. [đã đọc full text HTML, 2026-11-04] | https://arxiv.org/abs/2410.20334 |

### 2.5 Audio-LLM và emotion reasoning (2024-2026)

| Paper / model | Ý chính | Link |
|---|---|---|
| SALMONN, Qwen-Audio / Qwen2-Audio, WavLLM | Audio-LLM tổng quát; có SER như một task. Trên IEMOCAP+MELD trung bình khoảng 50 (WavLLM, Qwen2-Audio), SALMONN ~32 theo bảng của 2506.06820. | https://github.com/QwenLM/Qwen2-Audio |
| AIR-Bench | Benchmark audio-LLM, có task emotion. | https://arxiv.org/abs/2402.07729 |
| Beyond Classification: Towards Speech Emotion Reasoning with Multitask AudioLLMs | Không chỉ phân loại mà sinh lời giải thích; encoder Whisper + emotion2vec. | https://arxiv.org/abs/2506.06820 |
| EmoQ | Speech-aware Q-Former + LLM cho SER. | https://arxiv.org/abs/2509.15775 |
| EMO-RL | RL với emotion rules cho audio-LLM, SER tổng quát hóa tốt hơn. | https://arxiv.org/abs/2509.15654 |
| EmotionThinker | Prosody-aware RL cho explainable emotion reasoning. | https://arxiv.org/abs/2601.15668 |
| Towards LLM-Empowered Fine-Grained Speech Descriptors for Explainable ER | Mô tả cảm xúc chi tiết, zero-shot. | https://arxiv.org/abs/2505.23236 |
| Spoken DialogSum | Dataset hội thoại giàu cảm xúc; so cascade Whisper+LLM với audio-LLM end-to-end. | https://arxiv.org/abs/2512.14687 |
| Robust SER under Tone-Word Conflict | Khi giọng nói và nội dung từ mâu thuẫn (nói "tôi ổn" với giọng buồn). | https://arxiv.org/abs/2609.04236 |
| Joint MoE representation for Speech Enhancement + Robust ER | Joint SE + ER trong nhiễu. | https://arxiv.org/abs/2509.08470 |

### 2.6 Tiếng Việt

| Nguồn | Ý chính | Link |
|---|---|---|
| VLSP 2025 shared task ASR + SER | Task kết hợp ASR và SER cho tiếng Việt. Proceedings trên ACL Anthology. | https://aclanthology.org/2025.vlsp-1/ |
| Đội Cake by VPBank (VLSP 2025) | Whisper Small (ASR) + emotion2vec features + classifier nhẹ (SER). Hạng 4, SER acc 79.5%, WER 19.12%. | https://aclanthology.org/events/vlsp-2025 |
| Dat & Khanh: Speech recognition and SER approach for VLSP 2025 | Framework 3 giai đoạn joint ASR-SER. | https://aclanthology.org/2025.vlsp-1.2.pdf |
| DFAT (Truong et al.) | Dual-stage fusion acoustic + text cho SER. | https://aclanthology.org/2025.vlsp-1/ |

---

## 3. Dataset hay dùng

| Dataset | Ngôn ngữ | Đặc điểm |
|---|---|---|
| IEMOCAP | EN | ~12h, diễn xuất + ứng tác, có transcript. Chuẩn de-facto cho joint ASR-SER. Thường dùng 4 lớp (neutral, happy+excited, sad, angry). |
| MELD | EN | Hội thoại từ phim Friends, 7 lớp, có transcript, đa phương thức. |
| MSP-Podcast | EN | Giọng tự nhiên từ podcast, lớn, có cả nhãn categorical và dimensional. |
| CMU-MOSI / MOSEI | EN | Sentiment/emotion, video YouTube. |
| RAVDESS, CREMA-D, ESD | EN (ESD có ZH) | Diễn xuất, câu cố định, hợp để test robustness của ASR theo cảm xúc. |
| VLSP 2025 ASR-SER | VI | Dữ liệu cho shared task tiếng Việt (kiểm tra điều kiện truy cập). |

## 4. Metric

- ASR: **WER** (hoặc CER cho tiếng Việt/Trung).
- SER: **UA** (unweighted accuracy = recall trung bình các lớp), **WA** (accuracy thường), **macro-F1**.
- Thường báo WER theo từng lớp cảm xúc để thấy bias.

## 5. Các phát hiện quan trọng (dùng khi viết phần related work)

1. Giọng có cảm xúc mạnh làm ASR sai nhiều hơn, và lỗi không ngẫu nhiên mà theo loại từ, cảm xúc.
2. SER trên transcript chịu được WER khoảng 10% mà gần như không mất gì so với text chuẩn.
3. ASR làm task phụ (CTC) thường giúp SER, vì encoder học được thông tin ngôn ngữ.
4. Train chung kiểu ngây thơ có thể làm tệ cả hai task (gradient conflict); giải bằng two-stage, loss weighting.
5. Hướng mới: một model sinh token cảm xúc kèm transcript (SenseVoice), hoặc audio-LLM có reasoning.
6. Số liệu giữa các paper khó so trực tiếp (khác split, khác số lớp, khác metric).
7. [đã kiểm tra trong code, 2026-11-01] ENT (github ECNU-Cross-Innovation-Lab/ENT): không có tập dev; mỗi epoch
   đánh giá trên test và báo MAX_WA/MAX_UA/MIN_WER qua mọi epoch, tức là chọn model trên test. Số trong paper
   (72.43 WA / 26.47 WER) có thể lạc quan. Test batch cố định là 1; utterance train dài hơn 300000 mẫu bị lọc.
   [đã đọc paper, 2026-11-04] Claim "vượt SOTA" của ENT chỉ trong 2 bảng hẹp:
   - Table 1: SER theo câu, chỉ so với method **dùng feature wav2vec 2.0** (Pepino 2021, Corr Attentive, DCW+TsPA,
     Shiftformer, MSTR, EmotionNAS; Shiftformer cùng nhóm tác giả). ENT 72.43/73.88 vs tốt nhất trước đó
     Shiftformer 72.10/72.70 (WA +0.33, UA +1.18). Không so với WavLM/HuBERT/fine-tune, multimodal, hay Gao 2023 (~76.1).
   - Table 2: so với method joint ASR+SER, chỉ có WA và WER: e2e-ASR 2020 (68.60/35.70), co-attention Li 2022
     (63.40/32.70), RNN-T emotion tag (58.20/26.70). ENT 72.43/26.47, FENT 71.84/25.99. Văn bản ghi "+3% UA, -0.7% WER"
     nhưng bảng không có UA; WER giảm 0.7 là của FENT, ENT chỉ giảm 0.23. "Low WER" là tương đối với các baseline cũ
     này (Whisper/SenseVoice-FT trên IEMOCAP ~10-16).
8. Code có sẵn: SenseVoice (FunASR finetune), li-2024 (fusions_bimodal.py), ENT, OSUM (checkpoint HF, ~20 GB VRAM),
   emotion2vec. chen-2025 có code nhưng không công bố weights. Bansal không có code.
