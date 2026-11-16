# Emotion-Aware Speech Recognition: flow trình bày (khoảng 17 phút)

Report: `paper-review/report/main.tex`. Chữ trên slide bằng tiếng Anh (cùng ngôn ngữ với report), lời nói bằng tiếng Việt.
Số liệu lấy từ `notebooks/PROGRESS.md` (đợt Whisper ASR, encoder turbo), trung bình 5 fold leave-one-session-out.
Part 1 (Background) để đầy đủ trên slide cho thầy xem; lời nói chỉ lướt, đi nhanh sang bài của mình.

| # | Slide | Thời lượng | Cộng dồn |
|---|---|---|---|
| 1 | Title | 0:20 | 0:20 |
| 2 | Outline | 0:15 | 0:35 |
| 3 | Speech emotion recognition | 0:40 | 1:15 |
| 4 | History of SER | 0:50 | 2:05 |
| 5 | Emotion cues in speech | 0:40 | 2:45 |
| 6 | Datasets and metrics | 0:40 | 3:25 |
| 7 | Emotion-aware ASR: four directions | 0:50 | 4:15 |
| 8 | Open challenges | 0:40 | 4:55 |
| 9 | Task definition | 0:40 | 5:35 |
| 10 | Problem | 1:00 | 6:35 |
| 11 | System overview | 1:20 | 7:55 |
| 12 | One utterance through the model | 1:20 | 9:15 |
| 13 | Data and protocol | 0:50 | 10:05 |
| 14 | Steps 1-2: ASR | 1:00 | 11:05 |
| 15 | Step 3: audio encoder | 0:45 | 11:50 |
| 16 | Step 4: ablation | 1:10 | 13:00 |
| 17 | ASR quality and emotion | 1:10 | 14:10 |
| 18 | Contribution of each module | 1:00 | 15:10 |
| 19 | Published systems on IEMOCAP | 1:00 | 16:10 |
| 20 | Conclusion | 0:50 | 17:00 |
| 21 | Backup: UA per fold | khi được hỏi | |

---

### Slide 1. Title
**Trên slide**
- Emotion-Aware Speech Recognition with Separate ASR and Emotion Models
- Voice Processing, final project
- Tên, MSSV

**Lời nói**
"Em trình bày một hệ nhận diện lời nói có kèm cảm xúc. Hệ gồm hai model tách riêng: một model ASR và một model nhận diện cảm xúc."

**Nguồn**: report, Abstract.

---

### Slide 2. Outline
**Trên slide**
- Background, Problem, Method, Experiments, Analysis, Conclusion

**Lời nói**
"Bài đi qua sáu phần: tổng quan về nhận diện cảm xúc từ giọng nói, vấn đề, phương pháp, bốn bước thực nghiệm, phân tích và kết luận."

**Nguồn**: report.

---

### Slide 3. Speech emotion recognition
**Trên slide**
- Task: speech in, emotion out
- Categorical labels (basic emotions; IEMOCAP 4 classes) and dimensional labels (valence, arousal, dominance)
- Applications: call centers, mental health monitoring, voice assistants, education, driver monitoring
- Chip: Ekman, Cognition and Emotion 1992; Russell, J. Personality and Social Psychology 1980; Schuller, Communications of the ACM 2018

**Lời nói**
"Nhận diện cảm xúc từ giọng nói, SER, nhận một đoạn giọng nói và trả về cảm xúc. Nhãn có hai kiểu: nhãn rời rạc theo các cảm xúc cơ bản, như bốn lớp em dùng, và nhãn liên tục theo các chiều valence, arousal. SER được dùng trong tổng đài, theo dõi sức khỏe tâm thần, trợ lý ảo, giáo dục và giám sát tài xế."

**Nguồn**: Ekman 1992 (An argument for basic emotions); Russell 1980 (A circumplex model of affect); Schuller 2018, CACM (Speech emotion recognition: two decades in a nutshell, benchmarks, and ongoing trends).

---

### Slide 4. History of SER
**Trên slide**
- 2000s, hand-crafted features: MFCC, pitch, energy + GMM, HMM, SVM. Chips: Schuller et al. ICASSP 2003; Schuller et al. Interspeech 2009 (Emotion Challenge); Eyben et al. IEEE TAC 2016 (GeMAPS)
- 2014-2019, deep learning: DNN, CNN, LSTM on spectrograms or raw waveform, attention pooling. Chips: Han et al. Interspeech 2014; Trigeorgis et al. ICASSP 2016; Mirsamadi et al. ICASSP 2017
- 2020-2023, self-supervised encoders: pre-trained on unlabeled speech, then fine-tuned or frozen. Chips: Baevski et al. NeurIPS 2020 (wav2vec 2.0); Pepino et al. Interspeech 2021; Chen et al. IEEE JSTSP 2022 (WavLM)
- 2024-, audio LLMs: one model answers questions about speech, including emotion. Chips: Tang et al. ICLR 2024 (SALMONN); Chu et al. arXiv 2024 (Qwen2-Audio); Ma et al. Findings ACL 2024 (emotion2vec)

**Lời nói**
"SER đi qua bốn giai đoạn. Những năm 2000 dùng feature tự thiết kế như MFCC, pitch, năng lượng rồi đưa vào GMM, HMM hay SVM; Interspeech 2009 có challenge đầu tiên và sau đó có bộ feature chuẩn GeMAPS. Từ khoảng 2014, deep learning học feature trực tiếp từ spectrogram hay waveform. Từ 2020, các encoder self-supervised như wav2vec 2.0 và WavLM, train trên rất nhiều giọng nói không nhãn, trở thành nền cho SER. Từ 2024 có audio LLM trả lời được cả câu hỏi về cảm xúc. Hệ của em nằm ở giai đoạn ba: dùng encoder pre-trained."

**Nguồn**: Schuller, Rigoll, Lang, ICASSP 2003 (https://opus.bibliothek.uni-augsburg.de/opus4/frontdoor/index/index/year/2020/docId/76935); Schuller, Steidl, Batliner, Interspeech 2009 (https://www.isca-archive.org/interspeech_2009/schuller09_interspeech.html); Eyben et al., IEEE TAC 2016 (https://opus.bibliothek.uni-augsburg.de/opus4/38554); Han, Yu, Tashev, Interspeech 2014 (https://www.isca-archive.org/interspeech_2014/han14_interspeech.html); Trigeorgis et al., ICASSP 2016 (https://research.gold.ac.uk/17322); Mirsamadi, Barsoum, Zhang, ICASSP 2017 (https://dblp.uni-trier.de/rec/conf/icassp/MirsamadiBZ17.html); Pepino, Riera, Ferrer, Interspeech 2021 (https://www.isca-archive.org/interspeech_2021/pepino21_interspeech.html); SALMONN, ICLR 2024 (https://proceedings.iclr.cc/paper_files/paper/2024/hash/476ab8f369e489c04187ba84f68cfa68-Abstract-Conference.html); Qwen2-Audio, arXiv 2407.10759; emotion2vec, arXiv 2312.15185.

---

### Slide 5. Emotion cues in speech
**Trên slide**
- Acoustic: prosody (pitch, energy, speaking rate), voice quality, spectrum (MFCC). Used by audio-only SER
- Lexical: the words in the transcript. Needs ASR or a manual transcript. Used by text-only SER
- Context: previous turns in the dialogue. Used by conversational SER

**Lời nói**
"Cảm xúc nằm ở ba nguồn. Âm học: cao độ, năng lượng, tốc độ nói, chất giọng. Từ ngữ: nội dung câu nói. Ngữ cảnh: các câu trước trong hội thoại. Đi thẳng từ waveform qua MFCC rồi ra cảm xúc là chỉ dùng nguồn thứ nhất. Hệ của em dùng cả ba nguồn."

**Nguồn**: Eyben et al. 2016 (GeMAPS, feature âm học); Li et al. SLT 2024 (text từ ASR); Zhang và Poellabauer 2024, GenSEC (ngữ cảnh).

---

### Slide 6. Datasets and metrics
**Trên slide**
- Bảng dataset: IEMOCAP, MSP-Podcast, MELD, RAVDESS, CREMA-D, EmoDB, VLSP 2025 ASR-SER (ngôn ngữ, kiểu giọng nói, quy mô, nhãn, nguồn)
- Metrics: WER, UA, WA, F1

**Lời nói**
"Các dataset hay dùng chia làm hai nhóm: giọng diễn xuất như IEMOCAP, RAVDESS, CREMA-D, EmoDB, và giọng tự nhiên như MSP-Podcast. MELD lấy từ phim Friends. Tiếng Việt có shared task VLSP 2025 làm cả ASR và SER. Metric: WER cho ASR; UA, WA và F1 cho cảm xúc."

**Nguồn**: Busso et al. 2008 (IEMOCAP); Lotfian và Busso, IEEE TAC 2019 (MSP-Podcast; 409 giờ theo bài corpus 2025, arXiv 2509.09791); Poria et al. ACL 2019 (MELD); Livingstone và Russo, PLoS ONE 2018 (RAVDESS, 24 diễn viên, 7356 file); Cao et al. IEEE TAC 2014, vol. 5 no. 4 (CREMA-D, 91 diễn viên, 7442 clip; https://www.x-mol.com/paper/1212911514258579468); Burkhardt et al. Interspeech 2005 (EmoDB); VLSP 2025 (https://aclanthology.org/2025.vlsp-1/).

---

### Slide 7. Emotion-aware ASR: four directions
**Trên slide**
- ASR robust to emotional speech. Chip: Li et al. Interspeech 2023
- Joint ASR + SER model. Chips: Cai et al. Interspeech 2021; An et al. arXiv 2024 (SenseVoice)
- ASR then SER (this work). Chips: Li et al. SLT 2024; He et al. ICASSP 2024
- Audio LLMs. Chips: R3, arXiv 2024; Chen et al. arXiv 2025 (VIB-Emo)

**Lời nói**
"Riêng emotion-aware ASR có bốn hướng. Một là làm ASR chịu được giọng có cảm xúc. Hai là một model làm cả ASR và SER. Ba là chạy ASR trước rồi đoán cảm xúc, hướng của em. Bốn là dùng audio LLM sinh cả transcript lẫn nhãn cảm xúc."

**Nguồn**: `.claude/research/papers.md` mục 1 và 2; Li et al. Interspeech 2023 (arXiv 2305.16065); Cai, Yuan, Zheng, Huang, Church, Interspeech 2021 (https://unpaywall.org/10.21437%2FInterspeech.2021-1852); SenseVoice (arXiv 2407.04051); Li et al. SLT 2024; He et al. ICASSP 2024; R3 (arXiv 2409.15551); VIB-Emo (arXiv 2505.23236).

---

### Slide 8. Open challenges
**Trên slide**
- Acted vs natural speech (ours: future work on MSP-Podcast)
- Ambiguous labels (ours: 4 standard classes)
- Unseen speakers (ours: leave-one-session-out)
- Results hard to compare (ours: dev set for model selection, 5-fold mean)

**Lời nói**
"Còn bốn thách thức. Model train trên giọng diễn xuất thường kém hơn trên giọng tự nhiên. Nhãn cảm xúc mơ hồ, người gán nhãn hay bất đồng. Người nói ở tập test phải chưa xuất hiện lúc train. Và mỗi paper chia dữ liệu, chọn model, đo metric một kiểu, nên khó so. Mỗi thẻ ghi cách bài của em xử lý thách thức đó."

**Nguồn**: `.claude/research/papers.md` mục 5 (phát hiện 6, 7); Goncalves et al. Odyssey 2024 (SER challenge trên MSP-Podcast); Triantafyllopoulos et al. 2024 (arXiv 2406.06401, xem lại Emotion Challenge 2009).

---

### Slide 9. Task definition
**Trên slide**
- Input: one utterance (audio)
- Output 1: transcript (ASR, metric WER)
- Output 2: emotion, 4 classes: neutral, happy, angry, sad (SER, metric UA)

**Lời nói**
"Bài toán của em nhận một câu nói và trả về hai thứ: nội dung lời nói, đo bằng WER, và cảm xúc của người nói, đo bằng UA. UA là trung bình recall của bốn lớp, nên lớp ít mẫu cũng được tính ngang lớp nhiều mẫu."

**Nguồn**: report, Introduction và Setup.

---

### Slide 10. Problem
**Trên slide**
- Joint model (one model, two heads): Gao et al. Interspeech 2023, Shen et al. (ENT) ICASSP 2024. Limitation: conflicting gradients, the shared network drifts toward ASR.
- ASR then SER: Li et al. SLT 2024, He et al. (MF-AED-AEC) ICASSP 2024. Limitation: ASR errors reach the emotion model.

**Lời nói**
"Có hai hướng làm. Hướng thứ nhất dùng một model cho cả hai tác vụ. Gao và cộng sự (Interspeech 2023) chỉ ra rằng khi train chung, gradient của hai tác vụ thường ngược chiều, và mạng dùng chung nghiêng về ASR. Hướng thứ hai chạy ASR trước rồi đoán cảm xúc từ transcript. Hướng này bị lỗi ASR truyền sang, và Li và cộng sự (SLT 2024) cho thấy mức ảnh hưởng tùy vào WER. Em đi theo hướng thứ hai, tách hai model, và đo xem lỗi ASR thực sự tốn bao nhiêu."

**Nguồn**: Gao et al. 2023 (Introduction); Li et al. 2024 (Abstract); He et al. 2024; Shen et al. 2024.

---

### Slide 11. System overview
**Trên slide**
- Hình tổng quan (fig_overview)

**Lời nói**
"Audio đi vào hai nhánh song song. Nhánh trên là ASR module, Whisper large-v3-turbo đã fine-tune, cho ra transcript. Transcript cùng ba câu trước trong hội thoại đi vào RoBERTa-large để ra vector h text. Nhánh dưới là encoder của chính Whisper large-v3-turbo bản gốc, đóng băng, cho ra vector h audio. Như vậy cả hệ chỉ dùng một checkpoint Whisper. Gated fusion trộn hai vector theo trọng số học được, rồi classifier đoán cảm xúc. Hai auxiliary head chỉ dùng lúc train để nhánh nào cũng phải tự học được cảm xúc, tránh gate dồn hết về một nhánh."

**Nguồn**: report, Method.

---

### Slide 12. One utterance through the model
**Trên slide**
- Hình luồng (fig_flow)
- Trained vs frozen: Whisper turbo and RoBERTa fine-tuned; encoder of the original Whisper turbo frozen; layer weights, projection, gate, heads trained from scratch

**Lời nói**
"Đây là một câu đi qua model kèm kích thước tensor. Encoder Whisper trả về 33 lớp, mỗi lớp 1280 chiều. Em lấy trung bình theo các frame thật, bỏ phần padding, rồi trộn 33 lớp bằng trọng số softmax học được và chiếu về 1024 chiều. Gate là softmax trên hai nhánh, z là tổng có trọng số. Loss gồm cross-entropy của z cộng hai loss phụ trên h text và h audio."

**Nguồn**: report, Method (Fig. 2, phương trình 1-2).

---

### Slide 13. Data and protocol
**Trên slide**
- Bảng số câu theo session và lớp (5531 câu có nhãn trong 10039 câu)
- Leave-one-session-out, 5 folds; dev = 10% of training sessions; each ASR re-fine-tuned per fold

**Lời nói**
"IEMOCAP có 5 session, mỗi session 2 diễn viên. Em dùng 4 lớp chuẩn, gộp excited vào happy, được 5531 câu có nhãn. Các câu không có nhãn chỉ dùng làm ngữ cảnh. Mỗi fold để riêng một session làm test, nên người nói ở tập test chưa từng xuất hiện lúc train. ASR cũng được fine-tune lại trong từng fold."

**Nguồn**: report, Data and Setup, Table II.

---

### Slide 14. Steps 1-2: ASR
**Trên slide**
- Bảng WER zero-shot (Step 1) và fine-tuned (Step 2) của SenseVoice-Small (234M) và Whisper large-v3-turbo (809M)
- Chosen: Whisper large-v3-turbo, fine-tuned; SenseVoice-Large chưa được công bố
- Bảng WER theo cảm xúc sau fine-tune

**Lời nói**
"Bước 1 so hai ASR khi chưa fine-tune: SenseVoice tốt hơn, 11.20 so với 13.19, dù nhỏ hơn gần bốn lần. Bước 2 fine-tune cả hai trên từng fold: Whisper giảm mạnh hơn, còn 8.32, thấp hơn SenseVoice 9.73. Vì vậy ASR module là Whisper large-v3-turbo đã fine-tune. SenseVoice-Large có trong paper nhưng không được công bố weights, nên em dùng bản Small. Bảng phải: câu nói vui, phấn khích khó nhận dạng nhất với cả hai model, câu tức giận dễ nhất."

**Nguồn**: report, Table III; PROGRESS.md; WER theo cảm xúc tính từ `results/iemocap/*_finetune*/metrics_test.json`; SenseVoice chỉ công bố bản Small (https://github.com/FunAudioLLM/SenseVoice, https://huggingface.co/FunAudioLLM/SenseVoiceSmall; arXiv 2407.04051); số tham số: file fp32 936 MB của SenseVoice-Small, Whisper turbo 809M.

---

### Slide 15. Step 3: audio encoder
**Trên slide**
- Bảng WavLM-large và Whisper large-v3-turbo encoder trong hệ đầy đủ: UA, WA, wF1
- One Whisper checkpoint for both roles

**Lời nói**
"Giữ nguyên transcript Whisper và ngữ cảnh, em chỉ đổi encoder audio. Encoder Whisper turbo cao hơn WavLM ở cả ba metric, UA 81.80 so với 80.24. Kết quả này cùng chiều với benchmark EmoBox. Vì encoder lấy từ đúng checkpoint turbo dùng cho ASR, cả hệ chỉ cần một model Whisper: bản fine-tune làm ASR, encoder bản gốc đóng băng làm nhánh audio."

**Nguồn**: report, Table IV; Ma et al. 2024 (EmoBox).

---

### Slide 16. Step 4: ablation
**Trên slide**
- Bảng: Full, w/o audio, w/o context, w/o text; UA, WA, wF1, Δ UA

**Lời nói**
"Em bỏ từng module ra khỏi hệ đầy đủ. Bỏ text mất nhiều nhất, 9.20 UA. Bỏ ngữ cảnh mất 4.97. Bỏ audio mất 4.26. Module nào cũng có đóng góp: text nhiều nhất, ngữ cảnh và audio gần nhau."

**Nguồn**: report, Table V.

---

### Slide 17. ASR quality and emotion
**Trên slide**
- Bảng 1 (text only): ground-truth text 78.57, Whisper-FT transcript 77.54
- Bảng 2 (full system, cùng encoder turbo): SenseVoice-FT transcript 81.29, Whisper-FT transcript 81.80

**Lời nói**
"Lỗi ASR tốn bao nhiêu? Bảng trái: cùng chỉ dùng text, thay transcript bằng text đúng chỉ tăng 1.03 UA. Tức là dù ASR hoàn hảo, từ WER 8.3% về 0, cũng chỉ thêm được khoảng 1 UA. Bảng phải: giữ nguyên encoder audio, đổi sang transcript SenseVoice, WER 9.7%, hệ đầy đủ được 81.29, thấp hơn 0.51, nằm trong độ dao động giữa các fold. Nên ở mức WER 8 đến 10%, chọn ASR nào gần như không đổi kết quả cảm xúc, còn thêm audio được 4.26 và ngữ cảnh được 4.97. ASR được chọn theo WER, đúng nhiệm vụ của nó."

**Nguồn**: report, Analysis; Table V; Li et al. 2024.

---

### Slide 18. Contribution of each module
**Trên slide**
- Bảng recall theo lớp: Full, w/o audio, w/o context, w/o text
- Gate weight on audio: full 0.53, w/o context 0.55, WavLM 0.31

**Lời nói**
"Nhìn theo từng lớp: audio giúp nhiều nhất cho neutral, tăng 8.0 recall, và angry, tăng 6.6. Ngữ cảnh giúp neutral, tăng 8.2, và happy, tăng 7.1. Lớp sad gần như không đổi khi bỏ module nào. Gate trong hệ đầy đủ đặt trung bình 0.53 trọng số vào audio. Khi bỏ ngữ cảnh, nhánh text yếu đi nên gate nghiêng thêm sang audio, 0.55. Với WavLM, gate chỉ đặt 0.31 vào audio."

**Nguồn**: report, Table VI và Analysis.

---

### Slide 19. Published systems on IEMOCAP
**Trên slide**
- Bảng các hệ đã công bố: loại hệ, có dùng ASR text, audio, ngữ cảnh, WER, UA

**Lời nói**
"Đây là các hệ đã công bố trên IEMOCAP bốn lớp. Cách chia dữ liệu, tập dev và metric khác nhau nên không so trực tiếp được. Trong nhóm có làm cả ASR, hệ của em có WER thấp nhất, 8.32, và UA 81.80. PL-Distill cao hơn về UA nhưng chỉ làm nhận diện cảm xúc từ audio, không ra transcript."

**Nguồn**: report, Table I; Gao et al. 2023; Shen et al. 2024; Chen et al. 2025; He et al. 2024; Ma et al. 2024; Yang et al. 2026; Ueda et al. 2026.

---

### Slide 20. Conclusion
**Trên slide**
- Whisper large-v3-turbo fine-tuned: 8.32% WER
- Full system: 81.80% UA, one Whisper for ASR and audio
- Text > context > audio; ASR errors cost about 1 UA
- Future work: spontaneous speech (MSP-Podcast)

**Lời nói**
"Tóm lại: fine-tune làm đổi thứ hạng hai ASR, Whisper turbo đạt 8.32% WER. Hệ đầy đủ đạt 81.80% UA, chỉ dùng một model Whisper cho cả ASR và nhánh audio. Text đóng góp nhiều nhất, rồi ngữ cảnh, rồi audio. Ở mức WER này, lỗi ASR chỉ tốn khoảng 1 UA. Hướng tiếp theo là thử trên giọng nói tự nhiên như MSP-Podcast."

**Nguồn**: report, Conclusion.

---

### Slide 21. Backup: UA per fold
**Trên slide**
- Bảng UA theo session cho Full, WavLM, w/o audio, w/o context, w/o text

**Lời nói**
"Nếu được hỏi kết quả có ổn định không: hệ đầy đủ thắng bản bỏ audio, bản bỏ ngữ cảnh và bản bỏ text ở cả 5 fold. Encoder Whisper thắng WavLM ở 4 trên 5 fold, trừ session 2."

**Nguồn**: report, Table VII.
