# Emotion-Aware Speech Recognition: flow trình bày (khoảng 15 phút)

Report: `paper-review/report/main.tex`. Chữ trên slide bằng tiếng Anh (cùng ngôn ngữ với report), lời nói bằng tiếng Việt.
Số liệu lấy từ `notebooks/PROGRESS.md` (đợt Whisper ASR, encoder turbo), trung bình 5 fold leave-one-session-out.
Part 1 (Background) để đầy đủ trên slide cho thầy xem; lời nói chỉ lướt, đi nhanh sang bài của mình.
Chữ trên slide viết đơn giản: không viết tắt (finetune, without), không để ý phụ trong ngoặc, hệ đầy đủ gọi là "Our system".

| # | Slide | Thời lượng | Cộng dồn |
|---|---|---|---|
| 1 | Title | 0:20 | 0:20 |
| 2 | Outline | 0:15 | 0:35 |
| 3 | Speech emotion recognition | 0:40 | 1:15 |
| 4 | History of SER | 0:40 | 1:55 |
| 5 | Three sources of emotion | 0:40 | 2:35 |
| 6 | Datasets and metrics | 0:40 | 3:15 |
| 7 | Four approaches to emotion-aware ASR | 0:40 | 3:55 |
| 8 | Task definition | 0:40 | 4:35 |
| 9 | Existing approaches and their limits | 1:00 | 5:35 |
| 10 | System overview | 1:20 | 6:55 |
| 11 | System flow | 1:20 | 8:15 |
| 12 | Data and evaluation | 0:50 | 9:05 |
| 13 | Choosing the ASR model | 1:00 | 10:05 |
| 14 | Choosing the audio encoder | 0:45 | 10:50 |
| 15 | Ablation study | 1:10 | 12:00 |
| 16 | UA per test session | 0:30 | 12:30 |
| 17 | How ASR errors affect emotion | 1:00 | 13:30 |
| 18 | Recall of each class | 1:00 | 14:30 |
| 19 | Comparison with published systems | 1:10 | 15:40 |

---

### Slide 1. Title
**Trên slide**
- Emotion-Aware Speech Recognition with Separate ASR and Emotion Models
- Voice Processing, final project
- Tên, MSSV

**Lời nói**
"Em trình bày một hệ nhận diện lời nói có kèm cảm xúc. Hệ gồm hai model tách riêng: một model ASR và một model nhận diện cảm xúc. Em thực nghiệm trên IEMOCAP, chia 5 fold theo session."

**Nguồn**: report, Abstract.

---

### Slide 2. Outline
**Trên slide**
- Background, Problem, Method, Experiments, Analysis

**Lời nói**
"Bài đi qua năm phần: tổng quan về nhận diện cảm xúc từ giọng nói, vấn đề, phương pháp, thực nghiệm và phân tích."

**Nguồn**: report.

---

### Slide 3. Speech emotion recognition
**Trên slide**
- Two kinds of labels: categorical (basic emotions; IEMOCAP 4 classes), dimensional (valence, arousal)
- Applications: call centers, mental health monitoring, voice assistants, education, driver monitoring

**Lời nói**
"Nhận diện cảm xúc từ giọng nói, SER, nhận một đoạn giọng nói và trả về cảm xúc. Nhãn có hai kiểu: nhãn rời rạc theo các cảm xúc cơ bản, như bốn lớp em dùng, và nhãn liên tục: valence là cảm xúc tiêu cực hay tích cực, arousal là bình tĩnh hay kích động. SER được dùng trong tổng đài, theo dõi sức khỏe tâm thần, trợ lý ảo, giáo dục và giám sát tài xế."

**Nguồn**: Ekman 1992 (An argument for basic emotions); Russell 1980 (A circumplex model of affect); Schuller 2018, CACM (Speech emotion recognition: two decades in a nutshell, benchmarks, and ongoing trends).

---

### Slide 4. History of SER
**Trên slide**
- 2000s, hand-crafted features: MFCC, pitch, energy, classified by GMM, HMM or SVM. Ví dụ: GeMAPS
- 2014-2019, deep learning: CNN, LSTM learn features from spectrograms or raw audio
- 2020-2023, self-supervised models: pre-trained on unlabeled speech. Ví dụ: wav2vec 2.0, WavLM
- 2024-, audio LLMs: one model answers questions about speech, including emotion. Ví dụ: SALMONN, Qwen2-Audio

**Lời nói**
"SER đi qua bốn giai đoạn. Những năm 2000 dùng feature tự thiết kế như MFCC, pitch, năng lượng rồi đưa vào GMM, HMM hay SVM; sau đó có bộ feature chuẩn GeMAPS. Từ khoảng 2014, deep learning học feature trực tiếp từ spectrogram hay waveform. Từ 2020, các model self-supervised như wav2vec 2.0 và WavLM, train trên rất nhiều giọng nói không nhãn, trở thành nền cho SER. Từ 2024 có audio LLM trả lời được cả câu hỏi về cảm xúc. Hệ của em nằm ở giai đoạn ba: dùng encoder pre-trained."

**Nguồn**: Schuller, Rigoll, Lang, ICASSP 2003 (https://opus.bibliothek.uni-augsburg.de/opus4/frontdoor/index/index/year/2020/docId/76935); Schuller, Steidl, Batliner, Interspeech 2009 (https://www.isca-archive.org/interspeech_2009/schuller09_interspeech.html); Eyben et al., IEEE TAC 2016 (https://opus.bibliothek.uni-augsburg.de/opus4/38554); Han, Yu, Tashev, Interspeech 2014 (https://www.isca-archive.org/interspeech_2014/han14_interspeech.html); Trigeorgis et al., ICASSP 2016 (https://research.gold.ac.uk/17322); Mirsamadi, Barsoum, Zhang, ICASSP 2017 (https://dblp.uni-trier.de/rec/conf/icassp/MirsamadiBZ17.html); Pepino, Riera, Ferrer, Interspeech 2021 (https://www.isca-archive.org/interspeech_2021/pepino21_interspeech.html); SALMONN, ICLR 2024 (https://proceedings.iclr.cc/paper_files/paper/2024/hash/476ab8f369e489c04187ba84f68cfa68-Abstract-Conference.html); Qwen2-Audio, arXiv 2407.10759; emotion2vec, arXiv 2312.15185.

---

### Slide 5. Three sources of emotion
**Trên slide**
- Acoustic, how it is said: pitch, loudness, speaking speed; voice quality; spectrum such as MFCC. Used in audio-only SER
- Lexical, what is said: the words of the sentence; needs ASR or a manual transcript. Used in text-only SER
- Context, what came before: earlier turns; the other speaker's reaction. Used in conversational SER

**Lời nói**
"Cảm xúc nằm ở ba nguồn. Âm học: cao độ, độ lớn, tốc độ nói, chất giọng. Từ ngữ: nội dung câu nói. Ngữ cảnh: các câu trước trong hội thoại. Đi thẳng từ waveform qua MFCC rồi ra cảm xúc là chỉ dùng nguồn thứ nhất. Hệ của em dùng cả ba nguồn."

**Nguồn**: Eyben et al. 2016 (GeMAPS, feature âm học); Li et al. SLT 2024 (text từ ASR); Zhang và Poellabauer 2024, GenSEC (ngữ cảnh).

---

### Slide 6. Datasets and metrics
**Trên slide**
- Bảng dataset: IEMOCAP, MSP-Podcast, MELD (ngôn ngữ, kiểu giọng nói, quy mô, nhãn, nguồn)
- Metrics: WER, UA, WA, F1, mỗi cái một dòng giải thích

**Lời nói**
"Ba dataset hay dùng nhất: IEMOCAP là hội thoại do diễn viên đóng, MSP-Podcast là giọng tự nhiên từ podcast, MELD lấy từ phim Friends. Ngoài ra còn RAVDESS, CREMA-D, EmoDB, và tiếng Việt có VLSP 2025 làm cả ASR và SER. Metric: WER là tỉ lệ từ ASR nhận sai; UA là recall trung bình của các lớp, nên lớp ít mẫu được tính ngang lớp nhiều mẫu; WA là tỉ lệ câu đoán đúng; F1 cân bằng precision và recall."

**Nguồn**: Busso et al. 2008 (IEMOCAP); Lotfian và Busso, IEEE TAC 2019 (MSP-Podcast; 409 giờ theo bài corpus 2025, arXiv 2509.09791); Poria et al. ACL 2019 (MELD); Livingstone và Russo, PLoS ONE 2018 (RAVDESS); Cao et al. IEEE TAC 2014 (CREMA-D); Burkhardt et al. Interspeech 2005 (EmoDB); VLSP 2025 (https://aclanthology.org/2025.vlsp-1/).

---

### Slide 7. Four approaches to emotion-aware ASR
**Trên slide**
- Robust ASR on emotional speech
- One model for ASR and SER
- ASR first, then SER (this work)
- Audio LLMs

**Lời nói**
"Riêng emotion-aware ASR có bốn hướng. Một là làm ASR chịu được giọng có cảm xúc, ví dụ Li và cộng sự ở Interspeech 2023. Hai là một model làm cả ASR và SER, như SenseVoice. Ba là chạy ASR trước rồi đoán cảm xúc, hướng của em. Bốn là dùng audio LLM sinh cả transcript lẫn nhãn cảm xúc."

**Nguồn**: `.claude/research/papers.md` mục 1 và 2; Li et al. Interspeech 2023 (arXiv 2305.16065); Cai, Yuan, Zheng, Huang, Church, Interspeech 2021 (https://unpaywall.org/10.21437%2FInterspeech.2021-1852); SenseVoice (arXiv 2407.04051); Li et al. SLT 2024; He et al. ICASSP 2024; R3 (arXiv 2409.15551); VIB-Emo (arXiv 2505.23236).

---

### Slide 8. Task definition
**Trên slide**
- Input: audio of one sentence
- Output 1: transcript (ASR, WER, lower is better)
- Output 2: emotion, neutral, happy, angry or sad (SER, UA, higher is better)

**Lời nói**
"Bài toán của em nhận một câu nói và trả về hai thứ: nội dung lời nói, đo bằng WER, và cảm xúc của người nói, đo bằng UA."

**Nguồn**: report, Introduction và Setup.

---

### Slide 9. Existing approaches and their limits
**Trên slide**
- One model for both tasks: Gao et al. Interspeech 2023, Shen et al. (ENT) ICASSP 2024. Limitation: the two tasks compete inside one network, and it ends up favoring ASR.
- ASR first, then emotion from the text: Li et al. SLT 2024, He et al. (MF-AED-AEC) ICASSP 2024. Limitation: mistakes in the transcript lead to wrong emotions.

**Lời nói**
"Có hai hướng làm. Hướng thứ nhất dùng một model cho cả hai tác vụ. Gao và cộng sự chỉ ra rằng khi train chung, gradient của hai tác vụ thường ngược chiều, và mạng dùng chung nghiêng về ASR. Hướng thứ hai chạy ASR trước rồi đoán cảm xúc từ transcript. Hướng này bị lỗi ASR truyền sang, và Li và cộng sự cho thấy mức ảnh hưởng tùy vào WER. Em đi theo hướng thứ hai, tách hai model, và đo xem lỗi ASR thực sự tốn bao nhiêu."

**Nguồn**: Gao et al. 2023 (Introduction); Li et al. 2024 (Abstract); He et al. 2024; Shen et al. 2024.

---

### Slide 10. System overview
**Trên slide**
- Hình tổng quan (fig_overview, SVG)

**Lời nói**
"Audio đi vào hai nhánh song song. Nhánh trên là ASR module, Whisper large-v3-turbo đã finetune, cho ra transcript. Transcript cùng ba câu trước trong hội thoại đi vào RoBERTa-large để ra vector h text. Nhánh dưới là encoder của chính Whisper large-v3-turbo bản gốc, đóng băng, cho ra vector h audio. Như vậy cả hệ chỉ dùng một checkpoint Whisper. Gated fusion trộn hai vector theo trọng số học được, rồi classifier đoán cảm xúc. Hai auxiliary head chỉ dùng lúc train để nhánh nào cũng phải tự học được cảm xúc, tránh gate dồn hết về một nhánh."

**Nguồn**: report, Method.

---

### Slide 11. System flow
**Trên slide**
- Hình luồng (fig_flow, SVG), chiếm cả slide

**Lời nói**
"Đây là một câu đi qua model kèm kích thước tensor. Icon ổ khóa là phần đóng băng, icon ngọn lửa là phần được train. Whisper large-v3-turbo và RoBERTa-large được finetune; encoder Whisper turbo bản gốc đóng băng; layer weights, projection, gate, classifier và auxiliary heads train từ đầu. Encoder Whisper trả về 33 lớp, mỗi lớp 1280 chiều. Em lấy trung bình theo các frame thật, bỏ phần padding, rồi trộn 33 lớp bằng trọng số softmax học được và chiếu về 1024 chiều. Gate là softmax trên hai nhánh, z là tổng có trọng số. Loss gồm cross-entropy của z cộng hai loss phụ trên h text và h audio; auxiliary heads bị bỏ khi suy luận."

**Nguồn**: report, Method (Fig. 2, phương trình 1-2).

---

### Slide 12. Data and evaluation
**Trên slide**
- Bảng số câu theo session và lớp (5531 câu có nhãn trong 10039 câu)
- 5 folds, each tests on one session; validation set = 10% of training sessions, used to choose the best epoch

**Lời nói**
"IEMOCAP có 5 session, mỗi session 2 diễn viên. Em dùng 4 lớp chuẩn, gộp excited vào happy, được 5531 câu có nhãn. Các câu không có nhãn chỉ dùng làm ngữ cảnh. Mỗi fold để riêng một session làm test, nên người nói ở tập test chưa từng xuất hiện lúc train. Trong 4 session còn lại, em tách ngẫu nhiên 10% làm validation set để chọn epoch tốt nhất, phần còn lại để train. ASR cũng được finetune lại trong từng fold, nên transcript của tập test luôn đến từ một ASR chưa thấy session đó."

**Nguồn**: report, Data and Setup, Table II.

---

### Slide 13. Choosing the ASR model
**Trên slide**
- Bảng WER trước và sau finetune của SenseVoice-Small (234M) và Whisper large-v3-turbo (809M)
- Tag "chosen" ở dòng Whisper; pill: finetuning helps Whisper more (−3.63 vs −1.47)
- Card phải: bảng WER theo cảm xúc sau finetune, pill "happy is hardest for both"

**Lời nói**
"Đầu tiên em so hai ASR khi chưa finetune: SenseVoice tốt hơn, 11.20 so với 13.19, dù nhỏ hơn gần bốn lần. Sau khi finetune cả hai trên từng fold, Whisper giảm mạnh hơn, còn 9.56, thấp hơn SenseVoice 9.73 một chút. Vì vậy em chọn Whisper large-v3-turbo đã finetune. SenseVoice-Large có trong paper nhưng không được công bố weights, nên em dùng bản Small. Bảng phải: câu nói vui, phấn khích khó nhận dạng nhất với cả hai model, câu tức giận dễ nhất. Whisper thấp hơn ở mọi lớp trừ happy: có bốn câu test chỉ có tiếng cười hoặc tiếng thở, hai câu là happy, và Whisper lặp một từ đến hết độ dài cho phép. Bốn câu này cũng là lý do WER ở session 1 và 3 cao hơn các session khác."

**Nguồn**: report, Table III; PROGRESS.md; WER theo cảm xúc tính từ `results/iemocap/*_finetune*/metrics_test.json`; SenseVoice chỉ công bố bản Small (https://github.com/FunAudioLLM/SenseVoice, https://huggingface.co/FunAudioLLM/SenseVoiceSmall; arXiv 2407.04051); số tham số: file fp32 936 MB của SenseVoice-Small, Whisper turbo 809M.

---

### Slide 14. Choosing the audio encoder
**Trên slide**
- Bảng WavLM-large và Whisper large-v3-turbo encoder trong hệ đầy đủ: UA, WA, weighted F1, change in UA, audio gate weight (0.28, 0.51); tag "chosen" ở Whisper

**Lời nói**
"Giữ nguyên transcript Whisper và ngữ cảnh, em chỉ đổi encoder audio. Encoder Whisper turbo cao hơn WavLM ở cả ba metric, UA 81.17 so với 79.93. Gate cũng cho thấy điều này: với WavLM, model chỉ đặt 0.28 trọng số vào nhánh audio; với Whisper, con số là 0.51, tức là model tin nhánh audio nhiều hơn. Kết quả này cùng chiều với benchmark EmoBox. Vì encoder lấy từ đúng checkpoint turbo dùng cho ASR, cả hệ chỉ cần một model Whisper: bản finetune làm ASR, encoder bản gốc đóng băng làm nhánh audio."

**Nguồn**: report, Table IV; Ma et al. 2024 (EmoBox).

---

### Slide 15. Ablation study
**Trên slide**
- Bảng: Our system, Without audio, Without context, Without text; UA, WA, weighted F1, change in UA

**Lời nói**
"Em bỏ từng phần ra khỏi hệ đầy đủ. Bỏ text mất nhiều nhất, 8.58 UA. Bỏ ngữ cảnh mất 5.29. Bỏ audio mất 3.27. Phần nào cũng có đóng góp: text nhiều nhất, rồi ngữ cảnh, rồi audio."

**Nguồn**: report, Table V.

---

### Slide 16. UA per test session
**Trên slide**
- Bảng UA theo session cho Our system, WavLM encoder, Without audio, Without context, Without text

**Lời nói**
"Kết quả có ổn định qua các fold không? Hệ đầy đủ thắng bản bỏ audio, bản bỏ ngữ cảnh và bản bỏ text ở cả 5 session. Encoder Whisper thắng WavLM ở 4 trên 5 session, trừ session 5."

**Nguồn**: report, Table VII.

---

### Slide 17. How ASR errors affect emotion
**Trên slide**
- Bảng hệ đầy đủ với ba nguồn transcript: ground truth (WER 0, UA 81.89), Whisper finetune (9.56, UA 81.17, −0.72), SenseVoice finetune (9.73, UA 81.29, −0.60)

**Lời nói**
"Lỗi ASR tốn bao nhiêu? Em giữ nguyên hệ đầy đủ và chỉ đổi transcript. Với transcript đúng, WER bằng 0, hệ đạt 81.89. Với Whisper finetune, WER 9.56, hệ đạt 81.17, thấp hơn 0.72. Với SenseVoice finetune, WER 9.73, hệ đạt 81.29, thấp hơn 0.60. Nếu bỏ audio, chỉ dùng text và ngữ cảnh, thì text đúng được 78.57 so với 77.91 của transcript Whisper, mất 0.66 UA. Như vậy có hay không có audio, lỗi ASR ở mức WER khoảng 10% chỉ làm mất dưới một điểm UA, ít hơn nhiều so với phần audio đóng góp, 3.27, hay ngữ cảnh, 5.29. Hai transcript finetune chênh nhau 0.12 UA, nhỏ hơn nhiều so với độ dao động giữa các fold, nên UA không phân biệt được hai ASR; em chọn ASR theo WER, đúng tác vụ của nó."

**Nguồn**: report, Analysis; Table V; PROGRESS.md (đợt encoder turbo, `combined_gt_ctx_whisper_turbo*`); Li et al. 2024.

---

### Slide 18. Recall of each class
**Trên slide**
- Bảng recall theo lớp: Our system, Without audio, Without context, Without text
- Gate weight on audio: our system 0.51, without context 0.71, WavLM encoder 0.28

**Lời nói**
"Nhìn theo từng lớp: audio giúp nhiều nhất cho angry, tăng 7.4 recall. Ngữ cảnh giúp happy nhiều nhất, tăng 9.8, rồi angry, tăng 6.2. Lớp sad thay đổi ít nhất khi bỏ phần nào. Gate trong hệ đầy đủ đặt trung bình 0.51 trọng số vào audio. Khi bỏ ngữ cảnh, nhánh text yếu đi nên gate nghiêng hẳn sang audio, 0.71. Với WavLM, gate chỉ đặt 0.28 vào audio."

**Nguồn**: report, Table VI và Analysis.

---

### Slide 19. Comparison with published systems
**Trên slide**
- Bảng các hệ đã công bố: loại hệ, có dùng transcript, audio, ngữ cảnh, WER, UA

**Lời nói**
"Đây là các hệ đã công bố trên IEMOCAP bốn lớp. Cách chia dữ liệu, tập dev và metric khác nhau nên không so trực tiếp được. Trong nhóm có làm cả ASR, hệ của em có WER thấp nhất, 9.56, và UA 81.17. PL-Distill cao hơn về UA nhưng chỉ nhận diện cảm xúc từ audio, không ra transcript. Tóm lại: sau khi finetune, Whisper turbo đạt 9.56% WER; hệ đạt 81.17% UA và chỉ cần một model Whisper cho cả ASR và nhánh audio. Text đóng góp nhiều nhất, rồi ngữ cảnh, rồi audio; lỗi ASR ở mức này chỉ làm mất dưới một điểm UA. Hướng tiếp theo là thử trên giọng nói tự nhiên như MSP-Podcast."

**Nguồn**: report, Table I và Conclusion; Gao et al. 2023; Shen et al. 2024; Chen et al. 2025; He et al. 2024; Ma et al. 2024; Yang et al. 2026; Ueda et al. 2026.
