# Thực nghiệm Colab: nguồn sự thật (source of truth)

Cập nhật mỗi khi có kết quả, lỗi, hay quyết định mới. Khi context bị compact, đọc file này trước.
Mỗi job đang chạy phải có ETA (giờ Việt Nam, UTC+7). Giờ trên VM là UTC.

## Hạ tầng

- Chạy bằng Colab CLI trong `venv` (`venv/bin/colab`, google-colab-cli 0.7.5), auth oauth2 (token có sẵn ở `~/.config/colab-cli/`).
- Session: `eaasr`, GPU **L4** (23 GB VRAM, 52 GB RAM, 12 CPU, đĩa ~70 GB trống), **1.94 CU/giờ**.
  Số dư lúc bắt đầu (2026-11-01): 126.07 CU. Kiểm tra: `venv/bin/colab usage`.
- VM là máy ảo của Colab (cloud Google), bị xóa khi `colab stop` hoặc khi Colab thu hồi. **Tải `results/` về trước khi stop.**
- Không mount Drive (CLI không làm được `drive.mount` khi không có người). Mọi thứ nằm trên VM:
  `/content/vp-ck/notebooks/eaasr` (code), `/content/vp-ck/data` (manifest + tar), `/content/vp-ck/results`,
  `/content/vp-ck/cache`, `/content/eaasr_data/<dataset>/wav` (wav đã giải nén), `/content/vp-ck/jobs` (log job).
- Runner: `notebooks/tools/colab_run.py`.
  - `--sync`: upload `eaasr/`. Lệnh sync đôi khi treo; cách thay thế:
    `venv/bin/colab upload -s eaasr notebooks/eaasr/common.py /content/vp-ck/notebooks/eaasr/common.py`.
  - `--bg`: chạy cả notebook thành job `ipython` tách rời trên VM, poll log về `notebooks/runs/<name>.log`.
    Dùng cách này cho mọi job dài, vì websocket của CLI hay rớt ("Connection was lost").
  - `--attach NAME [--pid PID]`: bám lại job `--bg` đang chạy sau khi Claude Code/máy local bị tắt (job trên VM không chết).
    PID ghi ở `/content/vp-ck/jobs/<name>.pid` (job 12 chạy trước khi có file này, PID 384576).
  - `--set NAME=expr` thay dòng config; `--start N` bỏ qua các cell trước N; `--list` liệt kê cell.

## Quy ước đánh giá (eaasr/common.py)

- MELD: train/dev/test chính thức, 7 lớp. Loại clip < 0.2 s (158 câu) và > 60 s (2 câu test bị cắt video sai: dia38_utt4 305 s, dia220_utt0 235 s).
- Sau lọc: train 9879, dev 1093, test 2576 (gốc 9989/1109/2610).
- WER: Whisper English normalizer cho cả ref và hyp. SER: WA, UA, macro-F1, weighted-F1; nhãn ngoài tập nhãn tính là sai.
- Chọn model theo dev. ENT báo cả "max trên test" (cách của paper) và "epoch cuối" (công bằng).

## Dữ liệu

| Dataset | Trạng thái | Ghi chú |
|---|---|---|
| MELD | xong, trên VM (`meld_16k.tar` 1.4 GB) | thiếu mp4 dev/dia110_utt7; ffmpeg lỗi train/dia125_utt3; còn 7 transcript có ký tự không phải ASCII (từ mượn như Café), giữ nguyên |
| IEMOCAP (HF AbstractTTS) | 2026-11-02 09:02 bắt đầu chuỗi 01 → 10 → 20 → 12 (3 seed) → 21 (`asr_e2v`, `asr`, `gt2asr`, `gt`) bằng `scratchpad/iem.sh`, log `runs/*_iemocap.log`, ETA ~12:00 | người dùng: chưa có phản hồi USC, dùng bản HF luôn (học tập, không thương mại). 10039 câu khớp gốc; transcript có 2 dấu cách sau dấu câu như file gốc; **không có timestamp** nên IEMOCAP chạy không ngữ cảnh. Có 264 câu chứa tag gốc ([LAUGHTER] 166, [BREATHING] 57, [GARBAGE] 46) nên transcript là bản gốc, không phải ASR.
`major_emotion` = argmax soft vote cho mọi câu (4 lớp 6877, không có "xxx"); notebook 01 bỏ câu hòa phiếu (top1 - top2 ≤ 0.05) còn 5498 (neu 1654, joy 1663, ang 1110, sad 1071; gốc 5531).
IEMOCAP test (session 5, 1209 câu có ref): SV zero-shot WER 11.03, WA 67.18, UA 65.47; **SV fine-tune WER 9.97, WA 71.74, UA 73.09, wF1 71.65**; Whisper turbo WER 16.41;
**emotion2vec+ zero-shot UA 86.17 (WA 86.89): gần như chắc bị rò rỉ** (bản seed train trên EmoBox, có IEMOCAP) nên số audio e2v trên IEMOCAP chỉ để tham khảo; SenseVoice cũng có thể đã thấy IEMOCAP.
Chuỗi HF xong 11:26, đã tải về `results/iemocap/`, VM đã stop (còn 95.09 CU). Notebook 21 IEMOCAP (3 seed, UA ensemble / mean±std):
`asr_e2v` 88.94 / 88.88±0.74 (gate audio 0.82, rò rỉ); `asr` 72.04 / 71.64±0.99; `gt2asr` 71.97 / 70.91±0.93; `gt` 72.78 / 70.49±0.49.
Cascade: e2v probe UA 90.01 (rò rỉ), text ASR probe 67.24, text GT probe 65.72, fusion concat 89.14.
Kết luận: trên IEMOCAP lỗi ASR (WER ~10–16) gần như không hại SER từ text; text fine-tune > probe ~4.5 UA.
Notebook 21 đã thêm `AUDIO = "wavlm"` (WavLM-large, mọi layer, trọng số layer học được) để thay e2v trên IEMOCAP; chưa chạy.
Kết quả và cache bản HF đã dời sang `results/iemocap_hf/`, `cache/iemocap_hf/` (để bản chính thức dùng lại tên `iemocap`).
**Bản chính thức USC** (người dùng đăng ký, tải `dataset/IEMOCAP_full_release_withoutVideos.tar.gz` 12 GB, 14:12 2026-11-02):
giải nén riêng `sentences/wav`, `dialog/EmoEvaluation/Ses*.txt`, `dialog/transcriptions` vào `dataset/iemocap_extract/`;
`tools/prep_iemocap_official.py` (chạy local, chỉ stdlib) → `dataset/iemocap_official/` (manifest + `iemocap_16k.tar` 1.4 GB).
10039 câu, **4 lớp đúng 5531** (neu 1708, joy 1636, ang 1103, sad 1084), manifest có thêm `dialogue_id`, `start`, `end`.
Notebook 20 nay transcribe thêm câu ngoài 4 lớp (split `unlabeled`), notebook 21 lấy ngữ cảnh IEMOCAP từ **mọi** lượt trong
dialogue theo `start` (cả hai người nói). Upload lớn qua `colab upload` làm **VM chết** hai lần (14:19, 14:44, kể cả chia 20 MB),
nên dữ liệu đi qua HF: repo **private** `jasonhoang6201/iemocap-16k` (tar + manifest, license cấm phát tán: không bao giờ để public).
**Chạy bền (resume được)**: `caffeinate -i zsh notebooks/tools/run_iemocap_official.sh` từ thư mục project, log `runs/iem_off_chain.out`.
VM mới tự tải data + trạng thái từ HF (`tools/vm_hf_sync.py restore`, cần `hf auth login` trên Mac, token được upload lên VM);
mỗi 10 phút và cuối mỗi bước đẩy `cache/iemocap`, `results/iemocap` (+ checkpoint SenseVoice-FT sau bước 10) lên repo private
`jasonhoang6201/iemocap-runs`; bước đã có file kết quả trong `notebooks/` thì bỏ qua; notebook 21 bỏ qua seed đã có npz.
VM chết giữa chừng → chạy lại đúng lệnh trên. Thứ tự: 10 → 20 → 21 (`AUDIO="wavlm"`, 7 cấu hình gồm ctx) → 12 → 21 (e2v, tham khảo rò rỉ).
14:46 bắt đầu, ETA ~19:00.
Kết quả chính thức: SV zero-shot WER 11.15, UA 66.01; **SV-FT WER 9.77, WA 71.88, UA 73.69** (1222 câu). Transcript FT đủ 10039 câu.
21 WavLM seed 0 (test UA): asr_ctx_e2v 81.53 (gate audio 0.31), asr_ctx 79.20, asr_e2v 74.04, asr 71.77, gt_ctx 80.87, gt2asr 69.70, gt 70.36.
Sự cố: VM chết 16:37 nhưng `colab_run` retry tới 18:36 mới thoát; 18:49 mạng Mac rớt (DNS) bị nhận nhầm là VM chết.
Đã sửa: `colab_exec_capture` chỉ dừng khi thấy `Session '<s>' not found`, backoff tới 60 s, poll 30 lần; script chain tự `--attach`
nếu job còn sống trên VM. Chạy lại 18:55 (đã attach), ETA: 21 wavlm ~20:25, 12 ~21:00, 21 e2v ~21:35.
Report tổng hợp: `report.html` ở root.
**21 WavLM xong 20:24** (3 seed, test 1241 câu; UA ensemble / mean±std / WA / wF1): asr 71.77 / 70.38±1.69; gt2asr 71.96 / 70.32±0.55;
gt 71.73 / 71.06±0.93; asr_wavlm 74.87 / 73.05±0.87 (gate audio ~0.62); asr_ctx 81.19 / 80.20±1.14; gt_ctx 81.66 / 80.89±0.17;
**asr_ctx_wavlm 83.23 / 82.31±0.86, WA 83.16, wF1 83.24** (gate audio ~0.40). Thư mục kết quả `results/iemocap/combined_*_wavlm`.
Ngữ cảnh +9.4, WavLM +2–3, ASR ≈ GT ở WER ~10.
**Chuỗi xong 22:13 (ALL DONE), VM đã stop.** 12 (UA ens / mean±std): Whisper WER 16.37; text ASR probe 68.64 / 66.72±0.63;
text GT probe 66.45 / 66.77±1.11; e2v zero-shot 86.88; e2v probe 90.80 / 90.99±0.45; fusion concat 91.08, gated 91.12, gated_gt 91.16.
21 e2v: asr_e2v 90.10 / 89.71±1.09 (gate audio 0.78–0.99), asr_ctx_e2v 89.98 / 89.57±0.65. Mọi số e2v ~90 là rò rỉ; báo bản WavLM.
**Đợt 2 (2026-11-03 00:08)**: `caffeinate -i zsh notebooks/tools/run_iemocap_cv.sh` (log `runs/iem_cv_chain.out`; hạ tầng chung ở
`tools/chain_lib.sh`). Notebook 10/12/20/21 có `TEST_SESSION` (hậu tố `_f<k>`, session 5 không hậu tố); 12 có `AUDIO="wavlm"`.
Cascade WavLM session 5 xong 00:16 (UA ens / mean±std): wavlm probe 68.80 / 68.18±1.81; **concat 79.27 / 77.81±1.54**;
gated 76.59 / 75.90±1.65; gated_gt 77.46 / 76.85±1.28. Lưu ý: combined `asr_wavlm` (74.87) thua cascade concat; full 83.23 hơn +4.0.
CV fold 1–4 (10 → 20 → 21 seed 0, 6 cấu hình → 12 wavlm seed 0), ~80 phút/fold, ETA ~05:45.
VM chết 00:21 (Mac sleep) giữa 10 fold 1 (đã có zero-shot f1: WER 13.66, UA 66.20). Chạy lại 08:47 cùng lệnh, ETA ~14:15.
Fold 1: 10 xong 09:22 (38 phút), 20 xong 09:38; 21 chạy 09:38 → 10:51 thì "Connection was lost", chạy lại 10:52.
Thực đo ~2 giờ 15 phút/fold (21 với 6 cấu hình ~70 phút), ETA mới ~19:00.
**Fold 1 xong 11:00** (test session 1, 1 seed, UA): SV zero-shot 66.20 (WER 13.66); SV-FT 73.32 (WER 12.15); Whisper WER 13.70;
combined asr 68.82, gt 70.46, asr_wavlm 72.05, asr_ctx 75.33, gt_ctx 76.09, **asr_ctx_wavlm 76.43**; cascade wavlm probe 70.43,
text asr probe 63.71, text gt 65.36, concat 75.95, gated 77.01, gated_gt 77.16. Khác session 5: cascade gated ≈ combined full.
Fold 2: VM mất 11:38 vì **gập nắp chạy pin (Clamshell Sleep)**, ngay sau khi 20 f2 xong trên VM nhưng chưa kéo về; checkpoint f2
không lưu trên HF nên làm lại từ 10 f2. Chạy lại 12:58 nhưng máy vẫn gập nắp nên setup lỗi; chạy lại 13:46 (đã cắm sạc), ETA ~20:30.
**Fold 2 xong ~15:25** (UA): SV zero-shot 67.87 (WER 11.80); SV-FT 73.32 (WER 10.19); combined asr 75.52, gt 76.07,
asr_wavlm 78.80, asr_ctx 82.47, gt_ctx 85.01, asr_ctx_wavlm 81.78; cascade wavlm 70.98, text asr 70.89, text gt 71.48,
concat 80.20, gated 80.10, gated_gt 74.97. Fold 3: 21 bắt đầu 16:12. ETA cả chuỗi ~19:30.
**Người dùng cho dừng 16:45**, VM đã stop 16:55 (còn 67.67 CU). Fold 3 test UA (seed 0): asr 69.03, asr_e2v(=wavlm) 74.50,
asr_ctx 74.91, asr_ctx_e2v(=wavlm) 78.30. Trên HF có npz asr_ctx, asr_ctx_wavlm, asr_wavlm f3 (asr f3 chưa kịp đẩy, chạy lại ~5 phút),
transcript f3 có ở local nên không phải fine-tune lại SV f3. Chạy tiếp: `caffeinate -i zsh notebooks/tools/run_iemocap_cv.sh`
(còn 21 f3 phần còn lại, 12 f3, rồi cả fold 4, ~2 giờ 50 phút).
Chạy tiếp 21:22, ETA ~00:15 (2026-11-04).
21 f3 xong trên VM ~21:48 (gt_ctx 76.65, gt 69.53) nhưng kết nối rớt khi kéo về (22:20). Chạy lại 22:21, ETA ~00:40.
22:42 lại lỗi: gói kéo về 423 MB đứt giữa chừng, VM mất. Nguyên nhân: `--exclude` đặt sau đường dẫn trong `pull()` của
`chain_lib.sh` nên GNU tar bỏ qua, đóng gói cả cache WavLM/e2v. Đã sửa (đưa `--exclude` lên trước). Kết quả 21 f3 tải thẳng từ HF.
Chạy lại 22:45 (12 f3 → fold 4), ETA ~01:15.
**Fold 3 xong 22:50** (UA): SV zero-shot 64.62 (WER 12.19); SV-FT 71.16 (WER 10.28); combined asr 69.03, gt 69.53, asr_wavlm 74.50,
asr_ctx 74.91, gt_ctx 76.65, **asr_ctx_wavlm 78.30**; cascade wavlm 69.17, text asr 64.89, text gt 66.07, concat 76.54, gated 74.58,
gated_gt 73.84. **Fold 4 xong 01:14 (04/11), ALL DONE, đã stop VM `eaasr`** (UA): SV zero-shot 69.1 (WER 7.20); SV-FT 71.8 (WER 5.97);
combined asr 69.4, gt 71.0, asr_wavlm 73.6, asr_ctx 75.3, gt_ctx 76.5, asr_ctx_wavlm 77.0; cascade wavlm 65.9, text asr 64.6,
text gt 66.7, concat 72.6, gated 74.0, gated_gt 76.0.
**CV 5 fold (UA mean±std, seed 0)**: gt_ctx 79.02±3.87; **asr_ctx_wavlm 79.01±2.51** (hệ ASR tốt nhất); asr_ctx 77.44±3.31;
cascade concat 76.98±3.06, gated 76.67±2.48; asr_wavlm 74.61; SV-FT 72.65±1.11; combined gt 71.48, asr 70.91; SV zero-shot 66.77.
Whisper WER fold 1..5: 13.7 11.5 12.9 11.5 16.4. Tổng hợp: `tools/compare_results.py` (bỏ qua run chỉ có ASR)
→ `results/summary.md`.
Lần 1 (09:02) lỗi ghi manifest vì VM mới chưa có `/content/vp-ck/data`, đã thêm mkdir; chạy lại 09:06. Cache e2v của 21 đổi tên thành `e2v_plus_large_utt.npz` (khác định dạng của 12) |

## Tiến độ baseline (MELD)

| # | Notebook | Trạng thái | Kết quả test |
|---|---|---|---|
| 00 | data_meld | xong lại lúc 11:31 trên session mới (chỉ mất 7 phút); manifest backup ở `notebooks/data_backup/` | |
| 12 | cascade_fusion | xong 10:2x (09:00 → Whisper 09:56, e2v 10:03, classifier ~15 phút). Kết quả ở `notebooks/results/meld/` | xem bảng bên dưới |
| 10 | sensevoice | **xong 12:18**, kết quả đã tải về; checkpoint 2.7 GB chỉ nằm trên VM (`/content/vp-ck/checkpoints/sensevoice_meld`). Lần 2 mất do VM bị thu hồi; chạy lần 3 lúc 11:32 (log `runs/10_sensevoice_r2.log`). Zero-shot xong 11:35 (đã tải về). Fine-tune đo được ~0.34 s/bước, 558 bước/epoch, ~4 phút/epoch, nên **ETA ~12:20** (gồm eval test). Lưu ý: scheduler warmup mặc định 25000 bước > tổng ~5600 bước, nên LR chỉ lên tới khoảng 4.5e-5, chưa bao giờ đạt 2e-4 | zero-shot đã có (bảng dưới) |
| 11 | mtl_ctc | **xong 14:01**, cả 3 mode đã tải về. Lần 4 chạy ~12:37 với `BATCH_SECONDS=80` (L4), ~3 phút/epoch. Ba lần trước hỏng: (1) LayerDrop làm thiếu hidden state, sửa `layerdrop=0.0`; (2) grad cosine lệch kích thước khi một loss không dùng tham số, sửa bằng zero-fill; (3) head SER sập về một lớp (dev UA 14 = 1/7), sửa bằng layer norm từng layer trước weighted sum (như s3prl). Sau sửa: dev UA ep1 15.1, ep2 26.5 | |
| 14 | audio_llm (Qwen2-Audio) | **xong 14:33** (batch 8 trên L4, ~0.44 s/câu ASR). Bỏ câu dẫn chạy lại riêng bằng process Python mới (kernel chính của CLI bị lỗi numpy do pip nâng cấp giữa chừng). Kết quả đã tải về | xem bảng |
| 13 | ent | **xong 17:16**, kết quả + `runs/ent_train.log` đã tải về. Smoke test ổn 14:39 (pipeline chạy hết, 1 epoch 200 câu nên WER 100 là bình thường). Chạy đầy đủ từ 14:42 với `EPOCHS=10` (giảm từ 20 cho đỡ tốn CU). Đo được ~27 phút/epoch (GPU chỉ ~12%, nghẽn ở dataloader/CPU), **ETA ~19:15**. Epoch 0 trên test: WER 84.5, WA 34.8, UA 20.5. Log epoch nằm ở `/content/ent_train.log` trên VM, không ở log job | |

## Kết quả MELD test (n = 2576, sau lọc)

| Hệ thống | WER | WA | UA | wF1 | UA 5 seed (mean±std) | Ghi chú |
|---|---|---|---|---|---|---|
| Whisper large-v3-turbo | 31.83 | | | | | WER theo cảm xúc: surprise 47.2, joy 37.9, fear 33.1, neutral 31.2, anger 26.5, sadness 25.0, disgust 21.6 |
| emotion2vec+ large zero-shot | | 45.92 | 30.09 | 45.83 | | lệch về neutral |
| SenseVoice-Small zero-shot | 29.19 | 52.18 | 28.63 | 49.31 | | WER thấp hơn Whisper turbo; WER theo cảm xúc đều hơn (fear 38.6, surprise 33.7, còn lại 26–29) |
| e2v probe (audio) | | 37.84 | 30.84 | 41.53 | 30.94±0.49 | class weight làm WA giảm |
| text probe, ASR transcript | 31.83 | 38.46 | 38.11 | 40.87 | 38.01±1.82 | |
| text probe, GT text (cận trên) | | 55.21 | 47.57 | 57.72 | 46.47±0.84 | |
| fusion concat (e2v + ASR text) | 31.83 | 44.29 | 38.17 | 46.68 | 35.78±1.64 | |
| fusion gated (e2v + ASR text) | 31.83 | 43.28 | 36.69 | 46.20 | 35.44±1.22 | |
| fusion gated (e2v + GT text) | | 53.69 | 45.16 | 56.13 | 43.36±0.93 | |
| MTL wav2vec2-large CTC + SER, joint α=0.5 (best dev ep theo UA−WER) | 39.77 | 35.55 | 32.36 | 38.56 | | Dev WER dao động 39–45 giữa các epoch. Grad cosine CTC vs CE (17 điểm, mỗi 200 bước): trong [−0.11, 0.14], trung bình ≈ −0.01, 10/17 âm, tức gần trực giao, hơi xung đột |
| MTL, chỉ SER (α=0, best dev UA, ep5) | 52.44 (quên ASR) | 38.89 | 34.50 | 41.31 | | Dev UA đỉnh 36.0, ngang joint (36.7 ở ep6). Joint test UA thấp hơn chủ yếu vì tiêu chí chọn epoch UA−WER ưu tiên WER (chọn ep4). CTC loss trong joint giữ WER ~44 thay vì ~50 |
| MTL, chỉ ASR (α=1, best dev WER, ep4) | 38.42 | | | | | Chỉ hơn joint 1.35 WER. Dev WER cũng dao động 37–43 dù chỉ có CTC, nên dao động không do SER. WER theo cảm xúc: fear 48.2, surprise 46.6, joy 40.6, còn lại 32–37 |
| Qwen2-Audio-7B-Instruct zero-shot (raw) | 53.28 | 47.05 | 38.73 | 48.46 | | 1032/2574 câu ASR có câu dẫn ("The transcription is: '...'"), vài câu từ chối |
| Qwen2-Audio-7B zero-shot, bỏ câu dẫn | 31.87 | 47.05 | **38.73** | 48.46 | | WER ngang Whisper turbo. **UA audio cao nhất tới giờ**, ngang text probe trên transcript ASR (38.1). Có thể đã thấy MELD lúc train (rò rỉ dữ liệu). Surprise WER 51.1 |
| ENT (code chính thức, wav2vec2-base đóng băng + RNN-T, 10 ep, epoch cuối) | 56.77* | 47.44 | 26.57 | | | *WER theo chuẩn hóa riêng của ENT (chữ hoa, bỏ dấu), không phải Whisper normalizer. Kiểu paper (best trên test qua các epoch): WA 49.96, UA 27.50. Từ epoch 5 test loss tăng (overfit). ENT không xuất prediction nên không có F1 |
| SenseVoice-Small fine-tune (10 ep, epoch cuối) | **27.39** | 54.47 | 31.34 | 51.81 | | WER giảm 1.8, UA tăng 2.7 so với zero-shot. WER theo cảm xúc: fear 34.9, surprise 30.5, còn lại 22.7–27.8. Không chọn theo dev được (FunASR không có dev acc cho SenseVoice), nên dùng epoch cuối. LR thực tế ≤ ~4.5e-5 do warmup 25000 bước |

Cột WA/UA/wF1 là ensemble 5 seed. Nhận xét sơ bộ:
- Ở WER ~32% (MELD), text từ ASR mất khoảng 8–9 điểm UA so với text đúng (38 so với 46.5). Khác kết luận của li-2024
  ("WER ~10% gần như vô hại"), vì WER ở đây cao gấp 3.
- Text mạnh hơn audio trên MELD (emotion2vec probe UA ~31). Fusion chưa hơn text-only về UA, chỉ hơn về wF1, và gated
  không hơn concat (chênh nằm trong std). Ứng viên cải tiến: probe emotion2vec frame-level thay vì utterance, fine-tune text model.
- WER theo cảm xúc: surprise/joy cao nhất, nhưng câu surprise ngắn; cần tách ảnh hưởng của độ dài câu.

## Sự cố đã gặp

- 2026-11-01: `colab exec` chạy cell dài bị "Connection was lost", nên chuyển sang chạy job nền (`--bg`).
- 2026-11-01 10:40: cài `SenseVoice/requirements.txt` làm hạ numpy về 1.26 và nâng torch lên 2.15 (cu132, lệch torchaudio cu130),
  hỏng pandas/torchaudio. Đã sửa: `pip install "numpy>=2.1,<2.6"` và `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130`.
  Notebook 10 không còn cài file đó. **Không cài requirements.txt của repo ngoài khi chưa kiểm tra pin.**
  Môi trường chuẩn trên VM: numpy 2.5.4, pandas 2.2.3, torch/torchaudio 2.11.0+cu130, transformers 5.22.1, funasr 1.4.19, Python 3.13.
- `colab exec` trực tiếp đôi khi treo không trả về (không lỗi). Dùng `colab_exec_capture` (có timeout + retry) trong `tools/colab_run.py`.
- 2026-11-01: Claude Code khởi động lại (resume), các poller local chết nhưng job trên VM vẫn chạy; đã `--attach` lại.
- 2026-11-01 ~11:16: **Colab thu hồi VM giữa lúc SenseVoice fine-tune** (session biến mất, host không resolve được),
  dù poller vẫn đang poll mỗi 30 s. Mất dữ liệu MELD, cache (ASR Whisper, e2v, RoBERTa) và fine-tune dở. Kết quả
  notebook 12 đã tải về trước nên còn. 11:23 tạo session mới, chạy lại 00 rồi 10.
  Bài học: **sau mỗi job tải ngay `results/` (và cache nhỏ) về máy**, không để dồn tới cuối.
- 2026-11-01 17:20–20:54: session rảnh bị Colab tắt (idle).
- 2026-11-01 ~21:23: **VM bị thu hồi lần 2**, giữa lúc notebook 21 tính e2v (job chạy từ 21:11). Nguyên nhân: người dùng gập nắp Mac
  (clamshell sleep, pmset ghi 21:26). **Khi chạy job phải giữ máy thức, không gập nắp.** Chưa có kết quả nào. Log cũ: `runs/21_combined_lost1.log`.
  Từ đây: script `scratchpad/pull21.sh` (chạy trong Monitor) tải từng `combined/*.npz` và `e2v_plus_large.npz` về
  `notebooks/cache/meld/` mỗi 2 phút; `scratchpad/setup.sh` dựng session mới, upload lại các file này (21 sẽ bỏ qua run đã xong), rồi chạy 00 → 21.

## Giả thuyết cho model kết hợp (chờ số baseline)

- Gradient conflict giữa CTC và CE (notebook 11 đo cosine).
- Transcript ASR có lỗi so với text đúng: còn giúp SER bao nhiêu (notebook 12).
- Gated fusion audio (emotion2vec) + text.

## Nhắc cuối phiên

- Nếu người dùng đã gõ `/autocompact 300k` (chỉ muốn áp dụng cho session này), nhắc họ gõ `/autocompact auto` khi xong.
- `venv/bin/colab stop -s eaasr` sau khi đã tải `results/` về, để không tốn compute units.

## Chuẩn bị model kết hợp

- `cache/meld/asr_sensevoice_ft.csv` (đã tải về local): transcript SenseVoice fine-tune cho mọi split (notebook 20).
  WER train 23.6, dev 24.2, test 27.3, nên train không quá sạch so với test, dùng được để train nhánh text chịu lỗi ASR.
- Đề xuất v1: SenseVoice-FT ASR → RoBERTa-large fine-tune trên transcript ASR + emotion2vec+ (+ token cảm xúc SV),
  ablation: frozen/fine-tune, train trên ASR text/GT text, có/không audio.
- Người dùng chọn **(b)**: ngữ cảnh hội thoại là một ablation, báo cả có lẫn không có. Ngữ cảnh = 3 câu **trước** trong dialogue,
  lấy từ transcript ASR (không dùng text đúng), kèm tên người nói (nhãn MELD).
- Lúc 17:20 tất cả baseline MELD đã xong và tải về. Khoảng 20:54 phát hiện session `eaasr` đã bị Colab tắt khi rảnh
  (mất checkpoint SenseVoice-FT trên VM, không sao vì transcript đã có local). 20:55 tạo session L4 mới,
  upload `eaasr/` + `asr_sensevoice_ft.csv`, chạy lại 00 (log `runs/00_data_meld_r3.log`).
- Notebook **21_combined**: RoBERTa-large fine-tune + gated fusion e2v, 6 cấu hình (`gt2asr`, `asr`, `asr_ctx`, `asr_e2v`,
  `asr_ctx_e2v`, `gt_ctx`) × 3 seed, vòng ngoài là seed. Mỗi lần chạy lưu `cache/meld/combined/<cfg>_s<seed>.npz`
  (chạy lại sẽ bỏ qua cái đã xong). Cuối notebook đóng gói `/content/combined_out.tgz` để tải về.
- 22:06 lần chạy đầu `asr_ctx_e2v` s0 (không loss phụ): ~150 s/epoch, dev UA 34.0/33.9/34.4/35.1/34.1, test UA **31.71**,
  gate (text, audio) = (0.12, **0.88**): gate sập về audio, RoBERTa gần như không học (bằng e2v probe 30.8).
  Lưu lại thành `combined/collapsed_asr_ctx_e2v_s0.npz` (ablation "không loss phụ"), log `runs/21_combined_collapse.log`.
  Sửa: thêm head phụ cho từng nhánh (loss = CE fused + CE text + CE audio) và LayerNorm cho vector text. 22:21 chạy lại.
- Seed 0 xong 23:33 (~12.5 phút/run có ngữ cảnh, ~11 phút không ngữ cảnh). Test UA (best epoch theo dev):

  | cfg | best ep | dev UA | test UA | gate audio |
  |---|---|---|---|---|
  | asr_ctx_e2v (không loss phụ, sập) | 3 | 35.05 | 31.71 | 0.88 |
  | gt2asr | 2 | 43.38 | 39.36 | |
  | asr | 4 | 40.99 | 39.40 | |
  | asr_ctx | 2 | 46.06 | 40.66 | |
  | asr_e2v | 4 | 45.15 | 40.08 | 0.44 |
  | **asr_ctx_e2v** | 4 | 46.07 | **41.40** | 0.38 |
  | gt_ctx (cận trên) | 2 | 56.50 | 53.32 | |

  Nhận xét: ngữ cảnh +1.3, audio +0.7, cộng gần độc lập; train text đúng hay ASR như nhau trên test; bản có audio đạt best ở epoch cuối (có thể thiếu epoch).
  Ngữ cảnh giúp text đúng nhiều hơn hẳn (probe GT 47.6 → gt_ctx 53.3) so với text ASR (+1.3): lỗi ASR trong ngữ cảnh làm mất phần lớn lợi ích.
  Seed 1 bắt đầu 23:33, ETA đủ 3 seed ~01:45.
- **Xong 01:55 (2026-11-02)**, đã tải về `results/meld/combined_*` và `cache/meld/combined/*.npz`; VM đã `colab stop` (còn 99.64 CU).
  Kết quả cuối (MELD test, ensemble 3 seed; mean±std là từng seed; WER 27.34 = SenseVoice-FT):

  | cfg | WA | UA (ensemble) | wF1 | UA mean±std | wF1 mean±std | gate audio |
  |---|---|---|---|---|---|---|
  | gt2asr | 49.84 | 41.29 | 51.19 | 40.11±1.33 | 49.61±1.31 | |
  | asr | 50.04 | 41.04 | 51.34 | 40.09±0.59 | 49.62±0.21 | |
  | asr_e2v | 48.87 | 40.47 | 50.41 | 40.50±0.70 | 50.01±0.65 | 0.43 |
  | asr_ctx | 49.15 | 41.54 | 50.57 | 40.87±0.93 | 49.24±0.60 | |
  | **asr_ctx_e2v** | **51.13** | **43.21** | **52.38** | **41.98±0.51** | **51.75±0.88** | 0.41 |
  | gt_ctx (cận trên) | 62.94 | 55.84 | 64.28 | 53.77±0.40 | 62.18±1.49 | |

## Việc kế tiếp

1. Chờ 12 xong, ghi kết quả vào bảng.
2. Chạy lần lượt 10, 11, 14, 13 (smoke trước), rồi 90_compare; tải `results/` về `notebooks/results/`.

## Kiểm tra WER IEMOCAP thấp (2026-11-04)

WER SV-FT ~9.7 (CV) là đúng theo cách đo của project, không phải bug. Tính lại từ `results/iemocap/*/predictions_test.csv`
bằng normalizer cơ bản (lowercase, bỏ dấu câu và tag [..], KHÔNG dùng Whisper normalizer):
- Session 5 SV-FT: 11.63 với normalizer cơ bản so với 9.77 với Whisper normalizer, nên riêng normalizer đã giảm ~2 điểm
  (nó chuẩn hóa contraction, số, cách viết và bỏ từ đệm uh/um/hmm).
- WER theo fold dao động mạnh, fold 4 thấp với mọi hệ (SV-FT 6.55, Whisper 12.37).
- Gộp 5 fold, normalizer cơ bản: script / impro = SV-FT 8.17 / 13.45, SV zero-shot 9.71 / 15.92, Whisper turbo 8.46 / 20.67.
  Fine-tune giảm impro nhiều hơn script, nên không thấy dấu hiệu học thuộc câu kịch bản (5 session dùng chung 3 kịch bản).
- Test chỉ gồm câu có nhãn 4 lớp (labeled_only), không phải toàn bộ ~10k câu.
- Fine-tune SV mỗi fold loại session test (get_splits), nên không rò rỉ giữa train và test.
- Khi so WER với paper khác phải ghi rõ normalizer và tập câu; Li et al. Interspeech 2023 báo Whisper 12.3 trên IEMOCAP.

## Thử ý tưởng từ leaderboard sota2 (2026-11-05)

Nguồn và kiểm chứng paper: `.claude/research/papers.md`. Sửa `src/21_combined.py` (`src/*.py` mới là bản chạy, `.ipynb` cũ từ 02/11):
`AUDIO="whisper"` (encoder Whisper large v3, mọi layer, cache `whisper_large_v3_layers.npz`, không kéo về local) và cấu hình
`asr_ctx_pros`, `asr_ctx_pros_e2v` (tiền tố "pitch/volume/speed" z-score theo người nói, ngưỡng ±0.5 đặt trước, cache `prosody_raw.csv`).
Lưu ý: z-score dùng thống kê của chính người nói test, không dùng nhãn; Whisper không rõ có thấy IEMOCAP lúc train hay không.
Chạy: `caffeinate -i zsh notebooks/tools/run_iemocap_sota_ideas.sh` (log `runs/iem_ideas_chain.out`), session 5, 3 seed, bắt đầu 09:05, **ETA ~11:15**.
So với: `asr_ctx_wavlm` 83.23 / 82.31±0.86 và `asr_ctx` 81.19 / 80.20±1.14 (UA ensemble / mean±std).
Tiêu chí chạy đủ 5 fold (đặt trước khi có kết quả): mean hơn baseline quá 1 std của baseline.
**Session 5 xong 10:18** (n = 1241; UA ensemble / mean±std; McNemar exact trên dự đoán ensemble so với baseline):
`asr_ctx_whisper` 85.04 / 83.46±1.31, p = 0.014 (26 câu chỉ WavLM đúng, 48 câu chỉ Whisper đúng), hơn WavLM ở cả 3 seed;
`asr_ctx_pros` 82.96 / 81.66±0.89, p = 0.239 (54/68); `asr_ctx_pros_wavlm` 84.03 / 83.33±0.57, p = 1.000 (39/39, WA không đổi, UA tăng do dồn lại giữa các lớp).
Cả ba qua tiêu chí, nhưng chỉ Whisper có ý nghĩa thống kê. Gate audio pros_wavlm 0.23/0.38/0.17 so với WavLM 0.31/0.48/0.40.
Chạy tiếp fold 1–4 (seed 0, cùng script, resume được), bắt đầu 10:19 (log `runs/iem_ideas_cv_chain.out`), **ETA ~12:30**.
**10:26 chuỗi local bị kill** cùng session Claude Code đã khởi chạy nó (background task và monitor đều `killed` lúc 10:26:19), khi job whisper f1 sắp xong.
Job trên VM vẫn chạy hết: whisper f1 `[JOB DONE]` lúc 10:26, test UA 77.01, WA 74.56, wF1 73.79, gate audio 0.59 (`asr_ctx_wavlm` f1: 76.43). Kết quả mới nằm trên VM, chưa kéo về local.
VM `eaasr` rảnh, vẫn tốn 1.94 CU/giờ (còn 58.06 CU lúc 10:27). Còn lại: pros f1, rồi whisper + pros f2–f4. Chạy tiếp bằng cùng lệnh; không chạy làm background task của session Claude Code.
Chạy lại 10:29, tách khỏi session (`nohup` + `setsid`, PID 77666, ghi tiếp vào `runs/iem_ideas_cv_chain.out`), **ETA ~12:00**.
Pros f1 xong trên VM 10:47 (test UA): `asr_ctx_pros` 76.99 (`asr_ctx` f1: 75.33), `asr_ctx_pros_wavlm` 75.14, gate audio 0.13 (`asr_ctx_wavlm` f1: 76.43).
10:47–11:00 `colab exec` của bước `push` treo ở local dù VM đã hết tiến trình push (`push --wait` có timeout 31 phút nên sẽ chờ tới ~11:18); kill riêng lệnh đó thì chuỗi đi tiếp. **ETA mới ~12:10**.
11:05 `STEP FAILED pros_f1`: lệnh tar trong `pull` bị "Connection was lost", nên tải về gói cũ 10:33 mà vẫn in "pulled". Kết quả còn trên VM. Chạy lại 11:06, **ETA ~12:15**.
11:06–11:27 setup mất 21 phút; 11:28–11:47 `colab_run` có lúc treo rồi tự hồi, chưa rõ nguyên nhân.
Test UA seed 0, whisper: f2 85.41 (gate audio 0.43), f3 78.66 (gate audio 0.51).
Test UA seed 0, pros: f2 `asr_ctx_pros` 84.59, `asr_ctx_pros_wavlm` 85.01 (gate audio 0.38); f3 `asr_ctx_pros` 75.06, `asr_ctx_pros_wavlm` 77.50 (gate audio 0.19).
12:40 `STEP FAILED pros_f3`: job đã xong (`[JOB DONE]`) thì session `eaasr` mất (404/401), VM bị thu hồi nên `pull` không tải được, chưa rõ nguyên nhân. Chạy lại 12:41 trên VM mới, xong 12:52.
**Lưu ý f3:** `asr_ctx_pros` được khôi phục từ HF nên giữ 75.06, còn `asr_ctx_pros_wavlm` chưa kịp push nên train lại. Cùng seed 0 mà lần đầu ra 78.94 (best ep3, gate audio 0.27), lần sau ra 77.50 (best ep4, gate audio 0.19), lệch 1.44 UA. Vậy seed cố định không đảm bảo lặp lại kết quả, và các chênh lệch dưới ~1.5 UA giữa hai lần chạy đơn lẻ chưa đủ để kết luận. Dùng 77.50 làm số chính thức vì đây là bản có file kết quả; 78.94 chỉ còn trong log, predictions đã mất cùng VM.
Test UA seed 0, f4: `asr_ctx_whisper` 81.10 (gate audio 0.45), `asr_ctx_pros` 78.96, `asr_ctx_pros_wavlm` 80.02 (gate audio 0.22).
**ALL DONE 13:17.** Kết quả đã về `results/iemocap/`, `results/summary.md` đã cập nhật, VM đã `colab stop` (còn 51.64 CU).

**CV 5 fold mới (UA mean±std, seed 0)**: **`asr_ctx_whisper` 81.06±3.37** (wF1 79.77±3.75), `asr_ctx_pros_wavlm` 80.33±4.20, `asr_ctx_pros` 79.25±3.65; so với `asr_ctx_wavlm` 79.01±2.51, `asr_ctx` 77.44±3.31.
So sánh cặp (sign test trên 5 fold; McNemar exact gộp fold 1-4, n = 4290, vì fold 5 chỉ lưu predictions ensemble):
- Whisper so với `asr_ctx_wavlm`: +2.05 (+0.57 +3.64 +0.35 +4.12 +1.59), thắng 5/5, sign p = 0.0625, McNemar p = 0.0074 (300/237).
- `asr_ctx_pros` so với `asr_ctx`: +1.81 (+1.66 +2.13 +0.15 +3.66 +1.45), thắng 5/5, sign p = 0.0625, McNemar p = 0.0028 (311/240).
- `asr_ctx_pros_wavlm` so với `asr_ctx_wavlm`: +1.33 (-1.30 +3.23 -0.80 +3.03 +2.46), thắng 3/5, sign p = 1.000, McNemar p = 0.967 (286/288).
- Whisper so với `asr_ctx_pros_wavlm`: +0.73, thắng 4/5, sign p = 0.375, McNemar p = 0.0053 (296/231).

Kết luận: thay WavLM bằng Whisper có lợi nhất quán; prosody giúp hệ chỉ có text, nhưng thêm vào hệ đã có WavLM thì không rõ lợi ích (số câu đúng gần như không đổi).
Giới hạn: fold 1-4 mỗi fold một lần chạy, nhiễu cùng seed đã thấy 1.44 UA; 5 fold nên sign test không thể xuống dưới 0.0625; các p chưa hiệu chỉnh cho nhiều phép so sánh; McNemar gộp coi các câu là độc lập.

### Chuỗi tiếp theo: seed, pooling, Whisper + prosody (2026-11-05)

Chạy: `caffeinate -i zsh notebooks/tools/run_iemocap_seeds_pool.sh` (log `runs/iem_seeds_pool_chain.out`), mỗi cấu hình 3 seed.
1. Seed 1, 2 cho fold 1-4 của `asr_ctx_whisper` và `asr_ctx_wavlm`. Seed 0 dùng lại file đã lưu. `metrics_test.json` và `predictions_test.csv` của các fold này sẽ thành bản ensemble 3 seed; số seed 0 vẫn nằm trong `seeds` của metrics.
2. Pooling trên session 5, `POOL="attn"` so với `POOL="fmean"`. Thêm vào `src/21_combined.py`: frame Whisper layer 8, 16, 24, 32 ở 12.5 Hz (cache `.npy`, không lên HF), layer weight học được như cũ. `fmean` dùng đúng feature đó với mean có mask, nên phần chênh chỉ đến từ pooling.
3. `asr_ctx_pros_whisper` trên 5 fold, so với `asr_ctx_whisper` 3 seed.

Tiêu chí chạy attention pooling cho fold 1-4 (đặt trước khi có kết quả): UA mean của `attn` hơn `fmean` quá 1 std của `fmean` trên session 5.

**Tạm dừng 2026-11-05 17:35 theo yêu cầu người dùng.** VM `eaasr` đã tắt, số dư 45.84 CU. Xong fold 1-3 của mục 1, kết quả có ở local và HF.

| Fold | Whisper seed 0, 1, 2 | mean ± std | WavLM seed 0, 1, 2 | mean ± std |
|---|---|---|---|---|
| 1 | 77.01, 77.39, 79.67 | 78.02 ± 1.44 | 76.43, 76.25, 76.83 | 76.50 ± 0.29 |
| 2 | 85.41, 85.66, 86.03 | 85.70 ± 0.31 | 81.78, 85.33, 83.31 | 83.47 ± 1.78 |
| 3 | 78.66, 79.71, 80.61 | 79.66 ± 0.98 | 78.30, 75.07, 74.47 | 75.95 ± 2.06 |

Chạy lại 18:53 trên VM mới. Mục 1 xong 19:27, kết quả đủ 5 fold, mỗi fold 3 seed:

| Fold | Whisper mean ± std | ensemble | WavLM mean ± std | ensemble | Chênh mean |
|---|---|---|---|---|---|
| 1 | 78.02 ± 1.44 | 79.02 | 76.50 ± 0.29 | 77.37 | +1.52 |
| 2 | 85.70 ± 0.31 | 87.36 | 83.47 ± 1.78 | 83.96 | +2.23 |
| 3 | 79.66 ± 0.98 | 80.94 | 75.95 ± 2.06 | 79.93 | +3.71 |
| 4 | 81.53 ± 2.11 | 84.32 | 77.52 ± 0.77 | 78.56 | +4.01 |
| 5 | 83.46 ± 1.31 | 85.04 | 82.31 ± 0.86 | 83.23 | +1.15 |
| CV | 81.67 ± 3.03 | 83.34 | 79.15 ± 3.49 | 80.61 | +2.53 |

Whisper thắng 5/5 fold, sign test p 0.0625 (mức thấp nhất có thể với 5 fold). McNemar trên ensemble, 5531 utterance: 309 câu chỉ Whisper đúng, 166 câu chỉ WavLM đúng, p 5.2e-11. McNemar coi utterance độc lập nên p này lạc quan; chưa hiệu chỉnh đa so sánh; vẫn chưa rõ Whisper có thấy IEMOCAP lúc pretrain không.

Mục 2 xong 20:27, session 5, 3 seed:

| Pooling | Seed 0, 1, 2 | UA mean ± std | Ensemble | Gate audio |
|---|---|---|---|---|
| Mean cũ (layer gộp sẵn) | | 83.46 ± 1.31 | 85.04 | |
| `fmean` (mean có mask trên frame) | 83.27, 83.85, 84.96 | 84.03 ± 0.86 | 84.79 | 0.40 |
| `attn` | 83.72, 83.45, 84.14 | 83.77 ± 0.34 | 85.15 | 0.50 |

Không đạt tiêu chí (cần `attn` trên 84.89), nên không chạy attention cho fold 1-4. Attention pooling không cải thiện UA trên session 5; ensemble hơn `fmean` 0.36, nằm trong nhiễu. Kết luận chỉ dựa trên một fold, 3 seed.

Mục 3 xong 22:30, chuỗi `ALL DONE`, VM đã tắt. `asr_ctx_pros_whisper`, 3 seed mỗi fold, so với `asr_ctx_whisper`:

| Fold | Whisper + prosody mean ± std | ensemble | Whisper mean ± std | ensemble | Chênh mean |
|---|---|---|---|---|---|
| 1 | 77.44 ± 1.40 | 78.20 | 78.02 ± 1.44 | 79.02 | -0.58 |
| 2 | 86.56 ± 0.96 | 87.60 | 85.70 ± 0.31 | 87.36 | +0.86 |
| 3 | 78.66 ± 0.94 | 79.46 | 79.66 ± 0.98 | 80.94 | -1.00 |
| 4 | 83.63 ± 0.70 | 85.53 | 81.53 ± 2.11 | 84.32 | +2.10 |
| 5 | 84.67 ± 1.19 | 85.32 | 83.46 ± 1.31 | 85.04 | +1.21 |
| CV | 82.19 ± 3.95 | 83.22 | 81.67 ± 3.03 | 83.34 | +0.52 |

Thắng 3/5 fold, sign test p 1.0. McNemar trên ensemble: 187 câu chỉ prosody đúng, 225 câu chỉ không prosody đúng, p 0.068. Prosody không cải thiện rõ khi audio là Whisper; hai chiều mean và ensemble còn ngược dấu. `asr_ctx_whisper` vẫn là hệ thống chính vì đơn giản hơn. So với `asr_ctx_pros_wavlm` chưa so 3 seed được: hệ đó chỉ có seed 0.

Sự cố kết nối: `colab exec` ngay sau một job dài hay rớt `Connection was lost` hoặc treo, kể cả khi job trên VM đã xong (lần 17:12 phải kill tay). Đã sửa `chain_lib.sh`: `pull` có retry và chỉ tải khi có `tar ok`; `push` chạy nền trên VM và được hỏi bằng lệnh ngắn.

## Ý tưởng chưa làm (người dùng sẽ quay lại nếu cần)

- **Tận dụng phần cảm xúc của SenseVoice** (2026-11-04). Model combined hiện chỉ lấy transcript của SenseVoice-FT;
  token cảm xúc và vector ẩn của nó bị bỏ. Đề xuất v1 có ghi "+ token cảm xúc SV" nhưng notebook 21 không đưa vào.
  Lý do bỏ nghe hợp lý vì SV-FT yếu hơn (CV UA 72.65 so với asr_ctx_wavlm 79.01), nhưng chưa kiểm chứng.
  Hai cách rẻ:
  (1) thêm xác suất 4 lớp cảm xúc của SV làm nhánh thứ ba vào gate (ablation trong notebook 21);
  (2) dùng hidden state encoder SV thay hoặc thêm cạnh WavLM.
  Lưu ý: SenseVoice có thể đã thấy IEMOCAP lúc pretrain, nên nếu UA tăng thì một phần có thể do rò rỉ dữ liệu.
