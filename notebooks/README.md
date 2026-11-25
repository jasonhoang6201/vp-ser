# Code thực nghiệm: Emotion-Aware ASR trên IEMOCAP

Code cho mọi số liệu trong report: IEMOCAP 4 lớp, leave-one-session-out 5 fold, seed 42.

## Thư mục

| Đường dẫn | Nội dung |
|---|---|
| `src/*.py` | mã nguồn notebook (định dạng `# %%`); `*.ipynb` ở thư mục này sinh ra từ đây |
| `eaasr/common.py` | phần dùng chung: chia fold, WER (Whisper English normalizer), UA/WA/F1, lưu kết quả |
| `tools/prep_iemocap_official.py` | dựng manifest + tar wav 16 kHz từ bản IEMOCAP chính thức (chạy local) |
| `tools/compare_results.py` | tính lại mọi bảng kết quả của report từ `results/` (chạy local) |
| `tools/build_notebooks.py` | sinh `*.ipynb` từ `src/*.py` |
| `results/iemocap/<hệ>[_f<k>]/metrics_test.json` | kết quả từng fold; không hậu tố là test session 5, `_f1`..`_f4` là session 1..4 |
| `results/summary.md` | các bảng của report, sinh bởi `compare_results.py` |
| `demo/`, `30_demo` | chạy hệ đã train (fold S5, weight trên HF) trên vài clip ngoài IEMOCAP |

## Notebook

| Notebook | Bước trong report | Việc | Thời gian mỗi fold (L4) |
|---|---|---|---|
| `10_sensevoice` | Bước 1-2 | SenseVoice-Small zero-shot + fine-tune (WER) | ~40 phút |
| `15_whisper_ft` | Bước 1-2 | Whisper large-v3-turbo zero-shot + fine-tune (WER), transcript Whisper-FT cho mọi câu | ~80 phút |
| `20_sv_transcripts` | so sánh hai ASR | transcript SenseVoice-FT cho mọi câu (đọc checkpoint của `10`) | ~12 phút |
| `21_combined` | Bước 3-4 | hệ đề xuất (RoBERTa + ngữ cảnh + encoder audio, gated fusion) và ablation | ~10-15 phút mỗi cấu hình |
| `30_demo` | demo | tải weight fold S5 từ HF, chạy trên `demo/clips` | vài phút |

## Cách chạy

1. Dựng dữ liệu một lần (IEMOCAP không được phát tán lại, nên không có trong repo):
   `python notebooks/tools/prep_iemocap_official.py <IEMOCAP_full_release> <out>`, rồi chép
   `iemocap_manifest.csv` và `iemocap_16k.tar` vào `MyDrive/vp-ck/data/`.
2. Upload thư mục `notebooks/` lên `MyDrive/vp-ck/notebooks/`, mở notebook bằng Colab (GPU L4).
3. Mỗi notebook chạy cho một fold. Ô đầu tiên là phần cấu hình: đặt `TEST_SESSION` (5, 1, 2, 3, 4) rồi chạy từ trên xuống.
   Kết quả ghi vào `MyDrive/vp-ck/results/iemocap/`, transcript và embedding vào `MyDrive/vp-ck/cache/iemocap/`.
4. Thứ tự cho mỗi fold, và cấu hình của `21_combined` cho từng dòng trong report:

| # | Notebook | Cấu hình (ngoài `TEST_SESSION`) | Kết quả | Dòng trong report |
|---|---|---|---|---|
| 1 | `10_sensevoice` | mặc định | `sensevoice_small_zeroshot`, `sensevoice_small_finetune` | Bảng III, IV |
| 2 | `20_sv_transcripts` | mặc định | `cache/iemocap/asr_sensevoice_ft.csv` | |
| 3 | `15_whisper_ft` | mặc định | `whisper_turbo_zeroshot`, `whisper_turbo_finetune` | Bảng III, IV |
| 4 | `21_combined` | mặc định: `ASR="whisper"`, `AUDIO="whisper_turbo"`, 6 cấu hình | `combined_wasr_ctx_whisper_turbo` | hệ đề xuất |
| | | | `combined_wasr_ctx` | bỏ audio |
| | | | `combined_wasr_whisper_turbo` | bỏ ngữ cảnh |
| | | | `combined_whisper_turbo_only` | bỏ text |
| | | | `combined_gt_ctx_whisper_turbo` | text đúng, hệ đầy đủ |
| | | | `combined_gt_ctx` | text đúng, không audio |
| 5 | `21_combined` | `AUDIO="wavlm"`, `RUNS=["asr_ctx_audio"]` | `combined_wasr_ctx_wavlm` | encoder WavLM (Bảng V) |
| 6 | `21_combined` | `ASR="sensevoice"`, `RUNS=["asr_ctx_audio"]` | `combined_asr_ctx_whisper_turbo` | transcript SenseVoice-FT |

5. (Tùy chọn) Weight: `15` lưu Whisper-FT ở `checkpoints/whisper_turbo_iemocap{FOLD}`, `21` lưu hệ đầy đủ ở
   `checkpoints/ser_wasr_ctx_whisper_turbo{FOLD}`. Cell cuối của hai notebook (đang comment) nhập HF token rồi đẩy
   chúng lên repo HF private `fold_s{k}/asr` và `fold_s{k}/ser`, layout mà `30_demo` đọc.
6. Chép `results/iemocap/` về máy rồi chạy `python3 notebooks/tools/compare_results.py`: các bảng trong
   `results/summary.md` là số trong report (trung bình 5 fold).

## Quy ước đánh giá (`eaasr/common.py`)

- 4 lớp: neutral, happy (gộp excited), angry, sad; 5531 câu có nhãn trên 10039 lượt nói.
- Test là một session; dev là 10% ngẫu nhiên của bốn session còn lại. Mỗi ASR được fine-tune lại cho từng fold,
  nên transcript test không bao giờ đến từ model đã nghe session đó.
- Chọn epoch theo dev (dev WER cho ASR, dev UA cho SER), không theo test.
- WER: ref và hyp đều qua Whisper English normalizer. SER: UA (chính), WA, macro-F1, weighted-F1.

## Sửa notebook

File nguồn là `src/*.py`. Sau khi sửa, chạy `python notebooks/tools/build_notebooks.py` để sinh lại `*.ipynb`.
