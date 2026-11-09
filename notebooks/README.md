# Notebook Colab: baseline Emotion-Aware ASR

## Cách dùng

1. Upload cả thư mục `notebooks/` lên Google Drive thành `MyDrive/vp-ck/notebooks/`
   (code dùng chung nằm trong `eaasr/`, notebook import từ đó).
2. Mở notebook từ Drive bằng Colab, chạy từ trên xuống. Ô đầu tiên của mỗi notebook là phần cấu hình (`DATASET`, số epoch, ...).
3. Kết quả lưu ở `MyDrive/vp-ck/results/<dataset>/<system>/` (`metrics_test.json` + `predictions_test.csv`).

| Notebook | Việc | Runtime | Thời gian ước tính (MELD) |
|---|---|---|---|
| `00_data_meld` | tải MELD, tách wav 16 kHz, manifest | CPU | 30–60 phút, chạy 1 lần |
| `01_data_iemocap_hf` | IEMOCAP bản HF + kiểm tra ngẫu nhiên | CPU | 15–30 phút, chạy 1 lần |
| `10_sensevoice` | SenseVoice-Small zero-shot + fine-tune | A100 / L4 | 1–2 giờ |
| `11_mtl_ctc` | wav2vec2 + CTC + emotion head (asr / ser / joint) | A100 | khoảng 1.5 giờ mỗi mode |
| `12_cascade_fusion` | Whisper + emotion2vec + RoBERTa, fusion | A100 | khoảng 1 giờ |
| `13_ent` | ENT (code chính thức) | A100 | vài giờ (20 epoch) |
| `14_audio_llm` | Qwen2-Audio-7B zero-shot | A100 | khoảng 1 giờ |
| `90_compare` | bảng so sánh | CPU | vài giây |

Gợi ý thứ tự: `00` → `12` (nhanh, có ngay WER của Whisper và nhiều số SER) → `10` → `11` → `14` → `13` → `90`.
Thời gian ở trên là ước lượng, chưa đo thật.

## Quy ước đánh giá (giống nhau cho mọi notebook, định nghĩa ở `eaasr/common.py`)

- MELD: chia train/dev/test chính thức, 7 lớp. IEMOCAP: 4 lớp (excited gộp vào happy), test = Session 5,
  dev = 10% ngẫu nhiên của Session 1–4.
- Chọn checkpoint/epoch theo **dev**, không theo test (riêng ENT báo cả hai, xem notebook 13).
- WER: ref và hyp đều qua Whisper English normalizer.
- SER: WA, UA, macro-F1, weighted-F1. Nhãn dự đoán nằm ngoài tập nhãn tính là sai.

## Sửa notebook

File nguồn là `src/*.py` (định dạng `# %%`). Sau khi sửa, chạy:

```
python notebooks/tools/build_notebooks.py
```

để sinh lại các file `.ipynb`.
