# Tổng hợp kết quả (sinh tự động bởi tools/compare_results.py)

## MELD test

| system | WER | WA | UA | macro-F1 | wF1 | UA mean±std | seeds |
|---|---|---|---|---|---|---|---|
| combined_gt_ctx |  | 62.94 | 55.84 | 51.04 | 64.28 | 53.77±0.40 | 3 |
| cascade_text_gt_probe |  | 55.21 | 47.57 | 42.22 | 57.72 | 46.47±0.84 | 5 |
| cascade_fusion_gated_gt |  | 53.69 | 45.16 | 40.73 | 56.13 | 43.36±0.93 | 5 |
| combined_asr_ctx_e2v | 27.34 | 51.13 | 43.21 | 40.19 | 52.38 | 41.98±0.51 | 3 |
| combined_asr_ctx | 27.34 | 49.15 | 41.54 | 38.03 | 50.57 | 40.87±0.93 | 3 |
| combined_gt2asr | 27.34 | 49.84 | 41.29 | 38.17 | 51.19 | 40.11±1.33 | 3 |
| combined_asr | 27.34 | 50.04 | 41.04 | 38.26 | 51.34 | 40.09±0.59 | 3 |
| combined_asr_e2v | 27.34 | 48.87 | 40.47 | 37.40 | 50.41 | 40.50±0.70 | 3 |
| qwen2_audio_7b_zeroshot | 53.28 | 47.05 | 38.73 | 36.22 | 48.46 |  |  |
| qwen2_audio_7b_zeroshot_stripped | 31.87 | 47.05 | 38.73 | 36.22 | 48.46 |  |  |
| cascade_fusion_concat | 31.83 | 44.29 | 38.17 | 33.80 | 46.68 | 35.78±1.64 | 5 |
| cascade_text_asr_probe | 31.83 | 38.46 | 38.11 | 31.70 | 40.87 | 38.01±1.82 | 5 |
| cascade_fusion_gated | 31.83 | 43.28 | 36.69 | 33.00 | 46.20 | 35.44±1.22 | 5 |
| mtl_ctc_ser | 52.44 | 38.89 | 34.50 | 30.25 | 41.31 |  |  |
| mtl_ctc_joint | 39.77 | 35.55 | 32.36 | 27.62 | 38.56 |  |  |
| sensevoice_small_finetune | 27.39 | 54.47 | 31.34 | 32.19 | 51.81 |  |  |
| cascade_e2v_probe |  | 37.84 | 30.84 | 28.59 | 41.53 | 30.94±0.49 | 5 |
| emotion2vec_plus_large_zeroshot |  | 45.92 | 30.09 | 28.71 | 45.83 |  |  |
| sensevoice_small_zeroshot | 29.19 | 52.18 | 28.63 | 29.05 | 49.31 |  |  |
| ent | 56.77 | 47.44 | 26.57 | nan | nan |  |  |
| mtl_ctc_asr | 38.42 |  |  |  |  |  |  |
| whisper_whisper-large-v3-turbo | 31.83 |  |  |  |  |  |  |

### WER theo cảm xúc (%)

| system | neutral | joy | surprise | anger | sadness | disgust | fear |
|---|---|---|---|---|---|---|---|
| cascade_fusion_concat | 31.2 | 37.9 | 47.2 | 26.5 | 25.0 | 21.6 | 33.1 |
| cascade_fusion_gated | 31.2 | 37.9 | 47.2 | 26.5 | 25.0 | 21.6 | 33.1 |
| cascade_text_asr_probe | 31.2 | 37.9 | 47.2 | 26.5 | 25.0 | 21.6 | 33.1 |
| combined_asr | 26.6 | 27.7 | 30.4 | 27.8 | 26.7 | 22.5 | 35.1 |
| combined_asr_ctx | 26.6 | 27.7 | 30.4 | 27.8 | 26.7 | 22.5 | 35.1 |
| combined_asr_ctx_e2v | 26.6 | 27.7 | 30.4 | 27.8 | 26.7 | 22.5 | 35.1 |
| combined_asr_e2v | 26.6 | 27.7 | 30.4 | 27.8 | 26.7 | 22.5 | 35.1 |
| combined_gt2asr | 26.6 | 27.7 | 30.4 | 27.8 | 26.7 | 22.5 | 35.1 |
| mtl_ctc_asr | 37.4 | 40.6 | 46.6 | 36.4 | 35.9 | 31.9 | 48.2 |
| mtl_ctc_joint | 39.1 | 42.4 | 47.2 | 37.4 | 36.8 | 33.5 | 46.8 |
| mtl_ctc_ser | 52.0 | 57.5 | 66.2 | 45.6 | 47.0 | 46.2 | 58.1 |
| qwen2_audio_7b_zeroshot | 51.8 | 56.6 | 74.8 | 51.5 | 43.3 | 46.8 | 51.7 |
| qwen2_audio_7b_zeroshot_stripped | 30.1 | 33.0 | 51.1 | 30.6 | 25.5 | 25.0 | 37.7 |
| sensevoice_small_finetune | 26.6 | 27.8 | 30.5 | 27.8 | 26.8 | 22.7 | 34.9 |
| sensevoice_small_zeroshot | 28.3 | 28.9 | 33.7 | 29.3 | 28.7 | 25.7 | 38.6 |
| whisper_whisper-large-v3-turbo | 31.2 | 37.9 | 47.2 | 26.5 | 25.0 | 21.6 | 33.1 |

### Recall theo cảm xúc (%)

| system | neutral | joy | surprise | anger | sadness | disgust | fear |
|---|---|---|---|---|---|---|---|
| cascade_e2v_probe | 41.2 | 39.0 | 41.7 | 31.5 | 33.0 | 19.4 | 10.0 |
| cascade_fusion_concat | 48.5 | 46.9 | 52.2 | 28.5 | 38.8 | 34.3 | 18.0 |
| cascade_fusion_gated | 46.4 | 49.9 | 50.4 | 30.9 | 34.0 | 25.4 | 20.0 |
| cascade_fusion_gated_gt | 61.1 | 57.7 | 58.6 | 37.4 | 32.0 | 37.3 | 32.0 |
| cascade_text_asr_probe | 34.5 | 46.3 | 59.0 | 30.3 | 38.3 | 28.4 | 30.0 |
| cascade_text_gt_probe | 63.4 | 57.7 | 59.0 | 36.2 | 35.4 | 43.3 | 38.0 |
| combined_asr | 56.3 | 50.1 | 55.4 | 36.8 | 41.7 | 20.9 | 26.0 |
| combined_asr_ctx | 51.9 | 54.2 | 55.8 | 37.1 | 50.0 | 17.9 | 24.0 |
| combined_asr_ctx_e2v | 54.1 | 52.6 | 51.4 | 47.5 | 52.4 | 22.4 | 22.0 |
| combined_asr_e2v | 53.2 | 50.6 | 52.2 | 38.9 | 47.6 | 20.9 | 20.0 |
| combined_gt2asr | 57.4 | 48.9 | 50.0 | 34.7 | 44.2 | 23.9 | 30.0 |
| combined_gt_ctx | 67.0 | 67.5 | 74.5 | 51.9 | 43.2 | 44.8 | 42.0 |
| emotion2vec_plus_large_zeroshot | 56.7 | 55.2 | 39.9 | 24.3 | 32.5 | 0.0 | 2.0 |
| mtl_ctc_joint | 32.0 | 30.2 | 42.8 | 39.5 | 67.0 | 3.0 | 12.0 |
| mtl_ctc_ser | 37.4 | 34.5 | 45.0 | 47.2 | 48.5 | 14.9 | 14.0 |
| qwen2_audio_7b_zeroshot | 50.5 | 47.9 | 31.3 | 58.8 | 42.7 | 11.9 | 28.0 |
| qwen2_audio_7b_zeroshot_stripped | 50.5 | 47.9 | 31.3 | 58.8 | 42.7 | 11.9 | 28.0 |
| sensevoice_small_finetune | 77.3 | 37.0 | 37.4 | 44.2 | 20.4 | 3.0 | 0.0 |
| sensevoice_small_zeroshot | 75.6 | 39.5 | 27.7 | 42.4 | 13.6 | 1.5 | 0.0 |

## IEMOCAP test

| system | WER | WA | UA | macro-F1 | wF1 | UA mean±std | seeds |
|---|---|---|---|---|---|---|---|
| cascade_fusion_gated_gt |  | 90.73 | 91.16 | 90.94 | 90.76 | 90.37±1.00 | 3 |
| cascade_fusion_gated | 16.37 | 90.98 | 91.12 | 91.19 | 91.02 | 90.71±0.13 | 3 |
| cascade_fusion_concat | 16.37 | 90.81 | 91.08 | 90.96 | 90.85 | 90.50±1.07 | 3 |
| cascade_e2v_probe |  | 90.41 | 90.80 | 90.46 | 90.42 | 90.99±0.45 | 3 |
| combined_asr_e2v | 9.78 | 89.85 | 90.10 | 89.72 | 89.87 | 89.71±1.09 | 3 |
| combined_asr_ctx_e2v | 9.78 | 90.09 | 89.98 | 90.01 | 90.14 | 89.57±0.65 | 3 |
| emotion2vec_plus_large_zeroshot |  | 87.19 | 86.88 | 87.44 | 87.00 |  |  |
| combined_asr_ctx_pros_whisper | 9.78 | 84.13 | 85.32 | 84.09 | 84.15 | 84.67±1.19 | 3 |
| combined_asr_ctx_whisper_attn | 9.78 | 84.37 | 85.15 | 84.25 | 84.40 | 83.77±0.34 | 3 |
| combined_asr_ctx_whisper | 9.78 | 84.93 | 85.04 | 84.80 | 85.06 | 83.46±1.31 | 3 |
| combined_asr_ctx_whisper_fmean | 9.78 | 84.29 | 84.79 | 84.06 | 84.32 | 84.03±0.86 | 3 |
| combined_asr_ctx_pros_wavlm | 9.78 | 83.16 | 84.03 | 82.92 | 83.21 | 83.33±0.57 | 3 |
| combined_asr_ctx_wavlm | 9.78 | 83.16 | 83.23 | 82.77 | 83.24 | 82.31±0.86 | 3 |
| combined_asr_ctx_pros | 9.78 | 81.95 | 82.96 | 82.09 | 82.05 | 81.66±0.89 | 3 |
| combined_gt_ctx |  | 81.31 | 81.66 | 81.07 | 81.43 | 80.89±0.17 | 3 |
| combined_asr_ctx | 9.78 | 80.82 | 81.19 | 80.04 | 80.84 | 80.20±1.14 | 3 |
| cascade_fusion_concat_wavlm | 16.37 | 78.57 | 79.27 | 78.75 | 78.62 | 77.81±1.54 | 3 |
| cascade_fusion_gated_gt_wavlm |  | 77.03 | 77.46 | 77.26 | 77.15 | 76.85±1.28 | 3 |
| cascade_fusion_gated_wavlm | 16.37 | 76.07 | 76.59 | 76.48 | 76.14 | 75.90±1.65 | 3 |
| combined_asr_wavlm | 9.78 | 75.02 | 74.87 | 74.50 | 74.99 | 73.05±0.87 | 3 |
| sensevoice_small_finetune | 9.77 | 71.88 | 73.69 | 72.59 | 71.70 |  |  |
| combined_gt2asr | 9.78 | 70.19 | 71.96 | 69.96 | 70.33 | 70.32±0.55 | 3 |
| combined_asr | 9.78 | 70.35 | 71.77 | 69.85 | 70.62 | 70.38±1.69 | 3 |
| combined_gt |  | 70.67 | 71.73 | 70.56 | 70.80 | 71.06±0.93 | 3 |
| cascade_wavlm_probe |  | 69.86 | 68.80 | 69.31 | 69.55 | 68.18±1.81 | 3 |
| cascade_text_asr_probe | 16.37 | 67.20 | 68.64 | 66.51 | 67.46 | 66.72±0.63 | 3 |
| cascade_text_gt_probe |  | 64.30 | 66.45 | 64.40 | 64.49 | 66.77±1.11 | 3 |
| sensevoice_small_zeroshot | 11.15 | 67.93 | 66.01 | 67.29 | 68.12 |  |  |
| whisper_whisper-large-v3-turbo | 16.37 |  |  |  |  |  |  |

### WER theo cảm xúc (%)

| system | neutral | joy | anger | sadness |
|---|---|---|---|---|
| cascade_fusion_concat | 15.4 | 22.9 | 11.4 | 9.7 |
| cascade_fusion_concat_wavlm | 15.4 | 22.9 | 11.4 | 9.7 |
| cascade_fusion_gated | 15.4 | 22.9 | 11.4 | 9.7 |
| cascade_fusion_gated_wavlm | 15.4 | 22.9 | 11.4 | 9.7 |
| cascade_text_asr_probe | 15.4 | 22.9 | 11.4 | 9.7 |
| combined_asr | 8.5 | 11.5 | 8.3 | 9.6 |
| combined_asr_ctx | 8.5 | 11.5 | 8.3 | 9.6 |
| combined_asr_ctx_e2v | 8.5 | 11.5 | 8.3 | 9.6 |
| combined_asr_ctx_pros | 8.5 | 11.5 | 8.3 | 9.6 |
| combined_asr_ctx_pros_wavlm | 8.5 | 11.5 | 8.3 | 9.6 |
| combined_asr_ctx_pros_whisper | 8.5 | 11.5 | 8.3 | 9.6 |
| combined_asr_ctx_wavlm | 8.5 | 11.5 | 8.3 | 9.6 |
| combined_asr_ctx_whisper | 8.5 | 11.5 | 8.3 | 9.6 |
| combined_asr_ctx_whisper_attn | 8.5 | 11.5 | 8.3 | 9.6 |
| combined_asr_ctx_whisper_fmean | 8.5 | 11.5 | 8.3 | 9.6 |
| combined_asr_e2v | 8.5 | 11.5 | 8.3 | 9.6 |
| combined_asr_wavlm | 8.5 | 11.5 | 8.3 | 9.6 |
| combined_gt2asr | 8.5 | 11.5 | 8.3 | 9.6 |
| sensevoice_small_finetune | 8.4 | 11.4 | 8.3 | 9.7 |
| sensevoice_small_zeroshot | 9.7 | 13.0 | 9.5 | 11.0 |
| whisper_whisper-large-v3-turbo | 15.4 | 22.9 | 11.4 | 9.7 |

### Recall theo cảm xúc (%)

| system | neutral | joy | anger | sadness |
|---|---|---|---|---|
| cascade_e2v_probe | 89.6 | 89.4 | 91.2 | 93.1 |
| cascade_fusion_concat | 90.9 | 89.8 | 91.8 | 91.8 |
| cascade_fusion_concat_wavlm | 83.1 | 73.8 | 83.5 | 76.7 |
| cascade_fusion_gated | 91.4 | 90.3 | 91.8 | 91.0 |
| cascade_fusion_gated_gt | 89.6 | 90.0 | 92.4 | 92.7 |
| cascade_fusion_gated_gt_wavlm | 81.2 | 73.1 | 80.0 | 75.5 |
| cascade_fusion_gated_wavlm | 82.6 | 71.3 | 82.4 | 70.2 |
| cascade_text_asr_probe | 61.5 | 66.7 | 74.1 | 72.2 |
| cascade_text_gt_probe | 59.6 | 60.6 | 74.1 | 71.4 |
| cascade_wavlm_probe | 82.8 | 67.4 | 73.5 | 51.4 |
| combined_asr | 66.1 | 68.3 | 75.9 | 76.7 |
| combined_asr_ctx | 72.4 | 85.1 | 81.2 | 86.1 |
| combined_asr_ctx_e2v | 90.9 | 90.3 | 90.6 | 88.2 |
| combined_asr_ctx_pros | 78.9 | 81.0 | 87.1 | 84.9 |
| combined_asr_ctx_pros_wavlm | 77.1 | 84.8 | 87.6 | 86.5 |
| combined_asr_ctx_pros_whisper | 77.9 | 86.0 | 92.9 | 84.5 |
| combined_asr_ctx_wavlm | 79.2 | 84.8 | 81.2 | 87.8 |
| combined_asr_ctx_whisper | 85.4 | 84.2 | 85.3 | 85.3 |
| combined_asr_ctx_whisper_attn | 79.9 | 85.3 | 88.8 | 86.5 |
| combined_asr_ctx_whisper_fmean | 79.2 | 86.7 | 87.6 | 85.7 |
| combined_asr_e2v | 90.1 | 88.7 | 90.6 | 91.0 |
| combined_asr_wavlm | 82.8 | 70.1 | 74.7 | 71.8 |
| combined_gt | 67.7 | 69.2 | 75.3 | 74.7 |
| combined_gt2asr | 64.3 | 68.8 | 78.8 | 75.9 |
| combined_gt_ctx | 77.6 | 82.1 | 81.2 | 85.7 |
| emotion2vec_plus_large_zeroshot | 73.7 | 96.6 | 82.9 | 94.3 |
| sensevoice_small_finetune | 80.7 | 59.3 | 78.8 | 75.9 |
| sensevoice_small_zeroshot | 81.5 | 59.0 | 44.7 | 78.8 |

### IEMOCAP leave-one-session-out (trung bình các seed mỗi fold)

Std là độ lệch giữa các fold. Số seed khác nhau giữa các hệ thống, xem cột seeds available.

| system | folds | seeds available | UA per fold (1..5) | UA mean±std | wF1 mean±std | WER mean |
|---|---|---|---|---|---|---|
| combined_asr_ctx_pros_whisper | 5 | 3 3 3 3 3 | 77.4 86.6 78.7 83.6 84.7 | 82.19±3.95 | 80.57±4.54 | 9.68 |
| combined_asr_ctx_whisper | 5 | 3 3 3 3 3 | 78.0 85.7 79.7 81.5 83.5 | 81.67±3.03 | 80.54±3.60 | 9.68 |
| combined_asr_ctx_pros_wavlm | 5 | 1 1 1 1 3 | 75.1 85.0 77.5 80.0 83.3 | 80.20±4.06 | 78.48±4.71 | 9.68 |
| combined_asr_ctx_pros | 5 | 1 1 1 1 3 | 77.0 84.6 75.1 79.0 81.7 | 79.45±3.77 | 77.99±4.07 | 9.68 |
| combined_asr_ctx_wavlm | 5 | 3 3 3 3 3 | 76.5 83.5 75.9 77.5 82.3 | 79.15±3.49 | 78.27±3.96 | 9.68 |
| combined_gt_ctx | 5 | 1 1 1 1 3 | 76.1 85.0 76.7 76.5 80.9 | 79.02±3.88 | 77.41±4.64 |  |
| combined_asr_ctx | 5 | 1 1 1 1 3 | 75.3 82.5 74.9 75.3 80.2 | 77.64±3.47 | 76.50±3.73 | 9.68 |
| cascade_fusion_concat_wavlm | 5 | 1 1 1 1 3 | 76.0 80.2 76.5 72.6 77.8 | 76.63±2.77 | 75.59±2.57 | 13.19 |
| cascade_fusion_gated_wavlm | 5 | 1 1 1 1 3 | 77.0 80.1 74.6 74.0 75.9 | 76.31±2.42 | 75.21±1.91 | 13.19 |
| cascade_fusion_gated_gt_wavlm | 5 | 1 1 1 1 3 | 77.2 75.0 73.8 76.0 76.8 | 75.77±1.37 | 74.57±2.43 |  |
| combined_asr_wavlm | 5 | 1 1 1 1 3 | 72.0 78.8 74.5 73.6 73.0 | 74.41±2.61 | 73.32±2.82 | 9.68 |
| sensevoice_small_finetune | 5 | 1 1 1 1 1 | 73.3 73.3 71.2 71.8 73.7 | 72.65±1.11 | 70.96±0.94 | 9.67 |
| combined_gt | 5 | 1 1 1 1 3 | 70.5 76.1 69.5 71.0 71.1 | 71.62±2.56 | 70.12±2.15 |  |
| combined_asr | 5 | 1 1 1 1 3 | 68.8 75.5 69.0 69.4 70.4 | 70.63±2.80 | 69.42±2.46 | 9.68 |
| cascade_wavlm_probe | 5 | 1 1 1 1 3 | 70.4 71.0 69.2 65.9 68.2 | 68.93±2.01 | 67.73±1.52 |  |
| cascade_text_gt_probe | 5 | 1 1 1 1 3 | 65.4 71.5 66.1 66.7 66.8 | 67.28±2.42 | 65.64±2.48 |  |
| sensevoice_small_zeroshot | 5 | 1 1 1 1 1 | 66.2 67.9 64.6 69.1 66.0 | 66.77±1.76 | 67.18±1.88 | 11.20 |
| cascade_text_asr_probe | 5 | 1 1 1 1 3 | 63.7 70.9 64.9 64.6 66.7 | 66.15±2.87 | 64.17±3.80 | 13.19 |

### IEMOCAP leave-one-session-out (chỉ seed 0, mọi hệ thống cùng số lần chạy)

| system | folds | seeds available | UA per fold (1..5) | UA mean±std | wF1 mean±std | WER mean |
|---|---|---|---|---|---|---|
| combined_asr_ctx_pros_whisper | 5 | 3 3 3 3 3 | 76.4 87.3 77.6 83.6 85.5 | 82.08±4.83 | 80.45±5.30 | 9.68 |
| combined_asr_ctx_whisper | 5 | 3 3 3 3 3 | 77.0 85.4 78.7 81.1 83.1 | 81.06±3.37 | 79.77±3.75 | 9.68 |
| combined_asr_ctx_pros_wavlm | 5 | 1 1 1 1 3 | 75.1 85.0 77.5 80.0 84.0 | 80.33±4.20 | 78.49±4.72 | 9.68 |
| combined_asr_ctx_pros | 5 | 1 1 1 1 3 | 77.0 84.6 75.1 79.0 80.6 | 79.25±3.65 | 77.67±3.86 | 9.68 |
| combined_gt_ctx | 5 | 1 1 1 1 3 | 76.1 85.0 76.7 76.5 80.9 | 79.02±3.87 | 77.46±4.68 |  |
| combined_asr_ctx_wavlm | 5 | 3 3 3 3 3 | 76.4 81.8 78.3 77.0 81.5 | 79.01±2.51 | 78.35±3.37 | 9.68 |
| combined_asr_ctx | 5 | 1 1 1 1 3 | 75.3 82.5 74.9 75.3 79.2 | 77.44±3.31 | 76.32±3.56 | 9.68 |
| cascade_fusion_concat_wavlm | 5 | 1 1 1 1 3 | 76.0 80.2 76.5 72.6 79.6 | 76.98±3.06 | 75.96±2.96 | 13.19 |
| cascade_fusion_gated_wavlm | 5 | 1 1 1 1 3 | 77.0 80.1 74.6 74.0 77.7 | 76.67±2.48 | 75.36±1.96 | 13.19 |
| cascade_fusion_gated_gt_wavlm | 5 | 1 1 1 1 3 | 77.2 75.0 73.8 76.0 78.1 | 76.02±1.68 | 74.75±2.65 |  |
| combined_asr_wavlm | 5 | 1 1 1 1 3 | 72.0 78.8 74.5 73.6 74.0 | 74.61±2.52 | 73.55±2.82 | 9.68 |
| sensevoice_small_finetune | 5 | 1 1 1 1 1 | 73.3 73.3 71.2 71.8 73.7 | 72.65±1.11 | 70.96±0.94 | 9.67 |
| combined_gt | 5 | 1 1 1 1 3 | 70.5 76.1 69.5 71.0 70.4 | 71.48±2.62 | 70.07±2.16 |  |
| combined_asr | 5 | 1 1 1 1 3 | 68.8 75.5 69.0 69.4 71.8 | 70.91±2.84 | 69.82±2.53 | 9.68 |
| cascade_wavlm_probe | 5 | 1 1 1 1 3 | 70.4 71.0 69.2 65.9 67.1 | 68.71±2.18 | 67.58±1.36 |  |
| cascade_text_gt_probe | 5 | 1 1 1 1 3 | 65.4 71.5 66.1 66.7 66.0 | 67.13±2.48 | 65.56±2.51 |  |
| sensevoice_small_zeroshot | 5 | 1 1 1 1 1 | 66.2 67.9 64.6 69.1 66.0 | 66.77±1.76 | 67.18±1.88 | 11.20 |
| cascade_text_asr_probe | 5 | 1 1 1 1 3 | 63.7 70.9 64.9 64.6 66.1 | 66.02±2.85 | 63.93±3.73 | 13.19 |
