# gpu:
# %% [markdown]
# # 90. So sánh kết quả các baseline
#
# Đọc mọi `results/<dataset>/<system>/metrics_test.json` trên Drive, rồi lập bảng:
# - ASR: **WER** (thấp hơn là tốt hơn). Ở mọi hệ thống, ref và hyp đều qua cùng một bộ chuẩn hóa (Whisper normalizer),
#   trừ ENT (dùng chuẩn hóa riêng của nó, xem cột notes).
# - SER: **WA** (accuracy), **UA** (recall trung bình các lớp, không bị lớp đông như neutral lấn át),
#   **macro-F1**, **weighted-F1** (metric hay dùng nhất trên MELD).
# - WER theo từng cảm xúc: kiểm tra giả thuyết "giọng nhiều cảm xúc làm ASR sai nhiều hơn".
#
# Không cần GPU.

# %%
DATASET = "meld"

# %%
import os, sys
NB_ROOT = os.environ.get("EAASR_NB", "/content/drive/MyDrive/vp-ck/notebooks")
if NB_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
sys.path.insert(0, NB_ROOT)

# %%
import json
import pandas as pd
from eaasr.common import LABELS, RESULTS_DIR, load_results

pd.set_option("display.max_colwidth", 120)
res = load_results(DATASET)
cols = [c for c in ["system", "WER", "WA", "UA", "macro_f1", "weighted_f1", "notes"] if c in res]
res[cols].sort_values("UA", ascending=False, na_position="last").round(2)

# %% [markdown]
# ## WER theo cảm xúc (%)

# %%
rows = {}
for p in sorted((RESULTS_DIR / DATASET).glob("*/metrics_test.json")):
    m = json.load(open(p))
    if "asr" in m and "wer_by_emotion" in m["asr"]:
        rows[m["name"]] = {k: v * 100 for k, v in m["asr"]["wer_by_emotion"].items()}
wer_emo = pd.DataFrame(rows).T[LABELS[DATASET]] if rows else pd.DataFrame()
wer_emo.round(1).style.background_gradient(axis=1, cmap="Reds") if len(wer_emo) else "chưa có hệ thống ASR"

# %% [markdown]
# ## Recall theo cảm xúc (%)
# Cho thấy mỗi hệ thống nhận đúng lớp nào và bỏ sót lớp nào (trên MELD thường là fear, disgust).

# %%
rows = {}
for p in sorted((RESULTS_DIR / DATASET).glob("*/metrics_test.json")):
    m = json.load(open(p))
    if "ser" in m and "recall_by_emotion" in m["ser"]:
        rows[m["name"]] = {k: v * 100 for k, v in m["ser"]["recall_by_emotion"].items()}
rec = pd.DataFrame(rows).T[LABELS[DATASET]] if rows else pd.DataFrame()
rec.round(1).style.background_gradient(axis=None, cmap="Greens") if len(rec) else "chưa có hệ thống SER"

# %% [markdown]
# ## Dao động giữa các seed (chỉ các hệ thống cascade có chạy nhiều seed)

# %%
rows = []
for p in sorted((RESULTS_DIR / DATASET).glob("*/metrics_test.json")):
    m = json.load(open(p))
    if "ua_std" in m:
        rows.append({"system": m["name"], "UA": f"{m['ua_mean'] * 100:.2f} ± {m['ua_std'] * 100:.2f}",
                     "wF1": f"{m['wf1_mean'] * 100:.2f} ± {m['wf1_std'] * 100:.2f}"})
pd.DataFrame(rows)
