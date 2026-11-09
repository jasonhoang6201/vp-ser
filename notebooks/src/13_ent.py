# gpu: A100
# %% [markdown]
# # 13. Baseline ENT: Emotion Neural Transducer (shen-2024-ent)
#
# Chạy code chính thức `ECNU-Cross-Innovation-Lab/ENT`. Model: wav2vec2-base **đóng băng** + neural transducer
# (RNN-T) cho ASR, thêm emotion joint network cho SER.
#
# Code gốc chỉ đọc một file `pkl` chứa các key `Session1`…`Session5`, chạy fold `i` thì test trên `Session{i}`.
# Notebook này tạo pkl cùng cấu trúc từ manifest của mình:
# - **MELD**: `Session1` = test, `Session2..5` = train chia làm 4 phần, `NUM_FOLD = 1`.
# - **IEMOCAP**: mỗi session thật là một key. Nếu `FULL_5FOLD = False` thì đưa Ses05 lên làm `Session1`
#   để chạy 1 fold test trên Ses05, giống các notebook khác.
#
# **Lưu ý về độ tin cậy (đọc từ code):** repo không có tập dev. Mỗi epoch nó đánh giá trên test và báo
# **giá trị tốt nhất qua mọi epoch** (`MAX_WA`, `MIN_WER`). Như vậy là chọn model trên chính tập test, số sẽ đẹp
# hơn thực tế. Notebook báo cả hai:
# - `*_max`: theo cách của paper (lạc quan),
# - `*_last`: epoch cuối cùng, không chọn trên test (công bằng, dùng con số này để so với các model khác).

# %%
DATASET = "meld"
EPOCHS = 20            # paper: 50. MELD có ~10k câu train, 20 epoch trên A100 mất vài giờ
BATCH_SIZE = 2         # giữ như config gốc
FULL_5FOLD = False     # chỉ áp dụng cho IEMOCAP
SMOKE_TEST = False     # True: chỉ 200 câu train / 50 câu test, 1 epoch, để kiểm tra pipeline trước

# %%
import os, sys
NB_ROOT = os.environ.get("EAASR_NB", "/content/drive/MyDrive/vp-ck/notebooks")
if NB_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
sys.path.insert(0, NB_ROOT)

# %%
!test -d /content/ENT || git clone -q https://github.com/ECNU-Cross-Innovation-Lab/ENT /content/ENT
!pip -q install yacs timm jiwer sentencepiece whisper-normalizer soundfile

# %%
import os, pickle, re, subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from eaasr.common import CKPT_DIR, LABELS, get_splits, gpu_info, load_wav, save_metrics, stage_dataset

gpu_info()
ENT = Path("/content/ENT")
labels = LABELS[DATASET]
lab2id = {l: i for i, l in enumerate(labels)}
man = stage_dataset(DATASET)

# %% [markdown]
# ## 1. Chuẩn hóa text và tạo pkl giống `preprocess.py` + `toke.py` của ENT
# ENT: viết hoa, thay dấu câu bằng khoảng trắng, giữ dấu `'`. Mỗi ký tự tra trong `iemocap_char_units.txt`
# (A–Z, `'`, khoảng trắng là `▁`).

# %%
units = {}
for line in open(ENT / "dataset/IEMOCAP/iemocap_char_units.txt", encoding="utf8"):
    sym, i = line.split()
    units[" " if sym == "▁" else sym] = int(i)

def ent_text(s):
    s = s.upper()
    s = re.sub(r"[^A-Z' ]", " ", s)
    return " ".join(s.split())

def to_item(r):
    wav = torch.from_numpy(load_wav(r.path))
    text = ent_text(r.text)
    return {"audio": wav, "audio_length": int(wav.shape[0]), "label": lab2id[r.emotion], "text": text,
            "tlabel": [units.get(ch, units["<unk>"]) for ch in text]}

if DATASET == "meld":
    sp = get_splits(man, "meld")
    train = sp["train"].sample(frac=1, random_state=0).reset_index(drop=True)
    test = sp["test"]
    if SMOKE_TEST:
        train, test = train.head(200), test.head(50)
    train = train[train.text.map(ent_text).str.len() > 0]
    parts = np.array_split(np.arange(len(train)), 4)
    data_map = {"Session1": [to_item(r) for r in tqdm(test.itertuples(), total=len(test), desc="test")]}
    for k, p in enumerate(parts, 2):
        data_map[f"Session{k}"] = [to_item(r) for r in tqdm(train.iloc[p].itertuples(), total=len(p), desc=f"train{k}")]
    n_fold = 1
else:
    d = man[man.emotion.isin(labels)]
    if SMOKE_TEST:
        d = d.groupby("session").head(60)
    order = [1, 2, 3, 4, 5] if FULL_5FOLD else [5, 1, 2, 3, 4]
    data_map = {f"Session{k}": [to_item(r) for r in d[d.session == s].itertuples()] for k, s in enumerate(order, 1)}
    n_fold = 5 if FULL_5FOLD else 1

print({k: len(v) for k, v in data_map.items()})
print(data_map["Session1"][0]["text"], data_map["Session1"][0]["tlabel"][:20])
PKL = ENT / f"dataset/IEMOCAP/{DATASET}4char.pkl"
with open(PKL, "wb") as f:
    pickle.dump(data_map, f)
del data_map

# %% [markdown]
# ## 2. Sửa config cho dataset mới
# Tên file config do code tự ghép từ `DATASET` (giữ là `IEMOCAP` để không phải sửa code), model và mode.
# Chỉ đổi đường dẫn pkl, số lớp, số fold, số epoch và thư mục log.

# %%
cfg_path = ENT / "configs/char/IEMOCAP_ent_asrser.yaml"
orig = cfg_path.with_suffix(".yaml.orig")
if not orig.exists():
    orig.write_text(cfg_path.read_text())
cfg = yaml.safe_load(orig.read_text())
cfg["NUM_FOLD"] = n_fold
cfg["LOGPATH"] = f"./log/{DATASET}"
cfg["DATA"]["DATA_PATH"] = f"./dataset/IEMOCAP/{DATASET}4char.pkl"
cfg["DATA"]["BATCH_SIZE"] = BATCH_SIZE
cfg["MODEL"]["NUM_CLASSES"] = len(labels)
cfg["TRAIN"]["EPOCHS"] = 1 if SMOKE_TEST else EPOCHS
cfg["TRAIN"]["WARMUP_EPOCHS"] = min(cfg["TRAIN"]["WARMUP_EPOCHS"], max(cfg["TRAIN"]["EPOCHS"] // 10, 0))
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(cfg_path.read_text())

# %% [markdown]
# ## 3. Train
# Lệnh lấy từ README của ENT. Nếu báo lỗi liên quan tới `timm` (scheduler), xem cell sửa lỗi ngay bên dưới.

# %%
os.chdir(ENT)
!python main.py --model ent --headw 0.5 --rnntw 0.5 --jointw 0.5 --bpe_mode char 2>&1 | tee /content/ent_train.log | grep -vE "^\s*$" | tail -n 300

# %% [markdown]
# **Nếu lỗi kiểu `unexpected keyword argument 't_mul'`**: code viết cho `timm` bản cũ.
# Chạy cell dưới để đổi tên tham số theo `timm` mới, rồi chạy lại cell train.

# %%
lr = ENT / "lr_scheduler.py"
src = lr.read_text()
patched = src.replace("t_mul=", "cycle_mul=")
if patched != src:
    lr.write_text(patched)
    print("patched lr_scheduler.py")
else:
    print("không cần sửa")

# %% [markdown]
# ## 4. Đọc log, lấy số theo hai cách

# %%
log = Path("/content/ent_train.log").read_text()
pat = re.compile(r"WER: ([\d.]+), CER: ([\d.]+), WA: ([\d.]+), UA: ([\d.]+)")
ep = pd.DataFrame([tuple(map(float, m)) for m in pat.findall(log)], columns=["WER", "CER", "WA", "UA"])
ep.index = ep.index + 1
display(ep)
assert len(ep), "không tìm thấy dòng metric trong log, xem /content/ent_train.log"
last, best = ep.iloc[-1], {"WER": ep.WER.min(), "CER": ep.CER.min(), "WA": ep.WA.max(), "UA": ep.UA.max()}
print("last epoch:", last.to_dict())
print("paper protocol (best over epochs on test):", best)

# %%
name = "ent" + ("_smoke" if SMOKE_TEST else "")
save_metrics(name, DATASET, {
    "notes": f"official ENT code, frozen wav2vec2-base, {cfg['TRAIN']['EPOCHS']} ep; "
             "asr/ser = LAST epoch (fair); *_max = best epoch on test (paper protocol)",
    "asr": {"wer": last.WER / 100, "cer": last.CER / 100, "n_utts": None},
    "ser": {"wa": last.WA / 100, "ua": last.UA / 100, "macro_f1": float("nan"), "weighted_f1": float("nan"),
            "n_invalid_pred": 0},
    "wer_max": best["WER"], "wa_max": best["WA"], "ua_max": best["UA"],
    "epochs": ep.to_dict("records"),
    "wer_note": "ENT normalization (uppercase, punctuation stripped), not the Whisper normalizer",
})
dst = CKPT_DIR / f"ent_{DATASET}"
dst.mkdir(parents=True, exist_ok=True)
!cp /content/ent_train.log {dst}/
