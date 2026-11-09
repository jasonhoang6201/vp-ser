# gpu: L4
# %% [markdown]
# # 20. Transcript SenseVoice fine-tune cho mọi split (chuẩn bị cho model kết hợp)
#
# Model kết hợp dùng transcript của ASR tốt nhất (SenseVoice-Small fine-tune, WER test 27.4) làm đầu vào cho nhánh text.
# Để nhánh text học được cách chịu lỗi ASR, cần transcript ASR cho cả **train** và **dev**, không chỉ test.
#
# Lưu ý: SenseVoice đã fine-tune trên chính tập train, nên transcript train sẽ sạch hơn dev/test (WER thấp hơn).
# Notebook in WER từng split để thấy độ chênh này.
#
# Đầu ra: `cache/<dataset>/asr_sensevoice_ft.csv` (id, split, hyp_text, sv_emotion). SenseVoice chỉ sinh token
# cảm xúc, không có xác suất, nên chỉ lưu token.

# %%
DATASET = "meld"
BATCH = 64
TEST_SESSION = 5      # IEMOCAP: phải khớp với checkpoint của notebook 10 cùng fold
FOLD = "" if DATASET == "meld" or TEST_SESSION == 5 else f"_f{TEST_SESSION}"

# %%
import os, sys
NB_ROOT = os.environ.get("EAASR_NB", "/content/drive/MyDrive/vp-ck/notebooks")
if NB_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
sys.path.insert(0, NB_ROOT)

# %%
!pip -q install -U funasr whisper-normalizer soundfile
!test -d /content/SenseVoice || git clone -q https://github.com/FunAudioLLM/SenseVoice /content/SenseVoice

# %%
import re, shutil
from pathlib import Path

import pandas as pd
from funasr import AutoModel
from huggingface_hub import snapshot_download

from eaasr.common import CACHE_DIR, CKPT_DIR, asr_metrics, canon_emotion, get_splits, stage_dataset

SV_DIR = "/content/SenseVoice"
MODEL_DIR = snapshot_download("FunAudioLLM/SenseVoiceSmall", local_dir="/content/sv_model")
if not os.path.exists(f"{MODEL_DIR}/model.py"):
    shutil.copy(f"{SV_DIR}/model.py", MODEL_DIR)
ckpt = CKPT_DIR / f"sensevoice_{DATASET}{FOLD}" / "model.pt"
print(ckpt, ckpt.exists())
man = stage_dataset(DATASET)
splits = get_splits(man, DATASET, test_session=TEST_SESSION)
model = AutoModel(model=MODEL_DIR, init_param=str(ckpt), trust_remote_code=True,
                  remote_code=f"{SV_DIR}/model.py", device="cuda:0", disable_update=True)

# %%
TAG = re.compile(r"<\|([^|]*)\|>")
rows = []
for split, df in splits.items():
    res = model.generate(input=df["path"].tolist(), cache={}, language="en", use_itn=True,
                         batch_size=BATCH, ban_emo_unk=True)
    assert len(res) == len(df)
    for (_, r), out in zip(df.iterrows(), res):
        tags = TAG.findall(out["text"])
        rows.append({"id": r.id, "split": split, "hyp_text": TAG.sub("", out["text"]).strip(),
                     "sv_emotion": next((canon_emotion(t) for t in tags if canon_emotion(t)), None)})
    hyp = pd.DataFrame(rows[-len(df):])
    print(split, len(df), "WER", round(100 * asr_metrics(df.text.tolist(), hyp.hyp_text.tolist())["wer"], 2))

# IEMOCAP labels only ~55% of utterances with the 4 classes; the rest are still turns of the dialogue, so they
# get transcripts too and can serve as context. No metrics on them: the ASR model never saw their labels anyway.
seen = set(r["id"] for r in rows)
rest = man[~man.id.isin(seen) & (man.duration <= 60)]
if len(rest):
    res = model.generate(input=rest["path"].tolist(), cache={}, language="en", use_itn=True,
                         batch_size=BATCH, ban_emo_unk=True)
    for (_, r), o in zip(rest.iterrows(), res):
        rows.append({"id": r.id, "split": "unlabeled", "hyp_text": TAG.sub("", o["text"]).strip(), "sv_emotion": None})
    print("unlabeled", len(rest))

out = pd.DataFrame(rows)
dst = CACHE_DIR / DATASET / f"asr_sensevoice_ft{FOLD}.csv"
dst.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(dst, index=False)
print("saved", dst, len(out))
