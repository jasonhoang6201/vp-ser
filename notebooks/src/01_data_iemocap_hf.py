# gpu:
# %% [markdown]
# # 01. IEMOCAP từ HuggingFace (bản không chính thức) và kiểm tra ngẫu nhiên
#
# Dùng `AbstractTTS/IEMOCAP`, một bản IEMOCAP do người khác đăng lên HuggingFace, **không rõ license**.
# Trước khi dùng, notebook kiểm tra xem bản này có khớp với IEMOCAP gốc không:
#
# 1. số câu (IEMOCAP gốc có **10039** câu),
# 2. số câu của 4 lớp chuẩn (neutral, happy+excited, angry, sad): paper hay báo **5531** câu,
#    gồm neutral 1708, happy+excited 1636, angry 1103, sad 1084,
# 3. tên file đúng mẫu `Ses0{1-5}{F,M}_{impro,script}..._{F,M}xxx`, để chia **leave-one-session-out**
#    (người nói ở test không xuất hiện ở train),
# 4. sample rate, độ dài, câu trùng lặp,
# 5. **nghe ngẫu nhiên** vài câu và đối chiếu với transcript và nhãn.
#
# Nếu số liệu lệch nhiều thì dừng lại, chờ bản chính thức (form USC).
# Không cần GPU.

# %%
import os, sys
NB_ROOT = os.environ.get("EAASR_NB", "/content/drive/MyDrive/vp-ck/notebooks")
if NB_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
sys.path.insert(0, NB_ROOT)

# %%
!pip -q install -U datasets soundfile librosa whisper-normalizer

# %%
import io, re
from collections import Counter

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from IPython.display import Audio as Play, display
from tqdm.auto import tqdm

from eaasr.common import DATA_DIR, LOCAL_ROOT, SR, canon_emotion, clean_text

# decode=False: lấy bytes thô, tự đọc bằng soundfile, không phụ thuộc backend giải mã audio của `datasets`.
ds = load_dataset("AbstractTTS/IEMOCAP", split="train")
ds = ds.cast_column("audio", Audio(decode=False))
print(ds)
print(ds.features)

# %% [markdown]
# ## 1. Thống kê nhãn

# %%
meta = ds.remove_columns(["audio"]).to_pandas()
print("số câu:", len(meta), "(gốc: 10039)")
print(meta["major_emotion"].value_counts())

# %%
four = meta["major_emotion"].map(canon_emotion)
four = four.where(four.isin(["neutral", "joy", "anger", "sadness"]))
print("4 lớp, theo major_emotion:", four.notna().sum(), "(paper thường báo 5531)")
# major_emotion is the argmax of soft votes for every utterance, so it also labels the ones the
# original release marks "xxx" (no majority). A tie between the top two votes is that case;
# dropping ties brings the 4-class counts within 1% of the original (5498 vs 5531).
votes = meta[["frustrated", "angry", "sad", "disgust", "excited", "fear", "neutral", "surprise", "happy"]]
top2 = np.sort(votes.values, axis=1)[:, -2:]
four = four.where(top2[:, 1] - top2[:, 0] > 0.05)
print("4 lớp, bỏ câu hòa phiếu:", four.notna().sum())
print(four.value_counts(), "\n(gốc: neutral 1708, joy=hap+exc 1636, anger 1103, sad 1084)")

# %% [markdown]
# Kết quả kiểm tra (2026-11-02): cột `major_emotion` là argmax của điểm mềm (soft vote) cho **mọi** câu, kể cả câu
# bản gốc gán "xxx" (không đạt đa số), nên 4 lớp có 6877 câu thay vì 5531. Bỏ các câu mà hai điểm cao nhất bằng nhau
# (hòa phiếu) thì còn 5498 câu: neutral 1654, joy 1663, anger 1110, sadness 1071, lệch dưới 1% so với bản gốc.

# %% [markdown]
# ## 2. Tên file, session, người nói, trùng lặp

# %%
pat = re.compile(r"^Ses0([1-5])([FM])_(impro|script)\w*_([FM])\d+")
parsed = meta["file"].str.replace(".wav", "", regex=False).str.extract(pat)
print("tên file không đúng mẫu:", parsed[0].isna().sum())
meta["session"] = parsed[0].astype("Int64")
meta["speaker"] = "Ses0" + parsed[0] + parsed[3]
meta["type"] = parsed[2]
print("file trùng tên:", meta["file"].duplicated().sum())
print("transcript trùng (cùng nhãn):", meta.duplicated(["transcription", "major_emotion"]).sum())
# Transcript gốc có tag kiểu [LAUGHTER], [BREATHING]; transcript do ASR sinh ra thì không có.
tags = meta["transcription"].str.findall(r"\[[A-Za-z]+\]").explode().dropna()
print("câu có tag [..]:", meta["transcription"].str.contains(r"\[[A-Za-z]+\]").sum(), tags.value_counts().head(8).to_dict())
print(pd.crosstab(meta["session"], four, margins=True))

# %% [markdown]
# ## 3. Sample rate, độ dài (kiểm 300 câu ngẫu nhiên)

# %%
rng = np.random.RandomState(0)
idx = rng.choice(len(ds), 300, replace=False)
srs, durs = Counter(), []
for i in idx:
    a = ds[int(i)]["audio"]
    info = sf.info(io.BytesIO(a["bytes"]))
    srs[info.samplerate] += 1
    durs.append(info.duration)
print("sample rate:", srs)
print(pd.Series(durs).describe())

# %% [markdown]
# ## 4. Nghe ngẫu nhiên 8 câu
# Nghe và tự đánh giá: lời nói có khớp transcript không, cảm xúc nghe có hợp nhãn không.
# Một vài câu nghe "không giống nhãn" là bình thường (IEMOCAP do nhiều người gán, có bất đồng),
# nhưng **transcript sai lời** hoặc **audio lệch câu** thì bản này có vấn đề.

# %%
for i in rng.choice(len(ds), 8, replace=False):
    r = ds[int(i)]
    a = r["audio"]
    wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
    scores = {k: round(float(r[k]), 2) for k in ["neutral", "happy", "excited", "angry", "sad", "frustrated"] if k in r}
    print(f"{r['file']} | major={r['major_emotion']} | {scores}")
    print("   ", r["transcription"])
    display(Play(wav, rate=sr))

# %% [markdown]
# ## 5. Ghi wav 16 kHz + manifest (toàn bộ câu, cột `emotion` chỉ có giá trị cho 4 lớp)
# Chạy phần này sau khi bạn thấy kết quả kiểm tra ổn.

# %%
import librosa

OUT = LOCAL_ROOT / "iemocap" / "wav"
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for i in tqdm(range(len(ds))):
    if pd.isna(meta["session"].iloc[i]):
        continue
    r = ds[i]
    wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    stem = r["file"].replace(".wav", "")
    dst = OUT / f"{stem}.wav"
    sf.write(dst, wav, SR, subtype="PCM_16")
    m = meta.iloc[i]
    rows.append({
        "id": stem,
        "relpath": str(dst.relative_to(LOCAL_ROOT)),
        "session": int(m["session"]),
        "speaker": m["speaker"],
        "type": m["type"],
        "text": clean_text(r["transcription"]),
        "emotion": four.iloc[i] if isinstance(four.iloc[i], str) else "",
        "emotion_raw": r["major_emotion"],
        "duration": round(len(wav) / SR, 3),
    })
man = pd.DataFrame(rows)
DATA_DIR.mkdir(parents=True, exist_ok=True)
man.to_csv(DATA_DIR / "iemocap_manifest.csv", index=False)
print(man.emotion.replace("", "(khác)").value_counts())

# %%
!tar -cf {DATA_DIR}/iemocap_16k.tar -C {LOCAL_ROOT} iemocap/wav
!ls -lh {DATA_DIR}
