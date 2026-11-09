# gpu:
# %% [markdown]
# # 00. Chuẩn bị dữ liệu MELD
#
# MELD gồm các câu thoại trích từ phim *Friends*, mỗi câu có transcript (phụ đề) và nhãn cảm xúc (7 lớp).
# Bản gốc là video `.mp4`. Notebook này:
#
# 1. tải `MELD.Raw.tar.gz` (khoảng 10 GB),
# 2. tách audio thành wav **16 kHz mono** (dùng `ffmpeg`),
# 3. sửa lỗi ký tự trong phụ đề, tạo **manifest** (bảng: id, đường dẫn, transcript, cảm xúc, split),
# 4. đóng gói wav thành `meld_16k.tar` rồi lưu lên Drive. Các notebook sau chỉ cần giải nén file này.
#
# Notebook này **không cần GPU**: chọn Runtime > Change runtime type > CPU (High-RAM nếu có).
# Chỉ cần chạy **một lần**.

# %%
import os, sys
NB_ROOT = os.environ.get("EAASR_NB", "/content/drive/MyDrive/vp-ck/notebooks")
if NB_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
sys.path.insert(0, NB_ROOT)

# %%
!pip -q install whisper-normalizer soundfile
!apt-get -qq install -y ffmpeg > /dev/null

# %%
import os, re, subprocess, tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from eaasr.common import DATA_DIR, LOCAL_ROOT, canon_emotion, clean_text

RAW = Path("/content/meld_raw")
OUT = LOCAL_ROOT / "meld" / "wav"
RAW.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Tải và giải nén
# Dùng bản mirror trên HuggingFace của nhóm tác giả (`declare-lab/MELD`), thường nhanh hơn server gốc.

# %%
tgz = RAW / "MELD.Raw.tar.gz"
if not tgz.exists():
    !wget -c -q --show-progress -O {tgz} https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz
!tar -xzf {tgz} -C {RAW}

# Bên trong còn các archive con (train/dev/test), giải nén tiếp.
for inner in RAW.rglob("*.tar.gz"):
    if inner == tgz:
        continue
    print("extract", inner)
    with tarfile.open(inner) as t:
        t.extractall(inner.parent)

# %% [markdown]
# ## 2. Đọc file CSV (transcript + nhãn)

# %%
def find_csv(name):
    hits = [p for p in RAW.rglob(name) if not p.name.startswith("._")]
    if hits:
        return hits[0]
    url = f"https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/{name}"
    dst = RAW / name
    subprocess.run(["wget", "-q", "-O", str(dst), url], check=True)
    return dst

csvs = {s: pd.read_csv(find_csv(f"{s}_sent_emo.csv")) for s in ["train", "dev", "test"]}
for s, d in csvs.items():
    print(s, d.shape, list(d.columns))
csvs["train"].head()

# %% [markdown]
# ## 3. Tìm file mp4 cho từng câu
# Tên file là `dia{Dialogue_ID}_utt{Utterance_ID}.mp4`. Archive có lẫn file rác của macOS bắt đầu bằng `._`, bỏ qua.

# %%
mp4 = {"train": {}, "dev": {}, "test": {}}
for p in RAW.rglob("*.mp4"):
    if p.name.startswith("._"):
        continue
    parent = str(p.parent).lower()
    split = "train" if "train" in parent else "dev" if "dev" in parent else "test" if "test" in parent else None
    if split:
        mp4[split][p.name] = p
print({s: len(v) for s, v in mp4.items()})

# %%
jobs, missing = [], []
for split, d in csvs.items():
    for _, r in d.iterrows():
        name = f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}"
        src = mp4[split].get(name + ".mp4")
        if src is None:
            missing.append((split, name))
            continue
        jobs.append((split, name, src, OUT / split / f"{name}.wav"))
print("to convert:", len(jobs), "| missing mp4:", len(missing), missing[:5])

# %%
def convert(job):
    split, name, src, dst = job
    if dst.exists():
        return job, True
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
                        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
                       capture_output=True)
    return job, r.returncode == 0 and dst.exists()

failed = []
with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as ex:
    for job, ok in tqdm(ex.map(convert, jobs), total=len(jobs)):
        if not ok:
            failed.append(job[:2])
print("ffmpeg failed:", len(failed), failed[:10])

# %% [markdown]
# ## 4. Tạo manifest
# - `text`: phụ đề đã sửa lỗi ký tự (MELD có lỗi kiểu `â€™` thay cho dấu `'`).
# - `emotion`: tên nhãn chuẩn hóa (neutral, joy, surprise, anger, sadness, disgust, fear).
# - Câu có audio hỏng hoặc dài dưới 0.2 giây bị loại.

# %%
rows, bad = [], []
for split, d in csvs.items():
    for _, r in d.iterrows():
        name = f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}"
        wav = OUT / split / f"{name}.wav"
        if not wav.exists():
            continue
        try:
            dur = sf.info(str(wav)).duration
        except RuntimeError:
            bad.append(name)
            continue
        if dur < 0.2:
            bad.append(name)
            continue
        rows.append({
            "id": f"{split}_{name}",
            "relpath": str(wav.relative_to(LOCAL_ROOT)),
            "split": split,
            "text": clean_text(r.Utterance),
            "emotion": canon_emotion(r.Emotion),
            "emotion_raw": r.Emotion,
            "sentiment": r.Sentiment,
            "speaker": r.Speaker,
            "dialogue_id": r.Dialogue_ID,
            "duration": round(dur, 3),
        })
man = pd.DataFrame(rows)
print("dropped (unreadable/too short):", len(bad))
print(pd.crosstab(man.emotion, man.split, margins=True))
print(man.groupby("split").duration.describe())

# %%
weird = man[man.text.str.contains(r"[^\x00-\x7f]", regex=True)]
print("transcripts still containing non-ASCII:", len(weird))
weird[["id", "text"]].head(10)

# %% [markdown]
# ## 5. Nghe thử vài câu ngẫu nhiên
# Kiểm tra nhanh bằng tai: audio có đúng câu thoại không, nhãn cảm xúc có hợp lý không.

# %%
from IPython.display import Audio, display
for _, r in man.sample(5, random_state=0).iterrows():
    print(f"[{r.emotion}] {r.text}  ({r.duration}s)")
    display(Audio(str(LOCAL_ROOT / r.relpath)))

# %% [markdown]
# ## 6. Đóng gói và lưu lên Drive

# %%
man.to_csv(DATA_DIR / "meld_manifest.csv", index=False)
!tar -cf {DATA_DIR}/meld_16k.tar -C {LOCAL_ROOT} meld/wav
!ls -lh {DATA_DIR}
