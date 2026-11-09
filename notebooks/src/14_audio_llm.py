# gpu: A100
# %% [markdown]
# # 14. Audio-LLM zero-shot: Qwen2-Audio-7B-Instruct
#
# Đại diện cho hướng "speech LLM" (C-Gate, OSUM, chen-2025). Không train gì cả: đưa audio kèm câu lệnh (prompt),
# hỏi hai lần cho mỗi câu:
# 1. "Chép lại lời nói" (ASR)
# 2. "Người nói đang có cảm xúc gì, chọn một trong các nhãn" (SER)
#
# Vì sao không chạy đúng model của các paper LLM:
# - **chen-2025-descriptors**: repo có code nhưng **không công bố weights**. Muốn tái hiện phải tự train speech-LLM
#   (LLM 7B), quá nặng cho bài cuối kì.
# - **OSUM**: có checkpoint (cần khoảng 20 GB VRAM), nhưng môi trường cài đặt riêng (wenet). Để dành cho vòng sau
#   nếu kết quả Qwen2-Audio cho thấy hướng LLM đáng đào sâu.
#
# Lưu ý độ tin cậy: Qwen2-Audio có báo kết quả trên MELD trong paper của họ, nên có thể đã thấy **tập train** của
# MELD khi huấn luyện. Số zero-shot ở đây có thể lạc quan hơn một model thật sự chưa thấy dữ liệu.
# Runtime: **A100** (model bf16 khoảng 16 GB).

# %%
DATASET = "meld"
MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
N_TEST = None          # None = toàn bộ test; đặt 300 để thử nhanh
BATCH = 16

# %%
import os, sys
NB_ROOT = os.environ.get("EAASR_NB", "/content/drive/MyDrive/vp-ck/notebooks")
if NB_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
sys.path.insert(0, NB_ROOT)

# %%
!pip -q install -U transformers accelerate whisper-normalizer soundfile

# %%
import re
import time

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from eaasr.common import LABELS, emotion_from_text, evaluate, get_splits, gpu_info, load_wav, stage_dataset

gpu_info()
labels = LABELS[DATASET]
man = stage_dataset(DATASET)
test = get_splits(man, DATASET)["test"]
if N_TEST:
    test = test.sample(N_TEST, random_state=0).reset_index(drop=True)
print(len(test))

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "left"
model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()

# %%
SHOW = {"joy": "happy", "anger": "angry", "sadness": "sad"}
choices = ", ".join(SHOW.get(l, l) for l in labels)
PROMPTS = {
    "asr": "Transcribe the speech in this audio exactly as spoken, in English. Output only the transcript.",
    "ser": f"What emotion does the speaker express in this audio? Choose exactly one of: {choices}. "
           "Answer with one word only.",
}

def build_inputs(paths, prompt):
    texts, audios = [], []
    for p in paths:
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": p}, {"type": "text", "text": prompt}]}]
        texts.append(processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False))
        audios.append(load_wav(p))
    try:
        return processor(text=texts, audio=audios, sampling_rate=16000, return_tensors="pt", padding=True)
    except TypeError:
        return processor(text=texts, audios=audios, sampling_rate=16000, return_tensors="pt", padding=True)

@torch.no_grad()
def ask(paths, task, max_new_tokens):
    outs, t0 = [], time.time()
    for s in tqdm(range(0, len(paths), BATCH), desc=task):
        inp = build_inputs(paths[s:s + BATCH], PROMPTS[task]).to("cuda")
        gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        gen = gen[:, inp["input_ids"].shape[1]:]
        outs += [t.strip() for t in processor.batch_decode(gen, skip_special_tokens=True)]
        # tqdm does not refresh in a detached job's log, so report progress explicitly.
        if (s // BATCH) % 20 == 0:
            print(f"{task} {len(outs)}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
    return outs

# %%
paths = test.path.tolist()
test["hyp_text"] = ask(paths, "asr", 160)
test["raw_ser"] = ask(paths, "ser", 8)
test["hyp_emotion"] = test.raw_ser.map(lambda t: emotion_from_text(t, labels))
print(test[["text", "hyp_text", "emotion", "raw_ser"]].head(10).to_string())
print(test.raw_ser.str.lower().value_counts().head(15))

# %%
evaluate("qwen2_audio_7b_zeroshot", DATASET, test,
         notes=f"zero-shot, two prompts (ASR, SER), greedy; n={len(test)}; may have seen MELD train")

# %% [markdown]
# Khoảng 40% câu trả lời ASR có câu dẫn ("The exact transcription of the speech is: '...'") thay vì chỉ transcript,
# làm WER tăng vọt. Bỏ câu dẫn (chỉ khi phần trước dấu nháy/dấu hai chấm có từ khóa như "transcription", "speech")
# rồi đánh giá lại. Câu từ chối ("I'm sorry, but I cannot provide the transcript...") vẫn tính là lỗi.

# %%
KEY = re.compile(r"(?i)transcri|speech|content|spoken|audio")
QUOTED = re.compile(r"^(?P<pre>[^'\"‘“]{0,100}?)\s*['\"‘“](?P<body>.*)['\"’”]\s*\.?$", re.S)
COLON = re.compile(r"^(?P<pre>[^:]{0,100}):\s*(?P<body>.+)$", re.S)

def strip_preamble(t):
    for pat in (QUOTED, COLON):
        m = pat.match(t)
        if m and KEY.search(m["pre"]):
            return m["body"].strip()
    return t

test["hyp_text_raw"] = test.hyp_text
test["hyp_text"] = test.hyp_text_raw.astype(str).map(strip_preamble)
print("stripped:", (test.hyp_text != test.hyp_text_raw).sum())
evaluate("qwen2_audio_7b_zeroshot_stripped", DATASET, test,
         notes="as qwen2_audio_7b_zeroshot, with the instruction-following preamble removed from ASR output")
