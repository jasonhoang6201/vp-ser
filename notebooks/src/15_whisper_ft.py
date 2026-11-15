# gpu: L4
# %% [markdown]
# # 15. Whisper large-v3-turbo: zero-shot và fine-tune
#
# So WER với SenseVoice-Small (notebook 10) cho công bằng: cùng train/dev/test (`get_splits`), cùng Whisper normalizer.
#
# - **A. Zero-shot**: model gốc, cách giải mã giống notebook 12 (pipeline, chunk 30 s, tiếng Anh).
# - **B. Fine-tune** toàn bộ model trên tập train, chọn epoch theo **dev WER**, rồi giải mã test giống hệt A.
# - **C. Transcript cho mọi câu** (train, dev, test và câu không nhãn) bằng model đã fine-tune, cho notebook 21
#   (`ASR="whisper"`). Đầu ra: `cache/<dataset>/asr_whisper_ft{FOLD}.csv` (id, split, hyp_text), cùng dạng notebook 20.
#
# Không lưu checkpoint: transcript ở C là tất cả những gì notebook 21 cần.

# %%
DATASET = "iemocap"
WHISPER = "openai/whisper-large-v3-turbo"
EPOCHS = 3
LR = 1e-5
BATCH = 8
ACCUM = 2             # batch hiệu dụng 16, bằng notebook 21
SEED = 42
ZERO_SHOT = True      # False khi đã có kết quả zero-shot của fold này
TRANSCRIBE_ALL = True
TEST_SESSION = 5      # IEMOCAP: session để test (cross-validation 5 fold chạy lần lượt 1..5)
FOLD = "" if DATASET == "meld" or TEST_SESSION == 5 else f"_f{TEST_SESSION}"

# %%
import os, sys
NB_ROOT = os.environ.get("EAASR_NB", "/content/drive/MyDrive/vp-ck/notebooks")
if NB_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
sys.path.insert(0, NB_ROOT)

# %%
!pip -q install -U transformers whisper-normalizer soundfile

# %%
import time

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup, pipeline

import pandas as pd

from eaasr.common import CACHE_DIR, CKPT_DIR, asr_metrics, evaluate, get_splits, gpu_info, load_wav, set_seed, stage_dataset

gpu_info()
set_seed(SEED)
man = stage_dataset(DATASET)
splits = get_splits(man, DATASET, test_session=TEST_SESSION)
print({k: len(v) for k, v in splits.items()})
proc = WhisperProcessor.from_pretrained(WHISPER, language="en", task="transcribe")
GEN = {"language": "en", "task": "transcribe"}

def transcribe(model, df):
    asr = pipeline("automatic-speech-recognition", model=model, tokenizer=proc.tokenizer,
                   feature_extractor=proc.feature_extractor, torch_dtype=torch.float16, device="cuda:0")
    hyps, paths = [], df.path.tolist()
    for s in range(0, len(paths), 256):
        batch = [{"raw": load_wav(p), "sampling_rate": 16000} for p in paths[s:s + 256]]
        hyps += [o["text"].strip() for o in asr(batch, batch_size=32, chunk_length_s=30, generate_kwargs=GEN)]
    return hyps

# %% [markdown]
# ## A. Zero-shot

# %%
if ZERO_SHOT:
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER, torch_dtype=torch.float16).cuda().eval()
    test = splits["test"].copy()
    test["hyp_text"] = transcribe(model, test)
    evaluate(f"whisper_turbo_zeroshot{FOLD}", DATASET, test, notes=f"{WHISPER}, no training")
    del model; torch.cuda.empty_cache()

# %% [markdown]
# ## B. Fine-tune
#
# Câu dài hơn 30 s bị bỏ khỏi tập train: encoder chỉ nhận 30 s, phần audio bị cắt sẽ không khớp với transcript.

# %%
def features(df):
    return [proc.feature_extractor(load_wav(p), sampling_rate=16000, return_tensors="pt").input_features[0].half()
            for p in df.path]

START_ID = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

def label_ids(texts):
    ids = [proc.tokenizer(t).input_ids for t in texts]
    # The model prepends the start token itself when shifting labels right.
    return [x[1:] if x[0] == START_ID else x for x in ids]

train = splits["train"][splits["train"].duration <= 30].reset_index(drop=True)
dev = splits["dev"]
t0 = time.time()
tr_feat, dev_feat = features(train), features(dev)
tr_lab = label_ids(train.text.astype(str).tolist())
print(f"train {len(train)} dev {len(dev)} features {time.time() - t0:.0f}s")

# %%
model = WhisperForConditionalGeneration.from_pretrained(WHISPER, torch_dtype=torch.float32).cuda()
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.generation_config.language, model.generation_config.task = "en", "transcribe"
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * (len(train) // (BATCH * ACCUM))
sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

@torch.no_grad()
def dev_wer():
    model.eval(); model.config.use_cache = True
    hyps = []
    for s in range(0, len(dev), 32):
        x = torch.stack(dev_feat[s:s + 32]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(input_features=x, max_new_tokens=128, **GEN)
        hyps += [h.strip() for h in proc.batch_decode(out, skip_special_tokens=True)]
    model.train(); model.config.use_cache = False
    return asr_metrics(dev.text.tolist(), hyps)["wer"]

best, hist = None, []
model.train()
for ep in range(EPOCHS):
    perm = np.random.permutation(len(train))
    for i, s in enumerate(range(0, len(perm) - BATCH + 1, BATCH)):
        b = perm[s:s + BATCH]
        x = torch.stack([tr_feat[j] for j in b]).cuda().float()
        L = max(len(tr_lab[j]) for j in b)
        y = torch.full((len(b), L), -100, dtype=torch.long)
        for k, j in enumerate(b):
            y[k, :len(tr_lab[j])] = torch.tensor(tr_lab[j])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_features=x, labels=y.cuda()).loss / ACCUM
        loss.backward()
        if (i + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        if i % 100 == 0:
            print(f"  ep{ep} step {i} loss {loss.item() * ACCUM:.3f} {time.time() - t0:.0f}s", flush=True)
    w = dev_wer()
    hist.append({"epoch": ep, "dev_wer": w})
    print(f"  ep{ep} dev WER {w * 100:.2f} | {time.time() - t0:.0f}s", flush=True)
    if best is None or w < best["dev_wer"]:
        best = {"epoch": ep, "dev_wer": w, "state": {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}}

# %%
# AdamW state for 809M parameters is ~6.5 GB; left on the GPU it ran MELD's train-set decoding out of memory.
del opt, sched
model.zero_grad(set_to_none=True)
torch.cuda.empty_cache()
model.load_state_dict(best["state"])
del best["state"]
# Kept on the VM so a different decoding setting can be re-run without fine-tuning again.
model.save_pretrained(CKPT_DIR / f"whisper_turbo_{DATASET}{FOLD}")
model = model.half().eval()
model.config.use_cache = True
test = splits["test"].copy()
test["hyp_text"] = transcribe(model, test)
evaluate(f"whisper_turbo_finetune{FOLD}", DATASET, test,
         extra={"seed": SEED, "best_epoch": best["epoch"], "hist": hist},
         notes=f"{WHISPER} full fine-tune {EPOCHS} ep, lr {LR}, batch {BATCH * ACCUM}, seed {SEED}, "
               f"epoch picked on dev WER")

# %% [markdown]
# ## C. Transcript cho mọi câu

# %%
if TRANSCRIBE_ALL:
    rows = [{"id": i, "split": "test", "hyp_text": h} for i, h in zip(test.id, test.hyp_text)]
    for split in ["train", "dev"]:
        df = splits[split]
        hyps = transcribe(model, df)
        rows += [{"id": i, "split": split, "hyp_text": h} for i, h in zip(df.id, hyps)]
        print(split, len(df), "WER", round(100 * asr_metrics(df.text.tolist(), hyps)["wer"], 2), flush=True)
    # Unlabeled turns only serve as dialogue context; same 60 s cap as notebook 20.
    seen = set(r["id"] for r in rows)
    rest = man[~man.id.isin(seen) & (man.duration <= 60)]
    if len(rest):
        rows += [{"id": i, "split": "unlabeled", "hyp_text": h} for i, h in zip(rest.id, transcribe(model, rest))]
        print("unlabeled", len(rest))
    dst = CACHE_DIR / DATASET / f"asr_whisper_ft{FOLD}.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dst, index=False)
    print("saved", dst, len(rows))
