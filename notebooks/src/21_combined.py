# gpu: L4
# %% [markdown]
# # 21. Hệ đề xuất: SenseVoice-FT ASR → RoBERTa fine-tune + ngữ cảnh hội thoại + encoder audio, gated fusion
#
# - Khối ASR: **SenseVoice-Small fine-tune** (notebook 10, transcript cho mọi câu ở notebook 20).
# - Nhánh text: RoBERTa-large fine-tune trên transcript ASR (không dùng text đúng), ghép 3 câu trước trong dialogue.
# - Nhánh audio: encoder đông cứng (Whisper large v3 cho IEMOCAP), layer weight học được.
# - Gộp hai nhánh bằng **gated fusion** (model tự học trọng số text/audio cho từng câu).
#
# Ablation: mỗi cấu hình bỏ hoặc thay đúng một module của hệ đầy đủ.
#
# | Cấu hình | Text | Ngữ cảnh | Audio | Câu hỏi |
# |---|---|---|---|---|
# | `asr_ctx_e2v` | transcript ASR | 3 câu trước | có | **hệ đầy đủ** |
# | `asr_ctx` | transcript ASR | 3 câu trước | không | bỏ audio |
# | `asr_e2v` | transcript ASR | không | có | bỏ ngữ cảnh |
# | `e2v_only` | không | không | có | bỏ text |
# | `gt_ctx` | text đúng | 3 câu trước | không | cận trên: lỗi ASR làm mất bao nhiêu |
#
# Thay encoder audio (Whisper so với WavLM) bằng cách chạy lại `asr_ctx_e2v` với `AUDIO="wavlm"`.
# Trong tên kết quả, `e2v` được thay bằng tên encoder thật (ví dụ `asr_ctx_whisper`).
#
# Class-weighted cross-entropy, chọn epoch theo **dev UA**, mỗi cấu hình chạy một lần với `SEED`.

# %%
DATASET = "iemocap"
ASR = "sensevoice"     # nguồn transcript: "sensevoice" (notebook 20) hoặc "whisper" (Whisper turbo FT, notebook 15)
TEXT_MODEL = "roberta-large"
E2V = "emotion2vec/emotion2vec_plus_large"
AUDIO = "whisper"      # "e2v", "wavlm" hoặc "whisper"
WAVLM = "microsoft/wavlm-large"
WHISPER = "openai/whisper-large-v3"
CTX = 3
MAX_LEN = 192
EPOCHS = 5
BATCH = 16
LR = 1e-5
HEAD_LR = 1e-4
SEED = 42
RUNS = ["asr_ctx_e2v", "asr_ctx", "asr_e2v", "e2v_only", "gt_ctx"]
TEST_SESSION = 5      # IEMOCAP: session để test (cross-validation 5 fold chạy lần lượt 1..5)
FOLD = "" if DATASET == "meld" or TEST_SESSION == 5 else f"_f{TEST_SESSION}"
# IEMOCAP: ngữ cảnh dựng từ timestamp trong manifest (xem tools/prep_iemocap_official.py).

# %%
import os, sys
NB_ROOT = os.environ.get("EAASR_NB", "/content/drive/MyDrive/vp-ck/notebooks")
if NB_ROOT.startswith("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
sys.path.insert(0, NB_ROOT)

# %%
!pip -q install -U transformers funasr whisper-normalizer soundfile

# %%
import json, subprocess, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from eaasr.common import CACHE_DIR, LABELS, RESULTS_DIR, evaluate, get_splits, gpu_info, ser_metrics, set_seed, stage_dataset

gpu_info()
labels = LABELS[DATASET]
lab2id = {l: i for i, l in enumerate(labels)}
man = stage_dataset(DATASET)
splits = get_splits(man, DATASET, test_session=TEST_SESSION)
allu = pd.concat([splits[s].assign(_split=s) for s in ["train", "dev", "test"]], ignore_index=True)
cache = CACHE_DIR / DATASET
# Transcripts must come from the ASR fine-tuned without this fold's test session.
ASR_NAME = {"sensevoice": "SenseVoice-FT", "whisper": "Whisper-turbo-FT"}[ASR]
asr_all = pd.read_csv(cache / f"asr_{ASR}_ft{FOLD}.csv", keep_default_na=False)
allu = allu.merge(asr_all[["id", "hyp_text"]], on="id", how="left", validate="one_to_one")
assert allu.hyp_text.notna().all(), f"missing {ASR_NAME} transcripts"
y_all = allu.emotion.map(lab2id).values
idx = {s: np.where(allu._split.values == s)[0] for s in ["train", "dev", "test"]}
print({k: len(v) for k, v in idx.items()})

# %% [markdown]
# ## 1. Embedding audio
#
# - `e2v`: emotion2vec+ large, một vector cho mỗi câu. Chỉ dùng cho MELD: emotion2vec+ đã thấy IEMOCAP lúc train
#   (zero-shot UA 86).
# - `wavlm`: WavLM-large, chỉ học tự giám sát trên sách nói, **chưa từng thấy nhãn cảm xúc**. Lưu trung bình theo
#   thời gian của **mọi layer**, model tự học trọng số từng layer (kiểu SUPERB).
# - `whisper`: encoder Whisper large v3, cùng cách lưu mọi layer như `wavlm`. EmoBox đo cùng protocol: Whisper UA 73.54,
#   WavLM 69.47. Không rõ dữ liệu train của Whisper có IEMOCAP hay không.

# %%
from funasr import AutoModel

ecache = cache / {"e2v": "e2v_plus_large_utt.npz", "wavlm": "wavlm_large_layers.npz",
                  "whisper": "whisper_large_v3_layers.npz"}[AUDIO]
if AUDIO == "whisper" and not ecache.exists():
    from transformers import AutoFeatureExtractor, WhisperModel
    from eaasr.common import load_wav
    fe = AutoFeatureExtractor.from_pretrained(WHISPER)
    wm = WhisperModel.from_pretrained(WHISPER, torch_dtype=torch.float16).encoder.cuda().eval()
    paths, embs, t0 = allu.path.tolist(), [], time.time()
    with torch.no_grad():
        for s in range(0, len(paths), 16):
            wavs = [load_wav(p)[: 16000 * 20] for p in paths[s:s + 16]]
            x = fe(wavs, sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
            hs = torch.stack(wm(x, output_hidden_states=True).hidden_states, 1)
            for j, w in enumerate(wavs):
                # Input is always padded to 30 s: pool real frames only (320 samples per encoder frame),
                # otherwise short utterances average mostly padding.
                n = max(1, -(-len(w) // 320))
                embs.append(F.layer_norm(hs[j, :, :n].float(), hs.shape[-1:]).mean(1).half().cpu().numpy())
            if s % 1600 == 0:
                print(f"whisper {s}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
    emb = np.stack(embs)
    assert np.isfinite(emb).all(), "fp16 overflow in Whisper hidden states"
    np.savez(ecache, ids=np.array(allu.id.tolist()), emb=emb)
    del wm; torch.cuda.empty_cache()
if AUDIO == "wavlm" and not ecache.exists():
    from transformers import AutoFeatureExtractor, WavLMModel
    from eaasr.common import load_wav
    fe = AutoFeatureExtractor.from_pretrained(WAVLM)
    wl = WavLMModel.from_pretrained(WAVLM, torch_dtype=torch.float16).cuda().eval()
    embs, t0 = [], time.time()
    with torch.no_grad():
        for i, p in enumerate(allu.path):
            # 20 s cap keeps attention memory bounded; few utterances are longer.
            wav = load_wav(p)[: 16000 * 20]
            x = fe(wav, sampling_rate=16000, return_tensors="pt").input_values.cuda().half()
            hs = wl(x, output_hidden_states=True).hidden_states
            # Per-frame layer norm before pooling: raw layer scales differ by orders of magnitude.
            embs.append(torch.stack([F.layer_norm(h[0].float(), h.shape[-1:]).mean(0) for h in hs]).half().cpu().numpy())
            if i % 2000 == 0:
                print(f"wavlm {i}/{len(allu)} {time.time() - t0:.0f}s", flush=True)
    np.savez(ecache, ids=np.array(allu.id.tolist()), emb=np.stack(embs))
    del wl; torch.cuda.empty_cache()
if ecache.exists():
    z = np.load(ecache, allow_pickle=True)
    e2v_ids, e2v_emb = list(z["ids"]), z["emb"]
else:
    e2v = AutoModel(model=E2V, hub="hf", disable_update=True)
    embs, t0 = [], time.time()
    for i, p in enumerate(allu.path):
        r = e2v.generate(p, granularity="utterance", extract_embedding=True)[0]
        embs.append(np.asarray(r["feats"], dtype=np.float32).reshape(-1))
        if i % 2000 == 0:
            print(f"e2v {i}/{len(allu)} {time.time() - t0:.0f}s", flush=True)
    e2v_ids, e2v_emb = allu.id.tolist(), np.stack(embs)
    np.savez(ecache, ids=np.array(e2v_ids), emb=e2v_emb)
    del e2v; torch.cuda.empty_cache()
# The cache is shared across folds, whose row order differs, so align by id.
pos = {u: i for i, u in enumerate(e2v_ids)}
audio_all = torch.tensor(e2v_emb[[pos[u] for u in allu.id]], dtype=torch.float32)
print(AUDIO, tuple(audio_all.shape))

# %% [markdown]
# ## 2. Dựng input text: câu hiện tại, ghép với 3 câu trước trong dialogue
#
# RoBERTa nhận một cặp (ngữ cảnh, câu hiện tại). Nếu quá dài thì cắt **đầu** ngữ cảnh, để giữ các câu gần nhất.

# %%
from transformers import AutoConfig, AutoModel as HFAutoModel, AutoTokenizer, get_linear_schedule_with_warmup

tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
tok.truncation_side = "left"

def build_inputs(source, ctx):
    text = allu.text if source == "gt" else allu.hyp_text
    # IEMOCAP speaker ids are per session and the test session's speakers never appear in train,
    # so a name prefix carries no information there.
    if DATASET == "meld":
        turns = (allu.speaker.astype(str) + ": " + text.astype(str)).tolist()
    else:
        turns = text.astype(str).tolist()
    hist = [[] for _ in range(len(allu))]
    if ctx and DATASET == "meld":
        utt = allu.id.str.extract(r"utt(\d+)$")[0].astype(int)
        order = allu.assign(_utt=utt).sort_values(["_split", "dialogue_id", "_utt"]).index
        for key, g in allu.loc[order].groupby(["_split", "dialogue_id"], sort=False):
            rows = list(g.index)
            for j, r in enumerate(rows):
                hist[r] = [turns[k] for k in rows[max(0, j - ctx):j]]
    elif ctx:
        assert "start" in man.columns, "IEMOCAP context needs the official release (timestamps)"
        # Context runs over every turn of the dialogue, labeled or not, in time order: half of IEMOCAP's turns
        # fall outside the 4 classes, and skipping them would glue together turns that were not adjacent.
        full = man[man.duration <= 60].merge(asr_all[["id", "hyp_text"]], on="id", how="inner", validate="one_to_one")
        full = full.assign(_t=(full.text if source == "gt" else full.hyp_text).astype(str))
        prev_of = {}
        for _, g in full.sort_values(["dialogue_id", "start"]).groupby("dialogue_id", sort=False):
            ts = g._t.tolist()
            for j, uid in enumerate(g.id):
                prev_of[uid] = ts[max(0, j - ctx):j]
        hist = [prev_of[u] for u in allu.id]
    ids = []
    for r in range(len(allu)):
        c = " ".join(hist[r])
        # Whisper sometimes loops ("ha ha ha ..." for 400+ words); a turn that alone fills MAX_LEN leaves no room
        # for context, and only_first truncation then raises instead of cutting the turn.
        if c and len(tok(turns[r], add_special_tokens=False)["input_ids"]) > MAX_LEN - 8:
            c = ""
        enc = tok(c, turns[r], truncation="only_first", max_length=MAX_LEN) if c else \
              tok(turns[r], truncation=True, max_length=MAX_LEN)
        ids.append(enc["input_ids"])
    return ids

INPUTS = {}
def get_inputs(source, ctx):
    key = (source, ctx)
    if key not in INPUTS:
        INPUTS[key] = build_inputs(source, ctx)
        lens = np.array([len(x) for x in INPUTS[key]])
        print(f"inputs {source} ctx={ctx}: mean len {lens.mean():.0f}, max {lens.max()}, "
              f"truncated {(lens >= MAX_LEN).mean() * 100:.1f}%")
    return INPUTS[key]

print(tok.decode(get_inputs("asr", CTX if any("ctx" in r for r in RUNS) else 0)[idx["test"][5]]))

# %% [markdown]
# ## 3. Model và vòng train

# %%
CONFIGS = {
    "asr_ctx_e2v": dict(train="asr", eval="asr", ctx=CTX, audio=True, text=True),
    "asr_ctx":     dict(train="asr", eval="asr", ctx=CTX, audio=False, text=True),
    "asr_e2v":     dict(train="asr", eval="asr", ctx=0, audio=True, text=True),
    "e2v_only":    dict(train="asr", eval="asr", ctx=0, audio=True, text=False),
    "gt_ctx":      dict(train="gt", eval="gt", ctx=CTX, audio=False, text=True),
}

class TextAudio(nn.Module):
    def __init__(self, n_classes, audio, text):
        super().__init__()
        self.audio, self.text = audio, text
        d = AutoConfig.from_pretrained(TEXT_MODEL).hidden_size
        if text:
            self.enc = HFAutoModel.from_pretrained(TEXT_MODEL)
        if audio:
            if audio_all.dim() == 3:
                self.layer_w = nn.Parameter(torch.zeros(audio_all.shape[1]))
            self.a_proj = nn.Sequential(nn.LayerNorm(audio_all.shape[-1]), nn.Dropout(0.2),
                                        nn.Linear(audio_all.shape[-1], d), nn.Tanh())
        if audio and text:
            self.t_norm = nn.LayerNorm(d)
            self.gate = nn.Linear(2 * d, 2)
            self.head_t = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, n_classes))
            self.head_a = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, n_classes))
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, n_classes))

    def forward(self, ids, mask, a=None):
        if self.audio:
            if a.dim() == 3:
                a = (F.softmax(self.layer_w, 0)[None, :, None] * a).sum(1)
            ha = self.a_proj(a)
            if not self.text:
                return self.head(ha), None, ()
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        if not self.audio:
            return self.head(h), None, ()
        h = self.t_norm(h)
        g = F.softmax(self.gate(torch.cat([h, ha], -1)).float(), -1)
        fused = g[:, :1] * h + g[:, 1:] * ha
        return self.head(fused), g, (self.head_t(h), self.head_a(ha))

def batches(rows, inputs, bs):
    for s in range(0, len(rows), bs):
        b = rows[s:s + bs]
        L = max(len(inputs[r]) for r in b)
        ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(b), L), dtype=torch.long)
        for i, r in enumerate(b):
            ids[i, :len(inputs[r])] = torch.tensor(inputs[r])
            mask[i, :len(inputs[r])] = 1
        yield b, ids.cuda(), mask.cuda(), audio_all[b].cuda()

@torch.no_grad()
def predict(model, rows, inputs):
    model.eval()
    probs, gates = [], []
    for b, ids, mask, a in batches(rows, inputs, 64):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, g, _ = model(ids, mask, a)
        probs.append(F.softmax(logits.float(), -1).cpu())
        if g is not None:
            gates.append(g.cpu())
    return torch.cat(probs).numpy(), (torch.cat(gates).mean(0).tolist() if gates else None)

def ua_of(rows, prob):
    return ser_metrics([labels[i] for i in y_all[rows]], [labels[i] for i in prob.argmax(1)], labels)["ua"]

def train_one(name, seed):
    cfg = CONFIGS[name]
    set_seed(seed)
    tr_in, ev_in = get_inputs(cfg["train"], cfg["ctx"]), get_inputs(cfg["eval"], cfg["ctx"])
    model = TextAudio(len(labels), cfg["audio"], cfg["text"]).cuda()
    enc_p = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    other = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    groups = ([{"params": enc_p, "lr": LR}] if enc_p else []) + [{"params": other, "lr": HEAD_LR}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = EPOCHS * ((len(idx["train"]) + BATCH - 1) // BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    counts = np.bincount(y_all[idx["train"]], minlength=len(labels)).clip(min=1)
    w = torch.tensor(counts.sum() / (len(labels) * counts), dtype=torch.float32).cuda()
    y = torch.tensor(y_all).cuda()
    best, hist, t0 = None, [], time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(idx["train"]).tolist()
        for step, (b, ids, mask, a) in enumerate(batches(perm, tr_in, BATCH)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _, aux = model(ids, mask, a)
            loss = F.cross_entropy(logits.float(), y[b], weight=w)
            # Without per-branch losses the gate locked onto the fast-learning audio projection
            # (gate 0.88 audio, test UA = audio-only probe) and RoBERTa got too little gradient to learn.
            for x in aux:
                loss = loss + F.cross_entropy(x.float(), y[b], weight=w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if step % 200 == 0:
                print(f"  {name} ep{ep} step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        dev_p, _ = predict(model, idx["dev"].tolist(), ev_in)
        test_p, gate = predict(model, idx["test"].tolist(), ev_in)
        dua = ua_of(idx["dev"], dev_p)
        hist.append({"epoch": ep, "dev_ua": dua, "test_ua": ua_of(idx["test"], test_p), "sec": time.time() - t0})
        print(f"  {name} ep{ep} dev UA {dua * 100:.2f} | {time.time() - t0:.0f}s", flush=True)
        # Test probabilities are kept only for the epoch chosen on dev; test UA per epoch is logged, never used to pick.
        if best is None or dua > best["dev_ua"]:
            best = {"dev_ua": dua, "epoch": ep, "dev_prob": dev_p, "test_prob": test_p, "gate": gate}
    del model, opt; torch.cuda.empty_cache()
    return best, hist

# %% [markdown]
# ## 4. Chạy từng cấu hình
#
# Mỗi lần chạy lưu ra một file riêng, nên chạy lại notebook sẽ bỏ qua cấu hình đã xong (khi VM bị thu hồi giữa chừng).

# %%
OUT = cache / "combined"
OUT.mkdir(parents=True, exist_ok=True)

def tag(name):
    t = name.replace("e2v", AUDIO) if CONFIGS[name]["audio"] else name
    # Keeps the SenseVoice-transcript results under their old names next to the Whisper-transcript ones (wasr_*).
    if ASR == "whisper" and t.startswith("asr"):
        t = "w" + t
    return t + FOLD

for name in RUNS:
    f = OUT / f"{tag(name)}_s{SEED}.npz"
    if f.exists():
        continue
    print(f"== {name} | {time.strftime('%H:%M:%S')}", flush=True)
    best, hist = train_one(name, SEED)
    np.savez(f, dev_prob=best["dev_prob"], test_prob=best["test_prob"],
             meta=json.dumps({"best_epoch": best["epoch"], "dev_ua": best["dev_ua"], "gate": best["gate"], "hist": hist}))
    print(f"== {name} best ep{best['epoch']} dev UA {best['dev_ua'] * 100:.2f} "
          f"test UA {ua_of(idx['test'], best['test_prob']) * 100:.2f} gate(text, audio) {best['gate']}", flush=True)

# %% [markdown]
# ## 5. Lưu kết quả vào results/

# %%
test = allu.iloc[idx["test"]].reset_index(drop=True)
summary = []
for name in RUNS:
    cfg = CONFIGS[name]
    z = np.load(OUT / f"{tag(name)}_s{SEED}.npz")
    meta = json.loads(str(z["meta"]))
    out = test.copy()
    out["hyp_emotion"] = [labels[i] for i in z["test_prob"].argmax(1)]
    if cfg["eval"] == "gt":
        out = out.drop(columns=["hyp_text"])
    text_desc = f"{TEXT_MODEL} fine-tuned ({EPOCHS} ep, lr {LR}) on {cfg['train']} text" if cfg["text"] else "no text"
    m = evaluate(f"combined_{tag(name)}", DATASET, out,
                 extra={"seed": SEED, "best_epoch": meta["best_epoch"], "dev_ua": meta["dev_ua"],
                        "gate": meta["gate"], "config": cfg},
                 notes=f"{ASR_NAME} ASR; {text_desc}; ctx {cfg['ctx']}; "
                       f"audio {AUDIO if cfg['audio'] else 'none'}; seed {SEED}"
                       + ("; GROUND-TRUTH text (upper bound)" if cfg["eval"] == "gt" else ""))
    summary.append({"system": tag(name), "UA": m["ser"]["ua"] * 100, "WA": m["ser"]["wa"] * 100,
                    "wF1": m["ser"]["weighted_f1"] * 100,
                    "gate audio": meta["gate"][1] if meta["gate"] else None})
print(pd.DataFrame(summary).round(2).to_string())

# %%
# Single archive so results survive even if the VM is reclaimed right after the job.
subprocess.run(f"tar -czf /content/combined_out.tgz -C {RESULTS_DIR.parent} {RESULTS_DIR.name}/{DATASET} "
               f"-C {CACHE_DIR.parent} {CACHE_DIR.name}/{DATASET}/combined", shell=True, check=True)
print("archive /content/combined_out.tgz")
