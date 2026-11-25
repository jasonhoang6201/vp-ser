# gpu: L4
# %% [markdown]
# # 21. Hệ đề xuất: Whisper-FT ASR → RoBERTa fine-tune + ngữ cảnh hội thoại + encoder audio, gated fusion
#
# - Khối ASR: **Whisper large-v3-turbo fine-tune** (notebook 15, phần C sinh transcript cho mọi câu).
# - Nhánh text: RoBERTa-large fine-tune trên transcript ASR (không dùng text đúng), ghép 3 câu trước trong dialogue.
# - Nhánh audio: encoder đông cứng của Whisper large-v3-turbo gốc, layer weight học được.
# - Gộp hai nhánh bằng **gated fusion** (model tự học trọng số text/audio cho từng câu).
#
# Ablation: mỗi cấu hình bỏ hoặc thay đúng một module của hệ đầy đủ.
#
# | Cấu hình | Text | Ngữ cảnh | Audio | Dòng trong report |
# |---|---|---|---|---|
# | `asr_ctx_audio` | transcript ASR | 3 câu trước | có | **hệ đầy đủ** |
# | `asr_ctx` | transcript ASR | 3 câu trước | không | bỏ audio |
# | `asr_audio` | transcript ASR | không | có | bỏ ngữ cảnh |
# | `audio_only` | không | không | có | bỏ text |
# | `gt_ctx` | text đúng | 3 câu trước | không | text đúng, không audio |
# | `gt_ctx_audio` | text đúng | 3 câu trước | có | text đúng, hệ đầy đủ |
#
# Hai so sánh còn lại chạy lại `asr_ctx_audio` với tham số khác:
# - encoder audio WavLM (Bảng IV): `AUDIO="wavlm"`;
# - transcript SenseVoice-FT thay cho Whisper-FT: `ASR="sensevoice"` (cần notebook 10 và 20 của fold đó).
#
# Tên kết quả: `audio` được thay bằng tên encoder (`asr_ctx_whisper_turbo`), transcript Whisper-FT thêm tiền tố `w`
# (`combined_wasr_ctx_whisper_turbo`). Class-weighted cross-entropy, chọn epoch theo **dev UA**.

# %%
DATASET = "iemocap"
ASR = "whisper"        # nguồn transcript: "whisper" (notebook 15) hoặc "sensevoice" (notebook 20)
TEXT_MODEL = "roberta-large"
AUDIO = "whisper_turbo"  # "whisper_turbo" (encoder của large-v3-turbo) hoặc "wavlm"
WAVLM = "microsoft/wavlm-large"
WHISPER = "openai/whisper-large-v3-turbo"
CTX = 3
MAX_LEN = 192
EPOCHS = 5
BATCH = 16
LR = 1e-5
HEAD_LR = 1e-4
SEED = 42
RUNS = ["asr_ctx_audio", "asr_ctx", "asr_audio", "audio_only", "gt_ctx", "gt_ctx_audio"]
TEST_SESSION = 5      # session để test; chạy lần lượt 5, 1, 2, 3, 4 cho 5 fold
FOLD = "" if TEST_SESSION == 5 else f"_f{TEST_SESSION}"
# Ngữ cảnh dựng từ timestamp trong manifest (xem tools/prep_iemocap_official.py).

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
import json, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from eaasr.common import (CACHE_DIR, CKPT_DIR, LABELS, evaluate, get_splits, gpu_info, load_wav, ser_metrics, set_seed,
                          stage_dataset)

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
# - `whisper_turbo`: encoder của Whisper large-v3-turbo gốc (không phải bản fine-tune ASR). Lưu trung bình theo
#   thời gian của **mọi layer**, model tự học trọng số từng layer (kiểu SUPERB). EmoBox đo cùng protocol:
#   Whisper UA 73.54, WavLM 69.47.
# - `wavlm`: WavLM-large, chỉ học tự giám sát trên sách nói, **chưa từng thấy nhãn cảm xúc**. Cùng cách lưu mọi layer.
#
# Embedding tính một lần rồi lưu cache, các fold và cấu hình sau dùng lại.

# %%
ecache = cache / {"wavlm": "wavlm_large_layers.npz", "whisper_turbo": "whisper_large_v3_turbo_layers.npz"}[AUDIO]
if AUDIO == "whisper_turbo" and not ecache.exists():
    from transformers import AutoFeatureExtractor, WhisperModel
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
z = np.load(ecache, allow_pickle=True)
# The cache is shared across folds, whose row order differs, so align by id.
pos = {u: i for i, u in enumerate(z["ids"])}
audio_all = torch.tensor(z["emb"][[pos[u] for u in allu.id]], dtype=torch.float32)
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
    # Speaker ids are per session and the test session's speakers never appear in train,
    # so a name prefix would carry no information.
    turns = text.astype(str).tolist()
    hist = [[] for _ in range(len(allu))]
    if ctx:
        assert "start" in man.columns, "context needs the official release (timestamps)"
        # Context runs over every turn of the dialogue, labeled or not, in time order: half of IEMOCAP's turns
        # fall outside the 4 classes, and skipping them would glue together turns that were not adjacent.
        full = man.merge(asr_all[["id", "hyp_text"]], on="id", how="inner", validate="one_to_one")
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
    "asr_ctx_audio": dict(train="asr", eval="asr", ctx=CTX, audio=True, text=True),
    "asr_ctx":       dict(train="asr", eval="asr", ctx=CTX, audio=False, text=True),
    "asr_audio":     dict(train="asr", eval="asr", ctx=0, audio=True, text=True),
    "audio_only":    dict(train="asr", eval="asr", ctx=0, audio=True, text=False),
    "gt_ctx":        dict(train="gt", eval="gt", ctx=CTX, audio=False, text=True),
    "gt_ctx_audio":  dict(train="gt", eval="gt", ctx=CTX, audio=True, text=True),
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

def train_one(name, seed, keep_state=False):
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
            # and RoBERTa got too little gradient to learn.
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
            if keep_state:
                best["state"] = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    del model, opt; torch.cuda.empty_cache()
    return best, hist

# %% [markdown]
# ## 4. Chạy từng cấu hình

# %%
OUT = cache / "combined"
OUT.mkdir(parents=True, exist_ok=True)

def tag(name):
    t = name.replace("audio", AUDIO) if CONFIGS[name]["audio"] else name
    # SenseVoice-FT and Whisper-FT transcripts give results with the same config name, so the
    # Whisper-FT ones get a "w" prefix (wasr_*) to sit next to the SenseVoice-FT ones (asr_*).
    if ASR == "whisper" and t.startswith("asr"):
        t = "w" + t
    return t + FOLD

def save_weights(name, best, hist):
    """Weights of the full system, in the layout notebook 30 loads (model.safetensors, tokenizer, config.json)."""
    from safetensors.torch import save_file
    d = CKPT_DIR / f"ser_{tag(name)}"
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in best.pop("state").items()}, str(d / "model.safetensors"))
    tok.save_pretrained(d)
    cfg = {"class": "TextAudio (notebooks/src/21_combined.py)", "config": name, **CONFIGS[name], "labels": labels,
           "text_model": TEXT_MODEL, "max_len": MAX_LEN, "context_turns": CTX, "asr": ASR_NAME,
           # Inference must rebuild the audio features as in section 1: every layer, per-frame layer norm,
           # mean over real frames, first 20 s of audio.
           "audio_encoder": WHISPER if AUDIO == "whisper_turbo" else WAVLM, "audio_feature_shape": list(audio_all.shape[1:]),
           "test_session": TEST_SESSION, "seed": SEED, "best_epoch": best["epoch"], "dev_ua": best["dev_ua"],
           "gate_text_audio": best["gate"], "hist": hist}
    with open(d / "config.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    print("saved", d)

for name in RUNS:
    f = OUT / f"{tag(name)}_s{SEED}.npz"
    print(f"== {name} | {time.strftime('%H:%M:%S')}", flush=True)
    keep = name == "asr_ctx_audio"
    best, hist = train_one(name, SEED, keep_state=keep)
    if keep:
        save_weights(name, best, hist)
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
    if name == "asr_ctx_audio":
        with open(CKPT_DIR / f"ser_{tag(name)}" / "metrics_test.json", "w") as fh:
            json.dump(m, fh, indent=2, ensure_ascii=False)
    summary.append({"system": tag(name), "UA": m["ser"]["ua"] * 100, "WA": m["ser"]["wa"] * 100,
                    "wF1": m["ser"]["weighted_f1"] * 100,
                    "gate audio": meta["gate"][1] if meta["gate"] else None})
print(pd.DataFrame(summary).round(2).to_string())

# %% [markdown]
# ## 6. (Tùy chọn) Đẩy weight hệ đầy đủ lên Hugging Face Hub
#
# Bỏ comment để chạy: nhập HF token có quyền write, rồi weight của fold này lên `fold_s{TEST_SESSION}/ser`
# (layout notebook 30 đọc). Repo phải private: license IEMOCAP cấm phát tán dữ liệu.

# %%
# from huggingface_hub import HfApi, get_token, notebook_login
# HUB_REPO = "master-122025/eaasr-iemocap"
# if get_token() is None:
#     notebook_login()
# api = HfApi()
# api.create_repo(HUB_REPO, private=True, exist_ok=True)
# assert api.model_info(HUB_REPO).private, f"{HUB_REPO} is public, refusing to upload"
# api.upload_folder(folder_path=str(CKPT_DIR / f"ser_{tag('asr_ctx_audio')}"), path_in_repo=f"fold_s{TEST_SESSION}/ser",
#                   repo_id=HUB_REPO, commit_message=f"SER full system, test session {TEST_SESSION}")
