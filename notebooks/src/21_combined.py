# gpu: L4
# %% [markdown]
# # 21. Model kết hợp: SenseVoice-FT ASR → RoBERTa fine-tune (chịu lỗi ASR) + ngữ cảnh hội thoại + emotion2vec
#
# Rút ra từ baseline MELD:
# - ASR tốt nhất là **SenseVoice-Small fine-tune** (WER test 27.4). Giữ nguyên làm khối ASR.
# - SER từ text mạnh hơn từ audio (text probe UA 38 so với e2v probe UA 31), nhưng text đúng cho UA 47.6:
#   khoảng cách khoảng 9 điểm là do **lỗi ASR**, và do probe chỉ dùng embedding đông cứng (frozen).
# - Học chung ASR + SER (MTL, ENT) không giúp SER: gradient hai task gần như vuông góc (cosine khoảng 0).
#   Vì vậy model kết hợp **tách** ASR và SER, rồi tập trung làm nhánh text tốt hơn.
#
# Giả thuyết kiểm tra ở đây (mỗi dòng là một cấu hình, so với dòng ngay trên):
#
# | Cấu hình | Train trên | Test trên | Ngữ cảnh | Audio | Câu hỏi |
# |---|---|---|---|---|---|
# | `gt2asr` | text đúng | ASR | không | không | train trên text sạch rồi gặp text lỗi thì sao |
# | `asr` | ASR | ASR | không | không | train trên chính transcript ASR có giúp chịu lỗi không |
# | `asr_ctx` | ASR | ASR | 3 câu trước | không | ngữ cảnh hội thoại giúp bao nhiêu |
# | `asr_e2v` | ASR | ASR | không | có | audio bù được phần text mất do lỗi ASR không |
# | `asr_ctx_e2v` | ASR | ASR | 3 câu trước | có | **model kết hợp đầy đủ** |
# | `gt_ctx` | text đúng | text đúng | 3 câu trước | không | cận trên (không dùng thật được) |
#
# Ngữ cảnh chỉ lấy các câu **trước** trong cùng dialogue, và lấy từ **transcript ASR** (không dùng text đúng), đúng
# như lúc dùng thật. Tên người nói lấy từ nhãn của MELD; dùng thật thì cần thêm khối phân tách người nói (diarization).
#
# Audio: embedding emotion2vec+ large (đông cứng), ghép với vector [CLS] của RoBERTa bằng **gated fusion**
# (model tự học trọng số text/audio cho từng câu). Ghi lại trọng số gate trung bình để biết model dựa vào audio bao nhiêu.
# Mỗi nhánh có thêm một head phụ với loss riêng: lần chạy đầu không có loss phụ thì gate dồn 88% vào audio
# (nhánh audio học nhanh hơn), RoBERTa gần như không học, test UA chỉ bằng probe audio (31.7).
#
# Quy ước giống notebook 12: class-weighted cross-entropy, chọn epoch theo **dev UA**, báo mean ± std qua seed,
# và dự đoán cuối là trung bình xác suất các seed (ensemble).

# %%
DATASET = "meld"
TEXT_MODEL = "roberta-large"
E2V = "emotion2vec/emotion2vec_plus_large"
AUDIO = "e2v"          # "e2v", "wavlm" hoặc "whisper"
WAVLM = "microsoft/wavlm-large"
WHISPER = "openai/whisper-large-v3"
POOL = "mean"          # "mean": vector câu đã cache; "fmean", "attn": pool trên frame, chỉ dùng với AUDIO="whisper"
FRAME_LAYERS = [8, 16, 24, 32]
FRAME_STRIDE = 4
CTX = 3
MAX_LEN = 192
EPOCHS = 5
BATCH = 16
LR = 1e-5
HEAD_LR = 1e-4
SEEDS = [0, 1, 2]
RUNS = ["asr_ctx_e2v", "asr_ctx", "asr", "asr_e2v", "gt2asr", "gt_ctx"]
TEST_SESSION = 5      # IEMOCAP: session để test (cross-validation 5 fold chạy lần lượt 1..5)
FOLD = "" if DATASET == "meld" or TEST_SESSION == 5 else f"_f{TEST_SESSION}"
# IEMOCAP: ngữ cảnh cần bản chính thức của USC (có timestamp, xem tools/prep_iemocap_official.py). Bản HuggingFace
# không có timestamp nên chỉ chạy được các cấu hình không ngữ cảnh, ví dụ RUNS = ["asr_e2v", "asr", "gt2asr", "gt"].

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
import json, re, subprocess, time

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
# Transcripts must come from the SenseVoice fine-tuned without this fold's test session.
asr_all = pd.read_csv(cache / f"asr_sensevoice_ft{FOLD}.csv", keep_default_na=False)
allu = allu.merge(asr_all[["id", "hyp_text"]], on="id", how="left", validate="one_to_one")
assert allu.hyp_text.notna().all(), "missing SenseVoice-FT transcripts"
y_all = allu.emotion.map(lab2id).values
idx = {s: np.where(allu._split.values == s)[0] for s in ["train", "dev", "test"]}
print({k: len(v) for k, v in idx.items()})

# %% [markdown]
# ## 1. Embedding audio
#
# - `e2v`: emotion2vec+ large, một vector cho mỗi câu (khoảng 7 phút trên MELD).
# - `wavlm`: WavLM-large, chỉ học tự giám sát trên sách nói, **chưa từng thấy nhãn cảm xúc**. Dùng cho IEMOCAP vì
#   emotion2vec+ đã thấy IEMOCAP lúc train (zero-shot UA 86). Lưu trung bình theo thời gian của **mọi layer**,
#   model tự học trọng số từng layer (kiểu SUPERB).
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

# All 33 layers at 50 Hz would not fit on disk, so frame pooling keeps 4 layers at 12.5 Hz. The matrix is a .npy:
# the HF sync and the pull both skip that extension, and a fresh VM rebuilds it in minutes.
if POOL != "mean":
    assert AUDIO == "whisper", "frame pooling is only implemented for Whisper"
    fcache, fidx = cache / "whisper_large_v3_frames.npy", cache / "whisper_large_v3_frames_idx.npz"
    if not fcache.exists():
        from transformers import AutoFeatureExtractor, WhisperModel
        from eaasr.common import load_wav
        fe = AutoFeatureExtractor.from_pretrained(WHISPER)
        wm = WhisperModel.from_pretrained(WHISPER, torch_dtype=torch.float16).encoder.cuda().eval()
        paths, chunks, t0 = allu.path.tolist(), [], time.time()
        with torch.no_grad():
            for s in range(0, len(paths), 16):
                wavs = [load_wav(p)[: 16000 * 20] for p in paths[s:s + 16]]
                x = fe(wavs, sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
                hs = wm(x, output_hidden_states=True).hidden_states
                hs = torch.stack([hs[l] for l in FRAME_LAYERS], 1)
                for j, w in enumerate(wavs):
                    n = max(1, -(-len(w) // 320))
                    h = F.layer_norm(hs[j, :, :n].float(), hs.shape[-1:])
                    h = F.avg_pool1d(h.transpose(1, 2), FRAME_STRIDE, FRAME_STRIDE, ceil_mode=True).transpose(1, 2)
                    chunks.append(h.transpose(0, 1).half().cpu().numpy())
                if s % 1600 == 0:
                    print(f"whisper frames {s}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
        off = np.cumsum([0] + [len(c) for c in chunks])
        np.savez(fidx, ids=np.array(allu.id.tolist()), off=off)
        chunks = np.concatenate(chunks)
        assert np.isfinite(chunks).all(), "fp16 overflow in Whisper hidden states"
        np.save(fcache, chunks)
        del wm, chunks; torch.cuda.empty_cache()
    z = np.load(fidx)
    fpos = {u: i for i, u in enumerate(z["ids"].tolist())}
    off = z["off"].tolist()
    fspan = [(off[fpos[u]], off[fpos[u] + 1]) for u in allu.id]
    frames = torch.from_numpy(np.load(fcache))
    print("frames", tuple(frames.shape))

# %% [markdown]
# ## 2. Dựng input text: "Người nói: câu" cho câu hiện tại, ghép với 3 câu trước trong dialogue
#
# RoBERTa nhận một cặp (ngữ cảnh, câu hiện tại). Nếu quá dài thì cắt **đầu** ngữ cảnh, để giữ các câu gần nhất.
#
# Cấu hình `pros` thêm mô tả prosody vào đầu câu hiện tại, ví dụ "pitch high, volume low, speed normal. ...". Cao độ và
# âm lượng đo từ audio, tốc độ là số từ ASR trên giây. Mỗi đại lượng được z-score **theo từng người nói** trên mọi câu của
# người đó, không dùng nhãn, rồi chia 3 mức tại ±0.5 (ngưỡng đặt trước, không tinh chỉnh). Dùng thật thì cần thống kê
# chạy dần của người nói, vì ở đây thống kê của người nói trong tập test được tính từ chính các câu test.

# %%
from transformers import AutoModel as HFAutoModel, AutoTokenizer, get_linear_schedule_with_warmup

tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
tok.truncation_side = "left"

def raw_prosody(path):
    # Runs in joblib workers, whose sys.path does not include the eaasr package.
    import librosa
    wav, _ = librosa.load(path, sr=16000, mono=True)
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=256)[0]
    f0 = librosa.yin(wav, fmin=65, fmax=500, sr=16000, frame_length=1024, hop_length=256)
    n = min(len(rms), len(f0))
    rms, f0 = rms[:n], f0[:n]
    # Frames within 20 dB of the loudest one: pauses would otherwise pull down volume and add noise to pitch.
    speech = rms >= 0.1 * rms.max()
    return float(np.median(np.log(f0[speech]))), float(20 * np.log10(rms[speech].mean() + 1e-8))

def prosody_phrases():
    pcache = cache / "prosody_raw.csv"
    pool = man[man.duration <= 60]
    if not pcache.exists():
        from joblib import Parallel, delayed
        t0 = time.time()
        vals = Parallel(n_jobs=-1)(delayed(raw_prosody)(p) for p in pool.path)
        pd.DataFrame(vals, columns=["pitch", "volume"]).assign(id=pool.id.values).to_csv(pcache, index=False)
        print(f"prosody {len(pool)} utterances {time.time() - t0:.0f}s", flush=True)
    d = pool.merge(pd.read_csv(pcache), on="id", validate="one_to_one")
    d = d.merge(asr_all[["id", "hyp_text"]], on="id", validate="one_to_one")
    d["speed"] = d.hyp_text.astype(str).str.split().str.len() / d.duration
    z = d.groupby("speaker")[["pitch", "volume", "speed"]].transform(lambda s: (s - s.mean()) / s.std())
    level = lambda v, lo, hi: lo if v < -0.5 else hi if v > 0.5 else "normal"
    return {u: f"pitch {level(p, 'low', 'high')}, volume {level(v, 'low', 'high')}, speed {level(s, 'slow', 'fast')}."
            for u, p, v, s in zip(d.id, z.pitch, z.volume, z.speed)}

def build_inputs(source, ctx, pros=False):
    pros_of = prosody_phrases() if pros else None
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
        cur = f"{pros_of[allu.id[r]]} {turns[r]}" if pros else turns[r]
        enc = tok(c, cur, truncation="only_first", max_length=MAX_LEN) if c else \
              tok(cur, truncation=True, max_length=MAX_LEN)
        ids.append(enc["input_ids"])
    return ids

INPUTS = {}
def get_inputs(source, ctx, pros=False):
    key = (source, ctx, pros)
    if key not in INPUTS:
        INPUTS[key] = build_inputs(source, ctx, pros)
        lens = np.array([len(x) for x in INPUTS[key]])
        print(f"inputs {source} ctx={ctx} pros={pros}: mean len {lens.mean():.0f}, max {lens.max()}, "
              f"truncated {(lens >= MAX_LEN).mean() * 100:.1f}%")
    return INPUTS[key]

print(tok.decode(get_inputs("asr", CTX if any("ctx" in r for r in RUNS) else 0)[idx["test"][5]]))

# %% [markdown]
# ## 3. Model và vòng train

# %%
CONFIGS = {
    "gt2asr":      dict(train="gt", eval="asr", ctx=0, audio=False),
    "asr":         dict(train="asr", eval="asr", ctx=0, audio=False),
    "asr_ctx":     dict(train="asr", eval="asr", ctx=CTX, audio=False),
    "asr_e2v":     dict(train="asr", eval="asr", ctx=0, audio=True),
    "asr_ctx_e2v": dict(train="asr", eval="asr", ctx=CTX, audio=True),
    "gt_ctx":      dict(train="gt", eval="gt", ctx=CTX, audio=False),
    "gt":          dict(train="gt", eval="gt", ctx=0, audio=False),
    "asr_ctx_pros":     dict(train="asr", eval="asr", ctx=CTX, audio=False, pros=True),
    "asr_ctx_pros_e2v": dict(train="asr", eval="asr", ctx=CTX, audio=True, pros=True),
}

class TextAudio(nn.Module):
    def __init__(self, n_classes, audio):
        super().__init__()
        self.enc = HFAutoModel.from_pretrained(TEXT_MODEL)
        d = self.enc.config.hidden_size
        self.audio = audio
        if audio:
            self.t_norm = nn.LayerNorm(d)
            if audio_all.dim() == 3:
                self.layer_w = nn.Parameter(torch.zeros(audio_all.shape[1] if POOL == "mean" else len(FRAME_LAYERS)))
            if POOL == "attn":
                self.att = nn.Sequential(nn.Linear(audio_all.shape[-1], 128), nn.Tanh(), nn.Linear(128, 1))
            self.a_proj = nn.Sequential(nn.LayerNorm(audio_all.shape[-1]), nn.Dropout(0.2),
                                        nn.Linear(audio_all.shape[-1], d), nn.Tanh())
            self.gate = nn.Linear(2 * d, 2)
            self.head_t = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, n_classes))
            self.head_a = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, n_classes))
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, n_classes))

    def forward(self, ids, mask, a=None):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        if not self.audio:
            return self.head(h), None, ()
        h = self.t_norm(h)
        if isinstance(a, tuple):
            a, valid = a
            a = (F.softmax(self.layer_w, 0)[None, None, :, None] * a).sum(2)
            if POOL == "attn":
                w = F.softmax(self.att(a).squeeze(-1).float().masked_fill(~valid, float("-inf")), -1)
            else:
                w = valid.float() / valid.sum(1, keepdim=True)
            a = (w.unsqueeze(-1) * a).sum(1)
        elif a.dim() == 3:
            a = (F.softmax(self.layer_w, 0)[None, :, None] * a).sum(1)
        ha = self.a_proj(a)
        g = F.softmax(self.gate(torch.cat([h, ha], -1)).float(), -1)
        fused = g[:, :1] * h + g[:, 1:] * ha
        return self.head(fused), g, (self.head_t(h), self.head_a(ha))

def audio_batch(b):
    if POOL == "mean":
        return audio_all[b].cuda()
    spans = [fspan[r] for r in b]
    a = torch.zeros(len(b), max(e - s for s, e in spans), *frames.shape[1:])
    valid = torch.zeros(a.shape[:2], dtype=torch.bool)
    for i, (s, e) in enumerate(spans):
        a[i, :e - s] = frames[s:e].float()
        valid[i, :e - s] = True
    return a.cuda(), valid.cuda()

def batches(rows, inputs, bs):
    for s in range(0, len(rows), bs):
        b = rows[s:s + bs]
        L = max(len(inputs[r]) for r in b)
        ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(b), L), dtype=torch.long)
        for i, r in enumerate(b):
            ids[i, :len(inputs[r])] = torch.tensor(inputs[r])
            mask[i, :len(inputs[r])] = 1
        yield b, ids.cuda(), mask.cuda(), audio_batch(b)

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
    pros = cfg.get("pros", False)
    tr_in, ev_in = get_inputs(cfg["train"], cfg["ctx"], pros), get_inputs(cfg["eval"], cfg["ctx"], pros)
    model = TextAudio(len(labels), cfg["audio"]).cuda()
    enc_p = list(model.enc.parameters())
    other = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": LR}, {"params": other, "lr": HEAD_LR}], weight_decay=0.01)
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
                print(f"  {name} s{seed} ep{ep} step {step} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        dev_p, _ = predict(model, idx["dev"].tolist(), ev_in)
        test_p, gate = predict(model, idx["test"].tolist(), ev_in)
        dua = ua_of(idx["dev"], dev_p)
        hist.append({"epoch": ep, "dev_ua": dua, "test_ua": ua_of(idx["test"], test_p), "sec": time.time() - t0})
        print(f"  {name} s{seed} ep{ep} dev UA {dua * 100:.2f} | {time.time() - t0:.0f}s", flush=True)
        # Test probabilities are kept only for the epoch chosen on dev; test UA per epoch is logged, never used to pick.
        if best is None or dua > best["dev_ua"]:
            best = {"dev_ua": dua, "epoch": ep, "dev_prob": dev_p, "test_prob": test_p, "gate": gate}
    del model, opt; torch.cuda.empty_cache()
    return best, hist

# %% [markdown]
# ## 4. Chạy: vòng ngoài là seed, vòng trong là cấu hình
#
# Xong seed 0 là đã có số của mọi cấu hình. Mỗi lần chạy lưu ra một file riêng, nên chạy lại notebook sẽ bỏ qua
# các lần đã xong (khi VM bị thu hồi giữa chừng).

# %%
OUT = cache / "combined"
OUT.mkdir(parents=True, exist_ok=True)

def tag(name):
    audio = AUDIO if POOL == "mean" else f"{AUDIO}_{POOL}"
    return (name.replace("e2v", audio) if CONFIGS[name]["audio"] else name) + FOLD

for seed in SEEDS:
    for name in RUNS:
        f = OUT / f"{tag(name)}_s{seed}.npz"
        if f.exists():
            continue
        print(f"== {name} seed {seed} | {time.strftime('%H:%M:%S')}", flush=True)
        best, hist = train_one(name, seed)
        np.savez(f, dev_prob=best["dev_prob"], test_prob=best["test_prob"],
                 meta=json.dumps({"best_epoch": best["epoch"], "dev_ua": best["dev_ua"], "gate": best["gate"], "hist": hist}))
        print(f"== {name} s{seed} best ep{best['epoch']} dev UA {best['dev_ua'] * 100:.2f} "
              f"test UA {ua_of(idx['test'], best['test_prob']) * 100:.2f} gate(text, audio) {best['gate']}", flush=True)

# %% [markdown]
# ## 5. Tổng hợp: ensemble các seed, mean ± std, lưu vào results/

# %%
test = allu.iloc[idx["test"]].reset_index(drop=True)
summary = []
for name in RUNS:
    files = sorted(OUT.glob(f"{tag(name)}_s*.npz"))
    if not files:
        continue
    per_seed, probs = [], []
    for f in files:
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        m = ser_metrics(test.emotion.tolist(), [labels[i] for i in z["test_prob"].argmax(1)], labels)
        per_seed.append({"seed": int(re.search(r"_s(\d+)", f.name)[1]), "best_epoch": meta["best_epoch"],
                         "dev_ua": meta["dev_ua"], "test_ua": m["ua"], "test_wf1": m["weighted_f1"],
                         "gate": meta["gate"]})
        probs.append(z["test_prob"])
    ps = pd.DataFrame(per_seed)
    out = test.copy()
    out["hyp_emotion"] = [labels[i] for i in np.mean(probs, 0).argmax(1)]
    if CONFIGS[name]["eval"] == "gt":
        out = out.drop(columns=["hyp_text"])
    cfg = CONFIGS[name]
    extra = {"seeds": per_seed, "config": cfg,
             "ua_mean": ps.test_ua.mean(), "ua_std": ps.test_ua.std(),
             "wf1_mean": ps.test_wf1.mean(), "wf1_std": ps.test_wf1.std()}
    evaluate(f"combined_{tag(name)}", DATASET, out, extra=extra,
             notes=f"SenseVoice-FT ASR + {TEXT_MODEL} fine-tuned ({EPOCHS} ep, lr {LR}); train text {cfg['train']}, "
                   f"eval text {cfg['eval']}, ctx {cfg['ctx']}, audio {f'{AUDIO} {POOL}' if cfg['audio'] else 'none'}; "
                   f"{len(files)}-seed ensemble" + ("; GROUND-TRUTH text (upper bound)" if cfg["eval"] == "gt" else ""))
    summary.append({"system": tag(name), "seeds": len(files), "UA mean": ps.test_ua.mean() * 100,
                    "UA std": ps.test_ua.std() * 100, "wF1 mean": ps.test_wf1.mean() * 100,
                    "wF1 std": ps.test_wf1.std() * 100,
                    "gate audio": np.mean([g["gate"][1] for g in per_seed]) if cfg["audio"] else None})
print(pd.DataFrame(summary).round(2).to_string())

# %%
# Single archive so results survive even if the VM is reclaimed right after the job.
subprocess.run(f"tar -czf /content/combined_out.tgz -C {RESULTS_DIR.parent} {RESULTS_DIR.name}/{DATASET} "
               f"-C {CACHE_DIR.parent} {CACHE_DIR.name}/{DATASET}/combined", shell=True, check=True)
print("archive /content/combined_out.tgz")
