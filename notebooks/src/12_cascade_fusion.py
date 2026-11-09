# gpu: A100
# %% [markdown]
# # 12. Baseline cascade: Whisper + emotion2vec + text, fusion (li-2024-werfusion)
#
# Hệ thống ghép nối tiếp (cascade): mỗi khối là một model đã train sẵn.
#
# 1. **Whisper** chép lời, cho ra transcript và WER.
# 2. **emotion2vec+ large** trích embedding cảm xúc từ âm thanh (thử luôn zero-shot SER của nó).
# 3. **RoBERTa-large** trích embedding từ transcript của Whisper.
# 4. Train một **classifier nhỏ** trên embedding: chỉ audio, chỉ text, ghép (concat), và **gated fusion**
#    (model tự học trọng số cho từng modality ở mỗi câu, ý tưởng modality-gated fusion của li-2024).
#
# Câu hỏi paper đặt ra: transcript có lỗi (WER khoảng 10–20%) còn giúp SER không? Vì vậy chạy thêm
# `text_gt`: text đúng (ground truth). Đây là **cận trên**, không phải hệ thống dùng được thật, vì lúc dùng
# thật không có text đúng.
#
# Mỗi classifier train với 5 seed, báo mean ± std để biết chênh lệch có lớn hơn dao động hay không.

# %%
DATASET = "meld"
WHISPER = "openai/whisper-large-v3-turbo"
TEXT_MODEL = "roberta-large"
E2V = "emotion2vec/emotion2vec_plus_large"
SEEDS = [0, 1, 2, 3, 4]
# "wavlm": dùng embedding WavLM-large mọi layer do notebook 21 tạo (cache/<dataset>/wavlm_large_layers.npz) thay cho
# emotion2vec+, chỉ chạy các hệ thống có audio (text probe không đổi). Cần cho IEMOCAP vì emotion2vec+ đã thấy IEMOCAP.
AUDIO = "e2v"
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
!pip -q install -U transformers funasr whisper-normalizer soundfile

# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from eaasr.common import (CACHE_DIR, LABELS, canon_emotion, evaluate, get_splits, gpu_info, load_wav,
                          ser_metrics, set_seed, stage_dataset)

gpu_info()
labels = LABELS[DATASET]
lab2id = {l: i for i, l in enumerate(labels)}
man = stage_dataset(DATASET)
splits = get_splits(man, DATASET, test_session=TEST_SESSION)
allu = pd.concat([splits[s].assign(_split=s) for s in ["train", "dev", "test"]], ignore_index=True)
cache = CACHE_DIR / DATASET
cache.mkdir(parents=True, exist_ok=True)
print({k: len(v) for k, v in splits.items()})

# %% [markdown]
# ## 1. Whisper ASR (lưu cache lên Drive, chỉ chạy một lần)

# %%
from transformers import pipeline

wcache = cache / f"asr_{WHISPER.split('/')[-1]}.csv"
if wcache.exists():
    asr_df = pd.read_csv(wcache, keep_default_na=False)
else:
    asr = pipeline("automatic-speech-recognition", model=WHISPER, torch_dtype=torch.float16, device="cuda:0")
    hyps = []
    paths = allu.path.tolist()
    for s in tqdm(range(0, len(paths), 256)):
        batch = [{"raw": load_wav(p), "sampling_rate": 16000} for p in paths[s:s + 256]]
        out = asr(batch, batch_size=32, chunk_length_s=30,
                  generate_kwargs={"language": "en", "task": "transcribe"})
        hyps += [o["text"].strip() for o in out]
    asr_df = pd.DataFrame({"id": allu.id, "asr": hyps})
    asr_df.to_csv(wcache, index=False)
    del asr; torch.cuda.empty_cache()
allu = allu.merge(asr_df, on="id", how="left")
allu["asr"] = allu["asr"].fillna("")

test = allu[allu._split == "test"].copy()
test["hyp_text"] = test["asr"]
evaluate(f"whisper_{WHISPER.split('/')[-1]}{FOLD}", DATASET, test.drop(columns=["hyp_emotion"], errors="ignore"),
         notes="ASR only, no SER")

# %% [markdown]
# ## 2. emotion2vec+: embedding và zero-shot SER

# %%
from funasr import AutoModel

ecache = cache / "e2v_plus_large.npz"
if AUDIO == "wavlm":
    z = np.load(cache / "wavlm_large_layers.npz", allow_pickle=True)
    pos = {u: i for i, u in enumerate(z["ids"])}
    e2v_emb = z["emb"][[pos[u] for u in allu.id]].astype(np.float32)
    print("wavlm emb:", e2v_emb.shape)
elif FOLD:
    # This cache stores no ids and follows the row order of the session-5 split.
    raise ValueError("e2v cache is in session-5 row order; use AUDIO='wavlm' for other folds")
elif ecache.exists():
    z = np.load(ecache, allow_pickle=True)
    e2v_emb, e2v_scores, e2v_names = z["emb"], z["scores"], list(z["names"])
else:
    e2v = AutoModel(model=E2V, hub="hf", disable_update=True)
    embs, scores = [], []
    for p in tqdm(allu.path):
        r = e2v.generate(p, granularity="utterance", extract_embedding=True)[0]
        embs.append(np.asarray(r["feats"], dtype=np.float32).reshape(-1))
        scores.append(np.asarray(r["scores"], dtype=np.float32))
        e2v_names = r["labels"]
    e2v_emb, e2v_scores = np.stack(embs), np.stack(scores)
    np.savez(ecache, emb=e2v_emb, scores=e2v_scores, names=np.array(e2v_names))
    del e2v; torch.cuda.empty_cache()

# %%
# Zero-shot: chỉ giữ điểm của các lớp có trong dataset rồi argmax (không train gì).
if AUDIO == "e2v":
    print("emotion2vec classes:", e2v_names, "| emb dim:", e2v_emb.shape)
    cols = {canon_emotion(n): i for i, n in enumerate(e2v_names) if canon_emotion(n) in labels}
    print("map:", cols)
    order = [l for l in labels if l in cols]
    zs = test.copy()
    sub = e2v_scores[zs.index][:, [cols[l] for l in order]]
    zs["hyp_emotion"] = [order[i] for i in sub.argmax(1)]
    evaluate("emotion2vec_plus_large_zeroshot", DATASET, zs.drop(columns=["hyp_text"]),
             notes="restricted argmax over dataset classes, no training")

# %% [markdown]
# ## 3. Embedding text bằng RoBERTa-large (cho transcript của Whisper và text đúng)

# %%
from transformers import AutoModel as HFAutoModel, AutoTokenizer

def text_embed(texts, name):
    f = cache / f"text_{TEXT_MODEL.split('/')[-1]}_{name}{FOLD}.npy"
    if f.exists():
        return np.load(f)
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    mdl = HFAutoModel.from_pretrained(TEXT_MODEL, torch_dtype=torch.float16).cuda().eval()
    out = []
    with torch.no_grad():
        for s in tqdm(range(0, len(texts), 128), desc=name):
            b = tok([t if t.strip() else "." for t in texts[s:s + 128]], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to("cuda")
            h = mdl(**b).last_hidden_state.float()
            m = b.attention_mask.unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    arr = np.concatenate(out)
    np.save(f, arr)
    return arr

txt_asr = text_embed(allu.asr.tolist(), "asr")
txt_gt = text_embed(allu.text.tolist(), "gt")

# %% [markdown]
# ## 4. Classifier: chỉ audio, chỉ text, concat, gated fusion

# %%
class FusionHead(nn.Module):
    def __init__(self, dims, n_classes, mode, hid=256):
        super().__init__()
        self.mode = mode
        # A (layers, dim) input is a stack of encoder layers, mixed by learned weights as in SUPERB.
        self.layer_w = nn.ParameterList([nn.Parameter(torch.zeros(d[0])) for d in dims if len(d) == 2])
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(d[-1]), nn.Dropout(0.2), nn.Linear(d[-1], hid), nn.Tanh())
                                   for d in dims])
        if mode == "gated":
            self.gate = nn.Linear(hid * len(dims), len(dims))
        k = hid * len(dims) if mode == "concat" else hid
        self.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(k, hid), nn.ReLU(), nn.Linear(hid, n_classes))

    def forward(self, xs):
        lw = iter(self.layer_w)
        xs = [(F.softmax(next(lw), 0)[None, :, None] * x).sum(1) if x.dim() == 3 else x for x in xs]
        hs = [p(x) for p, x in zip(self.proj, xs)]
        if self.mode == "gated":
            g = F.softmax(self.gate(torch.cat(hs, -1)), -1)
            h = sum(g[:, i:i + 1] * hs[i] for i in range(len(hs)))
        elif self.mode == "concat":
            h = torch.cat(hs, -1)
        else:
            h = hs[0]
        return self.out(h)

y_all = allu.emotion.map(lab2id).values
idx = {s: np.where(allu._split.values == s)[0] for s in ["train", "dev", "test"]}

def fit_predict(feats, mode, seed, epochs=60, bs=64):
    set_seed(seed)
    X = [torch.tensor(f, dtype=torch.float32).cuda() for f in feats]
    y = torch.tensor(y_all).cuda()
    tr = idx["train"]
    counts = np.bincount(y_all[tr], minlength=len(labels)).clip(min=1)
    w = torch.tensor(counts.sum() / (len(labels) * counts), dtype=torch.float32).cuda()
    model = FusionHead([f.shape[1:] for f in feats], len(labels), mode).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state = -1, None
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        for s in range(0, len(perm), bs):
            b = torch.tensor(perm[s:s + bs]).cuda()
            loss = F.cross_entropy(model([x[b] for x in X]), y[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            dv = model([x[idx["dev"]] for x in X]).argmax(-1).cpu().numpy()
        ua = ser_metrics([labels[i] for i in y_all[idx["dev"]]], [labels[i] for i in dv], labels)["ua"]
        if ua > best:
            best, best_state = ua, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = F.softmax(model([x[idx["test"]] for x in X]), -1).cpu().numpy()
    return prob, best

SYSTEMS = {
    "e2v_probe":        ([e2v_emb], "single", False),
    "text_asr_probe":   ([txt_asr], "single", True),
    "text_gt_probe":    ([txt_gt], "single", False),
    "fusion_concat":    ([e2v_emb, txt_asr], "concat", True),
    "fusion_gated":     ([e2v_emb, txt_asr], "gated", True),
    "fusion_gated_gt":  ([e2v_emb, txt_gt], "gated", False),
}
if AUDIO == "wavlm":
    # Text-only probes do not depend on the audio encoder, so in WavLM mode run them only for new folds.
    uses_audio = lambda feats: any(f is e2v_emb for f in feats)
    SYSTEMS = {(k.replace("e2v", "wavlm") if "e2v" in k else k + "_wavlm") if uses_audio(v[0]) else k: v
               for k, v in SYSTEMS.items() if FOLD or uses_audio(v[0])}
SYSTEMS = {k + FOLD: v for k, v in SYSTEMS.items()}
summary = []
for name, (feats, mode, uses_asr) in SYSTEMS.items():
    probs, per_seed = [], []
    for seed in SEEDS:
        p, dev_ua = fit_predict(feats, mode, seed)
        probs.append(p)
        m = ser_metrics(test.emotion.tolist(), [labels[i] for i in p.argmax(1)], labels)
        per_seed.append({"seed": seed, "dev_ua": dev_ua, "test_ua": m["ua"], "test_wf1": m["weighted_f1"]})
    ps = pd.DataFrame(per_seed)
    out = test.copy()
    out["hyp_emotion"] = [labels[i] for i in np.mean(probs, 0).argmax(1)]
    if not uses_asr:
        out = out.drop(columns=["hyp_text"])
    extra = {"seeds": per_seed,
             "ua_mean": ps.test_ua.mean(), "ua_std": ps.test_ua.std(),
             "wf1_mean": ps.test_wf1.mean(), "wf1_std": ps.test_wf1.std()}
    evaluate(f"cascade_{name}", DATASET, out, extra=extra,
             notes=f"{mode}; predictions = {len(SEEDS)}-seed ensemble"
                   + ("; uses GROUND-TRUTH text (upper bound)" if "gt" in name else ""))
    summary.append({"system": name, "UA mean": ps.test_ua.mean() * 100, "UA std": ps.test_ua.std() * 100,
                    "wF1 mean": ps.test_wf1.mean() * 100, "wF1 std": ps.test_wf1.std() * 100})
pd.DataFrame(summary).round(2)
