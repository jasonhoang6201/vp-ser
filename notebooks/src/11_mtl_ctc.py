# gpu: A100
# %% [markdown]
# # 11. Baseline multi-task: CTC (ASR) + emotion head (bansal-2023-joint)
#
# Một encoder wav2vec 2.0 dùng chung, gắn hai head:
# - **CTC head** chép lời (ASR),
# - **emotion head**: lấy tổng có trọng số các layer, pooling theo thời gian, rồi phân loại cảm xúc (SER).
#
# Loss: `L = α · L_CTC + (1 − α) · L_CE`. Chạy 3 cấu hình để trả lời câu hỏi chính của paper,
# "học chung có giúp cả hai task không":
#
# | MODE | α | Ý nghĩa |
# |---|---|---|
# | `asr` | 1.0 | chỉ ASR |
# | `ser` | 0.0 | chỉ SER (CTC head không được train, WER của nó cho thấy ASR bị quên bao nhiêu) |
# | `joint` | 0.5 | học chung |
#
# Paper không công bố code, nên notebook này tự dựng lại. Encoder mặc định là `wav2vec2-large-960h-lv60-self`
# (đã fine-tune ASR trên LibriSpeech), vì MELD chỉ có khoảng 9 giờ, train CTC từ đầu thì WER rất cao.
# Chế độ `joint` còn đo **gradient conflict**: cosine giữa gradient của hai loss trên các layer encoder dùng chung.
# Cosine âm nghĩa là hai task kéo encoder về hai hướng ngược nhau.

# %%
DATASET = "meld"
ENCODER = "facebook/wav2vec2-large-960h-lv60-self"
MODES = ["joint", "ser", "asr"]   # chạy lần lượt
ALPHA_JOINT = 0.5
EPOCHS = 8
LR_ENC, LR_HEAD = 2e-5, 1e-3
BATCH_SECONDS = 160               # tổng số giây audio mỗi batch (A100 40GB, bf16)
MAX_TRAIN_SEC = 15.0
CLASS_WEIGHT = True               # MELD lệch lớp nặng (neutral ~47%), không cân thì UA rất thấp
SEED = 42

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
import copy, math, random, re, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from eaasr.common import (CKPT_DIR, LABELS, asr_metrics, evaluate, get_splits, gpu_info,
                          load_wav, ser_metrics, set_seed, stage_dataset)

gpu_info()
labels = LABELS[DATASET]
lab2id = {l: i for i, l in enumerate(labels)}
man = stage_dataset(DATASET)
splits = get_splits(man, DATASET)
splits["train"] = splits["train"][splits["train"].duration <= MAX_TRAIN_SEC].reset_index(drop=True)
print({k: len(v) for k, v in splits.items()})
processor = Wav2Vec2Processor.from_pretrained(ENCODER)
vocab = processor.tokenizer.get_vocab()

# %%
def ctc_text(s):
    """Map transcript to the CTC vocabulary (uppercase letters, apostrophe, space)."""
    s = s.upper().replace("-", " ")
    s = re.sub(r"[^A-Z' ]", " ", s)
    return " ".join(s.split())

class SpeechSet(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return load_wav(r.path), lab2id[r.emotion], ctc_text(r.text), i

def length_batches(df, max_seconds, shuffle, seed=0):
    order = np.argsort(df.duration.values)
    batches, cur, tot = [], [], 0.0
    for i in order:
        d = df.duration.values[i]
        if cur and (tot + d > max_seconds or len(cur) >= 64):
            batches.append(cur); cur, tot = [], 0.0
        cur.append(int(i)); tot += d
    if cur:
        batches.append(cur)
    if shuffle:
        random.Random(seed).shuffle(batches)
    return batches

def collate(items):
    wavs, labs, texts, idx = zip(*items)
    feats = processor.feature_extractor(list(wavs), sampling_rate=16000, padding=True,
                                        return_attention_mask=True, return_tensors="pt")
    tok = processor.tokenizer(list(texts), padding=True, return_tensors="pt")
    ctc_labels = tok.input_ids.masked_fill(tok.attention_mask == 0, -100)
    return {"input_values": feats.input_values, "attention_mask": feats.attention_mask,
            "emo": torch.tensor(labs), "ctc_labels": ctc_labels, "idx": torch.tensor(idx)}

# %%
class MTLModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # LayerDrop skips encoder layers in train mode and their hidden states go missing, so the
        # learned layer weights would no longer line up with hidden_states.
        self.w2v = Wav2Vec2ForCTC.from_pretrained(ENCODER, ctc_loss_reduction="mean", ctc_zero_infinity=True,
                                                  layerdrop=0.0)
        self.w2v.freeze_feature_encoder()
        cfg = self.w2v.config
        self.layer_w = nn.Parameter(torch.zeros(cfg.num_hidden_layers + 1))
        self.emo_head = nn.Sequential(nn.Linear(cfg.hidden_size, 256), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(256, n_classes))

    def forward(self, input_values, attention_mask, ctc_labels=None):
        out = self.w2v(input_values, attention_mask=attention_mask if self.w2v.config.feat_extract_norm == "layer" else None,
                       labels=ctc_labels, output_hidden_states=True)
        # wav2vec2-large hidden states differ in scale by orders of magnitude across layers; without a
        # per-layer norm the weighted sum is dominated by a few outlier layers and the head collapsed
        # to a single class (dev UA 14 = 1/7) in the first run.
        hs = torch.stack([F.layer_norm(x.float(), x.shape[-1:]) for x in out.hidden_states], 0)
        h = (F.softmax(self.layer_w, 0)[:, None, None, None] * hs).sum(0)
        frames = self.w2v._get_feat_extract_output_lengths(attention_mask.sum(-1))
        mask = (torch.arange(h.size(1), device=h.device)[None] < frames[:, None]).unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return out.logits, out.loss, self.emo_head(pooled)

# %%
@torch.no_grad()
def predict(model, df):
    model.eval()
    ds = SpeechSet(df)
    loader = DataLoader(ds, batch_sampler=length_batches(df, BATCH_SECONDS, False), collate_fn=collate, num_workers=4)
    hyp_text, hyp_emo = [None] * len(df), [None] * len(df)
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, emo_logits = model(b["input_values"].cuda(), b["attention_mask"].cuda())
        ids = logits.float().argmax(-1).cpu()
        texts = processor.batch_decode(ids)
        for k, i in enumerate(b["idx"].tolist()):
            hyp_text[i] = texts[k].lower()
            hyp_emo[i] = labels[int(emo_logits[k].argmax())]
    out = df.copy()
    out["hyp_text"], out["hyp_emotion"] = hyp_text, hyp_emo
    return out

def dev_score(mode, pred):
    ser = ser_metrics(pred.emotion.tolist(), pred.hyp_emotion.tolist(), labels)
    wer = asr_metrics(pred.text.tolist(), pred.hyp_text.tolist())["wer"]
    score = {"asr": -wer, "ser": ser["ua"], "joint": ser["ua"] - wer}[mode]
    return score, ser["ua"], ser["weighted_f1"], wer

def grad_cosine(model, b, ce_w):
    """Cosine between CTC and CE gradients on the shared transformer layers."""
    shared = [p for n, p in model.w2v.wav2vec2.encoder.named_parameters() if p.requires_grad]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, ctc_loss, emo_logits = model(b["input_values"].cuda(), b["attention_mask"].cuda(), b["ctc_labels"].cuda())
        ce = F.cross_entropy(emo_logits.float(), b["emo"].cuda(), weight=ce_w)
    g1 = torch.autograd.grad(ctc_loss, shared, retain_graph=True, allow_unused=True)
    g2 = torch.autograd.grad(ce, shared, allow_unused=True)
    # A parameter can be unused by one loss (e.g. a final norm only on the CTC path); zero-fill so
    # both vectors stay aligned parameter by parameter.
    v1 = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten().float() for g, p in zip(g1, shared)])
    v2 = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten().float() for g, p in zip(g2, shared)])
    return F.cosine_similarity(v1, v2, dim=0).item()

# %%
def train_mode(mode):
    set_seed(SEED)
    alpha = {"asr": 1.0, "ser": 0.0, "joint": ALPHA_JOINT}[mode]
    model = MTLModel(len(labels)).cuda()
    head_params = list(model.emo_head.parameters()) + [model.layer_w]
    enc_params = [p for n, p in model.w2v.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": LR_ENC}, {"params": head_params, "lr": LR_HEAD}],
                            weight_decay=0.01)
    tr = splits["train"]
    counts = tr.emotion.map(lab2id).value_counts().reindex(range(len(labels)), fill_value=1).values
    ce_w = torch.tensor(counts.sum() / (len(labels) * counts), dtype=torch.float).cuda() if CLASS_WEIGHT else None
    steps_per_epoch = len(length_batches(tr, BATCH_SECONDS, True))
    total = EPOCHS * steps_per_epoch
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else max(0.0, (total - s) / (total - warm)))
    best, best_state, log, cos_log, step = -1e9, None, [], [], 0
    for ep in range(EPOCHS):
        model.train()
        loader = DataLoader(SpeechSet(tr), batch_sampler=length_batches(tr, BATCH_SECONDS, True, seed=SEED + ep),
                            collate_fn=collate, num_workers=4)
        pbar = tqdm(loader, desc=f"{mode} ep{ep + 1}")
        for b in pbar:
            if mode == "joint" and step % 200 == 0:
                cos_log.append((step, grad_cosine(model, b, ce_w)))
                opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, ctc_loss, emo_logits = model(b["input_values"].cuda(), b["attention_mask"].cuda(),
                                                b["ctc_labels"].cuda() if alpha > 0 else None)
                ce = F.cross_entropy(emo_logits.float(), b["emo"].cuda(), weight=ce_w)
            loss = (alpha * ctc_loss if alpha > 0 else 0) + (1 - alpha) * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); step += 1
            pbar.set_postfix(ctc=float(ctc_loss) if alpha > 0 else 0, ce=float(ce))
            # tqdm does not refresh in a detached job's log, so print the losses explicitly.
            if step % 100 == 0:
                print(f"step {step} ctc {float(ctc_loss) if alpha > 0 else 0:.3f} ce {float(ce):.3f}", flush=True)
        dev = predict(model, splits["dev"])
        score, ua, wf1, wer = dev_score(mode, dev)
        log.append({"epoch": ep + 1, "dev_UA": ua * 100, "dev_wF1": wf1 * 100, "dev_WER": wer * 100})
        print(log[-1])
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, pd.DataFrame(log), cos_log

# %%
all_logs = {}
for mode in MODES:
    model, log, cos_log = train_mode(mode)
    all_logs[mode] = (log, cos_log)
    test = predict(model, splits["test"])
    extra = {"dev_log": log.to_dict("records")}
    if cos_log:
        extra["grad_cosine"] = cos_log
    alpha = {"asr": 1.0, "ser": 0.0, "joint": ALPHA_JOINT}[mode]
    # In "ser" mode the CTC head gets no loss but the encoder still moves, so its WER shows forgetting.
    if mode == "asr":
        test = test.drop(columns=["hyp_emotion"])
    evaluate(f"mtl_ctc_{mode}", DATASET, test,
             notes=f"{ENCODER.split('/')[-1]}, alpha={alpha}, {EPOCHS} ep, "
                   f"class_weight={CLASS_WEIGHT}, best epoch by dev", extra=extra)
    del model; torch.cuda.empty_cache()

# %% [markdown]
# ## Gradient conflict trong chế độ joint
# Mỗi 200 bước, đo cosine giữa gradient của CTC loss và CE loss trên các layer encoder dùng chung.
# - Gần 0: hai task gần như độc lập.
# - Âm: hai task xung đột (gradient conflict), là lý do để thử PCGrad hoặc train hai giai đoạn trong model kết hợp.

# %%
if "joint" in all_logs and all_logs["joint"][1]:
    cl = pd.DataFrame(all_logs["joint"][1], columns=["step", "cosine"])
    print(cl.describe())
    print("tỉ lệ bước có cosine < 0:", (cl.cosine < 0).mean())
    display(cl)
