# gpu: L4
# %% [markdown]
# # 30. Demo hệ đề xuất trên clip ngoài IEMOCAP
#
# Tải weight đã train (fold S5: train trên session 1-4 của IEMOCAP) từ repo HF private, rồi chạy toàn bộ hệ trên
# các clip trong `demo/clips` (nguồn và lý do chọn ở `demo/SOURCES.md`):
#
# 1. **ASR**: Whisper large-v3-turbo fine-tune, giải mã giống notebook 15.
# 2. **Nhánh text**: RoBERTa-large nhận (ngữ cảnh, câu hiện tại).
# 3. **Nhánh audio**: encoder Whisper large-v3-turbo đông cứng, pool mọi layer giống notebook 21.
# 4. **Gated fusion**: in ra trọng số gate text/audio và dự đoán riêng của từng nhánh, để giải thích các câu sai.
#
# Cần GPU (L4 hoặc T4) và token HF có quyền đọc repo.

# %%
HUB_REPO = "master-122025/eaasr-iemocap"
FOLD_DIR = "fold_s5"
CLIP_SET = "clips"    # thư mục trong demo/, chứa <CLIP_SET>.csv (file, gold, source); "candidates" là bộ ứng viên
ENCODER = "openai/whisper-large-v3-turbo"

# %%
!pip -q install -U transformers huggingface_hub safetensors soundfile librosa

# %%
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import get_token, snapshot_download
from IPython.display import Audio, display
from safetensors.torch import load_file
from transformers import (AutoConfig, AutoFeatureExtractor, AutoModel, AutoTokenizer, WhisperForConditionalGeneration,
                          WhisperModel, WhisperProcessor, pipeline)

if get_token() is None:
    from huggingface_hub import notebook_login
    notebook_login()
local = Path(snapshot_download(HUB_REPO, allow_patterns=[f"{FOLD_DIR}/*", "demo/*"]))
cfg = json.load(open(local / FOLD_DIR / "ser" / "config.json"))
labels = cfg["labels"]
print({k: cfg[k] for k in ["config", "asr", "text_model", "audio_encoder", "context_turns", "best_epoch", "dev_ua"]})
print("test UA của fold này trên IEMOCAP:",
      round(json.load(open(local / FOLD_DIR / "ser" / "metrics_test.json"))["ser"]["ua"] * 100, 2))

# %% [markdown]
# ## 1. ASR: Whisper turbo fine-tune

# %%
asr_dir = local / FOLD_DIR / "asr"
proc = WhisperProcessor.from_pretrained(asr_dir)
asr_model = WhisperForConditionalGeneration.from_pretrained(asr_dir, torch_dtype=torch.float16).cuda().eval()
asr = pipeline("automatic-speech-recognition", model=asr_model, tokenizer=proc.tokenizer,
               feature_extractor=proc.feature_extractor, torch_dtype=torch.float16, device="cuda:0")

def load_wav(p):
    return librosa.load(p, sr=16000, mono=True)[0]

def transcribe(wavs):
    out = asr([{"raw": w, "sampling_rate": 16000} for w in wavs], batch_size=8, chunk_length_s=30,
              generate_kwargs={"language": "en", "task": "transcribe"})
    return [o["text"].strip() for o in out]

# %% [markdown]
# ## 2. Feature audio: encoder Whisper turbo đông cứng
#
# Phải dựng đúng như lúc train (notebook 21): 20 s đầu, mọi hidden state của encoder, layer norm từng frame, trung
# bình trên các frame thật (bỏ phần pad tới 30 s).

# %%
fe = AutoFeatureExtractor.from_pretrained(ENCODER)
enc = WhisperModel.from_pretrained(ENCODER, torch_dtype=torch.float16).encoder.cuda().eval()

@torch.no_grad()
def audio_features(wavs):
    wavs = [w[: 16000 * 20] for w in wavs]
    x = fe(wavs, sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
    hs = torch.stack(enc(x, output_hidden_states=True).hidden_states, 1)
    embs = []
    for j, w in enumerate(wavs):
        n = max(1, -(-len(w) // 320))
        embs.append(F.layer_norm(hs[j, :, :n].float(), hs.shape[-1:]).mean(1).half())
    return torch.stack(embs).float()

# %% [markdown]
# ## 3. Model SER: RoBERTa-large + nhánh audio + gated fusion
#
# Cùng kiến trúc `TextAudio` của notebook 21 (cấu hình đầy đủ: có text, có audio).

# %%
class TextAudio(nn.Module):
    def __init__(self, text_model, n_classes, n_layers, dim):
        super().__init__()
        d = AutoConfig.from_pretrained(text_model).hidden_size
        self.enc = AutoModel.from_config(AutoConfig.from_pretrained(text_model))
        self.layer_w = nn.Parameter(torch.zeros(n_layers))
        self.a_proj = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.2), nn.Linear(dim, d), nn.Tanh())
        self.t_norm = nn.LayerNorm(d)
        self.gate = nn.Linear(2 * d, 2)
        self.head_t = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, n_classes))
        self.head_a = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, n_classes))
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d, n_classes))

    def forward(self, ids, mask, a):
        a = (F.softmax(self.layer_w, 0)[None, :, None] * a).sum(1)
        ha = self.a_proj(a)
        h = self.t_norm(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])
        g = F.softmax(self.gate(torch.cat([h, ha], -1)).float(), -1)
        fused = g[:, :1] * h + g[:, 1:] * ha
        return self.head(fused), g, self.head_t(h), self.head_a(ha)

ser_dir = local / FOLD_DIR / "ser"
tok = AutoTokenizer.from_pretrained(ser_dir)
tok.truncation_side = "left"
n_layers, dim = cfg["audio_feature_shape"]
ser = TextAudio(cfg["text_model"], len(labels), n_layers, dim)
ser.load_state_dict(load_file(str(ser_dir / "model.safetensors")), strict=True)
ser = ser.cuda().eval()

def encode(turn, context):
    c = " ".join(context[-cfg["context_turns"]:])
    max_len = cfg["max_len"]
    if c and len(tok(turn, add_special_tokens=False)["input_ids"]) > max_len - 8:
        c = ""
    return (tok(c, turn, truncation="only_first", max_length=max_len) if c else
            tok(turn, truncation=True, max_length=max_len))["input_ids"]

@torch.no_grad()
def predict(turns, contexts, feats):
    ids = [encode(t, c) for t, c in zip(turns, contexts)]
    L = max(map(len, ids))
    x = torch.full((len(ids), L), tok.pad_token_id, dtype=torch.long)
    m = torch.zeros_like(x)
    for i, s in enumerate(ids):
        x[i, :len(s)], m[i, :len(s)] = torch.tensor(s), 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, g, lt, la = ser(x.cuda(), m.cuda(), feats.cuda())
    p = lambda z: F.softmax(z.float(), -1).cpu().numpy()
    return p(logits), g.cpu().numpy(), p(lt), p(la)

# %% [markdown]
# ## 4. Chạy trên các clip
#
# Mỗi clip đứng riêng, không có câu trước, nên ngữ cảnh rỗng. Cột `text` và `audio` là dự đoán của riêng từng nhánh
# (head phụ lúc train); `gate audio` là phần trọng số model dành cho audio ở câu đó.

# %%
clip_dir = local / "demo" / CLIP_SET
CLIPS = pd.read_csv(clip_dir / f"{CLIP_SET}.csv")
paths = [clip_dir / f for f in CLIPS.file]
wavs = [load_wav(p) for p in paths]
hyps = transcribe(wavs)
# Batches of 16 keep the encoder's 30 s padded input within GPU memory for larger clip sets.
feats = torch.cat([audio_features(wavs[s:s + 16]) for s in range(0, len(wavs), 16)])
prob, gate, pt, pa = predict(hyps, [[] for _ in hyps], feats)

rows = []
for i, (f, gold) in enumerate(zip(CLIPS.file, CLIPS.gold)):
    rows.append({"clip": f[:-4], "gold": gold, "pred": labels[prob[i].argmax()],
                 **{l: round(float(prob[i, k]), 2) for k, l in enumerate(labels)},
                 "text": labels[pt[i].argmax()], "audio": labels[pa[i].argmax()],
                 "gate audio": round(float(gate[i, 1]), 2), "transcript": hyps[i]})
res = pd.DataFrame(rows)
print(f"đúng {(res.gold == res.pred).sum()}/{len(res)}")
print(pd.crosstab(res.gold, res.pred, rownames=["gold"], colnames=["pred"]))
pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 60)
display(res)

# %%
for p, h in zip(paths, hyps):
    print(p.name, "|", h)
    display(Audio(str(p)))

# %% [markdown]
# ## 5. Ngữ cảnh đổi dự đoán thế nào
#
# Cùng clip 06 (lời tích cực, giọng buồn), thêm câu trước vào ngữ cảnh. Trong hệ thật, ngữ cảnh là transcript ASR
# của 3 câu trước trong hội thoại; ở đây gõ tay để thấy rõ tác động.

# %%
k = list(CLIPS.file).index("06_sadvoice_amazing_book.wav")
contexts = {
    "không ngữ cảnh": [],
    "ngữ cảnh buồn": ["I finished it the night before the funeral.", "She gave me that book before she passed away."],
    "ngữ cảnh vui": ["I finally finished it this weekend!", "Honestly, it was the best thing I've read all year."],
}
p, g, _, _ = predict([hyps[k]] * len(contexts), list(contexts.values()), feats[[k] * len(contexts)])
display(pd.DataFrame([{"ngữ cảnh": n, "pred": labels[p[i].argmax()],
                       **{l: round(float(p[i, j]), 2) for j, l in enumerate(labels)},
                       "gate audio": round(float(g[i, 1]), 2)} for i, n in enumerate(contexts)]))
