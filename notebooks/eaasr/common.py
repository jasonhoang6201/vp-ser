"""Shared helpers for the emotion-aware ASR Colab notebooks.

Every notebook evaluates through `evaluate()` so that WER normalization, label
mapping and metric definitions are identical across systems.
"""
import json
import os
import random
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

DRIVE_ROOT = Path(os.environ.get("EAASR_DRIVE", "/content/drive/MyDrive/vp-ck"))
LOCAL_ROOT = Path(os.environ.get("EAASR_LOCAL", "/content/eaasr_data"))
DATA_DIR = DRIVE_ROOT / "data"
RESULTS_DIR = DRIVE_ROOT / "results"
CKPT_DIR = DRIVE_ROOT / "checkpoints"
CACHE_DIR = DRIVE_ROOT / "cache"

SR = 16000

# IEMOCAP uses the common 4-class protocol (excited merged into happy), expressed
# with MELD-style names so that one label space covers both corpora.
LABELS = {
    "meld": ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"],
    "iemocap": ["neutral", "joy", "anger", "sadness"],
}

_SYNONYMS = {
    "neutral": "neutral", "neu": "neutral",
    "joy": "joy", "joyful": "joy", "happy": "joy", "happiness": "joy", "hap": "joy",
    "excited": "joy", "excitement": "joy", "exc": "joy",
    "anger": "anger", "angry": "anger", "ang": "anger",
    "sadness": "sadness", "sad": "sadness",
    "surprise": "surprise", "surprised": "surprise", "sur": "surprise",
    "fear": "fear", "fearful": "fear", "scared": "fear", "afraid": "fear", "fea": "fear",
    "disgust": "disgust", "disgusted": "disgust", "dis": "disgust",
}


def canon_emotion(name):
    """Map a single label or tag (e.g. 'Joy', '<|HAPPY|>', '生气/angry') to a canonical name or None."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    s = str(name).strip().lower().split("/")[-1]
    return _SYNONYMS.get(re.sub(r"[^a-z]", "", s))


def emotion_from_text(text, labels):
    """First emotion word in free-form text (LLM answers) that belongs to `labels`."""
    for tok in re.findall(r"[a-z]+", str(text or "").lower()):
        c = _SYNONYMS.get(tok)
        if c in labels:
            return c
    return None


_CHAR_FIX = {
    "\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"', "\x85": "...", "\x96": "-", "\x97": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"', "…": "...",
    "–": "-", "—": "-", "\xa0": " ",
}


def clean_text(s):
    """Repair the cp1252/utf-8 mojibake found in MELD subtitles."""
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    if "â€" in s or "Â" in s:
        try:
            s = s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            s = s.replace("Â", "")
    for a, b in _CHAR_FIX.items():
        s = s.replace(a, b)
    return " ".join(s.split())


def _basic_normalize(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return " ".join(s.split())


_normalizer = None


def normalize_for_wer(s):
    """Whisper English normalizer when available, so hypotheses with digits/punctuation are not penalized."""
    global _normalizer
    if _normalizer is None:
        try:
            from whisper_normalizer.english import EnglishTextNormalizer
            _normalizer = EnglishTextNormalizer()
        except ImportError:
            print("[eaasr] whisper-normalizer not installed, using basic normalizer")
            _normalizer = _basic_normalize
    return " ".join(_normalizer(clean_text(s)).split())


def word_edits(ref, hyp):
    """Levenshtein distance between two word lists."""
    n = len(hyp)
    prev = list(range(n + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * n
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[n]


def asr_metrics(refs, hyps, groups=None):
    """Corpus WER, plus WER per group (e.g. per reference emotion). Empty references are skipped."""
    edits, nwords, keep_groups = [], [], []
    skipped = 0
    for i, (r, h) in enumerate(zip(refs, hyps)):
        rw = normalize_for_wer(r).split()
        if not rw:
            skipped += 1
            continue
        edits.append(word_edits(rw, normalize_for_wer(h).split()))
        nwords.append(len(rw))
        keep_groups.append(None if groups is None else groups[i])
    edits, nwords = np.array(edits), np.array(nwords)
    out = {"wer": float(edits.sum() / max(nwords.sum(), 1)), "n_utts": int(len(edits)),
           "n_ref_words": int(nwords.sum()), "n_empty_ref_skipped": skipped}
    if groups is not None:
        g = np.array([str(x) for x in keep_groups])
        out["wer_by_emotion"] = {str(k): float(edits[g == k].sum() / max(nwords[g == k].sum(), 1))
                                 for k in sorted(set(g))}
    return out


def ser_metrics(true, pred, labels):
    """WA = accuracy, UA = macro recall over `labels`; predictions outside `labels` count as errors."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
    t = [str(x) for x in true]
    p = [x if x in labels else "<invalid>" for x in pred]
    rec = recall_score(t, p, labels=labels, average=None, zero_division=0)
    return {
        "wa": float(accuracy_score(t, p)),
        "ua": float(rec.mean()),
        "macro_f1": float(f1_score(t, p, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(t, p, labels=labels, average="weighted", zero_division=0)),
        "recall_by_emotion": dict(zip(labels, map(float, rec))),
        "n": len(t),
        "n_invalid_pred": int(sum(x == "<invalid>" for x in p)),
        "labels": labels,
        "confusion": confusion_matrix(t, p, labels=labels).tolist(),
    }


def evaluate(name, dataset, df, split="test", notes="", extra=None):
    """Score a prediction table and store it under results/<dataset>/<name>/.

    `df` needs `id`, `text`, `emotion` (references) and at least one of
    `hyp_text`, `hyp_emotion`.
    """
    labels = LABELS[dataset]
    m = {"name": name, "dataset": dataset, "split": split, "notes": notes}
    if "hyp_text" in df:
        m["asr"] = asr_metrics(df["text"].tolist(), df["hyp_text"].fillna("").tolist(),
                               groups=df["emotion"].tolist())
    if "hyp_emotion" in df:
        m["ser"] = ser_metrics(df["emotion"].tolist(), df["hyp_emotion"].tolist(), labels)
    if extra:
        m.update(extra)
    out = RESULTS_DIR / dataset / name
    out.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["id", "text", "hyp_text", "emotion", "hyp_emotion"] if c in df]
    df[cols].to_csv(out / f"predictions_{split}.csv", index=False)
    with open(out / f"metrics_{split}.json", "w") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print_metrics(m)
    return m


def save_metrics(name, dataset, metrics, split="test"):
    """For systems that only expose aggregate numbers (no per-utterance predictions)."""
    out = RESULTS_DIR / dataset / name
    out.mkdir(parents=True, exist_ok=True)
    metrics = {"name": name, "dataset": dataset, "split": split, **metrics}
    with open(out / f"metrics_{split}.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print_metrics(metrics)
    return metrics


def print_metrics(m):
    print(f"== {m['name']} | {m['dataset']} / {m['split']}")
    if "asr" in m:
        a = m["asr"]
        print(f"   WER {a['wer'] * 100:.2f}%  ({a['n_utts']} utts)")
        if "wer_by_emotion" in a:
            print("   WER by emotion: " + ", ".join(f"{k} {v * 100:.1f}" for k, v in a["wer_by_emotion"].items()))
    if "ser" in m:
        s = m["ser"]
        print(f"   WA {s['wa'] * 100:.2f}  UA {s['ua'] * 100:.2f}  macro-F1 {s['macro_f1'] * 100:.2f}  "
              f"weighted-F1 {s['weighted_f1'] * 100:.2f}  (invalid preds: {s['n_invalid_pred']})")


def load_results(dataset, split="test"):
    rows = []
    for p in sorted((RESULTS_DIR / dataset).glob(f"*/metrics_{split}.json")):
        m = json.load(open(p))
        row = {"system": m["name"], "notes": m.get("notes", "")}
        if "asr" in m:
            row["WER"] = m["asr"]["wer"] * 100
        if "ser" in m:
            for k in ["wa", "ua", "macro_f1", "weighted_f1"]:
                row[k.upper() if k in ("wa", "ua") else k] = m["ser"][k] * 100
        for k in ("wer_last", "wa_last", "ua_last"):
            if k in m:
                row[k] = m[k]
        rows.append(row)
    return pd.DataFrame(rows)


def stage_dataset(dataset):
    """Copy the packed 16 kHz wav tar from Drive to local disk once per runtime; return the manifest."""
    marker = LOCAL_ROOT / dataset / ".ready"
    if not marker.exists():
        LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
        tar = DATA_DIR / f"{dataset}_16k.tar"
        print(f"[eaasr] extracting {tar} -> {LOCAL_ROOT}")
        subprocess.run(["tar", "-xf", str(tar), "-C", str(LOCAL_ROOT)], check=True)
        marker.touch()
    df = pd.read_csv(DATA_DIR / f"{dataset}_manifest.csv", keep_default_na=False)
    df["path"] = df["relpath"].map(lambda p: str(LOCAL_ROOT / p))
    df["emotion"] = df["emotion"].map(lambda x: x or None)
    return df


def get_splits(df, dataset, test_session=5, dev_frac=0.1, seed=42, labeled_only=True):
    """MELD: official train/dev/test. IEMOCAP: one held-out session as test, random dev from the rest."""
    if labeled_only:
        df = df[df["emotion"].isin(LABELS[dataset])]
    # Two MELD test clips (dia38_utt4, dia220_utt0) are minutes long but carry a few-word
    # subtitle: the video cut is wrong, and the extra speech would swamp corpus WER.
    df = df[df["duration"] <= 60]
    if dataset == "meld":
        return {s: df[df["split"] == s].reset_index(drop=True) for s in ["train", "dev", "test"]}
    test = df[df["session"] == test_session]
    rest = df[df["session"] != test_session]
    dev_mask = np.random.RandomState(seed).rand(len(rest)) < dev_frac
    return {"train": rest[~dev_mask].reset_index(drop=True),
            "dev": rest[dev_mask].reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def load_wav(path):
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert sr == SR, f"{path}: expected {SR} Hz, got {sr}"
    return wav


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        out = ""
    print(f"[eaasr] GPU: {out or 'none'}")
    return out
