"""Build the IEMOCAP manifest and 16 kHz wav tar from the official USC release (run locally, stdlib only).

Input: an extracted IEMOCAP_full_release/ holding Session*/sentences/wav, dialog/EmoEvaluation, dialog/transcriptions.
Output (same layout as the HuggingFace loader, so notebooks 10/12/20/21 need no change):
  <out>/iemocap_manifest.csv  id, relpath, session, speaker, type, text, emotion, emotion_raw, duration,
                              dialogue_id, start, end
  <out>/iemocap_16k.tar       iemocap/wav/<id>.wav

Usage: python notebooks/tools/prep_iemocap_official.py <IEMOCAP_full_release> <out_dir>
"""
import csv
import re
import sys
import tarfile
import wave
from collections import Counter
from pathlib import Path

FOUR = {"neu": "neutral", "hap": "joy", "exc": "joy", "ang": "anger", "sad": "sadness"}
EMO_LINE = re.compile(r"^\[(\d+\.\d+) - (\d+\.\d+)\]\t(Ses\w+)\t(\w+)\t")
TXT_LINE = re.compile(r"^(Ses\w+) \[[\d.]+-[\d.]+\]: ?(.*)$")
ID_PAT = re.compile(r"^Ses0([1-5])([FM])_(impro|script)\w*_([FM])\d+$")

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

labels, times, texts, wavs = {}, {}, {}, {}
for s in range(1, 6):
    sess = root / f"Session{s}"
    for f in sorted((sess / "dialog/EmoEvaluation").glob("Ses*.txt")):
        for line in f.read_text(errors="replace").splitlines():
            m = EMO_LINE.match(line)
            if m:
                labels[m[3]] = m[4]
                times[m[3]] = (float(m[1]), float(m[2]))
    for f in sorted((sess / "dialog/transcriptions").glob("Ses*.txt")):
        for line in f.read_text(errors="replace").splitlines():
            m = TXT_LINE.match(line.strip())
            if m:
                texts[m[1]] = " ".join(m[2].split())
    for w in (sess / "sentences/wav").glob("*/*.wav"):
        wavs[w.stem] = w

rows, problems = [], Counter()
for uid in sorted(wavs):
    m = ID_PAT.match(uid)
    if not m or uid not in labels or uid not in texts:
        problems["missing label/text/pattern"] += 1
        continue
    with wave.open(str(wavs[uid])) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, uid
        dur = w.getnframes() / 16000
    rows.append({
        "id": uid, "relpath": f"iemocap/wav/{uid}.wav", "session": int(m[1]), "speaker": f"Ses0{m[1]}{m[4]}",
        "type": m[3], "text": texts[uid], "emotion": FOUR.get(labels[uid], ""), "emotion_raw": labels[uid],
        "duration": round(dur, 3), "dialogue_id": uid.rsplit("_", 1)[0],
        "start": times[uid][0], "end": times[uid][1],
    })

print("wav files:", len(wavs), "rows:", len(rows), dict(problems))
print("raw labels:", Counter(r["emotion_raw"] for r in rows).most_common())
four = Counter(r["emotion"] for r in rows if r["emotion"])
print("4-class:", sum(four.values()), dict(four), "(expected 5531: neu 1708, joy 1636, ang 1103, sad 1084)")

with open(out / "iemocap_manifest.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
# Uncompressed: wav does not shrink under gzip, and notebooks extract it with plain `tar -xf`.
with tarfile.open(out / "iemocap_16k.tar", "w") as t:
    for r in rows:
        t.add(wavs[r["id"]], arcname=r["relpath"])
print("wrote", out)
