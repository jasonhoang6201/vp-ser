"""Final comparison tables from results/<dataset>/*/metrics_test.json (runs locally, stdlib only).

  python3 notebooks/tools/compare_results.py      writes notebooks/results/summary.md

Per dataset: one row per system (WER, WA, UA, macro-F1, weighted-F1, UA mean±std over seeds), WER and recall by
emotion. For IEMOCAP also leave-one-session-out cross-validation: fold k is stored with suffix `_f<k>`, fold 5 has
no suffix. Two CV tables: the mean over each fold's seeds, the headline number, and seed 0 only. Seed counts differ
between systems, so the seed 0 table is the one where every system gets the same number of runs per fold.
"""
import json
import re
import statistics
from pathlib import Path

NB = Path(__file__).resolve().parents[1]
RES = NB / "results"
LABELS = {"meld": ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"],
          "iemocap": ["neutral", "joy", "anger", "sadness"]}
FOLD = re.compile(r"_f([1-4])$")


def load(dataset):
    out = {}
    for p in sorted((RES / dataset).glob("*/metrics_test.json")):
        out[p.parent.name] = json.loads(p.read_text())
    return out


def pct(x):
    return "" if x is None else f"{x * 100:.2f}"


def table(head, rows):
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def system_table(ms):
    rows = []
    for name, m in ms.items():
        if FOLD.search(name):
            continue
        a, s = m.get("asr", {}), m.get("ser", {})
        spread = f"{m['ua_mean'] * 100:.2f}±{m['ua_std'] * 100:.2f}" if "ua_std" in m else ""
        rows.append([name, pct(a.get("wer")), pct(s.get("wa")), pct(s.get("ua")), pct(s.get("macro_f1")),
                     pct(s.get("weighted_f1")), spread, len(m.get("seeds", [])) or ""])
    rows.sort(key=lambda r: -float(r[3]) if r[3] else 1)
    return table(["system", "WER", "WA", "UA", "macro-F1", "wF1", "UA mean±std", "seeds"], rows)


def by_emotion(ms, labels, part, key):
    rows = [[n] + [f"{m[part][key][l] * 100:.1f}" if l in m[part][key] else "" for l in labels]
            for n, m in ms.items() if not FOLD.search(n) and key in m.get(part, {})]
    return table(["system"] + labels, rows)


def seed0(m, key):
    seeds = [s for s in m.get("seeds", []) if s["seed"] == 0]
    if seeds:
        return seeds[0][key]
    return m["ser"]["ua" if key == "test_ua" else "weighted_f1"]


def seed_mean(m, key):
    seeds = m.get("seeds", [])
    return statistics.mean(s[key] for s in seeds) if seeds else seed0(m, key)


def cv_table(ms, per_fold):
    folded = {}
    for name in ms:
        f = FOLD.search(name)
        if f:
            folded.setdefault(name[:f.start()], {})[int(f[1])] = ms[name]
    rows = []
    for base, folds in sorted(folded.items()):
        if base in ms:
            folds[5] = ms[base]
        ks = sorted(folds)
        if not all("ser" in folds[k] or folds[k].get("seeds") for k in ks):
            continue
        ua = [per_fold(folds[k], "test_ua") for k in ks]
        wf1 = [per_fold(folds[k], "test_wf1") for k in ks]
        wer = [folds[k]["asr"]["wer"] for k in ks if "asr" in folds[k]]
        sd = lambda v: statistics.stdev(v) * 100 if len(v) > 1 else 0.0
        counts = " ".join(str(len(folds[k].get("seeds", [])) or 1) for k in ks)
        rows.append([base, len(ks), counts, " ".join(f"{u * 100:.1f}" for u in ua),
                     f"{statistics.mean(ua) * 100:.2f}±{sd(ua):.2f}", f"{statistics.mean(wf1) * 100:.2f}±{sd(wf1):.2f}",
                     f"{statistics.mean(wer) * 100:.2f}" if wer else ""])
    rows.sort(key=lambda r: -float(r[4].split("±")[0]))
    head = ["system", "folds", "seeds available", "UA per fold (1..5)", "UA mean±std", "wF1 mean±std", "WER mean"]
    return table(head, rows) if rows else ""


def main():
    out = ["# Tổng hợp kết quả (sinh tự động bởi tools/compare_results.py)", ""]
    for ds in ["meld", "iemocap"]:
        ms = load(ds)
        if not ms:
            continue
        out += [f"## {ds.upper()} test", "", system_table(ms), "",
                "### WER theo cảm xúc (%)", "", by_emotion(ms, LABELS[ds], "asr", "wer_by_emotion"), "",
                "### Recall theo cảm xúc (%)", "", by_emotion(ms, LABELS[ds], "ser", "recall_by_emotion"), ""]
        cv = cv_table(ms, seed_mean) if ds == "iemocap" else ""
        if cv:
            out += ["### IEMOCAP leave-one-session-out (trung bình các seed mỗi fold)", "",
                    "Std là độ lệch giữa các fold. Số seed khác nhau giữa các hệ thống, xem cột seeds available.", "", cv, "",
                    "### IEMOCAP leave-one-session-out (chỉ seed 0, mọi hệ thống cùng số lần chạy)", "",
                    cv_table(ms, seed0), ""]
    (RES / "summary.md").write_text("\n".join(out))
    print(RES / "summary.md")


if __name__ == "__main__":
    main()
