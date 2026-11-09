"""Convert notebooks/src/*.py (percent format) into notebooks/*.ipynb for Colab.

Cells start with `# %%` (code) or `# %% [markdown]` (markdown, each line prefixed by `# `).
Usage: python notebooks/tools/build_notebooks.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
        if lines:
            cells.append((kind, lines))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def to_ipynb(cells, gpu):
    nb_cells = []
    for kind, lines in cells:
        src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        nb_cells.append(cell)
    meta = {"kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "machine_shape": "hm"}}
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = gpu
    return {"cells": nb_cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0}


def main():
    for src in sorted((ROOT / "src").glob("*.py")):
        text = src.read_text()
        gpu = None
        for line in text.splitlines()[:5]:
            if line.startswith("# gpu:"):
                gpu = line.split(":", 1)[1].strip() or None
        cells = parse(text)
        out = ROOT / (src.stem + ".ipynb")
        out.write_text(json.dumps(to_ipynb(cells, gpu), indent=1, ensure_ascii=False))
        print(f"{out.relative_to(ROOT)}: {len(cells)} cells")


if __name__ == "__main__":
    sys.exit(main())
