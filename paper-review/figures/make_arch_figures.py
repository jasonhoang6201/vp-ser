"""Architecture figures for the report and slides, as .drawio files with PNG exports next to them.

  python3 paper-review/figures/make_arch_figures.py [out_dir]

fig_overview: high-level modules only. fig_flow: one utterance through the model with tensor shapes and formulas.
Style helpers live in the research-paper skill (scripts/drawio_system.py).
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/.claude/skills/research-paper/scripts"))
from drawio_lib import Fig  # noqa: E402
import drawio_system as ds  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))


def overview():
    g = Fig("fig_overview")
    src = ds.io_input(g, 20, 262, "Audio")
    j = ds.dot(g, 215, 290)
    asr = ds.box(g, 280, 120, 210, 64, "Whisper large-v3-turbo", "pre", "flame")
    wsp = ds.box(g, 280, 400, 210, 64, "Whisper large-v3", "pre", "lock")
    hist = ds.cyl(g, 620, 250, 200, 90, "Dialogue history")
    rob = ds.box(g, 620, 120, 200, 64, "RoBERTa-large", "pre", "flame")
    fus = ds.box(g, 910, 250, 220, 76, "Gated Fusion", "core", "flame", 16)
    aux = ds.box(g, 925, 110, 190, 56, "Auxiliary heads", "new", "flame")
    cls = ds.box(g, 925, 410, 190, 56, "Classifier", "new", "flame")
    out_t = ds.box(g, 1200, 120, 160, 56, "Transcript", "io")
    out_e = ds.box(g, 1200, 410, 160, 56, "Emotion", "io")

    ds.arrow(g, src, j, head=False)
    ds.arrow(g, j, asr, points=[(215, 152)])
    ds.arrow(g, j, wsp, points=[(215, 432)])
    ds.arrow(g, asr, rob, "transcript")
    ds.arrow(g, asr, out_t, points=[(385, 85), (1280, 85)], exit=(0.5, 0), entry=(0.5, 0))
    ds.arrow(g, asr, hist, "store turn", points=[(440, 295)], exit=(0.75, 1), entry=(0, 0.5), label_pos=0.45)
    ds.arrow(g, hist, rob, "context", exit=(0.5, 0), entry=(0.5, 1))
    ds.arrow(g, rob, fus, "h<sub>text</sub>", points=[(865, 152), (865, 269)], exit=(1, 0.5), entry=(0, 0.25))
    ds.arrow(g, wsp, fus, "h<sub>audio</sub>", points=[(865, 432), (865, 307)], exit=(1, 0.5), entry=(0, 0.75))
    ds.arrow(g, fus, aux, exit=(0.5, 0), entry=(0.5, 1))
    ds.arrow(g, fus, cls, exit=(0.5, 1), entry=(0.5, 0))
    ds.arrow(g, cls, out_e)
    ds.legend(g, 280, 510)
    g.save(OUT)
    g.export(OUT)


def flow():
    g = Fig("fig_flow")
    # Text branch on the left, audio branch on the right, merged below. Shapes and formulas sit beside each block.
    sp = ds.io_input(g, 400, 20, "Audio", w=180, h=50)
    j = ds.dot(g, 490, 110)

    asr = ds.box(g, 60, 150, 220, 50, "Whisper large-v3-turbo", "pre", "flame")
    ctx = ds.cyl(g, 60, 250, 220, 76, "Context + current turn")
    rob = ds.box(g, 60, 380, 220, 50, "RoBERTa-large", "pre", "flame")
    ds.note(g, 295, 394, "[CLS] → h<sub>text</sub> ∈ ℝ<sup>1024</sup>")

    wsp = ds.box(g, 700, 150, 220, 50, "Whisper large-v3", "pre", "lock")
    ds.note(g, 935, 164, "33 layers × T frames × 1280")
    pool = ds.box(g, 700, 255, 220, 50, "Mean over frames", "new")
    ds.note(g, 935, 269, "33 × 1280")
    lw = ds.box(g, 700, 355, 220, 50, "Layer weights", "new", "flame")
    ds.note(g, 935, 369, "Σ<sub>l</sub> w<sub>l</sub> h<sub>l</sub> ∈ ℝ<sup>1280</sup>")
    proj = ds.box(g, 700, 455, 220, 50, "Projection", "new", "flame")
    ds.note(g, 935, 469, "h<sub>audio</sub> ∈ ℝ<sup>1024</sup>")

    gate = ds.box(g, 360, 570, 260, 64, "Gated Fusion", "core", "flame", 16)
    ds.note(g, 510, 650, "g = softmax(W[h<sub>text</sub>; h<sub>audio</sub>])")
    ds.note(g, 510, 675, "z = g<sub>text</sub> h<sub>text</sub> + g<sub>audio</sub> h<sub>audio</sub> ∈ ℝ<sup>1024</sup>")
    aux = ds.box(g, 700, 576, 220, 52, "Auxiliary heads", "new", "flame")
    ds.note(g, 935, 591, "L = CE(z) + CE(h<sub>text</sub>) + CE(h<sub>audio</sub>)")
    cls = ds.box(g, 380, 730, 220, 50, "Classifier", "new", "flame")
    ds.note(g, 615, 744, "p = softmax(W<sub>c</sub> z) ∈ ℝ<sup>4</sup>")
    out = ds.box(g, 380, 840, 220, 50, "Emotion", "io")

    ds.arrow(g, sp, j, head=False, exit=(0.5, 1))
    ds.arrow(g, j, asr, entry=(0.5, 0), points=[(170, 110)])
    ds.arrow(g, j, wsp, entry=(0.5, 0), points=[(810, 110)])
    ds.arrow(g, asr, ctx, exit=(0.5, 1), entry=(0.5, 0))
    ds.arrow(g, ctx, rob, exit=(0.5, 1), entry=(0.5, 0))
    ds.arrow(g, wsp, pool, exit=(0.5, 1), entry=(0.5, 0))
    ds.arrow(g, pool, lw, exit=(0.5, 1), entry=(0.5, 0))
    ds.arrow(g, lw, proj, exit=(0.5, 1), entry=(0.5, 0))
    ds.arrow(g, rob, gate, exit=(0.5, 1), entry=(0, 0.5), points=[(170, 602)])
    ds.arrow(g, proj, gate, exit=(0, 0.5), entry=(0.5, 0), points=[(490, 480)])
    ds.arrow(g, gate, aux, exit=(1, 0.5), entry=(0, 0.5))
    ds.arrow(g, gate, cls, exit=(0.5, 1), entry=(0.5, 0))
    ds.arrow(g, cls, out, exit=(0.5, 1), entry=(0.5, 0))
    ds.legend(g, 60, 930)
    g.save(OUT)
    g.export(OUT)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    overview()
    flow()
