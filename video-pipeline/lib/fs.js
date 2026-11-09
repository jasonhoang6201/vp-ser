// FastSpeech 2 toy for the clips: "xin chào" as 6 phonemes, the length regulator and the per-frame f0 / energy curves.
// Same numbers and curve shapes as the interactive page demo/15-fastspeech2.html.
(function (root) {
  const {COL, text, rrect, line, poly} = root.Clip;
  const PH = [{p: 'x', d: 9, v: false}, {p: 'i', d: 12, v: true}, {p: 'n', d: 8, v: true}, {p: 'ch', d: 8, v: false}, {p: 'a', d: 18, v: true}, {p: 'o', d: 16, v: true}];
  const PCOL = [COL.mute, COL.acc, COL.ok, COL.mute, COL.frame, COL.i];
  // Cell width is fixed to the slowest speed shown (101 frames) so cells do not jump when the speed changes.
  const MAXF = 104;

  const durs = (spd = 1, over = {}) => PH.map((o, i) => Math.max(1, Math.round((over[i] ?? o.d) / spd)));
  const frames = D => D.flatMap((n, i) => Array.from({length: n}, (_, j) => ({i, u: n > 1 ? j / (n - 1) : 0})));

  function f0At(i, u, mode, shift = 0) {
    const syl = i < 3 ? 0 : 1, su = (i - 3 * syl + u) / 3;
    let hz;
    if (mode === 'flat') hz = 190;
    else if (syl === 0) hz = 200;
    else if (mode === 'tone') hz = 175 - 45 * su;
    else hz = su < 0.55 ? 175 - 55 * su / 0.55 : 120 + 70 * (su - 0.55) / 0.45;
    return hz * 2 ** (shift / 12);
  }
  const enAt = (i, u, en = 1) => en * (PH[i].v ? (i === 1 || i === 4 ? 1 : 0.8) * Math.min(1, 4 * u + 0.4, 4 * (1 - u) + 0.6) : 0.35);

  // Phoneme boxes on top, one coloured cell per frame below, thin lines from each phoneme to the middle of its run.
  // `cells` is how many frame cells are drawn so far (for a left-to-right fill), `links` the opacity of the lines.
  function drawRow(B, D, cells = Infinity, links = 1) {
    const pw = 120, cw = B.w / MAXF, y2 = B.y + 200, total = D.reduce((a, b) => a + b, 0);
    const px = B.x + (B.w - PH.length * pw) / 2;
    PH.forEach((o, i) => {
      rrect(px + i * pw + 6, B.y, pw - 12, 64, 12, PCOL[i]);
      text(o.p, px + (i + 0.5) * pw, B.y + 32, {size: 30, weight: 700, color: '#fff', align: 'center'});
    });
    let acc = 0;
    D.forEach((d, i) => {
      if (links > 0) {
        root.Clip.g.save(); root.Clip.g.globalAlpha *= links;
        line(px + (i + 0.5) * pw, B.y + 66, B.x + (acc + d / 2) * cw, y2 - 4, PCOL[i], 2);
        root.Clip.g.restore();
      }
      for (let j = 0; j < d; j++) if (acc + j < cells) rrect(B.x + (acc + j) * cw, y2, Math.max(2, cw - 2), 64, 3, PCOL[i]);
      acc += d;
    });
    return {total, y2, cw};
  }

  // f0 dots (voiced frames only) and the energy line over the phoneme strip; `upto` reveals left to right.
  function drawCurves(B, D, o = {}) {
    const F = frames(D), cw = B.w / F.length, mode = o.mode || 'tone', upto = o.upto ?? 1;
    const Yf = hz => B.y + B.h - (hz - 80) / (320 - 80) * B.h, Ye = e => B.y + B.h - e / 1.6 * B.h;
    [100, 200, 300].forEach(hz => {
      line(B.x, Yf(hz), B.x + B.w, Yf(hz), COL.soft, 1.5);
      text(hz + ' Hz', B.x - 12, Yf(hz), {size: 20, color: COL.mute, align: 'right'});
    });
    let acc = 0;
    D.forEach((d, i) => {
      rrect(B.x + acc * cw, B.y + B.h + 10, d * cw - 2, 40, 6, PCOL[i]);
      text(PH[i].p, B.x + (acc + d / 2) * cw, B.y + B.h + 30, {size: 22, weight: 700, color: '#fff', align: 'center'});
      acc += d;
    });
    const n = Math.floor(F.length * upto);
    if (o.energy !== false) poly(F.map((_, t) => B.x + (t + 0.5) * cw), F.map(f => Ye(enAt(f.i, f.u, o.en ?? 1))), COL.acc, 3, [8, 6], upto);
    const g = root.Clip.g;
    F.slice(0, n).forEach((f, t) => {
      if (!PH[f.i].v) return;
      g.beginPath(); g.arc(B.x + (t + 0.5) * cw, Yf(f0At(f.i, f.u, mode, o.shift || 0)), 5, 0, 2 * Math.PI); g.fillStyle = COL.frame; g.fill();
    });
    return {Yf, Ye, cw};
  }

  root.FS = {PH, PCOL, MAXF, durs, frames, f0At, enAt, drawRow, drawCurves};
})(window);
