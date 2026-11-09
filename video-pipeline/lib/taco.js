// Illustrative Tacotron 2 attention paths and per-character Mel templates for "xin chào", same as
// demo/14-tacotron2.html (made up to show the failure modes, not taken from a trained model).
(function (root) {
  const rng = root.GMM.rng, clamp = root.Clip.clamp;
  const TXT = [...'xin chào'], NB = 40, MAXS = 90, DUR = [6, 8, 5, 4, 7, 6, 10, 9];
  const seq = idx => idx.flatMap(i => new Array(DUR[i]).fill(i));
  function pathFor(md) {
    if (md === 'good') return {path: seq([0, 1, 2, 3, 4, 5, 6, 7]), stop: true};
    if (md === 'repeat') return {path: seq([0, 1, 2, 3, 4, 5, 4, 5, 6, 7]), stop: true};
    if (md === 'skip') return {path: seq([0, 1, 4, 5, 6, 7]), stop: true};
    const p = seq([0, 1, 2, 3, 4, 5, 6, 7]);
    while (p.length < MAXS) p.push(7);
    return {path: p, stop: false};
  }
  const TEMPL = TXT.map(ch => {
    if (ch === ' ') return new Array(NB).fill(0.03);
    const r = rng(40 + ch.charCodeAt(0)), voiced = 'iaào'.includes(ch) || ch === 'n';
    const f = [3 + 10 * r(), 12 + 12 * r(), 25 + 10 * r()];
    return Array.from({length: NB}, (_, b) => (voiced ? 0.1 + f.reduce((s, c, k) => s + (0.9 - 0.25 * k) * Math.exp(-0.5 * ((b - c) / 2.2) ** 2), 0) : 0.1 + 0.6 * clamp((b - 20) / 15) * (0.6 + 0.4 * r())));
  });
  // What the decoder "reads": consecutive repeats of the attended character collapse to one.
  const readOut = (path, n) => { const out = []; for (let s = 0; s < n; s++) if (!s || path[s] !== path[s - 1]) out.push(TXT[path[s]]); return out.join(''); };

  // Attention matrix (top) and generated Mel (bottom) after `shown` decoder steps.
  function draw(B, md, shown) {
    const C = root.Clip, {path} = pathFor(md), cw = B.w / MAXS, rh = 30, mt = B.y + TXT.length * rh + 30, mh = B.mh || 200;
    for (let i = 0; i < TXT.length; i++) {
      for (let s = 0; s < MAXS; s++) {
        let v = 0;
        if (s < shown && s < path.length) { const d = Math.abs(path[s] - i); v = d === 0 ? 0.85 : d === 1 ? 0.25 : 0; }
        C.rrect(B.x + s * cw, B.y + i * rh, cw - 1, rh - 2, 1, v ? C.heat(v) : '#f4f3f0');
      }
      C.text(TXT[i] === ' ' ? '␣' : TXT[i], B.x - 14, B.y + i * rh + rh / 2, {size: 22, weight: 700, align: 'right', mono: true});
    }
    const n = Math.min(shown, path.length), bh = mh / NB;
    for (let s = 0; s < n; s++) TEMPL[path[s]].forEach((v, b) => C.rrect(B.x + s * cw, mt + mh - (b + 1) * bh, cw, bh + 0.5, 0, C.heat(clamp(v))));
    C.rrect(B.x, mt, B.w, mh, 2, null, C.COL.line2, 1.5);
    C.text('Mel', B.x - 14, mt + mh / 2, {size: 20, color: C.COL.mute, align: 'right'});
    for (let s = 0; s <= MAXS; s += 15) C.text(String(s), B.x + s * cw, mt + mh + 18, {size: 18, color: C.COL.mute, align: 'center'});
    return {n, read: readOut(path, n), total: path.length, mt, mh};
  }

  root.TACO = {TXT, MAXS, DUR, pathFor, readOut, draw};
})(window);
