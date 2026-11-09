// Shared building blocks for the clip scenes: cards, chips, the opening title and the closing terms table.
(function (root) {
  const {COL, P, text, rrect, line} = root.Clip;
  const PARTS = ['1 Vấn đề', '2 Log', '3 DCT', '4 Giữ 13 hệ số', '5 Cả câu: ma trận, delta', '6 MFCC ngày nay'];

  function card(x, y, w, h, o = {}) { rrect(x, y, w, h, 14, o.fill || COL.card, o.stroke || COL.line, o.lw || 2); }
  function heading(s, y = 140) { text(s, 960, y, {size: 44, weight: 700, align: 'center'}); }
  function fade(G, q, fn) { if (q <= 0) return; G.save(); G.globalAlpha *= Math.min(q, 1); fn(); G.restore(); }
  function chip(G, x, y, s, o = {}) {
    G.font = `${o.weight || 500} ${o.size || 24}px ${root.Clip.FONT}`;
    const w = G.measureText(s).width + 36;
    rrect(x, y, w, 52, 26, o.fill || COL.card, o.stroke || COL.line);
    text(s, x + w / 2, y + 26, {size: o.size || 24, weight: o.weight || 400, color: o.color || COL.mute, align: 'center'});
    return w;
  }
  const HOT = {fill: COL.frameSoft, stroke: COL.frame, color: COL.frame, weight: 600};
  // Left x of each chip so the row of chips is centred on 960 (widths measured the way chip() measures them).
  function chipXs(G, labels, o = {}, gap = 24) {
    G.font = `${o.weight || 500} ${o.size || 24}px ${root.Clip.FONT}`;
    const ws = labels.map(s => G.measureText(s).width + 36);
    let x = 960 - (ws.reduce((a, b) => a + b, 0) + gap * (ws.length - 1)) / 2;
    return ws.map(w => { const x0 = x; x += w + gap; return x0; });
  }

  // Title card plus the chip row of the page's parts with part `idx` (0-based) highlighted.
  const introShot = (title, sub, idx, parts = PARTS) => (G, t) => {
    fade(G, P(t, 0.3, 1.0), () => {
      text(title, 960, 400, {size: 76, weight: 700, align: 'center'});
      text(sub, 960, 490, {size: 34, color: COL.mute, align: 'center'});
    });
    G.font = `500 24px ${root.Clip.FONT}`;
    let x = 960 - (parts.reduce((s, p) => s + G.measureText(p).width + 52, 0) - 16) / 2;
    parts.forEach((s, j) => {
      G.save(); G.globalAlpha *= P(t, 0.9 + j * 0.12, 0.5);
      x += chip(G, x, 600, s, j === idx ? HOT : {}) + 16;
      G.restore();
    });
  };

  // rows: [plain words, term, meaning]
  const termsShot = (title, rows, segId) => (G, t) => {
    heading(title, 130);
    const s = root.Clip.S(segId), cx = [170, 640, 1220], y0 = 210, rh = Math.min(72, 560 / rows.length);
    ['Nói thường', 'Thuật ngữ', 'Ý nghĩa'].forEach((h, j) => text(h, cx[j], y0, {size: 24, weight: 600, color: COL.mute}));
    line(150, y0 + 26, 1770, y0 + 26, COL.line2, 2);
    rows.forEach((r, i) => fade(G, P(t, s + 0.2 + i * 0.4, 0.5), () => {
      const y = y0 + 64 + i * rh;
      text(r[0], cx[0], y, {size: 27});
      text(r[1], cx[1], y, {size: 27, weight: 700, color: COL.acc});
      text(r[2], cx[2], y, {size: 25, color: COL.mute});
      line(150, y + rh / 2, 1770, y + rh / 2, '#ecebe8', 1.5);
    }));
  };

  // Points j = 0..n-1 of a 40-filter curve on box B; returns the x positions and a y map.
  function curveBox(B, lo, hi, n = 40) {
    const X = j => B.x + (j + 0.5) / n * B.w, Y = v => B.y + B.h - (v - lo) / (hi - lo) * B.h;
    return {X, Y, xs: Array.from({length: n}, (_, j) => X(j))};
  }

  root.UI = {card, heading, fade, chip, chipXs, HOT, introShot, termsShot, curveBox};
})(window);
