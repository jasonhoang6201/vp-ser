// Deterministic clip engine: every frame is a pure function of time t, so the renderer can step frame by frame
// and the browser preview can play in real time with the same code.
(function (root) {
  const W = 1920, H = 1080, LEAD = 0.8, GAP = 0.5, XFADE = 0.35;
  let TAIL = 2.5;
  const COL = {page: '#fbfbfa', card: '#ffffff', ink: '#37352f', mute: '#787774', line: '#dcdbd7', line2: '#c4c2bc', soft: '#f1f0ed',
    acc: '#337ea9', accSoft: '#e7f3f8', frame: '#d9730d', frameSoft: '#fdf1e6', hl: '#d44c47', hlSoft: '#fdf3f2', ok: '#448361', i: '#9065b0'};
  const FONT = '"SF Pro Display", "Helvetica Neue", system-ui, sans-serif', MONO = 'ui-monospace, Menlo, monospace';

  const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
  const ease = x => { x = clamp(x); return x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2; };
  const lerp = (a, b, u) => a + (b - a) * u;

  let g, TL = {}, SHOTS = [], SPEC, TOTAL = 0, SUBS = [];

  function buildTimeline(spec, dur) {
    let t = LEAD;
    for (const s of spec.segments) {
      const d = dur[s.id];
      if (d == null) throw new Error('no audio duration for segment ' + s.id);
      TL[s.id] = {start: t, end: t + d, dur: d};
      // Sentences share the segment time in proportion to their length, close enough to the reading pace.
      const parts = s.sub.split(/(?<=[.?!])\s+/), tot = parts.reduce((a, p) => a + p.length, 0);
      let u = t;
      for (const p of parts) { const du = d * p.length / tot; SUBS.push({a: u, b: u + du, text: p}); u += du; }
      t += d + GAP;
    }
    TOTAL = t - GAP + TAIL;
  }

  const S = id => TL[id].start, E = id => TL[id].end, D = id => TL[id].dur;
  // Eased 0..1 progress of an animation that starts at time a and lasts d seconds.
  const P = (t, a, d = 0.8) => ease((t - a) / d);

  function init(opts) {
    SPEC = opts.spec;
    if (opts.tail) TAIL = opts.tail;
    buildTimeline(opts.spec, opts.durations);
    SHOTS = opts.shots.map(s => ({...s, a: S(s.from) - GAP / 2, b: s.to === '$end' ? TOTAL : E(s.to) + GAP / 2}));
    SHOTS[0].a = 0;
    const cv = document.getElementById('cv');
    cv.width = W; cv.height = H;
    g = cv.getContext('2d');
    root.TOTAL = TOTAL;
    root.TIMELINE = TL;
  }

  function renderAt(t) {
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.globalAlpha = 1;
    g.fillStyle = COL.page; g.fillRect(0, 0, W, H);
    for (const s of SHOTS) {
      if (t < s.a - XFADE || t > s.b + XFADE) continue;
      const al = Math.min(clamp((t - s.a) / XFADE + (s.a === 0 ? 1 : 0)), clamp((s.b - t) / XFADE + (s.b === TOTAL ? 1 : 0)));
      if (al <= 0) continue;
      g.save(); g.globalAlpha = al; s.draw(g, t); g.restore();
    }
    header(t);
    subtitle(t);
  }

  function header(t) {
    text(SPEC.title, 60, 52, {size: 22, color: COL.mute, weight: 500});
    g.fillStyle = COL.soft; g.fillRect(0, H - 6, W, 6);
    g.fillStyle = COL.line2; g.fillRect(0, H - 6, W * clamp(t / TOTAL), 6);
  }

  function subtitle(t) {
    const s = SUBS.find(x => t >= x.a && t < x.b + GAP * 0.8);
    if (!s) return;
    const al = Math.min(clamp((t - s.a) / 0.15), 1);
    g.save(); g.globalAlpha = al;
    g.font = `400 30px ${FONT}`;
    const lines = wrap(s.text, 1560), lh = 42, h = lines.length * lh + 28, y0 = H - 40 - h;
    let w = 0;
    for (const l of lines) w = Math.max(w, g.measureText(l).width);
    rrect((W - w) / 2 - 28, y0, w + 56, h, 12, 'rgba(255,255,255,0.94)', COL.line);
    lines.forEach((l, j) => text(l, W / 2, y0 + 14 + lh / 2 + j * lh, {size: 30, color: COL.ink, align: 'center'}));
    g.restore();
  }

  function wrap(s, maxW) {
    const words = s.split(' '), lines = [];
    let cur = '';
    for (const w of words) {
      const nx = cur ? cur + ' ' + w : w;
      if (g.measureText(nx).width > maxW && cur) { lines.push(cur); cur = w; } else cur = nx;
    }
    if (cur) lines.push(cur);
    return lines;
  }

  function text(s, x, y, o = {}) {
    g.font = `${o.weight || 400} ${o.size || 26}px ${o.mono ? MONO : FONT}`;
    g.fillStyle = o.color || COL.ink;
    g.textAlign = o.align || 'left';
    g.textBaseline = o.base || 'middle';
    if (o.maxW) {
      wrap(s, o.maxW).forEach((l, j) => g.fillText(l, x, y + j * (o.lh || (o.size || 26) * 1.4)));
      return;
    }
    g.fillText(s, x, y);
  }
  function rrect(x, y, w, h, r, fill, stroke, lw = 2) {
    g.beginPath(); g.roundRect(x, y, w, h, r);
    if (fill) { g.fillStyle = fill; g.fill(); }
    if (stroke) { g.strokeStyle = stroke; g.lineWidth = lw; g.stroke(); }
  }
  function line(x1, y1, x2, y2, color = COL.line, lw = 2, dash) {
    g.beginPath(); g.setLineDash(dash || []); g.moveTo(x1, y1); g.lineTo(x2, y2);
    g.strokeStyle = color; g.lineWidth = lw; g.stroke(); g.setLineDash([]);
  }
  // Polyline through (xs[i], ys[i]); `upto` in 0..1 draws only the first part, for line-drawing animations.
  function poly(xs, ys, color, lw = 3, dash, upto = 1) {
    const n = xs.length, m = (n - 1) * clamp(upto);
    if (m <= 0) return;
    g.beginPath(); g.setLineDash(dash || []);
    g.moveTo(xs[0], ys[0]);
    for (let i = 1; i <= Math.floor(m); i++) g.lineTo(xs[i], ys[i]);
    const f = m - Math.floor(m), i = Math.floor(m);
    if (f > 0 && i + 1 < n) g.lineTo(lerp(xs[i], xs[i + 1], f), lerp(ys[i], ys[i + 1], f));
    g.strokeStyle = color; g.lineWidth = lw; g.lineJoin = 'round'; g.stroke(); g.setLineDash([]);
  }
  function arrow(x1, y1, x2, y2, color = COL.mute, lw = 2.5, head = 12) {
    line(x1, y1, x2, y2, color, lw);
    const a = Math.atan2(y2 - y1, x2 - x1);
    g.beginPath(); g.moveTo(x2, y2);
    g.lineTo(x2 - head * Math.cos(a - 0.45), y2 - head * Math.sin(a - 0.45));
    g.lineTo(x2 - head * Math.cos(a + 0.45), y2 - head * Math.sin(a + 0.45));
    g.closePath(); g.fillStyle = color; g.fill();
  }
  // A plot box with linear x/y maps and optional light grid.
  function axes(B, xr, yr, o = {}) {
    const X = v => B.x + (v - xr[0]) / (xr[1] - xr[0]) * B.w, Y = v => B.y + B.h - (v - yr[0]) / (yr[1] - yr[0]) * B.h;
    if (o.gridY) for (const v of o.gridY) { line(B.x, Y(v), B.x + B.w, Y(v), '#ecebe8', 1.5); if (o.fmtY) text(o.fmtY(v), B.x - 12, Y(v), {size: 20, color: COL.mute, align: 'right'}); }
    if (o.ticksX) for (const v of o.ticksX) text(o.fmtX ? o.fmtX(v) : String(v), X(v), B.y + B.h + 24, {size: 20, color: COL.mute, align: 'center'});
    line(B.x, B.y + B.h, B.x + B.w, B.y + B.h, COL.line2, 2);
    if (o.xlabel) text(o.xlabel, B.x + B.w, B.y + B.h + 54, {size: 20, color: COL.mute, align: 'right'});
    if (o.ylabel) text(o.ylabel, B.x, B.y - 22, {size: 20, color: COL.mute});
    return {X, Y};
  }
  function heat(v) {
    const stops = [[251, 251, 250], [231, 243, 248], [51, 126, 169], [217, 115, 13], [140, 50, 20]];
    v = clamp(v) * (stops.length - 1);
    const i = Math.min(stops.length - 2, Math.floor(v)), f = v - i, a = stops[i], b = stops[i + 1];
    return `rgb(${a.map((c, k) => Math.round(lerp(c, b[k], f))).join(',')})`;
  }

  // Real-time preview: plays the narration track in sync when opened directly in a browser.
  function preview(audioSrc) {
    const au = new Audio(audioSrc), bar = document.getElementById('bar');
    let t0 = null, playing = false, off = 0;
    const tick = now => {
      if (!playing) return;
      if (t0 == null) t0 = now - off * 1000;
      const t = (now - t0) / 1000;
      renderAt(Math.min(t, TOTAL));
      if (t < TOTAL) requestAnimationFrame(tick); else playing = false;
    };
    document.addEventListener('keydown', e => {
      if (e.code !== 'Space') return;
      e.preventDefault();
      playing = !playing;
      if (playing) { off = au.currentTime; t0 = null; au.play().catch(() => {}); requestAnimationFrame(tick); } else au.pause();
    });
    if (bar) bar.textContent = 'Space: phát / dừng';
    renderAt(0);
  }

  root.Clip = {W, H, COL, FONT, clamp, ease, lerp, init, renderAt, S, E, D, P, text, rrect, line, poly, arrow, axes, heat, wrap, preview,
    get g() { return g; }};
  root.renderAt = t => renderAt(t);
})(window);
