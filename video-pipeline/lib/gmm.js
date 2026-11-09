// 2-D Gaussian mixtures and EM for the GMM clips. Same generator and data as demo/06-gmm.html, so the clips show
// the clouds the reader already played with.
(function (root) {
  function rng(seed) {
    let a = seed >>> 0;
    return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  const randn = r => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());

  // specs: [mean x, mean y, sd x, sd y, correlation, count] per cluster.
  function cloud(seed, specs) {
    const r = rng(seed), pts = [];
    specs.forEach(([mx, my, sx, sy, rho, n]) => {
      for (let j = 0; j < n; j++) {
        const z1 = randn(r), z2 = randn(r);
        pts.push([mx + sx * z1, my + sy * (rho * z1 + Math.sqrt(1 - rho * rho) * z2)]);
      }
    });
    return pts;
  }
  const A_PTS = cloud(11, [[-1.3, 0.9, 0.45, 0.28, 0.5, 120], [1.1, 1.1, 0.35, 0.6, -0.4, 90], [0.1, -0.9, 0.7, 0.3, 0.3, 90]]);
  const I_PTS = cloud(23, [[1.9, -1.4, 0.4, 0.3, 0.2, 90], [2.4, 0.1, 0.3, 0.45, 0.3, 70]]);

  // cov is [var x, cov xy, var y].
  function pdf2(p, g) {
    const [a, b, c] = g.cov, det = a * c - b * b, dx = p[0] - g.mu[0], dy = p[1] - g.mu[1];
    return Math.exp(-0.5 * (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / det) / (2 * Math.PI * Math.sqrt(det));
  }
  const mixPdf = (p, gs) => gs.reduce((s, g) => s + g.w * pdf2(p, g), 0);
  const avgLL = (pts, gs) => pts.reduce((s, p) => s + Math.log(mixPdf(p, gs) + 1e-300), 0) / pts.length;

  function initGmm(pts, K, seed) {
    const r = rng(seed), n = pts.length;
    const mx = pts.reduce((s, p) => s + p[0], 0) / n, my = pts.reduce((s, p) => s + p[1], 0) / n;
    const vx = pts.reduce((s, p) => s + (p[0] - mx) ** 2, 0) / n, vy = pts.reduce((s, p) => s + (p[1] - my) ** 2, 0) / n;
    return Array.from({length: K}, () => ({mu: pts[Math.floor(r() * n)].slice(), cov: [vx * 0.5, 0, vy * 0.5], w: 1 / K}));
  }
  function resp(pts, gs) {
    return pts.map(p => { const v = gs.map(g => g.w * pdf2(p, g)), s = v.reduce((x, y) => x + y, 0) || 1e-300; return v.map(x => x / s); });
  }
  function emStep(pts, gs) {
    const R = resp(pts, gs), n = pts.length;
    return gs.map((g, k) => {
      const Nk = R.reduce((s, r) => s + r[k], 0) + 1e-9;
      const mu = [0, 1].map(d => R.reduce((s, r, j) => s + r[k] * pts[j][d], 0) / Nk);
      let a = 0, b = 0, c = 0;
      R.forEach((r, j) => { const dx = pts[j][0] - mu[0], dy = pts[j][1] - mu[1]; a += r[k] * dx * dx; b += r[k] * dx * dy; c += r[k] * dy * dy; });
      // A small floor keeps a component from collapsing onto one point (variance -> 0, likelihood -> infinity).
      return {mu, cov: [a / Nk + 1e-3, b / Nk, c / Nk + 1e-3], w: Nk / n};
    });
  }
  // All EM states from the start, so a clip can show step i at any time without re-running.
  function emTrace(pts, K, seed, iters) {
    const out = [initGmm(pts, K, seed)];
    for (let i = 0; i < iters; i++) out.push(emStep(pts, out[i]));
    return out;
  }
  // Ellipse at k standard deviations, as point lists in data coordinates.
  function ellipsePts(g, k, n = 64) {
    const [a, b, c] = g.cov, m = (a + c) / 2, df = Math.sqrt(((a - c) / 2) ** 2 + b * b);
    const l1 = m + df, l2 = Math.max(m - df, 1e-9), th = 0.5 * Math.atan2(2 * b, a - c), ct = Math.cos(th), st = Math.sin(th);
    const xs = [], ys = [];
    for (let t = 0; t <= n; t++) {
      const u = 2 * Math.PI * t / n, e1 = k * Math.sqrt(l1) * Math.cos(u), e2 = k * Math.sqrt(l2) * Math.sin(u);
      xs.push(g.mu[0] + e1 * ct - e2 * st); ys.push(g.mu[1] + e1 * st + e2 * ct);
    }
    return [xs, ys];
  }

  // 1-D Gaussian density.
  const npdf = (x, mu, sd) => Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));

  // The (c1, c2) plane on box B. Keep B.w / B.h = 1.32 so a circle in data units stays round on screen.
  function plane(B, xr = [-3, 3.6], yr = [-2.6, 2.4]) {
    const C = root.Clip, COL = C.COL;
    const X = v => B.x + (v - xr[0]) / (xr[1] - xr[0]) * B.w, Y = v => B.y + B.h - (v - yr[0]) / (yr[1] - yr[0]) * B.h;
    const frame = () => {
      C.rrect(B.x, B.y, B.w, B.h, 8, '#fff', COL.line);
      C.text('c1 →', B.x + B.w, B.y + B.h + 26, {size: 20, color: COL.mute, align: 'right'});
      C.text('c2 ↑', B.x - 12, B.y + 14, {size: 20, color: COL.mute, align: 'right'});
    };
    const dot = (p, color, r = 4, stroke) => {
      const g = C.g;
      g.beginPath(); g.arc(X(p[0]), Y(p[1]), r, 0, 2 * Math.PI); g.fillStyle = color; g.fill();
      if (stroke) { g.strokeStyle = stroke; g.lineWidth = 2.5; g.stroke(); }
    };
    // upto in 0..1 reveals the points progressively; colorOf(i) may vary the color per point.
    const dots = (pts, color, r = 4, upto = 1, colorOf) => {
      const n = Math.floor(pts.length * C.clamp(upto));
      for (let i = 0; i < n; i++) {
        const [x, y] = pts[i];
        if (x > xr[0] && x < xr[1] && y > yr[0] && y < yr[1]) dot(pts[i], colorOf ? colorOf(i) : color, r);
      }
    };
    const ell = (gm, k, color, lw = 2.5, dash) => {
      const [xs, ys] = ellipsePts(gm, k), g = C.g;
      g.save(); g.beginPath(); g.rect(B.x, B.y, B.w, B.h); g.clip();
      C.poly(xs.map(X), ys.map(Y), color, lw, dash);
      g.restore();
    };
    const scale = B.w / (xr[1] - xr[0]);
    return {B, X, Y, frame, dot, dots, ell, scale};
  }

  // Offscreen image of f(x, y) over the plane, drawn once and blitted every frame.
  function field(P, f, cell = 6) {
    const cv = document.createElement('canvas'); cv.width = P.B.w; cv.height = P.B.h;
    const g = cv.getContext('2d');
    for (let px = 0; px < P.B.w; px += cell) for (let py = 0; py < P.B.h; py += cell) {
      const x = -3 + (px + cell / 2) / P.B.w * 6.6, y = 2.4 - (py + cell / 2) / P.B.h * 5;
      g.fillStyle = f(x, y); g.fillRect(px, py, cell, cell);
    }
    return cv;
  }

  root.GMM = {rng, randn, cloud, A_PTS, I_PTS, pdf2, mixPdf, avgLL, initGmm, resp, emStep, emTrace, ellipsePts, npdf, plane, field};
})(window);
