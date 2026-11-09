// One-hidden-layer tanh network on the three-region toy data of demo/09-dnn-hmm.html, trained full-batch with
// momentum exactly as on that page, so the clip shows the same boundaries the reader trained by hand.
(function (root) {
  const {rng, randn} = root.GMM;
  const KC = 3;
  const softmax = a => { const m = Math.max(...a), e = a.map(v => Math.exp(v - m)), s = e.reduce((x, y) => x + y, 0); return e.map(v => v / s); };

  const DATA = (() => {
    const r = rng(5), pts = [];
    for (let j = 0; j < 330; j++) {
      const cls = j % 3, a = r() * 2 * Math.PI;
      const rad = cls === 0 ? 0.75 * Math.sqrt(r()) : 1.25 + 0.5 * r();
      let ang = a;
      if (cls === 1) ang = Math.PI / 2 + r() * Math.PI;
      if (cls === 2) ang = -Math.PI / 2 + r() * Math.PI;
      pts.push({x: [rad * Math.cos(ang) + 0.08 * randn(r), rad * Math.sin(ang) + 0.08 * randn(r)], y: cls});
    }
    return pts;
  })();

  function init(H, seed = 1) {
    const r = rng(seed * 97), w = (n, s) => Array.from({length: n}, () => randn(r) * s), nin = H || 2;
    return {H, W1: H ? w(H * 2, 1.0) : [], b1: H ? w(H, 0.5) : [], W2: w(KC * nin, H ? 0.5 : 0.3), b2: new Array(KC).fill(0), vel: null};
  }
  function fwd(N, x) {
    const h = N.H ? Array.from({length: N.H}, (_, j) => Math.tanh(N.W1[2 * j] * x[0] + N.W1[2 * j + 1] * x[1] + N.b1[j])) : x;
    const nin = h.length, z = Array.from({length: KC}, (_, k) => { let s = N.b2[k]; for (let j = 0; j < nin; j++) s += N.W2[k * nin + j] * h[j]; return s; });
    return {h, z, p: softmax(z)};
  }
  function step(N, lr = 0.3) {
    const g = {W1: N.W1.map(() => 0), b1: N.b1.map(() => 0), W2: N.W2.map(() => 0), b2: N.b2.map(() => 0)}, n = DATA.length;
    DATA.forEach(({x, y}) => {
      const {h, p} = fwd(N, x), nin = h.length, dz = p.map((v, k) => (v - (k === y ? 1 : 0)) / n);
      for (let k = 0; k < KC; k++) { g.b2[k] += dz[k]; for (let j = 0; j < nin; j++) g.W2[k * nin + j] += dz[k] * h[j]; }
      if (N.H) for (let j = 0; j < N.H; j++) {
        let dh = 0;
        for (let k = 0; k < KC; k++) dh += dz[k] * N.W2[k * N.H + j];
        const da = dh * (1 - h[j] * h[j]);
        g.b1[j] += da; g.W1[2 * j] += da * x[0]; g.W1[2 * j + 1] += da * x[1];
      }
    });
    if (!N.vel) N.vel = {W1: g.W1.map(() => 0), b1: g.b1.map(() => 0), W2: g.W2.map(() => 0), b2: g.b2.map(() => 0)};
    for (const key of ['W1', 'b1', 'W2', 'b2']) for (let i = 0; i < g[key].length; i++) {
      N.vel[key][i] = 0.9 * N.vel[key][i] - lr * g[key][i];
      N[key][i] += N.vel[key][i];
    }
  }
  function stats(N) {
    let loss = 0, ok = 0;
    DATA.forEach(({x, y}) => { const p = fwd(N, x).p; loss -= Math.log(p[y] + 1e-12); if (p.indexOf(Math.max(...p)) === y) ok++; });
    return {loss: loss / DATA.length, acc: ok / DATA.length};
  }
  const clone = N => JSON.parse(JSON.stringify(N));
  // Snapshots after each step count in `at` (ascending), so a scene can scrub through training without retraining.
  function train(H, at, seed = 1) {
    const N = init(H, seed), out = {};
    let done = 0;
    for (const k of at) { while (done < k) { step(N); done++; } out[k] = {net: clone(N), ...stats(N)}; }
    return out;
  }

  // Class map of net N on box B (offscreen, `cell` px squares), coloured by the winning class.
  function field(N, B, cols, r = 2.2, cell = 8) {
    const cv = document.createElement('canvas'); cv.width = B.w; cv.height = B.h;
    const g = cv.getContext('2d');
    for (let px = 0; px < B.w; px += cell) for (let py = 0; py < B.h; py += cell) {
      const x = [-r + (px + cell / 2) / B.w * 2 * r, r - (py + cell / 2) / B.h * 2 * r], p = fwd(N, x).p, k = p.indexOf(Math.max(...p));
      g.globalAlpha = 0.1 + 0.3 * p[k]; g.fillStyle = cols[k]; g.fillRect(px, py, cell, cell);
    }
    return cv;
  }

  root.NN = {KC, DATA, softmax, init, fwd, step, stats, train, field};
})(window);
