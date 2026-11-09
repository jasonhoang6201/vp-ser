// The 60 synthetic frames of "a s i" and k-means on them, same generator as demo/12-ssl-whisper.html.
(function (root) {
  const {rng, randn} = root.GMM;
  const SEGS = [['a', [-1.2, 0.8], 22], ['s', [1.4, 1.2], 14], ['i', [0.6, -1.3], 24]];
  const FR = (() => {
    const r = rng(12), out = [];
    SEGS.forEach(([lab, c, n], si) => {
      for (let j = 0; j < n; j++) {
        const u = j / (n - 1), drift = si === 2 ? [0.5 * u - 0.25, 0.35 * u] : [0, 0];
        // Edges drift toward the neighbouring sound, which is what makes clusters straddle boundaries.
        const prev = si > 0 && u < 0.2 ? SEGS[si - 1][1] : null, nxt = si < 2 && u > 0.8 ? SEGS[si + 1][1] : null, pull = prev ? (0.2 - u) * 2 : nxt ? (u - 0.8) * 2 : 0, tgt = prev || nxt || c;
        out.push({lab, si, p: [c[0] + drift[0] + pull * (tgt[0] - c[0]) * 0.5 + 0.28 * randn(r), c[1] + drift[1] + pull * (tgt[1] - c[1]) * 0.5 + 0.28 * randn(r)]});
      }
    });
    return out;
  })();
  const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
  const assign = C => FR.map(f => { let b = 0; C.forEach((c, k) => { if (d2(f.p, c) < d2(f.p, C[b])) b = k; }); return b; });
  // Every iterate until assignments stop changing: [{cent, asg}], first entry is the random start.
  function kmeans(K, seed) {
    const r = rng(seed * 31);
    let C = Array.from({length: K}, () => FR[Math.floor(r() * FR.length)].p.slice()), A = assign(C);
    const out = [{cent: C, asg: A}];
    for (let it = 0; it < 30; it++) {
      C = C.map((c, k) => { const m = FR.filter((_, j) => A[j] === k); return m.length ? [m.reduce((s, f) => s + f.p[0], 0) / m.length, m.reduce((s, f) => s + f.p[1], 0) / m.length] : c; });
      const B = assign(C);
      out.push({cent: C, asg: B});
      if (B.join() === A.join()) break;
      A = B;
    }
    return out;
  }
  root.SSL = {SEGS, FR, kmeans};
})(window);
