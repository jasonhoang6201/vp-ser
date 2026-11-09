// XTTS toy for the clips: vector quantisation of 2-D "frames" and a toy 8-code GPT sampled at a temperature.
// Same data, seeding and sampling as the interactive page demo/17-xtts.html.
(function (root) {
  const {rng, randn} = root.GMM;
  const PTS = (() => {
    const r = rng(21), out = [], cs = [[-1.4, 1], [1.2, 1.3], [0.3, -1.2], [-1.2, -0.8], [1.6, -0.4]];
    for (let j = 0; j < 160; j++) { const c = cs[j % 5]; out.push([c[0] + 0.45 * randn(r), c[1] + 0.4 * randn(r)]); }
    return out;
  })();
  const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;

  function kmeans(K) {
    const r = rng(K * 13 + 1);
    // k-means++ style seeding so that small codebooks spread out instead of collapsing into one cluster.
    let C2 = [PTS[Math.floor(r() * PTS.length)].slice()];
    while (C2.length < K) {
      const w = PTS.map(p => Math.min(...C2.map(c => d2(p, c)))), s = w.reduce((a, b) => a + b, 0);
      let u = r() * s, j = 0;
      while (u > w[j]) { u -= w[j]; j++; }
      C2.push(PTS[Math.min(j, PTS.length - 1)].slice());
    }
    let A = [];
    for (let it = 0; it < 40; it++) {
      A = PTS.map(p => { let b = 0; C2.forEach((c, k) => { if (d2(p, c) < d2(p, C2[b])) b = k; }); return b; });
      C2 = C2.map((c, k) => { const m = PTS.filter((_, j) => A[j] === k); return m.length ? [m.reduce((s, p) => s + p[0], 0) / m.length, m.reduce((s, p) => s + p[1], 0) / m.length] : c; });
    }
    const err = PTS.reduce((s, p, j) => s + d2(p, C2[A[j]]), 0) / PTS.length;
    return {C2, A, err};
  }

  const NT = 8, LOGITS = (() => {
    const r = rng(4);
    return Array.from({length: NT}, (_, a) => Array.from({length: NT}, (_, b) => (b === (a + 1) % NT ? 2.2 : b === a ? 1.6 : b === (a + 3) % NT ? 1.0 : 0) + 0.6 * randn(r)));
  })();
  const softmax = (x, T) => { const m = Math.max(...x), e = x.map(v => Math.exp((v - m) / T)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); };

  // Five sequences of L tokens; `distinct` is the mean number of different (code, next code) pairs per sequence.
  function generate(T, seed = 1, L = 60) {
    const r = rng(seed * 7), seqs = [];
    let distinct = 0;
    for (let n = 0; n < 5; n++) {
      const seq = [Math.floor(r() * NT)];
      while (seq.length < L) {
        const p = softmax(LOGITS[seq[seq.length - 1]], T);
        let u = r(), k = 0;
        while (u > p[k] && k < NT - 1) { u -= p[k]; k++; }
        seq.push(k);
      }
      seqs.push(seq);
      distinct += new Set(seq.slice(1).map((v, i) => seq[i] * NT + v)).size;
    }
    return {seqs, distinct: distinct / 5};
  }

  root.XTTS = {PTS, kmeans, NT, LOGITS, softmax, generate};
})(window);
