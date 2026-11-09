// VITS toy for the clips: Monotonic Alignment Search on a made-up score matrix, and sampled durations.
// Same construction as the interactive page demo/16-vits.html (true durations, score shape, log-normal sampling).
(function (root) {
  const {rng, randn} = root.GMM;
  const PH = ['x', 'i', 'n', 'ch', 'a', 'o'], TRUE_D = [7, 10, 6, 7, 14, 12], TF = TRUE_D.reduce((a, b) => a + b, 0);

  function scores(noise, seed) {
    const r = rng(seed), truth = TRUE_D.flatMap((d, i) => new Array(d).fill(i));
    return PH.map((_, i) => Array.from({length: TF}, (_, j) => -1.5 * Math.abs(i - truth[j]) ** 1.3 + noise * randn(r)));
  }

  // Returns the best path (phoneme index per frame) and the filled Q table, so a scene can show the fill.
  function mas(L) {
    const N = L.length, T = L[0].length, Q = L.map(() => new Array(T).fill(-Infinity));
    Q[0][0] = L[0][0];
    for (let j = 1; j < T; j++) for (let i = 0; i < N; i++) {
      const stay = Q[i][j - 1], move = i ? Q[i - 1][j - 1] : -Infinity;
      Q[i][j] = L[i][j] + Math.max(stay, move);
    }
    const path = new Array(T);
    let i = N - 1;
    for (let j = T - 1; j >= 0; j--) { path[j] = i; if (j && i && Q[i - 1][j - 1] >= Q[i][j - 1]) i--; }
    return {path, Q};
  }
  const dursOf = path => PH.map((_, i) => path.filter(p => p === i).length);

  // Five readings of "xin chào": a shared tempo factor per reading times a per-phoneme factor, both log-normal.
  function readings(ns, seed, n = 5) {
    const r = rng(seed), rows = [];
    for (let k = 0; k < n; k++) {
      const tempo = Math.exp(0.15 * ns * randn(r));
      rows.push(TRUE_D.map(d => Math.max(1, Math.round(d * tempo * Math.exp(0.3 * ns * randn(r))))));
    }
    return rows;
  }

  root.VITS = {PH, TRUE_D, TF, scores, mas, dursOf, readings};
})(window);
