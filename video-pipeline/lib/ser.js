// SER toy for the clips: the MELD test class counts, a confusion matrix that drifts towards "neutral",
// and the valence-arousal positions of the emotion classes. Same counts and drift as demo/18-ser-co-ban.html.
(function (root) {
  const CL = ['trung tính', 'vui', 'ngạc nhiên', 'giận', 'buồn', 'ghê tởm', 'sợ'];
  const CNT = [1256, 402, 281, 345, 208, 68, 50], K = CL.length, N = CNT.reduce((a, b) => a + b, 0);

  // bias 0: every class 50% right, the rest spread evenly; bias 1: everything predicted as neutral.
  function cm(bias) {
    const M = CNT.map((n, c) => Array.from({length: K}, (_, d) => d === c ? 0.5 : 0.5 / (K - 1)).map((v, d) => ((1 - bias) * v + (d === 0 ? bias : 0)) * n));
    const col = Array.from({length: K}, (_, d) => M.reduce((s, r) => s + r[d], 0));
    const rec = M.map((r, c) => r[c] / CNT[c]), prec = M.map((_, c) => col[c] ? M[c][c] / col[c] : 0);
    const f1 = rec.map((r, c) => r + prec[c] ? 2 * r * prec[c] / (r + prec[c]) : 0);
    const WA = M.reduce((s, r, c) => s + r[c], 0) / N, UA = rec.reduce((a, b) => a + b, 0) / K;
    const mF1 = f1.reduce((a, b) => a + b, 0) / K, wF1 = f1.reduce((s, f, c) => s + f * CNT[c], 0) / N;
    return {M, rec, prec, f1, WA, UA, mF1, wF1};
  }

  // [label, valence, arousal] in [-1, 1]; a sketch of the usual circumplex placement, not measured data.
  const VA = [
    ['giận', -0.6, 0.75], ['sợ', -0.75, 0.5], ['ghê tởm', -0.7, 0.2], ['vui', 0.7, 0.55],
    ['ngạc nhiên', 0.25, 0.85], ['buồn', -0.6, -0.55], ['trung tính', 0, 0], ['thư thái', 0.6, -0.5],
  ];

  root.SER = {CL, CNT, K, N, cm, VA};
})(window);
