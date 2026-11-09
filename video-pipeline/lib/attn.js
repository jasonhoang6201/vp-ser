// Data for the attention clips, same generators as demo/11-attention.html: ten 2-D keys, the "xin chào" decoder
// attention rows, plus illustrative self-attention heads (patterns, not a trained model).
(function (root) {
  const {rng, randn} = root.GMM, softmax = root.NN.softmax;

  const KEYS = (() => { const r = rng(8); return Array.from({length: 10}, (_, j) => { const a = j * 0.63 + 0.3 * randn(r), l = 0.6 + 0.6 * r(); return [l * Math.cos(a), l * Math.sin(a)]; }); })();
  function weights(angleDeg, len) {
    const q = [len * Math.cos(angleDeg * Math.PI / 180), len * Math.sin(angleDeg * Math.PI / 180)];
    const sc = KEYS.map(k => (q[0] * k[0] + q[1] * k[1]) / Math.SQRT2);
    return {q, sc, w: softmax(sc)};
  }

  const OUT = ['x', 'i', 'n', ' ', 'c', 'h', 'à', 'o', '<eos>'], CENT = [9, 13, 17, 22, 27, 30, 34, 39, 47], TF = 50;
  const attnRow = i => { const r = rng(100 + i); return softmax(Array.from({length: TF}, (_, t) => -((t - CENT[i]) ** 2) / 6 + 0.4 * randn(r))); };

  // 36 frames of "a s i" (silence at the edges); each head returns a row-normalised 36×36 matrix.
  const SEG = Array.from({length: 36}, (_, t) => (t < 4 || t >= 33 ? 0 : t < 14 ? 1 : t < 22 ? 2 : 3));
  function head(kind) {
    return SEG.map((si, i) => softmax(SEG.map((sj, j) => {
      if (kind === 0) return -((i - j) ** 2) / 3;
      if (kind === 1) return si === sj ? 2.5 : 0;
      return sj !== 0 ? 1.5 : -2;
    })));
  }

  root.ATT = {KEYS, weights, OUT, CENT, TF, attnRow, SEG, head};
})(window);
