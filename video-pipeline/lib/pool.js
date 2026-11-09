// Frame-to-utterance toy for the clips: per-frame "anger evidence", mean vs attention pooling, and the
// illustrative layer weights per task. Same construction as the interactive page demo/19-frame-den-cau.html.
(function (root) {
  const {rng, randn} = root.GMM;
  const T = 75;
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const softmax = (x, temp = 1) => { const m = Math.max(...x), e = x.map(v => Math.exp((v - m) / temp)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); };

  // The angry stretch of `len` frames ends 6 frames before the end of the utterance.
  function evid(len) {
    const r = rng(9), s0 = Math.max(0, T - len - 6);
    return Array.from({length: T}, (_, t) => {
      const ang = t >= s0 && t < s0 + len;
      return {ang, v: clamp((ang ? 0.85 : 0.12) + 0.08 * randn(r), 0, 1)};
    });
  }
  function pool(len, sharp) {
    const E = evid(len), w = softmax(E.map(e => sharp * e.v));
    return {E, w, mean: E.reduce((s, e) => s + e.v, 0) / T, att: E.reduce((s, e, t) => s + w[t] * e.v, 0)};
  }

  // Illustrative layer weights (12 layers): a bump at the layer where the task's information is densest.
  const PEAK = {asr: [9.5, 2.2], ser: [6, 2.8], spk: [4, 2.2]};
  function layers(task) {
    const [mu, sd] = PEAK[task];
    return softmax(Array.from({length: 12}, (_, l) => -((l + 1 - mu) ** 2) / (2 * sd * sd) + (task === 'asr' && l === 11 ? -0.3 : 0)), 0.6);
  }

  root.POOL = {T, evid, pool, layers, softmax};
})(window);
