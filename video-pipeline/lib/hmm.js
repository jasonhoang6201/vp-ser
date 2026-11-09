// Left-to-right HMM over the word "mai" for the HMM clips: synthetic per-frame GMM scores, per-frame argmax,
// Viterbi, forward and forward-backward. One state per phone, as in demo/07-hmm.html.
(function (root) {
  const {rng, randn} = root.GMM;
  const STATES = ['m', 'a', 'i'];
  const TRUTH = [...Array(10).fill(0), ...Array(18).fill(1), ...Array(12).fill(2)];
  const lse = (a, b) => (a === -Infinity ? b : b === -Infinity ? a : Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b))));

  // Log emission score of every frame for every state: the true state sits `gap` above the others, plus noise.
  function scores(seed, noise = 1, gap = 1.6, truth = TRUTH, nS = 3) {
    const r = rng(seed);
    return truth.map(s => Array.from({length: nS}, (_, k) => (k === s ? 0 : -gap) + noise * randn(r)));
  }
  const argmax = v => v.indexOf(Math.max(...v));
  const perFrame = E => E.map(argmax);

  // Must start in state 0 and end in the last state; each frame either stays (log p) or moves on (log 1-p).
  function viterbi(E, p) {
    const T = E.length, N = E[0].length, ls = Math.log(p), lm = Math.log(1 - p);
    const V = [E[0].map((e, s) => (s === 0 ? e : -Infinity))], B = [E[0].map(() => -1)];
    for (let t = 1; t < T; t++) {
      V.push([]); B.push([]);
      for (let s = 0; s < N; s++) {
        const stay = V[t - 1][s] + ls, move = s > 0 ? V[t - 1][s - 1] + lm : -Infinity;
        V[t].push(Math.max(stay, move) + E[t][s]); B[t].push(stay >= move ? s : s - 1);
      }
    }
    const path = new Array(T); path[T - 1] = N - 1;
    for (let t = T - 1; t > 0; t--) path[t - 1] = B[t][path[t]];
    return {V, B, path, best: V[T - 1][N - 1]};
  }
  function forward(E, p) {
    const T = E.length, N = E[0].length, ls = Math.log(p), lm = Math.log(1 - p);
    const A = [E[0].map((e, s) => (s === 0 ? e : -Infinity))];
    for (let t = 1; t < T; t++) A.push(E[t].map((e, s) => lse(A[t - 1][s] + ls, s > 0 ? A[t - 1][s - 1] + lm : -Infinity) + e));
    return {A, total: A[T - 1][N - 1]};
  }
  // Posterior P(state s at frame t | all frames).
  function posterior(E, p) {
    const T = E.length, N = E[0].length, ls = Math.log(p), lm = Math.log(1 - p), {A, total} = forward(E, p);
    const Bk = new Array(T); Bk[T - 1] = E[0].map((_, s) => (s === N - 1 ? 0 : -Infinity));
    for (let t = T - 2; t >= 0; t--) Bk[t] = E[0].map((_, s) => lse(Bk[t + 1][s] + ls + E[t + 1][s], s < N - 1 ? Bk[t + 1][s + 1] + lm + E[t + 1][s + 1] : -Infinity));
    return A.map((a, t) => a.map((v, s) => Math.exp(v + Bk[t][s] - total)));
  }
  // Durations drawn from the self-loop model: geometric with stay probability p, at least one frame.
  function sampleDur(seed, p, n = 3) {
    const r = rng(seed);
    return Array.from({length: n}, () => { let d = 1; while (r() < p) d++; return d; });
  }

  const stateCol = s => { const C = root.Clip.COL; return [C.ok, C.acc, C.i][s]; };

  // One row of per-frame labels as small boxes; `upto` reveals frames left to right, `wrong` marks mismatches.
  function row(x, y, labels, o = {}) {
    const C = root.Clip, w = o.w || 38, h = o.h || 44, n = Math.floor(labels.length * C.clamp(o.upto == null ? 1 : o.upto));
    for (let t = 0; t < n; t++) {
      const s = labels[t], bad = o.truth && o.truth[t] !== s, c = stateCol(s);
      C.rrect(x + t * w, y, w - 4, h, 5, bad ? C.COL.hlSoft : c + '22', bad ? C.COL.hl : c, bad ? 3 : 1.5);
      C.text(STATES[s], x + t * w + (w - 4) / 2, y + h / 2, {size: o.size || 22, weight: 700, color: bad ? C.COL.hl : c, align: 'center'});
    }
  }

  // Trellis: rows = states, columns = frames, cell shade = emission score. Returns cell centre helpers.
  function trellis(B, E, o = {}) {
    const C = root.Clip, T = E.length, N = E[0].length, cw = B.w / T, ch = B.h / N;
    const lo = o.lo == null ? -4 : o.lo, hi = o.hi == null ? 1.5 : o.hi, cols = o.cols == null ? T : o.cols;
    for (let t = 0; t < T; t++) for (let s = 0; s < N; s++) {
      const v = C.clamp((E[t][s] - lo) / (hi - lo));
      C.rrect(B.x + t * cw + 1, B.y + s * ch + 1, cw - 2, ch - 2, 3, t < cols ? C.heat(0.15 + 0.5 * v) : '#f4f3f0');
    }
    for (let s = 0; s < N; s++) C.text((o.names || STATES)[s], B.x - 18, B.y + (s + 0.5) * ch, {size: 28, weight: 700, color: stateCol(s), align: 'right'});
    const cx = t => B.x + (t + 0.5) * cw, cy = s => B.y + (s + 0.5) * ch;
    return {cx, cy, cw, ch};
  }
  function path(tr, p, color, lw = 4, upto = 1) {
    const C = root.Clip;
    C.poly(p.map((_, t) => tr.cx(t)), p.map(s => tr.cy(s)), color, lw, null, upto);
  }

  // State box (120×90) with an optional self-loop drawn above it.
  function stateBox(x, y, s, o = {}) {
    const C = root.Clip, c = o.color || stateCol(s), g = C.g;
    C.rrect(x, y, 120, 90, 14, c + '22', c, 3);
    C.text(o.label || STATES[s], x + 60, y + 45, {size: o.size || 44, weight: 700, color: c, align: 'center'});
    if (o.loop) {
      g.beginPath(); g.arc(x + 60, y - 22, 26, Math.PI * 0.85, Math.PI * 2.15); g.strokeStyle = c; g.lineWidth = 2.5; g.stroke();
      C.arrow(x + 82, y - 10, x + 84, y - 2, c, 2.5, 10);
    }
  }

  root.HMM = {STATES, TRUTH, lse, scores, argmax, perFrame, viterbi, forward, posterior, sampleDur, stateCol, row, trellis, path, stateBox};
})(window);
