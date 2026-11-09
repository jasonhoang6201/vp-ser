// CTC helpers for the CTC clips: collapse rule, path enumeration and counting on the blank-extended lattice, and the
// synthetic 30-frame "mai" posteriors of demo/10-ctc.html (same generator, so the clip matches the page).
(function (root) {
  const {rng, randn} = root.GMM, softmax = root.NN.softmax;
  const collapse = s => s.replace(/(.)\1+/g, '$1');
  const reduce = s => collapse(s).replace(/-/g, '');

  function lattice(y) { const L = ['-']; for (const ch of y) L.push(ch, '-'); return L; }
  // A skip over a blank is only allowed between two different letters.
  const canSkip = (L, s) => s >= 2 && L[s] !== '-' && L[s] !== L[s - 2];

  function count(y, T) {
    const L = lattice(y), S = L.length, C = Array.from({length: T}, () => new Array(S).fill(0));
    C[0][0] = 1; if (S > 1) C[0][1] = 1;
    for (let t = 1; t < T; t++) for (let s = 0; s < S; s++) C[t][s] = C[t - 1][s] + (s ? C[t - 1][s - 1] : 0) + (canSkip(L, s) ? C[t - 1][s - 2] : 0);
    return y.length === 0 ? 1 : C[T - 1][S - 1] + (S > 1 ? C[T - 1][S - 2] : 0);
  }
  // Every path as the list of lattice states it visits.
  function paths(y, T, limit = 1000) {
    const L = lattice(y), S = L.length, out = [];
    function dfs(t, s, acc) {
      if (out.length >= limit) return;
      acc.push(s);
      if (t === T - 1) { if (s >= S - 2) out.push(acc.slice()); }
      else for (const ns of [s, s + 1, s + 2]) if (ns < S && (ns !== s + 2 || canSkip(L, ns))) dfs(t + 1, ns, acc);
      acc.pop();
    }
    dfs(0, 0, []); if (S > 1) dfs(0, 1, []);
    return {L, out};
  }

  const SYM = ['-', 'm', 'n', 'a', 'i'], T5 = 30, SPK = [[6, 'm'], [7, 'm'], [14, 'a'], [15, 'a'], [16, 'a'], [23, 'i']];
  function post(noise, seed) {
    const r = rng(seed);
    return Array.from({length: T5}, (_, t) => {
      const z = SYM.map(() => noise * randn(r)), sp = SPK.find(p => p[0] === t);
      if (sp) { z[SYM.indexOf(sp[1])] += 5; if (sp[1] === 'm') z[2] += 2.5; } else z[0] += 4;
      return softmax(z);
    });
  }
  const greedy = P => P.map(p => SYM[p.indexOf(Math.max(...p))]).join('');

  // One symbol per box; blanks drawn grey. `hl` marks indices to outline, `upto` reveals left to right.
  function row(x, y, str, o = {}) {
    const C = root.Clip, w = o.w || 44, h = o.h || 52, n = Math.ceil(str.length * C.clamp(o.upto == null ? 1 : o.upto));
    for (let k = 0; k < n; k++) {
      const ch = str[k], blank = ch === '-', hl = o.hl && o.hl.includes(k), c = blank ? C.COL.line2 : (o.color || C.COL.acc);
      C.rrect(x + k * w, y, w - 4, h, 6, hl ? C.COL.hlSoft : blank ? C.COL.soft : c + '1c', hl ? C.COL.hl : c, hl ? 3 : 1.5);
      C.text(blank ? '−' : ch, x + k * w + (w - 4) / 2, y + h / 2, {size: o.size || 26, weight: 700, color: hl ? C.COL.hl : blank ? C.COL.mute : c, align: 'center', mono: true});
    }
    return x + str.length * w;
  }

  root.CTC = {collapse, reduce, lattice, canSkip, count, paths, SYM, T5, post, greedy, row};
})(window);
