// Evaluation toy for the clips: picking the epoch on test vs on dev, confidence intervals, and WER with and
// without text normalisation. Same simulation and numbers as demo/24-danh-gia.html.
(function (root) {
  const {rng, randn} = root.GMM;

  // One run of `ep` epoch scores around a true UA of 30, plus the mean of last / dev-picked / test-max over M runs.
  function selection(ep, sd, seed = 1, M = 2000) {
    const r = rng(seed), one = Array.from({length: ep}, () => 30 + sd * randn(r));
    let sMax = 0, sLast = 0, sDev = 0;
    for (let m = 0; m < M; m++) {
      const t = Array.from({length: ep}, () => 30 + sd * randn(r)), d = Array.from({length: ep}, () => 30 + sd * randn(r));
      sMax += Math.max(...t); sLast += t[ep - 1]; sDev += t[d.indexOf(Math.max(...d))];
    }
    return {one, max: sMax / M, last: sLast / M, dev: sDev / M};
  }
  // Half-width of the 95% interval, in percentage points.
  const ci = (n, p) => 196 * Math.sqrt(p * (1 - p) / n);

  const NUM = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}, ABBR = {mr: 'mister', mrs: 'missus', dr: 'doctor'};
  const norm = s => s.toLowerCase().replace(/[.,!?;:"]/g, ' ').split(/\s+/).filter(Boolean).map(w => NUM[w] || ABBR[w] || w);
  const raw = s => s.split(/\s+/).filter(Boolean);
  function ed(a, b) {
    const D = Array.from({length: a.length + 1}, (_, i) => Array.from({length: b.length + 1}, (_, j) => i === 0 ? j : j === 0 ? i : 0));
    for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) D[i][j] = Math.min(D[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1), D[i - 1][j] + 1, D[i][j - 1] + 1);
    return D[a.length][b.length];
  }
  const REF = "Mr. Smith has 2 cats, doesn't he?", HYP = "mister smith has two cats doesn't he";
  const wer = (ref, hyp, normed) => { const a = normed ? norm(ref) : raw(ref), b = normed ? norm(hyp) : raw(hyp); return ed(a, b) / a.length; };

  root.EVX = {selection, ci, norm, raw, wer, REF, HYP};
})(window);
