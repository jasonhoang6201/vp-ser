// Keyword "emotion guesser" toy for the clips (not RoBERTa) and word error rate, as in demo/20-text-bert.html.
(function (root) {
  const EXS = [['I love this so much', 'I love these so much'], ['I hate this so much', 'I ate this so much'], ['I am not happy with you', 'I am happy with you']];
  const CLS = ['trung tính', 'vui', 'giận', 'buồn'];
  const LEX = {love: [0, 2.2, 0, 0], happy: [0, 2.2, 0, 0], great: [0, 1.6, 0, 0], glad: [0, 1.8, 0, 0], hate: [0, 0, 2.4, 0], stupid: [0, 0, 1.8, 0], angry: [0, 0, 2.2, 0], damn: [0, 0, 1.6, 0], sad: [0, 0, 0, 2.2], sorry: [0, 0, 0, 1.2], miss: [0, 0, 0, 1.4], cry: [0, 0, 0, 2]};
  const softmax = x => { const m = Math.max(...x), e = x.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); };

  function score(s) {
    const w = s.toLowerCase().replace(/[^a-z' ]/g, ' ').split(/\s+/).filter(Boolean), z = [1.0, 0, 0, 0];
    let neg = false;
    w.forEach(t => {
      if (t === 'not' || t === "don't" || t === 'never') { neg = true; return; }
      const v = LEX[t];
      if (!v) return;
      // A preceding negation flips positive words to anger/sadness, a crude stand-in for what context gives a real model.
      if (neg) { z[2] += v[1] * 0.6; z[3] += v[1] * 0.6; z[1] -= v[1] * 0.5; neg = false; }
      else v.forEach((x, k) => z[k] += x);
    });
    return softmax(z);
  }
  function wer(ref, hyp) {
    const a = ref.toLowerCase().split(/\s+/).filter(Boolean), b = hyp.toLowerCase().split(/\s+/).filter(Boolean);
    const D = Array.from({length: a.length + 1}, (_, i) => Array.from({length: b.length + 1}, (_, j) => i === 0 ? j : j === 0 ? i : 0));
    for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) D[i][j] = Math.min(D[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1), D[i - 1][j] + 1, D[i][j - 1] + 1);
    return a.length ? D[a.length][b.length] / a.length : 0;
  }
  const argmax = p => p.indexOf(Math.max(...p));

  root.BERT = {EXS, CLS, LEX, score, wer, argmax};
})(typeof window !== 'undefined' ? window : globalThis);
