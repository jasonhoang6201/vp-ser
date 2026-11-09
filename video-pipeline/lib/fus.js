// Fusion toy for the clips: per-source probabilities for three situations, late and gated fusion, and the
// "Thật à?" context example. Same numbers as the interactive page demo/21-fusion.html.
(function (root) {
  const CLS = ['trung tính', 'vui', 'giận', 'buồn'];
  const SCN = [
    {name: 'mỉa mai', a: [0.15, 0.1, 0.65, 0.1], t: [0.15, 0.7, 0.1, 0.05], truth: 2},
    {name: 'ASR chép sai', a: [0.2, 0.05, 0.1, 0.65], t: [0.6, 0.15, 0.1, 0.15], truth: 3},
    {name: 'hai nguồn đồng ý', a: [0.15, 0.6, 0.15, 0.1], t: [0.1, 0.8, 0.05, 0.05], truth: 1},
  ];
  const late = s => s.a.map((v, k) => (v + s.t[k]) / 2);
  const gated = (s, g) => s.a.map((v, k) => g * v + (1 - g) * s.t[k]);
  const argmax = p => p.indexOf(Math.max(...p));
  const CTX = [
    ['(không có ngữ cảnh)', [0.25, 0.3, 0.2, 0.25]],
    ['"Mình vừa đậu đại học!"', [0.08, 0.8, 0.04, 0.08]],
    ['"Con mèo của mình mất rồi."', [0.1, 0.03, 0.07, 0.8]],
    ['"Cậu làm hỏng hết rồi đấy."', [0.12, 0.03, 0.75, 0.1]],
  ];
  root.FUS = {CLS, SCN, late, gated, argmax, CTX};
})(window);
