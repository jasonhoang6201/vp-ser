// Multi-task toy for the clips: two quadratic losses (ASR, SER) over a 2-D "parameter space", gradient
// descent on alpha · ASR + (1 − alpha) · SER, optionally with PCGrad. Same numbers as demo/22-multitask.html.
(function (root) {
  const A = [1.6, 1.3], SA = [1, 2.2], Bm = [-1.3, 1.5], SB = [1.6, 0.6], START = [0.1, -1.6];
  const lossA = p => 0.5 * (SA[0] * (p[0] - A[0]) ** 2 + SA[1] * (p[1] - A[1]) ** 2);
  const lossB = p => 0.5 * (SB[0] * (p[0] - Bm[0]) ** 2 + SB[1] * (p[1] - Bm[1]) ** 2);
  const gA = p => [SA[0] * (p[0] - A[0]), SA[1] * (p[1] - A[1])], gB = p => [SB[0] * (p[0] - Bm[0]), SB[1] * (p[1] - Bm[1])];
  const dotv = (u, v) => u[0] * v[0] + u[1] * v[1], norm = u => Math.hypot(u[0], u[1]);
  const cos = p => { const a = gA(p), b = gB(p); return dotv(a, b) / (norm(a) * norm(b) + 1e-12); };

  function step(p, al, pc) {
    let a = gA(p).map(v => al * v), b = gB(p).map(v => (1 - al) * v);
    // PCGrad: when the two gradients conflict, project each onto the plane normal to the other before summing.
    if (pc && dotv(a, b) < 0) {
      const a0 = a.slice(), b0 = b.slice();
      if (norm(b0) > 1e-9) a = a0.map((v, i) => v - dotv(a0, b0) / dotv(b0, b0) * b0[i]);
      if (norm(a0) > 1e-9) b = b0.map((v, i) => v - dotv(b0, a0) / dotv(a0, a0) * a0[i]);
    }
    const lr = 0.08;
    return [p[0] - lr * (a[0] + b[0]), p[1] - lr * (a[1] + b[1])];
  }
  function path(al, pc = false, n = 120) {
    const out = [START.slice()];
    for (let k = 0; k < n; k++) out.push(step(out[out.length - 1], al, pc));
    return out;
  }

  root.MTL = {A, SA, Bm, SB, START, lossA, lossB, gA, gB, cos, path, norm};
})(typeof window !== 'undefined' ? window : globalThis);
