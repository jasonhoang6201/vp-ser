// Same synthetic frames and parameters as demo/05-mfcc.html, so numbers in the clips match the page.
(function (root) {
  const SR = 16000, WIN = 400, NFFT = 512, BINS = NFFT / 2 + 1, BIN_HZ = SR / NFFT, FMAX = 8000, NM = 40;
  const hz2mel = f => 2595 * Math.log10(1 + f / 700), mel2hz = m => 700 * (10 ** (m / 2595) - 1), MMAX = hz2mel(FMAX);

  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let b = n >> 1;
      for (; j & b; b >>= 1) j ^= b;
      j ^= b;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang), h = len >> 1;
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < h; k++) {
          const p = i + k, q = p + h;
          const vr = re[q] * cr - im[q] * ci, vi = re[q] * ci + im[q] * cr;
          re[q] = re[p] - vr; im[q] = im[p] - vi;
          re[p] += vr; im[p] += vi;
          const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
        }
      }
    }
  }

  const hann = Float64Array.from({length: WIN}, (_, n) => 0.5 - 0.5 * Math.cos(2 * Math.PI * n / (WIN - 1)));
  const gauss = (f, c, w) => Math.exp(-0.5 * ((f - c) / w) ** 2);
  const envA = f => 0.06 + gauss(f, 800, 160) + 0.8 * gauss(f, 1250, 180) + 0.25 * gauss(f, 2600, 220);
  const envI = f => 0.06 + gauss(f, 300, 110) + 0.7 * gauss(f, 2300, 220) + 0.4 * gauss(f, 3000, 220);
  const ENV = {a: envA, i: envI};

  const EDGES = Array.from({length: NM + 2}, (_, i) => mel2hz(i / (NM + 1) * MMAX));
  const BANK = Array.from({length: NM}, (_, i) => {
    const a = EDGES[i], b = EDGES[i + 1], c = EDGES[i + 2];
    return Float64Array.from({length: BINS}, (_, k) => { const f = k * BIN_HZ; return f <= a || f >= c ? 0 : f <= b ? (f - a) / (b - a) : (c - f) / (c - b); });
  });
  const melOf = p => BANK.map(w => { let s = 0; for (let k = 0; k < BINS; k++) s += w[k] * p[k]; return s; });
  // The floor keeps empty filters (no harmonic inside) from going to minus infinity after the log.
  const toDb = (a, floor) => a.map(v => 10 * Math.log10(Math.max(v, floor)));
  const dct = x => { const n = x.length; return Array.from({length: n}, (_, k) => { let s = 0; for (let j = 0; j < n; j++) s += x[j] * Math.cos(Math.PI * k * (j + 0.5) / n); return s * Math.sqrt((k ? 2 : 1) / n); }); };
  const idct = (c, K, n) => Array.from({length: n}, (_, j) => { let s = 0; for (let k = 0; k < K; k++) s += c[k] * Math.sqrt((k ? 2 : 1) / n) * Math.cos(Math.PI * k * (j + 0.5) / n); return s; });

  function framePow(env, f0) {
    const re = new Float64Array(NFFT), im = new Float64Array(NFFT);
    for (let n = 0; n < WIN; n++) {
      let v = 0;
      // Spread-out starting phases avoid one sharp pulse per period, which a 25 ms window would make uneven.
      for (let k = 1; k * f0 < 7800; k++) v += env(k * f0) * Math.sin(2 * Math.PI * k * f0 * n / SR + k * k * 0.7) / Math.sqrt(k);
      re[n] = v * hann[n];
    }
    fft(re, im);
    return Float64Array.from({length: BINS}, (_, k) => re[k] * re[k] + im[k] * im[k]);
  }
  function frameInfo(env, f0) {
    const pow = framePow(env, f0), mel = melOf(pow), ldb = toDb(mel, Math.max(...mel) * 1e-6);
    return {pow, mel, ldb, cc: dct(ldb)};
  }
  const dist = (a, b) => Math.sqrt(a.reduce((s, v, n) => s + (v - b[n]) ** 2, 0));
  function rng(seed) { let s = seed; return () => (s = (s * 16807) % 2147483647) / 2147483647; }
  // Random vowel-like frames: three formants anywhere in a plausible range, any f0 between 100 and 300 Hz.
  function randomFrames(n, seed = 11) {
    const r = rng(seed), out = [];
    for (let j = 0; j < n; j++) {
      const F1 = 250 + 600 * r(), F2 = Math.max(F1 + 300, 800 + 1600 * r()), F3 = 2400 + 700 * r(), f0 = 100 + 200 * r();
      const env = f => 0.06 + gauss(f, F1, 130) + 0.8 * gauss(f, F2, 190) + 0.3 * gauss(f, F3, 220);
      out.push({F1, F2, F3, f0, ...frameInfo(env, f0)});
    }
    return out;
  }
  function corr(xs, ys) {
    const n = xs.length, mx = xs.reduce((a, b) => a + b) / n, my = ys.reduce((a, b) => a + b) / n;
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; syy += (ys[i] - my) ** 2; }
    return sxy / Math.sqrt(sxx * syy);
  }

  root.DSP = {SR, WIN, NFFT, BINS, BIN_HZ, FMAX, NM, fft, hz2mel, mel2hz, EDGES, BANK, gauss, envA, envI, ENV, melOf, toDb, dct, idct,
    framePow, frameInfo, dist, rng, randomFrames, corr};
})(typeof window !== 'undefined' ? window : globalThis);
