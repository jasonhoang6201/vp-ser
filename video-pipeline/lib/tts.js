// Phase experiments for the TTS/vocoder clips, same signal and settings as demo/13-tts-vocoder.html: a synthetic
// 0.9 s "a" at 16 kHz, Hann 25 ms / hop 10 ms / FFT 512, full 257-bin magnitude (no Mel).
(function (root) {
  const fft = root.DSP.fft, rng = root.GMM.rng;
  const SR = 16000, WIN = 400, HOP = 160, NFFT = 512, LEN = Math.round(0.9 * SR);
  const hann = Float64Array.from({length: WIN}, (_, n) => 0.5 - 0.5 * Math.cos(2 * Math.PI * n / WIN));
  const gauss = (f, c, w) => Math.exp(-0.5 * ((f - c) / w) ** 2);
  const env = f => 0.03 + gauss(f, 750, 130) + 0.7 * gauss(f, 1200, 160) + 0.25 * gauss(f, 2600, 220);
  const SIG = (() => {
    const x = new Float32Array(LEN);
    let ph = 0;
    for (let n = 0; n < LEN; n++) {
      const t = n / SR, f0 = 130 + 25 * t + 3 * Math.sin(2 * Math.PI * 5 * t);
      ph += 2 * Math.PI * f0 / SR;
      let s = 0;
      for (let k = 1; k * f0 < 4000; k++) s += env(k * f0) / Math.sqrt(k) * Math.sin(k * ph + 0.3 * k * k);
      x[n] = s;
    }
    const fade = 800;
    let mx = 0;
    for (let n = 0; n < LEN; n++) { x[n] *= Math.min(1, n / fade, (LEN - n) / fade); mx = Math.max(mx, Math.abs(x[n])); }
    for (let n = 0; n < LEN; n++) x[n] *= 0.5 / mx;
    return x;
  })();
  const NFR = Math.floor((LEN - WIN) / HOP) + 1;
  function stft(x) {
    const out = [];
    for (let f = 0; f < NFR; f++) {
      const re = new Float64Array(NFFT), im = new Float64Array(NFFT);
      for (let n = 0; n < WIN; n++) re[n] = x[f * HOP + n] * hann[n];
      fft(re, im);
      out.push({re, im});
    }
    return out;
  }
  function istft(S) {
    const y = new Float64Array(LEN), wsum = new Float64Array(LEN);
    S.forEach(({re, im}, f) => {
      const r = Float64Array.from(re), i = Float64Array.from(im, v => -v);
      fft(r, i);
      for (let n = 0; n < WIN; n++) { y[f * HOP + n] += r[n] / NFFT * hann[n]; wsum[f * HOP + n] += hann[n] * hann[n]; }
    });
    const out = new Float32Array(LEN);
    for (let n = 0; n < LEN; n++) out[n] = wsum[n] > 1e-3 ? y[n] / wsum[n] : 0;
    return out;
  }
  // Rebuild a conjugate-symmetric spectrum from per-bin magnitude and phase.
  function spec(mag, phase) {
    const R = new Float64Array(NFFT), I = new Float64Array(NFFT);
    for (let k = 0; k <= NFFT / 2; k++) {
      R[k] = mag[k] * Math.cos(phase[k]); I[k] = mag[k] * Math.sin(phase[k]);
      if (k > 0 && k < NFFT / 2) { R[NFFT - k] = R[k]; I[NFFT - k] = -I[k]; }
    }
    I[0] = 0; I[NFFT / 2] = 0;
    return {re: R, im: I};
  }
  const S0 = stft(SIG);
  const MAG = S0.map(({re, im}) => Float64Array.from({length: NFFT / 2 + 1}, (_, k) => Math.hypot(re[k], im[k])));
  const PH = S => S.map(({re, im}) => Float64Array.from({length: NFFT / 2 + 1}, (_, k) => Math.atan2(im[k], re[k])));
  const rr = rng(77), RANDPH = S0.map(() => Float64Array.from({length: NFFT / 2 + 1}, () => 2 * Math.PI * rr()));
  const zero = new Float64Array(NFFT / 2 + 1);
  const VERS = {
    orig: SIG,
    true: istft(MAG.map((m, f) => spec(m, PH(S0)[f]))),
    zero: istft(MAG.map(m => spec(m, zero))),
    rand: istft(MAG.map((m, f) => spec(m, RANDPH[f]))),
  };
  const specErr = x => { const S = stft(x); let a = 0, b = 0; S.forEach(({re, im}, f) => { for (let k = 0; k <= NFFT / 2; k++) { const m = Math.hypot(re[k], im[k]); a += (m - MAG[f][k]) ** 2; b += MAG[f][k] ** 2; } }); return Math.sqrt(a / b); };
  // Griffin-Lim from the random-phase start; keeps only the iterations a scene asks for.
  function griffinLim(keep, iters = 100) {
    let x = VERS.rand;
    const out = {0: {x, err: specErr(x)}};
    for (let it = 1; it <= iters; it++) {
      const S = stft(x);
      x = istft(S.map(s => spec(MAG[S.indexOf(s)], PH([s])[0])));
      if (keep.includes(it)) out[it] = {x, err: specErr(x)};
    }
    return out;
  }
  const rms = x => Math.sqrt(x.reduce((s, v) => s + v * v, 0) / x.length);

  root.TTS = {SR, LEN, SIG, VERS, griffinLim, specErr, rms};
})(window);
