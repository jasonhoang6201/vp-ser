const C = {ink: '#37352f', mute: '#787774', line: '#dcdbd7', grey: '#9b9a97', soft: '#efeeeb', frame: '#d9730d', hl: '#d44c47', acc: '#337ea9', ok: '#448361', i: '#9065b0'};
const PALETTE = [C.acc, C.frame, C.ok, C.i, C.hl, C.grey];
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const r0 = v => v.toFixed(0), r1 = v => v.toFixed(1), r2 = v => v.toFixed(2);
const alpha = (hex, a) => hex + Math.round(a * 255).toString(16).padStart(2, '0');

const PAGES = [
  ['00-lo-trinh.html', 'Lộ trình'],
  ['01-spectrogram.html', 'Spectrogram'],
  ['02-f0-formant.html', 'f0 và formant'],
  ['03-cong-dan.html', 'Cộng dần họa âm'],
  ['04-mel.html', 'Mel'],
  ['05-mfcc.html', 'MFCC'],
  ['06-gmm.html', 'GMM'],
  ['07-hmm.html', 'HMM'],
  ['08-he-asr-co-dien.html', 'Hệ ASR cổ điển'],
  ['09-dnn-hmm.html', 'Mạng nơ-ron và DNN-HMM'],
  ['10-ctc.html', 'CTC'],
  ['11-attention.html', 'Attention, Transformer, Conformer'],
  ['12-ssl-whisper.html', 'wav2vec 2.0, HuBERT, Whisper'],
  ['13-tts-vocoder.html', 'TTS và vocoder'],
  ['14-tacotron2.html', 'Tacotron 2'],
  ['15-fastspeech2.html', 'FastSpeech 2'],
  ['16-vits.html', 'VITS'],
  ['17-xtts.html', 'XTTS'],
  ['18-ser-co-ban.html', 'SER cơ bản'],
  ['19-frame-den-cau.html', 'Từ frame đến cả câu'],
  ['20-text-bert.html', 'SER từ chữ: BERT, RoBERTa'],
  ['21-fusion.html', 'Ghép âm thanh và chữ'],
  ['22-multitask.html', 'Học ASR và SER cùng lúc'],
  ['23-audio-llm.html', 'Audio-LLM'],
  ['24-danh-gia.html', 'Đánh giá cho đúng'],
];

function buildNav() {
  const here = location.pathname.split('/').pop();
  const i = PAGES.findIndex(p => p[0] === here);
  if (i < 0) return;
  const link = j => j >= 0 && j < PAGES.length ? `<a href="${PAGES[j][0]}">${j < i ? '← ' : ''}${PAGES[j][1]}${j > i ? ' →' : ''}</a>` : '<span></span>';
  document.querySelectorAll('nav.pn').forEach(n => {
    n.innerHTML = link(i - 1) + `<a class="mid" href="00-lo-trinh.html">Lộ trình (trang ${i}/${PAGES.length - 1})</a>` + link(i + 1);
  });
}
document.addEventListener('DOMContentLoaded', buildNav);

function setup(id, h) {
  const c = document.getElementById(id), w = c.parentElement.clientWidth, d = window.devicePixelRatio || 1;
  c.width = w * d; c.height = h * d; c.style.width = w + 'px'; c.style.height = h + 'px';
  const g = c.getContext('2d');
  g.setTransform(d, 0, 0, d, 0, 0);
  g.textBaseline = 'middle';
  return {g, w, h, c};
}
function text(g, s, x, y, color = C.mute, align = 'left', bold = false, bg = false, size = 12) {
  g.font = (bold ? '600 ' : '') + size + 'px Inter, system-ui, sans-serif';
  g.textAlign = align;
  if (bg) {
    const w = g.measureText(s).width, x0 = align === 'center' ? x - w / 2 : align === 'right' ? x - w : x;
    g.fillStyle = 'rgba(255,255,255,.92)'; g.fillRect(x0 - 3, y - 9, w + 6, 18);
  }
  g.fillStyle = color; g.fillText(s, x, y);
}
function hline(g, x0, x1, y, color = C.line) {
  g.strokeStyle = color; g.lineWidth = 1; g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke();
}
function vline(g, x, y0, y1, color = C.line) {
  g.strokeStyle = color; g.lineWidth = 1; g.beginPath(); g.moveTo(x, y0); g.lineTo(x, y1); g.stroke();
}
function poly(g, xs, ys, color, lw = 2, dash = null) {
  g.save(); g.strokeStyle = color; g.lineWidth = lw; if (dash) g.setLineDash(dash);
  g.beginPath(); xs.forEach((xv, j) => j ? g.lineTo(xv, ys[j]) : g.moveTo(xv, ys[j])); g.stroke(); g.restore();
}
function dot(g, x, y, r, color, stroke = null) {
  g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fillStyle = color; g.fill();
  if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1.5; g.stroke(); }
}
function box(g, x, y, w, h, fill, stroke = null) {
  g.fillStyle = fill; g.fillRect(x, y, w, h);
  if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1; g.strokeRect(x + .5, y + .5, w - 1, h - 1); }
}

const STOPS = [[0, [251, 251, 250]], [0.35, [201, 225, 238]], [0.6, [90, 160, 200]], [0.8, [44, 95, 138]], [1, [27, 42, 73]]];
function cmap(v) {
  v = clamp(v);
  for (let j = 1; j < STOPS.length; j++) {
    if (v <= STOPS[j][0]) {
      const [a, ca] = STOPS[j - 1], [b, cb] = STOPS[j], u = (v - a) / (b - a);
      return ca.map((c, k) => Math.round(c + (cb[k] - c) * u));
    }
  }
  return STOPS[STOPS.length - 1][1];
}
const rgb = a => `rgb(${a[0]},${a[1]},${a[2]})`;

function toggle(attr, val) { document.querySelectorAll(`[data-${attr}]`).forEach(b => b.classList.toggle('on', b.dataset[attr] === val)); }

// Seeded so that the figures look the same on every reload.
function rng(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const randn = r => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
const softmax = (a, T = 1) => { const m = Math.max(...a), e = a.map(v => Math.exp((v - m) / T)), s = e.reduce((x, y) => x + y, 0); return e.map(v => v / s); };
const logsumexp = a => { const m = Math.max(...a); return m === -Infinity ? -Infinity : m + Math.log(a.reduce((s, v) => s + Math.exp(v - m), 0)); };

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

let actx;
function play(sig, sr = 16000) {
  actx = actx || new (window.AudioContext || window.webkitAudioContext)();
  const b = actx.createBuffer(1, sig.length, sr);
  b.copyToChannel(sig, 0);
  const src = actx.createBufferSource();
  src.buffer = b; src.connect(actx.destination); src.start();
}

function onResize(fn) {
  let t;
  window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(fn, 120); });
}
