// Render one clip: node tools/render.mjs 01-van-de [--stills 3,40,95] [--fps 30]
// Needs audio/<clip>/ from run_tts.sh. Writes build/<clip>.data.js, build/<clip>.wav and ../video/<page>/<clip>.mp4.
import fs from 'node:fs';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import puppeteer from 'puppeteer-core';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2), clip = args[0];
const opt = k => { const i = args.indexOf(k); return i < 0 ? null : args[i + 1]; };
const FPS = +(opt('--fps') || 30), stills = opt('--stills');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const p = (...a) => path.join(ROOT, ...a);
const PAGES = {
  gmm: '06-gmm', hmm: '07-hmm', asr: '08-he-asr-co-dien', dnn: '09-dnn-hmm', ctc: '10-ctc', att: '11-attention',
  ssl: '12-ssl-whisper', tts: '13-tts-vocoder', taco: '14-tacotron2', fs: '15-fastspeech2', vits: '16-vits', xtts: '17-xtts',
  ser: '18-ser-co-ban', pool: '19-frame-den-cau', bert: '20-text-bert', fus: '21-fusion', mtl: '22-multitask',
  llm: '23-audio-llm', eval: '24-danh-gia',
};
// MFCC clips were the first ones made and have no page prefix: 01-van-de, 02-log, ...
const clipPage = /^\d/.test(clip) ? '05-mfcc' : PAGES[clip.split('-')[0]];
if (!clipPage) throw new Error(`no demo page for clip ${clip}`);
const videoDir = path.join(ROOT, '..', 'video', clipPage);

const spec =JSON.parse(fs.readFileSync(p('narration', clip + '.json'), 'utf8'));
// --fake-dur lays out the clip before the narration exists (about 15 spoken characters per second), stills only.
const dur = args.includes('--fake-dur')
  ? {sr: 48000, durations: Object.fromEntries(spec.segments.map(s => [s.id, s.say.length / 15]))}
  : JSON.parse(fs.readFileSync(p('audio', clip, 'durations.json'), 'utf8'));
fs.mkdirSync(p('build'), {recursive: true});
fs.mkdirSync(videoDir, {recursive: true});
fs.writeFileSync(p('build', clip + '.data.js'), `window.SPEC = ${JSON.stringify(spec)};\nwindow.DUR = ${JSON.stringify(dur)};\n`);

const browser = await puppeteer.launch({executablePath: CHROME, headless: true, args: ['--allow-file-access-from-files']});
const page = await browser.newPage();
await page.setViewport({width: 1920, height: 1080});
const errors = [];
page.on('pageerror', e => errors.push(e.message));
page.on('console', m => m.type() === 'error' && errors.push(m.text()));
await page.evaluateOnNewDocument(() => { window.RENDERING = true; });
await page.goto('file://' + p('scenes', clip + '.html'));
await page.waitForFunction(() => window.TOTAL > 0, {timeout: 20000}).catch(() => {});
if (errors.length) { console.error(errors.join('\n')); await browser.close(); process.exit(1); }
const {total, tl} = await page.evaluate(() => ({total: window.TOTAL, tl: window.TIMELINE}));
console.log(`total ${total.toFixed(1)} s, ${Object.keys(tl).length} segments`);

const grab = t => page.evaluate((t, q) => { renderAt(t); return document.getElementById('cv').toDataURL(q ? 'image/png' : 'image/jpeg', 0.93); }, t, false)
  .then(u => Buffer.from(u.split(',')[1], 'base64'));

if (stills) {
  for (const s of stills.split(',')) {
    const t = isNaN(+s) ? tl[s].start + tl[s].dur * 0.85 : +s;
    const f = p('build', `${clip}_${s}.jpg`);
    fs.writeFileSync(f, await grab(t));
    console.log('still', f);
  }
  await browser.close();
  process.exit(0);
}

// Narration track: each segment's wav placed at its timeline start.
function readWav(f) {
  const b = fs.readFileSync(f);
  let o = 12, sr = 0, data = null;
  while (o < b.length) {
    const id = b.toString('ascii', o, o + 4), n = b.readUInt32LE(o + 4);
    if (id === 'fmt ') sr = b.readUInt32LE(o + 12);
    if (id === 'data') data = new Int16Array(b.buffer.slice(b.byteOffset + o + 8, b.byteOffset + o + 8 + n));
    o += 8 + n + (n & 1);
  }
  return {sr, data};
}
const SR = dur.sr, track = new Int16Array(Math.ceil(total * SR));
for (const [id, seg] of Object.entries(tl)) {
  const w = readWav(p('audio', clip, id + '.wav'));
  if (w.sr !== SR) throw new Error(`${id}.wav is ${w.sr} Hz, expected ${SR}`);
  track.set(w.data.subarray(0, track.length - Math.round(seg.start * SR)), Math.round(seg.start * SR));
}
const hdr = Buffer.alloc(44);
hdr.write('RIFF', 0); hdr.writeUInt32LE(36 + track.byteLength, 4); hdr.write('WAVEfmt ', 8);
hdr.writeUInt32LE(16, 16); hdr.writeUInt16LE(1, 20); hdr.writeUInt16LE(1, 22); hdr.writeUInt32LE(SR, 24);
hdr.writeUInt32LE(SR * 2, 28); hdr.writeUInt16LE(2, 32); hdr.writeUInt16LE(16, 34); hdr.write('data', 36); hdr.writeUInt32LE(track.byteLength, 40);
fs.writeFileSync(p('build', clip + '.wav'), Buffer.concat([hdr, Buffer.from(track.buffer)]));

const out = path.join(videoDir, clip + '.mp4');
const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
  '-i', p('build', clip + '.wav'), '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p',
  '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', '-shortest', out], {stdio: ['pipe', 'inherit', 'inherit']});
const n = Math.ceil(total * FPS), t0 = Date.now();
for (let i = 0; i < n; i++) {
  const buf = await grab(i / FPS);
  if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
  if (i % (FPS * 20) === 0) console.log(`frame ${i}/${n}  ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
ff.stdin.end();
await new Promise(r => ff.on('close', r));
await browser.close();
if (errors.length) console.error('page errors:\n' + errors.join('\n'));
console.log('wrote', out);
