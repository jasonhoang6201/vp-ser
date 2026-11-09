// Evaluate a JS expression in the renderer's Chrome after loading some lib files, to get numbers for a narration
// exactly as the scene will compute them.
// Usage (from tools/): node chromeval.mjs gmm,dsp,tts "JSON.stringify(TTS.rms(TTS.VERS.zero))"
import puppeteer from 'puppeteer-core';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const [libs, expr] = process.argv.slice(2);
const browser = await puppeteer.launch({executablePath: CHROME, headless: true, args: ['--allow-file-access-from-files']});
const page = await browser.newPage();
await page.goto('about:blank');
for (const f of ['engine', ...libs.split(',')]) await page.addScriptTag({path: path.join(root, 'lib', f + '.js')});
console.log(await page.evaluate(expr));
await browser.close();
