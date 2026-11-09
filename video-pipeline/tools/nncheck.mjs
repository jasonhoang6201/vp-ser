// Print the toy network's accuracy in the same Chrome the renderer uses: Chrome and Node can round tanh/exp
// differently, and 3000 momentum steps amplify that, so numbers quoted in a narration must come from here.
// Usage (from tools/): node nncheck.mjs 2 1000,2000,3000
import puppeteer from 'puppeteer-core';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const [H, at] = [+process.argv[2], process.argv[3].split(',').map(Number)];
const browser = await puppeteer.launch({executablePath: CHROME, headless: true, args: ['--allow-file-access-from-files']});
const page = await browser.newPage();
await page.goto('about:blank');
for (const f of ['lib/engine.js', 'lib/gmm.js', 'lib/nn.js']) await page.addScriptTag({path: path.join(root, f)});
const out = await page.evaluate((H, at) => Object.entries(NN.train(H, at)).map(([k, v]) => `${k}: ${(100 * v.acc).toFixed(1)}% loss ${v.loss.toFixed(3)}`), H, at);
console.log(out.join('\n'));
await browser.close();
