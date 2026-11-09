// Screenshot each <section> of a demo page and print its readouts and any page errors, to check a page without opening it.
// Usage (from tools/): node shotpage.mjs ../../paper-review/demo/05-mfcc.html /path/to/outdir
import puppeteer from 'puppeteer-core';
import path from 'node:path';

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const [page_, out] = process.argv.slice(2);
const browser = await puppeteer.launch({executablePath: CHROME, headless: true});
const page = await browser.newPage();
await page.setViewport({width: 1012, height: 900, deviceScaleFactor: 1});
page.on('pageerror', e => console.log('PAGEERROR', e.message));
page.on('console', m => m.type() === 'error' && console.log('CONSOLE', m.text()));
await page.goto('file://' + path.resolve(page_), {waitUntil: 'load'});
const secs = await page.$$('section');
for (let i = 0; i < secs.length; i++) await secs[i].screenshot({path: path.join(out, `s${String(i).padStart(2, '0')}.png`)});
console.log(await page.evaluate(() => [...document.querySelectorAll('.readout')].map(r => r.id + ': ' + r.innerText).join('\n')));
await browser.close();
