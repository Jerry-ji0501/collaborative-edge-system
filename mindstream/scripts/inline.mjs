// Post-build: inline the JS bundle into dist/index.html so the film is a single self-contained file.
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const dist = new URL('../dist/', import.meta.url).pathname;
let html = readFileSync(join(dist, 'index.html'), 'utf8');

html = html.replace(/<script type="module" crossorigin src="\.\/(assets\/[^"]+\.js)"><\/script>/g, (_m, p) => {
  const js = readFileSync(join(dist, p), 'utf8').replace(/<\/script/gi, '<\\/script');
  return `<script type="module">${js}</script>`;
});
html = html.replace(/<link rel="stylesheet" crossorigin href="\.\/(assets\/[^"]+\.css)">/g, (_m, p) => {
  return `<style>${readFileSync(join(dist, p), 'utf8')}</style>`;
});

writeFileSync(join(dist, 'mindstream.html'), html);
const assets = readdirSync(join(dist, 'assets'));
console.log(`inlined ${assets.length} asset(s) → dist/mindstream.html (${(html.length / 1024).toFixed(1)} KB)`);
