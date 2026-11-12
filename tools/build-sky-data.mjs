// 夜空のデータを d3-celestial（BSD-3, © Olaf Frohn）の data/ から作る。
//   恒星：XHIP（拡張 Hipparcos, Anderson & Francis 2012）由来、8等級まで
//   天の川：Jose R. Vieira "Milky Way Outline Catalog"（明るさ5段階の輪郭）
// 使い方: node tools/build-sky-data.mjs <d3-celestial/data のパス>
import { readFileSync, writeFileSync, copyFileSync } from 'node:fs';
import { join } from 'node:path';

const src = process.argv[2];
if (!src) { console.error('usage: node tools/build-sky-data.mjs <d3-celestial/data>'); process.exit(1); }

const stars = JSON.parse(readFileSync(join(src, 'stars.8.json'), 'utf8')).features;
// 1星 = 4 float32：赤経[度, 0..360), 赤緯[度], 実視等級 V, B−V（不明は 0.6）
const out = new Float32Array(stars.length * 4);
let n = 0;
for (const f of stars) {
  const [lon, lat] = f.geometry.coordinates;
  const mag = Number(f.properties.mag);
  let bv = Number(f.properties.bv);
  if (!Number.isFinite(mag)) continue;
  if (!Number.isFinite(bv)) bv = 0.6;
  out.set([(lon + 360) % 360, lat, mag, bv], n * 4);
  n++;
}
writeFileSync('public/data/stars.bin', Buffer.from(out.buffer, 0, n * 16));
copyFileSync(join(src, 'mw.json'), 'public/data/milkyway.json');
console.log(`stars: ${n}`);
