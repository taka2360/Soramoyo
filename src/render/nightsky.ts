/**
 * 夜空のデータの読み込み：
 *  - 天の川：輪郭カタログ（5段階）を正距円筒図（J2000 赤経・赤緯）に塗り重ね、ぼかしてテクスチャにする
 *  - 恒星：XHIP 由来の 8 等級までの星（public/data/stars.bin）→ GPU 用の配列
 */

interface MwFeature {
  id: string;
  geometry: { type: 'MultiPolygon' | 'Polygon'; coordinates: number[][][][] | number[][][] };
}

const MW_W = 2048, MW_H = 1024;

export async function loadMilkyWayTexture(device: GPUDevice): Promise<GPUTexture> {
  const res = await fetch('/data/milkyway.json');
  const json = (await res.json()) as { features: MwFeature[] };
  const canvas = new OffscreenCanvas(MW_W, MW_H);
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, MW_W, MW_H);
  ctx.globalCompositeOperation = 'lighter';
  // 5段階を重ねると、最も明るい所で 255 になる
  ctx.fillStyle = 'rgb(51,51,51)';
  const xOf = (ra: number) => (ra / 360) * MW_W;
  const yOf = (dec: number) => ((90 - dec) / 180) * MW_H;
  for (const f of json.features) {
    const polys = f.geometry.type === 'Polygon' ? [f.geometry.coordinates as number[][][]] : (f.geometry.coordinates as number[][][][]);
    for (const poly of polys) {
      // 経度の不連続（0°/360°）をまたぐ輪は、連続になるよう展開してから ±360° ずらして3回描く
      for (const shift of [-360, 0, 360]) {
        ctx.beginPath();
        for (const ring of poly) {
          let prev = NaN;
          ring.forEach(([lon, lat], i) => {
            let ra = ((lon % 360) + 360) % 360;
            if (!Number.isNaN(prev)) {
              while (ra - prev > 180) ra -= 360;
              while (ra - prev < -180) ra += 360;
            }
            prev = ra;
            const x = xOf(ra + shift), y = yOf(lat);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
          });
          ctx.closePath();
        }
        ctx.fill('evenodd');
      }
    }
  }
  // 輪郭の段差をならす：細かいぼかし（構造を残す）と大きいぼかし（段差を消す）を半分ずつ
  const blurred = new OffscreenCanvas(MW_W, MW_H);
  const bctx = blurred.getContext('2d')!;
  bctx.globalAlpha = 0.5;
  bctx.filter = 'blur(4px)';
  bctx.drawImage(canvas, 0, 0);
  bctx.globalCompositeOperation = 'lighter';
  bctx.filter = 'blur(14px)';
  bctx.drawImage(canvas, 0, 0);
  const bmp = blurred.transferToImageBitmap();
  const tex = device.createTexture({
    label: 'milkyway', size: [MW_W, MW_H], format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture({ source: bmp }, { texture: tex }, [MW_W, MW_H]);
  return tex;
}

/** 星の色：B−V → 色温度（Ballesteros 2012）→ 黒体の色（線形 sRGB、輝度 1 に正規化） */
function bvToRgb(bv: number): [number, number, number] {
  const T = 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62));
  // 黒体スペクトルを CIE 近似等色関数で積分
  const h = 6.626e-34, c = 2.998e8, k = 1.381e-23;
  let X = 0, Y = 0, Z = 0;
  for (let l = 380; l <= 780; l += 5) {
    const lm = l * 1e-9;
    const B = 1 / (Math.pow(lm, 5) * (Math.exp((h * c) / (lm * k * T)) - 1));
    const g = (x: number, m: number, s1: number, s2: number) => Math.exp(-0.5 * ((x - m) / (x < m ? s1 : s2)) ** 2);
    const xb = 1.056 * g(l, 599.8, 37.9, 31.0) + 0.362 * g(l, 442.0, 16.0, 26.7) - 0.065 * g(l, 501.1, 20.4, 26.2);
    const yb = 0.821 * g(l, 568.8, 46.9, 40.5) + 0.286 * g(l, 530.9, 16.3, 31.1);
    const zb = 1.217 * g(l, 437.0, 11.8, 36.0) + 0.681 * g(l, 459.0, 26.0, 13.8);
    X += B * xb; Y += B * yb; Z += B * zb;
  }
  const r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z;
  const gg = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
  const b = 0.0557 * X - 0.204 * Y + 1.057 * Z;
  const rgb: [number, number, number] = [Math.max(r, 0), Math.max(gg, 0), Math.max(b, 0)];
  const lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
  return [rgb[0] / lum, rgb[1] / lum, rgb[2] / lum];
}

export interface StarData {
  buffer: GPUBuffer;
  count: number;
}

/**
 * GPU 用の星の配列：1星 = 8 float
 *   [x, y, z（J2000 の単位ベクトル）, V 等級, r, g, b（輝度1の色）, 乱数（瞬きの位相）]
 */
export async function loadStars(device: GPUDevice): Promise<StarData> {
  const res = await fetch('/data/stars.bin');
  const raw = new Float32Array(await res.arrayBuffer());
  const n = raw.length / 4;
  const out = new Float32Array(n * 8);
  const cache = new Map<number, [number, number, number]>();
  for (let i = 0; i < n; i++) {
    const ra = (raw[i * 4] * Math.PI) / 180, dec = (raw[i * 4 + 1] * Math.PI) / 180;
    const bvKey = Math.round(raw[i * 4 + 3] * 50) / 50;
    let col = cache.get(bvKey);
    if (!col) { col = bvToRgb(Math.max(-0.4, Math.min(2.0, bvKey))); cache.set(bvKey, col); }
    out.set([Math.cos(dec) * Math.cos(ra), Math.cos(dec) * Math.sin(ra), Math.sin(dec), raw[i * 4 + 2], ...col, Math.random()], i * 8);
  }
  const buffer = device.createBuffer({ label: 'stars', size: out.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, out);
  return { buffer, count: n };
}
