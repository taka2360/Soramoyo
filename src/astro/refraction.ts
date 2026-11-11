/**
 * 大気差（光の屈折による天体の浮き上がり）。
 *  - 真高度 → 見かけ高度：Saemundsson (1986)
 *  - 見かけ高度 → 真高度：Bennett (1982)
 * 標準大気（1010 hPa, 10℃）基準で、気圧・気温で補正する。
 * 地平線付近では実際の気温分布によって ±0.2° 程度ばらつく（ここでは標準値）。
 * 高度 -2° より下は計算しない（式の適用範囲外）。
 */
import { DEG, RAD } from './time';

function factor(pressureHpa: number, tempC: number): number {
  return (pressureHpa / 1010) * (283 / (273 + tempC));
}

/** 真高度 h[度] に対する大気差 [度] */
export function refractionFromTrue(hDeg: number, pressureHpa = 1010, tempC = 10): number {
  const h = Math.max(hDeg, -2);
  const R = 1.02 / Math.tan((h + 10.3 / (h + 5.11)) * RAD); // 分角
  return Math.max(R, 0) / 60 * factor(pressureHpa, tempC);
}

/** 見かけ高度 h[度] に対する大気差 [度] */
export function refractionFromApparent(hDeg: number, pressureHpa = 1010, tempC = 10): number {
  const h = Math.max(hDeg, -2);
  const R = 1 / Math.tan((h + 7.31 / (h + 4.4)) * RAD);
  return Math.max(R, 0) / 60 * factor(pressureHpa, tempC);
}

/** ENU 単位ベクトルの真の方向 → 見かけの方向 */
export function applyRefraction(v: [number, number, number], pressureHpa = 1010, tempC = 10): [number, number, number] {
  const h = Math.asin(Math.max(-1, Math.min(1, v[2]))) * DEG;
  const ha = (h + refractionFromTrue(h, pressureHpa, tempC)) * RAD;
  const horiz = Math.hypot(v[0], v[1]);
  if (horiz < 1e-9) return v;
  const c = Math.cos(ha) / horiz;
  return [v[0] * c, v[1] * c, Math.sin(ha)];
}
