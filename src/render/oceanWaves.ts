/**
 * 海面の波の表（sky.wgsl の oceanWaves が読む）。
 * 波長は固定（2 cm〜600 m を対数で等分）で、位相は ω = √(gk) を倍精度で積分して 2π の剰余で持つ。
 * 風速は「どの波に傾斜の分散を配るか」（Pierson–Moskowitz のピーク波長より長い波には配らない）だけを変える。
 * → 時間がどれだけ経っても、風速が変わっても、波の位置が跳ばない
 *  （以前はシェーダーで ω·t を計算していたため t を 3600 秒で巻き戻すたびに全波が跳び、
 *   風速で波長そのものが変わるので遠くの波の模様も跳んでいた）。
 */
export const N_WAVES = 64;
const L_MIN = 0.02;
const L_MAX_ALL = 600;
const G = 9.81;

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 風速 U [m/s] での波の最長波長 [m]（Pierson–Moskowitz のピーク波長 ≈ 2πU² / (0.877² g)） */
export function peakWavelength(windSpeed: number): number {
  const U = Math.max(windSpeed, 0.5);
  return Math.max(4, 0.83 * U * U);
}

export class OceanWaves {
  private lam = new Float64Array(N_WAVES);
  private angle = new Float64Array(N_WAVES);
  private phase = new Float64Array(N_WAVES);
  /** 分散の配分に使う風速（急に変わらないよう平滑化） */
  private windSmooth = -1;
  /** GPU に渡す表：(λ [m], 風下からの向きのずれ [rad], 位相 [rad], 傾斜の分散の配分（合計 1）) × N_WAVES */
  readonly data: Float32Array<ArrayBuffer> = new Float32Array(N_WAVES * 4);

  constructor(seed = 20261112) {
    const rnd = mulberry32(seed);
    for (let i = 0; i < N_WAVES; i++) {
      this.lam[i] = L_MIN * Math.pow(L_MAX_ALL / L_MIN, (i + rnd()) / N_WAVES);
      this.angle[i] = (rnd() - 0.5) * 2.4; // 風下 ±69°
      this.phase[i] = rnd() * 2 * Math.PI;
    }
  }

  /** dtSec [s] 進めて表を更新する */
  update(dtSec: number, windSpeed: number): Float32Array<ArrayBuffer> {
    const TWO_PI = 2 * Math.PI;
    this.windSmooth = this.windSmooth < 0 ? windSpeed : this.windSmooth + (windSpeed - this.windSmooth) * (1 - Math.exp(-dtSec / 5));
    const lMax = peakWavelength(this.windSmooth);
    // ピーク波長の前後（対数で ±0.15 桁）でなめらかに切る
    const w = new Float64Array(N_WAVES);
    let wsum = 0;
    for (let i = 0; i < N_WAVES; i++) {
      const x = (Math.log10(this.lam[i] / lMax) + 0.15) / 0.3;
      const c = Math.min(1, Math.max(0, x));
      w[i] = 1 - c * c * (3 - 2 * c);
      wsum += w[i];
    }
    for (let i = 0; i < N_WAVES; i++) {
      const omega = Math.sqrt((G * TWO_PI) / this.lam[i]);
      let ph = (this.phase[i] + omega * dtSec) % TWO_PI;
      if (ph < 0) ph += TWO_PI;
      this.phase[i] = ph;
      this.data[i * 4] = this.lam[i];
      this.data[i * 4 + 1] = this.angle[i];
      this.data[i * 4 + 2] = ph;
      this.data[i * 4 + 3] = w[i] / wsum;
    }
    return this.data;
  }
}
