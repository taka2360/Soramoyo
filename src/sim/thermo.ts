/**
 * 熱力学定数と関数（WGSL 側 src/shaders/sim/common.wgsl と同じ式を使うこと）。
 * 飽和水蒸気圧は Bolton (1980)（水面）。氷相は段階6で追加。
 */
export const G = 9.81;
export const RD = 287.04;
export const RV = 461.5;
export const CP = 1004.5;
export const LV = 2.5e6;
export const P00 = 1e5;
export const KAPPA = RD / CP;
export const EPS = RD / RV; // 0.622

/** 飽和水蒸気圧 [Pa]（T[K]） */
export function esat(T: number): number {
  const tc = T - 273.15;
  return 611.2 * Math.exp((17.67 * tc) / (tc + 243.5));
}

/** 飽和混合比 [kg/kg] */
export function qvsat(T: number, p: number): number {
  const es = Math.min(esat(T), p * 0.5);
  return (EPS * es) / (p - es);
}

export function exner(p: number): number {
  return Math.pow(p / P00, KAPPA);
}

/**
 * 飽和調整（液相のみ）：qv が過飽和なら凝結、雲水があり未飽和なら蒸発。
 * θ は温位、pi はエクスナー関数。戻り値は調整後の [θ, qv, qc]。
 */
export function saturationAdjust(theta: number, qv: number, qc: number, p: number, pi: number): [number, number, number] {
  let th = theta, v = qv, c = qc;
  for (let it = 0; it < 3; it++) {
    const T = th * pi;
    const qs = qvsat(T, p);
    if (v <= qs && c <= 0) break;
    const tc = T - 273.15;
    const dlnes = (17.67 * 243.5) / ((tc + 243.5) * (tc + 243.5));
    let dq = (v - qs) / (1 + (LV / CP) * qs * dlnes);
    dq = Math.max(dq, -c);
    th += (LV / (CP * pi)) * dq;
    v -= dq;
    c += dq;
  }
  return [th, v, c];
}
