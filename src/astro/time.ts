export const DEG = 180 / Math.PI;
export const RAD = Math.PI / 180;

export function norm360(x: number): number {
  return ((x % 360) + 360) % 360;
}

/**
 * ΔT = TT − UT [秒]。Espenak & Meeus の多項式（2005〜2050年）とそれ以外の粗い近似。
 * 数秒の誤差は太陽で 0.0001°、月で 0.002° 程度の差にしかならない。
 */
export function deltaT(jdUT: number): number {
  const y = 2000 + (jdUT - 2451545.0) / 365.25;
  if (y >= 2005 && y < 2050) {
    const t = y - 2000;
    return 62.92 + 0.32217 * t + 0.005589 * t * t;
  }
  const u = (y - 1820) / 100;
  return -20 + 32 * u * u;
}

/** グリニッジ平均恒星時 [度]（Meeus 12.4） */
export function gmstDeg(jdUT: number): number {
  const T = (jdUT - 2451545.0) / 36525;
  return norm360(
    280.46061837 + 360.98564736629 * (jdUT - 2451545.0) + 0.000387933 * T * T - (T * T * T) / 38710000,
  );
}

/** 章動の主項（Δψ, Δε）[度]。Meeus 22 章の簡略式（精度 0.5″程度） */
export function nutation(jdUT: number): { dPsi: number; dEps: number } {
  const T = (jdUT - 2451545.0) / 36525;
  const omega = (125.04452 - 1934.136261 * T) * RAD;
  const L = (280.4665 + 36000.7698 * T) * RAD;
  const Lp = (218.3165 + 481267.8813 * T) * RAD;
  const dPsi = (-17.2 * Math.sin(omega) - 1.32 * Math.sin(2 * L) - 0.23 * Math.sin(2 * Lp) + 0.21 * Math.sin(2 * omega)) / 3600;
  const dEps = (9.2 * Math.cos(omega) + 0.57 * Math.cos(2 * L) + 0.1 * Math.cos(2 * Lp) - 0.09 * Math.cos(2 * omega)) / 3600;
  return { dPsi, dEps };
}

/** 地方視恒星時 [度] */
export function lastDeg(jdUT: number, lonDeg: number): number {
  const { dPsi } = nutation(jdUT);
  const T = (jdUT - 2451545.0) / 36525;
  const eps = 23.4392911 - 0.0130042 * T;
  return norm360(gmstDeg(jdUT) + dPsi * Math.cos(eps * RAD) + lonDeg);
}

/**
 * 赤道座標 → 観測地点の ENU 単位ベクトル（x=東, y=北, z=上）。大気差なし。
 */
export function equatorialToEnu(raDeg: number, decDeg: number, latDeg: number, lastDegV: number): [number, number, number] {
  const H = (lastDegV - raDeg) * RAD;
  const d = decDeg * RAD;
  const p = latDeg * RAD;
  const east = -Math.cos(d) * Math.sin(H);
  const north = Math.cos(p) * Math.sin(d) - Math.sin(p) * Math.cos(d) * Math.cos(H);
  const up = Math.sin(p) * Math.sin(d) + Math.cos(p) * Math.cos(d) * Math.cos(H);
  return [east, north, up];
}

type Mat3 = number[]; // 行優先 3x3

function mul(a: Mat3, b: Mat3): Mat3 {
  const r = new Array(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) r[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
  return r;
}
const rx = (t: number): Mat3 => [1, 0, 0, 0, Math.cos(t), Math.sin(t), 0, -Math.sin(t), Math.cos(t)];
const rz = (t: number): Mat3 => [Math.cos(t), Math.sin(t), 0, -Math.sin(t), Math.cos(t), 0, 0, 0, 1];
const ry = (t: number): Mat3 => [Math.cos(t), 0, -Math.sin(t), 0, 1, 0, Math.sin(t), 0, Math.cos(t)];

export function applyMat3(m: Mat3, v: number[]): [number, number, number] {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}

/**
 * J2000.0 の赤道座標（直交、x=春分点, z=天の北極）→ 観測地点の ENU への回転行列。
 * 歳差（Meeus 21.2–21.4、IAU 1976）・章動（主項）・地方視恒星時・緯度を含む。
 * 年周光行差（最大 20″）と恒星の固有運動は含まない。
 */
export function j2000ToEnuMatrix(jdUT: number, latDeg: number, lonDeg: number): Mat3 {
  const jde = jdUT + deltaT(jdUT) / 86400;
  const T = (jde - 2451545.0) / 36525;
  const as = RAD / 3600;
  const zeta = (2306.2181 * T + 0.30188 * T * T + 0.017998 * T ** 3) * as;
  const z = (2306.2181 * T + 1.09468 * T * T + 0.018203 * T ** 3) * as;
  const theta = (2004.3109 * T - 0.42665 * T * T - 0.041833 * T ** 3) * as;
  // 歳差：P = Rz(−z) · Ry(θ) · Rz(−ζ)（座標変換としての回転）
  const P = mul(rz(-z), mul(ry(theta), rz(-zeta)));
  const { dPsi, dEps } = nutation(jdUT);
  const eps = (23.4392911 - 0.0130042 * T) * RAD;
  const N = mul(rx(-(eps + dEps * RAD)), mul(rz(-dPsi * RAD), rx(eps)));
  // 真の赤道座標 → 時角座標 → ENU
  const lst = lastDeg(jdUT, lonDeg) * RAD;
  const phi = latDeg * RAD;
  // 赤道直交 (x→春分点) を、地方子午線基準（x→南中方向）に回す：Rz(LST)
  const H = rz(lst);
  // 赤道（x=子午線と赤道の交点, y=東向き90°, z=北極）→ ENU
  //   E = y, N = −sin φ · x + cos φ · z, U = cos φ · x + sin φ · z
  const toEnu: Mat3 = [0, 1, 0, -Math.sin(phi), 0, Math.cos(phi), Math.cos(phi), 0, Math.sin(phi)];
  return mul(toEnu, mul(H, mul(N, P)));
}

export function raDecToVec(raDeg: number, decDeg: number): [number, number, number] {
  const a = raDeg * RAD, d = decDeg * RAD;
  return [Math.cos(d) * Math.cos(a), Math.cos(d) * Math.sin(a), Math.sin(d)];
}
