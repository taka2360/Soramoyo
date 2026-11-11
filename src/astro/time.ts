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
