/**
 * 月の位置（Meeus "Astronomical Algorithms" 第47章、ELP-2000/82 の主要項）。
 * 経度は約60項、緯度は約30項、距離は約45項に打ち切り（精度：経度・緯度で約 0.003°、距離で数 km）。
 * 返すのは地心の視位置（章動込み）。地表から見た方向（視差）は topocentric() で求める。
 */
import { DEG, RAD, deltaT, norm360, nutation } from './time';

// [D, M, M', F, Σl(1e-6 度), Σr(1e-3 km)]
const LR: number[][] = [
  [0, 0, 1, 0, 6288774, -20905355], [2, 0, -1, 0, 1274027, -3699111], [2, 0, 0, 0, 658314, -2955968],
  [0, 0, 2, 0, 213618, -569925], [0, 1, 0, 0, -185116, 48888], [0, 0, 0, 2, -114332, -3149],
  [2, 0, -2, 0, 58793, 246158], [2, -1, -1, 0, 57066, -152138], [2, 0, 1, 0, 53322, -170733],
  [2, -1, 0, 0, 45758, -204586], [0, 1, -1, 0, -40923, -129620], [1, 0, 0, 0, -34720, 108743],
  [0, 1, 1, 0, -30383, 104755], [2, 0, 0, -2, 15327, 10321], [0, 0, 1, 2, -12528, 0],
  [0, 0, 1, -2, 10980, 79661], [4, 0, -1, 0, 10675, -34782], [0, 0, 3, 0, 10034, -23210],
  [4, 0, -2, 0, 8548, -21636], [2, 1, -1, 0, -7888, 24208], [2, 1, 0, 0, -6766, 30824],
  [1, 0, -1, 0, -5163, -8379], [1, 1, 0, 0, 4987, -16675], [2, -1, 1, 0, 4036, -12831],
  [2, 0, 2, 0, 3994, -10445], [4, 0, 0, 0, 3861, -11650], [2, 0, -3, 0, 3665, 14403],
  [0, 1, -2, 0, -2689, -7003], [2, 0, -1, 2, -2602, 0], [2, -1, -2, 0, 2390, 10056],
  [1, 0, 1, 0, -2348, 6322], [2, -2, 0, 0, 2236, -9884], [0, 1, 2, 0, -2120, 5751],
  [0, 2, 0, 0, -2069, 0], [2, -2, -1, 0, 2048, -4950], [2, 0, 1, -2, -1773, 4130],
  [2, 0, 0, 2, -1595, 0], [4, -1, -1, 0, 1215, -3958], [0, 0, 2, 2, -1110, 0],
  [3, 0, -1, 0, -892, 3258], [2, 1, 1, 0, -810, 2616], [4, -1, -2, 0, 759, -1897],
  [0, 2, -1, 0, -713, -2117], [2, 2, -1, 0, -700, 2354], [2, 1, -2, 0, 691, 0],
  [2, -1, 0, -2, 596, 0], [4, 0, 1, 0, 549, -1423], [0, 0, 4, 0, 537, -1117],
  [4, -1, 0, 0, 520, -1571], [1, 0, -2, 0, -487, -1739], [2, 1, 0, -2, -399, 0],
  [0, 0, 2, -2, -381, -4421], [1, 1, 1, 0, 351, 0], [3, 0, -2, 0, -340, 0],
  [4, 0, -3, 0, 330, 0], [2, -1, 2, 0, 327, 0], [0, 2, 1, 0, -323, 1165],
  [1, 1, -1, 0, 299, 0], [2, 0, 3, 0, 294, 0], [2, 0, -1, -2, 0, 8752],
];

// [D, M, M', F, Σb(1e-6 度)]
const B: number[][] = [
  [0, 0, 0, 1, 5128122], [0, 0, 1, 1, 280602], [0, 0, 1, -1, 277693], [2, 0, 0, -1, 173237],
  [2, 0, -1, 1, 55413], [2, 0, -1, -1, 46271], [2, 0, 0, 1, 32573], [0, 0, 2, 1, 17198],
  [2, 0, 1, -1, 9266], [0, 0, 2, -1, 8822], [2, -1, 0, -1, 8216], [2, 0, -2, -1, 4324],
  [2, 0, 1, 1, 4200], [2, 1, 0, -1, -3359], [2, -1, -1, 1, 2463], [2, -1, 0, 1, 2211],
  [2, -1, -1, -1, 2065], [0, 1, -1, -1, -1870], [4, 0, -1, -1, 1828], [0, 1, 0, 1, -1794],
  [0, 0, 0, 3, -1749], [0, 1, -1, 1, -1565], [1, 0, 0, 1, -1491], [0, 1, 1, 1, -1475],
  [0, 1, 1, -1, -1410], [0, 1, 0, -1, -1344], [1, 0, 0, -1, -1335], [0, 0, 3, 1, 1107],
  [4, 0, 0, -1, 1021], [4, 0, -1, 1, 833],
];

export interface MoonPosition {
  /** 地心の視黄経・視黄緯 [度]、距離 [km] */
  lonDeg: number;
  latDeg: number;
  distanceKm: number;
  /** 地心の視赤経・視赤緯 [度] */
  raDeg: number;
  decDeg: number;
}

export function moonPosition(jdUT: number): MoonPosition {
  const jde = jdUT + deltaT(jdUT) / 86400;
  const T = (jde - 2451545.0) / 36525;
  const T2 = T * T, T3 = T2 * T, T4 = T3 * T;
  const Lp = norm360(218.3164477 + 481267.88123421 * T - 0.0015786 * T2 + T3 / 538841 - T4 / 65194000);
  const D = norm360(297.8501921 + 445267.1114034 * T - 0.0018819 * T2 + T3 / 545868 - T4 / 113065000);
  const M = norm360(357.5291092 + 35999.0502909 * T - 0.0001536 * T2 + T3 / 24490000);
  const Mp = norm360(134.9633964 + 477198.8675055 * T + 0.0087414 * T2 + T3 / 69699 - T4 / 14712000);
  const F = norm360(93.272095 + 483202.0175233 * T - 0.0036539 * T2 - T3 / 3526000 + T4 / 863310000);
  const A1 = norm360(119.75 + 131.849 * T);
  const A2 = norm360(53.09 + 479264.29 * T);
  const A3 = norm360(313.45 + 481266.484 * T);
  const E = 1 - 0.002516 * T - 0.0000074 * T2;
  const eFac = (m: number) => (Math.abs(m) === 1 ? E : Math.abs(m) === 2 ? E * E : 1);

  let sl = 0, sr = 0, sb = 0;
  for (const [d, m, mp, f, l, r] of LR) {
    const arg = (d * D + m * M + mp * Mp + f * F) * RAD;
    const e = eFac(m);
    sl += l * e * Math.sin(arg);
    sr += r * e * Math.cos(arg);
  }
  for (const [d, m, mp, f, b] of B) {
    const arg = (d * D + m * M + mp * Mp + f * F) * RAD;
    sb += b * eFac(m) * Math.sin(arg);
  }
  sl += 3958 * Math.sin(A1 * RAD) + 1962 * Math.sin((Lp - F) * RAD) + 318 * Math.sin(A2 * RAD);
  sb += -2235 * Math.sin(Lp * RAD) + 382 * Math.sin(A3 * RAD) + 175 * Math.sin((A1 - F) * RAD) +
    175 * Math.sin((A1 + F) * RAD) + 127 * Math.sin((Lp - Mp) * RAD) - 115 * Math.sin((Lp + Mp) * RAD);

  const { dPsi, dEps } = nutation(jdUT);
  const lon = norm360(Lp + sl / 1e6 + dPsi);
  const lat = sb / 1e6;
  const dist = 385000.56 + sr / 1000;
  const eps0 = 23 + 26 / 60 + (21.448 - 46.815 * T - 0.00059 * T2 + 0.001813 * T3) / 3600;
  const eps = (eps0 + dEps) * RAD;
  const l = lon * RAD, b = lat * RAD;
  const ra = norm360(Math.atan2(Math.sin(l) * Math.cos(eps) - Math.tan(b) * Math.sin(eps), Math.cos(l)) * DEG);
  const dec = Math.asin(Math.sin(b) * Math.cos(eps) + Math.cos(b) * Math.sin(eps) * Math.sin(l)) * DEG;
  return { lonDeg: lon, latDeg: lat, distanceKm: dist, raDeg: ra, decDeg: dec };
}

const EARTH_EQ_RADIUS_KM = 6378.14;

/**
 * 地心方向（ENU 単位ベクトル）と距離から、観測地点（標高 h）から見た方向へ直す（地平視差）。
 * 地球は球として近似（測地緯度と地心緯度の差による誤差は 0.003° 程度）。
 */
export function topocentric(geoDir: [number, number, number], distKm: number, elevationM: number): [number, number, number] {
  const r = EARTH_EQ_RADIUS_KM + elevationM / 1000;
  const x = geoDir[0] * distKm, y = geoDir[1] * distKm, z = geoDir[2] * distKm - r;
  const n = Math.hypot(x, y, z);
  return [x / n, y / n, z / n];
}

/** 月の視半径 [rad]（観測者からの距離 distKm） */
export function moonAngularRadius(distKm: number): number {
  return Math.asin(1737.4 / distKm);
}

/**
 * 月相：位相角 i [度]（太陽–月–地球の角）と輝面比 k。
 * sunDirGeo, moonDirGeo は地心から見た単位ベクトル（同じ座標系）。
 */
export function moonPhase(sunDirGeo: number[], sunDistKm: number, moonDirGeo: number[], moonDistKm: number): { phaseAngleDeg: number; illuminated: number } {
  const cosPsi = sunDirGeo[0] * moonDirGeo[0] + sunDirGeo[1] * moonDirGeo[1] + sunDirGeo[2] * moonDirGeo[2];
  const psi = Math.acos(Math.max(-1, Math.min(1, cosPsi))); // 離角
  const i = Math.atan2(sunDistKm * Math.sin(psi), moonDistKm - sunDistKm * Math.cos(psi));
  return { phaseAngleDeg: i * DEG, illuminated: (1 + Math.cos(i)) / 2 };
}

/**
 * 月の明るさ：大気上端での照度 [lx]（V 等級から）。
 * 等級は Allen の式 m = −12.73 + 0.026|i| + 4×10⁻⁹ i⁴（平均距離）に距離補正。
 */
export function moonIlluminanceLux(phaseAngleDeg: number, distKm: number): number {
  const i = Math.abs(phaseAngleDeg);
  const m = -12.73 + 0.026 * i + 4e-9 * i ** 4 + 5 * Math.log10(distKm / 384400);
  return 2.54e-6 * Math.pow(10, -0.4 * m);
}
