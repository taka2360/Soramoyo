/**
 * 太陽の位置（Meeus "Astronomical Algorithms" 第25章の低精度式 + 章動の主項）。
 * 精度：赤経・赤緯で約 0.01°（1900〜2100 年）。
 * 大気差は refraction.ts で別途加える（ここで返すのは「真の」方向）。
 */
import { DEG, RAD, deltaT, norm360 } from './time';

export interface EquatorialPos {
  raDeg: number;
  decDeg: number;
  /** 距離 [AU]（太陽）または [km]（月） */
  distance: number;
}

export function sunEquatorial(jdUT: number): EquatorialPos & { eclLonDeg: number; obliquityDeg: number } {
  const jdTT = jdUT + deltaT(jdUT) / 86400;
  const T = (jdTT - 2451545.0) / 36525;
  const L0 = norm360(280.46646 + 36000.76983 * T + 0.0003032 * T * T);
  const M = norm360(357.52911 + 35999.05029 * T - 0.0001537 * T * T);
  const e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T * T;
  const Mr = M * RAD;
  const C =
    (1.914602 - 0.004817 * T - 0.000014 * T * T) * Math.sin(Mr) +
    (0.019993 - 0.000101 * T) * Math.sin(2 * Mr) +
    0.000289 * Math.sin(3 * Mr);
  const trueLon = L0 + C;
  const v = (M + C) * RAD;
  const R = (1.000001018 * (1 - e * e)) / (1 + e * Math.cos(v));
  const omega = (125.04 - 1934.136 * T) * RAD;
  const lambda = trueLon - 0.00569 - 0.00478 * Math.sin(omega);
  const eps0 = 23 + 26 / 60 + (21.448 - 46.815 * T - 0.00059 * T * T + 0.001813 * T * T * T) / 3600;
  const eps = eps0 + 0.00256 * Math.cos(omega);
  const lr = lambda * RAD, er = eps * RAD;
  const ra = norm360(Math.atan2(Math.cos(er) * Math.sin(lr), Math.cos(lr)) * DEG);
  const dec = Math.asin(Math.sin(er) * Math.sin(lr)) * DEG;
  return { raDeg: ra, decDeg: dec, distance: R, eclLonDeg: lambda, obliquityDeg: eps };
}

/** 太陽の視半径 [rad]（地球–太陽距離 R[AU] による変化込み） */
export function sunAngularRadius(distanceAU: number): number {
  return ((959.63 / 3600) * RAD) / distanceAU;
}
