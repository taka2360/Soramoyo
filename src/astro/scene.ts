import type { Observer } from '../core/clock';
import { sunEquatorial, sunAngularRadius } from './sun';
import { equatorialToEnu, lastDeg } from './time';
import { applyRefraction } from './refraction';
import { solarIlluminanceRgb } from '../atmosphere/spectrum';

const SUN_TOA = solarIlluminanceRgb();
const SUN_WHITE = (() => {
  const m = (SUN_TOA[0] + SUN_TOA[1] + SUN_TOA[2]) / 3;
  return SUN_TOA.map((c) => c / m);
})();

export interface Ephemeris {
  sunDir: [number, number, number];
  sunDirApparent: [number, number, number];
  sunAngularRadius: number;
  sunDistanceAU: number;
  sunIlluminance: number[];
  whitePoint: number[];
  /** 大気差の倍率（観測高度の気圧による） */
  refractionScale: number;
}

/**
 * 太陽の方向など。viewerAltitudeM は大気差の気圧補正用（標高＋視点の高さ）。
 *
 * 注：大気散乱 LUT は 1AU の太陽で計算しているので、地球–太陽距離による
 * ±3.4% の明るさの変化は太陽面・直達光にも入れていない（空と整合させるため）。
 */
export function computeEphemeris(jdUT: number, obs: Observer, viewerAltitudeM: number): Ephemeris {
  const sun = sunEquatorial(jdUT);
  const last = lastDeg(jdUT, obs.lonDeg);
  const dir = equatorialToEnu(sun.raDeg, sun.decDeg, obs.latDeg, last);
  const pressureScale = Math.exp(-viewerAltitudeM / 8400);
  const app = applyRefraction(dir, 1010 * pressureScale);
  return {
    sunDir: dir,
    sunDirApparent: app,
    sunAngularRadius: sunAngularRadius(sun.distance),
    sunDistanceAU: sun.distance,
    sunIlluminance: SUN_TOA,
    whitePoint: SUN_WHITE,
    refractionScale: pressureScale,
  };
}
