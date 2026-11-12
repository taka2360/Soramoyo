import type { Observer } from '../core/clock';
import { sunEquatorial, sunAngularRadius } from './sun';
import { equatorialToEnu, lastDeg, j2000ToEnuMatrix } from './time';
import { applyRefraction } from './refraction';
import { moonPosition, topocentric, moonAngularRadius, moonPhase, moonIlluminanceLux } from './moon';
import { solarIlluminanceRgb } from '../atmosphere/spectrum';

const SUN_TOA = solarIlluminanceRgb();
const SUN_Y = 0.2126 * SUN_TOA[0] + 0.7152 * SUN_TOA[1] + 0.0722 * SUN_TOA[2];
const SUN_WHITE = (() => {
  const m = (SUN_TOA[0] + SUN_TOA[1] + SUN_TOA[2]) / 3;
  return SUN_TOA.map((c) => c / m);
})();
const AU_KM = 149597870.7;

export interface Ephemeris {
  sunDir: [number, number, number];
  sunDirApparent: [number, number, number];
  sunAngularRadius: number;
  sunDistanceAU: number;
  /** 地球–太陽距離による明るさの倍率 (1AU/d)²（sunIlluminance には掛け済み） */
  sunDistScale: number;
  sunIlluminance: number[];
  whitePoint: number[];
  /** 大気差の倍率（観測高度の気圧による） */
  refractionScale: number;

  /** 月（観測地点から見た真の方向・見かけの方向・視半径） */
  moonDir: [number, number, number];
  moonDirApparent: [number, number, number];
  moonAngularRadius: number;
  moonPhaseAngleDeg: number;
  moonIlluminatedFraction: number;
  /** 大気上端での月の照度（線形 sRGB, lx）。色は太陽光と同じと近似 */
  moonIlluminance: number[];
  /** 月面輝度 = 月の照度 × この係数 × LS(μ0,μ)（Lommel–Seeliger 則の円盤平均で正規化） */
  moonLuminanceScale: number;
  /** 地球照で光る月の暗い側の輝度 [cd/m²] */
  earthshine: number;
  /** J2000 赤道座標 → ENU の回転行列（行優先 3x3） */
  j2000ToEnu: number[];
}

/** Lommel–Seeliger 則の、見える円盤上での平均値（位相角 i） */
function lsDiskMean(phaseRad: number): number {
  const s = [Math.sin(phaseRad), 0, Math.cos(phaseRad)]; // 月から見た太陽方向（観測者方向が +z）
  let sum = 0, cnt = 0;
  const N = 48;
  for (let a = 0; a < N; a++) {
    for (let b = 0; b < N; b++) {
      const x = ((a + 0.5) / N) * 2 - 1, y = ((b + 0.5) / N) * 2 - 1;
      const r2 = x * x + y * y;
      if (r2 > 1) continue;
      const mu = Math.sqrt(1 - r2);
      const mu0 = x * s[0] + mu * s[2];
      if (mu0 > 0) sum += mu0 / (mu0 + mu);
      cnt++;
    }
  }
  return sum / cnt;
}

/**
 * 太陽・月の方向など。viewerAltitudeM は大気差の気圧補正用（標高＋視点の高さ）。
 *
 * 地球–太陽距離による明るさの変化（±3.4%）：sunIlluminance に (1AU/d)² を掛け、
 * 1AU で計算した大気散乱 LUT の出力にも描画時に同じ倍率（sunDistScale）を掛ける。
 * 月の照度は月の距離・位相で別に求めている（太陽距離の変化は無視、±3%）。
 */
export function computeEphemeris(jdUT: number, obs: Observer, viewerAltitudeM: number): Ephemeris {
  const sun = sunEquatorial(jdUT);
  const last = lastDeg(jdUT, obs.lonDeg);
  const dir = equatorialToEnu(sun.raDeg, sun.decDeg, obs.latDeg, last);
  const pressureScale = Math.exp(-viewerAltitudeM / 8400);
  const app = applyRefraction(dir, 1010 * pressureScale);

  const moon = moonPosition(jdUT);
  const moonGeo = equatorialToEnu(moon.raDeg, moon.decDeg, obs.latDeg, last);
  const moonTopo = topocentric(moonGeo, moon.distanceKm, obs.elevationM);
  const moonApp = applyRefraction(moonTopo, 1010 * pressureScale);
  const ph = moonPhase(dir, sun.distance * AU_KM, moonGeo, moon.distanceKm);
  const lux = moonIlluminanceLux(ph.phaseAngleDeg, moon.distanceKm);
  const moonColor = SUN_TOA.map((c) => (c / SUN_Y) * lux);
  const angR = moonAngularRadius(moon.distanceKm);
  const omega = Math.PI * angR * angR;
  const mean = Math.max(lsDiskMean((ph.phaseAngleDeg * Math.PI) / 180), 1e-4);
  // 地球照：月から見た地球の輝面比 × 新月付近での月面輝度（約 0.5 cd/m²、満月の約 1/5000）
  const earthFrac = (1 - Math.cos((ph.phaseAngleDeg * Math.PI) / 180)) / 2;

  return {
    sunDir: dir,
    sunDirApparent: app,
    sunAngularRadius: sunAngularRadius(sun.distance),
    sunDistanceAU: sun.distance,
    sunDistScale: 1 / (sun.distance * sun.distance),
    sunIlluminance: SUN_TOA.map((c) => c / (sun.distance * sun.distance)),
    whitePoint: SUN_WHITE,
    refractionScale: pressureScale,
    moonDir: moonTopo,
    moonDirApparent: moonApp,
    moonAngularRadius: angR,
    moonPhaseAngleDeg: ph.phaseAngleDeg,
    moonIlluminatedFraction: ph.illuminated,
    moonIlluminance: moonColor,
    moonLuminanceScale: 1 / (omega * mean),
    earthshine: 0.5 * earthFrac,
    j2000ToEnu: j2000ToEnuMatrix(jdUT, obs.latDeg, obs.lonDeg),
  };
}
