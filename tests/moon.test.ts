import { describe, it, expect } from 'vitest';
import { moonPosition, moonIlluminanceLux } from '../src/astro/moon';
import { deltaT, j2000ToEnuMatrix, applyMat3, raDecToVec, equatorialToEnu, lastDeg } from '../src/astro/time';

describe('月の位置（Meeus 例題 47.a: 1992-04-12 0h TD）', () => {
  const jde = 2448724.5;
  const m = moonPosition(jde - deltaT(jde) / 86400);
  it('視黄経・黄緯・距離', () => {
    expect(m.lonDeg).toBeCloseTo(133.167265, 2);
    expect(m.latDeg).toBeCloseTo(-3.229126, 2);
    expect(m.distanceKm).toBeCloseTo(368409.7, -1);
  });
  it('視赤経・視赤緯', () => {
    expect(m.raDeg).toBeCloseTo(134.688470, 2);
    expect(m.decDeg).toBeCloseTo(13.768368, 2);
  });
});

describe('月の明るさ', () => {
  it('満月（位相角 0）で約 0.27 lx', () => {
    expect(moonIlluminanceLux(0, 384400)).toBeGreaterThan(0.2);
    expect(moonIlluminanceLux(0, 384400)).toBeLessThan(0.35);
  });
});

describe('歳差・章動（Meeus 例題 21.b: θ Persei）', () => {
  it('J2000 → 2028-11-13.19 TD の赤道座標と一致（固有運動ぶん 0.01° 以内）', () => {
    const jde = 2462088.69;
    const jd = jde - deltaT(jde) / 86400;
    const lat = 35, lon = 139;
    const M = j2000ToEnuMatrix(jd, lat, lon);
    const v = applyMat3(M, raDecToVec(41.054063, 49.22775));
    // 期待値：その日の平均赤道座標（Meeus の結果）を、同じ恒星時で ENU にしたもの（章動の差は 0.005° 程度）
    const w = equatorialToEnu(41.547214, 49.348483, lat, lastDeg(jd, lon));
    const ang = Math.acos(Math.min(1, v[0] * w[0] + v[1] * w[1] + v[2] * w[2])) * 180 / Math.PI;
    expect(ang).toBeLessThan(0.012);
  });
  it('北極星の高度はほぼ緯度に等しい', () => {
    const M = j2000ToEnuMatrix(2461356.5, 35.7, 139.7);
    const v = applyMat3(M, raDecToVec(37.954, 89.264));
    expect(Math.abs(Math.asin(v[2]) * 180 / Math.PI - 35.7)).toBeLessThan(0.8);
  });
});
