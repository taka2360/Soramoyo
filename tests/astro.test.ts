import { describe, it, expect } from 'vitest';
import { sunEquatorial } from '../src/astro/sun';
import { deltaT, gmstDeg, equatorialToEnu, lastDeg } from '../src/astro/time';
import { refractionFromTrue, refractionFromApparent } from '../src/astro/refraction';

describe('太陽位置（Meeus 例題）', () => {
  it('例 25.a: 1992-10-13 0h TD の視赤経・視赤緯', () => {
    const jde = 2448908.5;
    const jdUT = jde - deltaT(jde) / 86400;
    const s = sunEquatorial(jdUT);
    expect(s.raDeg).toBeCloseTo(198.38083, 3);
    expect(s.decDeg).toBeCloseTo(-7.78507, 3);
    expect(s.distance).toBeCloseTo(0.99766, 4);
  });
});

describe('恒星時', () => {
  it('例 12.a: 1987-04-10 0h UT の GMST = 13h10m46.3668s', () => {
    expect(gmstDeg(2446895.5)).toBeCloseTo((13 + 10 / 60 + 46.3668 / 3600) * 15, 4);
  });
});

describe('座標変換', () => {
  it('天の北極は北の空、高度=緯度', () => {
    const v = equatorialToEnu(0, 90, 35.7, 123);
    expect(Math.asin(v[2]) * 180 / Math.PI).toBeCloseTo(35.7, 6);
    expect(v[0]).toBeCloseTo(0, 9);
    expect(v[1]).toBeGreaterThan(0);
  });
  it('南中時は真南', () => {
    const lst = lastDeg(2460000.5, 139.7);
    const v = equatorialToEnu(lst, -10, 35.7, lst);
    expect(v[0]).toBeCloseTo(0, 9);
    expect(v[1]).toBeLessThan(0);
  });
});

describe('大気差', () => {
  it('地平線で約 34′、高度 45° で約 1′', () => {
    expect(refractionFromApparent(0) * 60).toBeCloseTo(34.5, 0);
    expect(refractionFromTrue(45) * 60).toBeCloseTo(1.0, 1);
  });
  it('真→見かけ と 見かけ→真 がおおむね整合', () => {
    for (const h of [0, 2, 10, 30]) {
      const ha = h + refractionFromTrue(h);
      expect(ha - refractionFromApparent(ha)).toBeCloseTo(h, 2);
    }
  });
});
