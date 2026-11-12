import { describe, it, expect } from 'vitest';
import { saturationAdjust, qvsat, exner, LV, CP } from '../src/sim/thermo';
import { makeGrid, stretchedLevels, buildBaseState, uniformLevels } from '../src/sim/grid';

describe('熱力学', () => {
  it('20℃・1000hPa の飽和混合比は約 14.7 g/kg', () => {
    expect(qvsat(293.15, 1e5) * 1000).toBeCloseTo(14.7, 0);
  });
  it('飽和調整は全水量と液水温位を保存し、飽和状態に落ち着く', () => {
    const p = 9e4, pi = exner(p);
    const [th, qv, qc] = saturationAdjust(295, 0.02, 0, p, pi);
    expect(qv + qc).toBeCloseTo(0.02, 10);
    expect(th - (LV / (CP * pi)) * qc).toBeCloseTo(295, 3);
    expect(qv).toBeCloseTo(qvsat(th * pi, p), 6);
  });
});

describe('格子と基本場', () => {
  it('ストレッチ格子は単調で上端に一致', () => {
    const zf = stretchedLevels(50, 1000, 1.05, 400, 15000);
    expect(zf[0]).toBe(0);
    expect(zf[zf.length - 1]).toBeCloseTo(15000, 6);
    for (let k = 1; k < zf.length; k++) expect(zf[k]).toBeGreaterThan(zf[k - 1]);
  });
  it('等温位大気の静水圧：密度は高さとともに減り、地表付近で約 1.16 kg/m³', () => {
    const g = makeGrid({ nx: 2, ny: 1, dx: 100, dy: 100, zf: uniformLevels(50, 200) });
    const b = buildBaseState(g, { surfacePressure: 1e5, theta: () => 300, qv: () => 0, u: () => 0, v: () => 0 });
    expect(b.rho0[0]).toBeCloseTo(1.16, 1);
    for (let k = 1; k < g.nz; k++) expect(b.rho0[k]).toBeLessThan(b.rho0[k - 1]);
  });
});

describe('不等間隔格子の移流の重み', () => {
  it('等間隔なら従来の 5 次・3 次風上の重みと一致する', async () => {
    const { verticalAdvectionWeights } = await import('../src/sim/grid');
    const g = makeGrid({ nx: 2, ny: 1, dx: 100, dy: 100, zf: uniformLevels(20, 50) });
    const a = verticalAdvectionWeights(g);
    const s5 = a.scalar[10];
    [2, -13, 47, 27, -3, 0].forEach((v, i) => expect(s5[i]).toBeCloseTo(v / 60, 10));
    [0, -3, 27, 47, -13, 2].forEach((v, i) => expect(s5[6 + i]).toBeCloseTo(v / 60, 10));
    const s3 = a.scalar[2];
    [0, -1, 5, 2, 0, 0].forEach((v, i) => expect(s3[i]).toBeCloseTo(v / 6, 10));
    const w5 = a.w[10];
    [2, -13, 47, 27, -3, 0].forEach((v, i) => expect(w5[i]).toBeCloseTo(v / 60, 10));
  });
  it('不等間隔でも 4 次までの多項式のセル平均から面の値を正確に再構成する', async () => {
    const { verticalAdvectionWeights } = await import('../src/sim/grid');
    const g = makeGrid({ nx: 2, ny: 1, dx: 100, dy: 100, zf: stretchedLevels(20, 100, 1.1, 300, 3000) });
    const a = verticalAdvectionWeights(g);
    const f = (z: number) => 1 + 2e-3 * z - 3e-6 * z * z + 1e-9 * z ** 3 + 2e-13 * z ** 4;
    const F = (z: number) => z + 1e-3 * z * z - 1e-6 * z ** 3 + 2.5e-10 * z ** 4 + 4e-14 * z ** 5;
    const avg = (k: number) => (F(g.zf[k + 1]) - F(g.zf[k])) / g.dzc[k];
    for (const k of [5, 12, 20]) {
      let vp = 0, vn = 0;
      for (let l = 0; l < 6; l++) {
        const c = Math.min(g.nz - 1, Math.max(0, k - 3 + l));
        vp += a.scalar[k][l] * avg(c);
        vn += a.scalar[k][6 + l] * avg(c);
      }
      expect(vp).toBeCloseTo(f(g.zf[k]), 8);
      expect(vn).toBeCloseTo(f(g.zf[k]), 8);
    }
  });
});
