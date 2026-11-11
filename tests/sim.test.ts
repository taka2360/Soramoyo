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
