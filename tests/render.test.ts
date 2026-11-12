import { describe, it, expect } from 'vitest';
import { computeEphemeris } from '../src/astro/scene';
import { TOKYO } from '../src/core/clock';
import { OceanWaves, N_WAVES, peakWavelength } from '../src/render/oceanWaves';

const jd = (y: number, m: number, d: number) => Date.UTC(y, m - 1, d) / 86400_000 + 2440587.5;

describe('地球–太陽距離による明るさ', () => {
  it('近日点（1月初め）で約 +3.4%、遠日点（7月初め）で約 −3.3%', () => {
    expect(computeEphemeris(jd(2026, 1, 3), TOKYO, 40).sunDistScale).toBeCloseTo(1.034, 2);
    expect(computeEphemeris(jd(2026, 7, 6), TOKYO, 40).sunDistScale).toBeCloseTo(0.967, 2);
  });
  it('太陽照度にも同じ倍率が掛かる', () => {
    const a = computeEphemeris(jd(2026, 1, 3), TOKYO, 40);
    const b = computeEphemeris(jd(2026, 7, 6), TOKYO, 40);
    expect(a.sunIlluminance[1] / b.sunIlluminance[1]).toBeCloseTo(a.sunDistScale / b.sunDistScale, 6);
  });
});

describe('海の波の表', () => {
  it('分散の配分は合計 1、ピーク波長より十分長い波には配らない', () => {
    const w = new OceanWaves();
    const d = w.update(0.016, 7);
    let sum = 0;
    for (let i = 0; i < N_WAVES; i++) {
      sum += d[i * 4 + 3];
      if (d[i * 4] > peakWavelength(7) * 1.5) expect(d[i * 4 + 3]).toBe(0);
    }
    expect(sum).toBeCloseTo(1, 5);
  });
  it('長時間たっても位相は連続（1フレームの変化が ω·dt に一致）', () => {
    const w = new OceanWaves();
    for (let i = 0; i < 4000; i++) w.update(1, 7); // 1 時間以上
    const before = Float64Array.from(w.data);
    const after = w.update(0.016, 7);
    for (let i = 0; i < N_WAVES; i++) {
      const omega = Math.sqrt((9.81 * 2 * Math.PI) / before[i * 4]);
      let dph = after[i * 4 + 2] - before[i * 4 + 2];
      dph = ((dph % (2 * Math.PI)) + 3 * Math.PI) % (2 * Math.PI) - Math.PI;
      expect(Math.abs(dph - ((omega * 0.016 + Math.PI) % (2 * Math.PI) - Math.PI))).toBeLessThan(1e-4);
      expect(after[i * 4]).toBe(before[i * 4]); // 波長は風速・時間で変わらない
    }
  });
});
