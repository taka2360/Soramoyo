import { describe, it, expect } from 'vitest';
import { solarIlluminanceRgb, cie1931 } from '../src/atmosphere/spectrum';

describe('分光変換', () => {
  it('等色関数のピーク（ȳ は 555nm 付近で約1）', () => {
    expect(cie1931(555)[1]).toBeCloseTo(1.0, 1);
  });
  it('大気上端の太陽照度は約 13 万 lx', () => {
    const [r, g, b] = solarIlluminanceRgb();
    const Y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    expect(Y).toBeGreaterThan(120000);
    expect(Y).toBeLessThan(140000);
  });
});
