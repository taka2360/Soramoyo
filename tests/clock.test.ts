import { describe, it, expect } from 'vitest';
import { Clock, TOKYO } from '../src/core/clock';

describe('Clock', () => {
  it('現地時刻の設定と取得が往復する', () => {
    const c = new Clock(0);
    c.setLocal(TOKYO, '2026-11-11', 15.5);
    expect(c.localDateString(TOKYO)).toBe('2026-11-11');
    expect(c.localHours(TOKYO)).toBeCloseTo(15.5, 6);
    expect(new Date(c.utcMs).toISOString()).toBe('2026-11-11T06:30:00.000Z');
  });
  it('J2000.0 のユリウス日', () => {
    const c = new Clock(Date.UTC(2000, 0, 1, 12));
    expect(c.julianDayUT).toBeCloseTo(2451545.0, 9);
  });
  it('早送りと停止', () => {
    const c = new Clock(0);
    c.speed = 60;
    expect(c.advance(1)).toBe(60);
    c.paused = true;
    expect(c.advance(1)).toBe(0);
  });
});
