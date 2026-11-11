/**
 * 時間管理。シミュレーション内の時刻（UTC ミリ秒）と再生速度を持つ。
 * 表示用の「現地時刻」は観測地点の UTC オフセット（固定値、夏時間なし）で求める。
 *
 * 描画フレームとシミュレーションの時間刻みは分離する：
 * フレームごとに advance() で進めた「目標時刻」に、雲シミュレーション（段階3以降）が
 * 自分の dt で追いつこうとする。追いつけない場合は speed を自動制限する（limitSpeed）。
 */
export interface Observer {
  latDeg: number;
  lonDeg: number;
  elevationM: number;
  utcOffsetHours: number;
  name: string;
}

export const TOKYO: Observer = {
  name: '東京',
  latDeg: 35.6895,
  lonDeg: 139.6917,
  elevationM: 40,
  utcOffsetHours: 9,
};

export class Clock {
  /** シミュレーション内の UTC 時刻 [ms] */
  utcMs: number;
  /** 再生倍率（1 = 実時間）。0 で停止。 */
  speed = 1;
  paused = false;
  /** シミュレーションが追いつける上限倍率（Infinity = 制限なし） */
  speedLimit = Infinity;

  constructor(utcMs: number = Date.now()) {
    this.utcMs = utcMs;
  }

  get effectiveSpeed(): number {
    return this.paused ? 0 : Math.min(this.speed, this.speedLimit);
  }

  /** 実時間 dtReal 秒ぶん進める。進めたシミュレーション秒を返す。 */
  advance(dtRealSec: number): number {
    const dt = dtRealSec * this.effectiveSpeed;
    this.utcMs += dt * 1000;
    return dt;
  }

  /** 観測地点の現地時刻（0〜24 時） */
  localHours(obs: Observer): number {
    const d = new Date(this.utcMs + obs.utcOffsetHours * 3600_000);
    return d.getUTCHours() + d.getUTCMinutes() / 60 + d.getUTCSeconds() / 3600;
  }

  /** 現地日付 YYYY-MM-DD */
  localDateString(obs: Observer): string {
    const d = new Date(this.utcMs + obs.utcOffsetHours * 3600_000);
    return d.toISOString().slice(0, 10);
  }

  localTimeString(obs: Observer): string {
    const d = new Date(this.utcMs + obs.utcOffsetHours * 3600_000);
    return d.toISOString().slice(11, 19);
  }

  /** 現地の日付と時刻（時、小数可）を指定して設定 */
  setLocal(obs: Observer, dateStr: string, hours: number): void {
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(dateStr.trim());
    if (!m) return;
    const midnightUtc = Date.UTC(+m[1], +m[2] - 1, +m[3]);
    this.utcMs = midnightUtc + (hours - obs.utcOffsetHours) * 3600_000;
  }

  /** ユリウス日（UT） */
  get julianDayUT(): number {
    return this.utcMs / 86400_000 + 2440587.5;
  }
}
