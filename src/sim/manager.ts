import { CloudModel } from './model';
import { type SimCase, armCumulus, weismanKlemp, caseMeanWind } from './cases';

export const SCENARIOS: Record<string, () => SimCase> = {
  '晴天積雲（地表加熱, 128², 100m）': () => armCumulus({ nx: 128, dx: 100, dzMin: 40, shf: 120, lhf: 400 }),
  '晴天積雲（地表加熱, 256², 100m）': () => armCumulus({ nx: 256, dx: 100, dzMin: 40, shf: 120, lhf: 400 }),
  '積乱雲（暖気泡, 128², 500m）': () => weismanKlemp({ nx: 128, dx: 500, dzMin: 250, shear: 10, qv0: 0.014 }),
  '積乱雲（暖気泡, 256², 250m）': () => weismanKlemp({ nx: 256, dx: 250, dzMin: 150, shear: 10, qv0: 0.014 }),
};

/**
 * 雲シミュレーションと時計の橋渡し。
 * 時計の早送り速度ぶんだけモデルを進めようとし、追いつかなければ実効速度が下がる。
 * 計算領域は平均風で動く座標系なので、世界座標での領域の位置も時間とともに動かす。
 */
export class SimManager {
  model: CloudModel | null = null;
  simCase: SimCase | null = null;
  scenario = Object.keys(SCENARIOS)[0];
  running = true;
  maxStepsPerFrame = 3;
  /** 実効速度（シミュレーション秒 / 実秒）の平滑値 */
  effectiveSpeed = 0;
  /** 領域の原点（世界座標 m）。観測者が領域の中央に来るように始める */
  origin: [number, number] = [0, 0];
  private wind: [number, number] = [0, 0];
  private debt = 0;
  busy = '';

  constructor(private device: GPUDevice) {}

  load(name = this.scenario): void {
    this.scenario = name;
    this.model?.destroy();
    this.simCase = SCENARIOS[name]();
    this.model = new CloudModel(this.device, this.simCase.config, this.simCase.initial);
    const g = this.simCase.config.grid;
    this.origin = [-g.nx * g.dx / 2, -g.ny * g.dy / 2];
    this.wind = caseMeanWind(this.simCase.name);
    this.debt = 0;
  }

  unload(): void {
    this.model?.destroy();
    this.model = null;
    this.simCase = null;
  }

  /**
   * 1フレームぶん進める。wantSec = 時計が進めたいシミュレーション秒。
   * 実際に進んだ秒数を返す（時計はこれに合わせて進める）。
   */
  frame(wantSec: number, dtReal: number): number {
    const m = this.model;
    if (!m || !this.running || this.busy) {
      return wantSec;
    }
    this.debt = Math.min(this.debt + wantSec, Math.max(m.dt * this.maxStepsPerFrame * 2, 1));
    let adv = 0;
    let n = 0;
    while (this.debt >= m.dt * 0.5 && n < this.maxStepsPerFrame) {
      const a = m.step(1);
      this.debt -= a;
      adv += a;
      n++;
    }
    this.moveOrigin(adv);
    const sp = dtReal > 0 ? adv / dtReal : 0;
    this.effectiveSpeed = this.effectiveSpeed * 0.95 + sp * 0.05;
    return adv;
  }

  private moveOrigin(sec: number): void {
    this.origin[0] += this.wind[0] * sec;
    this.origin[1] += this.wind[1] * sec;
  }

  /** 描画せずに早回しする（雲が育つまで待つ用）。進めた秒数を返す */
  async spinUp(seconds: number, onProgress: (f: number) => void): Promise<number> {
    const m = this.model;
    if (!m) return 0;
    this.busy = '早回し中';
    let done = 0;
    try {
      while (done < seconds && this.model === m) {
        done += m.step(10);
        await this.device.queue.onSubmittedWorkDone();
        onProgress(done / seconds);
      }
    } finally {
      this.busy = '';
    }
    this.moveOrigin(done);
    return done;
  }
}
