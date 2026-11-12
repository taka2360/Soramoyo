import { CloudModel } from './model';
import { type SimCase, weismanKlemp, bomex, landDay, dycoms } from './cases';

export const SCENARIOS: Record<string, (latDeg: number) => SimCase> = {
  '夏の晴れた日・陸上（128², 100m）': (lat) => landDay({
    name: '夏の晴れた日（陸上）', description: '米国中部の夏の朝のサウンディング（ARM SGP）。日射で地面が温まり、昼前から積雲が湧く。',
    sounding: 'arm', nx: 128, dx: 100, dzMin: 40, ztop: 6000, soilWet: 0.8, windMs: 8, latDeg: lat,
  }),
  '夏の晴れた日・陸上（256², 100m）': (lat) => landDay({
    name: '夏の晴れた日（陸上・高解像度）', description: '同上、領域 25.6 km。',
    sounding: 'arm', nx: 256, dx: 100, dzMin: 40, ztop: 6000, soilWet: 0.8, windMs: 8, latDeg: lat,
  }),
  '乾いた高気圧の日（128², 100m）': (lat) => landDay({
    name: '乾いた高気圧の日', description: '同じ気温で水蒸気が 6 割。混合層は育つが雲はほとんどできない。',
    sounding: 'armDry', nx: 128, dx: 100, dzMin: 40, ztop: 6000, soilWet: 0.35, windMs: 5, latDeg: lat,
  }),
  '蒸し暑く不安定な日（256², 250m）': (lat) => landDay({
    name: '蒸し暑く不安定な日', description: '条件付き不安定（Weisman–Klemp 型）。午後に積乱雲が立つことがある。',
    sounding: 'wk', nx: 256, dx: 250, dzMin: 80, ztop: 16000, soilWet: 1.0, windMs: 4, latDeg: lat,
  }),
  '貿易風の海（BOMEX, 128², 100m）': () => bomex({ nx: 128, dx: 100, dz: 40 }),
  '海上の層積雲・雲海（DYCOMS-II, 128², 50m）': () => dycoms({ nx: 128, dx: 50, dzFine: 10 }),
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
  /** 1フレームに雲の計算へ割く GPU 時間の目安 [ms]（1ステップを塊に分けて、この範囲で進める） */
  simBudgetMs = 8;
  /** 追いつき中の1フレームの GPU 時間の目安 [ms]（描画はほぼ止まってよい） */
  catchUpBudgetMs = 120;
  /** 塊1つあたりの GPU 時間の実測（指数移動平均）[ms] */
  msPerChunk = 2;
  private measuring = false;
  /** 実効速度（シミュレーション秒 / 実秒）の平滑値 */
  effectiveSpeed = 0;
  /** 領域の原点（世界座標 m）。観測者が領域の中央に来るように始める */
  origin: [number, number] = [0, 0];
  private wind: [number, number] = [0, 0];
  /** 計算座標系の移動速度（≒領域の平均風）[m/s] */
  get windVec(): [number, number] { return this.wind; }
  private debt = 0;
  busy = '';

  constructor(private device: GPUDevice) {}

  /** シミュレーション開始時刻（UTC ms）。時計 = startUtc + model.time */
  startUtc = 0;

  /** 追いつき（時刻ジャンプや早送り）の目標時刻（UTC ms）。null なら通常 */
  catchUpTarget: number | null = null;
  /** 雲粒数濃度（地表日射の計算用） */
  nDrop = 3e8;

  /**
   * シナリオを読み込み、targetUtc の時刻の空になるよう準備する。
   * 日変化のシナリオは、その日の夜明け（現地 startHour）から計算を始めて targetUtc まで追いつく。
   */
  loadFor(name: string, latDeg: number, targetUtc: number, utcOffsetHours: number): void {
    const sc = SCENARIOS[name](latDeg);
    let start = targetUtc;
    if (sc.diurnal) {
      const localMs = targetUtc + utcOffsetHours * 3600_000;
      const midnightLocal = Math.floor(localMs / 86400_000) * 86400_000;
      start = midnightLocal + sc.diurnal.startHour * 3600_000 - utcOffsetHours * 3600_000;
      if (start > targetUtc) start -= 86400_000;
    }
    this.loadCase(name, sc, start);
    this.catchUpTarget = targetUtc > start + 1000 ? targetUtc : null;
  }

  load(name: string, latDeg: number, startUtc: number): void {
    this.loadCase(name, SCENARIOS[name](latDeg), startUtc);
  }

  private loadCase(name: string, sc: SimCase, startUtc: number): void {
    this.scenario = name;
    this.model?.destroy();
    this.simCase = sc;
    this.model = new CloudModel(this.device, this.simCase.config, this.simCase.initial);
    const g = this.simCase.config.grid;
    this.origin = [-g.nx * g.dx / 2, -g.ny * g.dy / 2];
    this.wind = this.simCase.config.frame ?? [0, 0];
    this.debt = 0;
    this.startUtc = startUtc;
  }

  /** 現在のシミュレーション時刻（UTC ms） */
  get simUtc(): number {
    return this.startUtc + (this.model?.time ?? 0) * 1000;
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
    m.nDrop = this.nDrop;
    if (this.catchUpTarget !== null) {
      // 追いつき中：時計に関係なく多めに進める
      const chunks = Math.max(1, Math.floor(this.catchUpBudgetMs / this.msPerChunk));
      const adv = this.run(m, chunks);
      if (this.simUtc >= this.catchUpTarget && !m.midStep) this.catchUpTarget = null;
      this.moveOrigin(adv);
      this.debt = 0;
      this.effectiveSpeed = this.effectiveSpeed * 0.9 + (dtReal > 0 ? adv / dtReal : 0) * 0.1;
      return adv;
    }
    this.debt = Math.min(this.debt + wantSec, Math.max(m.dt * this.maxStepsPerFrame * 2, 1));
    let adv = 0;
    // 時間の借りがあるか、ステップの途中なら、予算の範囲で塊を進める
    if (m.midStep || this.debt >= m.dt * 0.5) {
      const maxChunks = this.maxStepsPerFrame * CloudModel.CHUNKS_PER_STEP;
      const chunks = Math.min(maxChunks, Math.max(1, Math.floor(this.simBudgetMs / this.msPerChunk)));
      adv = this.run(m, chunks);
      this.debt -= adv;
    }
    this.moveOrigin(adv);
    const sp = dtReal > 0 ? adv / dtReal : 0;
    this.effectiveSpeed = this.effectiveSpeed * 0.95 + sp * 0.05;
    return adv;
  }

  /** 塊を進め、GPU 時間を測る（提出から完了までの時間。他の処理の待ちを含むので控えめな見積もり） */
  private run(m: CloudModel, chunks: number): number {
    const t0 = performance.now();
    const { used, advanced } = m.runChunks(chunks);
    if (!this.measuring) {
      this.measuring = true;
      void this.device.queue.onSubmittedWorkDone().then(() => {
        const ms = (performance.now() - t0) / used;
        this.msPerChunk = Math.min(200, Math.max(0.05, this.msPerChunk * 0.8 + ms * 0.2));
        this.measuring = false;
      });
    }
    return advanced;
  }

  private moveOrigin(sec: number): void {
    this.origin[0] += this.wind[0] * sec;
    this.origin[1] += this.wind[1] * sec;
  }

  /** 時計が新しい時刻に飛んだときの処理。近い未来なら追いつき、そうでなければ計算し直す */
  jumpTo(targetUtc: number, latDeg: number, utcOffsetHours: number): void {
    const now = this.simUtc;
    if (targetUtc > now && targetUtc - now < 18 * 3600_000) {
      this.catchUpTarget = targetUtc;
    } else {
      this.loadFor(this.scenario, latDeg, targetUtc, utcOffsetHours);
    }
  }
}
