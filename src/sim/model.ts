import commonWgsl from '../shaders/sim/common.wgsl?raw';
import dynamicsWgsl from '../shaders/sim/dynamics.wgsl?raw';
import fftWgsl from '../shaders/sim/fft.wgsl?raw';
import tridiagWgsl from '../shaders/sim/tridiag.wgsl?raw';
import microWgsl from '../shaders/sim/micro.wgsl?raw';
import statsWgsl from '../shaders/sim/stats.wgsl?raw';
import { type Grid, type BaseState, type LargeScaleForcing, packColumn, COL_STRIDE } from './grid';
import { LV, CP } from './thermo';

/** ディスパッチ1回分（パイプライン名・バインドグループ・パラメータの段・グループ数） */
interface Dispatch { name: string; bg: GPUBindGroup; slot: number; x: number; y: number; z: number }

export const NFIELDS = 8;
export const F = { U: 0, V: 1, W: 2, TH: 3, QV: 4, QC: 5, QR: 6, QG: 7 } as const;
const NSTAT = 16;
const PARAM_SLOT = 256;

export interface ModelConfig {
  grid: Grid;
  base: BaseState;
  moist: boolean;
  rain: boolean;
  smagorinsky: boolean;
  /** 定数の粘性 [m²/s]（Smagorinsky と併用時は加算） */
  nuConst: number;
  cs: number;
  /** 乱流プラントル数（スカラーの拡散係数 = ν/Pr）。LES では 1/3、Straka テストは 1 */
  prandtl: number;
  /** スポンジ層の下端 [m] と最大減衰率 [1/s] */
  spongeZ: number;
  spongeRate: number;
  /** 固定の地表顕熱・潜熱フラックス [W/m²]（surfaceEnergy=false のとき） */
  shf: number;
  lhf: number;
  /** 大規模強制（沈降・放射・地衡風・ナッジング） */
  forcing?: LargeScaleForcing;
  /** コリオリ係数 f [1/s] */
  fcor?: number;
  /** 計算座標系の移動速度（地面に対する）[m/s] */
  frame?: [number, number];
  /** 地表摩擦：固定 u* [m/s]（0 ならバルク式）、粗度 z0 [m]（0 なら摩擦なし） */
  ustar?: number;
  z0?: number;
  /** 地表エネルギー収支（日射→地温→顕熱・潜熱）を使う */
  surfaceEnergy?: boolean;
  surface?: { albedo: number; soilWet: number; tDeep: number; cSoil: number };
  /** 水物質の正値保存リミッタ */
  positiveDefinite?: boolean;
  /** 氷相（温度で雲水/雲氷・雨/雪を分ける） */
  ice?: boolean;
  /** 雲の長波放射（雲頂冷却） */
  longwave?: boolean;
  /** DYCOMS-II の放射式の第3項（逆転層上の沈降に伴う項） */
  dycomsRadiation?: boolean;
  /** 時間刻みの上限 [s] と目標 CFL 数 */
  dtMax: number;
  cfl: number;
}

export interface LevelStats {
  sumTh: number; sumQv: number; sumQc: number; sumQr: number;
  maxU: number; maxV: number; maxW: number; minW: number;
  maxQc: number; maxQr: number; cloudCells: number; sumThl: number;
  minThp: number; maxThp: number;
}

export interface SurfaceStats { ts: number; h: number; le: number; sw: number }

export interface ModelStats {
  levels: LevelStats[];
  /** 全水量 [kg]（大気中 + 地表に落ちた雨 − 負値切り詰めで増えた分） */
  totalWater: number;
  airWater: number;
  precip: number;
  clipped: number;
  maxW: number; minW: number; maxU: number; maxV: number; maxQc: number; maxQr: number;
  minThp: number; maxThp: number;
  cloudBase: number | null; cloudTop: number | null; cloudCover: number;
  /** 液水温位の質量加重平均 [K] */
  meanThl: number;
  /** 雲水量の柱積算の領域平均 [g/m²] */
  lwp: number;
  /** 雲量（どこかの高さに雲がある柱の割合…の代わりに、最大の層雲量） */
  surface: SurfaceStats;
  time: number;
}

function isPow2(n: number) {
  return n > 0 && (n & (n - 1)) === 0;
}

/**
 * GPU 上の雲解像モデル。RK3（Wicker–Skamarock）で各段ごとに圧力射影を行う。
 * 状態は3本のバッファを回して使う（n, 段の中間, 出力）。
 */
export class CloudModel {
  readonly n: number;
  readonly cfg: ModelConfig;
  time = 0;
  steps = 0;
  dt: number;
  lastStats: ModelStats | null = null;

  private states: GPUBuffer[];
  private cur = 0;
  private colBuf: GPUBuffer;
  private cplx: GPUBuffer;
  private diag: GPUBuffer;
  private statsBuf: GPUBuffer;
  private paramBuf: GPUBuffer;
  private layout: GPUBindGroupLayout;
  private pipes = new Map<string, GPUComputePipeline>();
  private bindCache = new Map<string, GPUBindGroup>();
  private initialWater = 0;
  private evapAccum = 0;
  private lastStatTime = 0;

  constructor(private device: GPUDevice, cfg: ModelConfig, initial: Float32Array<ArrayBuffer>) {
    this.cfg = cfg;
    const g = cfg.grid;
    if (!isPow2(g.nx) || !isPow2(g.ny) || g.nx < 2) throw new Error('nx, ny は 2 の冪にしてください（FFT のため）');
    if (g.nx > 1024 || g.ny > 1024) throw new Error('nx, ny は 1024 以下');
    if (g.nz > 256) throw new Error('nz は 256 以下');
    this.n = g.nx * g.ny * g.nz;
    this.statsEvery = Math.min(32, Math.max(8, 2 ** Math.round(Math.log2(6e7 / this.n))));
    this.dt = cfg.dtMax;
    const n = this.n;

    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    this.states = [0, 1, 2].map((i) => device.createBuffer({ label: `state${i}`, size: n * NFIELDS * 4, usage: S }));
    device.queue.writeBuffer(this.states[0], 0, initial);
    this.colBuf = device.createBuffer({ label: 'column', size: g.nz * COL_STRIDE * 4, usage: S });
    device.queue.writeBuffer(this.colBuf, 0, packColumn(g, cfg.base, cfg.forcing));
    this.cplx = device.createBuffer({ label: 'cplx', size: n * 8, usage: S });
    const plane = g.nx * g.ny;
    this.diag = device.createBuffer({ label: 'diag', size: (7 * n + 8 * plane) * 4, usage: S });
    // 地表面温度の初期値 = 最下層の気温
    const ts0 = new Float32Array(plane).fill(cfg.base.th0[0] * cfg.base.pi0[0]);
    device.queue.writeBuffer(this.diag, (n + 2 * plane) * 4, ts0);
    // 正値保存リミッタの係数の初期値（使わないときも 1 にしておく）
    device.queue.writeBuffer(this.diag, 2 * n * 4, new Float32Array(4 * n).fill(1));
    // 「柱に凝結物あり」の印は、最初の微物理までは全列 1（初期状態に雲がある場合のため）
    device.queue.writeBuffer(this.diag, (n + 6 * plane) * 4, new Float32Array(plane).fill(1));
    this.statsBuf = device.createBuffer({ label: 'stats', size: (g.nz + 1) * NSTAT * 4, usage: S });
    this.paramBuf = device.createBuffer({ size: PARAM_SLOT * 3, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const st = (type: GPUBufferBindingType): GPUBindGroupLayoutEntry['buffer'] => ({ type });
    const C = GPUShaderStage.COMPUTE;
    this.layout = device.createBindGroupLayout({
      label: 'sim',
      entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 144 } },
        { binding: 1, visibility: C, buffer: st('read-only-storage') },
        { binding: 2, visibility: C, buffer: st('read-only-storage') },
        { binding: 3, visibility: C, buffer: st('read-only-storage') },
        { binding: 4, visibility: C, buffer: st('storage') },
        { binding: 5, visibility: C, buffer: st('storage') },
        { binding: 6, visibility: C, buffer: st('storage') },
        { binding: 7, visibility: C, buffer: st('storage') },
      ],
    });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const mk = (label: string, code: string, entryPoint: string) =>
      this.pipes.set(label, device.createComputePipeline({ label, layout: pl, compute: { module: device.createShaderModule({ label, code }), entryPoint } }));

    const dyn = commonWgsl + dynamicsWgsl;
    for (const ep of ['csActDilate', 'csNu', 'csPdRatio', 'csScalars', 'csMomentum', 'csDivergence', 'csProject']) mk(ep, dyn, ep);
    // MODE 0/1 = x 方向の実数 FFT（順/逆）、2 = y 方向の複素 FFT（kx = 0..nx/2 の列だけ）
    const fft = (N: number, dir: number, mode: number) =>
      commonWgsl +
      fftWgsl
        .replaceAll('__N__', String(N))
        .replaceAll('__LOG2N__', String(Math.log2(N)))
        .replaceAll('__DIR__', dir < 0 ? '-1.0' : '1.0')
        .replaceAll('__MODE__', String(mode));
    mk('fftXf', fft(g.nx, -1, 0), 'csFft');
    mk('fftXi', fft(g.nx, 1, 1), 'csFft');
    if (g.ny > 1) {
      mk('fftYf', fft(g.ny, -1, 2), 'csFft');
      mk('fftYi', fft(g.ny, 1, 2), 'csFft');
    }
    mk('tridiag', commonWgsl + tridiagWgsl, 'csTridiag');
    for (const ep of ['csColumnPre', 'csMicroCell', 'csSediment']) mk(ep, commonWgsl + microWgsl, ep);
    mk('stats', commonWgsl + statsWgsl, 'csStats');

    this.initialWater = NaN; // 最初の統計で確定
  }

  private bind(nI: number, cI: number, oI: number): GPUBindGroup {
    const key = `${nI}${cI}${oI}`;
    let bg = this.bindCache.get(key);
    if (!bg) {
      bg = this.device.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: this.paramBuf, size: 144 } },
          { binding: 1, resource: { buffer: this.colBuf } },
          { binding: 2, resource: { buffer: this.states[nI] } },
          { binding: 3, resource: { buffer: this.states[cI] } },
          { binding: 4, resource: { buffer: this.states[oI] } },
          { binding: 5, resource: { buffer: this.cplx } },
          { binding: 6, resource: { buffer: this.diag } },
          { binding: 7, resource: { buffer: this.statsBuf } },
        ],
      });
      this.bindCache.set(key, bg);
    }
    return bg;
  }

  /** 最新状態のバッファ（描画用） */
  get stateBuffer(): GPUBuffer {
    return this.states[this.cur];
  }
  get columnBuffer(): GPUBuffer {
    return this.colBuf;
  }
  get diagBuffer(): GPUBuffer {
    return this.diag;
  }

  /** 雲粒数濃度 [1/m³]（地表日射の雲の光学的厚さに使う。描画と同じ値） */
  nDrop = 3e8;
  /** 日射の条件（毎フレーム、天文計算から与える） */
  env = { cosZen: 0, swClear: 0 };

  private writeParams(dt: number): void {
    const g = this.cfg.grid;
    const c = this.cfg;
    const flags = (c.moist ? 1 : 0) | (c.smagorinsky ? 2 : 0) | (c.rain ? 4 : 0) |
      (c.surfaceEnergy ? 8 : 0) | (c.positiveDefinite !== false ? 16 : 0) |
      (c.ice ? 32 : 0) | (c.longwave ? 64 : 0) | (c.dycomsRadiation ? 128 : 0);
    const sf = c.surface ?? { albedo: 0.2, soilWet: 0.3, tDeep: 295, cSoil: 2e5 };
    const fr = c.frame ?? [0, 0];
    const buf = new ArrayBuffer(PARAM_SLOT * 3);
    [dt / 3, dt / 2, dt].forEach((dts, s) => {
      const u = new Uint32Array(buf, s * PARAM_SLOT, 36);
      const f = new Float32Array(buf, s * PARAM_SLOT, 36);
      u[0] = g.nx; u[1] = g.ny; u[2] = g.nz; u[3] = this.n;
      f[4] = g.dx; f[5] = g.dy; f[6] = dts; f[7] = dt;
      u[8] = s; u[9] = flags; f[10] = c.nuConst; f[11] = c.cs;
      f[12] = c.spongeZ; f[13] = c.spongeRate; f[14] = g.ztop; f[15] = c.prandtl;
      f[16] = c.shf; f[17] = c.lhf; f[18] = c.fcor ?? 0; f[19] = c.ustar ?? 0;
      f[20] = fr[0]; f[21] = fr[1]; f[22] = this.env.cosZen; f[23] = this.env.swClear;
      f[24] = sf.albedo; f[25] = sf.soilWet; f[26] = c.z0 ?? 0; f[27] = this.nDrop;
      f[28] = sf.tDeep; f[29] = 86400; f[30] = sf.cSoil; f[31] = 0;
    });
    this.device.queue.writeBuffer(this.paramBuf, 0, buf);
  }

  /** 検証用：0 より大きければ時間刻みをこの値に固定する [s] */
  fixedDt = 0;

  /**
   * 時間刻みの決め方（実行ごとに同じ列になるよう、読み出しのタイミングに依らない形）：
   *  統計はステップ番号が K（statsEvery）の倍数のステップだけで集計して読み出す。
   *  刻みはステップ s = m·K の始めにだけ決め直し、ちょうどステップ (m−1)K と (m−2)K の統計を使う。
   *  その統計がまだ届いていなければ、新しいステップを始めずに待つ（runChunks が早めに戻る）。
   *  読み出しは提出の次のフレームに届くので、1フレームに進められるのは少なくとも K ステップ。
   *  K は格子が小さい（1ステップが軽く、追いつきで1フレームに多くのステップを進める）ほど大きくする（8〜32）。
   *  統計は K〜2K−1 ステップ遅れなので、速さを2点から線形外挿し、刻みの増加も 1 回 1.2 倍までに抑える。
   */
  readonly statsEvery: number;
  /** 集計したステップの番号 → そのステップの後の CFL の速さ max(|u|/dx, |v|/dy, |w|/dz) [1/s] */
  private rates = new Map<number, number>();
  /** 刻みを決め直したときの値（次に決め直すまで使う） */
  private dtCur = 0;

  /** ステップ s の時間刻み。必要な統計が届いていなければ null */
  private chooseDt(s: number): number | null {
    if (this.fixedDt > 0) return this.fixedDt;
    const K = this.statsEvery;
    if (s < K) return Math.min(this.cfg.dtMax, 1);
    if (s % K !== 0) return this.dtCur;
    const r1 = this.rates.get(s - K);
    if (r1 === undefined) return null;
    const r0 = this.rates.get(s - 2 * K) ?? r1;
    const rate = Math.max(r1 + Math.max(0, r1 - r0), 1e-6);
    return Math.min(this.cfg.dtMax, this.cfg.cfl / rate, 1.2 * this.dtCur);
  }

  /** 途中まで進めたステップの残りの塊（空なら次は新しいステップ）。塊 = ディスパッチの並び */
  private pending: Dispatch[][] = [];
  private pendingDt = 0;
  /** 1ステップを分ける塊の数（RK3 の各段 ×（移流・発散 / 圧力・射影）＋ 微物理） */
  static readonly CHUNKS_PER_STEP = 7;
  /** ステップの途中か（途中でも描画用の最新状態 states[cur] は完全な状態のまま） */
  get midStep(): boolean {
    return this.pending.length > 0;
  }

  /** 新しいステップの塊を用意する。時間刻みを決める統計がまだ届いていなければ false */
  private beginStep(): boolean {
    const g = this.cfg.grid;
    const s = this.steps;
    const dt = this.chooseDt(s);
    if (dt === null) return false;
    // 使い終わった統計は捨てる
    for (const k of this.rates.keys()) if (k < s - 2 * this.statsEvery) this.rates.delete(k);
    if (s % this.statsEvery === 0) this.dtCur = dt;
    this.pendingCapture = s % this.statsEvery === 0;
    this.dt = dt;
    this.pendingDt = dt;
    this.writeParams(dt);
    const r = this.cur;
    const a = (r + 1) % 3, b = (r + 2) % 3;
    const perms: [number, number, number][] = [[r, r, a], [r, a, b], [r, b, a]];
    const g3 = [Math.ceil(g.nx / 8), Math.ceil(g.ny / 8), g.nz] as const;
    const g2 = [Math.ceil(g.nx / 8), Math.ceil(g.ny / 8), 1] as const;
    const g3z2 = [Math.ceil(g.nx / 8), Math.ceil(g.ny / 8), Math.ceil(g.nz / 2)] as const;
    // 実数 FFT：x は線2本で1ワークグループ（fft.wgsl の XGROUPS = 32768 個ずつ折り返す）、y と三重対角は kx = 0..nx/2
    const pairs = Math.ceil((g.ny * g.nz) / 2);
    const fftX = [Math.min(pairs, 32768), Math.ceil(pairs / 32768), 1] as const;
    const fftYx = Math.ceil((g.nx / 2 + 1) / Math.min(8, Math.max(1, Math.floor(2048 / g.ny))));
    const gHalf = [Math.ceil((g.nx / 2 + 1) / 8), Math.ceil(g.ny / 8), 1] as const;
    const D = (name: string, bg: GPUBindGroup, slot: number, x: number, y = 1, z = 1): Dispatch => ({ name, bg, slot, x, y, z });
    const chunks: Dispatch[][] = [];
    perms.forEach(([nI, cI, oI], stage) => {
      const bg = this.bind(nI, cI, oI);
      const A: Dispatch[] = [];
      if (stage === 0) A.push(D('csActDilate', bg, stage, ...g2), D('csNu', bg, stage, ...g3));
      if (stage === 2 && this.cfg.moist && this.cfg.positiveDefinite !== false) A.push(D('csPdRatio', bg, stage, ...g3));
      A.push(D('csScalars', bg, stage, ...g3z2), D('csMomentum', bg, stage, ...g3z2), D('csDivergence', bg, stage, ...g3));
      chunks.push(A);
      const B: Dispatch[] = [D('fftXf', bg, stage, ...fftX)];
      if (g.ny > 1) B.push(D('fftYf', bg, stage, fftYx, g.nz));
      B.push(D('tridiag', bg, stage, ...gHalf));
      if (g.ny > 1) B.push(D('fftYi', bg, stage, fftYx, g.nz));
      B.push(D('fftXi', bg, stage, ...fftX), D('csProject', bg, stage, Math.ceil(g.nx / 32), Math.ceil(g.ny / 2), g.nz));
      chunks.push(B);
    });
    const last = this.bind(r, b, a);
    const fin = [D('csColumnPre', last, 2, ...g2), D('csMicroCell', last, 2, ...g3), D('csSediment', last, 2, ...g2)];
    if (this.pendingCapture) fin.push(D('stats', last, 2, g.nz + 1));
    chunks.push(fin);
    this.pending = chunks;
    this.pendingNext = a;
    return true;
  }
  private pendingNext = 0;
  /** このステップで統計を集計して読み出すか */
  private pendingCapture = false;
  /** 統計の読み出し用バッファ（使い終わったものを再利用） */
  private readPool: GPUBuffer[] = [];

  private encode(pass: GPUComputePassEncoder, d: Dispatch): void {
    pass.setPipeline(this.pipes.get(d.name)!);
    pass.setBindGroup(0, d.bg, [d.slot * PARAM_SLOT]);
    pass.dispatchWorkgroups(d.x, d.y, d.z);
  }

  /**
   * 1ステップをカーネルごとに GPU のタイムスタンプで計測する（開発用。timestamp-query が必要）。
   * 戻り値：カーネル名 → 1ステップ中の合計 [ms]
   */
  async profileStep(): Promise<Record<string, number>> {
    const d = this.device;
    if (!d.features.has('timestamp-query')) throw new Error('timestamp-query が使えません');
    while (this.pending.length === 0 && !this.beginStep()) await new Promise((r) => setTimeout(r, 5));
    const capture = this.pendingCapture ? { step: this.steps, time: this.time + this.pendingDt } : null;
    const list = this.pending.flat();
    this.pending = [];
    const qs = d.createQuerySet({ type: 'timestamp', count: list.length * 2 });
    const res = d.createBuffer({ size: list.length * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const rd = d.createBuffer({ size: list.length * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = d.createCommandEncoder();
    list.forEach((x, i) => {
      const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
      this.encode(pass, x);
      pass.end();
    });
    enc.resolveQuerySet(qs, 0, list.length * 2, res, 0);
    enc.copyBufferToBuffer(res, 0, rd, 0, list.length * 16);
    const sb = capture ? this.copyStats(enc) : null;
    d.queue.submit([enc.finish()]);
    this.cur = this.pendingNext;
    this.time += this.pendingDt;
    this.steps++;
    if (capture && sb) this.readStats(sb, capture.step, capture.time);
    await rd.mapAsync(GPUMapMode.READ);
    const t = new BigInt64Array(rd.getMappedRange().slice(0));
    rd.unmap();
    const out: Record<string, number> = {};
    list.forEach((x, i) => { out[x.name] = (out[x.name] ?? 0) + Number(t[2 * i + 1] - t[2 * i]) / 1e6; });
    [qs, res, rd].forEach((o) => o.destroy());
    return out;
  }

  /**
   * 最大 maxChunks 個の塊を実行する（1つのコマンドバッファ）。ステップが完了したら時刻を進める。
   * 時間刻みを決める統計が届いていないときは、新しいステップを始めずに早めに戻る。
   * 戻り値：使った塊の数と、進んだシミュレーション時間 [s]。
   */
  runChunks(maxChunks: number): { used: number; advanced: number } {
    const d = this.device;
    let used = 0;
    let advanced = 0;
    let enc: GPUCommandEncoder | null = null;
    while (used < maxChunks) {
      if (this.pending.length === 0 && !this.beginStep()) break;
      enc ??= d.createCommandEncoder();
      const pass = enc.beginComputePass({ label: 'sim-chunk' });
      for (const x of this.pending.shift()!) this.encode(pass, x);
      pass.end();
      used++;
      if (this.pending.length === 0) {
        // ステップ完了。集計したステップなら統計を写す。次のステップの書き込み先に影響しないよう、ここで一旦提出する
        const capture = this.pendingCapture;
        const sb = capture ? this.copyStats(enc) : null;
        d.queue.submit([enc.finish()]);
        enc = null;
        const step = this.steps;
        this.cur = this.pendingNext;
        this.time += this.pendingDt;
        this.steps++;
        advanced += this.pendingDt;
        if (sb) this.readStats(sb, step, this.time);
      }
    }
    if (enc) d.queue.submit([enc.finish()]);
    return { used, advanced };
  }

  /**
   * count ステップ進める（途中のステップがあれば、まずそれを終える）。
   * 統計待ちで進めなくなったらそこでやめる。進めたシミュレーション時間 [s] を返す。
   */
  step(count: number): number {
    let advanced = 0;
    for (let s = 0; s < count; s++) {
      const n = this.pending.length > 0 ? this.pending.length : CloudModel.CHUNKS_PER_STEP;
      const r = this.runChunks(n);
      advanced += r.advanced;
      if (r.used < n) break;
    }
    return advanced;
  }

  private copyStats(enc: GPUCommandEncoder): GPUBuffer {
    const size = (this.cfg.grid.nz + 1) * NSTAT * 4;
    const buf = this.readPool.pop() ?? this.device.createBuffer({ label: 'stats-read', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(this.statsBuf, 0, buf, 0, size);
    return buf;
  }

  private readStats(buf: GPUBuffer, step: number, time: number): void {
    const g = this.cfg.grid;
    const base = this.cfg.base;
    buf.mapAsync(GPUMapMode.READ).then(() => {
      const a = new Float32Array(buf.getMappedRange().slice(0));
      buf.unmap();
      this.readPool.push(buf);
      const levels: LevelStats[] = [];
      const area = g.nx * g.dx * g.ny * g.dy;
      const cellArea = g.dx * g.dy;
      let airWater = 0, thlMass = 0, mass = 0;
      let cloudBase: number | null = null, cloudTop: number | null = null, cover = 0;
      const agg = { maxW: -1e9, minW: 1e9, maxU: 0, maxV: 0, maxQc: 0, maxQr: 0, minThp: 1e9, maxThp: -1e9 };
      for (let k = 0; k < g.nz; k++) {
        const o = k * NSTAT;
        const L: LevelStats = {
          sumTh: a[o], sumQv: a[o + 1], sumQc: a[o + 2], sumQr: a[o + 3],
          maxU: a[o + 4], maxV: a[o + 5], maxW: a[o + 6], minW: a[o + 7],
          maxQc: a[o + 8], maxQr: a[o + 9], cloudCells: a[o + 10], sumThl: a[o + 11],
          minThp: a[o + 12], maxThp: a[o + 13],
        };
        levels.push(L);
        const m = base.rho0[k] * g.dzc[k] * cellArea;
        airWater += (L.sumQv + L.sumQc + L.sumQr) * m;
        thlMass += L.sumThl * m;
        mass += m * g.nx * g.ny;
        agg.maxW = Math.max(agg.maxW, L.maxW); agg.minW = Math.min(agg.minW, L.minW);
        agg.maxU = Math.max(agg.maxU, L.maxU); agg.maxV = Math.max(agg.maxV, L.maxV);
        agg.maxQc = Math.max(agg.maxQc, L.maxQc); agg.maxQr = Math.max(agg.maxQr, L.maxQr);
        agg.minThp = Math.min(agg.minThp, L.minThp); agg.maxThp = Math.max(agg.maxThp, L.maxThp);
        if (L.cloudCells > 0) {
          if (cloudBase === null) cloudBase = g.zf[k];
          cloudTop = g.zf[k + 1];
          cover = Math.max(cover, L.cloudCells / (g.nx * g.ny));
        }
      }
      const so = g.nz * NSTAT;
      const precip = a[so] * cellArea;
      const clipped = a[so + 1] * cellArea;
      const np = g.nx * g.ny;
      const surface: SurfaceStats = { ts: a[so + 2] / np, h: a[so + 3] / np, le: a[so + 4] / np, sw: a[so + 5] / np };
      let lwp = 0;
      for (let k = 0; k < g.nz; k++) lwp += (levels[k].sumQc / np) * base.rho0[k] * g.dzc[k];
      // 地表からの蒸発で入った水を差し引いて収支を見る（エネルギー収支モードでは積算値）
      this.evapAccum += (this.cfg.surfaceEnergy ? surface.le : this.cfg.lhf) / LV * area * (time - this.lastStatTime);
      this.lastStatTime = time;
      const evap = this.evapAccum;
      const total = airWater + precip - clipped - evap;
      if (Number.isNaN(this.initialWater)) this.initialWater = total;
      this.lastStats = {
        levels, totalWater: total, airWater, precip, clipped, ...agg,
        cloudBase, cloudTop, cloudCover: cover, meanThl: thlMass / mass, lwp: lwp * 1000, surface, time,
      };
      // 時間刻みに使う CFL の速さ
      const dzMin = Math.min(...g.dzc);
      this.rates.set(step, Math.max(agg.maxU / g.dx, g.ny > 1 ? agg.maxV / g.dy : 0, Math.max(agg.maxW, -agg.minW) / dzMin));
    }).catch(() => { /* 破棄後の読み出しは無視 */ });
  }

  /** 全水量の初期値からの相対変化 */
  get waterDrift(): number {
    const s = this.lastStats;
    if (!s || !(this.initialWater > 0)) return 0;
    return (s.totalWater - this.initialWater) / this.initialWater;
  }

  destroy(): void {
    // 送信済みのコマンドが使い終わってから解放する
    void this.device.queue.onSubmittedWorkDone().then(() => [...this.states, this.colBuf, this.cplx, this.diag, this.statsBuf, this.paramBuf, ...this.readPool].forEach((b) => b.destroy()));
  }
}

export { LV, CP };
