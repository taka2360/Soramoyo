import cloudCommonWgsl from '../shaders/clouds/cloudcommon.wgsl?raw';
import cloudVolWgsl from '../shaders/clouds/cloudvol.wgsl?raw';
import shadowWgsl from '../shaders/clouds/shadow.wgsl?raw';
import noiseWgsl from '../shaders/clouds/noise.wgsl?raw';
import type { CloudModel } from '../sim/model';
import type { Grid } from '../sim/grid';

export interface CloudSettings {
  /** 雲粒の数濃度 [個/cm³]（海洋性 ~100、大陸性 ~300〜1000） */
  dropletsPerCc: number;
  /** 細部合成の強さ（0 = 計算格子のまま） */
  detail: number;
  /** 雲を描く最大距離 [km] */
  maxDistKm: number;
  /** レイマーチの刻み：距離に対する比（小さいほど細かく重い）と最小刻み [m] */
  stepScale: number;
  minStepM: number;
  /** レイマーチの最大ステップ数 */
  maxSteps: number;
  /** 薄明光線（雲の影）の視線上のサンプル数 */
  shaftSamples: number;
  /** 細部合成の細かい段（近くの 25〜90 m の凹凸）を使うか */
  detailHQ: boolean;
}

/**
 * 計算モデルの状態から、描画用の雲ボリューム（密度・太陽方向の光学的厚さ）を作る。
 * 計算領域は水平に周期的なので、そのままタイル状に並べて地平線まで埋める（仮実装）。
 */
export class CloudVolume {
  readonly uniform: GPUBuffer;
  readonly samplerRepeat: GPUSampler;
  /** ノイズ用（3方向とも繰り返し） */
  readonly samplerNoise: GPUSampler;
  noise: GPUTexture;
  vol: GPUTexture;
  /** セル中心の風（領域に対する相対値）[m/s] */
  vel: GPUTexture;
  shadow: GPUTexture;
  /** 影の続き（y 反転・xy 反転の太陽方向）。遠方タイルの反転用 */
  shadow2: GPUTexture;
  zmap: GPUTexture;
  /** テクスチャを作り直すたびに増える（バインドグループ再作成の判定用） */
  generation = 1;
  enabled = false;

  private dims: GPUBuffer;
  private volPipe: GPUComputePipeline;
  private shadowPipe: GPUComputePipeline;
  private model: CloudModel | null = null;
  private grid: Grid | null = null;
  private cdata = new Float32Array(28);
  slabBot = 0;
  slabTop = 3000;

  constructor(private device: GPUDevice) {
    this.uniform = device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dims = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.samplerRepeat = device.createSampler({
      magFilter: 'linear', minFilter: 'linear',
      addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'clamp-to-edge',
    });
    this.samplerNoise = device.createSampler({
      magFilter: 'linear', minFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', addressModeW: 'repeat',
    });
    const mod = (label: string, code: string) => device.createShaderModule({ label, code });
    this.volPipe = device.createComputePipeline({
      label: 'toVolume', layout: 'auto', compute: { module: mod('cloudvol', cloudCommonWgsl + cloudVolWgsl), entryPoint: 'csToVolume' },
    });
    this.shadowPipe = device.createComputePipeline({
      label: 'cloudShadow', layout: 'auto', compute: { module: mod('shadow', cloudCommonWgsl + shadowWgsl), entryPoint: 'csShadow' },
    });

    // ノイズ（起動時に1回）
    this.noise = device.createTexture({
      label: 'cloudNoise', size: [128, 128, 128], dimension: '3d', format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    const noisePipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod('noise', noiseWgsl), entryPoint: 'csNoise' } });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(noisePipe);
    pass.setBindGroup(0, device.createBindGroup({ layout: noisePipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: this.noise.createView() }] }));
    pass.dispatchWorkgroups(32, 32, 32);
    pass.end();
    device.queue.submit([enc.finish()]);

    // モデルが無いときのダミー
    this.vol = this.tex3d(1, 1, 1);
    this.vel = this.tex3d(1, 1, 1);
    this.shadow = this.tex3d(1, 1, 1);
    this.shadow2 = this.tex3d(1, 1, 1);
    this.zmap = this.makeZmap(null);
  }

  private tex3d(x: number, y: number, z: number): GPUTexture {
    return this.device.createTexture({
      size: [x, y, z], dimension: '3d', format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
  }

  /** 高さ → 小数の層番号 の対応表（1024 点、0〜ztop） */
  private makeZmap(g: Grid | null): GPUTexture {
    const t = this.device.createTexture({ size: [1024, 1], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const a = new Float32Array(1024);
    if (g) {
      for (let i = 0; i < 1024; i++) {
        const z = (i / 1023) * g.ztop;
        let kf = 0;
        if (z <= g.zc[0]) kf = 0;
        else if (z >= g.zc[g.nz - 1]) kf = g.nz - 1;
        else {
          for (let k = 1; k < g.nz; k++) {
            if (z <= g.zc[k]) { kf = k - 1 + (z - g.zc[k - 1]) / (g.zc[k] - g.zc[k - 1]); break; }
          }
        }
        a[i] = kf;
      }
    }
    this.device.queue.writeTexture({ texture: t }, a, { bytesPerRow: 4096 }, [1024, 1]);
    return t;
  }

  attach(model: CloudModel | null): void {
    const old = [this.vol, this.vel, this.shadow, this.shadow2, this.zmap];
    this.model = model;
    if (model) {
      const g = model.cfg.grid;
      this.grid = g;
      this.vol = this.tex3d(g.nx, g.ny, g.nz);
      this.vel = this.tex3d(g.nx, g.ny, g.nz);
      // 影（太陽方向の光学的厚さ）はなめらかな量なので水平は半分の解像度で（近くはレイマーチで直接たどる）
      this.shadow = this.tex3d(g.nx / 2, g.ny > 1 ? g.ny / 2 : 1, g.nz);
      this.shadow2 = this.tex3d(g.nx / 2, g.ny > 1 ? g.ny / 2 : 1, g.nz);
      this.zmap = this.makeZmap(g);
      const d = new ArrayBuffer(32);
      new Uint32Array(d, 0, 4).set([g.nx, g.ny, g.nz, model.n]);
      new Float32Array(d, 16, 4).set([g.dx, g.dy, model.cfg.ice ? 1 : 0, 2]);
      this.device.queue.writeBuffer(this.dims, 0, d);
      this.enabled = true;
    } else {
      this.grid = null;
      this.vol = this.tex3d(1, 1, 1);
      this.vel = this.tex3d(1, 1, 1);
      this.shadow = this.tex3d(1, 1, 1);
      this.shadow2 = this.tex3d(1, 1, 1);
      this.zmap = this.makeZmap(null);
      this.enabled = false;
    }
    this.generation++;
    this.lastSteps = -1;
    this.lastShadowTime = -1e9;
    this.shadowSlicesLeft = 0;
    void this.device.queue.onSubmittedWorkDone().then(() => old.forEach((t) => t.destroy()));
  }

  /** 毎フレームの定数（領域の位置、太陽方向など） */
  private frameNo = 0;
  private lastSun = [0, 0, 0];
  private lastSteps = -1;
  writeUniform(origin: [number, number], sunDir: number[], time: number, s: CloudSettings, groundAlbedo: number, histAlpha: number): void {
    const g = this.grid;
    const c = this.cdata;
    const lx = g ? g.nx * g.dx : 1, ly = g ? g.ny * g.dy : 1;
    // 雲粒の代表直径：雲水量 0.5 g/m³ のときの有効半径から
    const n = s.dropletsPerCc * 1e6;
    const rv = Math.cbrt((3 * 0.5e-3) / (4 * Math.PI * 1000 * n));
    const dUm = Math.min(50, Math.max(5, 2 * 1.1 * rv * 1e6));
    // ノイズの周期は領域の大きさを割り切る値に（継ぎ目を出さないため）
    const period = lx / Math.max(1, Math.round(lx / 1500));
    c.set([origin[0], origin[1], lx, ly], 0);
    c.set([g ? g.ztop : 1, this.slabBot, this.slabTop, g ? g.nz : 1], 4);
    c.set([n, s.detail, time, s.maxDistKm * 1000], 8);
    this.frameNo = (this.frameNo + 1) % 1024;
    this.sunNow = sunDir;
    c.set([sunDir[0], sunDir[1], sunDir[2], this.frameNo], 12);
    c.set([period, dUm, groundAlbedo, this.enabled ? 1 : 0], 16);
    c.set([histAlpha, g ? g.dx : 100, s.stepScale, s.minStepM], 20);
    c.set([s.shaftSamples, s.detailHQ ? 1 : 0, s.maxSteps, 0], 24);
    this.device.queue.writeBuffer(this.uniform, 0, c);
  }

  private sunNow: number[] = [0, 0, 1];
  private sliceBuf(i: number): GPUBuffer {
    if (!this.sliceBufs[i]) {
      this.sliceBufs[i] = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    return this.sliceBufs[i];
  }
  private lastShadowTime = -1e9;
  /** 影の計算を何フレームかに分ける：残りの塊の数（0 なら計算中でない） */
  private shadowSlicesLeft = 0;
  private static readonly SHADOW_SLICES = 4;
  private sliceBufs: GPUBuffer[] = [];
  /**
   * 計算結果から描画用ボリュームと影を更新する。
   * ボリュームは計算が進んだときだけ。影（重い：各点から5方向に積分）は、計算時間で 20 秒ごとか、
   * 光源が 0.5° 以上動いたときだけ作り直す（雲の形の変化に対して影は少し遅れる）。
   */
  update(): void {
    const m = this.model;
    const g = this.grid;
    if (!m || !g) return;
    const s = this.sunNow, l = this.lastSun;
    const cosMove = s[0] * l[0] + s[1] * l[1] + s[2] * l[2];
    const doVol = m.steps !== this.lastSteps;
    if (this.shadowSlicesLeft === 0 && (m.time - this.lastShadowTime >= 20 || m.time < this.lastShadowTime || cosMove < Math.cos((0.5 * Math.PI) / 180))) {
      this.shadowSlicesLeft = CloudVolume.SHADOW_SLICES;
      this.lastSun = [...s];
      this.lastShadowTime = m.time;
    }
    const doShadow = this.shadowSlicesLeft > 0;
    if (!doVol && !doShadow) return;
    this.lastSteps = m.steps;
    // 雲のある高さの範囲（統計から。余裕を持たせる）
    const st = m.lastStats;
    if (st) {
      this.slabBot = Math.max(0, (st.cloudBase ?? 0) - 300);
      this.slabTop = Math.min(g.ztop, (st.cloudTop ?? 1000) + 500);
      // 雨は地表まで落ちるので、降水があれば下端は地表
      if (st.maxQr > 1e-6) this.slabBot = 0;
      // 雲も降水も無ければ雲の層は空（影の計算もすぐ終わり、0 になる）
      if (st.cloudTop === null && st.maxQr <= 0 && st.maxQc <= 0) { this.slabBot = 0; this.slabTop = -1; }
    }
    const d = this.device;
    const enc = d.createCommandEncoder();
    const pass = enc.beginComputePass({ label: 'cloud-volume' });
    if (doVol) {
    pass.setPipeline(this.volPipe);
    pass.setBindGroup(0, d.createBindGroup({
      layout: this.volPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.dims } },
        { binding: 1, resource: { buffer: m.stateBuffer } },
        { binding: 2, resource: { buffer: m.columnBuffer } },
        { binding: 3, resource: { buffer: m.diagBuffer } },
        { binding: 4, resource: this.vol.createView() },
        { binding: 5, resource: this.vel.createView() },
      ],
    }));
    pass.dispatchWorkgroups(Math.ceil(g.nx / 8), Math.ceil(g.ny / 8), g.nz);
    }
    if (doShadow) {
    // 高さ方向に SHADOW_SLICES 個に分けて、1フレームに1つずつ
    const slice = CloudVolume.SHADOW_SLICES - this.shadowSlicesLeft;
    const per = Math.ceil(g.nz / CloudVolume.SHADOW_SLICES);
    const z0 = Math.min(g.nz, slice * per), z1 = Math.min(g.nz, z0 + per);
    this.shadowSlicesLeft--;
    if (z1 > z0) {
    d.queue.writeBuffer(this.sliceBuf(slice), 0, new Uint32Array([z0, 0, 0, 0]));
    pass.setPipeline(this.shadowPipe);
    pass.setBindGroup(0, d.createBindGroup({
      layout: this.shadowPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.dims } },
        { binding: 1, resource: { buffer: this.uniform } },
        { binding: 2, resource: { buffer: m.columnBuffer } },
        { binding: 3, resource: this.vol.createView() },
        { binding: 4, resource: this.samplerRepeat },
        { binding: 5, resource: this.zmap.createView() },
        { binding: 6, resource: this.shadow.createView() },
        { binding: 7, resource: this.shadow2.createView() },
        { binding: 8, resource: { buffer: this.sliceBuf(slice) } },
      ],
    }));
    pass.dispatchWorkgroups(Math.ceil(g.nx / 16), Math.ceil(g.ny / 16), z1 - z0);
    }
    }
    pass.end();
    d.queue.submit([enc.finish()]);
  }
}
