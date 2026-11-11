import functionsWgsl from '../shaders/atmosphere/functions.wgsl?raw';
import precomputeWgsl from '../shaders/atmosphere/precompute.wgsl?raw';
import {
  type AtmosphereSettings,
  coefficientsAt,
  RGB_LAMBDAS,
  srgbCoefficients,
  wavelengthGroups,
} from './spectrum';

export const BOTTOM_RADIUS_KM = 6360;
export const TOP_RADIUS_KM = 6420;
export const SUN_ANGULAR_RADIUS = 0.004675;
const MU_S_MIN = Math.cos((120 * Math.PI) / 180); // 太陽天頂角 120°（高度 -30°）まで計算（深い薄明用）

const T_W = 256, T_H = 64;
const S_W = 256, S_H = 128, S_D = 32;
const I_W = 64, I_H = 16;

export interface PrecomputeOptions {
  /** 事前計算に使う3波長グループ数（波長数 = 3×groups） */
  wavelengthGroups: number;
  /** 散乱の次数（1=単散乱のみ） */
  scatteringOrders: number;
}

/**
 * Bruneton 2017 方式の大気散乱 LUT を GPU で事前計算し、描画用テクスチャを保持する。
 * 設定（エアロゾル量など）を変えたら precompute() を呼び直す。
 */
export class AtmosphereLut {
  readonly format: GPUTextureFormat;
  readonly uniform: GPUBuffer; // 描画用 Atmo（半径・g など）
  readonly sampler: GPUSampler;
  transmittance!: GPUTexture;
  irradiance!: GPUTexture;
  scattering!: GPUTexture;
  singleMie!: GPUTexture;
  /** 最後の事前計算にかかった時間 [ms] */
  lastPrecomputeMs = 0;
  /** 世代番号（再計算のたびに増える。バインドグループの作り直し判定用） */
  generation = 0;

  private pipes = new Map<string, GPUComputePipeline>();
  private computeUniform: GPUBuffer;

  constructor(private device: GPUDevice, float32Filterable: boolean) {
    this.format = float32Filterable ? 'rgba32float' : 'rgba16float';
    const code = functionsWgsl + precomputeWgsl.replaceAll('__FMT__', this.format);
    const module = device.createShaderModule({ label: 'atmo-precompute', code });
    for (const ep of [
      'csTransmittance',
      'csDirectIrradiance',
      'csSingleScattering',
      'csScatteringDensity',
      'csIndirectIrradiance',
      'csMultipleScattering',
      'csFinalizeScattering',
      'csFinalizeIrradiance',
    ]) {
      this.pipes.set(ep, device.createComputePipeline({ label: ep, layout: 'auto', compute: { module, entryPoint: ep } }));
    }
    this.uniform = device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.computeUniform = device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
    });
  }

  private tex2d(label: string, w: number, h: number): GPUTexture {
    return this.device.createTexture({
      label, size: [w, h], format: this.format,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
  }
  private tex3d(label: string): GPUTexture {
    return this.device.createTexture({
      label, size: [S_W, S_H, S_D], dimension: '3d', format: this.format,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
  }

  /** Atmo 構造体（144 バイト）を詰める */
  private packAtmo(
    s: AtmosphereSettings,
    lambdas: number[],
    lum: number[][] | null,
    order: number,
  ): Float32Array<ArrayBuffer> {
    const c = lambdas.map((l) => coefficientsAt(l, s));
    const f = new Float32Array(36);
    f.set([c[0].solar, c[1].solar, c[2].solar, SUN_ANGULAR_RADIUS], 0);
    f.set([c[0].rayleigh, c[1].rayleigh, c[2].rayleigh, BOTTOM_RADIUS_KM], 4);
    f.set([c[0].mieScattering, c[1].mieScattering, c[2].mieScattering, TOP_RADIUS_KM], 8);
    f.set([c[0].mieExtinction, c[1].mieExtinction, c[2].mieExtinction, s.mieG], 12);
    f.set([c[0].ozone, c[1].ozone, c[2].ozone, MU_S_MIN], 16);
    f.set([s.groundAlbedo, s.mieScaleHeight, order, 0], 20);
    const m = lum ?? [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    f.set([...m[0], 0], 24);
    f.set([...m[1], 0], 28);
    f.set([...m[2], 0], 32);
    return f;
  }

  private bind(ep: string, entries: Record<number, GPUBindingResource>): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.pipes.get(ep)!.getBindGroupLayout(0),
      entries: Object.entries(entries).map(([b, resource]) => ({ binding: +b, resource })),
    });
  }

  private run(ep: string, bg: GPUBindGroup, x: number, y: number, z = 1): void {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass({ label: ep });
    pass.setPipeline(this.pipes.get(ep)!);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(x, y, z);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  async precompute(s: AtmosphereSettings, opt: PrecomputeOptions, onProgress?: (f: number) => void): Promise<void> {
    const t0 = performance.now();
    const d = this.device;
    const U = this.computeUniform;
    const smp = this.sampler;

    // 作業用テクスチャ
    const tGroup = this.tex2d('T(group)', T_W, T_H);
    const dIrr = this.tex2d('ΔE', I_W, I_H);
    const dRay = this.tex3d('ΔS_R');
    const dMie = this.tex3d('ΔS_M');
    const dMul = this.tex3d('ΔS');
    const dDen = this.tex3d('ΔJ');
    const accSize = S_W * S_H * S_D * 16;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const scatAcc = d.createBuffer({ size: accSize, usage });
    const mieAcc = d.createBuffer({ size: accSize, usage });
    const irrAcc = d.createBuffer({ size: I_W * I_H * 16, usage });

    // 出力テクスチャ（描画中のものとは別に作り、完成後に差し替える）
    const transmittance = this.tex2d('transmittance', T_W, T_H);
    const irradiance = this.tex2d('irradiance', I_W, I_H);
    const scattering = this.tex3d('scattering');
    const singleMie = this.tex3d('singleMie');

    const v = (t: GPUTexture) => t.createView();
    const bTrans = (out: GPUTexture) => this.bind('csTransmittance', { 0: { buffer: U }, 8: v(out) });
    const bgTransG = bTrans(tGroup);
    const bgDirect = this.bind('csDirectIrradiance', { 0: { buffer: U }, 1: smp, 2: v(tGroup), 9: v(dIrr) });
    const bgSingle = this.bind('csSingleScattering', {
      0: { buffer: U }, 1: smp, 2: v(tGroup), 10: v(dRay), 11: v(dMie), 12: { buffer: scatAcc }, 13: { buffer: mieAcc },
    });
    const bgDensity = this.bind('csScatteringDensity', {
      0: { buffer: U }, 1: smp, 2: v(tGroup), 3: v(dIrr), 4: v(dRay), 5: v(dMie), 6: v(dMul), 14: v(dDen),
    });
    const bgIndirect = this.bind('csIndirectIrradiance', {
      0: { buffer: U }, 1: smp, 4: v(dRay), 5: v(dMie), 6: v(dMul), 9: v(dIrr), 15: { buffer: irrAcc },
    });
    const bgMultiple = this.bind('csMultipleScattering', {
      0: { buffer: U }, 1: smp, 2: v(tGroup), 7: v(dDen), 16: v(dMul), 12: { buffer: scatAcc },
    });

    const { lambdas, dLambda } = wavelengthGroups(opt.wavelengthGroups);
    const steps = opt.wavelengthGroups * (3 + 3 * (opt.scatteringOrders - 1)) + 2;
    let done = 0;
    const tick = async () => {
      done++;
      onProgress?.(done / steps);
      // 長時間の GPU 占有（Windows の TDR によるデバイス喪失）を避けるため、こまめに完了を待つ
      await d.queue.onSubmittedWorkDone();
    };

    const g8 = (n: number) => Math.ceil(n / 8);
    const g4 = (n: number) => Math.ceil(n / 4);

    for (const ls of lambdas) {
      // 3波長ぶんの放射輝度 → sRGB 輝度 の変換行列（行 = R,G,B / 列 = 波長）
      const cs = ls.map((l) => srgbCoefficients(l));
      const lum = [0, 1, 2].map((row) => cs.map((c) => c[row] * dLambda));
      const write = (order: number) => d.queue.writeBuffer(U, 0, this.packAtmo(s, ls, lum, order));

      write(0);
      this.run('csTransmittance', bgTransG, g8(T_W), g8(T_H));
      await tick();
      this.run('csDirectIrradiance', bgDirect, g8(I_W), g8(I_H));
      await tick();
      this.run('csSingleScattering', bgSingle, g4(S_W), g4(S_H), g4(S_D));
      await tick();
      for (let order = 2; order <= opt.scatteringOrders; order++) {
        write(order);
        this.run('csScatteringDensity', bgDensity, g4(S_W), g4(S_H), g4(S_D));
        await tick();
        write(order - 1);
        this.run('csIndirectIrradiance', bgIndirect, g8(I_W), g8(I_H));
        await tick();
        this.run('csMultipleScattering', bgMultiple, g4(S_W), g4(S_H), g4(S_D));
        await tick();
      }
    }

    // 描画用：代表3波長（680/550/440nm）での透過率
    d.queue.writeBuffer(U, 0, this.packAtmo(s, RGB_LAMBDAS, null, 0));
    this.run('csTransmittance', bTrans(transmittance), g8(T_W), g8(T_H));
    await tick();
    this.run(
      'csFinalizeScattering',
      this.bind('csFinalizeScattering', { 17: v(scattering), 18: v(singleMie), 20: { buffer: scatAcc }, 21: { buffer: mieAcc } }),
      g4(S_W), g4(S_H), g4(S_D),
    );
    this.run(
      'csFinalizeIrradiance',
      this.bind('csFinalizeIrradiance', { 19: v(irradiance), 22: { buffer: irrAcc } }),
      g8(I_W), g8(I_H),
    );
    await tick();

    // 差し替え
    const old = [this.transmittance, this.irradiance, this.scattering, this.singleMie];
    this.transmittance = transmittance;
    this.irradiance = irradiance;
    this.scattering = scattering;
    this.singleMie = singleMie;
    d.queue.writeBuffer(this.uniform, 0, this.packAtmo(s, RGB_LAMBDAS, null, 0));
    this.generation++;
    await d.queue.onSubmittedWorkDone();
    old.forEach((t) => t?.destroy());
    [tGroup, dIrr, dRay, dMie, dMul, dDen].forEach((t) => t.destroy());
    [scatAcc, mieAcc, irrAcc].forEach((b) => b.destroy());
    this.lastPrecomputeMs = performance.now() - t0;
  }
}
