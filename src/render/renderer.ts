import type { Gpu } from '../core/gpu';
import type { Camera } from '../core/camera';
import type { AtmosphereLut } from '../atmosphere/atmosphere';
import { BOTTOM_RADIUS_KM } from '../atmosphere/atmosphere';
import commonWgsl from '../shaders/common.wgsl?raw';
import atmoFunctionsWgsl from '../shaders/atmosphere/functions.wgsl?raw';
import atmoRenderWgsl from '../shaders/atmosphere/render.wgsl?raw';
import cloudCommonWgsl from '../shaders/clouds/cloudcommon.wgsl?raw';
import cloudsWgsl from '../shaders/clouds/clouds.wgsl?raw';
import resolveWgsl from '../shaders/clouds/resolve.wgsl?raw';
import skyWgsl from '../shaders/sky.wgsl?raw';
import exposureWgsl from '../shaders/exposure.wgsl?raw';
import tonemapWgsl from '../shaders/tonemap.wgsl?raw';
import starsWgsl from '../shaders/stars.wgsl?raw';
import type { StarData } from './nightsky';
import { GpuTimer } from './gpuTimer';
import type { CloudVolume } from './clouds';

export const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const FRAME_FLOATS = 72;

export interface RenderSettings {
  /** 露出補正 [EV] */
  exposureEV: number;
  autoExposure: boolean;
  grid: boolean;
  /** 描画解像度の倍率（画質設定） */
  renderScale: number;
  /** 雲の描画解像度（画面に対する比） */
  cloudScale: number;
  /** 太陽光を白とみなす色順応 */
  whiteBalance: boolean;
}

/** フレームごとに CPU で求める天体・環境の状態 */
export interface SceneState {
  /** 地表の標高 [m] */
  groundElevationM: number;
  sunDir: number[];
  sunDirApparent: number[];
  sunAngularRadius: number;
  /** 大気上端の太陽照度（線形 sRGB, lx） */
  sunIlluminance: number[];
  whitePoint: number[];
  groundAlbedo: number;
  refractionScale: number;
  moonDir: number[];
  moonDirApparent: number[];
  moonAngularRadius: number;
  moonIlluminance: number[];
  moonLuminanceScale: number;
  earthshine: number;
  j2000ToEnu: number[];
  /** 光害による天頂の輝度 [cd/m²]（0 = 光害なし） */
  lightPollution: number;
  /** 大気光（夜光）の天頂輝度 [cd/m²] */
  nightGlow: number;
  milkyWayScale: number;
  scotopic: boolean;
  /** 地面の種類（0 = 陸, 1 = 海） */
  surface: number;
  /** 海上の風速 [m/s] と風が吹いていく向き [rad, 東から反時計回り] */
  windSpeed: number;
  windAngle: number;
}

/**
 * 描画パイプライン：
 *   [雲（低解像度）] → [空・地面 ＋ 雲の合成] → HDR(rgba16float) → [自動露出] → [トーンマップ] → キャンバス
 */
export class Renderer {
  private frameBuf: GPUBuffer;
  private cloudFrameBuf: GPUBuffer;
  private frameData = new Float32Array(FRAME_FLOATS);
  private expoBuf: GPUBuffer;
  private skyPipe: GPURenderPipeline;
  private cloudPipe: GPURenderPipeline;
  private resolvePipe: GPURenderPipeline;
  private resolveBuf: GPUBuffer;
  private resolveBind: GPUBindGroup[] = [];
  private linearClamp: GPUSampler;
  private cloudRaw!: GPUTexture;
  private cloudDepth!: GPUTexture;
  private prevCam: number[] | null = null;
  private historyValid = false;
  /** 雲の時間方向の蓄積の混合率（小さいほどなめらかだが変化に遅れる） */
  taaAlpha = 0.1;
  private expoPipe: GPUComputePipeline;
  private tonePipe: GPURenderPipeline;
  private starPipe: GPURenderPipeline;
  private starBind: GPUBindGroup[] = [];
  private stars: StarData | null = null;
  private mwTex: GPUTexture;
  private nightGen = 0;
  private skyBind: GPUBindGroup[] = [];
  private cloudBind: GPUBindGroup[] = [];
  private parity = 0;
  private bindKey = '';
  private expoBind!: GPUBindGroup;
  private toneBind!: GPUBindGroup;
  private hdr!: GPUTexture;
  private cloudTargets: GPUTexture[] = [];
  private width = 0;
  private height = 0;
  private cw = 0;
  private ch = 0;
  readonly timer: GpuTimer;

  constructor(private gpu: Gpu, private canvas: HTMLCanvasElement, private atmo: AtmosphereLut, private clouds: CloudVolume) {
    const { device } = gpu;
    const U = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
    this.frameBuf = device.createBuffer({ size: FRAME_FLOATS * 4, usage: U });
    this.cloudFrameBuf = device.createBuffer({ size: FRAME_FLOATS * 4, usage: U });
    this.expoBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

    const fs = (label: string, code: string, format: GPUTextureFormat) => {
      const module = device.createShaderModule({ label, code });
      return device.createRenderPipeline({
        label,
        layout: 'auto',
        vertex: { module, entryPoint: 'vsFullscreen' },
        fragment: { module, entryPoint: 'fsMain', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
      });
    };
    const atmoBase = atmoFunctionsWgsl + atmoRenderWgsl + commonWgsl + cloudCommonWgsl;
    this.skyPipe = fs('sky', atmoBase + skyWgsl, HDR_FORMAT);
    {
      const module = device.createShaderModule({ label: 'clouds', code: atmoBase + cloudsWgsl });
      this.cloudPipe = device.createRenderPipeline({
        label: 'clouds', layout: 'auto',
        vertex: { module, entryPoint: 'vsFullscreen' },
        fragment: { module, entryPoint: 'fsMain', targets: [{ format: HDR_FORMAT }, { format: 'r32float' }] },
        primitive: { topology: 'triangle-list' },
      });
    }
    this.resolvePipe = fs('cloudResolve', commonWgsl + resolveWgsl, HDR_FORMAT);
    this.resolveBuf = device.createBuffer({ size: 112, usage: U });
    this.linearClamp = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.tonePipe = fs('tonemap', commonWgsl + tonemapWgsl, gpu.canvasFormat);
    const starMod = device.createShaderModule({ label: 'stars', code: atmoFunctionsWgsl + commonWgsl + starsWgsl });
    this.starPipe = device.createRenderPipeline({
      label: 'stars',
      layout: 'auto',
      vertex: { module: starMod, entryPoint: 'vsStar' },
      fragment: {
        module: starMod, entryPoint: 'fsStar',
        targets: [{ format: HDR_FORMAT, blend: { color: { srcFactor: 'one', dstFactor: 'one' }, alpha: { srcFactor: 'zero', dstFactor: 'one' } } }],
      },
      primitive: { topology: 'triangle-list' },
    });
    this.mwTex = device.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING });
    const expoMod = device.createShaderModule({ label: 'exposure', code: commonWgsl + exposureWgsl });
    this.expoPipe = device.createComputePipeline({ label: 'exposure', layout: 'auto', compute: { module: expoMod, entryPoint: 'csMain' } });

    this.timer = new GpuTimer(device, gpu.hasTimestamps, ['clouds', 'cloudTAA', 'sky', 'stars', 'exposure', 'tonemap']);
  }

  private resize(scale: number, cloudScale: number): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2) * scale;
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    const cw = Math.max(1, Math.floor(w * cloudScale));
    const ch = Math.max(1, Math.floor(h * cloudScale));
    if (w === this.width && h === this.height && cw === this.cw && ch === this.ch) return;
    this.width = w;
    this.height = h;
    this.cw = cw;
    this.ch = ch;
    this.canvas.width = w;
    this.canvas.height = h;
    const { device } = this.gpu;
    this.hdr?.destroy();
    this.cloudTargets.forEach((t) => t.destroy());
    this.cloudRaw?.destroy();
    this.cloudDepth?.destroy();
    this.historyValid = false;
    this.hdr = device.createTexture({
      label: 'hdr', size: [w, h], format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    // 雲：レイマーチの生の結果（色・透過率）と代表距離 → 時間方向の蓄積で cloudTargets[p] へ。
    // cloudTargets は2枚を交互に使う（前フレームの結果を履歴として読む）
    this.cloudRaw = device.createTexture({
      label: 'cloudRaw', size: [cw, ch], format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.cloudDepth = device.createTexture({
      label: 'cloudDepth', size: [cw, ch], format: 'r32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.cloudTargets = [0, 1].map((i) => device.createTexture({
      label: `clouds${i}`, size: [cw, ch], format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    }));
    this.expoBind = device.createBindGroup({
      layout: this.expoPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.frameBuf } },
        { binding: 1, resource: this.hdr.createView() },
        { binding: 2, resource: { buffer: this.expoBuf } },
      ],
    });
    this.toneBind = device.createBindGroup({
      layout: this.tonePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.frameBuf } },
        { binding: 1, resource: this.hdr.createView() },
        { binding: 2, resource: { buffer: this.expoBuf } },
      ],
    });
    this.bindKey = '';
  }

  /** 夜空のデータ（天の川テクスチャ・恒星）を設定 */
  setNightSky(mw: GPUTexture, stars: StarData): void {
    this.mwTex = mw;
    this.stars = stars;
    this.nightGen++;
    this.bindKey = '';
  }

  private ensureBinds(): boolean {
    const a = this.atmo;
    const c = this.clouds;
    if (a.generation === 0) return false;
    const key = `${a.generation}/${c.generation}/${this.width}x${this.height}/${this.cw}/${this.nightGen}`;
    if (key === this.bindKey) return true;
    const atmoEntries = (frame: GPUBuffer): GPUBindGroupEntry[] => [
      { binding: 0, resource: { buffer: a.uniform } },
      { binding: 1, resource: a.sampler },
      { binding: 2, resource: a.transmittance.createView() },
      { binding: 3, resource: a.irradiance.createView() },
      { binding: 4, resource: a.scattering.createView() },
      { binding: 5, resource: a.singleMie.createView() },
      { binding: 9, resource: { buffer: frame } },
      { binding: 10, resource: { buffer: this.expoBuf } },
    ];
    const d = this.gpu.device;
    // parity p：雲を cloudTargets[p] に描き、履歴として cloudTargets[1-p] を読む
    this.skyBind = [0, 1].map((p) => d.createBindGroup({
      layout: this.skyPipe.getBindGroupLayout(0),
      entries: [
        ...atmoEntries(this.frameBuf),
        { binding: 11, resource: this.cloudTargets[p].createView() },
        { binding: 12, resource: c.shadow.createView() },
        { binding: 20, resource: c.shadow2.createView() },
        { binding: 13, resource: c.zmap.createView() },
        { binding: 15, resource: { buffer: c.uniform } },
        { binding: 16, resource: c.samplerRepeat },
        { binding: 17, resource: this.mwTex.createView() },
      ],
    }));
    if (this.stars) {
      const st = this.stars;
      this.starBind = [0, 1].map((p) => d.createBindGroup({
        layout: this.starPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: a.uniform } },
          { binding: 1, resource: a.sampler },
          { binding: 2, resource: a.transmittance.createView() },
          { binding: 9, resource: { buffer: this.frameBuf } },
          { binding: 10, resource: { buffer: this.expoBuf } },
          { binding: 11, resource: this.cloudTargets[p].createView() },
          { binding: 20, resource: { buffer: st.buffer } },
        ],
      }));
    }
    this.cloudBind = [0, 1].map(() => d.createBindGroup({
      layout: this.cloudPipe.getBindGroupLayout(0),
      entries: [
        ...atmoEntries(this.cloudFrameBuf),
        { binding: 11, resource: c.vol.createView() },
        { binding: 12, resource: c.shadow.createView() },
        { binding: 20, resource: c.shadow2.createView() },
        { binding: 13, resource: c.zmap.createView() },
        { binding: 14, resource: c.noise.createView() },
        { binding: 18, resource: c.vel.createView() },
        { binding: 19, resource: c.samplerNoise },
        { binding: 15, resource: { buffer: c.uniform } },
        { binding: 16, resource: c.samplerRepeat },
      ],
    }));
    this.resolveBind = [0, 1].map((p) => d.createBindGroup({
      layout: this.resolvePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.resolveBuf } },
        { binding: 1, resource: this.cloudRaw.createView() },
        { binding: 2, resource: this.cloudDepth.createView() },
        { binding: 3, resource: this.cloudTargets[1 - p].createView() },
        { binding: 4, resource: this.linearClamp },
      ],
    }));
    this.bindKey = key;
    return true;
  }

  get size(): [number, number] {
    return [this.width, this.height];
  }

  /** 雲の時間方向の蓄積を捨てる（時刻ジャンプなどで雲が入れ替わったとき） */
  resetHistory(): void {
    this.historyValid = false;
  }

  /** 自動露出の状態をリセット（時刻ジャンプ時など、即座に順応させたいとき） */
  resetAdaptation(): void {
    // 0 にすると最初のフレームのプリエクスポージャーが極端（×10⁴）になり、雲が半精度の上限を超えるため、
    // 昼間程度の値から順応し直す（[0] 順応輝度、[1] は露出パスが毎フレーム書く）
    this.gpu.device.queue.writeBuffer(this.expoBuf, 0, new Float32Array([1000, 0.001, 1, 0]));
  }

  private packFrame(cam: Camera, s: RenderSettings, sc: SceneState, dtReal: number, timeSec: number, w: number, h: number): Float32Array<ArrayBuffer> {
    const { fwd, right, up } = cam.basis();
    const camR = BOTTOM_RADIUS_KM + (sc.groundElevationM + cam.heightM) / 1000;
    const f = new Float32Array(FRAME_FLOATS);
    const wp = s.whiteBalance ? sc.whitePoint : [1, 1, 1];
    f.set([...fwd, Math.tan(cam.fovY * 0.5)], 0);
    f.set([...right, w / h], 4);
    f.set([...up, s.exposureEV], 8);
    f.set([0, 0, camR, s.autoExposure ? 1 : 0], 12);
    f.set([...sc.sunDir, sc.sunAngularRadius], 16);
    f.set([...sc.sunDirApparent, dtReal], 20);
    f.set([...sc.sunIlluminance, s.grid ? 1 : 0], 24);
    f.set([...wp, timeSec % 3600], 28);
    f.set([w, h, sc.groundAlbedo, sc.refractionScale], 32);
    f.set([...sc.moonDir, sc.moonAngularRadius], 36);
    f.set([...sc.moonDirApparent, sc.moonLuminanceScale], 40);
    f.set([...sc.moonIlluminance, sc.earthshine], 44);
    // 雲を照らす主光源：太陽が地平線下 6° より低ければ月
    const Y = (c: number[]) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    const useMoon = sc.sunDir[2] < -0.1;
    const lDir = useMoon ? sc.moonDir : sc.sunDir;
    const lIll = useMoon ? sc.moonIlluminance : sc.sunIlluminance;
    f.set([...lDir, Y(lIll) / Y(sc.sunIlluminance)], 48);
    f.set([...lIll, sc.lightPollution], 52);
    const m = sc.j2000ToEnu;
    f.set([m[0], m[1], m[2], sc.nightGlow], 56);
    f.set([m[3], m[4], m[5], sc.scotopic ? 1 : 0], 60);
    f.set([m[6], m[7], m[8], sc.milkyWayScale], 64);
    f.set([sc.surface, sc.windSpeed, sc.windAngle, cam.heightM], 68);
    return f;
  }

  render(cam: Camera, s: RenderSettings, sc: SceneState, dtReal: number, timeSec: number): void {
    this.resize(s.renderScale, s.cloudScale);
    if (!this.ensureBinds()) return;
    const { device } = this.gpu;
    this.frameData = this.packFrame(cam, s, sc, dtReal, timeSec, this.width, this.height);
    device.queue.writeBuffer(this.frameBuf, 0, this.frameData);
    device.queue.writeBuffer(this.cloudFrameBuf, 0, this.packFrame(cam, s, sc, dtReal, timeSec, this.cw, this.ch));

    const enc = device.createCommandEncoder();
    const fullscreen = (view: GPUTextureView, pipe: GPURenderPipeline, bg: GPUBindGroup, timerName: string, clear: number[]) => {
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view, loadOp: 'clear', storeOp: 'store', clearValue: clear }],
        timestampWrites: this.timer.writes(timerName),
      });
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    };
    this.parity = 1 - this.parity;
    const p = this.parity;
    {
      const pass = enc.beginRenderPass({
        colorAttachments: [
          { view: this.cloudRaw.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
          { view: this.cloudDepth.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
        ],
        timestampWrites: this.timer.writes('clouds'),
      });
      pass.setPipeline(this.cloudPipe);
      pass.setBindGroup(0, this.cloudBind[p]);
      pass.draw(3);
      pass.end();
    }
    // 時間方向の蓄積（前フレームのカメラへ再投影）
    {
      const { fwd, right, up } = cam.basis();
      const camH = sc.groundElevationM + cam.heightM;
      const curCam = [...fwd, Math.tan(cam.fovY * 0.5), ...right, this.cw / this.ch, ...up, camH];
      const prev = this.prevCam ?? curCam;
      device.queue.writeBuffer(this.resolveBuf, 0, new Float32Array([...curCam, ...prev, this.taaAlpha, this.historyValid && this.prevCam ? 1 : 0, 0, 0]));
      this.prevCam = curCam;
      this.historyValid = true;
    }
    fullscreen(this.cloudTargets[p].createView(), this.resolvePipe, this.resolveBind[p], 'cloudTAA', [0, 0, 0, 1]);
    fullscreen(this.hdr.createView(), this.skyPipe, this.skyBind[p], 'sky', [0, 0, 0, 1]);
    if (this.stars && this.starBind.length) {
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: this.hdr.createView(), loadOp: 'load', storeOp: 'store' }],
        timestampWrites: this.timer.writes('stars'),
      });
      pass.setPipeline(this.starPipe);
      pass.setBindGroup(0, this.starBind[p]);
      pass.draw(6, this.stars.count);
      pass.end();
    }
    {
      const pass = enc.beginComputePass({ timestampWrites: this.timer.computeWrites('exposure') });
      pass.setPipeline(this.expoPipe);
      pass.setBindGroup(0, this.expoBind);
      pass.dispatchWorkgroups(1);
      pass.end();
    }
    fullscreen(this.gpu.context.getCurrentTexture().createView(), this.tonePipe, this.toneBind, 'tonemap', [0, 0, 0, 1]);
    this.timer.resolve(enc);
    device.queue.submit([enc.finish()]);
    this.timer.readback();
  }
}
