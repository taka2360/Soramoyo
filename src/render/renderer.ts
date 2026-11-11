import type { Gpu } from '../core/gpu';
import type { Camera } from '../core/camera';
import type { AtmosphereLut } from '../atmosphere/atmosphere';
import { BOTTOM_RADIUS_KM } from '../atmosphere/atmosphere';
import commonWgsl from '../shaders/common.wgsl?raw';
import atmoFunctionsWgsl from '../shaders/atmosphere/functions.wgsl?raw';
import skyWgsl from '../shaders/sky.wgsl?raw';
import exposureWgsl from '../shaders/exposure.wgsl?raw';
import tonemapWgsl from '../shaders/tonemap.wgsl?raw';
import { GpuTimer } from './gpuTimer';

export const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const FRAME_FLOATS = 40;

export interface RenderSettings {
  /** 露出補正 [EV] */
  exposureEV: number;
  autoExposure: boolean;
  grid: boolean;
  /** 描画解像度の倍率（画質設定） */
  renderScale: number;
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
  /** 大気上端の太陽照度（線形 sRGB, lx）。地球–太陽距離の補正込み */
  sunIlluminance: number[];
  whitePoint: number[];
  groundAlbedo: number;
  refractionScale: number;
}

/**
 * 描画パイプライン：
 *   [空・地面] → HDR(rgba16float, cd/m²) → [自動露出] → [トーンマップ] → キャンバス
 * 段階4以降で、空と地面の間に雲レイマーチ（半解像度+TAA）が入る。
 */
export class Renderer {
  private frameBuf: GPUBuffer;
  private frameData = new Float32Array(FRAME_FLOATS);
  private expoBuf: GPUBuffer;
  private skyPipe: GPURenderPipeline;
  private expoPipe: GPUComputePipeline;
  private tonePipe: GPURenderPipeline;
  private skyBind?: GPUBindGroup;
  private skyBindGen = -1;
  private expoBind!: GPUBindGroup;
  private toneBind!: GPUBindGroup;
  private hdr!: GPUTexture;
  private width = 0;
  private height = 0;
  readonly timer: GpuTimer;

  constructor(private gpu: Gpu, private canvas: HTMLCanvasElement, private atmo: AtmosphereLut) {
    const { device } = gpu;
    this.frameBuf = device.createBuffer({ size: FRAME_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.expoBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

    const skyMod = device.createShaderModule({ label: 'sky', code: atmoFunctionsWgsl + commonWgsl + skyWgsl });
    this.skyPipe = device.createRenderPipeline({
      label: 'sky',
      layout: 'auto',
      vertex: { module: skyMod, entryPoint: 'vsFullscreen' },
      fragment: { module: skyMod, entryPoint: 'fsMain', targets: [{ format: HDR_FORMAT }] },
      primitive: { topology: 'triangle-list' },
    });

    const expoMod = device.createShaderModule({ label: 'exposure', code: commonWgsl + exposureWgsl });
    this.expoPipe = device.createComputePipeline({ label: 'exposure', layout: 'auto', compute: { module: expoMod, entryPoint: 'csMain' } });

    const toneMod = device.createShaderModule({ label: 'tonemap', code: commonWgsl + tonemapWgsl });
    this.tonePipe = device.createRenderPipeline({
      label: 'tonemap',
      layout: 'auto',
      vertex: { module: toneMod, entryPoint: 'vsFullscreen' },
      fragment: { module: toneMod, entryPoint: 'fsMain', targets: [{ format: gpu.canvasFormat }] },
      primitive: { topology: 'triangle-list' },
    });

    this.timer = new GpuTimer(device, gpu.hasTimestamps, ['sky', 'exposure', 'tonemap']);
  }

  private resize(scale: number): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2) * scale;
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (w === this.width && h === this.height) return;
    this.width = w;
    this.height = h;
    this.canvas.width = w;
    this.canvas.height = h;
    this.hdr?.destroy();
    this.hdr = this.gpu.device.createTexture({
      label: 'hdr',
      size: [w, h],
      format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    const { device } = this.gpu;
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
  }

  private ensureSkyBind(): boolean {
    const a = this.atmo;
    if (a.generation === 0) return false;
    if (this.skyBindGen !== a.generation) {
      this.skyBind = this.gpu.device.createBindGroup({
        layout: this.skyPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: a.uniform } },
          { binding: 1, resource: a.sampler },
          { binding: 2, resource: a.transmittance.createView() },
          { binding: 3, resource: a.irradiance.createView() },
          { binding: 4, resource: a.scattering.createView() },
          { binding: 5, resource: a.singleMie.createView() },
          { binding: 9, resource: { buffer: this.frameBuf } },
          { binding: 10, resource: { buffer: this.expoBuf } },
        ],
      });
      this.skyBindGen = a.generation;
    }
    return true;
  }

  get size(): [number, number] {
    return [this.width, this.height];
  }

  /** 自動露出の状態をリセット（時刻ジャンプ時など、即座に順応させたいとき） */
  resetAdaptation(): void {
    this.gpu.device.queue.writeBuffer(this.expoBuf, 0, new Float32Array([0, 0, 0, 0]));
  }

  render(cam: Camera, s: RenderSettings, sc: SceneState, dtReal: number, timeSec: number): void {
    this.resize(s.renderScale);
    if (!this.ensureSkyBind()) return;
    const { device } = this.gpu;
    const { fwd, right, up } = cam.basis();
    const camR = BOTTOM_RADIUS_KM + (sc.groundElevationM + cam.heightM) / 1000;
    const f = this.frameData;
    const wp = s.whiteBalance ? sc.whitePoint : [1, 1, 1];
    f.set([...fwd, Math.tan(cam.fovY * 0.5)], 0);
    f.set([...right, this.width / this.height], 4);
    f.set([...up, s.exposureEV], 8);
    f.set([0, 0, camR, s.autoExposure ? 1 : 0], 12);
    f.set([...sc.sunDir, sc.sunAngularRadius], 16);
    f.set([...sc.sunDirApparent, dtReal], 20);
    f.set([...sc.sunIlluminance, s.grid ? 1 : 0], 24);
    f.set([...wp, timeSec % 3600], 28);
    f.set([this.width, this.height, sc.groundAlbedo, sc.refractionScale], 32);
    device.queue.writeBuffer(this.frameBuf, 0, f);

    const enc = device.createCommandEncoder();
    {
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: this.hdr.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
        timestampWrites: this.timer.writes('sky'),
      });
      pass.setPipeline(this.skyPipe);
      pass.setBindGroup(0, this.skyBind!);
      pass.draw(3);
      pass.end();
    }
    {
      const pass = enc.beginComputePass({ timestampWrites: this.timer.computeWrites('exposure') });
      pass.setPipeline(this.expoPipe);
      pass.setBindGroup(0, this.expoBind);
      pass.dispatchWorkgroups(1);
      pass.end();
    }
    {
      const pass = enc.beginRenderPass({
        colorAttachments: [
          { view: this.gpu.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] },
        ],
        timestampWrites: this.timer.writes('tonemap'),
      });
      pass.setPipeline(this.tonePipe);
      pass.setBindGroup(0, this.toneBind);
      pass.draw(3);
      pass.end();
    }
    this.timer.resolve(enc);
    device.queue.submit([enc.finish()]);
    this.timer.readback();
  }
}
