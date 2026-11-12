/**
 * 雲モデルの検証ページ：標準テストケースを走らせ、断面図と統計を表示する。
 */
import GUI from 'lil-gui';
import { initGpu, showFatal } from '../core/gpu';
import { CloudModel, F } from '../sim/model';
import { type SimCase, straka, bryanFritsch, weismanKlemp, armCumulus, bomex, dycoms } from '../sim/cases';
import vizWgsl from '../shaders/sim/viz.wgsl?raw';

const CASES: Record<string, () => SimCase> = {
  'Straka 密度流 (2D, dx=100m)': () => straka(100),
  'Straka 密度流 (2D, dx=50m)': () => straka(50),
  'Bryan–Fritsch 湿潤気泡 (2D, dx=100m)': () => bryanFritsch(100),
  'Weisman–Klemp 積乱雲 (3D, 128², dx=500m)': () => weismanKlemp({ nx: 128, dx: 500, dzMin: 250, shear: 10, qv0: 0.014 }),
  'Weisman–Klemp 積乱雲 (3D, 256², dx=250m)': () => weismanKlemp({ nx: 256, dx: 250, dzMin: 150, shear: 10, qv0: 0.014 }),
  'DYCOMS-II 層積雲 (3D, 64², dx=50m, dz=10m)': () => dycoms({ nx: 64, dx: 50, dzFine: 10 }),
  'DYCOMS-II 層積雲 (3D, 128², dx=50m, dz=5m)': () => dycoms({ nx: 128, dx: 50, dzFine: 5 }),
  'BOMEX 貿易風積雲 (3D, 64², dx=100m)': () => bomex({ nx: 64, dx: 100, dz: 40 }),
  'BOMEX 貿易風積雲 (3D, 128², dx=100m)': () => bomex({ nx: 128, dx: 100, dz: 40 }),
  '晴天積雲 ARM SGP (3D, 128², dx=100m)': () => armCumulus({ nx: 128, dx: 100, dzMin: 40, shf: 120, lhf: 400 }),
  '晴天積雲 ARM SGP (3D, 256², dx=100m)': () => armCumulus({ nx: 256, dx: 100, dzMin: 40, shf: 120, lhf: 400 }),
};

interface FieldDef { name: string; f: number; cmap: 0 | 1; range: [number, number] }
const FIELDS: Record<string, FieldDef> = {
  "θ' [K]": { name: "θ'", f: F.TH, cmap: 0, range: [-4, 4] },
  'w [m/s]': { name: 'w', f: F.W, cmap: 0, range: [-15, 15] },
  'u [m/s]': { name: 'u', f: F.U, cmap: 0, range: [-20, 20] },
  "qv' [kg/kg]": { name: "qv'", f: F.QV, cmap: 0, range: [-0.004, 0.004] },
  'qc 雲水 [kg/kg]': { name: 'qc', f: F.QC, cmap: 1, range: [0, 0.004] },
  'qr 雨水 [kg/kg]': { name: 'qr', f: F.QR, cmap: 1, range: [0, 0.006] },
  'qg あられ [kg/kg]': { name: 'qg', f: F.QG, cmap: 1, range: [0, 0.006] },
};

async function main() {
  const canvas = document.getElementById('view') as HTMLCanvasElement;
  const hud = document.getElementById('hud')!;
  const gpu = await initGpu(canvas);
  const { device } = gpu;

  const vizMod = device.createShaderModule({ label: 'viz', code: vizWgsl });
  const vizPipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: vizMod, entryPoint: 'vsMain' },
    fragment: { module: vizMod, entryPoint: 'fsMain', targets: [{ format: gpu.canvasFormat }] },
  });
  const vizBuf = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const ui = {
    caseName: Object.keys(CASES)[0],
    running: true,
    stopAtEnd: true,
    stepsPerFrame: 4,
    field: "θ' [K]",
    mode: 0,
    slice: 0,
    autoRange: false,
    vmin: -4,
    vmax: 4,
    contour: 1e-5,
    vExag: 1,
    reset: () => load(),
  };

  let model: CloudModel | null = null;
  let simCase: SimCase | null = null;
  let vizBind: GPUBindGroup | null = null;
  let msPerStep = 0;
  let frontX: number | null = null;
  let rowBusy = false;
  const rowRead = { buf: null as GPUBuffer | null };

  const load = () => {
    model?.destroy();
    simCase = CASES[ui.caseName]();
    model = new CloudModel(device, simCase.config, simCase.initial);
    const g = simCase.config.grid;
    ui.slice = ui.mode === 0 ? Math.floor(g.ny / 2) : Math.min(g.nz - 1, 4);
    sliceCtl.max(ui.mode === 0 ? g.ny - 1 : g.nz - 1).updateDisplay();
    ui.vExag = g.ny === 1 && g.nx * g.dx / g.ztop > 5 ? 2 : 1;
    vexCtl.updateDisplay();
    rowRead.buf?.destroy();
    rowRead.buf = device.createBuffer({ size: g.nx * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    frontX = null;
    vizBind = null;
  };

  const gui = new GUI({ title: '雲モデル検証' });
  gui.add(ui, 'caseName', Object.keys(CASES)).name('ケース').onChange(load);
  gui.add(ui, 'running').name('実行');
  gui.add(ui, 'stepsPerFrame', 1, 40, 1).name('1フレームのステップ数');
  gui.add(ui, 'stopAtEnd').name('標準時間で停止');
  gui.add(ui, 'reset').name('リセット');
  gui.add(ui, 'field', Object.keys(FIELDS)).name('表示する量').onChange((k: string) => {
    [ui.vmin, ui.vmax] = FIELDS[k].range;
    gui.controllersRecursive().forEach((c) => c.updateDisplay());
  });
  gui.add(ui, 'mode', { 'x–z 鉛直断面': 0, 'x–y 水平断面': 1 }).name('断面').onChange(() => {
    const g = simCase!.config.grid;
    ui.slice = ui.mode === 0 ? Math.floor(g.ny / 2) : Math.min(g.nz - 1, 4);
    sliceCtl.max(ui.mode === 0 ? g.ny - 1 : g.nz - 1).updateDisplay();
  });
  const sliceCtl = gui.add(ui, 'slice', 0, 255, 1).name('断面の位置（j or k）');
  gui.add(ui, 'vmin').name('色の下限');
  gui.add(ui, 'vmax').name('色の上限');
  gui.add(ui, 'contour', { 'なし': 0, 'qc=0.01 g/kg': 1e-5, 'qc=0.1 g/kg': 1e-4 }).name('雲の輪郭');
  const vexCtl = gui.add(ui, 'vExag', 1, 4, 0.5).name('縦の誇張');
  load();

  const fmt = (v: number | null | undefined, d = 1) => (v === null || v === undefined ? '—' : v.toFixed(d));

  const frame = async () => {
    if (!model || !simCase) return requestAnimationFrame(frame);
    const model0 = model;
    const g = simCase.config.grid;
    const limit = ui.stopAtEnd ? simCase.duration : Infinity;
    if (ui.running && model.time < limit - 1e-6) {
      const t0 = performance.now();
      // 終了時刻をまたがないよう、残り時間を目安にステップ数を減らす
      const remain = Math.ceil((limit - model.time) / Math.max(model.dt, 1e-3));
      model.step(Math.max(1, Math.min(ui.stepsPerFrame, remain)));
      await device.queue.onSubmittedWorkDone();
      const ms = (performance.now() - t0) / ui.stepsPerFrame;
      msPerStep = msPerStep ? msPerStep * 0.9 + ms * 0.1 : ms;
      // 待っている間にケースが切り替わったら、この回の描画はやめる
      if (model !== model0) return requestAnimationFrame(frame);

      // 2D の場合：地表の θ' から前線位置（θ' < −1K の最も右）を読む
      if (g.ny === 1 && !simCase.config.moist && !rowBusy && rowRead.buf) {
        rowBusy = true;
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(model.stateBuffer, F.TH * model.n * 4, rowRead.buf, 0, g.nx * 4);
        device.queue.submit([enc.finish()]);
        const buf = rowRead.buf;
        const th0 = simCase.config.base.th0[0];
        buf.mapAsync(GPUMapMode.READ).then(() => {
          const a = new Float32Array(buf.getMappedRange().slice(0));
          buf.unmap();
          let x: number | null = null;
          for (let i = g.nx - 1; i >= g.nx / 2; i--) {
            if (a[i] - th0 < -1) { x = (i + 0.5) * g.dx - g.nx * g.dx / 2; break; }
          }
          frontX = x;
          rowBusy = false;
        }).catch(() => (rowBusy = false));
      }
    }

    // 描画領域（縦横比を保つ。2D の細長い領域は縦を誇張できる）
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.floor(canvas.clientWidth * dpr), H = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    const lx = g.nx * g.dx, ly = g.ny * g.dy;
    const physW = lx, physH = ui.mode === 0 ? g.ztop * ui.vExag : ly;
    const margin = 40 * dpr;
    const s = Math.min((W - 2 * margin) / physW, (H - 2 * margin - 120 * dpr) / physH);
    const rw = physW * s, rh = physH * s;
    const x0 = (W - rw) / 2, y0 = (H - rh) / 2 + 40 * dpr;

    const fd = FIELDS[ui.field];
    let vmin = ui.vmin, vmax = ui.vmax;
    if (ui.autoRange && model.lastStats) {
      void 0;
    }
    const buf = new ArrayBuffer(80);
    const u32 = new Uint32Array(buf), f32 = new Float32Array(buf);
    u32[0] = g.nx; u32[1] = g.ny; u32[2] = g.nz; u32[3] = model.n;
    u32[4] = ui.mode; u32[5] = Math.min(ui.slice, ui.mode === 0 ? g.ny - 1 : g.nz - 1); u32[6] = fd.f; u32[7] = fd.cmap;
    f32[8] = vmin; f32[9] = vmax; f32[10] = lx; f32[11] = ly;
    f32[12] = g.ztop; f32[13] = ui.contour;
    f32.set([x0, y0, x0 + rw, y0 + rh], 16);
    device.queue.writeBuffer(vizBuf, 0, buf);
    // バッファが回るので毎フレーム作り直す（安価）
    vizBind = device.createBindGroup({
      layout: vizPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: vizBuf } },
        { binding: 1, resource: { buffer: model.stateBuffer } },
        { binding: 2, resource: { buffer: model.columnBuffer } },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: gpu.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0.08, 0.08, 0.09, 1] }],
    });
    pass.setPipeline(vizPipe);
    pass.setBindGroup(0, vizBind);
    pass.draw(3);
    pass.end();
    device.queue.submit([enc.finish()]);

    const st = model.lastStats;
    const cells = g.nx * g.ny * g.nz;
    const realtime = msPerStep > 0 ? (model.dt / (msPerStep / 1000)) : 0;
    hud.textContent =
      `${simCase.name}\n${simCase.description}\n` +
      `格子 ${g.nx}×${g.ny}×${g.nz} = ${(cells / 1e6).toFixed(2)}M セル  dx=${g.dx.toFixed(0)}m  dz=${Math.min(...g.dzc).toFixed(0)}〜${Math.max(...g.dzc).toFixed(0)}m  上端 ${(g.ztop / 1000).toFixed(1)}km\n` +
      `時刻 ${model.time.toFixed(1)} s / ${simCase.duration} s   ステップ ${model.steps}   dt ${model.dt.toFixed(2)} s\n` +
      `計算 ${msPerStep.toFixed(2)} ms/step  → 実時間の ×${realtime.toFixed(0)}（描画なし換算）\n` +
      (st
        ? `w 最大 ${fmt(st.maxW)} / 最小 ${fmt(st.minW)} m/s   |u| 最大 ${fmt(st.maxU)} m/s\n` +
          `θ' 最小 ${fmt(st.minThp, 2)} / 最大 ${fmt(st.maxThp, 2)} K\n` +
          (simCase.config.moist
            ? `qc 最大 ${fmt(st.maxQc * 1000, 2)} g/kg  qr 最大 ${fmt(st.maxQr * 1000, 2)} g/kg   雲底 ${fmt(st.cloudBase === null ? null : st.cloudBase / 1000, 2)} km  雲頂 ${fmt(st.cloudTop === null ? null : st.cloudTop / 1000, 2)} km  最大雲量 ${(st.cloudCover * 100).toFixed(1)}%\n` +
              `水の収支：全水量の変化 ${(model.waterDrift * 100).toExponential(2)} %${simCase.config.forcing ? '（大規模強制で水が出入りするため保存量ではない）' : ''}   地表降水 ${(st.precip / 1e6).toFixed(3)} kt   負値切り詰め ${(st.clipped / 1e6).toExponential(2)} kt\n`
            : '') +
          `液水温位の平均 ${st.meanThl.toFixed(5)} K   LWP ${st.lwp.toFixed(1)} g/m²   最大の層雲量 ${(st.cloudCover * 100).toFixed(1)}%\n` +
          `地表：Ts ${(st.surface.ts - 273.15).toFixed(1)}℃  顕熱 ${st.surface.h.toFixed(0)} W/m²  潜熱 ${st.surface.le.toFixed(0)} W/m²  日射 ${st.surface.sw.toFixed(0)} W/m²\n` +
          (simCase.reference ? `${simCase.reference}\n` : '')
        : '') +
      (frontX !== null ? `前線位置 ${(frontX / 1000).toFixed(2)} km（標準解 900秒で ≈15.5km）\n` : '');

    const lt = document.getElementById('legendText')!;
    lt.textContent = `${fd.name}: ${vmin} 〜 ${vmax}`;
    requestAnimationFrame(frame);
  };

  // 凡例
  const bar = document.getElementById('bar') as HTMLCanvasElement;
  const bctx = bar.getContext('2d')!;
  const drawBar = () => {
    const fd = FIELDS[ui.field];
    for (let x = 0; x < 260; x++) {
      const t = x / 259;
      let c: number[];
      if (fd.cmap === 0) {
        const a = t < 0.5 ? [0.12, 0.25, 0.65] : [0.97, 0.97, 0.97];
        const b = t < 0.5 ? [0.97, 0.97, 0.97] : [0.7, 0.12, 0.12];
        const u = t < 0.5 ? t * 2 : t * 2 - 1;
        const sm = u * u * (3 - 2 * u);
        c = a.map((v, i) => v + (b[i] - v) * sm);
      } else {
        c = [t, t, t].map((v, i) => [0.05 + 0.95 * v, 0.07 + 0.93 * v, 0.18 + 0.82 * v][i]);
      }
      bctx.fillStyle = `rgb(${c.map((v) => Math.round(Math.pow(v, 1 / 2.2) * 255)).join(',')})`;
      bctx.fillRect(x, 0, 1, 10);
    }
  };
  setInterval(drawBar, 500);

  (window as unknown as { __sim: unknown }).__sim = { ui, get model() { return model; } };
  requestAnimationFrame(frame);
}

main().catch((e) => {
  console.error(e);
  showFatal(String(e?.message ?? e));
});
