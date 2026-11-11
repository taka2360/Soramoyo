import { initGpu, showFatal } from './core/gpu';
import { Clock, TOKYO } from './core/clock';
import { Camera } from './core/camera';
import { Renderer, type RenderSettings } from './render/renderer';
import { buildPanel } from './ui/panel';
import { CompassLabels } from './ui/labels';
import { AtmosphereLut, type PrecomputeOptions } from './atmosphere/atmosphere';
import { DEFAULT_ATMOSPHERE } from './atmosphere/spectrum';
import { computeEphemeris } from './astro/scene';

async function main(): Promise<void> {
  const canvas = document.getElementById('sky') as HTMLCanvasElement;
  const hud = document.getElementById('hud')!;
  const gpu = await initGpu(canvas);

  const obs = { ...TOKYO };
  const clock = new Clock();
  const cam = new Camera();
  cam.attach(canvas);

  const rs: RenderSettings = { exposureEV: 0, autoExposure: true, grid: false, renderScale: 1, whiteBalance: true };
  const atmoSettings = { ...DEFAULT_ATMOSPHERE };
  const lutOpt: PrecomputeOptions = { wavelengthGroups: 8, scatteringOrders: 6 };

  const atmo = new AtmosphereLut(gpu.device, gpu.hasFloat32Filterable);
  const renderer = new Renderer(gpu, canvas, atmo);
  const labels = new CompassLabels(document.getElementById('labels')!);

  // 大気 LUT の再計算（設定変更が続いたら最後の1回だけ実行）
  let lutStatus = '';
  let lutBusy = false;
  let lutDirty = true;
  const recompute = async () => {
    if (lutBusy) { lutDirty = true; return; }
    lutBusy = true;
    lutDirty = false;
    try {
      await atmo.precompute(atmoSettings, lutOpt, (p) => (lutStatus = `大気 LUT 計算中 ${(p * 100).toFixed(0)}%`));
      lutStatus = `大気 LUT ${lutOpt.wavelengthGroups * 3}波長・${lutOpt.scatteringOrders}次散乱 ${(atmo.lastPrecomputeMs / 1000).toFixed(1)}秒`;
    } finally {
      lutBusy = false;
      if (lutDirty) recompute();
    }
  };
  recompute();

  const panel = buildPanel({ clock, obs, rs, cam, atmo: atmoSettings, lutOpt, onAtmosphereChange: recompute, onJump: () => renderer.resetAdaptation() });

  // 開発・検証用（コンソールから時刻や視線を操作する）
  (window as unknown as { __sora: unknown }).__sora = { clock, obs, cam, rs, atmoSettings, lutOpt, recompute, renderer };

  let last = performance.now();
  let fps = 60;
  let uiTick = 0;
  const frame = (now: number) => {
    const dtReal = Math.min(0.1, (now - last) / 1000);
    last = now;
    fps = fps * 0.95 + (1 / Math.max(dtReal, 1e-4)) * 0.05;

    clock.advance(dtReal);
    // ここに段階3以降で「雲シミュレーションを目標時刻まで進める」処理が入る

    const eph = computeEphemeris(clock.julianDayUT, obs, obs.elevationM + cam.heightM);
    renderer.render(
      cam,
      rs,
      { ...eph, groundElevationM: obs.elevationM, groundAlbedo: atmoSettings.groundAlbedo },
      dtReal,
      now / 1000,
    );
    labels.update(cam, rs.grid, canvas.clientWidth, canvas.clientHeight);

    if (++uiTick % 10 === 0) {
      panel.refresh();
      const [w, h] = renderer.size;
      const gpuMs = [...renderer.timer.ms].map(([k, v]) => `${k} ${v.toFixed(2)}`).join('  ');
      const deg = 180 / Math.PI;
      const sunAlt = Math.asin(eph.sunDirApparent[2]) * deg;
      const sunAz = ((Math.atan2(eph.sunDir[0], eph.sunDir[1]) * deg) + 360) % 360;
      hud.textContent =
        `${obs.name}  ${clock.localDateString(obs)} ${clock.localTimeString(obs)} (UTC${obs.utcOffsetHours >= 0 ? '+' : ''}${obs.utcOffsetHours})` +
        `  再生 ×${clock.effectiveSpeed}\n` +
        `太陽 高度 ${sunAlt.toFixed(2)}°（大気差込み） 方位 ${sunAz.toFixed(2)}°\n` +
        `視線 方位 ${(cam.azimuth * deg).toFixed(1)}°  高度 ${(cam.altitude * deg).toFixed(1)}°  画角 ${(cam.fovY * deg).toFixed(1)}°  視点 ${cam.heightM.toFixed(0)} m\n` +
        `${w}×${h}  ${fps.toFixed(0)} fps` +
        (renderer.timer.enabled ? `  GPU[ms] ${gpuMs}` : '  (GPU 計測なし)') +
        `\n${lutStatus}${atmo.format === 'rgba16float' ? '（float32 フィルタ非対応のため半精度）' : ''}`;
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main().catch((e) => {
  console.error(e);
  showFatal(String(e?.message ?? e));
});
