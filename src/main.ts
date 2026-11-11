import { initGpu, showFatal } from './core/gpu';
import { Clock, TOKYO } from './core/clock';
import { Camera } from './core/camera';
import { Renderer, type RenderSettings } from './render/renderer';
import { buildPanel } from './ui/panel';
import { CompassLabels } from './ui/labels';
import { AtmosphereLut, type PrecomputeOptions } from './atmosphere/atmosphere';
import { DEFAULT_ATMOSPHERE } from './atmosphere/spectrum';
import { computeEphemeris } from './astro/scene';
import { CloudVolume, type CloudSettings } from './render/clouds';
import { SimManager, SCENARIOS } from './sim/manager';
import { SkewT } from './ui/skewt';

async function main(): Promise<void> {
  const canvas = document.getElementById('sky') as HTMLCanvasElement;
  const hud = document.getElementById('hud')!;
  const gpu = await initGpu(canvas);

  const obs = { ...TOKYO };
  const clock = new Clock();
  const cam = new Camera();
  cam.attach(canvas);

  const rs: RenderSettings = { exposureEV: 0, autoExposure: true, grid: false, renderScale: 1, cloudScale: 0.5, whiteBalance: true };
  const cs: CloudSettings = { dropletsPerCc: 300, detail: 0.6, maxDistKm: 120 };
  const atmoSettings = { ...DEFAULT_ATMOSPHERE };
  const lutOpt: PrecomputeOptions = { wavelengthGroups: 8, scatteringOrders: 6 };

  const atmo = new AtmosphereLut(gpu.device, gpu.hasFloat32Filterable);
  const cloudVol = new CloudVolume(gpu.device);
  const renderer = new Renderer(gpu, canvas, atmo, cloudVol);
  const sim = new SimManager(gpu.device);
  sim.loadFor(sim.scenario, obs.latDeg, clock.utcMs, obs.utcOffsetHours);
  cloudVol.attach(sim.model);
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

  // ---- 雲シミュレーションの UI
  let simStatus = '';
  const simUi = {
    scenario: sim.scenario,
    enabled: true,
    ahead: () => { sim.catchUpTarget = sim.simUtc + 2 * 3600_000; },
    reset: () => { sim.loadFor(simUi.scenario, obs.latDeg, clock.utcMs, obs.utcOffsetHours); cloudVol.attach(sim.model); },
  };
  const fs = panel.gui.addFolder('雲シミュレーション');
  fs.add(simUi, 'scenario', Object.keys(SCENARIOS)).name('大気の状態').onChange(() => simUi.reset());
  fs.add(simUi, 'enabled').name('雲を計算・表示').onChange((v: boolean) => {
    if (v) simUi.reset(); else { sim.unload(); cloudVol.attach(null); }
  });
  fs.add(sim, 'running').name('計算を進める');
  fs.add(sim, 'maxStepsPerFrame', 1, 20, 1).name('1フレームの最大ステップ');
  fs.add(sim, 'catchUpSteps', 1, 60, 1).name('追いつき時のステップ数');
  fs.add(simUi, 'ahead').name('2時間先へ早送り');
  fs.add(simUi, 'reset').name('初期状態に戻す');
  fs.add(cs, 'dropletsPerCc', 30, 1500, 10).name('雲粒の数 [個/cm³]');
  fs.add(cs, 'detail', 0, 2, 0.05).name('細部合成の強さ');
  fs.add(cs, 'maxDistKm', 10, 200, 5).name('雲を描く距離 [km]');
  const skewT = new SkewT(document.body);
  const skewUi = { show: false };
  fs.add(skewUi, 'show').name('Skew-T 図を表示').onChange((v: boolean) => skewT.setVisible(v));

  // 開発・検証用（コンソールから時刻や視線を操作する）
  (window as unknown as { __sora: unknown }).__sora = { clock, obs, cam, rs, cs, atmoSettings, lutOpt, recompute, renderer, sim, cloudVol, simUi };

  let last = performance.now();
  let fps = 60;
  let uiTick = 0;
  let lastCamKey = '';
  let lastClockSet = clock.utcMs;
  let pendingJump: { target: number; at: number } | null = null;
  const frame = (now: number) => {
    const dtReal = Math.min(0.1, (now - last) / 1000);
    last = now;
    fps = fps * 0.95 + (1 / Math.max(dtReal, 1e-4)) * 0.05;

    // 時計と雲シミュレーション：計算が追いつく範囲で時計を進める
    // ユーザーが時計を動かしたら、操作が落ち着いてから（0.4秒）追いつき or 計算し直し
    if (sim.model && Math.abs(clock.utcMs - lastClockSet) > 1000) {
      pendingJump = { target: clock.utcMs, at: now };
      lastClockSet = clock.utcMs;
    }
    if (pendingJump && sim.model) {
      if (now - pendingJump.at > 400) {
        if (Math.abs(pendingJump.target - sim.simUtc) > 120_000) {
          sim.jumpTo(pendingJump.target, obs.latDeg, obs.utcOffsetHours);
          cloudVol.attach(sim.model);
          renderer.resetAdaptation();
        }
        pendingJump = null;
      }
    } else if (sim.model && sim.running && !sim.busy) {
      // 地表の日射条件（太陽高度は計算の時刻で）
      const sunNow = computeEphemeris(2440587.5 + sim.simUtc / 86400_000, obs, obs.elevationM);
      const cz = sunNow.sunDir[2];
      sim.model.env.cosZen = cz;
      sim.model.env.swClear = cz > 0 ? 1098 * cz * Math.exp(-0.057 / cz) : 0; // Haurwitz の晴天日射
      sim.nDrop = cs.dropletsPerCc * 1e6;
      const want = dtReal * clock.effectiveSpeed;
      sim.frame(want, dtReal);
      clock.utcMs = sim.simUtc;
      lastClockSet = clock.utcMs;
    } else if (!sim.model) {
      clock.advance(dtReal);
    }
    simStatus = sim.catchUpTarget !== null ? `追いつき中（あと ${((sim.catchUpTarget - sim.simUtc) / 3600_000).toFixed(1)} 時間）` : '';

    const eph = computeEphemeris(clock.julianDayUT, obs, obs.elevationM + cam.heightM);
    // カメラが動いたら雲の時間蓄積をやり直す（再投影はまだ無いため）
    const camKey = `${cam.azimuth.toFixed(5)},${cam.altitude.toFixed(5)},${cam.fovY.toFixed(5)},${cam.heightM.toFixed(2)},${canvas.clientWidth}x${canvas.clientHeight},${rs.renderScale},${rs.cloudScale}`;
    const histAlpha = camKey === lastCamKey ? 0.15 : 1;
    lastCamKey = camKey;
    cloudVol.writeUniform(sim.origin, eph.sunDir, sim.model?.time ?? 0, cs, atmoSettings.groundAlbedo, histAlpha);
    if (sim.model && !sim.busy) cloudVol.update();
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
      skewT.draw(sim.model);
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
        (sim.model
          ? `\n雲: ${sim.simCase!.name}  経過 ${(sim.model.time / 3600).toFixed(2)} 時間  dt ${sim.model.dt.toFixed(2)} s  実効 ×${sim.effectiveSpeed.toFixed(0)}` +
            (sim.model.lastStats
              ? `\n雲底 ${((sim.model.lastStats.cloudBase ?? NaN) / 1000).toFixed(2)} km  雲頂 ${((sim.model.lastStats.cloudTop ?? NaN) / 1000).toFixed(2)} km  w最大 ${sim.model.lastStats.maxW.toFixed(1)} m/s  LWP ${sim.model.lastStats.lwp.toFixed(1)} g/m²` +
                (sim.simCase!.config.surfaceEnergy
                  ? `  地表 ${(sim.model.lastStats.surface.ts - 273.15).toFixed(1)}℃  日射 ${sim.model.lastStats.surface.sw.toFixed(0)}  顕熱 ${sim.model.lastStats.surface.h.toFixed(0)}  潜熱 ${sim.model.lastStats.surface.le.toFixed(0)} W/m²`
                  : '')
              : '') +
            (simStatus ? `  ${simStatus}` : '')
          : '') +
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
