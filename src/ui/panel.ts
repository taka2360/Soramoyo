import GUI from 'lil-gui';
import type { Clock, Observer } from '../core/clock';
import type { Camera } from '../core/camera';
import type { RenderSettings } from '../render/renderer';
import type { AtmosphereSettings } from '../atmosphere/spectrum';
import type { PrecomputeOptions } from '../atmosphere/atmosphere';

const SPEEDS: Record<string, number> = {
  '実時間 ×1': 1,
  '×10': 10,
  '×60（1秒=1分）': 60,
  '×300': 300,
  '×1800': 1800,
  '×3600（1秒=1時間）': 3600,
};

export interface PanelDeps {
  clock: Clock;
  obs: Observer;
  rs: RenderSettings;
  cam: Camera;
  atmo: AtmosphereSettings;
  lutOpt: PrecomputeOptions;
  onAtmosphereChange: () => void;
  onJump: () => void;
}

export function buildPanel(d: PanelDeps): { refresh: () => void; gui: GUI } {
  const { clock, obs, rs, cam } = d;
  const gui = new GUI({ title: 'Soramoyo' });

  const t = {
    date: clock.localDateString(obs),
    hour: clock.localHours(obs),
    speed: 60,
    paused: false,
    now: () => {
      clock.utcMs = Date.now();
      d.onJump();
    },
  };
  clock.speed = t.speed;

  const ft = gui.addFolder('時間');
  const dateCtl = ft.add(t, 'date').name('日付 (現地)').onFinishChange((v: string) => {
    clock.setLocal(obs, v, t.hour);
    d.onJump();
  });
  const hourCtl = ft
    .add(t, 'hour', 0, 24, 0.01)
    .name('時刻 (現地, 時)')
    .onChange((v: number) => clock.setLocal(obs, t.date, v));
  ft.add(t, 'speed', SPEEDS).name('早送り').onChange((v: number) => (clock.speed = v));
  const pauseCtl = ft.add(t, 'paused').name('一時停止 [Space]').onChange((v: boolean) => (clock.paused = v));
  ft.add(t, 'now').name('現在時刻へ');

  const fo = gui.addFolder('観測地点');
  fo.add(obs, 'name').name('名前');
  fo.add(obs, 'latDeg', -90, 90, 0.0001).name('緯度 [°]');
  fo.add(obs, 'lonDeg', -180, 180, 0.0001).name('経度 [°]');
  fo.add(obs, 'elevationM', 0, 4000, 1).name('標高 [m]');
  fo.add(obs, 'utcOffsetHours', -12, 14, 0.25).name('UTC オフセット [h]');
  fo.close();

  const fc = gui.addFolder('視点');
  const h = { logH: Math.log10(cam.heightM) };
  fc.add(h, 'logH', Math.log10(1.7), Math.log10(40000), 0.01)
    .name('高さ (log10 m)')
    .onChange((v: number) => (cam.heightM = 10 ** v));

  const fa = gui.addFolder('大気');
  const change = () => d.onAtmosphereChange();
  fa.add(d.atmo, 'aerosol', 0, 8, 0.05).name('エアロゾル量（倍）').onFinishChange(change);
  fa.add(d.atmo, 'mieScaleHeight', 0.5, 4, 0.1).name('エアロゾル層の厚さ [km]').onFinishChange(change);
  fa.add(d.atmo, 'mieG', 0.6, 0.95, 0.01).name('エアロゾル前方散乱 g').onFinishChange(change);
  fa.add(d.atmo, 'ozoneDU', 0, 500, 10).name('オゾン全量 [DU]').onFinishChange(change);
  fa.add(d.atmo, 'groundAlbedo', 0.02, 0.9, 0.01).name('地面の反射率').onFinishChange(change);
  fa.add(d.lutOpt, 'wavelengthGroups', { '15波長': 5, '24波長': 8, '36波長': 12 }).name('計算波長数').onFinishChange(change);
  fa.add(d.lutOpt, 'scatteringOrders', 1, 10, 1).name('散乱の次数').onFinishChange(change);
  fa.close();

  const fv = gui.addFolder('表示');
  fv.add(rs, 'autoExposure').name('自動露出（目の順応）');
  fv.add(rs, 'exposureEV', -10, 10, 0.1).name('露出補正 [EV]');
  fv.add(rs, 'whiteBalance').name('太陽光を白に（色順応）');
  fv.add(rs, 'glare').name('まぶしさ（目の中の散乱）');
  fv.add(rs, 'grid').name('方位・高度の目盛り');
  fv.add(rs, 'renderScale', { '50%': 0.5, '75%': 0.75, '100%': 1 }).name('描画解像度');
  fv.add(rs, 'cloudScale', { '1/4': 0.25, '1/3': 0.333, '1/2': 0.5, '等倍': 1 }).name('雲の描画解像度');

  window.addEventListener('keydown', (e) => {
    if ((e.target as HTMLElement).tagName === 'INPUT') return;
    if (e.code === 'Space') {
      t.paused = !t.paused;
      clock.paused = t.paused;
      pauseCtl.updateDisplay();
      e.preventDefault();
    }
  });

  const refresh = () => {
    const active = document.activeElement;
    if (!dateCtl.domElement.contains(active)) {
      t.date = clock.localDateString(obs);
      dateCtl.updateDisplay();
    }
    if (!hourCtl.domElement.contains(active)) {
      t.hour = clock.localHours(obs);
      hourCtl.updateDisplay();
    }
  };
  return { refresh, gui };
}
