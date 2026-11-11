import type { CloudModel } from '../sim/model';
import { CP, G, LV, RD, EPS, qvsat } from '../sim/thermo';

/**
 * Skew-T log-p 図（気象のサウンディング図）。
 *  - 実線：計算領域の水平平均の気温（赤）と露点（緑）
 *  - 点線：初期サウンディング
 *  - 灰の線：最下層の空気塊を持ち上げたときの温度（乾燥断熱 → 凝結後は湿潤断熱）
 * CAPE（対流有効位置エネルギー）も表示する。
 */
export class SkewT {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  visible = false;

  constructor(parent: HTMLElement) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = 420;
    this.canvas.height = 460;
    Object.assign(this.canvas.style, {
      position: 'fixed', right: '260px', bottom: '10px', width: '360px', height: '394px',
      background: 'rgba(250,250,247,0.92)', borderRadius: '4px', display: 'none', pointerEvents: 'none',
    });
    parent.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d')!;
  }

  setVisible(v: boolean): void {
    this.visible = v;
    this.canvas.style.display = v ? 'block' : 'none';
  }

  draw(model: CloudModel | null): void {
    if (!this.visible) return;
    const c = this.ctx;
    const W = this.canvas.width, H = this.canvas.height;
    const L = 40, R = W - 10, T = 26, B = H - 30;
    const pTop = model ? Math.max(100, model.cfg.base.p0[model.cfg.grid.nz - 1] / 100) : 100;
    const pBot = model ? Math.ceil(model.cfg.base.p0[0] / 100 / 10) * 10 + 10 : 1050;
    const yOf = (p: number) => B - ((Math.log(pBot) - Math.log(p)) / (Math.log(pBot) - Math.log(pTop))) * (B - T);
    // 温度の範囲は地表気温の付近に合わせ、傾き（スキュー）は標準的な気温減率の線がほぼ垂直になるよう決める
    const tSfc = model ? model.cfg.base.th0[0] * model.cfg.base.pi0[0] - 273.15 : 25;
    const tMax = Math.ceil((tSfc + 12) / 10) * 10, tMin = tMax - 60;
    const kx = (R - L) / (tMax - tMin);
    const ztopKm = model ? model.cfg.grid.ztop / 1000 : 10;
    const skew = (kx * 6.5 * ztopKm) / (B - T);
    const xOf = (tc: number, p: number) => L + (tc - tMin) * kx + skew * (B - yOf(p));
    c.clearRect(0, 0, W, H);
    c.save();
    c.beginPath();
    c.rect(L, T, R - L, B - T);
    c.clip();
    // 等温線・等圧線・乾燥断熱線
    c.lineWidth = 1;
    c.strokeStyle = '#ccc';
    for (let tc = -150; tc <= 60; tc += 10) {
      c.beginPath(); c.moveTo(xOf(tc, pBot), yOf(pBot)); c.lineTo(xOf(tc, pTop), yOf(pTop)); c.stroke();
    }
    c.strokeStyle = '#e8d8c8';
    for (let th = 250; th <= 450; th += 10) {
      c.beginPath();
      for (let p = pBot; p >= pTop; p -= 10) {
        const tc = th * Math.pow(p / 1000, RD / CP) - 273.15;
        const x = xOf(tc, p), y = yOf(p);
        if (p === pBot) c.moveTo(x, y); else c.lineTo(x, y);
      }
      c.stroke();
    }
    c.strokeStyle = '#bbb';
    for (const p of [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]) {
      if (p < pTop) continue;
      c.beginPath(); c.moveTo(L, yOf(p)); c.lineTo(R, yOf(p)); c.stroke();
    }
    let cape = 0, cin = 0, lcl = NaN;
    if (model) {
      const g = model.cfg.grid, b = model.cfg.base;
      const np = g.nx * g.ny;
      const st = model.lastStats;
      const line = (temps: (k: number) => number, color: string, dash: number[] = [], width = 2) => {
        c.strokeStyle = color; c.lineWidth = width; c.setLineDash(dash);
        c.beginPath();
        for (let k = 0; k < g.nz; k++) {
          const p = b.p0[k] / 100;
          const x = xOf(temps(k), p), y = yOf(p);
          if (k === 0) c.moveTo(x, y); else c.lineTo(x, y);
        }
        c.stroke();
        c.setLineDash([]);
      };
      const dew = (qv: number, p: number) => {
        const e = Math.max((qv * p) / (EPS + qv), 1e-3);
        const x = Math.log(e / 611.2);
        return (243.5 * x) / (17.67 - x);
      };
      line((k) => b.th0[k] * b.pi0[k] - 273.15, '#d33', [4, 4], 1.5);
      line((k) => dew(b.qv0[k], b.p0[k]), '#393', [4, 4], 1.5);
      if (st) {
        const th = (k: number) => st.levels[k].sumTh / np;
        const qv = (k: number) => st.levels[k].sumQv / np;
        line((k) => th(k) * b.pi0[k] - 273.15, '#c00');
        line((k) => dew(qv(k), b.p0[k]), '#080');
        // 最下層の空気塊を持ち上げる（凝結熱込み、可逆でない偽断熱）
        let T0 = th(0) * b.pi0[0];
        const q0 = qv(0);
        const parcel: number[] = [];
        let Tp = T0;
        let qp = q0;
        for (let k = 0; k < g.nz; k++) {
          const p = b.p0[k];
          if (k > 0) {
            const dz = g.zc[k] - g.zc[k - 1];
            const qs = qvsat(Tp, p);
            if (qp < qs) {
              Tp -= (G / CP) * dz;
            } else {
              if (Number.isNaN(lcl)) lcl = g.zc[k - 1];
              // 湿潤断熱減率
              const gm = (G * (1 + (LV * qs) / (RD * Tp))) / (CP + (LV * LV * qs * EPS) / (RD * Tp * Tp));
              Tp -= gm * dz;
              qp = qvsat(Tp, p);
            }
          }
          parcel.push(Tp);
          const Te = th(k) * b.pi0[k] * (1 + 0.608 * qv(k));
          const Tpv = Tp * (1 + 0.608 * Math.min(qp, qvsat(Tp, p)));
          const buoy = (G * (Tpv - Te)) / Te;
          const dz = g.dzc[k];
          if (buoy > 0) cape += buoy * dz; else if (cape === 0) cin += buoy * dz;
        }
        T0 = 0;
        void T0;
        line((k) => parcel[k] - 273.15, '#555', [2, 3], 1.5);
      }
    }
    c.restore();
    // 目盛りと文字
    c.fillStyle = '#333';
    c.font = '11px system-ui, sans-serif';
    for (const p of [1000, 850, 700, 500, 300, 200, 100]) {
      if (p < pTop) continue;
      c.fillText(String(p), 4, yOf(p) + 4);
    }
    for (let tc = tMin; tc <= tMax; tc += 10) c.fillText(`${tc}`, xOf(tc, pBot) - 8, B + 14);
    c.font = '12px system-ui, sans-serif';
    c.fillText('Skew-T（hPa / ℃）  実線：現在の領域平均  点線：初期', 8, 16);
    if (model) {
      c.fillText(`CAPE ${cape.toFixed(0)} J/kg` + (cape > 0 ? `   CIN ${cin.toFixed(0)} J/kg` : '') + `   持ち上げ凝結高度 ${Number.isNaN(lcl) ? '—' : (lcl / 1000).toFixed(2) + ' km'}`, 8, H - 6);
    }
  }
}
