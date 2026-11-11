import type { Camera } from '../core/camera';

/** 地平線上の方位ラベル（北・東・南・西と 45° ごと）と高度目盛り */
export class CompassLabels {
  private items: { el: HTMLElement; dir: number[] }[] = [];

  constructor(root: HTMLElement) {
    const names = ['北', '北東', '東', '南東', '南', '南西', '西', '北西'];
    names.forEach((n, i) => {
      const az = (i * Math.PI) / 4;
      this.add(root, n, [Math.sin(az), Math.cos(az), 0]);
    });
    for (const alt of [30, 60]) {
      const a = (alt * Math.PI) / 180;
      for (let i = 0; i < 4; i++) {
        const az = (i * Math.PI) / 2;
        this.add(root, `${alt}°`, [Math.sin(az) * Math.cos(a), Math.cos(az) * Math.cos(a), Math.sin(a)]);
      }
    }
    this.add(root, '天頂', [0, 0, 1]);
  }

  private add(root: HTMLElement, text: string, dir: number[]): void {
    const el = document.createElement('div');
    el.className = 'lbl';
    el.textContent = text;
    root.appendChild(el);
    this.items.push({ el, dir });
  }

  update(cam: Camera, visible: boolean, w: number, h: number): void {
    for (const it of this.items) {
      const p = visible ? cam.project(it.dir, w, h) : null;
      if (!p || p[0] < -50 || p[0] > w + 50 || p[1] < -50 || p[1] > h + 50) {
        it.el.style.display = 'none';
      } else {
        it.el.style.display = 'block';
        it.el.style.left = `${p[0]}px`;
        it.el.style.top = `${p[1]}px`;
      }
    }
  }
}
