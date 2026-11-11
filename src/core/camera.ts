/**
 * 見回し用カメラ。座標系は ENU（x=東, y=北, z=上）、単位 m。
 * - azimuth: 北を 0 として東回り [rad]
 * - altitude: 水平を 0 として上向き [rad]
 * - fovY: 縦画角 [rad]（ホイールでズーム）
 * - heightM: 地表からの高さ（雲の上に出る視点用）
 */
export class Camera {
  azimuth = Math.PI * 0.5; // 東向き
  altitude = 0.25;
  fovY = (70 * Math.PI) / 180;
  heightM = 1.7;

  static readonly MIN_FOV = (1 * Math.PI) / 180;
  static readonly MAX_FOV = (120 * Math.PI) / 180;

  /** 前・右・上の基底ベクトル */
  basis(): { fwd: number[]; right: number[]; up: number[] } {
    const ca = Math.cos(this.altitude), sa = Math.sin(this.altitude);
    const cz = Math.cos(this.azimuth), sz = Math.sin(this.azimuth);
    const fwd = [sz * ca, cz * ca, sa];
    const right = [cz, -sz, 0];
    // up = right × fwd
    const up = [
      right[1] * fwd[2] - right[2] * fwd[1],
      right[2] * fwd[0] - right[0] * fwd[2],
      right[0] * fwd[1] - right[1] * fwd[0],
    ];
    return { fwd, right, up };
  }

  /** 方向ベクトル（ENU）を画面座標 [px] に投影。背後なら null。 */
  project(dir: number[], width: number, height: number): [number, number] | null {
    const { fwd, right, up } = this.basis();
    const z = dir[0] * fwd[0] + dir[1] * fwd[1] + dir[2] * fwd[2];
    if (z <= 1e-4) return null;
    const x = dir[0] * right[0] + dir[1] * right[1] + dir[2] * right[2];
    const y = dir[0] * up[0] + dir[1] * up[1] + dir[2] * up[2];
    const t = Math.tan(this.fovY * 0.5);
    const aspect = width / height;
    const sx = x / z / (t * aspect);
    const sy = y / z / t;
    return [(sx * 0.5 + 0.5) * width, (0.5 - sy * 0.5) * height];
  }

  attach(el: HTMLElement): void {
    let dragging = false;
    let lx = 0, ly = 0;
    el.addEventListener('pointerdown', (e) => {
      dragging = true;
      lx = e.clientX; ly = e.clientY;
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener('pointerup', (e) => {
      dragging = false;
      el.releasePointerCapture(e.pointerId);
    });
    el.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const dx = e.clientX - lx, dy = e.clientY - ly;
      lx = e.clientX; ly = e.clientY;
      // 画面の高さぶんドラッグ = 縦画角ぶん回転（つかんで引っ張る感覚）
      const k = this.fovY / el.clientHeight;
      this.azimuth -= dx * k;
      this.altitude += dy * k;
      this.altitude = Math.max(-Math.PI / 2 + 1e-3, Math.min(Math.PI / 2 - 1e-3, this.altitude));
      this.azimuth = ((this.azimuth % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    });
    el.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        const f = Math.exp(e.deltaY * 0.001);
        this.fovY = Math.max(Camera.MIN_FOV, Math.min(Camera.MAX_FOV, this.fovY * f));
      },
      { passive: false },
    );
  }
}
