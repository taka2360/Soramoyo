import { CP, G, P00, RD, KAPPA } from './thermo';

/**
 * 計算格子：Arakawa C 格子。水平は等間隔・周期境界、鉛直は不等間隔（ストレッチ）可。
 *   u: x 面 (i-1/2)、v: y 面、w: 下面 (k-1/2)、スカラー: セル中心
 * 地表 w(k=0)=0、上端 w(k=nz)=0（剛体蓋＋スポンジ層）。
 */
export interface GridSpec {
  nx: number;
  ny: number;
  dx: number;
  dy: number;
  /** 鉛直のセル境界の高さ zf[0..nz]（zf[0]=0） [m] */
  zf: number[];
}

export interface Grid extends GridSpec {
  nz: number;
  zc: number[];
  /** セルの厚さ zf[k+1]-zf[k] */
  dzc: number[];
  /** 中心間隔 zc[k]-zc[k-1]（k>=1、w 点の厚さ） */
  dzw: number[];
  ztop: number;
}

export function uniformLevels(nz: number, dz: number): number[] {
  return Array.from({ length: nz + 1 }, (_, k) => k * dz);
}

/**
 * 不等間隔：地表から zConst までは dzMin、そこから上は比率 r で伸ばし、dzMax で頭打ち。
 * ztop に達するまで層を積む（nz は結果で決まる）。
 */
export function stretchedLevels(dzMin: number, zConst: number, ratio: number, dzMax: number, ztop: number): number[] {
  const zf = [0];
  let dz = dzMin;
  while (zf[zf.length - 1] < ztop - 1e-6) {
    const z = zf[zf.length - 1];
    if (z >= zConst) dz = Math.min(dz * ratio, dzMax);
    zf.push(Math.min(z + dz, ztop));
  }
  return zf;
}

export function makeGrid(s: GridSpec): Grid {
  const nz = s.zf.length - 1;
  const zc: number[] = [], dzc: number[] = [], dzw: number[] = [];
  for (let k = 0; k < nz; k++) {
    zc.push(0.5 * (s.zf[k] + s.zf[k + 1]));
    dzc.push(s.zf[k + 1] - s.zf[k]);
  }
  for (let k = 0; k < nz; k++) dzw.push(k === 0 ? dzc[0] * 0.5 : zc[k] - zc[k - 1]);
  return { ...s, nz, zc, dzc, dzw, ztop: s.zf[nz] };
}

/** 環境場（サウンディング）：高さの関数として与える */
export interface Sounding {
  surfacePressure: number;
  theta: (z: number) => number;
  qv: (z: number) => number;
  u: (z: number) => number;
  v: (z: number) => number;
}

/** 基本場（各層の値）。column バッファに詰めて GPU に渡す */
export interface BaseState {
  th0: number[]; qv0: number[]; pi0: number[]; p0: number[]; rho0: number[]; // セル中心
  rhoF: number[]; // 下面（w 点）
  u0: number[]; v0: number[];
}

/**
 * 大規模場の強制（計算領域より大きなスケールの効果を外から与える）。
 * 周期境界の領域では、地表から熱と水が入り続けるので、これで収支を閉じる。
 */
export interface LargeScaleForcing {
  /** 地衡風（地面に対する速度）[m/s] */
  ug: (z: number) => number;
  vg: (z: number) => number;
  /** 大規模な沈降（負が下降）[m/s] */
  wsub: (z: number) => number;
  /** θ への大規模移流＋放射の強制 [K/s] */
  thLs: (z: number) => number;
  /** qv への大規模移流の強制 [1/s] */
  qvLs: (z: number) => number;
  /** 基本場（初期サウンディング）へのナッジング率 [1/s] */
  nudge: (z: number) => number;
}

export const NO_FORCING: LargeScaleForcing = {
  ug: () => 0, vg: () => 0, wsub: () => 0, thLs: () => 0, qvLs: () => 0, nudge: () => 0,
};

/** 静水圧平衡をエクスナー関数で積分して基本場を作る（仮温位を使用） */
export function buildBaseState(grid: Grid, snd: Sounding): BaseState {
  const pis = Math.pow(snd.surfacePressure / P00, KAPPA);
  // 細かい刻みで積分し、必要な高さの値を拾う
  const piAt = (z: number): number => {
    const n = Math.max(1, Math.ceil(z / 10));
    const h = z / n;
    let pi = pis;
    for (let i = 0; i < n; i++) {
      const zm = (i + 0.5) * h;
      const thv = snd.theta(zm) * (1 + 0.608 * snd.qv(zm));
      pi -= (G / (CP * thv)) * h;
    }
    return pi;
  };
  const b: BaseState = { th0: [], qv0: [], pi0: [], p0: [], rho0: [], rhoF: [], u0: [], v0: [] };
  const rhoOf = (z: number, pi: number) => {
    const p = P00 * Math.pow(pi, 1 / KAPPA);
    const T = snd.theta(z) * pi;
    return p / (RD * T * (1 + 0.608 * snd.qv(z)));
  };
  for (let k = 0; k < grid.nz; k++) {
    const z = grid.zc[k];
    const pi = piAt(z);
    b.th0.push(snd.theta(z));
    b.qv0.push(snd.qv(z));
    b.pi0.push(pi);
    b.p0.push(P00 * Math.pow(pi, 1 / KAPPA));
    b.rho0.push(rhoOf(z, pi));
    b.u0.push(snd.u(z));
    b.v0.push(snd.v(z));
    const zf = grid.zf[k];
    b.rhoF.push(rhoOf(zf, piAt(zf)));
  }
  return b;
}

/** column バッファの1層あたりの要素数（common.wgsl の Col と一致させる） */
export const COL_STRIDE = 24;

export function packColumn(grid: Grid, b: BaseState, f: LargeScaleForcing = NO_FORCING): Float32Array<ArrayBuffer> {
  const a = new Float32Array(grid.nz * COL_STRIDE);
  for (let k = 0; k < grid.nz; k++) {
    a.set(
      [
        grid.zc[k], grid.zf[k], grid.dzc[k], grid.dzw[k],
        b.rho0[k], b.rhoF[k], b.th0[k], b.qv0[k],
        b.pi0[k], b.p0[k], b.u0[k], b.v0[k],
        f.ug(grid.zc[k]), f.vg(grid.zc[k]), f.wsub(grid.zf[k]), f.thLs(grid.zc[k]),
        f.qvLs(grid.zc[k]), f.nudge(grid.zc[k]), 0, 0,
        0, 0, 0, 0,
      ],
      k * COL_STRIDE,
    );
  }
  return a;
}
