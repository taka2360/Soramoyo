import { type Grid, type Sounding, buildBaseState, makeGrid, uniformLevels, stretchedLevels } from './grid';
import { type ModelConfig, NFIELDS, F } from './model';
import { CP, G, LV, qvsat, saturationAdjust, exner, P00, KAPPA } from './thermo';

export interface SimCase {
  name: string;
  description: string;
  /** 推奨する計算時間 [s] */
  duration: number;
  config: ModelConfig;
  initial: Float32Array<ArrayBuffer>;
}

function alloc(g: Grid): Float32Array<ArrayBuffer> {
  return new Float32Array(g.nx * g.ny * g.nz * NFIELDS);
}

/** 決定的な乱数（初期擾乱を毎回同じにするため） */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function fillBase(g: Grid, base: ReturnType<typeof buildBaseState>, a: Float32Array): void {
  const n = g.nx * g.ny * g.nz;
  const plane = g.nx * g.ny;
  for (let k = 0; k < g.nz; k++) {
    for (let p = 0; p < plane; p++) {
      const id = p + plane * k;
      a[F.U * n + id] = base.u0[k];
      a[F.V * n + id] = base.v0[k];
      a[F.TH * n + id] = base.th0[k];
      a[F.QV * n + id] = base.qv0[k];
    }
  }
}

// ---------------------------------------------------------------------------
/**
 * Straka et al. (1993) 密度流（乾燥・2D）。中立成層に冷気塊を置き、地表を這う重力流を見る。
 * 標準解（900秒）：前線位置 ≈ 15.5 km、最小 θ' ≈ −9.8 K（圧縮性モデルの収束解）。
 */
export function straka(dx = 100): SimCase {
  const nx = Math.round(51200 / dx);
  const pow2 = 2 ** Math.round(Math.log2(nx));
  const grid = makeGrid({ nx: pow2, ny: 1, dx: 51200 / pow2, dy: 51200 / pow2, zf: uniformLevels(Math.round(6400 / dx), dx) });
  const snd: Sounding = { surfacePressure: 1e5, theta: () => 300, qv: () => 0, u: () => 0, v: () => 0 };
  const base = buildBaseState(grid, snd);
  const a = alloc(grid);
  fillBase(grid, base, a);
  const n = grid.nx * grid.ny * grid.nz;
  for (let k = 0; k < grid.nz; k++) {
    for (let i = 0; i < grid.nx; i++) {
      const x = (i + 0.5) * grid.dx - 25600;
      const z = grid.zc[k];
      const L = Math.hypot(x / 4000, (z - 3000) / 2000);
      if (L <= 1) {
        const dT = -15 * (Math.cos(Math.PI * L) + 1) / 2;
        a[F.TH * n + i + grid.nx * grid.ny * k] += dT / base.pi0[k];
      }
    }
  }
  return {
    name: 'Straka 密度流（乾燥 2D）',
    description: '冷気塊が落下して地表を広がる。900秒で前線 ≈15.5km、最小θ′ ≈ −9.8K が標準解（圧縮性モデル）。',
    duration: 900,
    config: {
      grid, base, moist: false, rain: false, smagorinsky: false, nuConst: 75, cs: 0.18, prandtl: 1,
      spongeZ: 1e9, spongeRate: 0, dtMax: 1.0, cfl: 0.8,
    },
    initial: a,
  };
}

// ---------------------------------------------------------------------------
/**
 * Bryan & Fritsch (2002) 型の湿潤気泡（2D、雨なし）。
 * 環境は「全水量 0.02 で飽和した湿潤中立」。本家は圧縮性・厳密な熱力学なので、
 * ここではこのモデル自身の熱力学で湿潤中立になるよう、飽和した空気塊を持ち上げて環境を作る。
 */
export function bryanFritsch(dx = 100): SimCase {
  const nx = 2 ** Math.round(Math.log2(20000 / dx));
  const grid = makeGrid({ nx, ny: 1, dx: 20000 / nx, dy: 20000 / nx, zf: uniformLevels(Math.round(10000 / dx), dx) });
  const qt = 0.02;
  // 地表で飽和（θ=289.8K 付近）、その後は可逆な湿潤断熱で持ち上げる
  const levels: { z: number; th: number; qv: number; qc: number }[] = [];
  {
    let p = 1e5;
    let th = 289.8;
    let st = saturationAdjust(th, qt, 0, p, exner(p));
    let z = 0;
    const dz = 10;
    levels.push({ z, th: st[0], qv: st[1], qc: st[2] });
    while (z < 10000 + dx) {
      // 静水圧（仮温位、凝結物の荷重込み）
      const thv = st[0] * (1 + 0.608 * st[1] - st[2]);
      const pi = exner(p) - (G / (CP * thv)) * dz;
      p = P00 * Math.pow(pi, 1 / KAPPA);
      z += dz;
      // 全水量・液水温位を保存して持ち上げ → 飽和調整
      const thl = st[0] - (LV / (CP * exner(p))) * st[2];
      st = saturationAdjust(thl, qt, 0, p, pi);
      th = st[0];
      levels.push({ z, th, qv: st[1], qc: st[2] });
    }
  }
  const at = (z: number, key: 'th' | 'qv' | 'qc') => {
    const i = Math.min(levels.length - 2, Math.max(0, Math.floor(z / 10)));
    const f = (z - levels[i].z) / 10;
    return levels[i][key] * (1 - f) + levels[i + 1][key] * f;
  };
  // 基本場の θ は「仮温位が同じになる乾燥大気」ではなく、そのままの θ と qv（qc は初期状態に入れる）
  const snd: Sounding = { surfacePressure: 1e5, theta: (z) => at(z, 'th'), qv: (z) => at(z, 'qv'), u: () => 0, v: () => 0 };
  const base = buildBaseState(grid, snd);
  const a = alloc(grid);
  fillBase(grid, base, a);
  const n = grid.nx * grid.ny * grid.nz;
  const plane = grid.nx * grid.ny;
  for (let k = 0; k < grid.nz; k++) {
    for (let i = 0; i < grid.nx; i++) {
      const id = i + plane * k;
      const x = (i + 0.5) * grid.dx - 10000;
      const z = grid.zc[k];
      let th = base.th0[k];
      const L = Math.hypot(x / 2000, (z - 2000) / 2000);
      if (L <= 1) th += 2 * Math.cos((Math.PI * L) / 2) ** 2;
      // 飽和を保つ：qv = qvs(T), qc = qt − qv
      const qs = qvsat(th * base.pi0[k], base.p0[k]);
      a[F.TH * n + id] = th;
      a[F.QV * n + id] = Math.min(qs, qt);
      a[F.QC * n + id] = Math.max(qt - qs, 0);
    }
  }
  // 雲水の荷重を基本場の浮力と整合させるため、qc の水平平均分を基本場の qv0 に反映…はせず、
  // 浮力は (θ−θ0)/θ0 + 0.608(qv−qv0) − qc なので、環境の qc による一様な負の浮力は
  // 水平平均として圧力に吸収される（周期境界・非弾性のため）。
  return {
    name: 'Bryan–Fritsch 型 湿潤気泡（2D）',
    description: '飽和した湿潤中立の大気中で暖気泡が上昇。凝結熱で加速しながら雲の中を昇る（雨なし）。',
    duration: 1000,
    config: {
      grid, base, moist: true, rain: false, smagorinsky: true, nuConst: 0, cs: 0.18, prandtl: 1 / 3,
      spongeZ: 1e9, spongeRate: 0, dtMax: 1.0, cfl: 0.8,
    },
    initial: a,
  };
}

// ---------------------------------------------------------------------------
/**
 * Weisman & Klemp (1982) の解析的サウンディング＋暖気泡（3D、雨あり）。
 * 条件付き不安定な大気で、暖気泡から積雲が立ち、積乱雲へ発達して雨を降らせる。
 */
export function weismanKlemp(opts: { nx: number; dx: number; dzMin: number; shear: number; qv0: number }): SimCase {
  const ztr = 12000, thTr = 343, Ttr = 213, th0s = 300;
  const zf = stretchedLevels(opts.dzMin, 1000, 1.04, Math.max(opts.dzMin, 500), 20000);
  const grid = makeGrid({ nx: opts.nx, ny: opts.nx, dx: opts.dx, dy: opts.dx, zf });
  const theta = (z: number) =>
    z <= ztr ? th0s + (thTr - th0s) * Math.pow(z / ztr, 1.25) : thTr * Math.exp((G / (CP * Ttr)) * (z - ztr));
  const rh = (z: number) => (z <= ztr ? 1 - 0.75 * Math.pow(z / ztr, 1.25) : 0.25);
  // qv は RH から求める（気圧が必要なので、乾燥で一度静水圧を積分して近似）
  const pDry = (z: number) => {
    let pi = 1;
    const n = Math.max(1, Math.ceil(z / 50));
    for (let i = 0; i < n; i++) pi -= (G / (CP * theta((i + 0.5) * (z / n)))) * (z / n);
    return P00 * Math.pow(pi, 1 / KAPPA);
  };
  const qv = (z: number) => {
    const p = pDry(z);
    const T = theta(z) * Math.pow(p / P00, KAPPA);
    return Math.min(opts.qv0, rh(z) * qvsat(T, p));
  };
  // 計算を軽くするため qv(z) は表にしておく
  const qtab = Array.from({ length: 401 }, (_, i) => qv(i * 50));
  const qvf = (z: number) => {
    const x = Math.min(399.999, z / 50);
    const i = Math.floor(x);
    return qtab[i] + (qtab[i + 1] - qtab[i]) * (x - i);
  };
  const us = opts.shear;
  // 平均風で動く座標系（Galilean 変換）：u − us/2
  const snd: Sounding = {
    surfacePressure: 1e5,
    theta,
    qv: qvf,
    u: (z) => us * Math.tanh(z / 3000) - us / 2,
    v: () => 0,
  };
  const base = buildBaseState(grid, snd);
  const a = alloc(grid);
  fillBase(grid, base, a);
  const n = grid.nx * grid.ny * grid.nz;
  const plane = grid.nx * grid.ny;
  const r = rng(12345);
  const Lx = grid.nx * grid.dx;
  for (let k = 0; k < grid.nz; k++) {
    const z = grid.zc[k];
    for (let j = 0; j < grid.ny; j++) {
      for (let i = 0; i < grid.nx; i++) {
        const id = i + grid.nx * j + plane * k;
        const x = (i + 0.5) * grid.dx - Lx * 0.4;
        const y = (j + 0.5) * grid.dy - Lx * 0.5;
        const L = Math.hypot(x / 10000, y / 10000, (z - 1400) / 1400);
        let dth = 0;
        if (L <= 1) dth += 2 * Math.cos((Math.PI * L) / 2) ** 2;
        if (z < 1000) dth += (r() - 0.5) * 0.2; // 対称性を崩す小さな擾乱
        a[F.TH * n + id] += dth;
      }
    }
  }
  return {
    name: 'Weisman–Klemp 積乱雲（3D）',
    description: `条件付き不安定な大気に暖気泡を置く。積雲→積乱雲→降水。シア ${us} m/s、地表 qv ${opts.qv0 * 1000} g/kg。`,
    duration: 7200,
    config: {
      grid, base, moist: true, rain: true, smagorinsky: true, nuConst: 0, cs: 0.18, prandtl: 1 / 3,
      spongeZ: 15000, spongeRate: 1 / 300, dtMax: 6, cfl: 0.8,
    },
    initial: a,
  };
}
