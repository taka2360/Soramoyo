import { type Grid, type Sounding, type LargeScaleForcing, buildBaseState, makeGrid, uniformLevels, stretchedLevels } from './grid';
import { type ModelConfig, NFIELDS, F } from './model';
import { CP, G, LV, qvsat, saturationAdjust, exner, P00, KAPPA } from './thermo';

export interface SimCase {
  name: string;
  description: string;
  /** 日射と結合した日変化のケースか（開始時刻を現地 startHour に合わせる） */
  diurnal?: { startHour: number };
  /** 検証用の参照値（表示用） */
  reference?: string;
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
      grid, base, moist: false, rain: false, smagorinsky: false, nuConst: 75, cs: 0.18, prandtl: 1, shf: 0, lhf: 0,
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
      grid, base, moist: true, rain: false, smagorinsky: true, nuConst: 0, cs: 0.18, prandtl: 1 / 3, shf: 0, lhf: 0,
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
      grid, base, moist: true, rain: true, smagorinsky: true, nuConst: 0, cs: 0.18, prandtl: 1 / 3, shf: 0, lhf: 0,
      spongeZ: 15000, spongeRate: 1 / 300, dtMax: 6, cfl: 0.8, frame: [us / 2, 0], ice: true, longwave: true,
    },
    initial: a,
  };
}

// ---------------------------------------------------------------------------
/**
 * 陸上の晴天積雲：ARM SGP（米国中部, 1997/6/21）の初期サウンディング（Brown et al. 2002）。
 * 地表加熱（顕熱・潜熱フラックス）で対流混合層が育ち、持ち上げ凝結高度に達すると積雲が湧く。
 * 段階4では地表フラックスは一定値（正午付近の値）。段階5で日射・地表エネルギー収支と結合する。
 */
export function armCumulus(opts: { nx: number; dx: number; dzMin: number; shf: number; lhf: number }): SimCase {
  const zs = [0, 50, 350, 650, 700, 1300, 2500, 5500];
  const ths = [299.0, 301.5, 302.5, 303.53, 303.7, 307.13, 314.0, 343.2];
  const qts = [15.2, 15.17, 14.98, 14.8, 14.7, 13.5, 3.0, 3.0].map((q) => q / 1000);
  const lerp = (arr: number[]) => (z: number) => {
    if (z <= zs[0]) return arr[0];
    for (let i = 1; i < zs.length; i++) {
      if (z <= zs[i]) return arr[i - 1] + ((arr[i] - arr[i - 1]) * (z - zs[i - 1])) / (zs[i] - zs[i - 1]);
    }
    const n = zs.length - 1;
    return arr[n] + ((arr[n] - arr[n - 1]) * (z - zs[n])) / (zs[n] - zs[n - 1]);
  };
  const zf = stretchedLevels(opts.dzMin, 3000, 1.06, 250, 6000);
  const grid = makeGrid({ nx: opts.nx, ny: opts.nx, dx: opts.dx, dy: opts.dx, zf });
  // 平均風 u=10 m/s で動く座標系で計算する（領域ごと東へ流れる）
  const snd: Sounding = { surfacePressure: 97000, theta: lerp(ths), qv: lerp(qts), u: () => 0, v: () => 0 };
  const base = buildBaseState(grid, snd);
  const a = alloc(grid);
  fillBase(grid, base, a);
  const n = grid.nx * grid.ny * grid.nz;
  const plane = grid.nx * grid.ny;
  const r = rng(777);
  for (let k = 0; k < grid.nz; k++) {
    if (grid.zc[k] > 500) break;
    for (let p = 0; p < plane; p++) {
      a[F.TH * n + p + plane * k] += (r() - 0.5) * 0.2;
      a[F.QV * n + p + plane * k] += (r() - 0.5) * 2e-5;
    }
  }
  return {
    name: '陸上の晴天積雲（ARM SGP 型）',
    description: `地表加熱で混合層が育ち、積雲が湧く。顕熱 ${opts.shf} W/m²・潜熱 ${opts.lhf} W/m²（一定）。平均風 10 m/s。`,
    duration: 4 * 3600,
    config: {
      grid, base, moist: true, rain: true, smagorinsky: true, nuConst: 0, cs: 0.18, prandtl: 1 / 3,
      shf: opts.shf, lhf: opts.lhf, spongeZ: 4500, spongeRate: 1 / 300, dtMax: 3, cfl: 0.7,
      frame: [10, 0], z0: 0.1,
    },
    initial: a,
  };
}

// ---------------------------------------------------------------------------
function profile(zs: number[], vs: number[]) {
  return (z: number) => {
    if (z <= zs[0]) return vs[0];
    for (let i = 1; i < zs.length; i++) {
      if (z <= zs[i]) return vs[i - 1] + ((vs[i] - vs[i - 1]) * (z - zs[i - 1])) / (zs[i] - zs[i - 1]);
    }
    const n = zs.length - 1;
    return vs[n] + ((vs[n] - vs[n - 1]) * (z - zs[n])) / (zs[n] - zs[n - 1]);
  };
}

function perturb(g: Grid, a: Float32Array, zMax: number, thAmp: number, qAmp: number, seed: number) {
  const n = g.nx * g.ny * g.nz;
  const plane = g.nx * g.ny;
  const r = rng(seed);
  for (let k = 0; k < g.nz; k++) {
    if (g.zc[k] > zMax) break;
    for (let p = 0; p < plane; p++) {
      a[F.TH * n + p + plane * k] += (r() - 0.5) * 2 * thAmp;
      a[F.QV * n + p + plane * k] += (r() - 0.5) * 2 * qAmp;
    }
  }
}

/**
 * BOMEX（Siebesma et al. 2003）：貿易風帯の海上の浅い積雲。標準的な LES 相互比較ケース。
 * 地表フラックス固定、大規模沈降・放射冷却・乾燥移流・地衡風で準定常状態になる。
 * 参照値（3〜6時間の平均、LES 群の範囲）：雲底 ≈ 500 m、雲頂 ≈ 1.5〜1.8 km、
 * 層ごとの雲量の最大 ≈ 5〜8%、雲水量の柱積算（LWP）≈ 5〜12 g/m²。
 */
export function bomex(opts: { nx: number; dx: number; dz: number }): SimCase {
  const zf = uniformLevels(Math.round(3000 / opts.dz), opts.dz);
  const grid = makeGrid({ nx: opts.nx, ny: opts.nx, dx: opts.dx, dy: opts.dx, zf });
  const thl = profile([0, 520, 1480, 2000, 3000], [298.7, 298.7, 302.4, 308.2, 311.85]);
  const qt = profile([0, 520, 1480, 2000, 3000], [17.0e-3, 16.3e-3, 10.7e-3, 4.2e-3, 3.0e-3]);
  const uG = (z: number) => (z <= 700 ? -8.75 : -8.75 + 1.8e-3 * (z - 700));
  const frame: [number, number] = [-8, 0];
  const snd: Sounding = { surfacePressure: 101500, theta: thl, qv: qt, u: (z) => uG(z) - frame[0], v: () => 0 };
  const forcing: LargeScaleForcing = {
    ug: (z) => -10 + 1.8e-3 * z,
    vg: () => 0,
    wsub: (z) => (z <= 1500 ? (-0.0065 * z) / 1500 : z <= 2100 ? -0.0065 + (0.0065 * (z - 1500)) / 600 : 0),
    thLs: (z) => (-2 / 86400) * (z <= 1500 ? 1 : z <= 2500 ? 1 - (z - 1500) / 1000 : 0),
    qvLs: (z) => -1.2e-8 * (z <= 300 ? 1 : z <= 500 ? 1 - (z - 300) / 200 : 0),
    nudge: () => 0,
  };
  const base = buildBaseState(grid, snd);
  const a = alloc(grid);
  fillBase(grid, base, a);
  perturb(grid, a, 1600, 0.1, 2.5e-5, 4242);
  const rho = base.rho0[0];
  return {
    name: 'BOMEX 貿易風積雲（検証）',
    description: '海上の浅い積雲。地表フラックス・大規模強制は固定（Siebesma et al. 2003）。',
    reference: '参照（3〜6h）：雲底≈0.5km 雲頂≈1.5〜1.8km 最大層雲量≈5〜8% LWP≈5〜12 g/m²',
    duration: 6 * 3600,
    config: {
      grid, base, moist: true, rain: false, smagorinsky: true, nuConst: 0, cs: 0.18, prandtl: 1 / 3,
      shf: rho * 1004.5 * 8e-3, lhf: rho * 2.5e6 * 5.2e-5, forcing, fcor: 0.376e-4, frame,
      ustar: 0.28, z0: 0.0002, spongeZ: 2500, spongeRate: 1 / 300, dtMax: 3, cfl: 0.7,
    },
    initial: a,
  };
}

/** 観測地点の緯度からコリオリ係数 */
export function coriolis(latDeg: number): number {
  return 2 * 7.2921e-5 * Math.sin((latDeg * Math.PI) / 180);
}

/**
 * 日射と結合した陸上の一日（地表エネルギー収支で顕熱・潜熱が決まる）。
 * 夜明け（現地 startHour）に初期サウンディングから始め、日射で地面が温まると混合層が育ち、
 * 条件がそろえば積雲が湧き、夕方に加熱が止まると崩れる。
 */
export function landDay(opts: {
  name: string;
  description: string;
  sounding: 'arm' | 'wk' | 'armDry';
  nx: number; dx: number; dzMin: number; ztop: number;
  soilWet: number; windMs: number; latDeg: number;
}): SimCase {
  let theta: (z: number) => number;
  let qv: (z: number) => number;
  if (opts.sounding === 'wk') {
    const ztr = 12000, thTr = 343, Ttr = 213;
    // 夜明け前の地表付近は放射冷却で少し安定（下層 300m を最大 2K 冷やす）
    const thWk = (z: number) => (z <= ztr ? 300 + (thTr - 300) * Math.pow(z / ztr, 1.25) : thTr * Math.exp((G / (CP * Ttr)) * (z - ztr)));
    theta = (z) => thWk(z) - (z < 300 ? 2 * (1 - z / 300) : 0);
    const rh = (z: number) => (z <= ztr ? 1 - 0.75 * Math.pow(z / ztr, 1.25) : 0.25);
    qv = (z) => {
      const pi = Math.max(0.2, 1 - (G * z) / (CP * 305));
      const p = P00 * Math.pow(pi, 1 / KAPPA);
      return Math.min(0.014, rh(z) * qvsat(thWk(z) * pi, p));
    };
  } else {
    const zs = [0, 50, 350, 650, 700, 1300, 2500, 5500];
    theta = profile(zs, [299.0, 301.5, 302.5, 303.53, 303.7, 307.13, 314.0, 343.2]);
    const q = profile(zs, [15.2, 15.17, 14.98, 14.8, 14.7, 13.5, 3.0, 3.0].map((x) => x / 1000));
    const f = opts.sounding === 'armDry' ? 0.6 : 1;
    qv = (z) => q(z) * f;
  }
  const deep = opts.ztop > 8000;
  const zf = stretchedLevels(opts.dzMin, deep ? 1500 : 3000, 1.05, deep ? 400 : 200, opts.ztop);
  const grid = makeGrid({ nx: opts.nx, ny: opts.nx, dx: opts.dx, dy: opts.dx, zf });
  const frame: [number, number] = [opts.windMs, 0];
  const snd: Sounding = { surfacePressure: 100000, theta, qv, u: () => 0, v: () => 0 };
  const zn = deep ? 9000 : 3000;
  const forcing: LargeScaleForcing = {
    ug: () => opts.windMs, vg: () => 0, wsub: () => 0,
    // 晴天の大気の放射冷却（約 −1.5 K/日）。地表の長波放射は地表エネルギー収支で別に扱う
    thLs: () => -1.5 / 86400,
    qvLs: () => 0,
    // 上空は数時間で元の状態へ引き戻す（周期領域で熱・水が溜まり続けるのを防ぐ）
    nudge: (z) => (z < zn ? 0 : z < zn + 1500 ? (1 / 10800) * ((z - zn) / 1500) : 1 / 10800),
  };
  const base = buildBaseState(grid, snd);
  const a = alloc(grid);
  fillBase(grid, base, a);
  perturb(grid, a, 500, 0.1, 1e-5, 99);
  const t0 = base.th0[0] * base.pi0[0];
  return {
    name: opts.name,
    description: opts.description,
    diurnal: { startHour: 5.5 },
    duration: 24 * 3600,
    config: {
      grid, base, moist: true, rain: true, smagorinsky: true, nuConst: 0, cs: 0.18, prandtl: 1 / 3,
      shf: 0, lhf: 0, forcing, fcor: coriolis(opts.latDeg), frame, z0: 0.1,
      surfaceEnergy: true, ice: true, longwave: true,
      surface: { albedo: 0.2, soilWet: opts.soilWet, tDeep: t0 + 1, cSoil: 2.0e5 },
      spongeZ: opts.ztop * 0.8, spongeRate: 1 / 300, dtMax: 4, cfl: 0.7,
    },
    initial: a,
  };
}

// ---------------------------------------------------------------------------
/** 指定した高さの帯だけ細かくした鉛直格子（逆転層を解像するため） */
export function bandLevels(dzFine: number, zFineBot: number, zFineTop: number, dzCoarse: number, ztop: number): number[] {
  const zf = [0];
  while (zf[zf.length - 1] < ztop - 1e-6) {
    const z = zf[zf.length - 1];
    // 帯の外は dzCoarse、帯の中は dzFine、境目はなめらかに（1.15 倍ずつ）
    const prev = zf.length > 1 ? z - zf[zf.length - 2] : dzCoarse;
    const target = z >= zFineBot - 3 * dzCoarse && z < zFineTop ? dzFine : dzCoarse;
    const dz = target < prev ? Math.max(target, prev / 1.15) : Math.min(target, prev * 1.15);
    zf.push(Math.min(z + dz, ztop));
  }
  return zf;
}

/**
 * DYCOMS-II RF01（Stevens et al. 2005）：カリフォルニア沖の夜の層積雲。
 * 強い逆転層（約 840 m）の下に一様な雲の層ができ、雲頂の長波放射冷却で対流が駆動される。
 * 「雲海」の典型。解像度への要求が厳しく（本家は鉛直 5 m）、粗いと雲頂で乾いた空気を取り込みすぎて雲が薄くなる。
 * 参照（LES 群、2〜4時間）：雲頂 ≈ 850〜870 m、雲底 ≈ 600 m、雲量 ≈ 1、LWP ≈ 20〜60 g/m²（観測 ≈ 40〜60）。
 */
export function dycoms(opts: { nx: number; dx: number; dzFine: number }): SimCase {
  const zi = 840;
  const zf = bandLevels(opts.dzFine, 650, 1000, 25, 1500);
  const grid = makeGrid({ nx: opts.nx, ny: opts.nx, dx: opts.dx, dy: opts.dx, zf });
  const thl = (z: number) => (z <= zi ? 289.0 : 297.5 + Math.cbrt(z - zi));
  const qt = (z: number) => (z <= zi ? 9.0e-3 : 1.5e-3);
  // 初期状態は θl, qt。雲の中は飽和調整して θ, qv, qc に分ける
  const frame: [number, number] = [7, -5.5];
  const snd: Sounding = { surfacePressure: 101780, theta: thl, qv: qt, u: () => 0, v: () => 0 };
  const forcing: LargeScaleForcing = {
    ug: () => 7, vg: () => -5.5,
    wsub: (z) => -3.75e-6 * z,
    thLs: () => 0, qvLs: () => 0, nudge: () => 0,
  };
  const base = buildBaseState(grid, snd);
  const a = alloc(grid);
  fillBase(grid, base, a);
  const n = grid.nx * grid.ny * grid.nz;
  const plane = grid.nx * grid.ny;
  const r = rng(2024);
  for (let k = 0; k < grid.nz; k++) {
    for (let p = 0; p < plane; p++) {
      const id = p + plane * k;
      let th = base.th0[k];
      if (grid.zc[k] < 800) th += (r() - 0.5) * 0.2;
      const [t2, v2, c2] = saturationAdjust(th, qt(grid.zc[k]), 0, base.p0[k], base.pi0[k]);
      a[F.TH * n + id] = t2;
      a[F.QV * n + id] = v2;
      a[F.QC * n + id] = c2;
    }
  }
  const rho = base.rho0[0];
  return {
    name: 'DYCOMS-II RF01 層積雲（検証）',
    description: '夜の海上の層積雲（雲海）。雲頂の放射冷却で対流が起こる。雨なし（Stevens et al. 2005）。',
    reference: '参照（2〜4h）：雲頂≈0.85km 雲底≈0.6km 雲量≈100% LWP≈20〜60 g/m²',
    duration: 4 * 3600,
    config: {
      grid, base, moist: true, rain: false, smagorinsky: true, nuConst: 0, cs: 0.18, prandtl: 1 / 3,
      shf: 15, lhf: 115, forcing, fcor: 7.62e-5, frame, ustar: 0.25, z0: 0.0002,
      spongeZ: 1250, spongeRate: 1 / 300, dtMax: 2, cfl: 0.6,
      longwave: true, dycomsRadiation: true,
    },
    initial: a,
  };
  void rho;
}
