// =============================================================================
// 雲解像モデル 共通定義
// 格子：Arakawa C。index = i + nx*(j + ny*k)。水平は周期境界。
// state バッファ：7 変数を連続配置 [U, V, W, TH, QV, QC, QR] × n
//   U[i] は x 面 i-1/2、V[j] は y 面 j-1/2、W[k] は下面 k-1/2（W[k=0]=0）
// =============================================================================

const F_U: u32 = 0u;
const F_V: u32 = 1u;
const F_W: u32 = 2u;
const F_TH: u32 = 3u;
const F_QV: u32 = 4u;
const F_QC: u32 = 5u;
const F_QR: u32 = 6u;
const NFIELDS: u32 = 7u;

// diag バッファ
const D_NU: u32 = 0u;     // 渦粘性 [m²/s]（セル中心）
const D_PRECIP: u32 = 1u; // 地表の積算降水量 [kg/m² = mm]（k=0 面のみ使用）

const GRAV: f32 = 9.81;
const CP: f32 = 1004.5;
const RD: f32 = 287.04;
const LV: f32 = 2.5e6;
const EPSV: f32 = 0.62198;

const FLAG_MOIST: u32 = 1u;
const FLAG_SMAG: u32 = 2u;
const FLAG_RAIN: u32 = 4u;

struct Params {
  nx: u32, ny: u32, nz: u32, n: u32,
  dx: f32, dy: f32, dt: f32, dtFull: f32,   // dt = この RK 段の刻み、dtFull = 1ステップの刻み
  stage: u32, flags: u32, nuConst: f32, cs: f32,
  spongeZ: f32, spongeRate: f32, ztop: f32, prandtl: f32,
};

struct Col {
  zc: f32, zf: f32, dzc: f32, dzw: f32,
  rho0: f32, rhoF: f32, th0: f32, qv0: f32,
  pi0: f32, p0: f32, u0: f32, v0: f32,
  pad0: f32, pad1: f32, pad2: f32, pad3: f32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> col: array<Col>;
@group(0) @binding(2) var<storage, read> sN: array<f32>;     // ステップ開始時の状態
@group(0) @binding(3) var<storage, read> sC: array<f32>;     // 現在の RK 段の入力状態
@group(0) @binding(4) var<storage, read_write> sO: array<f32>; // 出力状態
@group(0) @binding(5) var<storage, read_write> cplx: array<vec2f>; // 圧力用の複素バッファ
@group(0) @binding(6) var<storage, read_write> diag: array<f32>;
@group(0) @binding(7) var<storage, read_write> stats: array<f32>;

fn wrapX(i: i32) -> u32 { let n = i32(P.nx); return u32(((i % n) + n) % n); }
fn wrapY(j: i32) -> u32 { let n = i32(P.ny); return u32(((j % n) + n) % n); }
fn clampK(k: i32) -> u32 { return u32(clamp(k, 0, i32(P.nz) - 1)); }

fn idx(i: i32, j: i32, k: i32) -> u32 {
  return wrapX(i) + P.nx * (wrapY(j) + P.ny * clampK(k));
}

// 現在の段の状態（sC）を読む
fn c(f: u32, i: i32, j: i32, k: i32) -> f32 { return sC[f * P.n + idx(i, j, k)]; }
// W は k=0 と k=nz で 0
fn cw(i: i32, j: i32, k: i32) -> f32 {
  if (k <= 0 || k >= i32(P.nz)) { return 0.0; }
  return sC[F_W * P.n + idx(i, j, k)];
}

// ---------------------------------------------------------------- 熱力学
fn esatLiq(T: f32) -> f32 {
  let tc = T - 273.15;
  return 611.2 * exp(17.67 * tc / (tc + 243.5));
}
fn qvsatLiq(T: f32, p: f32) -> f32 {
  let es = min(esatLiq(T), p * 0.5);
  return EPSV * es / (p - es);
}

// ---------------------------------------------------------------- 移流スキーム
// 面 (a-1/2) での値。m3..p2 は a-3 .. a+2 の値、vel はその面の速度
fn interp5(m3: f32, m2: f32, m1: f32, p0: f32, p1: f32, p2: f32, vel: f32) -> f32 {
  let c6 = (37.0 * (p0 + m1) - 8.0 * (p1 + m2) + (p2 + m3)) / 60.0;
  let d = (10.0 * (p0 - m1) - 5.0 * (p1 - m2) + (p2 - m3)) / 60.0;
  return c6 - sign(vel) * d;
}
fn interp3(m2: f32, m1: f32, p0: f32, p1: f32, vel: f32) -> f32 {
  let c4 = (7.0 * (p0 + m1) - (p1 + m2)) / 12.0;
  let d = (3.0 * (p0 - m1) - (p1 - m2)) / 12.0;
  return c4 - sign(vel) * d;
}

fn spongeCoef(z: f32) -> f32 {
  if (z <= P.spongeZ || P.spongeRate <= 0.0) { return 0.0; }
  let s = sin(0.5 * 3.14159265 * (z - P.spongeZ) / (P.ztop - P.spongeZ));
  return P.spongeRate * s * s;
}
