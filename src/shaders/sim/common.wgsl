// =============================================================================
// 雲解像モデル 共通定義
// 格子：Arakawa C。index = i + nx*(j + ny*k)。水平は周期境界。
// state バッファ：10 変数を連続配置 [U, V, W, TH, QV, QC, QI, QR, QS, QG] × n
//   U[i] は x 面 i-1/2、V[j] は y 面 j-1/2、W[k] は下面 k-1/2（W[k=0]=0）
// =============================================================================

const F_U: u32 = 0u;
const F_V: u32 = 1u;
const F_W: u32 = 2u;
const F_TH: u32 = 3u;
const F_QV: u32 = 4u;
const F_QC: u32 = 5u;   // 雲水（0℃以下でも過冷却水として残る。−40℃で凍る）
const F_QI: u32 = 6u;   // 雲氷（氷晶。ゆっくり落ちる）
const F_QR: u32 = 7u;   // 雨（過冷却の雨も含む）
const F_QS: u32 = 8u;   // 雪
const F_QG: u32 = 9u;   // あられ・ひょう（速く落ちる）
const NFIELDS: u32 = 10u;

// diag バッファ（n = 全セル数, plane = nx*ny）
//   [0, n)                 渦粘性 [m²/s]
//   [n, n+16plane)         地表の面の量（dSurf(slot)）：
//     0 地表の積算降水量 [kg/m²]   1 負値の切り詰めで増えた水量の累計 [kg/m²]
//     2 地表面温度 Ts [K]          3 地表の顕熱フラックス H [W/m²]   4 潜熱フラックス LE [W/m²]
//     5 地表の日射 [W/m²]          6 柱に凝結物があるか（1/0。微物理が書く）
//     7 上を水平 ±9 列に広げたもの（ステップの最初に作る。0 の列では凝結物の移流を省く）
//     8 地表からの蒸発の累計 [kg/m²]   9 大規模強制（沈降・qv の強制・ナッジング）で入った水の累計 [kg/m²]
//   [n+16plane, +6n)       正値保存リミッタの係数（qv, qc, qi, qr, qs, qg）。微物理の後は落下速度の置き場
//   [7n+16plane, +n)       三重対角ソルバの作業領域（前進消去の係数 c'）
const D_NU: u32 = 0u;

const GRAV: f32 = 9.81;
const CP: f32 = 1004.5;
const RD: f32 = 287.04;
const LV: f32 = 2.5e6;
const EPSV: f32 = 0.62198;

const FLAG_MOIST: u32 = 1u;
const FLAG_SMAG: u32 = 2u;
const FLAG_RAIN: u32 = 4u;
const FLAG_SURF_ENERGY: u32 = 8u; // 地表エネルギー収支で顕熱・潜熱を求める（0 なら固定フラックス）
const FLAG_PD: u32 = 16u;         // 水物質の正値保存リミッタ
const FLAG_ICE: u32 = 32u;        // 氷相（雲氷・雪）を温度で分ける
const FLAG_LW: u32 = 64u;         // 雲の長波放射（雲頂の冷却・雲底の加熱）
const FLAG_DYCOMS: u32 = 128u;    // DYCOMS-II の放射式の第3項（逆転層より上の沈降に伴う項）
const FLAG_SUBS: u32 = 512u;      // 大規模沈降がある（凝結物にも沈降の移流をかける）
const FLAG_RAD: u32 = 256u;       // 日射の計算（csRadiation：地表の日射と雲の日射加熱）をこのステップで行った

struct Params {
  nx: u32, ny: u32, nz: u32, n: u32,
  dx: f32, dy: f32, dt: f32, dtFull: f32,   // dt = この RK 段の刻み、dtFull = 1ステップの刻み
  stage: u32, flags: u32, nuConst: f32, cs: f32,
  spongeZ: f32, spongeRate: f32, ztop: f32, prandtl: f32,
  shf: f32, lhf: f32, fcor: f32, ustar: f32,          // 固定の地表フラックス [W/m²]、コリオリ係数 [1/s]、固定 u*（0 ならバルク式）
  frameU: f32, frameV: f32, cosZen: f32, swClear: f32,  // 計算座標系の移動速度 [m/s]、太陽天頂角の余弦、晴天時の地表日射 [W/m²]
  albedo: f32, soilWet: f32, z0: f32, nDrop: f32,       // 地表アルベド、土壌の湿り（蒸発効率 β）、粗度 [m]、雲粒数濃度 [1/m³]
  tDeep: f32, tauSoil: f32, cSoil: f32, lai: f32,       // 深層の地温 [K]、復元時間 [s]、地表層の熱容量 [J/m²/K]、葉面積指数
  sunX: f32, sunY: f32, rsMin: f32, pad0: f32,          // 太陽方向の単位ベクトルの水平成分（x 東, y 北）、最小気孔抵抗 [s/m]
};

struct Col {
  zc: f32, zf: f32, dzc: f32, dzw: f32,
  rho0: f32, rhoF: f32, th0: f32, qv0: f32,
  pi0: f32, p0: f32, u0: f32, v0: f32,
  ug: f32, vg: f32, wsub: f32, thLs: f32,       // 地衡風、大規模沈降 [m/s]、θ の大規模・放射強制 [K/s]
  qvLs: f32, nudge: f32, pad0: f32, pad1: f32,  // qv の大規模強制 [1/s]、基本場へのナッジング率 [1/s]
  pad2: f32, pad3: f32, pad4: f32, pad5: f32,
  // 鉛直移流の補間の重み（不等間隔格子用。grid.ts の verticalAdvectionWeights）。6 点 m3..p2 に掛ける：
  //  sA = 速度が正のときの m3..p0、sB = 負のときの m3..p0、sC = (正の p1, p2, 負の p1, p2)
  //  s* はスカラー（と u, v）の下面 k、w* は w のセル中心 k
  sA: vec4f, sB: vec4f, sC: vec4f,
  wA: vec4f, wB: vec4f, wC: vec4f,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> col: array<Col>;
@group(0) @binding(2) var<storage, read> sN: array<f32>;     // ステップ開始時の状態
@group(0) @binding(3) var<storage, read> sC: array<f32>;     // 現在の RK 段の入力状態
@group(0) @binding(4) var<storage, read_write> sO: array<f32>; // 出力状態
@group(0) @binding(5) var<storage, read_write> cplx: array<vec2f>; // 圧力用の複素バッファ
@group(0) @binding(6) var<storage, read_write> diag: array<f32>;
@group(0) @binding(7) var<storage, read_write> stats: array<f32>;

// 水平の格子数は FFT の都合で常に 2 の冪なので、周期境界はビット演算で（整数の剰余は GPU では遅い）。
// 負の添字も 2 の補数なので u32 にしてマスクすれば正しく折り返る。ny = 1 ならマスク 0 で常に 0。
fn wrapX(i: i32) -> u32 { return u32(i) & (P.nx - 1u); }
fn wrapY(j: i32) -> u32 { return u32(j) & (P.ny - 1u); }
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
// 氷面の飽和水蒸気圧（Magnus 型、Alduchov & Eskridge 1996）
fn esatIce(T: f32) -> f32 {
  let tc = T - 273.15;
  return 611.2 * exp(22.46 * tc / (tc + 272.62));
}
fn qvsatIce(T: f32, p: f32) -> f32 {
  let es = min(esatIce(T), p * 0.5);
  return EPSV * es / (p - es);
}

const LS: f32 = 2.834e6; // 昇華熱
const LF: f32 = 3.34e5;  // 融解熱（= LS − LV）
const T_0: f32 = 273.16; // 融点
const T_HOM: f32 = 233.15; // 均質凍結（−40℃）

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

// 鉛直の補間（不等間隔の重み）。v0 = (m3, m2, m1, p0)、v1 = (p1, p2)
fn zInterpS(k: i32, v0: vec4f, v1: vec2f, vel: f32) -> f32 {
  let cl = col[k];
  if (vel >= 0.0) { return dot(cl.sA, v0) + dot(cl.sC.xy, v1); }
  return dot(cl.sB, v0) + dot(cl.sC.zw, v1);
}
fn zInterpW(kc: i32, v0: vec4f, v1: vec2f, vel: f32) -> f32 {
  let cl = col[kc];
  if (vel >= 0.0) { return dot(cl.wA, v0) + dot(cl.wC.xy, v1); }
  return dot(cl.wB, v0) + dot(cl.wC.zw, v1);
}

fn spongeCoef(z: f32) -> f32 {
  if (z <= P.spongeZ || P.spongeRate <= 0.0) { return 0.0; }
  let s = sin(0.5 * 3.14159265 * (z - P.spongeZ) / (P.ztop - P.spongeZ));
  return P.spongeRate * s * s;
}

fn plane() -> u32 { return P.nx * P.ny; }
fn dSurf(slot: u32, colId: u32) -> u32 { return P.n + slot * plane() + colId; } // slot は上の diag の説明を参照
fn dPd(f: u32, id: u32) -> u32 { return P.n + 16u * plane() + (f - F_QV) * P.n + id; }
fn dTri(id: u32) -> u32 { return 7u * P.n + 16u * plane() + id; }
