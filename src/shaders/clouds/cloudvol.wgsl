// 雲ボリュームの準備：
//  csToVolume : 計算モデルの状態 → 描画用 3D テクスチャ（rgba16float）
//               r = 雲水 [g/m³]、g = 雲氷 [g/m³]、b = 降水（雨換算。雪は 5 倍、あられは等倍）[g/m³]、a = 渦粘性 [m²/s]
//  csShadow   : 各点から太陽方向への光学的厚さ（r）と真上方向の光学的厚さ（g）
//  csNoise    : 細部合成用のタイル可能なノイズ（Perlin–Worley, 128³）
// （cloudcommon.wgsl の後ろに連結）

struct Dims { nx: u32, ny: u32, nz: u32, n: u32, dx: f32, dy: f32, ice: f32, pad1: f32 };
struct Col {
  zc: f32, zf: f32, dzc: f32, dzw: f32,
  rho0: f32, rhoF: f32, th0: f32, qv0: f32,
  pi0: f32, p0: f32, u0: f32, v0: f32,
  ug: f32, vg: f32, wsub: f32, thLs: f32,       // 地衡風、大規模沈降 [m/s]、θ の大規模・放射強制 [K/s]
  qvLs: f32, nudge: f32, pad0: f32, pad1: f32,  // qv の大規模強制 [1/s]、基本場へのナッジング率 [1/s]
  pad2: f32, pad3: f32, pad4: f32, pad5: f32,
  adv: array<vec4f, 6>,                          // 鉛直移流の補間の重み（計算モデル用。ここでは使わない）
};

@group(0) @binding(0) var<uniform> D: Dims;
@group(0) @binding(1) var<storage, read> st: array<f32>;
@group(0) @binding(2) var<storage, read> col: array<Col>;
@group(0) @binding(3) var<storage, read> diag: array<f32>;
@group(0) @binding(4) var volOut: texture_storage_3d<rgba16float, write>;
// セル中心の風（計算領域に対する相対値）[m/s]。細部合成を流れに乗せるため
@group(0) @binding(5) var velOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn csToVolume(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= D.nx || g.y >= D.ny || g.z >= D.nz) { return; }
  let id = g.x + D.nx * (g.y + D.ny * g.z);
  let rho = col[g.z].rho0;
  // 状態の並び：U V W TH QV QC QI QR QS QG（sim/common.wgsl）
  let qc = max(st[5u * D.n + id], 0.0);
  let qi = max(st[6u * D.n + id], 0.0);
  let qr = max(st[7u * D.n + id], 0.0);
  let qs = max(st[8u * D.n + id], 0.0);
  let qg = max(st[9u * D.n + id], 0.0);  // あられ：消散は雨とほぼ同じ（N0 = 4×10⁵, 密度 500 で σ ≈ 3.2 ρq）
  let nu = diag[id];
  let k = rho * 1000.0;
  // セル中心の風（U は x 面 i−1/2、V は y 面 j−1/2、W は下面 k−1/2 にある。水平は周期境界）
  let ip = ((g.x + 1u) % D.nx) + D.nx * (g.y + D.ny * g.z);
  let jp = g.x + D.nx * (((g.y + 1u) % D.ny) + D.ny * g.z);
  let u = 0.5 * (st[id] + st[ip]);
  let v = 0.5 * (st[D.n + id] + st[D.n + jp]);
  var wTop = 0.0;
  if (g.z + 1u < D.nz) { wTop = st[2u * D.n + id + D.nx * D.ny]; }
  let w = 0.5 * (st[2u * D.n + id] + wTop);
  textureStore(velOut, g, vec4f(u, v, w, 0.0));
  textureStore(volOut, g, vec4f(qc * k, qi * k, (qr + 5.0 * qs + qg) * k, nu));
}
