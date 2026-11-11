// 雲ボリュームの準備：
//  csToVolume : 計算モデルの状態 → 描画用 3D テクスチャ（rgba16float）
//               r = 雲水量 [g/m³]、g = 雨水量 [g/m³]、b = 渦粘性 [m²/s]、a = 鉛直流 [m/s]
//  csShadow   : 各点から太陽方向への光学的厚さ（r）と真上方向の光学的厚さ（g）
//  csNoise    : 細部合成用のタイル可能なノイズ（Perlin–Worley, 128³）
// （cloudcommon.wgsl の後ろに連結）

struct Dims { nx: u32, ny: u32, nz: u32, n: u32, dx: f32, dy: f32, pad0: f32, pad1: f32 };
struct Col {
  zc: f32, zf: f32, dzc: f32, dzw: f32,
  rho0: f32, rhoF: f32, th0: f32, qv0: f32,
  pi0: f32, p0: f32, u0: f32, v0: f32,
  pad0: f32, pad1: f32, pad2: f32, pad3: f32,
};

@group(0) @binding(0) var<uniform> D: Dims;
@group(0) @binding(1) var<storage, read> st: array<f32>;
@group(0) @binding(2) var<storage, read> col: array<Col>;
@group(0) @binding(3) var<storage, read> diag: array<f32>;
@group(0) @binding(4) var volOut: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn csToVolume(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= D.nx || g.y >= D.ny || g.z >= D.nz) { return; }
  let id = g.x + D.nx * (g.y + D.ny * g.z);
  let rho = col[g.z].rho0;
  let qc = max(st[5u * D.n + id], 0.0);
  let qr = max(st[6u * D.n + id], 0.0);
  let w0 = st[2u * D.n + id];
  var w1 = 0.0;
  if (g.z + 1u < D.nz) { w1 = st[2u * D.n + id + D.nx * D.ny]; }
  let nu = diag[id];
  textureStore(volOut, g, vec4f(qc * rho * 1000.0, qr * rho * 1000.0, nu, 0.5 * (w0 + w1)));
}
