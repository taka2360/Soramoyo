// 雲の描画で共有する定義（cloudvol.wgsl / clouds.wgsl）

struct CloudU {
  origin: vec2f, size: vec2f,          // 計算領域の原点（世界座標 m）と大きさ
  ztop: f32, slabBot: f32, slabTop: f32, nz: f32,
  nDrop: f32, detail: f32, time: f32, maxDist: f32,  // 雲粒数濃度 [1/m³]、細部合成の強さ
  sunDir: vec3f, frame: f32,
  noisePeriod: f32, dropDiamUm: f32, groundAlbedo: f32, enabled: f32,
  histAlpha: f32, cellSize: f32, pad1: f32, pad2: f32,   // 時間方向の蓄積の混合率、水平格子幅 [m]
};

// 雲水量 [kg/m³] → 消散係数 [1/m]。有効半径は雲粒数濃度から：
//   r_v = (3 LWC / (4π ρw N))^(1/3)、r_eff ≈ 1.1 r_v、σ = 1.5 LWC / (ρw r_eff)
//   → σ = 0.02198 · LWC^(2/3) · N^(1/3)
fn sigmaCloud(lwcKg: f32, n: f32) -> f32 {
  if (lwcKg <= 1e-9) { return 0.0; }
  return 0.02198 * pow(lwcKg, 2.0 / 3.0) * pow(n, 1.0 / 3.0);
}
// 雨：有効半径 0.5 mm と仮定
fn sigmaRain(rwcKg: f32) -> f32 { return 3.0 * max(rwcKg, 0.0); }

// 高さ z [m] → 3D テクスチャの w 座標（不等間隔の鉛直格子に対応する対応表を引く）
fn zToW(zmap: texture_2d<f32>, z: f32, ztop: f32, nz: f32) -> f32 {
  let x = clamp(z / ztop, 0.0, 1.0) * 1023.0;
  let i = u32(floor(x));
  let f = x - floor(x);
  let a = textureLoad(zmap, vec2u(i, 0u), 0).r;
  let b = textureLoad(zmap, vec2u(min(i + 1u, 1023u), 0u), 0).r;
  return (mix(a, b, f) + 0.5) / nz;
}
