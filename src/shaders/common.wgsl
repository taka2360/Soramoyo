// 全パス共通のフレーム定数。レイアウトは src/render/frame.ts と一致させること。
// 座標系：観測地点の ENU（x=東, y=北, z=上）。地球中心が原点で、観測地点は +z 軸上。長さは km。
struct Frame {
  fwd: vec3f, tanHalfFov: f32,
  right: vec3f, aspect: f32,
  up: vec3f, exposureEV: f32,        // 露出補正 [EV]
  camPos: vec3f, autoExposure: f32,  // カメラ位置（地球中心基準, km）
  sunDir: vec3f, sunAngRadius: f32,  // 太陽の真の方向（散乱計算用）
  sunDirApp: vec3f, dtReal: f32,     // 太陽の見かけの方向（大気差込み）
  sunIllum: vec3f, gridOn: f32,      // 大気上端の太陽照度（線形 sRGB, lx）
  whitePoint: vec3f, time: f32,      // 白色点（太陽光を白とみなす色順応）
  resolution: vec2f, groundAlbedo: f32, refractionScale: f32,
};

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,   // NDC [-1,1]
};

// 画面全体を覆う三角形1枚
@vertex
fn vsFullscreen(@builtin(vertex_index) i: u32) -> VsOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  var o: VsOut;
  o.pos = vec4f(p, 0.0, 1.0);
  o.uv = p;
  return o;
}

fn viewDir(f: Frame, ndc: vec2f) -> vec3f {
  let x = ndc.x * f.tanHalfFov * f.aspect;
  let y = ndc.y * f.tanHalfFov;
  return normalize(f.fwd + x * f.right + y * f.up);
}

const PI: f32 = 3.14159265358979;

fn luminance(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
