// 各ボクセルから太陽方向・真上方向への光学的厚さ（周期境界でタイル状に並ぶ雲と整合）
// （cloudcommon.wgsl の後ろに連結）

struct Dims { nx: u32, ny: u32, nz: u32, n: u32, dx: f32, dy: f32, pad0: f32, pad1: f32 };
struct Col {
  zc: f32, zf: f32, dzc: f32, dzw: f32,
  rho0: f32, rhoF: f32, th0: f32, qv0: f32,
  pi0: f32, p0: f32, u0: f32, v0: f32,
  ug: f32, vg: f32, wsub: f32, thLs: f32,       // 地衡風、大規模沈降 [m/s]、θ の大規模・放射強制 [K/s]
  qvLs: f32, nudge: f32, pad0: f32, pad1: f32,  // qv の大規模強制 [1/s]、基本場へのナッジング率 [1/s]
  pad2: f32, pad3: f32, pad4: f32, pad5: f32,
};

@group(0) @binding(0) var<uniform> D: Dims;
@group(0) @binding(1) var<uniform> C: CloudU;
@group(0) @binding(2) var<storage, read> col: array<Col>;
@group(0) @binding(3) var vol: texture_3d<f32>;
@group(0) @binding(4) var smpRep: sampler;
@group(0) @binding(5) var zmap: texture_2d<f32>;
@group(0) @binding(6) var shadowOut: texture_storage_3d<rgba16float, write>;

fn sigmaAt(p: vec3f) -> f32 {
  let uvw = vec3f(p.x / C.size.x, p.y / C.size.y, zToW(zmap, p.z, C.ztop, C.nz));
  let v = sampleVolSmoothZ(vol, smpRep, uvw);
  return sigmaVolume(v, C.nDrop);
}

fn opticalDepth(p0: vec3f, dir: vec3f) -> f32 {
  var od = 0.0;
  var t = 0.0;
  var ds = 0.5 * min(D.dx, 100.0);
  for (var i = 0; i < 96; i++) {
    let p = p0 + dir * (t + 0.5 * ds);
    if (p.z > C.slabTop || p.z < 0.0) { break; }
    od += sigmaAt(p) * ds;
    t += ds;
    ds = min(ds * 1.08, 4.0 * D.dx);
    if (t > 30000.0 || od > 60.0) { break; }
  }
  return od;
}

@compute @workgroup_size(8, 8, 1)
fn csShadow(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= D.nx || g.y >= D.ny || g.z >= D.nz) { return; }
  let p = vec3f((f32(g.x) + 0.5) * D.dx, (f32(g.y) + 0.5) * D.dy, col[g.z].zc);
  var sun = C.sunDir;
  sun.z = max(sun.z, 0.03); // 太陽が地平線近く・下のときは浅い角度で打ち切る
  sun = normalize(sun);
  var odSun = 0.0;
  var odUp = 0.0;
  if (p.z <= C.slabTop) {
    odSun = opticalDepth(p, sun);
    odUp = opticalDepth(p, vec3f(0.0, 0.0, 1.0));
  }
  textureStore(shadowOut, g, vec4f(odSun, odUp, 0.0, 0.0));
}
