// Bruneton 2017 の描画用関数（functions.wgsl の後ろに連結）。空・地面・雲の空気遠近で共有。

// ---------------------------------------------------------------- Bruneton の描画関数
struct SkyOut { radiance: vec3f, transmittance: vec3f };

fn skyScattering(r: f32, mu: f32, muS: f32, nu: f32, hitGround: bool) -> vec3f {
  let ray = sampleScattering(rayleighTex, r, mu, muS, nu, hitGround);
  let mie = sampleScattering(mieTex, r, mu, muS, nu, hitGround);
  return ray * rayleighPhase(nu) + mie * miePhase(A.mie_g, nu);
}

fn getSkyRadiance(camIn: vec3f, view: vec3f, sun: vec3f) -> SkyOut {
  var o: SkyOut;
  var cam = camIn;
  var r = length(cam);
  var rmu = dot(cam, view);
  let distToTop = -rmu - sqrt(rmu * rmu - r * r + A.top_radius * A.top_radius);
  if (distToTop > 0.0) {
    cam = cam + view * distToTop;
    r = A.top_radius;
    rmu += distToTop;
  } else if (r > A.top_radius) {
    o.radiance = vec3f(0.0);
    o.transmittance = vec3f(1.0);
    return o;
  }
  let mu = rmu / r;
  let muS = dot(cam, sun) / r;
  let nu = dot(view, sun);
  let hitGround = rayIntersectsGround(r, mu);
  o.transmittance = select(getTransmittanceToTop(r, mu), vec3f(0.0), hitGround);
  o.radiance = skyScattering(r, mu, muS, nu, hitGround);
  return o;
}

fn getSkyRadianceToPoint(camIn: vec3f, point: vec3f, sun: vec3f) -> SkyOut {
  var o: SkyOut;
  let view = normalize(point - camIn);
  var cam = camIn;
  var r = length(cam);
  var rmu = dot(cam, view);
  let distToTop = -rmu - sqrt(rmu * rmu - r * r + A.top_radius * A.top_radius);
  if (distToTop > 0.0) {
    cam = cam + view * distToTop;
    r = A.top_radius;
    rmu += distToTop;
  }
  let mu = rmu / r;
  let muS = dot(cam, sun) / r;
  let nu = dot(view, sun);
  let d = length(point - cam);
  let hitGround = rayIntersectsGround(r, mu);
  o.transmittance = getTransmittance(r, mu, d, hitGround);

  let ray0 = sampleScattering(rayleighTex, r, mu, muS, nu, hitGround);
  let mie0 = sampleScattering(mieTex, r, mu, muS, nu, hitGround);
  let rP = clampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
  let muP = (r * mu + d) / rP;
  let muSP = (r * muS + d * nu) / rP;
  let ray1 = sampleScattering(rayleighTex, rP, muP, muSP, nu, hitGround);
  let mie1 = sampleScattering(mieTex, rP, muP, muSP, nu, hitGround);
  let ray = max(ray0 - o.transmittance * ray1, vec3f(0.0));
  var mie = max(mie0 - o.transmittance * mie1, vec3f(0.0));
  mie *= smoothstep(0.0, 0.01, muS); // Bruneton と同じ：地平線下の太陽での Mie の誤差を抑える
  o.radiance = ray * rayleighPhase(nu) + mie * miePhase(A.mie_g, nu);
  return o;
}

