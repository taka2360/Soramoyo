/**
 * 大気の分光パラメータと、分光値 → 線形 sRGB 輝度 の変換。
 *
 * 出典：
 *  - 太陽分光放射照度・オゾン吸収断面積・Rayleigh/Mie 係数：
 *    Bruneton 2017 のデモ (precomputed_atmospheric_scattering/atmosphere/demo/demo.cc) の値
 *    （360〜830nm, 10nm 刻み）。太陽は大気上端の値 [W/m²/nm]。
 *  - CIE 1931 2° 等色関数：Wyman, Sloan, Shirley 2013 の多峰ガウス近似（誤差 ~1%）。
 *    ※ 正式な表ではなく近似式を使っている点に注意。
 */

export const LAMBDA_MIN = 360;
export const LAMBDA_MAX = 830;

const SOLAR_IRRADIANCE = [
  1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887, 1.61253, 1.91198, 2.03474, 2.02042, 2.02212,
  1.93377, 1.95809, 1.91686, 1.8298, 1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533,
  1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482, 1.41018, 1.36775, 1.34188,
  1.31429, 1.28303, 1.26758, 1.2367, 1.2082, 1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992,
];

const OZONE_CROSS_SECTION = [
  1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27, 8.451e-27, 1.582e-26, 2.316e-26,
  3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26, 1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25,
  4.266e-25, 4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25, 2.662e-25, 2.238e-25,
  1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26, 6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26,
  2.451e-26, 2.801e-26, 2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27,
];

const DOBSON_UNIT = 2.687e20; // [1/m²]
const MAX_OZONE_NUMBER_DENSITY = (300 * DOBSON_UNIT) / 15000; // [1/m³]（オゾン全量 300DU）
const RAYLEIGH_K = 1.24062e-6; // [1/m] × λ[µm]^4
const MAX_LUMINOUS_EFFICACY = 683; // [lm/W]

/** XYZ → 線形 sRGB (D65) */
const XYZ_TO_SRGB = [3.2406, -1.5372, -0.4986, -0.9689, 1.8758, 0.0415, 0.0557, -0.204, 1.057];

function table(t: number[], lambda: number): number {
  const x = (lambda - LAMBDA_MIN) / 10;
  if (x <= 0) return t[0];
  if (x >= t.length - 1) return t[t.length - 1];
  const i = Math.floor(x);
  const f = x - i;
  return t[i] * (1 - f) + t[i + 1] * f;
}

export const solarIrradiance = (l: number) => table(SOLAR_IRRADIANCE, l);

function lobe(l: number, mu: number, s1: number, s2: number): number {
  const t = (l - mu) / (l < mu ? s1 : s2);
  return Math.exp(-0.5 * t * t);
}

/** CIE 1931 2° 等色関数（近似） */
export function cie1931(l: number): [number, number, number] {
  const x = 1.056 * lobe(l, 599.8, 37.9, 31.0) + 0.362 * lobe(l, 442.0, 16.0, 26.7) - 0.065 * lobe(l, 501.1, 20.4, 26.2);
  const y = 0.821 * lobe(l, 568.8, 46.9, 40.5) + 0.286 * lobe(l, 530.9, 16.3, 31.1);
  const z = 1.217 * lobe(l, 437.0, 11.8, 36.0) + 0.681 * lobe(l, 459.0, 26.0, 13.8);
  return [x, y, z];
}

/** 波長 λ の単位分光量（/nm）を線形 sRGB 輝度に変換する係数（683 lm/W を含む） */
export function srgbCoefficients(l: number): [number, number, number] {
  const [x, y, z] = cie1931(l);
  const k = MAX_LUMINOUS_EFFICACY;
  return [
    k * (XYZ_TO_SRGB[0] * x + XYZ_TO_SRGB[1] * y + XYZ_TO_SRGB[2] * z),
    k * (XYZ_TO_SRGB[3] * x + XYZ_TO_SRGB[4] * y + XYZ_TO_SRGB[5] * z),
    k * (XYZ_TO_SRGB[6] * x + XYZ_TO_SRGB[7] * y + XYZ_TO_SRGB[8] * z),
  ];
}

export interface AtmosphereSettings {
  /** エアロゾル（Mie）の量の倍率。1 = Bruneton デモの標準（かなり澄んだ空） */
  aerosol: number;
  /** エアロゾルのスケールハイト [km] */
  mieScaleHeight: number;
  mieG: number;
  /** オゾン全量 [DU] */
  ozoneDU: number;
  groundAlbedo: number;
}

export const DEFAULT_ATMOSPHERE: AtmosphereSettings = {
  aerosol: 1,
  mieScaleHeight: 1.2,
  mieG: 0.8,
  ozoneDU: 300,
  groundAlbedo: 0.1,
};

/** 波長 λ[nm] での各係数 [1/km] */
export function coefficientsAt(l: number, s: AtmosphereSettings) {
  const um = l / 1000;
  const rayleigh = (RAYLEIGH_K * Math.pow(um, -4)) * 1000;
  const mieExt = ((5.328e-3 * s.aerosol) / (s.mieScaleHeight * 1000)) * 1000;
  const ozone = MAX_OZONE_NUMBER_DENSITY * (s.ozoneDU / 300) * table(OZONE_CROSS_SECTION, l) * 1000;
  return { rayleigh, mieScattering: mieExt * 0.9, mieExtinction: mieExt, ozone, solar: solarIrradiance(l) };
}

/** 事前計算に使う波長（3つずつのグループ） */
export function wavelengthGroups(numGroups: number): { lambdas: number[][]; dLambda: number } {
  const dLambda = (LAMBDA_MAX - LAMBDA_MIN) / (3 * numGroups);
  const lambdas: number[][] = [];
  for (let i = 0; i < numGroups; i++) {
    lambdas.push([0.5, 1.5, 2.5].map((k) => LAMBDA_MIN + (3 * i + k) * dLambda));
  }
  return { lambdas, dLambda };
}

/** 透過率テクスチャ・描画時に使う代表波長（Bruneton と同じ） */
export const RGB_LAMBDAS = [680, 550, 440];

/** 大気上端での太陽の照度（線形 sRGB, lx） */
export function solarIlluminanceRgb(numSamples = 471): [number, number, number] {
  const out: [number, number, number] = [0, 0, 0];
  const dl = (LAMBDA_MAX - LAMBDA_MIN) / (numSamples - 1);
  for (let i = 0; i < numSamples; i++) {
    const l = LAMBDA_MIN + i * dl;
    const c = srgbCoefficients(l);
    const e = solarIrradiance(l) * dl * (i === 0 || i === numSamples - 1 ? 0.5 : 1);
    out[0] += c[0] * e;
    out[1] += c[1] * e;
    out[2] += c[2] * e;
  }
  return out;
}
