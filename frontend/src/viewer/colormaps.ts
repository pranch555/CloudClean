import * as THREE from 'three';

/*
 * Data colour maps (dataviz method):
 *  - diverging: deviation from CAD. Two hues (blue = material missing / inside, red = excess / outside) around a
 *    neutral grey band that covers the tolerance, so "in tolerance" never reads as a colour.
 *  - sequential: one hue light → dark for magnitudes such as scan density / distance.
 */
const NEG = ['#9ec5f4', '#5598e7', '#256abf', '#104281'];
const POS = ['#f3b1a9', '#e66767', '#c43a3a', '#8b1d1d'];
const NEUTRAL = '#aab0b9';
const SEQ = ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'];
export const NAN_COLOR = new THREE.Color('#4a505b');

const hex = (h: string) => new THREE.Color(h);

function sampleStops(stops: string[], t: number, out: THREE.Color): THREE.Color {
  const x = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const i = Math.min(Math.floor(x), stops.length - 2);
  return out.copy(hex(stops[i])).lerp(hex(stops[i + 1]), x - i);
}

export interface ScalarStyle {
  kind: 'diverging' | 'sequential';
  min: number;       // sequential lower bound / diverging −range
  max: number;       // sequential upper bound / diverging +range
  tolerance: number; // diverging only: |v| ≤ tolerance is neutral
  steps: number;     // 0 = continuous, otherwise banded (easier to read exact values)
  part?: [number, number]; // sequential only: the stretch of the ramp used, 0..1 (default all of it)
}

/** Colour for one value (used for meshes, legends and tooltips). */
export function scalarColor(v: number, s: ScalarStyle, out = new THREE.Color()): THREE.Color {
  if (!Number.isFinite(v)) return out.copy(NAN_COLOR);
  if (s.kind === 'sequential') {
    const span = s.max - s.min;
    let t = (v - s.min) / (Math.abs(span) < 1e-12 ? 1e-12 : span); // min > max reverses the ramp
    if (s.steps > 1) t = Math.min(Math.floor(Math.max(t, 0) * s.steps) / (s.steps - 1), 1);
    if (s.part) t = s.part[0] + Math.min(Math.max(t, 0), 1) * (s.part[1] - s.part[0]);
    return sampleStops(SEQ, t, out);
  }
  const range = Math.max(Math.abs(s.max), Math.abs(s.min), s.tolerance * 1.0001, 1e-12);
  const a = Math.abs(v);
  if (a <= s.tolerance) return out.copy(hex(NEUTRAL));
  let t = (a - s.tolerance) / Math.max(range - s.tolerance, 1e-12);
  if (s.steps > 1) t = Math.min((Math.floor(Math.min(t, 0.9999) * s.steps) + 0.5) / s.steps, 1);
  return sampleStops(v < 0 ? NEG : POS, t, out);
}

/**
 * One texel of an sRGB lookup texture. THREE.Color holds linear values (colour management): storing those as the
 * bytes of an sRGB texture decoded them twice, and the points showed every data colour much darker than its legend
 * (the dense end of the live density ramp came out almost black).
 */
const srgb = { r: 0, g: 0, b: 0 };
function putSRGB(c: THREE.Color, data: Uint8Array, i: number) {
  c.getRGB(srgb, THREE.SRGBColorSpace);
  data.set([Math.round(srgb.r * 255), Math.round(srgb.g * 255), Math.round(srgb.b * 255), 255], i * 4);
}

/** 1-D lookup texture spanning [lo, hi] used by the point shader. */
export function lutTexture(style: ScalarStyle, size = 1024): { texture: THREE.DataTexture; lo: number; hi: number } {
  const lo = style.kind === 'diverging' ? -Math.max(Math.abs(style.min), Math.abs(style.max)) : style.min;
  const hi = style.kind === 'diverging' ? -lo : style.max;
  const data = new Uint8Array(size * 4);
  const c = new THREE.Color();
  for (let i = 0; i < size; i++) putSRGB(scalarColor(lo + ((i + 0.5) / size) * (hi - lo), style, c), data, i);
  const texture = new THREE.DataTexture(data, size, 1, THREE.RGBAFormat);
  texture.magFilter = style.steps > 1 ? THREE.NearestFilter : THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return { texture, lo, hi };
}

export function gradientCss(style: ScalarStyle, stops = 48): string {
  const lo = style.kind === 'diverging' ? -Math.max(Math.abs(style.min), Math.abs(style.max)) : style.min;
  const hi = style.kind === 'diverging' ? -lo : style.max;
  const c = new THREE.Color();
  const parts: string[] = [];
  for (let i = 0; i <= stops; i++) {
    const v = lo + (i / stops) * (hi - lo);
    parts.push(`#${scalarColor(v, style, c).getHexString()} ${((i / stops) * 100).toFixed(1)}%`);
  }
  return `linear-gradient(90deg, ${parts.join(', ')})`;
}

/*
 * Height (Display -> Colour -> Height, and the live scan's "Height"): the sequential ramp from the bottom to the top
 * of what is on screen. Its dark blues are left out so the shape shading still reads on every part, and like the
 * density colours it flips with the stage so the top always stands out: on the light Paper stage the bottom is light
 * and the top a stronger blue, on the dark Carbon stage the other way round.
 */
const HEIGHT_LO = 0.04;
const HEIGHT_HI = 0.6;

/** Colour at height t (0 bottom .. 1 top). */
export function heightColor(t: number, dark: boolean, out = new THREE.Color()): THREE.Color {
  const u = Math.min(Math.max(t, 0), 1);
  return sampleStops(SEQ, HEIGHT_LO + (dark ? 1 - u : u) * (HEIGHT_HI - HEIGHT_LO), out);
}

/** Lookup texture for the shaders (bottom at u = 0). */
export function heightLut(dark: boolean, size = 256): THREE.DataTexture {
  const data = new Uint8Array(size * 4);
  const c = new THREE.Color();
  for (let i = 0; i < size; i++) putSRGB(heightColor((i + 0.5) / size, dark, c), data, i);
  const texture = new THREE.DataTexture(data, size, 1, THREE.RGBAFormat);
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.needsUpdate = true;
  return texture;
}

/** CSS gradient of the height colours, bottom on the left (the keys next to the colour choices). */
export function heightGradientCss(dark: boolean, stops = 24): string {
  const c = new THREE.Color();
  const parts = Array.from({ length: stops + 1 }, (_, i) => `#${heightColor(i / stops, dark, c).getHexString()} ${((i / stops) * 100).toFixed(1)}%`);
  return `linear-gradient(90deg, ${parts.join(', ')})`;
}
