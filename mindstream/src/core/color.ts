// Palette + colour helpers. Colours are RGB tuples so they can be mixed across scenes.
import { clamp } from './math';

export type RGB = [number, number, number];

export function hex(h: string): RGB {
  const n = parseInt(h.replace('#', ''), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function mix(a: RGB, b: RGB, t: number): RGB {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

export function rgba(c: RGB, a = 1): string {
  return `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${a < 0 ? 0 : a > 1 ? 1 : a.toFixed(3)})`;
}

export function ramp(stops: RGB[], t: number): RGB {
  const x = clamp(t) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  return mix(stops[i], stops[i + 1], x - i);
}

export const lighten = (c: RGB, k: number): RGB => mix(c, [255, 255, 255], k);
export const darken = (c: RGB, k: number): RGB => mix(c, [0, 0, 0], k);

export const C = {
  cream: hex('#F2E8CF'),
  creamIn: hex('#F6EEDB'),
  creamOut: hex('#E6D7B5'),
  ink: hex('#2F2536'),
  orange: hex('#EE7B3A'),
  orangeDeep: hex('#C4552A'),
  orangeLight: hex('#F7A56C'),
  yellow: hex('#F6C443'),
  gold: hex('#EDA630'),
  violet: hex('#8A63E0'),
  cyan: hex('#3FBCC8'),
  magenta: hex('#D84A86'),
  softRed: hex('#E4575A'),
  navyIn: hex('#1B2046'),
  navyOut: hex('#0A0C1E'),
  white: [255, 255, 255] as RGB,
  warmWhite: hex('#FFF4DC'),
};

/** Seven spectral bands — slightly muted, painterly rather than neon. */
export const SPECTRUM: RGB[] = ['#EE5A5E', '#F6953F', '#F6CE4A', '#88CC6B', '#45C1CD', '#5B7FE6', '#9468E4'].map(hex);
