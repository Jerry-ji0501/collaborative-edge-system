// Seeded value noise (1D / 2D) and fBm — used for line boil, wobble and paper.
import { hashInt } from './rng';

const fade = (t: number): number => t * t * (3 - 2 * t);

function h1(i: number, seed: number): number {
  return hashInt(Math.imul(i, 0x9e3779b1) ^ Math.imul(seed + 0x632be5ab, 0x85ebca77)) / 4294967296;
}

function h2(ix: number, iy: number, seed: number): number {
  return (
    hashInt(Math.imul(ix, 0x27d4eb2d) ^ Math.imul(iy, 0x165667b1) ^ Math.imul(seed + 0x3c6ef372, 0x85ebca77)) /
    4294967296
  );
}

/** 1D value noise in [-1, 1]. */
export function noise1(x: number, seed = 0): number {
  const i = Math.floor(x);
  const u = fade(x - i);
  const a = h1(i, seed);
  return (a + (h1(i + 1, seed) - a) * u) * 2 - 1;
}

/** 2D value noise in [-1, 1]. */
export function noise2(x: number, y: number, seed = 0): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const ux = fade(x - ix);
  const uy = fade(y - iy);
  const a = h2(ix, iy, seed);
  const b = h2(ix + 1, iy, seed);
  const c = h2(ix, iy + 1, seed);
  const d = h2(ix + 1, iy + 1, seed);
  const top = a + (b - a) * ux;
  const bot = c + (d - c) * ux;
  return (top + (bot - top) * uy) * 2 - 1;
}

export function fbm2(x: number, y: number, oct = 4, seed = 0): number {
  let s = 0;
  let amp = 0.5;
  let f = 1;
  let norm = 0;
  for (let o = 0; o < oct; o++) {
    s += amp * noise2(x * f, y * f, seed + o * 101);
    norm += amp;
    amp *= 0.5;
    f *= 2.03;
  }
  return s / norm;
}
