// Seeded randomness — every frame of the film is reproducible.
import { TAU } from './math';

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class RNG {
  private f: () => number;
  constructor(seed: number) {
    this.f = mulberry32(seed);
  }
  next(): number {
    return this.f();
  }
  range(a: number, b: number): number {
    return a + (b - a) * this.f();
  }
  int(a: number, b: number): number {
    return Math.floor(this.range(a, b + 1));
  }
  sign(): number {
    return this.f() < 0.5 ? -1 : 1;
  }
  chance(p: number): boolean {
    return this.f() < p;
  }
  gauss(): number {
    let u = 0;
    while (u === 0) u = this.f();
    const v = this.f();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v);
  }
  pick<T>(arr: readonly T[]): T {
    return arr[Math.floor(this.f() * arr.length) % arr.length];
  }
}

export function hashInt(n: number): number {
  let x = n | 0;
  x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
  x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
  x ^= x >>> 16;
  return x >>> 0;
}

export const hash01 = (n: number): number => hashInt(n) / 4294967296;
export const hash2 = (a: number, b: number): number =>
  hashInt(Math.imul(a | 0, 0x27d4eb2d) ^ Math.imul((b | 0) + 0x165667b1, 0x85ebca77)) / 4294967296;
