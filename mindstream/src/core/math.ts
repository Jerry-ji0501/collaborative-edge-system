// Small math toolkit: clamps, easing, segment helpers, damped springs.
export const W = 1920;
export const H = 1080;
export const TAU = Math.PI * 2;

export const clamp = (x: number, a = 0, b = 1): number => (x < a ? a : x > b ? b : x);
export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;
export const invLerp = (a: number, b: number, x: number): number => clamp((x - a) / (b - a));
export const smoothstep = (a: number, b: number, x: number): number => {
  const t = invLerp(a, b, x);
  return t * t * (3 - 2 * t);
};
export const smootherstep = (a: number, b: number, x: number): number => {
  const t = invLerp(a, b, x);
  return t * t * t * (t * (t * 6 - 15) + 10);
};

export type Ease = (t: number) => number;
export const E = {
  linear: (t: number) => t,
  inQuad: (t: number) => t * t,
  outQuad: (t: number) => t * (2 - t),
  inOutQuad: (t: number) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2),
  inCubic: (t: number) => t * t * t,
  outCubic: (t: number) => 1 - Math.pow(1 - t, 3),
  inOutCubic: (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
  inQuart: (t: number) => t * t * t * t,
  outQuart: (t: number) => 1 - Math.pow(1 - t, 4),
  inOutQuart: (t: number) => (t < 0.5 ? 8 * t * t * t * t : 1 - Math.pow(-2 * t + 2, 4) / 2),
  inExpo: (t: number) => (t <= 0 ? 0 : Math.pow(2, 10 * t - 10) * (1 + 1 / 1023) - 1 / 1023),
  outExpo: (t: number) => (t >= 1 ? 1 : 1 - Math.pow(2, -10 * t)),
  inSine: (t: number) => 1 - Math.cos((t * Math.PI) / 2),
  outSine: (t: number) => Math.sin((t * Math.PI) / 2),
  inOutSine: (t: number) => -(Math.cos(Math.PI * t) - 1) / 2,
  outBack: (t: number) => {
    const c1 = 1.70158;
    const c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  },
};

/** Progress of t through [a,b], clamped and eased. */
export const seg = (t: number, a: number, b: number, e: Ease = E.linear): number => e(invLerp(a, b, t));

/** 0 → 1 → 0 sine bump over [a,b]. */
export const bump = (t: number, a: number, b: number): number => {
  if (t <= a || t >= b) return 0;
  return Math.sin(Math.PI * ((t - a) / (b - a)));
};

/** Damped oscillation that starts at 0 (a nudge that rings out). */
export const dampedSin = (tau: number, freq: number, decay: number): number =>
  tau <= 0 ? 0 : Math.exp(-decay * tau) * Math.sin(TAU * freq * tau);

/** Damped oscillation that starts at 1 (an impact that rings out). */
export const dampedCos = (tau: number, freq: number, decay: number): number =>
  tau <= 0 ? 0 : Math.exp(-decay * tau) * Math.cos(TAU * freq * tau);

export const mod = (a: number, n: number): number => ((a % n) + n) % n;

/** When a rising function f first reaches `target` within [lo, hi] (bisection). */
export function reach(f: (t: number) => number, target: number, lo: number, hi: number): number {
  for (let i = 0; i < 40; i++) {
    const m = (lo + hi) / 2;
    if (f(m) < target) lo = m;
    else hi = m;
  }
  return hi;
}

/** ∫ smoothstep(a,b,τ) dτ from -∞ to t — used to ramp angular velocities deterministically. */
export function smoothInt(t: number, a: number, b: number): number {
  if (t <= a) return 0;
  const d = b - a;
  if (t >= b) return d * 0.5 + (t - b);
  const x = (t - a) / d;
  return d * (x * x * x - 0.5 * x * x * x * x);
}

export const rotX = (x: number, y: number, a: number): number => x * Math.cos(a) - y * Math.sin(a);
export const rotY = (x: number, y: number, a: number): number => x * Math.sin(a) + y * Math.cos(a);
