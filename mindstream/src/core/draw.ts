// Hand-drawn drawing primitives: smoothed paths, tapered ink ribbons, wobbly circles, glow sprites.
import { RGB, rgba } from './color';
import { noise2 } from './noise';
import { TAU } from './math';

export type Ctx = CanvasRenderingContext2D;

/** Line boil: shapes re-jitter at 12 fps, like hand-drawn animation shot "on twos". */
export const boilOf = (t: number, fps = 12): number => Math.floor(t * fps + 1e-6);

export function makeCanvas(w: number, h: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

/** Smooth path through flat points [x0,y0,x1,y1,...] using quadratic midpoints. */
export function tracePath(ctx: Ctx, pts: ArrayLike<number>, count: number, closed: boolean, move = true): void {
  const n = count;
  if (n < 2) return;
  if (!closed) {
    if (move) ctx.moveTo(pts[0], pts[1]);
    else ctx.lineTo(pts[0], pts[1]);
    if (n === 2) {
      ctx.lineTo(pts[2], pts[3]);
      return;
    }
    for (let i = 1; i < n - 1; i++) {
      const x = pts[2 * i];
      const y = pts[2 * i + 1];
      ctx.quadraticCurveTo(x, y, (x + pts[2 * i + 2]) * 0.5, (y + pts[2 * i + 3]) * 0.5);
    }
    ctx.lineTo(pts[2 * n - 2], pts[2 * n - 1]);
  } else {
    ctx.moveTo((pts[2 * n - 2] + pts[0]) * 0.5, (pts[2 * n - 1] + pts[1]) * 0.5);
    for (let i = 0; i < n; i++) {
      const j = i + 1 < n ? i + 1 : 0;
      const x = pts[2 * i];
      const y = pts[2 * i + 1];
      ctx.quadraticCurveTo(x, y, (x + pts[2 * j]) * 0.5, (y + pts[2 * j + 1]) * 0.5);
    }
    ctx.closePath();
  }
}

/** Straight-segment polygon (for shapes that must keep sharp corners). */
export function polyPath(ctx: Ctx, pts: ArrayLike<number>, count: number): void {
  if (count < 2) return;
  ctx.moveTo(pts[0], pts[1]);
  for (let i = 1; i < count; i++) ctx.lineTo(pts[2 * i], pts[2 * i + 1]);
  ctx.closePath();
}

const RL: number[] = [];
const RR: number[] = [];
/**
 * Variable-width ink stroke ("ribbon") along a centreline, added as a closed sub-path.
 * width(i) returns full width at point i.
 */
export function ribbonPath(
  ctx: Ctx,
  pts: ArrayLike<number>,
  count: number,
  width: (i: number) => number,
  smooth = true,
): void {
  if (count < 2) return;
  RL.length = 0;
  RR.length = 0;
  for (let i = 0; i < count; i++) {
    const i0 = i > 0 ? i - 1 : 0;
    const i1 = i < count - 1 ? i + 1 : count - 1;
    const dx = pts[2 * i1] - pts[2 * i0];
    const dy = pts[2 * i1 + 1] - pts[2 * i0 + 1];
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len;
    const ny = dx / len;
    const w = width(i) * 0.5;
    const x = pts[2 * i];
    const y = pts[2 * i + 1];
    RL.push(x + nx * w, y + ny * w);
    RR.push(x - nx * w, y - ny * w);
  }
  // right side reversed in place
  for (let a = 0, b = count - 1; a < b; a++, b--) {
    const ax = RR[2 * a];
    const ay = RR[2 * a + 1];
    RR[2 * a] = RR[2 * b];
    RR[2 * a + 1] = RR[2 * b + 1];
    RR[2 * b] = ax;
    RR[2 * b + 1] = ay;
  }
  if (smooth) {
    tracePath(ctx, RL, count, false, true);
    tracePath(ctx, RR, count, false, false);
  } else {
    ctx.moveTo(RL[0], RL[1]);
    for (let i = 1; i < count; i++) ctx.lineTo(RL[2 * i], RL[2 * i + 1]);
    for (let i = 0; i < count; i++) ctx.lineTo(RR[2 * i], RR[2 * i + 1]);
  }
  ctx.closePath();
}

const WC: number[] = [];
/** Wobbly (hand-drawn) circle sub-path. Noise is sampled on a circle so it loops seamlessly. */
export function wobbleCirclePath(
  ctx: Ctx,
  cx: number,
  cy: number,
  r: number,
  amp: number,
  seed: number,
  boil: number,
  n = 72,
  freq = 1.6,
  phase = 0,
): void {
  WC.length = 0;
  for (let i = 0; i < n; i++) {
    const a = (i / n) * TAU;
    const rr = r + amp * noise2(Math.cos(a - phase) * freq + seed * 3.1, Math.sin(a - phase) * freq + boil * 1.37, seed);
    WC.push(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr);
  }
  tracePath(ctx, WC, n, true);
}

// ---------------------------------------------------------------- glow sprites
const glowCache = new Map<string, HTMLCanvasElement>();
function q(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v / 6) * 6));
}
export function glowSprite(c: RGB, hardness = 1): HTMLCanvasElement {
  const r = q(c[0]);
  const g = q(c[1]);
  const b = q(c[2]);
  const key = `${r},${g},${b},${hardness}`;
  let s = glowCache.get(key);
  if (s) return s;
  if (glowCache.size > 600) glowCache.clear();
  const size = 128;
  s = makeCanvas(size, size);
  const gx = s.getContext('2d')!;
  const grd = gx.createRadialGradient(64, 64, 0, 64, 64, 64);
  for (let i = 0; i <= 14; i++) {
    const x = i / 14;
    const a = Math.exp(-x * x * 5 * hardness) * (1 - x * x);
    grd.addColorStop(x, rgba([r, g, b], a));
  }
  gx.fillStyle = grd;
  gx.fillRect(0, 0, size, size);
  glowCache.set(key, s);
  return s;
}

export function glow(ctx: Ctx, x: number, y: number, r: number, c: RGB, a: number, hardness = 1): void {
  if (a <= 0.003 || r <= 0.3) return;
  ctx.globalAlpha = a > 1 ? 1 : a;
  ctx.drawImage(glowSprite(c, hardness), x - r, y - r, r * 2, r * 2);
}

/**
 * Glow that behaves on both paper and night: painted (source-over) on light grounds,
 * emitted (additive) on dark grounds. `dark` is 0 on cream and 1 in space.
 */
export function softGlow(ctx: Ctx, x: number, y: number, r: number, c: RGB, a: number, dark: number, hardness = 1): void {
  const prev = ctx.globalCompositeOperation;
  if (dark < 0.999) {
    ctx.globalCompositeOperation = 'source-over';
    glow(ctx, x, y, r, c, a * (1 - dark) * 0.7, hardness);
  }
  if (dark > 0.001) {
    ctx.globalCompositeOperation = 'lighter';
    glow(ctx, x, y, r, c, a * dark, hardness);
  }
  ctx.globalCompositeOperation = prev;
}

/** Four-point twinkle. */
export function sparkle(ctx: Ctx, x: number, y: number, r: number, rot: number, c: RGB, a: number): void {
  if (a <= 0.003 || r <= 0.3) return;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rot);
  ctx.globalAlpha = a > 1 ? 1 : a;
  ctx.fillStyle = rgba(c, 1);
  const k = r * 0.12;
  ctx.beginPath();
  ctx.moveTo(0, -r);
  ctx.quadraticCurveTo(k, -k, r, 0);
  ctx.quadraticCurveTo(k, k, 0, r);
  ctx.quadraticCurveTo(-k, k, -r, 0);
  ctx.quadraticCurveTo(-k, -k, 0, -r);
  ctx.fill();
  ctx.restore();
}

// ---------------------------------------------------------------- batched particles
/**
 * Batches thousands of short strokes by (palette colour, alpha level, width level)
 * so a whole galaxy is drawn with ~100 stroke calls.
 */
export class StrokeBatch {
  private buckets = new Map<number, number[]>();
  add(pal: number, alpha: number, wLevel: number, x0: number, y0: number, x1: number, y1: number): void {
    if (alpha <= 0.02) return;
    const aL = Math.min(9, Math.max(0, Math.round(alpha * 9)));
    if (aL === 0) return;
    const key = pal * 1000 + aL * 10 + wLevel;
    let arr = this.buckets.get(key);
    if (!arr) {
      arr = [];
      this.buckets.set(key, arr);
    }
    arr.push(x0, y0, x1, y1);
  }
  flush(ctx: Ctx, palette: string[], widths: number[], alphaScale = 1, clear = true): void {
    ctx.lineCap = 'round';
    for (const [key, arr] of this.buckets) {
      if (arr.length === 0) continue;
      if (alphaScale <= 0.003) {
        if (clear) arr.length = 0;
        continue;
      }
      const pal = Math.floor(key / 1000);
      const aL = Math.floor((key % 1000) / 10);
      const wL = key % 10;
      ctx.strokeStyle = palette[pal];
      ctx.globalAlpha = Math.min(1, (aL / 9) * alphaScale);
      ctx.lineWidth = widths[wL];
      ctx.beginPath();
      for (let i = 0; i < arr.length; i += 4) {
        ctx.moveTo(arr[i], arr[i + 1]);
        ctx.lineTo(arr[i + 2], arr[i + 3]);
      }
      ctx.stroke();
      if (clear) arr.length = 0;
    }
    ctx.globalAlpha = 1;
  }
}
