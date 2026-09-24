// Background stars with depth parallax, gentle twinkle, optional lensing and radial streaking.
import { Ctx } from '../core/draw';
import { RGB, hex, rgba } from '../core/color';
import { RNG } from '../core/rng';
import { TAU } from '../core/math';

const N = 950;
const sx = new Float32Array(N);
const sy = new Float32Array(N);
const sd = new Float32Array(N);
const ss = new Float32Array(N);
const sp = new Float32Array(N);
const sc = new Uint8Array(N);
const COLS: RGB[] = ['#FFFFFF', '#CFE0FF', '#FFE9C7', '#D9CCFF'].map(hex);
const CSS = COLS.map((c) => rgba(c));

(function build() {
  const r = new RNG(99);
  for (let i = 0; i < N; i++) {
    sx[i] = r.range(-1.35, 1.35) * 960;
    sy[i] = r.range(-1.35, 1.35) * 540;
    sd[i] = Math.pow(r.next(), 1.6);
    ss[i] = r.chance(0.06) ? r.range(2.2, 3.2) : r.range(0.9, 1.8);
    sp[i] = r.range(0, TAU);
    sc[i] = r.int(0, 3);
  }
})();

export interface StarOpts {
  alpha: number;
  scale: number; // camera zoom relative to the reference framing
  scalePrev?: number; // for radial streaks during fast pushes
  rot: number;
  cx?: number;
  cy?: number;
  lens?: { x: number; y: number; r: number };
  clipBelow?: (x: number) => number; // hide stars below a horizon
}

export function drawStars(ctx: Ctx, t: number, o: StarOpts): void {
  if (o.alpha <= 0.01) return;
  const cx = o.cx ?? 960;
  const cy = o.cy ?? 540;
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.lineCap = 'round';
  for (let i = 0; i < N; i++) {
    const d = sd[i];
    const k = Math.pow(o.scale, 0.05 + 0.2 * d);
    const a = o.rot * (0.3 + 0.7 * d);
    const ca = Math.cos(a);
    const sa = Math.sin(a);
    let px = cx + (sx[i] * ca - sy[i] * sa) * k;
    let py = cy + (sx[i] * sa + sy[i] * ca) * k;
    if (o.lens) {
      const dx = px - o.lens.x;
      const dy = py - o.lens.y;
      const dd = dx * dx + dy * dy;
      const rr = o.lens.r * o.lens.r;
      if (dd < rr * 60) {
        const f = 1 + (rr * 1.3) / Math.max(dd, 1);
        px = o.lens.x + dx * f;
        py = o.lens.y + dy * f;
      }
    }
    if (px < -20 || px > 1940 || py < -20 || py > 1100) continue;
    if (o.clipBelow && py > o.clipBelow(px) - 4) continue;
    const tw = 0.55 + 0.45 * Math.sin(t * (0.8 + d * 2.2) + sp[i]);
    ctx.globalAlpha = Math.min(1, o.alpha * tw * (0.35 + 0.65 * d));
    const s = ss[i];
    if (o.scalePrev !== undefined && o.scalePrev !== o.scale) {
      const k0 = Math.pow(o.scalePrev, 0.05 + 0.2 * d);
      const qx = cx + (sx[i] * ca - sy[i] * sa) * k0;
      const qy = cy + (sx[i] * sa + sy[i] * ca) * k0;
      ctx.strokeStyle = CSS[sc[i]];
      ctx.lineWidth = s;
      ctx.beginPath();
      ctx.moveTo(qx, qy);
      ctx.lineTo(px, py);
      ctx.stroke();
    } else {
      ctx.fillStyle = CSS[sc[i]];
      ctx.fillRect(px - s / 2, py - s / 2, s, s);
    }
  }
  ctx.restore();
}
