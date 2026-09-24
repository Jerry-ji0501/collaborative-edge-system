// Scene 08 — inside the horizon: a hand-inked tunnel of rings, dashes and beads we fly through.
import { Ctx, boilOf, glow, wobbleCirclePath } from '../core/draw';
import { RGB, hex, rgba } from '../core/color';
import { E, TAU, clamp, lerp, mod, seg, smoothInt, smoothstep } from '../core/math';
import { RNG } from '../core/rng';

const F = 250;
const NR = 64;
const T0 = 20.3;
interface Ring {
  z: number;
  col: RGB;
  type: number;
  w: number;
  dir: number;
  seed: number;
  dots: number;
}
const PAL: RGB[] = ['#FFC857', '#FF9150', '#E8508E', '#9A72FF', '#5FD0E0', '#6C8CFF', '#F5E6C8'].map(hex);
const rings: Ring[] = [];
const ND = 240;
const dA = new Float32Array(ND);
const dR = new Float32Array(ND);
const dZ = new Float32Array(ND);
(function build() {
  const r = new RNG(5150);
  for (let i = 0; i < NR; i++) {
    rings.push({
      z: 0.3 + i * 0.3,
      col: PAL[(i * 2 + r.int(0, 1)) % PAL.length],
      type: r.pick([0, 0, 1, 2, 3]),
      w: r.range(0.6, 1.4),
      dir: r.sign(),
      seed: i * 7 + 3,
      dots: r.int(30, 56),
    });
  }
  for (let j = 0; j < ND; j++) {
    dA[j] = r.range(0, TAU);
    dR[j] = r.range(0.55, 1.2);
    dZ[j] = r.range(0, 10);
  }
})();

export function tunnelZ(t: number): number {
  if (t < T0) return 0;
  const d = t - T0;
  return 0.7 * d + 2.3 * (d - smoothInt(t, 22.1, 22.75));
}
const axX = (z: number): number => 0.12 * Math.sin(0.8 * z);
const axY = (z: number): number => 0.08 * Math.cos(0.6 * z + 1);

/** Draw the tunnel. With a clip circle it is seen through the black hole's horizon. */
export function drawTunnel(ctx: Ctx, t: number, clip?: { x: number; y: number; r: number }): void {
  if (t < T0 || t > 23.05) return;
  const boil = boilOf(t);
  const sway = 1 - seg(t, 22.2, 22.6);
  let vx = 960 + 30 * Math.sin(t * 0.9) * sway;
  let vy = 540 + 18 * Math.sin(t * 0.7 + 1) * sway;
  if (clip) {
    const k = seg(t, 20.55, 21.0, E.inOutSine);
    vx = lerp(clip.x, vx, k);
    vy = lerp(clip.y, vy, k);
  }
  ctx.save();
  if (clip) {
    ctx.beginPath();
    ctx.arc(clip.x, clip.y, clip.r, 0, TAU);
    ctx.clip();
  }
  const bg = ctx.createRadialGradient(vx, vy, 0, vx, vy, 1200);
  bg.addColorStop(0, '#221B46');
  bg.addColorStop(0.45, '#100D26');
  bg.addColorStop(1, '#05050D');
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, 1920, 1080);

  const zc = tunnelZ(t);
  const condense = seg(t, 22.45, 22.8, E.inCubic);
  const ringA = 1 - seg(t, 22.7, 22.86);
  const enter = clip ? seg(t, T0, 20.7) : 1;
  ctx.lineCap = 'round';
  for (let i = NR - 1; i >= 0; i--) {
    const rg = rings[i];
    const dz = rg.z - zc;
    if (dz < 0.06 || dz > 13) continue;
    const R = (F / dz) * (1 - condense * 0.97);
    if (R > 1700) continue;
    const ox = (axX(rg.z) - axX(zc)) * (F / dz) * (1 - condense);
    const oy = (axY(rg.z) - axY(zc)) * (F / dz) * (1 - condense);
    const cx = vx + ox;
    const cy = vy + oy;
    const al = smoothstep(13, 7.5, dz) * smoothstep(0.06, 0.25, dz) * ringA * enter;
    if (al <= 0.01) continue;
    const w = clamp(rg.w * (F / dz) * 0.024, 0.9, 30);
    const rot = rg.dir * 0.5 * t + rg.seed;
    const n = Math.round(clamp(R / 6, 24, 120));
    // soft luminous halo
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = al * 0.13;
    ctx.strokeStyle = rgba(rg.col);
    ctx.lineWidth = w * 3.4 + 2;
    ctx.beginPath();
    wobbleCirclePath(ctx, cx, cy, R, R * 0.012 + 0.5, rg.seed, boil, n, 1.6, rot);
    ctx.stroke();
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = al;
    ctx.strokeStyle = rgba(rg.col);
    ctx.fillStyle = rgba(rg.col);
    if (rg.type === 0) {
      ctx.lineWidth = w;
      ctx.beginPath();
      wobbleCirclePath(ctx, cx, cy, R, R * 0.012 + 0.5, rg.seed, boil, n, 1.6, rot);
      ctx.stroke();
      ctx.globalAlpha = al * 0.4;
      ctx.lineWidth = Math.max(0.6, w * 0.35);
      ctx.beginPath();
      wobbleCirclePath(ctx, cx, cy, R * 1.03, R * 0.015 + 0.5, rg.seed + 1, boil, n, 1.6, rot);
      ctx.stroke();
    } else if (rg.type === 1) {
      ctx.lineWidth = w;
      ctx.setLineDash([R * 0.22, R * 0.11]);
      ctx.lineDashOffset = -rot * R;
      ctx.beginPath();
      wobbleCirclePath(ctx, cx, cy, R, R * 0.01 + 0.5, rg.seed, boil, n, 1.6, rot);
      ctx.stroke();
      ctx.setLineDash([]);
    } else if (rg.type === 2) {
      ctx.beginPath();
      const dr = Math.max(0.8, w * 0.75);
      for (let d = 0; d < rg.dots; d++) {
        const a = (d / rg.dots) * TAU + rot;
        const x = cx + Math.cos(a) * R;
        const y = cy + Math.sin(a) * R;
        ctx.moveTo(x + dr, y);
        ctx.arc(x, y, dr, 0, TAU);
      }
      ctx.fill();
    } else {
      ctx.lineWidth = Math.max(0.6, w * 0.4);
      ctx.beginPath();
      wobbleCirclePath(ctx, cx, cy, R, R * 0.01 + 0.5, rg.seed, boil, n, 1.6, rot);
      ctx.stroke();
      for (let d = 0; d < 8; d++) {
        const a = (d / 8) * TAU - rot * 1.3;
        const x = cx + Math.cos(a) * R;
        const y = cy + Math.sin(a) * R;
        ctx.globalCompositeOperation = 'lighter';
        glow(ctx, x, y, w * 6 + 3, rg.col, al * 0.8);
        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = al;
        ctx.beginPath();
        ctx.arc(x, y, w * 1.2 + 0.6, 0, TAU);
        ctx.fill();
      }
    }
  }
  // drifting dust on the walls
  ctx.globalCompositeOperation = 'lighter';
  ctx.strokeStyle = 'rgb(245,230,200)';
  ctx.lineWidth = 1.7;
  ctx.globalAlpha = 0.5 * ringA * enter;
  ctx.beginPath();
  for (let j = 0; j < ND; j++) {
    const dz = mod(dZ[j] - zc, 10);
    if (dz < 0.12 || dz > 7) continue;
    const k1 = (F * dR[j]) / dz;
    const k0 = (F * dR[j]) / (dz + 0.14);
    const c = Math.cos(dA[j]);
    const s = Math.sin(dA[j]);
    const x1 = vx + c * k1;
    const y1 = vy + s * k1;
    if (x1 < -50 || x1 > 1970 || y1 < -50 || y1 > 1130) continue;
    ctx.moveTo(vx + c * k0, vy + s * k0);
    ctx.lineTo(x1, y1);
  }
  ctx.stroke();
  // the far light at the end of the tunnel, gathering itself before the pillar
  const pulse = 1 + 0.12 * Math.sin(t * 3.1);
  glow(ctx, vx, vy, (60 + 150 * condense) * pulse, hex('#FFE9C8'), 0.55 + 0.45 * condense);
  glow(ctx, vx, vy, 16 + 30 * condense, hex('#FFFFFF'), 0.9);
  ctx.restore();
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
}
