// Scene 10 — the blue glow condenses into a tiny orb that drifts down beside the creature.
import { Ctx, boilOf, softGlow, wobbleCirclePath } from '../core/draw';
import { C, hex, rgba } from '../core/color';
import { E, TAU, bump, dampedSin, lerp, seg } from '../core/math';
import { Cam, applyCam } from '../core/camera';
import { darkness } from '../core/timeline';
import { GROUND_Y } from '../core/geometry';
import { drawCharacter, drawGround, finalePose } from './character';

const DOT_R = 11;
const D0 = { x: 585, y: 420 };
const FINAL_C = { x: 705, y: 590 };
const FINAL_Z = 1.35;
const Z0 = 29 / DOT_R;

function dotPos(t: number): [number, number] {
  const tau = seg(t, 27.95, 28.55, E.inOutSine);
  const x = D0.x + 16 * Math.sin(tau * Math.PI * 1.6) * (1 - tau);
  let y = lerp(D0.y, GROUND_Y - DOT_R - 1, tau);
  y -= Math.abs(dampedSin(t - 28.55, 1.7, 4.5)) * 16;
  y -= 13 * bump(t, 29.27, 29.5);
  return [x, y];
}

function finaleCam(t: number): Cam {
  const k = seg(t, 27.95, 28.85, E.inOutCubic);
  const [dx, dy] = dotPos(t);
  return { x: lerp(dx, FINAL_C.x, k), y: lerp(dy, FINAL_C.y, k), zoom: Z0 * Math.pow(FINAL_Z / Z0, k), rot: 0 };
}

function drawOrb(ctx: Ctx, x: number, y: number, r: number, dark: number, a: number, boil: number, lw: number): void {
  softGlow(ctx, x, y, r * 2.4, hex('#6FB7FF'), 0.6 * a, dark);
  ctx.globalAlpha = a;
  const g = ctx.createRadialGradient(x - 0.35 * r, y - 0.38 * r, r * 0.05, x, y, r);
  g.addColorStop(0, '#EAF7FF');
  g.addColorStop(0.35, '#8CC9FF');
  g.addColorStop(0.8, '#3F86E6');
  g.addColorStop(1, '#2A62C4');
  ctx.fillStyle = g;
  ctx.beginPath();
  wobbleCirclePath(ctx, x, y, r, r * 0.02, 71, boil, 48);
  ctx.fill();
  if (dark < 0.99) {
    ctx.globalAlpha = a * (1 - dark);
    ctx.strokeStyle = rgba(C.ink);
    ctx.lineWidth = lw;
    ctx.stroke();
  }
  ctx.globalAlpha = a * 0.9 * (1 - Math.min(1, Math.max(0, (r - 70) / 160)));
  ctx.fillStyle = 'rgba(255,255,255,0.9)';
  ctx.beginPath();
  ctx.ellipse(x - 0.38 * r, y - 0.42 * r, r * 0.22, r * 0.13, -0.6, 0, TAU);
  ctx.fill();
  ctx.globalAlpha = 1;
}

export function drawFinale(ctx: Ctx, t: number): void {
  if (t < 27.38) return;
  const boil = boilOf(t);
  const dark = darkness(t);
  if (t < 27.95) {
    const r = lerp(1300, 29, seg(t, 27.38, 27.9, E.inOutCubic));
    drawOrb(ctx, 960, 540, r, dark, seg(t, 27.38, 27.46), boil, 2.4);
    if (t < 27.9) return;
  }
  const cam = finaleCam(t);
  const endA = 1 - seg(t, 29.74, 30.0, E.inOutSine);
  ctx.save();
  applyCam(ctx, cam);
  drawGround(ctx, t, seg(t, 27.95, 28.5) * endA);
  // the dot's shadow once it nears the ground
  const [x, y] = dotPos(t);
  const near = seg(y, GROUND_Y - 120, GROUND_Y - DOT_R);
  if (near > 0) {
    ctx.globalAlpha = 0.16 * near * endA;
    ctx.fillStyle = rgba(C.ink);
    ctx.beginPath();
    ctx.ellipse(x + 1, GROUND_Y + 3, 13 * (0.5 + 0.5 * near), 3.5, 0, 0, TAU);
    ctx.fill();
  }
  drawCharacter(ctx, finalePose(t), t);
  const pulse = 1 + 0.06 * Math.sin(t * 4.2) + 0.12 * bump(t, 29.1, 29.5);
  drawOrb(ctx, x, y, DOT_R * pulse, dark, endA, boil, 2.4 / cam.zoom);
  ctx.restore();
  ctx.globalAlpha = 1;
}
