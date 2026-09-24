// Scene 01 / 10 — the little orange creature. Pose is a pure function of time.
import { Ctx, boilOf, glow, tracePath } from '../core/draw';
import { C, RGB, hex, mix, rgba } from '../core/color';
import { E, TAU, bump, clamp, dampedCos, dampedSin, lerp, seg } from '../core/math';
import { noise1 } from '../core/noise';
import { CREATURE_X, GROUND_Y } from '../core/geometry';

export interface Pose {
  x: number;
  y: number;
  s: number;
  tilt: number;
  sx: number;
  sy: number;
  lift: number; // walk bounce (negative = up)
  air: number; // jump height (positive = up)
  legPhase: number;
  legAmp: number;
  lookX: number;
  lookY: number;
  eyeScale: number;
  blink: number;
  mouth: number; // -1 surprised "o"
  smile: number; // 0..1
  antenna: number;
  bulb: number;
  bulbCol: RGB;
  alpha: number;
}

const BW = 168;
const BH = 138;
const LEG = 30;

function basePose(): Pose {
  return {
    x: CREATURE_X,
    y: GROUND_Y,
    s: 1,
    tilt: 0,
    sx: 1,
    sy: 1,
    lift: 0,
    air: 0,
    legPhase: 0,
    legAmp: 0,
    lookX: 0.35,
    lookY: 0,
    eyeScale: 1,
    blink: 0,
    mouth: 0,
    smile: 0,
    antenna: 0,
    bulb: 0,
    bulbCol: hex('#FFD66B'),
    alpha: 1,
  };
}

const blinkAt = (t: number, t0: number, dur = 0.16): number => {
  const x = (t - t0) / dur;
  if (x <= 0 || x >= 1) return 0;
  return Math.pow(Math.sin(Math.PI * x), 0.6);
};

/** Squash → leap → land, as offsets. */
function hop(p: Pose, t: number, tA: number, tUp: number, tLand: number, height: number): void {
  const ant = bump(t, tA, tUp + 0.02);
  p.sy -= 0.1 * ant;
  p.sx += 0.07 * ant;
  if (t > tUp && t < tLand) {
    const x = (t - tUp) / (tLand - tUp);
    const s = Math.sin(Math.PI * x);
    p.air += height * s;
    p.sy += 0.08 * Math.sin(Math.PI * Math.min(1, x * 1.6));
    p.sx -= 0.05 * Math.sin(Math.PI * Math.min(1, x * 1.6));
  }
  const imp = dampedCos(t - tLand, 2.4, 6.5);
  p.sy -= 0.12 * imp;
  p.sx += 0.09 * imp;
  p.antenna += 0.22 * dampedSin(t - tUp, 2.2, 3.2) - 0.3 * dampedSin(t - tLand, 2.6, 3.4);
}

// --- walk-in kinematics: cruise at V0, then brake linearly to a stop at T2
const V0 = 392;
const T1 = 2.0;
const T2 = 2.9;
const X0 = CREATURE_X - (V0 * T1 + (V0 * (T2 - T1)) / 2);

function walkDist(t: number): number {
  if (t <= T1) return V0 * t;
  if (t <= T2) {
    const tau = t - T1;
    return V0 * T1 + V0 * (tau - (tau * tau) / (2 * (T2 - T1)));
  }
  return V0 * T1 + (V0 * (T2 - T1)) / 2;
}
function walkSpeed(t: number): number {
  if (t <= T1) return V0;
  if (t <= T2) return V0 * (1 - (t - T1) / (T2 - T1));
  return 0;
}

export function introPose(t: number): Pose {
  const p = basePose();
  const d = walkDist(t);
  const wa = walkSpeed(t) / V0;
  p.x = X0 + d;
  const ph = (d / 96) * Math.PI;
  p.legPhase = ph;
  p.legAmp = wa;
  const sinp = Math.sin(ph);
  const cosp = Math.cos(ph);
  p.lift = -Math.abs(cosp) * 10 * wa;
  const contact = Math.pow(Math.abs(sinp), 6) * wa;
  p.sy = 1 - 0.065 * contact;
  p.sx = 1 + 0.05 * contact;
  p.tilt = 0.065 * wa + 0.03 * sinp * wa;
  // momentum carries forward when it brakes, then rings out
  const tau = t - T2;
  p.tilt += 0.08 * dampedSin(tau, 1.5, 4.2);
  p.sy -= 0.05 * dampedSin(tau, 2.3, 5);
  p.sx += 0.035 * dampedSin(tau, 2.3, 5);
  p.antenna = -p.tilt * 1.3 + 0.18 * Math.sin(ph - 0.7) * wa + 0.32 * dampedSin(tau, 1.9, 3.2);

  // notices the spark
  const nt = seg(t, 3.1, 3.3, E.outBack);
  p.lookX = lerp(lerp(0.5, 0.15, seg(t, 2.8, 3.05)), 1.0, nt);
  p.lookY = lerp(0.04, -0.8, nt);
  p.eyeScale = 1 + 0.2 * Math.max(0, dampedCos(t - 3.12, 1.4, 5));
  p.mouth = -seg(t, 3.12, 3.22) * (1 - seg(t, 4.3, 4.6));
  p.smile = seg(t, 4.3, 4.6) * 0.6;
  p.blink = blinkAt(t, 1.35) + blinkAt(t, 2.98, 0.12);
  p.antenna += 0.2 * dampedSin(t - 3.12, 2.8, 4);
  p.bulb = seg(t, 3.14, 3.34) * (1 - seg(t, 4.6, 5.0));

  hop(p, t, 3.24, 3.36, 3.62, 34);

  // the network overtakes it: it shrinks away into the distance
  const k = seg(t, 3.7, 5.3, E.inOutCubic);
  p.s = lerp(1, 0.58, k);
  p.x += -150 * k;
  p.y += 40 * k;
  p.alpha = 1 - seg(t, 4.55, 5.3);
  return p;
}

export function finalePose(t: number): Pose {
  const p = basePose();
  const breathe = Math.sin((t - 27) * 3.1);
  p.sy = 1 + 0.012 * breathe;
  p.sx = 1 - 0.008 * breathe;
  // turns its head toward the landed light
  const tk = seg(t, 28.75, 29.0, E.outBack);
  p.lookX = lerp(0.35, -1.0, tk);
  p.lookY = lerp(0.0, 0.35, tk);
  p.tilt = -0.05 * seg(t, 28.75, 28.95, E.outCubic) + 0.06 * dampedSin(t - 28.75, 1.8, 4.5);
  p.sy -= 0.05 * dampedSin(t - 28.75, 2.5, 6);
  p.antenna = -0.35 * dampedSin(t - 28.75, 2.0, 3.0);
  p.blink = blinkAt(t, 29.0, 0.15);
  p.smile = seg(t, 29.08, 29.22, E.outCubic);
  p.bulb = seg(t, 29.1, 29.3) * (1 - seg(t, 29.9, 30));
  p.bulbCol = hex('#9ED4FF');
  hop(p, t, 29.14, 29.24, 29.52, 42);
  p.tilt += 0.04 * bump(t, 29.24, 29.52);
  p.alpha = 1 - seg(t, 29.74, 30.0, E.inOutSine);
  return p;
}

// ------------------------------------------------------------------ drawing
function bodyOutline(boil: number): number[] {
  const hw = BW / 2;
  // slightly different corner radii and a tilted top edge → hand-cut, asymmetric silhouette
  const corners = [
    { cx: -hw + 50, cy: -BH + 50, r: 50, a0: Math.PI, a1: Math.PI * 1.5 },
    { cx: hw - 38, cy: -BH + 6 + 38, r: 38, a0: Math.PI * 1.5, a1: TAU },
    { cx: hw - 29, cy: -29, r: 29, a0: 0, a1: Math.PI * 0.5 },
    { cx: -hw + 33, cy: -33, r: 33, a0: Math.PI * 0.5, a1: Math.PI },
  ];
  const raw: number[] = [];
  for (let c = 0; c < 4; c++) {
    const k = corners[c];
    for (let i = 0; i <= 8; i++) {
      const a = k.a0 + ((k.a1 - k.a0) * i) / 8;
      raw.push(k.cx + Math.cos(a) * k.r, k.cy + Math.sin(a) * k.r);
    }
    const nk = corners[(c + 1) % 4];
    const ex = k.cx + Math.cos(k.a1) * k.r;
    const ey = k.cy + Math.sin(k.a1) * k.r;
    const sx = nk.cx + Math.cos(nk.a0) * nk.r;
    const sy = nk.cy + Math.sin(nk.a0) * nk.r;
    for (let i = 1; i < 5; i++) raw.push(ex + ((sx - ex) * i) / 5, ey + ((sy - ey) * i) / 5);
  }
  const n = raw.length / 2;
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const a = (i - 1 + n) % n;
    const b = (i + 1) % n;
    const dx = raw[2 * b] - raw[2 * a];
    const dy = raw[2 * b + 1] - raw[2 * a + 1];
    const l = Math.hypot(dx, dy) || 1;
    const off = 1.5 * noise1(i * 0.37 + boil * 3.7, 77);
    out.push(raw[2 * i] + (dy / l) * off, raw[2 * i + 1] - (dx / l) * off);
  }
  return out;
}

const INK = C.ink;

function drawLegs(ctx: Ctx, p: Pose): void {
  const hipY = -LEG + 10 + p.lift - p.air;
  for (const side of [-1, 1]) {
    const dir = side === -1 ? 1 : -1;
    const hx = side * 38;
    const swing = Math.sin(p.legPhase) * dir * 20 * p.legAmp;
    const footLift = Math.max(0, Math.cos(p.legPhase) * dir) * 12 * p.legAmp;
    const fx = hx + swing + side * 4 * clamp(p.air / 30);
    const fy = -footLift - p.air * 0.82;
    const mx = (hx + fx) / 2 - dir * 4 * p.legAmp;
    const my = (hipY + fy) / 2;
    ctx.lineCap = 'round';
    ctx.strokeStyle = rgba(INK);
    ctx.lineWidth = 18;
    ctx.beginPath();
    ctx.moveTo(hx, hipY);
    ctx.quadraticCurveTo(mx, my, fx, fy - 2);
    ctx.stroke();
    ctx.strokeStyle = rgba(C.orangeDeep);
    ctx.lineWidth = 10.5;
    ctx.beginPath();
    ctx.moveTo(hx, hipY);
    ctx.quadraticCurveTo(mx, my, fx, fy - 2);
    ctx.stroke();
  }
}

function drawFace(ctx: Ctx, p: Pose): void {
  const fx = p.lookX * 18;
  const fy = -BH * 0.56 + p.lookY * 7;
  // cheeks
  ctx.fillStyle = rgba(hex('#F48C74'), 0.42 + 0.35 * p.smile);
  for (const s of [-1, 1]) {
    ctx.beginPath();
    ctx.ellipse(fx + s * 47, fy + 19 - 2 * p.smile, 13, 7.5, 0, 0, TAU);
    ctx.fill();
  }
  // eyes
  const open = 1 - p.blink;
  ctx.fillStyle = rgba(INK);
  ctx.strokeStyle = rgba(INK);
  for (const s of [-1, 1]) {
    const ex = fx + s * 27;
    const ey = fy;
    if (open > 0.2) {
      ctx.beginPath();
      ctx.ellipse(ex, ey, 8.6 * p.eyeScale, 12.2 * p.eyeScale * open, 0, 0, TAU);
      ctx.fill();
      ctx.fillStyle = 'rgba(255,250,240,0.95)';
      ctx.beginPath();
      ctx.arc(ex - 2.4 + p.lookX * 1.6, ey - 4.2 * open + p.lookY * 1.5, 2.9 * p.eyeScale, 0, TAU);
      ctx.fill();
      ctx.fillStyle = rgba(INK);
    } else {
      ctx.lineWidth = 3.6;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(ex - 8.5, ey);
      ctx.quadraticCurveTo(ex, ey + 4.5, ex + 8.5, ey);
      ctx.stroke();
    }
  }
  // mouth
  const mx = fx + 2;
  const my = fy + 25;
  if (p.mouth < -0.05) {
    const k = -p.mouth;
    ctx.beginPath();
    ctx.ellipse(mx, my + 2, 5 * k + 1.5, 7 * k + 1.5, 0, 0, TAU);
    ctx.fill();
  } else {
    const sm = p.smile;
    const w = 9 + 8 * sm;
    const dep = 3 + 9 * sm;
    ctx.lineWidth = 3.5;
    ctx.lineCap = 'round';
    if (sm > 0.55) {
      ctx.beginPath();
      ctx.moveTo(mx - w, my - 2);
      ctx.quadraticCurveTo(mx, my + dep * 1.7, mx + w, my - 2);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.save();
      ctx.clip();
      ctx.fillStyle = rgba(hex('#EF7F86'));
      ctx.beginPath();
      ctx.ellipse(mx + 1, my + dep * 0.95, w * 0.5, dep * 0.45, 0, 0, TAU);
      ctx.fill();
      ctx.restore();
    } else {
      ctx.beginPath();
      ctx.moveTo(mx - w, my);
      ctx.quadraticCurveTo(mx, my + dep, mx + w, my);
      ctx.stroke();
    }
  }
}

function drawAntenna(ctx: Ctx, p: Pose): void {
  const bx = 14;
  const by = -BH + 7;
  const L = 40;
  const a = p.antenna;
  const tx = bx + Math.sin(a) * L;
  const ty = by - Math.cos(a) * L;
  const cx = bx + Math.sin(a * 0.35) * L * 0.5;
  const cy = by - L * 0.55;
  ctx.strokeStyle = rgba(INK);
  ctx.lineWidth = 4.2;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(bx, by);
  ctx.quadraticCurveTo(cx, cy, tx, ty);
  ctx.stroke();
  if (p.bulb > 0.01) {
    ctx.globalCompositeOperation = 'source-over';
    glow(ctx, tx, ty, 46, p.bulbCol, 0.55 * p.bulb * p.alpha);
    ctx.globalAlpha = p.alpha;
  }
  ctx.fillStyle = rgba(mix(hex('#F7D774'), p.bulbCol === undefined ? C.warmWhite : mix(C.warmWhite, p.bulbCol, 0.4), p.bulb));
  ctx.beginPath();
  ctx.arc(tx, ty, 8.5, 0, TAU);
  ctx.fill();
  ctx.lineWidth = 3.4;
  ctx.stroke();
}

export function drawCharacter(ctx: Ctx, p: Pose, t: number): void {
  if (p.alpha <= 0.01) return;
  const boil = boilOf(t);
  ctx.save();
  ctx.globalAlpha = p.alpha;
  ctx.translate(p.x, p.y);
  ctx.scale(p.s, p.s);
  // contact shadow, shrinking as it leaves the ground
  const sk = 1 - clamp(p.air / 80) * 0.45 + p.lift * 0.004;
  ctx.fillStyle = rgba(INK, 0.14);
  ctx.beginPath();
  ctx.ellipse(4, 4, 80 * sk, 11 * sk, 0, 0, TAU);
  ctx.fill();
  ctx.rotate(p.tilt);
  drawLegs(ctx, p);
  ctx.save();
  ctx.translate(0, -LEG + 6 + p.lift - p.air);
  ctx.scale(p.sx, p.sy);
  const body = bodyOutline(boil);
  const n = body.length / 2;
  ctx.beginPath();
  tracePath(ctx, body, n, true);
  ctx.fillStyle = rgba(C.orange);
  ctx.fill();
  ctx.save();
  ctx.clip();
  ctx.fillStyle = rgba(C.orangeDeep, 0.32);
  ctx.beginPath();
  ctx.ellipse(12, 10, BW * 0.78, 42, -0.04, 0, TAU);
  ctx.fill();
  ctx.fillStyle = rgba(C.orangeLight, 0.6);
  ctx.beginPath();
  ctx.ellipse(-BW * 0.27, -BH * 0.83, 34, 13, -0.55, 0, TAU);
  ctx.fill();
  ctx.restore();
  ctx.lineWidth = 4.3;
  ctx.lineJoin = 'round';
  ctx.strokeStyle = rgba(INK);
  ctx.beginPath();
  tracePath(ctx, body, n, true);
  ctx.stroke();
  drawFace(ctx, p);
  drawAntenna(ctx, p);
  ctx.restore();
  ctx.restore();
}

/** A faint hand-drawn ground line with a couple of tufts. */
export function drawGround(ctx: Ctx, t: number, alpha: number): void {
  if (alpha <= 0.01) return;
  const boil = boilOf(t);
  ctx.save();
  const g = ctx.createLinearGradient(140, 0, 1780, 0);
  g.addColorStop(0, rgba(INK, 0));
  g.addColorStop(0.18, rgba(INK, 0.5 * alpha));
  g.addColorStop(0.82, rgba(INK, 0.5 * alpha));
  g.addColorStop(1, rgba(INK, 0));
  ctx.strokeStyle = g;
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  const pts: number[] = [];
  for (let x = 140; x <= 1780; x += 40) pts.push(x, GROUND_Y + 3 + noise1(x * 0.011 + boil * 0.9, 3) * 1.7);
  ctx.beginPath();
  tracePath(ctx, pts, pts.length / 2, false);
  ctx.stroke();
  ctx.strokeStyle = rgba(INK, 0.45 * alpha);
  ctx.lineWidth = 2.4;
  for (const [x, s] of [
    [430, 1],
    [1395, 0.8],
    [1530, 0.6],
  ]) {
    ctx.beginPath();
    for (let i = -1; i <= 1; i++) {
      const w = noise1(boil * 0.8 + i + x, 9) * 1.5;
      ctx.moveTo(x + i * 7 * s, GROUND_Y + 2);
      ctx.quadraticCurveTo(x + i * 9 * s + w, GROUND_Y - 8 * s, x + i * 13 * s + w, GROUND_Y - (14 - Math.abs(i) * 4) * s);
    }
    ctx.stroke();
  }
  ctx.fillStyle = rgba(INK, 0.16 * alpha);
  ctx.beginPath();
  ctx.ellipse(1250, GROUND_Y + 16, 9, 4, 0, 0, TAU);
  ctx.ellipse(575, GROUND_Y + 20, 6, 3, 0, 0, TAU);
  ctx.fill();
  ctx.restore();
}
