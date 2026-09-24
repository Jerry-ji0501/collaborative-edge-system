// Scene 09 — the light pillar lies down, sinks into a lunar horizon, and a stylised Earth rises.
import { Ctx, boilOf, glow, sparkle, tracePath, wobbleCirclePath } from '../core/draw';
import { RGB, hex, rgba } from '../core/color';
import { E, TAU, lerp, seg } from '../core/math';
import { noise1 } from '../core/noise';
import { RNG } from '../core/rng';
import { Cam, applyCam, toScreen } from '../core/camera';

export const EARTH_X = 1060;
export const RE = 245;

export function horizonY(x: number): number {
  const dx = (x - 960) / 960;
  return 792 + 26 * dx * dx + 14 * Math.sin(x * 0.0042 + 0.8) + 7 * Math.sin(x * 0.011 + 2.1) + 3 * noise1(x * 0.02, 5);
}

export function earthCY(t: number): number {
  return lerp(1060, 512, seg(t, 23.9, 27.2, E.outSine));
}

function glintWorld(t: number): [number, number] {
  return [EARTH_X - 0.36 * RE, earthCY(t) - 0.3 * RE];
}

/** Gentle push during the earthrise, then a fast dive into the sun-glint on the ocean. */
export function earthCam(t: number): Cam {
  const k = seg(t, 24.0, 26.9, E.inOutSine);
  const base: Cam = { x: 960 + 30 * k, y: 540 - 14 * k, zoom: 1 + 0.05 * k, rot: 0 };
  const z = seg(t, 26.9, 27.42, E.inCubic);
  if (z <= 0) return base;
  const G = glintWorld(t);
  const s0 = toScreen(base, G[0], G[1]);
  const zoom = base.zoom * Math.pow(11 / base.zoom, z);
  const m = E.inOutCubic(z);
  const sx = lerp(s0[0], 960, m);
  const sy = lerp(s0[1], 540, m);
  return { x: G[0] - (sx - 960) / zoom, y: G[1] - (sy - 540) / zoom, zoom, rot: 0 };
}

// ------------------------------------------------ sphere blobs (continents, clouds)
interface Blob {
  p: Float32Array; // xyz triples
  n: number;
}
function makeBlob(lat: number, lon: number, rad: number, seed: number, n = 48): Blob {
  const c = [Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon)];
  // e1 = normalize(up × c), e2 = c × e1
  let e1 = [c[2], 0, -c[0]];
  const l = Math.hypot(e1[0], e1[2]) || 1;
  e1 = [e1[0] / l, 0, e1[2] / l];
  const e2 = [c[1] * e1[2] - c[2] * e1[1], c[2] * e1[0] - c[0] * e1[2], c[0] * e1[1] - c[1] * e1[0]];
  const p = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const phi = (i / n) * TAU;
    const rho = rad * (1 + 0.35 * noise1(phi * 1.5 + seed * 10, 3) + 0.15 * noise1(phi * 5 + seed, 4));
    const cr = Math.cos(rho);
    const sr = Math.sin(rho);
    const cp = Math.cos(phi);
    const sp = Math.sin(phi);
    for (let k = 0; k < 3; k++) p[i * 3 + k] = c[k] * cr + (e1[k] * cp + e2[k] * sp) * sr;
  }
  return { p, n };
}
const LAND: Blob[] = [
  [0.35, 0.2, 0.45],
  [-0.3, 0.95, 0.36],
  [0.62, 2.2, 0.5],
  [-0.5, 2.8, 0.33],
  [0.12, 3.9, 0.55],
  [-0.22, 5.1, 0.42],
  [0.85, 4.6, 0.3],
].map(([a, b, r], i) => makeBlob(a, b, r, i + 1));
const CLOUDS: Blob[] = (() => {
  const r = new RNG(61);
  return Array.from({ length: 11 }, (_, i) => makeBlob(r.range(-1.0, 1.0), r.range(0, TAU), r.range(0.12, 0.26), 40 + i, 36));
})();
const LAND_COL: RGB[] = ['#79AE6A', '#C9B27A', '#6FA565', '#8DB871', '#D2B97F', '#77AA66', '#E8EEF4'].map(hex);

const PJ = new Float32Array(200);
function projectBlob(b: Blob, lam: number, tilt: number, cx: number, cy: number, R: number): number {
  const cl = Math.cos(lam);
  const sl = Math.sin(lam);
  const ct = Math.cos(tilt);
  const st = Math.sin(tilt);
  let front = 0;
  for (let i = 0; i < b.n; i++) {
    const X = b.p[i * 3];
    const Y = b.p[i * 3 + 1];
    const Z = b.p[i * 3 + 2];
    const x1 = X * cl + Z * sl;
    const z1 = -X * sl + Z * cl;
    let x2 = x1 * ct - Y * st;
    let y2 = x1 * st + Y * ct;
    if (z1 > 0) front++;
    else {
      const l = Math.hypot(x2, y2) || 1;
      x2 /= l;
      y2 /= l;
    }
    PJ[2 * i] = cx + x2 * R;
    PJ[2 * i + 1] = cy - y2 * R;
  }
  return front;
}

function drawEarth(ctx: Ctx, t: number, cx: number, cy: number, boil: number): void {
  const R = RE;
  // atmosphere halo
  ctx.globalCompositeOperation = 'lighter';
  glow(ctx, cx, cy, R * 1.55, hex('#4F9BFF'), 0.42);
  glow(ctx, cx, cy, R * 1.22, hex('#8CC6FF'), 0.25);
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1;
  ctx.save();
  ctx.beginPath();
  wobbleCirclePath(ctx, cx, cy, R, 0.9, 51, boil, 120);
  const og = ctx.createRadialGradient(cx - 0.35 * R, cy - 0.4 * R, 0, cx, cy, R * 1.35);
  og.addColorStop(0, '#83C3F5');
  og.addColorStop(0.45, '#3F85DB');
  og.addColorStop(1, '#1A48A0');
  ctx.fillStyle = og;
  ctx.fill();
  ctx.clip();
  const lam = 0.12 * t + 0.4;
  const tilt = 0.35;
  LAND.forEach((b, i) => {
    const f = projectBlob(b, lam, tilt, cx, cy, R);
    if (f === 0) return;
    ctx.beginPath();
    tracePath(ctx, PJ, b.n, true);
    ctx.fillStyle = rgba(LAND_COL[i]);
    ctx.fill();
    ctx.strokeStyle = 'rgba(40,80,60,0.35)';
    ctx.lineWidth = 1.6;
    ctx.stroke();
  });
  // clouds: soft blobs + a few wind-streaks
  CLOUDS.forEach((b) => {
    const f = projectBlob(b, lam * 1.18 + 0.5, tilt, cx, cy, R);
    if (f === 0) return;
    ctx.beginPath();
    tracePath(ctx, PJ, b.n, true);
    ctx.fillStyle = 'rgba(248,252,255,0.78)';
    ctx.fill();
  });
  ctx.strokeStyle = 'rgba(248,252,255,0.55)';
  ctx.lineCap = 'round';
  for (let s = 0; s < 4; s++) {
    const lat = -0.6 + s * 0.42;
    ctx.lineWidth = 5 - s * 0.6;
    ctx.beginPath();
    let started = false;
    for (let q = 0; q <= 24; q++) {
      const lon = s * 1.7 + (q / 24) * 1.5 - lam * 1.3;
      const X = Math.cos(lat) * Math.cos(lon);
      const Y = Math.sin(lat) + 0.03 * Math.sin(q * 0.8 + s);
      const Z = Math.cos(lat) * Math.sin(lon);
      if (Z <= 0.05) {
        started = false;
        continue;
      }
      const x2 = X * Math.cos(tilt) - Y * Math.sin(tilt);
      const y2 = X * Math.sin(tilt) + Y * Math.cos(tilt);
      if (!started) ctx.moveTo(cx + x2 * R, cy - y2 * R);
      else ctx.lineTo(cx + x2 * R, cy - y2 * R);
      started = true;
    }
    ctx.stroke();
  }
  // night side
  const sg = ctx.createLinearGradient(cx - 0.85 * R, cy - 0.3 * R, cx + 0.9 * R, cy + 0.32 * R);
  sg.addColorStop(0, 'rgba(5,8,28,0)');
  sg.addColorStop(0.4, 'rgba(5,8,28,0.03)');
  sg.addColorStop(0.6, 'rgba(5,8,28,0.5)');
  sg.addColorStop(0.78, 'rgba(5,8,28,0.84)');
  sg.addColorStop(1, 'rgba(5,8,28,0.93)');
  ctx.fillStyle = sg;
  ctx.fillRect(cx - R, cy - R, 2 * R, 2 * R);
  // soft sheen on the day side
  const hl = ctx.createRadialGradient(cx - 0.42 * R, cy - 0.45 * R, 0, cx - 0.42 * R, cy - 0.45 * R, R * 0.75);
  hl.addColorStop(0, 'rgba(255,255,255,0.28)');
  hl.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = hl;
  ctx.fillRect(cx - R, cy - R, 2 * R, 2 * R);
  // atmospheric rim inside the limb
  const rg = ctx.createRadialGradient(cx, cy, R * 0.84, cx, cy, R);
  rg.addColorStop(0, 'rgba(160,215,255,0)');
  rg.addColorStop(1, 'rgba(170,220,255,0.5)');
  ctx.fillStyle = rg;
  ctx.fillRect(cx - R, cy - R, 2 * R, 2 * R);
  ctx.restore();
  // lit limb + hand-drawn outline
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'rgba(214,238,255,0.85)';
  ctx.lineWidth = 3.5;
  ctx.beginPath();
  ctx.arc(cx, cy, R - 1, Math.PI * 0.82, Math.PI * 1.62);
  ctx.stroke();
  ctx.strokeStyle = 'rgba(150,200,255,0.35)';
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  wobbleCirclePath(ctx, cx, cy, R + 2, 1.2, 52, boil, 120);
  ctx.stroke();
}

// ------------------------------------------------ moon
const CRATERS = (() => {
  const r = new RNG(2024);
  return Array.from({ length: 15 }, () => {
    const y = r.range(835, 1065);
    const k = (y - 830) / 235;
    const rx = lerp(22, 125, k) * r.range(0.7, 1.2);
    return { x: r.range(-40, 1960), y, rx, ry: rx * lerp(0.17, 0.28, k) };
  });
})();

function drawMoon(ctx: Ctx, t: number, top: (x: number) => number, a: number, detail: number, boil: number): void {
  if (a <= 0.01) return;
  ctx.globalAlpha = a;
  ctx.globalCompositeOperation = 'source-over';
  const g = ctx.createLinearGradient(0, 760, 0, 1100);
  g.addColorStop(0, '#2A3047');
  g.addColorStop(0.25, '#1A1E30');
  g.addColorStop(1, '#090B13');
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.moveTo(-80, top(-80));
  for (let x = -60; x <= 2000; x += 20) ctx.lineTo(x, top(x) + noise1(x * 0.05 + boil * 0.7, 8) * 0.6);
  ctx.lineTo(2000, 1300);
  ctx.lineTo(-80, 1300);
  ctx.closePath();
  ctx.fill();
  if (detail > 0.01) {
    for (const c of CRATERS) {
      ctx.globalAlpha = a * detail;
      ctx.fillStyle = 'rgba(12,14,24,0.75)';
      ctx.beginPath();
      ctx.ellipse(c.x, c.y, c.rx, c.ry, 0, 0, TAU);
      ctx.fill();
      ctx.strokeStyle = 'rgba(78,90,130,0.55)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(c.x, c.y, c.rx, c.ry, 0, Math.PI * 1.05, Math.PI * 1.95);
      ctx.stroke();
      ctx.strokeStyle = 'rgba(52,60,92,0.5)';
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.ellipse(c.x, c.y + c.ry * 0.25, c.rx * 0.8, c.ry * 0.6, 0, Math.PI * 0.15, Math.PI * 0.85);
      ctx.stroke();
    }
  }
  // earthlight along the horizon
  ctx.globalCompositeOperation = 'lighter';
  for (const [w, al] of [
    [12, 0.05],
    [3.5, 0.22],
    [1.3, 0.5],
  ] as [number, number][]) {
    ctx.globalAlpha = a * al;
    ctx.strokeStyle = '#9FC4FF';
    ctx.lineWidth = w;
    ctx.beginPath();
    ctx.moveTo(-80, top(-80));
    for (let x = -60; x <= 2000; x += 20) ctx.lineTo(x, top(x));
    ctx.stroke();
  }
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1;
  void t;
}

// ------------------------------------------------ light pillar → horizon
function drawLightBar(ctx: Ctx, t: number): void {
  if (t < 22.74 || t > 24.35) return;
  const grow = seg(t, 22.74, 23.0, E.outExpo);
  const rotP = seg(t, 23.05, 23.45, E.inOutCubic);
  const sinkP = seg(t, 23.4, 23.95, E.inOutCubic);
  const flash = t > 22.78 ? Math.exp(-(t - 22.78) * 5) : 0;
  const inten = 1 - seg(t, 23.85, 24.35);
  const layers: [number, number, string][] = [
    [70, 0.07, '#FFE7B8'],
    [26, 0.22, '#FFF4E0'],
    [7, 0.95, '#FFFFFF'],
  ];
  ctx.globalCompositeOperation = 'lighter';
  ctx.lineCap = 'round';
  if (sinkP <= 0) {
    const ang = ((1 - rotP) * Math.PI) / 2;
    const half = lerp(640 * grow, 1300, rotP);
    const dx = Math.cos(ang) * half;
    const dy = Math.sin(ang) * half;
    for (const [w, a, c] of layers) {
      ctx.globalAlpha = a * inten;
      ctx.strokeStyle = c;
      ctx.lineWidth = w * (1 + flash);
      ctx.beginPath();
      ctx.moveTo(960 - dx, 540 - dy);
      ctx.lineTo(960 + dx, 540 + dy);
      ctx.stroke();
    }
  } else {
    for (const [w, a, c] of layers) {
      ctx.globalAlpha = a * inten;
      ctx.strokeStyle = c;
      ctx.lineWidth = w * lerp(1, 0.45, sinkP);
      ctx.beginPath();
      for (let x = -60; x <= 1980; x += 30) {
        const y = lerp(540, horizonY(x), sinkP);
        if (x === -60) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }
  if (flash > 0.01) {
    glow(ctx, 960, 540, 1100, hex('#FFF2DC'), 0.45 * flash, 0.8);
    glow(ctx, 960, 540, 260, hex('#FFFFFF'), 0.9 * flash);
  }
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1;
}

export function drawEarthAct(ctx: Ctx, t: number): void {
  if (t < 22.74 || t > 27.58) return;
  const boil = boilOf(t);
  const cam = earthCam(t);
  const actA = 1 - seg(t, 27.4, 27.56);
  ctx.save();
  applyCam(ctx, cam);
  ctx.globalAlpha = actA;
  // blue glow welling up behind the horizon
  const bg = seg(t, 23.7, 24.8, E.inOutSine);
  if (bg > 0) {
    ctx.globalCompositeOperation = 'lighter';
    const g = ctx.createRadialGradient(EARTH_X, 900, 0, EARTH_X, 900, 820);
    g.addColorStop(0, 'rgba(58,116,255,0.34)');
    g.addColorStop(0.5, 'rgba(58,116,255,0.12)');
    g.addColorStop(1, 'rgba(58,116,255,0)');
    ctx.globalAlpha = bg * actA;
    ctx.fillStyle = g;
    ctx.fillRect(-200, 0, 2400, 1200);
    ctx.globalCompositeOperation = 'source-over';
  }
  if (t > 23.9) {
    ctx.save();
    ctx.globalAlpha = actA;
    drawEarth(ctx, t, EARTH_X, earthCY(t), boil);
    ctx.restore();
  }
  const sinkP = seg(t, 23.4, 23.95, E.inOutCubic);
  if (sinkP > 0) {
    const top = (x: number): number => lerp(540, horizonY(x), sinkP);
    drawMoon(ctx, t, top, seg(t, 23.4, 23.9) * actA, seg(t, 23.8, 24.5) * actA, boil);
  }
  // sun-glint on the ocean — the point we will fall toward
  const gl = seg(t, 26.45, 26.8, E.outBack);
  if (gl > 0) {
    const [gx, gy] = glintWorld(t);
    const z = cam.zoom;
    ctx.globalCompositeOperation = 'lighter';
    glow(ctx, gx, gy, (36 + 26 * seg(t, 26.9, 27.4)) * gl, hex('#DDF1FF'), 0.9 * actA);
    ctx.globalCompositeOperation = 'source-over';
    sparkle(ctx, gx, gy, (22 * gl * (1 + 0.15 * Math.sin(t * 11))) / Math.pow(z, 0.25), t * 0.7, hex('#FFFFFF'), actA);
    sparkle(ctx, gx, gy, (12 * gl) / Math.pow(z, 0.25), -t * 0.5 + 0.78, hex('#FFFFFF'), 0.8 * actA);
  }
  ctx.restore();
  drawLightBar(ctx, t);
  ctx.globalAlpha = 1;
}
