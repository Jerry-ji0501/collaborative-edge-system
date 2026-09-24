// Scene 03 / 04 — white beam → hand-drawn prism → spectrum → rainbow bands bend into concentric rings.
import { Ctx, boilOf, glow, ribbonPath, tracePath, wobbleCirclePath } from '../core/draw';
import { C, SPECTRUM, hex, mix, rgba } from '../core/color';
import { E, TAU, clamp, lerp, mod, seg, smoothInt } from '../core/math';
import { noise1, noise2 } from '../core/noise';
import { hash01 } from '../core/rng';
import { CENTER, ENTRY, EXIT, PRISM_V, RING_R, RING_W, THETA_S } from '../core/geometry';
import { applyCam, cosmicCam } from '../core/camera';

/** Ring radius incl. a gentle breath (shared with the sunflower so the hand-off is exact). */
export function ringR(i: number, t: number): number {
  return RING_R[i] * (1 + 0.02 * Math.sin(2.6 * t + 0.8 * i) * seg(t, 10.2, 10.8) * (1 - seg(t, 11.05, 11.8)));
}
/** Alternating ring spin, eased in via an integrated smoothstep (no velocity pop). */
export function ringRot(i: number, t: number): number {
  const dir = i % 2 === 0 ? 1 : -1;
  return dir * (0.55 + 0.1 * i) * smoothInt(t, 9.9, 10.7);
}
function ringSpan(i: number, t: number): number {
  const gap = 0.55 + 0.4 * hash01(i * 13 + 5);
  return lerp(TAU - gap, TAU + 0.03, seg(t, 10.45, 10.95, E.inOutSine));
}

// constant-curvature curve starting at (sx,sy) with heading phi
function curvePoint(sx: number, sy: number, phi: number, kappa: number, s: number, out: number[]): void {
  if (Math.abs(kappa) < 1e-7) {
    out[0] = sx + Math.cos(phi) * s;
    out[1] = sy + Math.sin(phi) * s;
    out[2] = phi;
    return;
  }
  const a = phi + kappa * s;
  out[0] = sx + (Math.sin(a) - Math.sin(phi)) / kappa;
  out[1] = sy + (Math.cos(phi) - Math.cos(a)) / kappa;
  out[2] = a;
}

function prismOutline(boil: number, sc: number): number[] {
  const pts: number[] = [];
  const cx = 960;
  const cy = 600;
  for (let e = 0; e < 3; e++) {
    const [ax, ay] = PRISM_V[e];
    const [bx, by] = PRISM_V[(e + 1) % 3];
    pts.push(cx + (ax - cx) * sc, cy + (ay - cy) * sc);
    for (let i = 1; i <= 14; i++) {
      const f = 0.05 + (0.9 * i) / 15;
      const x = ax + (bx - ax) * f;
      const y = ay + (by - ay) * f;
      const l = Math.hypot(bx - ax, by - ay);
      const off = 1.8 * noise1(i * 0.6 + e * 17 + boil * 2.3, 31);
      pts.push(cx + (x - cx + ((by - ay) / l) * off) * sc, cy + (y - cy - ((bx - ax) / l) * off) * sc);
    }
  }
  return pts;
}

const WHITE = C.white;
const LAV = hex('#E6E0FF');

function beamLine(ctx: Ctx, x0: number, y0: number, x1: number, y1: number, a: number, core = 3.6): void {
  if (a <= 0.003) return;
  const layers: [number, number, typeof WHITE][] = [
    [34, 0.05, LAV],
    [13, 0.18, LAV],
    [core, 0.95, WHITE],
  ];
  for (const [w, al, c] of layers) {
    ctx.globalAlpha = al * a;
    ctx.strokeStyle = rgba(c);
    ctx.lineWidth = w;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  }
}

const PT = [0, 0, 0];
const BAND = new Float32Array(600);
const BS = new Float32Array(300);

export function drawPrismAct(ctx: Ctx, t: number): void {
  if (t < 7.2 || t > 11.15) return;
  const boil = boilOf(t);
  const cam = cosmicCam(t);
  ctx.save();
  applyCam(ctx, cam);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  const it = t - 8.05; // impact time
  const lit = seg(t, 8.05, 8.2) * (1 - seg(t, 9.7, 10.4));

  // ------------------------------------------------ incoming white beam
  const beamOn = seg(t, 7.58, 7.72) * (1 - seg(t, 9.75, 10.4));
  if (beamOn > 0) {
    ctx.globalCompositeOperation = 'lighter';
    const frontP = seg(t, 7.55, 8.05, E.inQuad);
    const xf = lerp(-120, ENTRY.x, frontP);
    beamLine(ctx, -120, ENTRY.y, ENTRY.x, ENTRY.y, 0.35 * beamOn, 2.2);
    const flick = 1 + 0.06 * Math.sin(t * 37) * Math.sin(t * 13);
    beamLine(ctx, -120, ENTRY.y, xf, ENTRY.y, beamOn * flick, 3.8);
    if (frontP < 1) glow(ctx, xf, ENTRY.y, 60, hex('#FFF6E4'), 0.9 * beamOn);
  }

  // ------------------------------------------------ prism body
  const drawP = seg(t, 7.25, 8.0, E.inOutSine);
  const prismA = 1 - seg(t, 9.7, 10.45);
  if (drawP > 0 && prismA > 0) {
    const sc = lerp(1, 0.9, seg(t, 9.7, 10.45, E.inCubic));
    const pts = prismOutline(boil, sc);
    const n = pts.length / 2;
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = prismA * drawP;
    const g = ctx.createLinearGradient(960, 346, 960, 727);
    g.addColorStop(0, 'rgba(190,200,255,0.13)');
    g.addColorStop(1, 'rgba(255,255,255,0.04)');
    ctx.fillStyle = g;
    ctx.beginPath();
    tracePath(ctx, pts, n, true);
    ctx.fill();
    const flash = it > 0 ? Math.exp(-it * 2.4) * 0.45 + 0.1 * lit : 0;
    if (flash > 0.01) {
      ctx.globalCompositeOperation = 'lighter';
      ctx.globalAlpha = prismA * flash;
      ctx.fillStyle = 'rgba(235,230,255,0.5)';
      ctx.fill();
      ctx.globalCompositeOperation = 'source-over';
    }
    // depth: a faint back face + connecting edges make it a chunky glass block, not a flat icon
    {
      const ox = 38 * sc;
      const oy = -24 * sc;
      const V = PRISM_V.map(([x, y]) => [960 + (x - 960) * sc, 600 + (y - 600) * sc]);
      ctx.globalAlpha = prismA * drawP * 0.1;
      ctx.fillStyle = rgba(LAV);
      ctx.beginPath();
      ctx.moveTo(V[0][0], V[0][1]);
      ctx.lineTo(V[2][0], V[2][1]);
      ctx.lineTo(V[2][0] + ox, V[2][1] + oy);
      ctx.lineTo(V[0][0] + ox, V[0][1] + oy);
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = prismA * drawP * 0.05;
      ctx.beginPath();
      ctx.moveTo(V[0][0], V[0][1]);
      ctx.lineTo(V[0][0] + ox, V[0][1] + oy);
      ctx.lineTo(V[1][0] + ox, V[1][1] + oy);
      ctx.lineTo(V[1][0], V[1][1]);
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = prismA * 0.34;
      ctx.strokeStyle = rgba(LAV);
      ctx.lineWidth = 1.8;
      ctx.setLineDash([1400 * drawP, 1400]);
      ctx.translate(ox, oy);
      ctx.beginPath();
      tracePath(ctx, prismOutline(boil + 3, sc), n, true);
      ctx.stroke();
      ctx.translate(-ox, -oy);
      ctx.setLineDash([]);
      ctx.globalAlpha = prismA * 0.34 * drawP;
      ctx.beginPath();
      for (const [x, y] of V) {
        ctx.moveTo(x, y);
        ctx.lineTo(x + ox, y + oy);
      }
      ctx.stroke();
    }
    // glassy inner highlights
    ctx.globalAlpha = prismA * drawP * 0.4;
    ctx.strokeStyle = rgba(LAV);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(lerp(960, 740, 0.18) + 16, lerp(346, 727, 0.18) + 2);
    ctx.lineTo(lerp(960, 740, 0.62) + 16, lerp(346, 727, 0.62) + 2);
    ctx.moveTo(1004, 452);
    ctx.lineTo(1022, 484);
    ctx.stroke();
    // outline "drawn in" by a dash that grows
    const per = 1400;
    ctx.globalAlpha = prismA;
    ctx.setLineDash([per * drawP, per]);
    ctx.strokeStyle = rgba(mix(LAV, WHITE, clamp(flash)));
    ctx.lineWidth = 3.2;
    ctx.beginPath();
    tracePath(ctx, pts, n, true);
    ctx.stroke();
    ctx.globalAlpha = prismA * 0.35;
    ctx.lineWidth = 1.4;
    ctx.translate(2.5, -1.5);
    ctx.beginPath();
    tracePath(ctx, prismOutline(boil + 7, sc), n, true);
    ctx.stroke();
    ctx.translate(-2.5, 1.5);
    ctx.setLineDash([]);
  }

  // ------------------------------------------------ impact flash
  if (it > 0 && it < 0.9) {
    const k = it / 0.9;
    ctx.globalCompositeOperation = 'lighter';
    glow(ctx, ENTRY.x, ENTRY.y, 90 + 160 * k, hex('#FFF8EC'), 0.95 * Math.exp(-it * 4));
    ctx.globalAlpha = 0.6 * (1 - k);
    ctx.strokeStyle = rgba(LAV);
    ctx.lineWidth = 3 * (1 - k) + 0.6;
    ctx.beginPath();
    wobbleCirclePath(ctx, ENTRY.x, ENTRY.y, 24 + 420 * E.outCubic(k), 4, 8, boil, 72);
    ctx.stroke();
  }

  // ------------------------------------------------ light inside the glass
  if (it > 0) {
    ctx.globalCompositeOperation = 'lighter';
    const p = seg(t, 8.05, 8.22);
    const x = lerp(ENTRY.x, EXIT.x, p);
    beamLine(ctx, ENTRY.x, ENTRY.y, x, EXIT.y, 0.8 * lit, 5);
    if (t > 8.2) glow(ctx, EXIT.x, EXIT.y, 70, hex('#FFFFFF'), 0.7 * lit * (0.85 + 0.15 * Math.sin(t * 9)));
  }

  // ------------------------------------------------ spectrum → arcs → rings
  drawBands(ctx, t, boil);

  ctx.restore();
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
}

function drawBands(ctx: Ctx, t: number, boil: number): void {
  const Lfan = 1350 * seg(t, 8.2, 8.8, E.outCubic);
  if (Lfan <= 1) return;
  const fanP = seg(t, 8.2, 8.95, E.outCubic);
  const bend = seg(t, 9.45, 10.55, E.inOutCubic);
  const addK = 1 - seg(t, 10.1, 10.7);
  const paintA = 1 - seg(t, 11.05, 11.13);
  const beadA = seg(t, 10.2, 10.5) * (1 - seg(t, 10.95, 11.1));
  for (let i = 0; i < 7; i++) {
    const col = SPECTRUM[i];
    const aFull = 0.3 + i * 0.055;
    const a = lerp(0.41, aFull, fanP) + 0.005 * Math.sin(t * 2.3 + i * 1.1) + 0.004 * noise1(t * 1.3, i + 20);
    const R = ringR(i, t);
    const th = THETA_S + ringRot(i, t);
    const sx = lerp(EXIT.x, CENTER.x + R * Math.cos(th), bend);
    const sy = lerp(EXIT.y, CENTER.y + R * Math.sin(th), bend);
    const phi = lerp(a, th + Math.PI / 2, bend);
    const kappa = lerp(0, 1 / R, bend);
    let L = lerp(Lfan, R * ringSpan(i, t), bend);
    if (kappa > 1e-7) L = Math.min(L, (TAU + 0.03) / kappa);
    const n = Math.max(2, Math.min(280, Math.ceil(L / 12) + 1));
    for (let q = 0; q < n; q++) {
      const s = (L * q) / (n - 1);
      curvePoint(sx, sy, phi, kappa, s, PT);
      const off = 1.6 * noise2(s * 0.015, boil * 0.5 + i * 7) * bend;
      BAND[2 * q] = PT[0] - Math.sin(PT[2]) * off;
      BAND[2 * q + 1] = PT[1] + Math.cos(PT[2]) * off;
      BS[q] = s;
    }
    const width = (q: number): number => {
      const s = BS[q];
      const wf = 2.5 + s * 0.05 * fanP;
      const wr = RING_W * (1 + 0.1 * noise1(s * 0.012 + i * 3.3, 7));
      return lerp(wf, wr, bend);
    };
    // luminous pass (light on dark)
    if (addK > 0.001) {
      ctx.globalCompositeOperation = 'lighter';
      const ex = sx + Math.cos(phi) * L * 0.95;
      const ey = sy + Math.sin(phi) * L * 0.95;
      const g = ctx.createLinearGradient(sx, sy, ex, ey);
      g.addColorStop(0, rgba(col, 0.62));
      g.addColorStop(0.55, rgba(col, lerp(0.42, 0.62, bend)));
      g.addColorStop(1, rgba(col, lerp(0.0, 0.62, bend)));
      ctx.fillStyle = g;
      ctx.globalAlpha = addK * paintA;
      ctx.beginPath();
      ribbonPath(ctx, BAND, n, width);
      ctx.fill();
      ctx.globalAlpha = 0.11 * addK * paintA;
      ctx.beginPath();
      ribbonPath(ctx, BAND, n, (q) => width(q) * 2.6 + 6);
      ctx.fill();
    }
    // painted pass (flat ink colour, ready to become petals)
    if (addK < 0.999) {
      ctx.globalCompositeOperation = 'source-over';
      ctx.globalAlpha = (1 - addK) * paintA;
      ctx.fillStyle = rgba(col);
      ctx.beginPath();
      ribbonPath(ctx, BAND, n, width);
      ctx.fill();
    }
    // beads riding each ring make the counter-rotation readable
    if (beadA > 0) {
      for (let k = 0; k < 2; k++) {
        const s = mod(i * 333 + (k * L) / 2 + t * 150 * (i % 2 ? -1 : 1), L);
        curvePoint(sx, sy, phi, kappa, s, PT);
        ctx.globalCompositeOperation = 'lighter';
        glow(ctx, PT[0], PT[1], 22, mix(col, WHITE, 0.6), 0.8 * beadA);
        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = beadA;
        ctx.fillStyle = rgba(mix(col, WHITE, 0.75));
        ctx.beginPath();
        ctx.arc(PT[0], PT[1], 4, 0, TAU);
        ctx.fill();
      }
    }
  }
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1;
}
