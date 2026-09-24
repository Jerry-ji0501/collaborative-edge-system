// Scene 05 — the rainbow rings grow petals, the inner rings break into golden-angle seeds.
// Then (13.8s+) petals peel away as glowing stars while the seed field unwinds into a galaxy.
import { Ctx, boilOf, polyPath, ribbonPath, softGlow, wobbleCirclePath } from '../core/draw';
import { C, RGB, SPECTRUM, hex, mix, ramp, rgba } from '../core/color';
import { E, TAU, clamp, lerp, seg, smoothInt, smoothstep, smootherstep } from '../core/math';
import { noise1 } from '../core/noise';
import { RNG, hash01 } from '../core/rng';
import { CENTER, RING_W } from '../core/geometry';
import { applyCam, cosmicCam, diskView, projectDisk } from '../core/camera';
import { darkness } from '../core/timeline';
import { ringR } from './prism';

export const N_SEEDS = 520;
const GOLDEN = Math.PI * (3 - Math.sqrt(5));
const SEED_R = 138;

export interface Seed {
  r: number;
  phi: number;
  ring: number;
  size: number;
  col: RGB;
  colIdx: number;
  u: number;
  launch: number;
}

export const SEED_PAL: RGB[] = ['#3A2116', '#4A2A1A', '#5C3520', '#6E3F22', '#874C22', '#A65A24'].map(hex);

export const SEEDS: Seed[] = [];
for (let k = 0; k < N_SEEDS; k++) {
  const r = 6.05 * Math.sqrt(k + 0.6);
  const u = r / SEED_R;
  const colIdx = clamp(Math.floor(u * 4 + hash01(k * 3 + 1) * 2), 0, 5);
  SEEDS.push({
    r,
    phi: k * GOLDEN,
    ring: r < 95 ? 6 : 5,
    size: 2.7 + 2.3 * u,
    col: SEED_PAL[colIdx],
    colIdx,
    u,
    launch: 13.85 + 0.55 * (1 - u) + 0.2 * hash01(k * 7 + 3),
  });
}

/** Flower spin: lazy drift, then a sharp acceleration as it becomes a galaxy. */
export function flowerRot(t: number): number {
  return 0.09 * (t - 11.0) + 2.3 * smootherstep(13.75, 15.4, t);
}

export function seedMorph(k: number, t: number): number {
  const M = seg(t, 11.05, 12.45);
  const s = SEEDS[k];
  return E.inOutCubic(clamp((M - 0.12 - 0.3 * (1 - s.u)) / 0.58));
}

/** Disk-plane polar position of seed k while it is a flower seed. */
export function seedPolar(k: number, t: number, out: [number, number]): void {
  const s = SEEDS[k];
  const e = seedMorph(k, t);
  out[0] = lerp(ringR(s.ring, t), s.r, e);
  out[1] = s.phi + flowerRot(t) + 0.7 * Math.sin(Math.PI * e) * (s.ring === 6 ? 1 : -1);
}

interface Layer {
  ring: number;
  n: number;
  off: number;
  tip: number;
  base: number;
  width: number;
  col: RGB;
  colTip: RGB;
  outline: number;
}
const LAYERS: Layer[] = [
  { ring: 3, n: 13, off: 0.25, tip: 322, base: 138, width: 40, col: hex('#4F8440'), colTip: hex('#86B45A'), outline: 0.75 },
  { ring: 0, n: 21, off: 0.0, tip: 404, base: 136, width: 64, col: hex('#DB8423'), colTip: hex('#EFAE3A'), outline: 0.9 },
  { ring: 1, n: 21, off: 0.5, tip: 380, base: 136, width: 60, col: hex('#EE9F2B'), colTip: hex('#FAD257'), outline: 0.95 },
  { ring: 2, n: 34, off: 0.25, tip: 180, base: 140, width: 17, col: hex('#C9661E'), colTip: hex('#F2B53A'), outline: 0.6 },
];
interface PR {
  tip: number;
  w: number;
  jit: number;
  bend: number;
  rnd: number;
}
const PRS: PR[][] = LAYERS.map((L, li) => {
  const r = new RNG(900 + li);
  return Array.from({ length: L.n }, () => ({
    tip: r.range(0.93, 1.05),
    w: r.range(0.88, 1.08),
    jit: r.range(-0.035, 0.035),
    bend: r.range(-0.12, 0.12),
    rnd: r.next(),
  }));
});

const STAR_RAMP: RGB[] = ['#F6C443', '#F59A45', '#E0679A', '#A88CFF', '#D8E6FF'].map(hex);
const INK = C.ink;
const K = 14;
const POLY = new Float32Array((K + 1) * 4 + 16);

function petalDetach(li: number, j: number, t: number): number {
  const L = LAYERS[li];
  const pr = PRS[li][j];
  if (li === 0 || li === 3) return 0;
  const frac = ((j + L.off) / L.n) % 1;
  const td = 13.82 + 0.45 * frac + 0.5 * pr.rnd + (li === 1 ? 0.1 : 0);
  return E.inOutSine(clamp((t - td) / 0.85));
}

function petalPoly(li: number, j: number, t: number, M: number, boil: number): { cnt: number; psi: number; m: number; d: number; mid: number } {
  const L = LAYERS[li];
  const pr = PRS[li][j];
  const frac = ((j + L.off) / L.n) % 1;
  // a gentle travelling stagger, so the ring "unzips" into petals around the circle
  const m = clamp((M - 0.14 * (frac * 0.75 + 0.25 * pr.rnd)) / 0.86);
  const mE = E.inOutCubic(m);
  const tipK = E.inOutCubic(clamp(m / 0.62));
  const baseK = E.inOutCubic(clamp((m - 0.28) / 0.72));
  const d = petalDetach(li, j, t);
  // sepals and florets shrink away as the galaxy forms
  const shrink = li === 0 || li === 3 ? seg(t, 13.8, 14.5, E.inCubic) : 0;
  const R = ringR(L.ring, t);
  let psi = flowerRot(t) + ((j + L.off) * TAU) / L.n + pr.jit * mE + (0.9 + 0.8 * (pr.rnd - 0.5)) * d;
  let rIn = lerp(R - RING_W / 2, L.base, baseK);
  let rOut = lerp(R + RING_W / 2, L.tip * pr.tip, tipK);
  if (shrink > 0) rOut = lerp(rOut, rIn + 4, shrink);
  const mid0 = (rIn + rOut) / 2;
  if (d > 0) {
    const len = (rOut - rIn) * (1 - 0.95 * smoothstep(0, 0.65, d));
    const mid = mid0 + (220 + 420 * pr.rnd) * d;
    rIn = mid - len / 2;
    rOut = mid + len / 2;
  }
  const hwRing = (Math.PI / L.n) * 1.02;
  const wScale = (1 - 0.95 * smoothstep(0, 0.65, d)) * (1 - shrink);
  // the tip pinches first, the base lets go last
  const shapeK = (v: number): number => {
    const f = clamp(m * 1.7 - (1 - v) * 0.7);
    return f * f * (3 - 2 * f);
  };
  let c = 0;
  // left edge base→tip
  for (let q = 0; q <= K; q++) {
    const v = q / K;
    const r = lerp(rIn, rOut, v);
    const prof = Math.pow(Math.sin(Math.PI * Math.pow(v, 0.72)), 0.85);
    const hwP = ((L.width * pr.w * 0.5 * prof) / Math.max(r, 20)) * wScale;
    const hw = lerp(hwRing, hwP, shapeK(v));
    const a = psi + pr.bend * v * v * mE;
    const wob = 0.9 * noise1(q * 0.9 + boil * 2.1 + j * 5, li) * mE;
    POLY[2 * c] = Math.cos(a - hw) * (r + wob);
    POLY[2 * c + 1] = Math.sin(a - hw) * (r + wob);
    c++;
  }
  // tip point
  {
    const a = psi + pr.bend * mE;
    POLY[2 * c] = Math.cos(a) * rOut;
    POLY[2 * c + 1] = Math.sin(a) * rOut;
    c++;
  }
  for (let q = K; q >= 0; q--) {
    const v = q / K;
    const r = lerp(rIn, rOut, v);
    const prof = Math.pow(Math.sin(Math.PI * Math.pow(v, 0.72)), 0.85);
    const hwP = ((L.width * pr.w * 0.5 * prof) / Math.max(r, 20)) * wScale;
    const hw = lerp(hwRing, hwP, shapeK(v));
    const a = psi + pr.bend * v * v * mE;
    const wob = 0.9 * noise1(q * 0.9 + boil * 2.1 + j * 5 + 40, li) * mE;
    POLY[2 * c] = Math.cos(a + hw) * (r + wob);
    POLY[2 * c + 1] = Math.sin(a + hw) * (r + wob);
    c++;
  }
  // base cap
  POLY[2 * c] = Math.cos(psi) * rIn;
  POLY[2 * c + 1] = Math.sin(psi) * rIn;
  c++;
  psi = psi + pr.bend * 0.5 * mE;
  return { cnt: c, psi, m, d, mid: (rIn + rOut) / 2 };
}

function drawStem(ctx: Ctx, t: number): void {
  const g = seg(t, 12.05, 12.95, E.outCubic);
  const a = 1 - seg(t, 13.8, 14.3);
  if (g <= 0 || a <= 0) return;
  const P = [
    [0, 120],
    [-10, 330],
    [-60, 520],
    [-40, 800],
  ];
  const n = 32;
  const pts: number[] = [];
  const cnt = Math.max(2, Math.round(n * g));
  for (let i = 0; i < cnt; i++) {
    const u = (i / (n - 1)) * 1;
    const w0 = (1 - u) ** 3;
    const w1 = 3 * (1 - u) ** 2 * u;
    const w2 = 3 * (1 - u) * u * u;
    const w3 = u ** 3;
    pts.push(
      CENTER.x + w0 * P[0][0] + w1 * P[1][0] + w2 * P[2][0] + w3 * P[3][0],
      CENTER.y + w0 * P[0][1] + w1 * P[1][1] + w2 * P[2][1] + w3 * P[3][1],
    );
  }
  ctx.globalAlpha = a;
  ctx.beginPath();
  ribbonPath(ctx, pts, cnt, (i) => lerp(26, 18, i / (n - 1)) * (i === cnt - 1 && g < 1 ? 0.6 : 1));
  ctx.fillStyle = rgba(hex('#6E9A4A'));
  ctx.fill();
  ctx.strokeStyle = rgba(INK);
  ctx.lineWidth = 2.6;
  ctx.stroke();
  // one leaf
  const ls = seg(t, 12.5, 12.95, E.outBack);
  if (ls > 0 && cnt > 18) {
    const lx = pts[2 * 17];
    const ly = pts[2 * 17 + 1];
    ctx.save();
    ctx.translate(lx - 8, ly);
    ctx.rotate(-2.55 + 0.05 * Math.sin(t * 1.7));
    ctx.scale(ls, ls);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.quadraticCurveTo(60, -42, 135, -4);
    ctx.quadraticCurveTo(70, 34, 0, 0);
    ctx.fillStyle = rgba(hex('#7DAA55'));
    ctx.fill();
    ctx.stroke();
    ctx.globalAlpha = a * 0.5;
    ctx.beginPath();
    ctx.moveTo(6, 0);
    ctx.quadraticCurveTo(60, -6, 118, -4);
    ctx.stroke();
    ctx.restore();
  }
  ctx.globalAlpha = 1;
}

const SP: [number, number] = [0, 0];
const WP: [number, number] = [0, 0];

export function drawFlower(ctx: Ctx, t: number): void {
  if (t < 11.04 || t > 17.9) return;
  const cam = cosmicCam(t);
  const { iota, beta } = diskView(t);
  const M = seg(t, 11.05, 12.45);
  const cm = smoothstep(0.05, 0.7, M);
  const boil = boilOf(t);
  const dark = darkness(t);
  const galFade = seg(t, 13.9, 14.8);

  ctx.save();
  applyCam(ctx, cam);
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  drawStem(ctx, t);

  ctx.save();
  ctx.translate(CENTER.x, CENTER.y);
  ctx.rotate(beta);
  ctx.scale(1, iota);

  // ------------------------------------------------ sepals, back petals, front petals
  const bloom = seg(t, 12.2, 12.8);
  const breathe = 1 + 0.014 * Math.sin(2.1 * (t - 11)) * bloom * (1 - galFade);
  ctx.scale(breathe, breathe);
  for (let li = 0; li < 3; li++) {
    const L = LAYERS[li];
    const gr = ctx.createRadialGradient(0, 0, L.base, 0, 0, L.tip);
    gr.addColorStop(0, rgba(mix(SPECTRUM[L.ring], L.col, cm)));
    gr.addColorStop(1, rgba(mix(SPECTRUM[L.ring], L.colTip, cm)));
    for (let j = 0; j < L.n; j++) {
      const p = petalPoly(li, j, t, M, boil);
      if (p.d >= 0.999) continue;
      const alpha = (1 - smoothstep(0.45, 0.75, p.d)) * (li === 0 ? 1 - seg(t, 13.9, 14.5) : 1);
      if (alpha <= 0.01) continue;
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      polyPath(ctx, POLY, p.cnt);
      ctx.fillStyle = p.d > 0.02 ? rgba(ramp(STAR_RAMP, p.d * 0.9)) : gr;
      ctx.fill();
      const ol = smoothstep(0.42, 0.95, p.m) * L.outline * (1 - smoothstep(0.1, 0.5, p.d));
      if (ol > 0.01) {
        ctx.globalAlpha = alpha * ol;
        ctx.strokeStyle = rgba(INK);
        ctx.lineWidth = 2.5;
        ctx.stroke();
        if (li > 0) {
          ctx.globalAlpha = alpha * ol * 0.28;
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.moveTo(Math.cos(p.psi) * (L.base + 24), Math.sin(p.psi) * (L.base + 24));
          ctx.lineTo(Math.cos(p.psi) * p.mid * 1.25, Math.sin(p.psi) * p.mid * 1.25);
          ctx.stroke();
        }
      }
    }
  }

  // ------------------------------------------------ disk
  const diskA = smoothstep(0.15, 0.42, M) * (1 - galFade);
  if (diskA > 0.01) {
    const g = ctx.createRadialGradient(-20, -20, 10, 0, 0, 150);
    g.addColorStop(0, rgba(hex('#2B1810')));
    g.addColorStop(1, rgba(hex('#5E3520')));
    ctx.globalAlpha = diskA;
    ctx.fillStyle = g;
    ctx.beginPath();
    wobbleCirclePath(ctx, 0, 0, 148, 1.2, 5, boil, 60);
    ctx.fill();
  }
  // rim ring (the cyan band) — shrinks to hug the disk and turns rust
  {
    const mE = E.inOutCubic(seg(M, 0.1, 0.8));
    const r = lerp(ringR(4, t), 147, mE);
    const a = 1 - galFade;
    if (a > 0.01) {
      ctx.globalAlpha = a;
      ctx.strokeStyle = rgba(mix(SPECTRUM[4], hex('#A0521F'), smoothstep(0.1, 0.7, M)));
      ctx.lineWidth = lerp(RING_W, 10, mE);
      ctx.beginPath();
      wobbleCirclePath(ctx, 0, 0, r, 1.4 * mE, 6, boil, 72);
      ctx.stroke();
      if (mE > 0.3) {
        ctx.globalAlpha = a * smoothstep(0.3, 0.9, mE) * 0.8;
        ctx.strokeStyle = rgba(INK);
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        wobbleCirclePath(ctx, 0, 0, r + 5.5, 1.4, 16, boil, 72);
        ctx.stroke();
      }
    }
  }
  // florets ring (the yellow band)
  {
    const L = LAYERS[3];
    const gr = ctx.createRadialGradient(0, 0, L.base, 0, 0, L.tip);
    gr.addColorStop(0, rgba(mix(SPECTRUM[L.ring], L.col, cm)));
    gr.addColorStop(1, rgba(mix(SPECTRUM[L.ring], L.colTip, cm)));
    const a = 1 - seg(t, 13.9, 14.5);
    if (a > 0.01) {
      for (let j = 0; j < L.n; j++) {
        const p = petalPoly(3, j, t, M, boil);
        ctx.globalAlpha = a;
        ctx.beginPath();
        polyPath(ctx, POLY, p.cnt);
        ctx.fillStyle = gr;
        ctx.fill();
        const ol = smoothstep(0.3, 0.9, p.m) * L.outline;
        if (ol > 0.01) {
          ctx.globalAlpha = a * ol;
          ctx.strokeStyle = rgba(INK);
          ctx.lineWidth = 1.8;
          ctx.stroke();
        }
      }
    }
  }

  // ------------------------------------------------ seeds
  const bandA = 1 - seg(M, 0.08, 0.3);
  if (bandA > 0.01) {
    for (const ring of [5, 6]) {
      ctx.globalAlpha = bandA;
      ctx.strokeStyle = rgba(SPECTRUM[ring]);
      ctx.lineWidth = RING_W;
      ctx.beginPath();
      ctx.arc(0, 0, ringR(ring, t), 0, TAU);
      ctx.stroke();
    }
  }
  for (let k = 0; k < N_SEEDS; k++) {
    const s = SEEDS[k];
    const gp = clamp((t - s.launch) / 1.35);
    const a = 1 - smoothstep(0.0, 0.35, gp);
    if (a <= 0.01) continue;
    const e = seedMorph(k, t);
    seedPolar(k, t, SP);
    const x = Math.cos(SP[1]) * SP[0];
    const y = Math.sin(SP[1]) * SP[0];
    const size = lerp(RING_W * 0.55, s.size, e);
    let col = mix(SPECTRUM[s.ring], s.col, smoothstep(0.1, 0.7, e));
    if (bloom > 0) {
      const w = 0.5 + 0.5 * Math.sin(TAU * ((k % 21) / 21) + 3.2 * t);
      col = mix(col, hex('#D08A3A'), 0.3 * w * bloom * (1 - galFade));
    }
    ctx.globalAlpha = a;
    ctx.fillStyle = rgba(col);
    ctx.beginPath();
    ctx.arc(x, y, size, 0, TAU);
    ctx.fill();
  }
  ctx.restore();

  // ------------------------------------------------ petals reborn as stars (drawn in world space)
  for (let li = 1; li < 3; li++) {
    const L = LAYERS[li];
    for (let j = 0; j < L.n; j++) {
      const d = petalDetach(li, j, t);
      if (d < 0.2) continue;
      const pr = PRS[li][j];
      let r = (L.base + L.tip * pr.tip) / 2 + (220 + 420 * pr.rnd) * d;
      let ang = flowerRot(t) + ((j + L.off) * TAU) / L.n + pr.jit + (0.9 + 0.8 * (pr.rnd - 0.5)) * d;
      const c = seg(t, 17.0, 17.9, E.inCubic);
      r = lerp(r, 180, c);
      ang += 2.2 * smoothInt(t, 17.0, 17.9);
      projectDisk(r, ang, iota, beta, WP);
      const k = smoothstep(0.2, 0.6, d);
      const tw = 0.8 + 0.2 * Math.sin(t * 5 + j * 1.7 + li);
      const col = ramp(STAR_RAMP, 0.45 + 0.55 * k);
      const a = k * (1 - c) * (0.4 + 0.6 * PRS[li][(j * 7 + 3) % L.n].rnd);
      softGlow(ctx, WP[0], WP[1], (26 - 8 * k) * tw, col, 0.9 * a, dark);
      ctx.globalAlpha = a;
      ctx.fillStyle = rgba(mix(col, C.white, 0.6));
      ctx.beginPath();
      ctx.arc(WP[0], WP[1], 2.8 * tw, 0, TAU);
      ctx.fill();
    }
  }
  ctx.restore();
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
}
