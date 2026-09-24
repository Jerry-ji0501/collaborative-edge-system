// Scenes 06–07 — every sunflower seed splits into nine stars that spiral out into a two-armed galaxy;
// the core then collapses into a black hole whose accretion disk is those same particles.
import { Ctx, StrokeBatch, boilOf, glow, wobbleCirclePath } from '../core/draw';
import { RGB, hex, rgba } from '../core/color';
import { E, TAU, clamp, lerp, mod, seg, smoothInt, smoothstep } from '../core/math';
import { RNG } from '../core/rng';
import { CENTER } from '../core/geometry';
import { Cam, applyCam, cosmicCam, diskView, projectDisk, toScreen } from '../core/camera';
import { darkness } from '../core/timeline';
import { N_SEEDS, SEEDS, SEED_PAL, flowerRot, seedPolar } from './sunflower';

const M = 9;
const NP = N_SEEDS * M;
const REF = 13.8;

const GAL_PAL: RGB[] = ['#FFF2D6', '#FFC96B', '#FF9A4D', '#FF77A8', '#A58CFF', '#7AA2FF', '#BFE6FF', '#FFFFFF'].map(hex);
const DISK_PAL: RGB[] = ['#FFF6E0', '#FFE2A0', '#FFC261', '#FF9448', '#F2607E', '#C9509E', '#8C6BFF', '#5D7CFF'].map(hex);
const PAL_CSS = [...SEED_PAL, ...GAL_PAL, ...DISK_PAL].map((c) => rgba(c));
const P_SEED = 0;
const P_GAL = 6;
const P_DISK = 14;

const pSeed = new Int16Array(NP);
const rG = new Float32Array(NP);
const thG = new Float32Array(NP);
const wG = new Float32Array(NP);
const rD = new Float32Array(NP);
const wD = new Float32Array(NP);
const aB = new Float32Array(NP);
const colG = new Uint8Array(NP);
const colD = new Uint8Array(NP);
const tf = new Float32Array(NP);
const tc = new Float32Array(NP);
const wl = new Uint8Array(NP);
const bright = new Uint8Array(NP);
const wisp = new Uint8Array(NP);

function diskCol(r: number): number {
  const edges = [118, 140, 170, 205, 240, 280, 320];
  for (let i = 0; i < edges.length; i++) if (r < edges[i]) return i;
  return 7;
}

(function build() {
  const rng = new RNG(777);
  for (let k = 0; k < N_SEEDS; k++) {
    const s = SEEDS[k];
    const seedRef = s.phi + flowerRot(REF);
    for (let c = 0; c < M; c++) {
      const i = k * M + c;
      pSeed[i] = k;
      const u = s.u;
      let r: number;
      let th: number;
      let col: number;
      const bulge = u < 0.24 && rng.chance(0.75);
      if (bulge) {
        r = Math.abs(rng.gauss()) * 55 + 6;
        th = rng.range(0, TAU);
        col = rng.pick([0, 0, 1, 1, 2, 7]);
      } else {
        const uu = clamp(u + rng.gauss() * 0.07, 0.04, 1);
        r = 40 + 640 * Math.pow(uu, 1.25);
        const arm = (k * 7 + c) % 2;
        const base = arm * Math.PI - Math.log(r / 40) * 3.1;
        const spread = rng.chance(0.2) ? rng.range(0, TAU) : rng.gauss() * (0.2 + 0.22 * uu);
        th = base + spread;
        r *= 1 + rng.gauss() * 0.05;
        const x = rng.next();
        col = uu < 0.35 && x < 0.3 ? 1 : x < 0.33 ? 6 : x < 0.58 ? 5 : x < 0.78 ? 4 : x < 0.93 ? 7 : 3;
      }
      rG[i] = r;
      thG[i] = seedRef + mod(th - seedRef, TAU);
      wG[i] = 0.42 / (1 + r / 160);
      const rd = bulge ? 96 + rng.range(0, 40) : 98 + 262 * Math.pow(Math.min(r, 700) / 700, 0.85) + rng.gauss() * 5;
      rD[i] = Math.max(94, rd);
      wD[i] = 2.3 * Math.pow(100 / rD[i], 1.5);
      colG[i] = col;
      colD[i] = diskCol(rD[i]);
      aB[i] = rng.range(0.45, 1) * (bulge ? 1 : 0.95);
      tf[i] = s.launch + rng.range(0, 0.06);
      tc[i] = 17.0 + 0.3 * Math.min(1, r / 700);
      const x = rng.next();
      wl[i] = x < 0.6 ? 0 : x < 0.92 ? 1 : 2;
      bright[i] = rng.chance(0.03) ? 1 : 0;
      wisp[i] = !bulge && rng.chance(0.025) ? 1 : 0;
    }
  }
})();

// infalling streams
const NI = 220;
const iP = new Float32Array(NI);
const iPh = new Float32Array(NI);
const iTh = new Float32Array(NI);
const iR = new Float32Array(NI);
(function () {
  const r = new RNG(313);
  for (let i = 0; i < NI; i++) {
    iP[i] = r.range(1.8, 3.0);
    iPh[i] = r.next();
    iTh[i] = r.range(0, TAU);
    iR[i] = r.range(520, 780);
  }
})();

const seedB = new StrokeBatch();
const farB = new StrokeBatch();
const nearB = new StrokeBatch();
const lensB = new StrokeBatch();
const H0: [number, number] = [0, 0];
const T0: [number, number] = [0, 0];
const SPOL: [number, number] = [0, 0];
const seedR = new Float32Array(N_SEEDS);
const seedA = new Float32Array(N_SEEDS);

export function horizonR(t: number): number {
  return 86 * seg(t, 17.25, 18.25, E.outCubic);
}

export type Portal = (ctx: Ctx, t: number, x: number, y: number, r: number) => void;

export function drawGalaxy(ctx: Ctx, t: number, portal?: Portal): void {
  if (t < 13.85 || t > 21.05) return;
  const cam = cosmicCam(t);
  const camPrev = cosmicCam(t - 0.022);
  const diveFade = 1 - seg(t, 20.35, 20.72, E.inOutSine);
  const { iota, beta } = diskView(t);
  const dark = darkness(t);
  const boil = boilOf(t);
  const h = horizonR(t);
  const bhOn = t > 17.2;
  const diveK = camPrev.zoom / cam.zoom; // < 1 while diving: radial streaks toward centre
  const frRate = (flowerRot(t + 0.01) - flowerRot(t - 0.01)) / 0.02;
  const shutter = 0.08;
  const lensOn = seg(t, 17.6, 18.4) * (1 - seg(t, 20.2, 20.6));
  const glowList: number[] = [];
  const wispList: number[] = [];
  // per-seed values once per frame (not per particle)
  const fr = flowerRot(t);
  for (let k = 0; k < N_SEEDS; k++) {
    seedPolar(k, t, SPOL);
    seedR[k] = SPOL[0];
    seedA[k] = SPOL[1];
  }
  const cb = Math.cos(beta);
  const sb = Math.sin(beta);
  const proj = (r: number, a: number, out: [number, number]): void => {
    const px = r * Math.cos(a);
    const py = r * Math.sin(a) * iota;
    out[0] = CENTER.x + px * cb - py * sb;
    out[1] = CENTER.y + px * sb + py * cb;
  };

  for (let i = 0; i < NP; i++) {
    const fp = (t - tf[i]) / 1.35;
    if (fp <= 0) continue;
    const eF = E.inOutCubic(clamp(fp));
    const k = pSeed[i];
    const D = thG[i] - SEEDS[k].phi - fr + wG[i] * (t - REF);
    let th = seedA[k] + eF * D;
    let r = lerp(seedR[k], rG[i], eF);
    let c = 0;
    let wBH = 0;
    if (t > tc[i]) {
      c = E.inOutCubic(clamp((t - tc[i]) / 1.2));
      r = lerp(r, rD[i], c);
      th += wD[i] * smoothInt(t, tc[i], tc[i] + 1.2);
      wBH = wD[i] * smoothstep(tc[i], tc[i] + 1.2, t);
    }
    const w = wG[i] * eF + (1 - eF) * frRate + wBH;
    const dth = Math.min(0.55, w * shutter * (1 + 2 * c));
    proj(r, th, H0);
    proj(r, th - dth, T0);
    if (diveK < 0.999) {
      T0[0] = CENTER.x + (T0[0] - CENTER.x) * diveK;
      T0[1] = CENTER.y + (T0[1] - CENTER.y) * diveK;
    }
    // a hair of length so dots render even when still
    if (Math.abs(H0[0] - T0[0]) + Math.abs(H0[1] - T0[1]) < 0.3) T0[0] -= 0.3;
    const a = aB[i] * diveFade;
    if (a <= 0.02) continue;
    const near = bhOn && Math.sin(th) > 0;
    const B = near ? nearB : farB;
    if (eF < 1) seedB.add(P_SEED + SEEDS[k].colIdx, a * (1 - eF), wl[i], H0[0], H0[1], T0[0], T0[1]);
    if (c < 1) B.add(P_GAL + colG[i], a * eF * (1 - c), wl[i], H0[0], H0[1], T0[0], T0[1]);
    if (c > 0) B.add(P_DISK + colD[i], Math.min(1, a * 1.15) * c, wl[i], H0[0], H0[1], T0[0], T0[1]);
    if (lensOn > 0 && c > 0.2 && (i & 1) === 0) {
      const rho = h * (1.1 + 0.95 * clamp((r - 95) / 270));
      const s = Math.sin(th);
      const la = a * c * lensOn * (s < 0 ? 0.75 : 0.22);
      const x0 = rho * Math.cos(th);
      const y0 = rho * s;
      const x1 = rho * Math.cos(th - dth);
      const y1 = rho * Math.sin(th - dth);
      lensB.add(P_DISK + colD[i], la, wl[i], CENTER.x + x0 * cb - y0 * sb, CENTER.y + x0 * sb + y0 * cb, CENTER.x + x1 * cb - y1 * sb, CENTER.y + x1 * sb + y1 * cb);
    }
    if (bright[i] && eF > 0.3) glowList.push(H0[0], H0[1], a * eF, colG[i], c);
    if (wisp[i] && eF > 0.5) wispList.push(H0[0], H0[1], eF, colG[i]);
  }

  // infalling streams (black hole only)
  const inOn = seg(t, 17.8, 18.4) * (1 - seg(t, 20.1, 20.5));
  if (inOn > 0) {
    for (let i = 0; i < NI; i++) {
      const tau = mod((t - 17.6) / iP[i] + iPh[i], 1);
      if (tau < 0.04) continue;
      const rr = (x: number): number => lerp(iR[i], 99, E.inQuad(x));
      const tt = (x: number): number => iTh[i] + 6 * x * x + t * 0.3;
      const th = tt(tau);
      projectDisk(rr(tau), th, iota, beta, H0);
      projectDisk(rr(tau - 0.035), tt(tau - 0.035), iota, beta, T0);
      const a = Math.pow(Math.sin(Math.PI * tau), 0.6) * inOn * 0.8;
      (Math.sin(th) > 0 ? nearB : farB).add(P_DISK + diskCol(rr(tau)), a, 1, H0[0], H0[1], T0[0], T0[1]);
    }
  }

  const pxW = [1.45, 2.2, 3.2].map((w) => w / cam.zoom);

  ctx.save();
  applyCam(ctx, cam);

  // seed-coloured children: painted
  ctx.globalCompositeOperation = 'source-over';
  seedB.flush(ctx, PAL_CSS, pxW.map((w) => w * 1.4));

  // galaxy core glow & wisps (fade as the core collapses)
  const coreG = seg(t, 14.4, 15.5) * (1 - seg(t, 17.0, 17.6));
  ctx.globalCompositeOperation = 'lighter';
  if (coreG > 0) {
    ctx.save();
    ctx.translate(CENTER.x, CENTER.y);
    ctx.rotate(beta);
    ctx.scale(1, iota);
    const pulse = 1 + 0.04 * Math.sin(t * 2.2);
    glow(ctx, 0, 0, 300 * pulse, hex('#FFB870'), 0.3 * coreG);
    glow(ctx, 0, 0, 125 * pulse, hex('#FFE3B0'), 0.55 * coreG);
    glow(ctx, 0, 0, 42, hex('#FFFFFF'), 0.85 * coreG);
    ctx.restore();
  }
  const wispA = seg(t, 14.6, 15.6) * (1 - seg(t, 16.9, 17.5)) * dark;
  for (let q = 0; q < wispList.length; q += 4) {
    glow(ctx, wispList[q], wispList[q + 1], 70, GAL_PAL[wispList[q + 3]], 0.06 * wispA * wispList[q + 2]);
  }

  // accretion disk glow (behind)
  const diskG = seg(t, 17.3, 18.3) * (1 - seg(t, 20.3, 20.8));
  if (diskG > 0) {
    ctx.save();
    ctx.translate(CENTER.x, CENTER.y);
    ctx.rotate(beta);
    ctx.scale(1, iota);
    const g = ctx.createRadialGradient(0, 0, Math.max(1, h), 0, 0, 390);
    g.addColorStop(0, 'rgba(255,210,140,0)');
    g.addColorStop(0.04, 'rgba(255,200,120,0.5)');
    g.addColorStop(0.22, 'rgba(255,130,80,0.28)');
    g.addColorStop(0.5, 'rgba(200,80,160,0.15)');
    g.addColorStop(0.8, 'rgba(90,80,220,0.07)');
    g.addColorStop(1, 'rgba(60,60,200,0)');
    ctx.globalAlpha = diskG;
    ctx.fillStyle = g;
    ctx.fillRect(-400, -400, 800, 800);
    ctx.restore();
  }

  // far side
  const addW = clamp(dark * 1.4 - 0.2);
  if (addW < 1) {
    ctx.globalCompositeOperation = 'source-over';
    farB.flush(ctx, PAL_CSS, pxW, 1 - addW, false);
  }
  ctx.globalCompositeOperation = 'lighter';
  farB.flush(ctx, PAL_CSS, pxW, addW);

  // bright stars
  for (let q = 0; q < glowList.length; q += 5) {
    const cc = glowList[q + 4];
    const col = cc > 0.5 ? DISK_PAL[1] : GAL_PAL[glowList[q + 3]];
    glow(ctx, glowList[q], glowList[q + 1], (9 + 5 * Math.sin(t * 4 + q)) / Math.sqrt(cam.zoom), col, 0.7 * glowList[q + 2] * dark);
  }

  // event horizon
  if (h > 0.5) {
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#020205';
    ctx.beginPath();
    wobbleCirclePath(ctx, CENTER.x, CENTER.y, h, 0.8, 21, boil, 80);
    ctx.fill();
  }
  ctx.restore();

  // the tunnel seen through the horizon as we fall in
  if (portal && h > 0.5 && t > 20.3) {
    const [px, py] = toScreen(cam, CENTER.x, CENTER.y);
    portal(ctx, t, px, py, h * cam.zoom * 0.985);
  }

  ctx.save();
  applyCam(ctx, cam);
  if (h > 0.5) {
    // lensed image of the far disk, arched over the top
    if (lensOn > 0) {
      ctx.save();
      ctx.translate(CENTER.x, CENTER.y);
      ctx.rotate(beta);
      const g = ctx.createRadialGradient(0, 0, h * 1.0, 0, 0, h * 2.15);
      g.addColorStop(0, 'rgba(255,230,180,0)');
      g.addColorStop(0.03, 'rgba(255,230,180,0.6)');
      g.addColorStop(0.25, 'rgba(255,160,90,0.32)');
      g.addColorStop(0.65, 'rgba(210,90,170,0.12)');
      g.addColorStop(1, 'rgba(120,80,220,0)');
      ctx.globalCompositeOperation = 'lighter';
      ctx.fillStyle = g;
      ctx.globalAlpha = 0.28 * lensOn;
      ctx.fillRect(-h * 2.2, -h * 2.2, h * 4.4, h * 4.4);
      ctx.beginPath();
      ctx.rect(-h * 2.2, -h * 2.2, h * 4.4, h * 2.2);
      ctx.clip();
      ctx.globalAlpha = 0.55 * lensOn;
      ctx.fillRect(-h * 2.2, -h * 2.2, h * 4.4, h * 4.4);
      ctx.restore();
      ctx.globalCompositeOperation = 'lighter';
      lensB.flush(ctx, PAL_CSS, pxW);
    }
    // photon ring
    const ringA = seg(t, 17.5, 18.2);
    ctx.globalCompositeOperation = 'lighter';
    for (const [w, a] of [
      [16, 0.1],
      [6, 0.3],
      [2.2, 0.95],
    ] as [number, number][]) {
      ctx.globalAlpha = a * ringA;
      ctx.strokeStyle = 'rgb(255,236,200)';
      ctx.lineWidth = w / Math.sqrt(cam.zoom / 1.3);
      ctx.beginPath();
      wobbleCirclePath(ctx, CENTER.x, CENTER.y, h * 1.03, 0.9, 22, boil, 90);
      ctx.stroke();
    }
  }
  // near side passes in front of the hole
  ctx.globalCompositeOperation = 'lighter';
  nearB.flush(ctx, PAL_CSS, pxW);
  ctx.restore();
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
}

export function galaxyCam(t: number): Cam {
  return cosmicCam(t);
}
