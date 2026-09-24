// Scene 02 — a spark swells into a living dendritic network; one branch pulls taut into a beam of light.
import { Ctx, boilOf, glow, ribbonPath, softGlow, sparkle, tracePath, wobbleCirclePath } from '../core/draw';
import { C, RGB, hex, mix, rgba } from '../core/color';
import { E, TAU, clamp, lerp, mod, seg } from '../core/math';
import { RNG } from '../core/rng';
import { ENTRY, SPARK } from '../core/geometry';
import { darkness } from '../core/timeline';

interface Seg {
  pts: Float32Array;
  n: number;
  d0: number;
  w0: number;
  w1: number;
  depth: number;
  bid: number;
  parent: number;
  kids: number;
  root: number;
}
interface Sat {
  x: number;
  y: number;
  r: number;
  d: number;
  bid: number;
}
interface Spark {
  x: number;
  y: number;
  vx: number;
  vy: number;
  t: number;
  life: number;
  col: RGB;
}
interface Pulse {
  path: Float32Array;
  cum: Float32Array;
  n: number;
  L: number;
  d0: number;
  t0: number;
  speed: number;
  inward: boolean;
  col: RGB;
  bid: number;
}
interface Node {
  x: number;
  y: number;
  r: number;
  events: number[];
  phase: number;
  bid: number;
  d: number;
}

const STEP = 11;
const BURST = 3.62;
const segs: Seg[] = [];
const sats: Sat[] = [];
const speedOf: number[] = [];
const pulses: Pulse[] = [];
const sparks: Spark[] = [];
const nodes: Node[] = [];
const nodeOfSeg = new Map<number, number>();
let selPath: Float32Array = new Float32Array(0);
let selCum: Float32Array = new Float32Array(0);
let selN = 0;
let selL = 1;

const rng = new RNG(4242);

function grow(x: number, y: number, a: number, len: number, w: number, depth: number, d: number, bid: number, parent: number, maxDepth: number, root: number): void {
  const n = Math.max(3, Math.round(len / STEP)) + 1;
  const pts = new Float32Array(n * 2);
  pts[0] = x;
  pts[1] = y;
  const curl = rng.gauss() * 0.035;
  for (let i = 1; i < n; i++) {
    a += curl + rng.gauss() * 0.11;
    if (rng.chance(0.08)) a += rng.gauss() * 0.42; // occasional lightning kink
    x += Math.cos(a) * STEP;
    y += Math.sin(a) * STEP;
    pts[2 * i] = x;
    pts[2 * i + 1] = y;
  }
  const idx = segs.length;
  segs.push({ pts, n, d0: d, w0: w, w1: w * 0.64, depth, bid, parent, kids: 0, root });
  if (depth < maxDepth) {
    const k = depth < 2 ? 2 : rng.chance(0.72) ? 2 : 1;
    for (let j = 0; j < k; j++) {
      const da = k === 1 ? rng.range(-0.3, 0.3) : (j === 0 ? -1 : 1) * rng.range(0.3, 0.72);
      grow(x, y, a + da, len * rng.range(0.64, 0.84), w * 0.64, depth + 1, d + (n - 1) * STEP, bid, idx, maxDepth, root);
    }
    segs[idx].kids = k;
  }
}

function chainOf(s: number): number[] {
  const chain: number[] = [];
  while (s !== -1) {
    chain.unshift(s);
    s = segs[s].parent;
  }
  return chain;
}

function buildPath(chain: number[], start: [number, number] | null): { path: Float32Array; cum: Float32Array; n: number; L: number; offsets: number[] } {
  const p: number[] = [];
  const offsets: number[] = [];
  if (start) p.push(start[0], start[1]);
  for (const s of chain) {
    const sg = segs[s];
    offsets.push(p.length / 2 - (p.length > 0 ? 1 : 0));
    for (let i = p.length > 0 ? 1 : 0; i < sg.n; i++) p.push(sg.pts[2 * i], sg.pts[2 * i + 1]);
  }
  const n = p.length / 2;
  const cum = new Float32Array(n);
  for (let i = 1; i < n; i++) cum[i] = cum[i - 1] + Math.hypot(p[2 * i] - p[2 * i - 2], p[2 * i + 1] - p[2 * i - 1]);
  return { path: new Float32Array(p), cum, n, L: cum[n - 1], offsets: offsets.map((o) => cum[Math.max(0, o)]) };
}

(function build() {
  for (let b = 0; b < 9; b++) {
    const a = (b / 9) * TAU + rng.range(-0.22, 0.22);
    speedOf[b] = rng.range(0.86, 1.14);
    grow(Math.cos(a) * 16, Math.sin(a) * 16, a, rng.range(150, 205), 7.5, 0, 16, b, -1, 5, 0);
  }
  for (let s = 0; s < 6; s++) {
    const a = (s / 6) * TAU + 0.3 + rng.range(-0.2, 0.2);
    const r = rng.range(380, 540);
    const bid = 9 + s;
    speedOf[bid] = rng.range(0.9, 1.1);
    const sat: Sat = { x: Math.cos(a) * r, y: Math.sin(a) * r, r: rng.range(9, 14), d: r, bid };
    sats.push(sat);
    for (let k = 0; k < 5; k++) {
      const a2 = (k / 5) * TAU + rng.range(-0.3, 0.3);
      grow(sat.x + Math.cos(a2) * 10, sat.y + Math.sin(a2) * 10, a2, rng.range(70, 115), 4.4, 2, r + 10, bid, -1, 4, s + 1);
    }
  }
  // junction nodes
  segs.forEach((sg, i) => {
    if (sg.parent !== -1 && sg.depth <= 3) {
      nodeOfSeg.set(i, nodes.length);
      nodes.push({ x: sg.pts[0], y: sg.pts[1], r: 2 + (4 - sg.depth) * 0.9, events: [], phase: rng.range(0, TAU), bid: sg.bid, d: sg.d0 });
    }
  });
  const leaves = segs.map((s, i) => ({ s, i })).filter((o) => o.s.kids === 0);
  const warm: RGB[] = [hex('#FFC25A'), hex('#E6508E'), hex('#F58A3C'), hex('#FFE08A')];
  for (let p = 0; p < 58; p++) {
    const main = p < 46;
    const pool = leaves.filter((o) => (main ? o.s.root === 0 : o.s.root > 0));
    const leaf = rng.pick(pool);
    const chain = chainOf(leaf.i);
    const sg = segs[leaf.i];
    const start: [number, number] | null = sg.root === 0 ? [0, 0] : [sats[sg.root - 1].x, sats[sg.root - 1].y];
    const bp = buildPath(chain, start);
    const inward = rng.chance(0.35);
    const pulse: Pulse = {
      path: bp.path,
      cum: bp.cum,
      n: bp.n,
      L: bp.L,
      d0: sg.root === 0 ? 0 : sats[sg.root - 1].d,
      t0: rng.range(4.0, 7.0),
      speed: rng.range(520, 840),
      inward,
      col: warm[p % warm.length],
      bid: sg.bid,
    };
    pulses.push(pulse);
    chain.forEach((s, ci) => {
      const ni = nodeOfSeg.get(s);
      if (ni === undefined) return;
      const off = bp.offsets[ci];
      const tp = pulse.t0 + (inward ? pulse.L - off : off) / pulse.speed;
      nodes[ni].events.push(tp);
    });
    // sparks thrown off along the way
    for (let sd = 70; sd < pulse.L; sd += rng.range(80, 140)) {
      const [ex, ey] = pointAt(pulse.path, pulse.cum, pulse.n, sd);
      const te = pulse.t0 + (inward ? pulse.L - sd : sd) / pulse.speed;
      for (let k = 0; k < 3; k++) {
        const a = rng.range(0, TAU);
        const v = rng.range(60, 180);
        sparks.push({ x: ex, y: ey, vx: Math.cos(a) * v, vy: Math.sin(a) * v, t: te, life: rng.range(0.28, 0.55), col: warm[(p + k) % warm.length] });
      }
    }
  }
  // the branch that will become the beam: the main leaf farthest to screen-left
  let best = -1;
  let bestX = Infinity;
  const r = 0.07;
  for (const o of leaves) {
    if (o.s.root !== 0) continue;
    const x = o.s.pts[2 * o.s.n - 2] * Math.cos(r) - o.s.pts[2 * o.s.n - 1] * Math.sin(r);
    if (x < bestX) {
      bestX = x;
      best = o.i;
    }
  }
  const bp = buildPath(chainOf(best), [0, 0]);
  const rev = new Float32Array(bp.n * 2);
  for (let i = 0; i < bp.n; i++) {
    rev[2 * i] = bp.path[2 * (bp.n - 1 - i)];
    rev[2 * i + 1] = bp.path[2 * (bp.n - 1 - i) + 1];
  }
  selPath = rev;
  selN = bp.n;
  selCum = new Float32Array(selN);
  for (let i = 1; i < selN; i++) selCum[i] = selCum[i - 1] + Math.hypot(rev[2 * i] - rev[2 * i - 2], rev[2 * i + 1] - rev[2 * i - 1]);
  selL = selCum[selN - 1];
})();

function pointAt(path: Float32Array, cum: Float32Array, n: number, s: number): [number, number] {
  if (s <= 0) return [path[0], path[1]];
  if (s >= cum[n - 1]) return [path[2 * n - 2], path[2 * n - 1]];
  let i = 1;
  while (i < n - 1 && cum[i] < s) i++;
  const f = (s - cum[i - 1]) / (cum[i] - cum[i - 1] || 1);
  return [lerp(path[2 * i - 2], path[2 * i], f), lerp(path[2 * i - 1], path[2 * i + 1], f)];
}

function front(t: number, bid: number): number {
  const tau = t - BURST;
  if (tau <= 0) return 0;
  return 780 * speedOf[bid] * (1 - Math.pow(1 - clamp(tau / 1.55), 1.7));
}

export function netCam(t: number): { ax: number; ay: number; zoom: number; rot: number } {
  const k = seg(t, 3.7, 5.5, E.inOutCubic);
  return {
    ax: lerp(SPARK.x, 960, k),
    ay: lerp(SPARK.y, 540, k),
    zoom: 1 + 1.35 * k + 0.35 * seg(t, 5.5, 7.8, E.inOutSine),
    rot: 0.1 * seg(t, 3.7, 7.8, E.inOutSine),
  };
}

const TMP = new Float32Array(512);

export function drawNeurons(ctx: Ctx, t: number): void {
  if (t < 2.9 || t > 7.85) return;
  const dark = darkness(t);
  const boil = boilOf(t);
  const cam = netCam(t);
  const netA = 1 - seg(t, 6.55, 7.45, E.inOutSine);
  const lineCol = mix(C.ink, hex('#A99BE0'), dark);

  ctx.save();
  ctx.translate(cam.ax, cam.ay);
  ctx.rotate(cam.rot);
  ctx.scale(cam.zoom, cam.zoom);

  // ---- the spark (before the burst)
  if (t < BURST + 0.5) {
    const on = seg(t, 2.95, 3.25, E.outBack);
    const pre = 1 - seg(t, BURST, BURST + 0.3);
    const tw = 1 + 0.25 * Math.sin(t * 19) * (1 - seg(t, 3.4, BURST));
    const swell = 1 + 1.2 * seg(t, 3.3, BURST, E.inCubic);
    softGlow(ctx, 0, 0, 70 * on * swell, hex('#FFC857'), 0.9 * pre, dark);
    ctx.globalAlpha = 1;
    sparkle(ctx, 0, 0, 30 * on * tw * swell, t * 0.8, hex('#F5A623'), 0.9 * pre);
    sparkle(ctx, 0, 0, 16 * on * tw * swell, -t * 0.6 + 0.78, hex('#FFF2C4'), pre);
    ctx.globalAlpha = pre;
    ctx.fillStyle = rgba(hex('#FFF6D8'));
    ctx.beginPath();
    ctx.arc(0, 0, 5.5 * on * swell, 0, TAU);
    ctx.fill();
  }
  // burst ring
  const bt = t - BURST;
  if (bt > 0 && bt < 0.8) {
    const k = bt / 0.8;
    ctx.globalAlpha = (1 - k) * 0.7;
    ctx.strokeStyle = rgba(hex('#F08A3C'));
    ctx.lineWidth = 3 * (1 - k) + 0.5;
    ctx.beginPath();
    wobbleCirclePath(ctx, 0, 0, 20 + 260 * E.outCubic(k), 4, 3, boil, 64);
    ctx.stroke();
  }

  if (t >= BURST) {
    // ---- branches (one union fill)
    ctx.globalAlpha = netA;
    ctx.fillStyle = rgba(lineCol);
    ctx.beginPath();
    for (const sg of segs) {
      const fr = front(t, sg.bid);
      const f = (fr - sg.d0) / STEP;
      if (f <= 0) continue;
      const m = Math.min(sg.n - 1, Math.floor(f));
      let cnt = 0;
      for (let i = 0; i <= m; i++) {
        TMP[2 * cnt] = sg.pts[2 * i];
        TMP[2 * cnt + 1] = sg.pts[2 * i + 1];
        cnt++;
      }
      if (m < sg.n - 1) {
        const fr2 = f - m;
        TMP[2 * cnt] = lerp(sg.pts[2 * m], sg.pts[2 * m + 2], fr2);
        TMP[2 * cnt + 1] = lerp(sg.pts[2 * m + 1], sg.pts[2 * m + 3], fr2);
        cnt++;
      }
      if (cnt < 2) continue;
      const leafTaper = sg.kids === 0;
      ribbonPath(ctx, TMP, cnt, (i) => {
        const u = i / (sg.n - 1);
        const d = sg.d0 + i * STEP;
        let w = lerp(sg.w0, sg.w1, u) * (1 + 0.22 * Math.sin(d * 0.05 + sg.bid * 1.7));
        if (leafTaper) w *= lerp(1, 0.25, u);
        w *= clamp((fr - d) / 26, 0.2, 1);
        return w;
      });
    }
    ctx.fill();

    // growth cones: bright tips while branches are still extending
    for (const sg of segs) {
      if (sg.kids !== 0) continue;
      const fr = front(t, sg.bid);
      const f = (fr - sg.d0) / STEP;
      if (f <= 0 || f >= sg.n - 1) continue;
      const m = Math.floor(f);
      const x = lerp(sg.pts[2 * m], sg.pts[2 * m + 2], f - m);
      const y = lerp(sg.pts[2 * m + 1], sg.pts[2 * m + 3], f - m);
      softGlow(ctx, x, y, 16, hex('#FFB347'), 0.7 * netA, dark);
    }

    // ---- nodes that breathe and flare as signals pass
    for (const nd of nodes) {
      if (front(t, nd.bid) < nd.d) continue;
      let flare = 0;
      for (const te of nd.events) {
        const x = (t - te) / 0.1;
        if (x > -3 && x < 3) flare += Math.exp(-x * x);
      }
      const r = nd.r * (1 + 0.22 * Math.sin(t * 3.1 + nd.phase)) * (1 + 0.8 * flare);
      if (flare > 0.05) softGlow(ctx, nd.x, nd.y, r * 7, hex('#FFB347'), Math.min(1, flare) * 0.8 * netA, dark);
      ctx.globalAlpha = netA;
      ctx.fillStyle = rgba(mix(mix(C.ink, hex('#C98BE8'), dark), hex('#FFD27A'), clamp(flare)));
      ctx.beginPath();
      ctx.arc(nd.x, nd.y, r, 0, TAU);
      ctx.fill();
    }

    // ---- satellite cell bodies
    for (const s of sats) {
      const on = seg(front(t, s.bid), s.d - 40, s.d + 20, E.outBack);
      if (on <= 0) continue;
      const r = s.r * on * (1 + 0.1 * Math.sin(t * 2.4 + s.x));
      softGlow(ctx, s.x, s.y, r * 4, hex('#F5A04A'), 0.5 * netA, dark);
      ctx.globalAlpha = netA;
      ctx.fillStyle = rgba(mix(hex('#F2A54A'), hex('#FFCF73'), dark));
      ctx.strokeStyle = rgba(lineCol);
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      wobbleCirclePath(ctx, s.x, s.y, r, 1.2, s.bid, boil, 24);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = rgba(lineCol, 0.8);
      ctx.beginPath();
      ctx.arc(s.x + r * 0.15, s.y - r * 0.1, r * 0.34, 0, TAU);
      ctx.fill();
    }

    // ---- root soma (the spark, grown up)
    const somaA = 1 - seg(t, 6.6, 7.3);
    if (somaA > 0) {
      const r = lerp(8, 19, seg(t, BURST, BURST + 0.4, E.outBack)) * (1 + 0.08 * Math.sin(t * 3.4));
      softGlow(ctx, 0, 0, r * 5, hex('#FFB84D'), 0.85 * somaA, dark);
      ctx.globalAlpha = somaA;
      ctx.fillStyle = rgba(hex('#F7B24A'));
      ctx.strokeStyle = rgba(lineCol);
      ctx.lineWidth = 3;
      ctx.beginPath();
      wobbleCirclePath(ctx, 0, 0, r, 1.4, 1, boil, 28);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = rgba(hex('#FFF1C9'));
      ctx.beginPath();
      ctx.arc(-r * 0.2, -r * 0.2, r * 0.38, 0, TAU);
      ctx.fill();
    }

    // ---- travelling signals
    for (const p of pulses) {
      let s = (t - p.t0) * p.speed;
      if (s < 0 || s > p.L) continue;
      if (p.inward) s = p.L - s;
      if (front(t, p.bid) < p.d0 + s) continue;
      for (let k = 5; k >= 0; k--) {
        const ss = p.inward ? s + k * 13 : s - k * 13;
        if (ss < 0 || ss > p.L) continue;
        const [x, y] = pointAt(p.path, p.cum, p.n, ss);
        const f = 1 - k / 6;
        softGlow(ctx, x, y, 7 + 12 * f, p.col, 0.55 * f * netA, dark);
      }
      const [hx, hy] = pointAt(p.path, p.cum, p.n, s);
      ctx.globalAlpha = netA;
      ctx.fillStyle = rgba(mix(p.col, C.white, 0.6 + 0.4 * dark));
      ctx.beginPath();
      ctx.arc(hx, hy, 3.4, 0, TAU);
      ctx.fill();
    }

    // ---- sparks flicking off the signals
    ctx.lineCap = 'round';
    for (const sp of sparks) {
      const tau = t - sp.t;
      if (tau < 0 || tau > sp.life) continue;
      const k = tau / sp.life;
      const damp = (sp.life * (1 - Math.exp((-3 * tau) / sp.life))) / 3;
      const x = sp.x + sp.vx * damp;
      const y = sp.y + sp.vy * damp;
      const len = 0.045 * (1 - k);
      ctx.globalAlpha = (1 - k) * netA;
      ctx.globalCompositeOperation = dark > 0.5 ? 'lighter' : 'source-over';
      ctx.strokeStyle = rgba(dark > 0.5 ? sp.col : mix(sp.col, hex('#E0662E'), 0.4));
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x - sp.vx * len, y - sp.vy * len);
      ctx.stroke();
    }
    ctx.globalCompositeOperation = 'source-over';
  }
  ctx.restore();

  drawSelected(ctx, t, cam);
  ctx.globalAlpha = 1;
}

const SEL = new Float32Array(1024);

/** The chosen branch glows, pulls taut and becomes the horizontal beam line. */
function drawSelected(ctx: Ctx, t: number, cam: { ax: number; ay: number; zoom: number; rot: number }): void {
  if (t < 6.45 || t > 7.8) return;
  const glowUp = seg(t, 6.45, 6.85, E.inOutSine);
  const fade = 1 - seg(t, 7.62, 7.78);
  const cs = Math.cos(cam.rot);
  const sn = Math.sin(cam.rot);
  for (let i = 0; i < selN; i++) {
    const lx = selPath[2 * i];
    const ly = selPath[2 * i + 1];
    const px = cam.ax + cam.zoom * (lx * cs - ly * sn);
    const py = cam.ay + cam.zoom * (lx * sn + ly * cs);
    const u = selCum[i] / selL;
    const tx = lerp(-120, ENTRY.x, u);
    const ty = ENTRY.y;
    const s = E.inOutCubic(clamp((t - 6.85 - 0.3 * (1 - u)) / 0.55));
    SEL[2 * i] = lerp(px, tx, s);
    SEL[2 * i + 1] = lerp(py, ty, s);
  }
  const taut = seg(t, 6.9, 7.7);
  const col = mix(hex('#C8B8FF'), C.white, taut);
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  const layers: [number, number][] = [
    [30, 0.06],
    [12, 0.2],
    [3.6, 0.95],
  ];
  for (const [w, a] of layers) {
    ctx.globalAlpha = a * glowUp * fade;
    ctx.strokeStyle = rgba(w < 5 ? C.white : col);
    ctx.lineWidth = w * lerp(cam.zoom * 0.5, 1, taut);
    ctx.beginPath();
    tracePath(ctx, SEL, selN, false);
    ctx.stroke();
  }
  // energy flowing toward the right, where the prism will be
  for (let k = 0; k < 3; k++) {
    const s = mod((t - 6.45) * 950 + (k * selL) / 3, selL);
    const [x, y] = pointAt(SEL, selCum, selN, s * 1);
    glow(ctx, x, y, 34, hex('#FFF1D6'), 0.8 * glowUp * fade);
  }
  ctx.restore();
}
