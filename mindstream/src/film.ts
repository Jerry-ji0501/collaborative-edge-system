// The director: background, scene layers in order, and the paper / grain finish.
import { Ctx, boilOf, wobbleCirclePath } from './core/draw';
import { C, RGB, hex, mix, ramp, rgba } from './core/color';
import { E, H, W, lerp, seg } from './core/math';
import { hashInt } from './core/rng';
import { CENTER } from './core/geometry';
import { cosmicCam, toScreen } from './core/camera';
import { darkness } from './core/timeline';
import type { Textures } from './core/texture';
import { drawCharacter, drawGround, introPose } from './scenes/character';
import { drawNeurons } from './scenes/neurons';
import { drawPrismAct } from './scenes/prism';
import { drawFlower } from './scenes/sunflower';
import { drawGalaxy, horizonR } from './scenes/galaxy';
import { drawStars } from './scenes/starfield';
import { drawTunnel } from './scenes/tunnel';
import { drawEarthAct } from './scenes/earthrise';
import { drawFinale } from './scenes/finale';

const SUN_IN: RGB[] = [C.creamIn, hex('#F4C68E'), hex('#E08A7E'), hex('#8E4A86'), hex('#3A2A66'), hex('#13153A')];
const SUN_OUT: RGB[] = [C.creamOut, hex('#E9A873'), hex('#B8566F'), hex('#5A2D66'), hex('#1E1742'), hex('#05060F')];
const DUSK_IN: RGB[] = [C.creamIn, hex('#D9CCE0'), hex('#8E82B8'), hex('#3B3A72'), C.navyIn];
const DUSK_OUT: RGB[] = [C.creamOut, hex('#B9A7C4'), hex('#5E5690'), hex('#23244E'), C.navyOut];
const SPACE_IN = hex('#13153A');
const SPACE_OUT = hex('#05060F');
const BH_IN = hex('#0C0D22');
const BH_OUT = hex('#020308');
const EARTH_IN = hex('#0B0E24');
const EARTH_OUT = hex('#020308');

function bgColors(t: number): [RGB, RGB] {
  if (t < 6.2) return [C.creamIn, C.creamOut];
  if (t < 7.7) {
    // dusk falls through lavender rather than grey
    return [ramp(DUSK_IN, seg(t, 6.35, 7.7, E.inOutSine)), ramp(DUSK_OUT, seg(t, 6.2, 7.45, E.inOutSine))];
  }
  if (t < 13.85) return [C.navyIn, C.navyOut];
  if (t < 15.4) return [ramp(SUN_IN, seg(t, 14.15, 15.35)), ramp(SUN_OUT, seg(t, 13.85, 15.0))];
  if (t < 17.0) return [SPACE_IN, SPACE_OUT];
  if (t < 20.3) {
    const k = seg(t, 17.0, 18.4);
    return [mix(SPACE_IN, BH_IN, k), mix(SPACE_OUT, BH_OUT, k)];
  }
  if (t < 27.45) return [EARTH_IN, EARTH_OUT];
  // dawn: the galaxy's sunset played backwards, centre first
  return [ramp(SUN_IN, 1 - seg(t, 27.45, 27.98, E.inOutSine)), ramp(SUN_OUT, 1 - seg(t, 27.55, 28.12, E.inOutSine))];
}

function paintGround(ctx: Ctx, inner: RGB, outer: RGB): void {
  const g = ctx.createRadialGradient(W / 2, H * 0.47, 0, W / 2, H * 0.47, 1200);
  g.addColorStop(0, rgba(inner));
  g.addColorStop(1, rgba(outer));
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
}

/** A cream disc with a hand-cut edge that spreads out from a point (radial transition). */
function creamWash(ctx: Ctx, t: number, x: number, y: number, r: number): void {
  if (r <= 1) return;
  const boil = boilOf(t);
  ctx.save();
  ctx.beginPath();
  wobbleCirclePath(ctx, x, y, r, 5 + r * 0.01, 90, boil, 96, 2.2);
  ctx.clip();
  paintGround(ctx, C.creamIn, C.creamOut);
  ctx.restore();
}

function drawBackground(ctx: Ctx, t: number): void {
  const [inner, outer] = bgColors(t);
  paintGround(ctx, inner, outer);
  if (t >= 11.25 && t < 13.85) {
    const [x, y] = toScreen(cosmicCam(t), CENTER.x, CENTER.y);
    creamWash(ctx, t, x, y, lerp(0, 1800, seg(t, 11.25, 12.45, E.inOutCubic)));
  }
}

function drawSky(ctx: Ctx, t: number): void {
  if (t >= 13.9 && t < 21.0) {
    const cam = cosmicCam(t);
    const prev = cosmicCam(t - 0.025);
    const h = horizonR(t);
    const [lx, ly] = toScreen(cam, CENTER.x, CENTER.y);
    drawStars(ctx, t, {
      alpha: seg(t, 14.2, 15.4) * (1 - seg(t, 20.6, 21.0)),
      scale: Math.max(1, cam.zoom / 0.8),
      scalePrev: t > 20.1 ? Math.max(1, prev.zoom / 0.8) : undefined,
      rot: cam.rot,
      cx: lx,
      cy: ly,
      lens: h > 1 ? { x: lx, y: ly, r: h * cam.zoom } : undefined,
    });
  }
  if (t >= 22.8 && t < 27.5) {
    drawStars(ctx, t, {
      alpha: 0.75 * seg(t, 23.2, 24.2) * (1 - seg(t, 27.0, 27.4)),
      scale: 1 + 0.04 * seg(t, 24, 27),
      rot: 0.02 * seg(t, 23, 27),
    });
  }
}

function portal(ctx: Ctx, t: number, x: number, y: number, r: number): void {
  if (t >= 21.0) return;
  drawTunnel(ctx, t, { x, y, r });
}

function overlays(ctx: Ctx, t: number, tex: Textures): void {
  const dark = darkness(t);
  const boil = boilOf(t);
  ctx.globalCompositeOperation = 'source-over';
  const vc = mix(hex('#5A3F24'), [0, 0, 0], dark);
  const vg = ctx.createRadialGradient(W / 2, H / 2, 380, W / 2, H / 2, 1180);
  vg.addColorStop(0, rgba(vc, 0));
  vg.addColorStop(1, rgba(vc, 0.2 + 0.3 * dark));
  ctx.fillStyle = vg;
  ctx.fillRect(0, 0, W, H);
  // paper tooth
  ctx.globalCompositeOperation = 'soft-light';
  ctx.globalAlpha = lerp(0.8, 0.55, dark);
  ctx.drawImage(tex.paper, 0, 0, W, H);
  // grain, re-rolled 12x a second like hand-shot film
  ctx.globalCompositeOperation = 'overlay';
  ctx.globalAlpha = lerp(0.075, 0.11, dark);
  const hsh = hashInt(boil * 7919 + 13);
  const ox = hsh & 255;
  const oy = (hsh >>> 8) & 255;
  ctx.translate(ox, oy);
  ctx.fillStyle = tex.grain[boil & 3];
  ctx.fillRect(-ox, -oy, W, H);
  ctx.translate(-ox, -oy);
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1;
}

export function renderFilm(ctx: Ctx, t: number, tex: Textures): void {
  ctx.save();
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
  ctx.lineJoin = 'round';
  drawBackground(ctx, t);

  // Act I — creature, spark, network
  if (t < 5.4) drawGround(ctx, t, seg(t, 0, 0.35) * (1 - seg(t, 4.2, 5.1)));
  drawNeurons(ctx, t);
  if (t < 5.4) drawCharacter(ctx, introPose(t), t);

  // Act II — beam, prism, spectrum, rings, sunflower
  drawPrismAct(ctx, t);
  drawSky(ctx, t);
  drawFlower(ctx, t);

  // Act III — galaxy, black hole, tunnel, earthrise
  drawGalaxy(ctx, t, portal);
  if (t >= 21.0) drawTunnel(ctx, t);
  drawEarthAct(ctx, t);

  // Coda — back to paper
  drawFinale(ctx, t);

  overlays(ctx, t, tex);
  ctx.restore();
}
