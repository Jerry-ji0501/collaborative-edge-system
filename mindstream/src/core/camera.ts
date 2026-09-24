// Cameras. Each act has one continuous camera function of time, so pushes carry momentum across seams.
import { CENTER } from './geometry';
import { E, H, W, lerp, seg } from './math';
import type { Ctx } from './draw';

export interface Cam {
  x: number;
  y: number;
  zoom: number;
  rot: number;
}

export function applyCam(ctx: Ctx, c: Cam): void {
  ctx.translate(W / 2, H / 2);
  if (c.rot) ctx.rotate(c.rot);
  ctx.scale(c.zoom, c.zoom);
  ctx.translate(-c.x, -c.y);
}

export function toScreen(c: Cam, x: number, y: number): [number, number] {
  const dx = (x - c.x) * c.zoom;
  const dy = (y - c.y) * c.zoom;
  const cs = Math.cos(c.rot);
  const sn = Math.sin(c.rot);
  return [W / 2 + dx * cs - dy * sn, H / 2 + dx * sn + dy * cs];
}

/**
 * The long middle-act camera: prism → rings → sunflower → galaxy → black hole → dive.
 * It is one function so every push flows into the next.
 */
export function cosmicCam(t: number): Cam {
  let x = W / 2;
  let y = H / 2;
  let zoom = 1;
  const k1 = seg(t, 9.6, 11.0, E.inOutCubic);
  x = lerp(x, CENTER.x, k1);
  y = lerp(y, CENTER.y, k1);
  zoom = lerp(1, 1.12, k1);
  const k2 = seg(t, 11.8, 13.4, E.inOutSine);
  y += 55 * k2;
  zoom *= lerp(1, 0.93, k2);
  const k3 = seg(t, 13.8, 15.5, E.inOutCubic);
  y = lerp(y, CENTER.y, k3);
  zoom = lerp(zoom, 0.8, k3);
  zoom *= lerp(1, 1.18, seg(t, 15.3, 17.3, E.inOutSine));
  zoom *= lerp(1, 1.18, seg(t, 17.0, 18.4, E.inOutCubic));
  zoom *= lerp(1, 1.25, seg(t, 18.4, 20.2, E.inOutSine));
  zoom *= lerp(1, 13, seg(t, 20.15, 21.0, E.inExpo));
  const rot = -0.05 * seg(t, 14.0, 20.0, E.inOutSine);
  return { x, y, zoom, rot };
}

/** Disk-plane view of the galaxy / accretion disk: inclination ι (y-squash) and tilt β. */
export function diskView(t: number): { iota: number; beta: number } {
  let iota = lerp(1, 0.58, seg(t, 14.1, 15.6, E.inOutSine));
  iota = lerp(iota, 0.3, seg(t, 17.0, 18.3, E.inOutCubic));
  iota = lerp(iota, 1, seg(t, 20.1, 20.8, E.inOutCubic));
  let beta = lerp(0, -0.32, seg(t, 14.1, 15.6, E.inOutSine));
  beta = lerp(beta, -0.14, seg(t, 17.0, 18.3, E.inOutCubic));
  beta = lerp(beta, 0, seg(t, 20.1, 20.8, E.inOutCubic));
  return { iota, beta };
}

/** Project a disk-plane polar point to world coordinates. */
export function projectDisk(r: number, th: number, iota: number, beta: number, out: [number, number]): void {
  const px = r * Math.cos(th);
  const py = r * Math.sin(th) * iota;
  const cb = Math.cos(beta);
  const sb = Math.sin(beta);
  out[0] = CENTER.x + px * cb - py * sb;
  out[1] = CENTER.y + px * sb + py * cb;
}
