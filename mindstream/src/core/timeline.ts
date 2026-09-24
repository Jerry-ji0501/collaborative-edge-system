// Master timeline. All scenes read absolute film time; these keys keep the seams in one place.
import { E, seg } from './math';

export const FPS = 30;
export const DURATION = 30;
export const TOTAL_FRAMES = FPS * DURATION;

export const K = {
  walkStop: 2.9,
  sparkOn: 2.95,
  notice: 3.12,
  hopUp: 3.36,
  hopLand: 3.62,
  burst: 3.62,
  darkStart: 6.2,
  darkEnd: 7.7,
  impact: 8.05,
  morphStart: 11.05,
  galaxyStart: 13.8,
  collapse: 17.0,
  dive: 20.15,
  pillar: 22.74,
  glint: 26.45,
  zoomIn: 26.9,
  land: 28.55,
  turn: 28.75,
  end: 30,
};

/** 0 on paper, 1 in space. Drives glow blend modes and line colours. */
export function darkness(t: number): number {
  if (t < 6.2) return 0;
  if (t < 7.7) return seg(t, 6.3, 7.6, E.inOutSine);
  if (t < 11.3) return 1;
  if (t < 12.3) return 1 - seg(t, 11.3, 12.3, E.inOutSine);
  if (t < 13.9) return 0;
  if (t < 15.2) return seg(t, 13.9, 15.2, E.inOutSine);
  if (t < 27.45) return 1;
  if (t < 28.1) return 1 - seg(t, 27.45, 28.05, E.inOutSine);
  return 0;
}

export const SECTIONS: [number, string][] = [
  [0, 'Creature'],
  [4, 'Neurons'],
  [7.5, 'Prism'],
  [10, 'Rings'],
  [11.5, 'Sunflower'],
  [14, 'Galaxy'],
  [17, 'Black hole'],
  [20.5, 'Tunnel'],
  [23.5, 'Earthrise'],
  [27, 'Return'],
];
