// Shared anchor points so neighbouring scenes line up exactly at their seams.

/** Where the white beam enters / leaves the prism (screen coords, camera at rest). */
export const ENTRY = { x: 850, y: 536.5 };
export const EXIT = { x: 1070, y: 536.5 };
export const PRISM_V: [number, number][] = [
  [960, 346],
  [740, 727],
  [1180, 727],
];

/** Heading of the spectrum's centre ray; rings start where each ray leaves the prism. */
export const FAN_MID = 0.465;
export const THETA_S = FAN_MID - Math.PI / 2;

/** The seven spectral rings (red outermost). Same radii become petals, sepals, rim, seeds. */
export const RING_R = [330, 300, 270, 240, 210, 180, 150];
export const RING_W = 22;

/** Common centre of rings → sunflower → galaxy → black hole. */
export const CENTER = {
  x: EXIT.x - RING_R[3] * Math.cos(THETA_S),
  y: EXIT.y - RING_R[3] * Math.sin(THETA_S),
};

/** Scene-1 ground line and the little creature's resting spot. */
export const GROUND_Y = 700;
export const CREATURE_X = 810;
export const SPARK = { x: 1100, y: 505 };
