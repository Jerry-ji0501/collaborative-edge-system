// The score: what sounds when. It is the film's cue sheet for the ear — beats come from the same
// timeline keys and seeded scene data the frames are drawn from, and every continuous layer is a pure
// function of film time (like every frame), so the soundtrack can be entered anywhere and always agrees.
import { FPS, K } from '../core/timeline';
import { E, clamp, lerp, seg } from '../core/math';
import { hash01 } from '../core/rng';
import { noise1 } from '../core/noise';
import { toScreen } from '../core/camera';
import { CREATURE_X, ENTRY, GROUND_Y, SPARK } from '../core/geometry';
import { HOPS, Hop, footfalls } from '../scenes/character';
import { branchTension, growthRate, neuronCues, signalLoad } from '../scenes/neurons';
import { beamFront } from '../scenes/prism';
import { horizonR } from '../scenes/galaxy';
import { ringPasses, tunnelSpeed } from '../scenes/tunnel';
import { earthCY } from '../scenes/earthrise';
import { dotTouchdowns, finaleCam } from '../scenes/finale';
import * as S from './synth';

/** A one-shot at film time t; `play` receives the audio-clock time of that moment. */
export interface Cue {
  t: number;
  play: (k: S.Kit, at: number) => void;
}
/** An AudioParam that follows a pure function of film time. */
export interface Lane {
  param: AudioParam;
  at: (t: number) => number;
}
interface Add {
  cue: (t: number, play: Cue['play']) => void;
  lane: (param: AudioParam, at: Lane['at']) => void;
}

const mtof = (m: number): number => 440 * Math.pow(2, (m - 69) / 12);
/** The first frame that shows a moment: a sound the eye can see coming lands exactly on it. */
const onFrame = (t: number): number => Math.ceil(t * FPS - 1e-6) / FPS;
/** Screen x → stereo position (kept a little inside the speakers). */
const panOf = (x: number): number => 0.8 * clamp((x - 960) / 960, -1, 1);

/** Eased, log-domain glide through [time, value] points — for filter cutoffs. */
function glideLog(t: number, pts: [number, number][]): number {
  if (t <= pts[0][0]) return pts[0][1];
  for (let i = 1; i < pts.length; i++) {
    const [t1, v1] = pts[i];
    if (t > t1) continue;
    const [t0, v0] = pts[i - 1];
    return v0 * Math.pow(v1 / v0, E.inOutSine((t - t0) / (t1 - t0)));
  }
  return pts[pts.length - 1][1];
}

// ---------------------------------------------------------------- I · the creature

function creature({ cue }: Add): void {
  // footsteps on the gait's own plants; the first ones come from off-screen left, before it is seen
  footfalls().forEach((f, i) => {
    const gain = (f.k > 0 ? 0.3 + 0.45 * f.k : 0.22) * (f.x < -30 ? 0.75 : 1);
    const pitch = (f.side < 0 ? 1 : 0.93) * (0.97 + 0.06 * hash01(i * 17 + 3));
    cue(onFrame(f.t), (k, at) => S.step(k, at, { gain, pan: panOf(f.x), pitch, seed: i * 31 + 7 }));
  });
  // it notices the spark: "叮", from where the spark hangs
  cue(onFrame(K.notice), (k, at) => S.ding(k, at, { f: mtof(81), gain: 0.2, pan: panOf(SPARK.x), decay: 1.6 }));
  hop(cue, HOPS.intro, panOf(CREATURE_X), 1);
}

/** Wind-up in the squash, the spring on take-off, both feet on the landing — each on its own frame. */
function hop(cue: Add['cue'], h: Hop, pan: number, level: number): void {
  const a = onFrame(h.tA);
  const up = onFrame(h.tUp);
  const down = onFrame(h.tLand);
  const lift = Math.pow(h.height / HOPS.intro.height, 0.3);
  cue(a, (k, at) => S.boing(k, { tA: at, tUp: at + up - a, tLand: at + down - a, lift, gain: 0.2 * level, pan }));
  cue(down, (k, at) => S.land(k, at, { gain: 0.9 * level, pan, seed: Math.round(h.tLand * 100) }));
}

// ---------------------------------------------------------------- II · the network, the beam

function network({ cue, lane }: Add, k: S.Kit): void {
  const { ac, to, buf } = k;
  // the spark bursts into a network, on the frame the creature lands
  cue(onFrame(K.burst), (kk, at) => S.burst(kk, at, { gain: 1, pan: panOf(SPARK.x) }));
  // every spark thrown, junction flaring and cell lighting up, from where it is on screen. These are
  // dense, so they keep their exact times (a frame grid would buzz at 30 Hz); the cells land on frames.
  const cells = [74, 76, 78, 81, 83, 86]; // they light up in turn: a rising D pentatonic
  let cell = 0;
  neuronCues().forEach((c, i) => {
    const pan = panOf(c.x);
    if (c.kind === 'flick') {
      cue(c.t, (kk, at) => S.flick(kk, at, { gain: 0.3 * c.a, pan, seed: i * 13 + 1 }));
    } else if (c.kind === 'flare') {
      const f = (1800 + 2200 * clamp((4.7 - c.r) / 1.8)) * (0.92 + 0.16 * hash01(i * 5 + 2));
      cue(c.t, (kk, at) => S.flare(kk, at, { gain: 2.4 * c.a, pan, f, seed: i * 7 + 3 }));
    } else {
      const f = mtof(cells[cell++ % cells.length]);
      cue(onFrame(c.t), (kk, at) => S.plip(kk, at, { gain: c.a, pan, f, seed: i * 3 + 9 }));
    }
  });

  // the living network's bed: crackle that follows the growth front and the signals in flight
  const cr = S.gain(ac, 0);
  S.bed(ac, buf.crackle, 21).connect(S.bq(ac, 'highpass', 1800)).connect(cr).connect(to.air);
  lane(cr.gain, (t) => 0.05 * growthRate(t) + 0.004 * signalLoad(t));
  const hiss = S.gain(ac, 0);
  S.bed(ac, buf.white, 22).connect(S.bq(ac, 'bandpass', 6500, 0.8)).connect(hiss).connect(to.air);
  lane(hiss.gain, (t) => 0.012 * growthRate(t) + 0.0012 * signalLoad(t));

  // the chosen branch pulled taut: a thin fifth rising an octave, A4 → A5 — the dominant of the
  // D-major ring the prism is about to strike
  const wg = S.gain(ac, 0);
  wg.connect(to.glass);
  const w1 = S.drone(ac, mtof(69));
  const w2 = S.drone(ac, mtof(76));
  w1.connect(wg);
  w2.connect(S.gain(ac, 0.5)).connect(wg);
  const rise = (t: number): number => 12 * E.inOutSine(branchTension(t).taut);
  lane(w1.frequency, (t) => mtof(69 + rise(t)));
  lane(w2.frequency, (t) => mtof(76 + rise(t)));
  lane(wg.gain, (t) => {
    const b = branchTension(t);
    return 0.02 * b.glow * (0.35 + 0.65 * b.taut);
  });

  // the white beam racing for the prism, left to right, rising into the strike
  const bp = S.bq(ac, 'bandpass', 500, 1.2);
  const bg = S.gain(ac, 0);
  const bpan = S.panner(ac, 0, to.air);
  S.bed(ac, buf.pink, 23).connect(bp).connect(bg).connect(bpan);
  const on = (t: number): number => (t < 7.55 || t > K.impact + 0.05 ? 0 : 1 - seg(t, K.impact, K.impact + 0.04));
  lane(bp.frequency, (t) => 500 * Math.pow(10, beamFront(t)));
  lane(bg.gain, (t) => 0.22 * on(t) * beamFront(t) ** 2);
  lane(bpan.pan, (t) => panOf(lerp(-120, ENTRY.x, beamFront(t))));
}

// ---------------------------------------------------------------- III · the prism

function prism({ cue, lane }: Add, k: S.Kit): void {
  const { ac, to } = k;
  // white light strikes the glass…
  cue(onFrame(K.impact), (kk, at) => S.strike(kk, at, { gain: 1, pan: panOf(ENTRY.x) }));
  // …and rings on as seven partials, one per spectral band (red lowest): a D-major chord that blooms
  // as the fan opens, shimmers like struck glass (a beat counted from the strike, so it is the same
  // every time), breathes with the rings (the same sin(2.6t + 0.8i) they breathe with), and hands over
  // to the pad as the rings become petals
  const chord = [62, 69, 74, 78, 81, 86, 88];
  chord.forEach((m, i) => {
    const g = S.gain(ac, 0);
    S.drone(ac, mtof(m)).connect(g).connect(S.panner(ac, -0.45 + 0.15 * i, to.glass));
    const amp = 0.04 * Math.pow(0.84, i);
    const tau = 3.2 - 0.25 * i;
    const beat = 0.7 + 0.14 * i;
    lane(g.gain, (t) => {
      if (t < K.impact || t > 12.4) return 0;
      const hit = 0.6 * Math.exp(-(t - K.impact) / 0.35);
      const fan = seg(t, 8.2, 8.95, E.outCubic) * (t < 9 ? 1 : Math.exp(-(t - 9) / tau));
      const shimmer = 0.8 + 0.2 * Math.cos(2 * Math.PI * beat * (t - K.impact));
      const breath = 1 + 0.25 * Math.sin(2.6 * t + 0.8 * i) * seg(t, 10.2, 10.8) * (1 - seg(t, 11.05, 11.8));
      return amp * (hit + fan) * shimmer * breath * (1 - seg(t, 11.0, 12.3, E.inOutSine));
    });
  });
}

// ---------------------------------------------------------------- IV · sunflower → galaxy → black hole

/** I (sunflower) → vi (sunset) → IV (galaxy) → i (collapse) → open fifths (black hole). */
const CHORDS: [number, number[]][] = [
  [11.0, [50, 57, 62, 66, 69, 76]],
  [13.8, [47, 54, 62, 66, 69, 73]],
  [15.4, [43, 50, 59, 66, 69, 74]],
  [17.0, [38, 45, 53, 62, 69, 74]],
  [18.4, [38, 45, 50, 57, 62, 74]],
];
/** The top voices thin out as the black hole deepens: [from, to, how much]. */
const THIN: ([number, number, number] | null)[] = [null, null, null, null, [19.2, 20.2, 0.6], [18.6, 19.6, 1]];

function pad({ lane }: Add, k: S.Kit): void {
  const { ac, to } = k;
  const wave = S.warmWave(ac);
  const lp1 = S.bq(ac, 'lowpass', 700, 0.6);
  const lp2 = S.bq(ac, 'lowpass', 700, 0.6);
  const out = S.gain(ac, 0);
  lp1.connect(lp2).connect(out).connect(to.space);
  // it spreads from the centre exactly as the cream wash does, then keeps widening into space
  const spread = (t: number): number => seg(t, 11.25, 12.45, E.inOutCubic) * (1 + 0.3 * seg(t, 13.9, 15.5));
  CHORDS[0][1].forEach((m0, j) => {
    // each voice glides a beat after the one below it, the way the rings unzip into petals
    const pitch = (t: number): number => {
      let m = m0;
      for (let c = 1; c < CHORDS.length; c++) {
        const t0 = CHORDS[c][0] + 0.12 * j;
        m = lerp(m, CHORDS[c][1][j], seg(t, t0, t0 + 1.3, E.inOutSine));
      }
      return m;
    };
    const level = 0.03 * [1, 0.85, 0.75, 0.65, 0.55, 0.4][j];
    const vg = S.gain(ac, level);
    vg.connect(lp1);
    const thin = THIN[j];
    if (thin) lane(vg.gain, (t) => level * (1 - thin[2] * seg(t, thin[0], thin[1])));
    for (const side of [-1, 1]) {
      const o = S.drone(ac, mtof(m0), wave);
      const p = S.panner(ac, 0, vg);
      o.connect(p);
      lane(o.frequency, (t) => mtof(pitch(t)));
      lane(o.detune, (t) => side * lerp(5, 11, seg(t, 13.9, 17)));
      lane(p.pan, (t) => side * spread(t) * (0.25 + 0.1 * j));
    }
  });
  // warm and open on paper, closing down through the sunset, dark and low at the black hole
  const cutoff = (t: number): number =>
    glideLog(t, [
      [11.05, 700],
      [12.6, 2400],
      [13.8, 2400],
      [15.4, 1400],
      [17.0, 1050],
      [18.3, 520],
      [20.2, 300],
      [20.8, 150],
    ]);
  lane(lp1.frequency, cutoff);
  lane(lp2.frequency, cutoff);
  lane(out.gain, (t) => {
    if (t < 11.0 || t > 20.9) return 0;
    // the sunflower breathes (1 + 0.014 sin 2.1(t−11) in the picture); the pad breathes with it
    const breathe = 1 + 0.1 * Math.sin(2.1 * (t - 11)) * seg(t, 12.2, 12.8) * (1 - seg(t, 13.9, 14.8));
    return seg(t, 11.05, 12.8, E.inOutSine) * breathe * (1 - seg(t, 20.15, 20.75, E.inOutSine));
  });
}

function blackHole({ lane }: Add, k: S.Kit): void {
  const { ac, to, buf } = k;
  // the rumble is born with the event horizon, swells with the camera's push, surges as we fall in
  const level = (t: number): number => {
    if (t < 17.2 || t > 21.2) return 0;
    const push = 0.75 + 0.25 * seg(t, 18.4, 20.15, E.inOutSine);
    const fall = 1 + 0.4 * seg(t, 20.15, 20.6, E.inCubic);
    return (horizonR(t) / 86) * push * fall * (1 - seg(t, 20.55, 21.15, E.inOutSine));
  };
  // D1 (with D2 for small speakers): it sags as the hole deepens, then is dragged up into the fall
  const f = (t: number): number => mtof(26) * (1 - 0.05 * seg(t, 18.4, 20.15)) * (1 + 0.8 * seg(t, 20.15, 21.0, E.inExpo));
  const sub = S.drone(ac, mtof(26));
  const sub2 = S.drone(ac, mtof(38));
  const sg = S.gain(ac, 0);
  const s2 = S.gain(ac, 0);
  sub.connect(sg).connect(to.sub);
  sub2.connect(s2).connect(to.sub);
  lane(sub.frequency, f);
  lane(sub2.frequency, (t) => 2 * f(t));
  lane(sg.gain, (t) => 0.1 * level(t));
  lane(s2.gain, (t) => 0.05 * level(t));
  // restless brown-noise rumble below 150 Hz
  const rl = S.bq(ac, 'lowpass', 90, 0.8);
  const rg = S.gain(ac, 0);
  S.bed(ac, buf.brown, 31).connect(rl).connect(rg).connect(to.sub);
  lane(rl.frequency, (t) => 80 + 60 * seg(t, 18.4, 20.4));
  lane(rg.gain, (t) => 0.2 * level(t) * (0.85 + 0.15 * noise1(t * 1.3, 3)));
  // a slowly heaving growl, the part of the rumble small speakers can carry
  const gb = S.bq(ac, 'bandpass', 95, 2.2);
  const gg = S.gain(ac, 0);
  S.bed(ac, buf.brown, 37).connect(gb).connect(gg).connect(to.air);
  lane(gb.frequency, (t) => 95 + 45 * seg(t, 18.4, 20.5));
  lane(gg.gain, (t) => 0.13 * level(t) * (0.55 + 0.45 * Math.sin(Math.PI * 0.55 * t) ** 2));
}

// ---------------------------------------------------------------- V · tunnel → earthrise → home

function tunnel({ cue, lane }: Add, k: S.Kit): void {
  const { ac, to, buf } = k;
  // falling through the horizon: a rush that climbs with the camera's exponential push
  const fall = (t: number): number => seg(t, 20.15, 21.0, E.inExpo);
  const db = S.bq(ac, 'bandpass', 250, 0.8);
  const dg = S.gain(ac, 0);
  S.stereoBed(ac, buf.pink, 41).connect(db).connect(dg).connect(to.air);
  lane(db.frequency, (t) => 250 * Math.pow(12, fall(t)));
  lane(dg.gain, (t) => (t < 20.1 || t > 21.6 ? 0 : 0.22 * fall(t) * (1 - seg(t, 20.95, 21.5, E.inOutSine))));

  // the tunnel's wind: a soft body and three wandering whistles, riding the forward speed; as the rings
  // condense into the light the air is drawn in after them and is gone
  const speed = (t: number): number => tunnelSpeed(t) / 3;
  const gather = (t: number): number => seg(t, 22.45, 22.8, E.inCubic);
  const wind = (t: number): number => seg(t, 20.3, 20.95, E.inOutSine) * (1 - seg(t, 22.45, 22.85, E.inCubic));
  const bl = S.bq(ac, 'lowpass', 600, 0.5);
  const bg = S.gain(ac, 0);
  S.stereoBed(ac, buf.pink, 43).connect(bl).connect(bg).connect(to.air);
  lane(bl.frequency, (t) => 450 + 900 * speed(t) + 3000 * gather(t));
  lane(bg.gain, (t) => 0.16 * wind(t) * (0.45 + 0.55 * speed(t)));
  const whistles: [number, number][] = [
    [520, -0.55],
    [1150, 0.5],
    [2300, 0.1],
  ];
  whistles.forEach(([f0, p], i) => {
    const b = S.bq(ac, 'bandpass', f0, 7);
    const g = S.gain(ac, 0);
    S.bed(ac, buf.pink, 47 + i).connect(b).connect(g).connect(S.panner(ac, p, to.air));
    lane(b.frequency, (t) => f0 * (1 + 0.18 * noise1(t * (0.6 + 0.3 * i), 50 + i)) * (0.8 + 0.25 * speed(t)) * (1 + 1.2 * gather(t)));
    lane(g.gain, (t) => 0.8 * wind(t) * speed(t) * (0.7 + 0.3 * noise1(t * 2.1, 60 + i)));
  });
  // each ring rushing past the frame edge: a gust (softer while the tunnel is still seen through the hole)
  for (const r of ringPasses()) {
    const gain = 0.35 * (0.35 + 0.65 * seg(r.t, 20.45, 21.0)) * (r.type === 0 ? 1 : 0.8);
    const bright = r.type === 2 ? 1.3 : r.type === 3 ? 1.15 : 1;
    cue(r.t, (kk, at) => S.gust(kk, at, { gain, pan: 1.2 * (hash01(r.seed) - 0.5), bright, seed: r.seed }));
  }
}

function earthrise({ lane }: Add, k: S.Kit): void {
  const { ac, to } = k;
  // the stillest sound of the film: A3 with a whisper of its octave and twelfth, breathing slowly. It
  // appears with the pillar of light, lies down with it into the horizon, lifts a little with the
  // rising Earth, and is gone before the orb comes down.
  const lp = S.bq(ac, 'lowpass', 1800, 0.5);
  const g = S.gain(ac, 0);
  lp.connect(g).connect(to.space);
  const partials: [number, number][] = [
    [57, 1],
    [69, 0.3],
    [76, 0.1],
  ];
  for (const [m, a] of partials) S.drone(ac, mtof(m)).connect(S.gain(ac, a)).connect(lp);
  lane(g.gain, (t) => {
    if (t < K.pillar || t > 28.1) return 0;
    const risen = (1060 - earthCY(t)) / 548;
    const breath = 1 + 0.12 * Math.sin(2 * Math.PI * 0.2 * (t - K.pillar));
    return 0.026 * seg(t, K.pillar, 23.9, E.inOutSine) * (0.8 + 0.2 * risen) * breath * (1 - seg(t, 27.0, 28.0, E.inOutSine));
  });
  // the sun-glint on the ocean: a faint high shimmer that pulses with its sparkle (1 + 0.15 sin 11t)
  const gl = S.gain(ac, 0);
  S.drone(ac, mtof(93)).connect(gl).connect(to.glass);
  lane(gl.gain, (t) =>
    t < K.glint || t > 27.5 ? 0 : 0.008 * seg(t, K.glint, 26.8, E.outBack) * (1 + 0.15 * Math.sin(t * 11)) * (1 - seg(t, 26.9, 27.45)),
  );
}

function finale({ cue }: Add): void {
  // the orb lands, rebounds twice, and later hops along with the creature: a soft D5 each time
  for (const d of dotTouchdowns()) {
    cue(onFrame(d.t), (k, at) => S.tink(k, at, { f: mtof(74), gain: 0.14 * d.k, pan: panOf(d.x) }));
  }
  // the creature sees it and its bulb lights: the opening bell again, now resolved from A to D
  const pan = panOf(toScreen(finaleCam(K.bulb), CREATURE_X, GROUND_Y)[0]);
  cue(onFrame(K.bulb), (k, at) => S.ding(k, at, { f: mtof(86), gain: 0.12, pan, decay: 2.4 }));
  hop(cue, HOPS.finale, pan, 0.55);
}

// ----------------------------------------------------------------

export function buildScore(k: S.Kit): { cues: Cue[]; lanes: Lane[] } {
  const cues: Cue[] = [];
  const lanes: Lane[] = [];
  const add: Add = {
    cue: (t, play) => cues.push({ t, play }),
    lane: (param, at) => lanes.push({ param, at }),
  };
  creature(add);
  network(add, k);
  prism(add, k);
  pad(add, k);
  blackHole(add, k);
  tunnel(add, k);
  earthrise(add, k);
  finale(add);
  return { cues: cues.sort((a, b) => a.t - b.t), lanes };
}
