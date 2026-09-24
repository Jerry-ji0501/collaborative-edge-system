// Instruments. Everything is synthesised when the audio starts — noise colours, the crackle of the
// network, the reverb tails — and every voice is built from oscillators, filters and envelopes.
// No samples, no files. Randomness is seeded, so the live mix and an offline render sound identical.
import { hash01, mulberry32 } from '../core/rng';

export type BusName = 'paper' | 'glass' | 'space' | 'air' | 'sub';

export interface Buffers {
  white: AudioBuffer;
  pink: AudioBuffer;
  brown: AudioBuffer;
  crackle: AudioBuffer;
}

/** What a voice needs: the context, a destination per kind of sound, and the shared buffers. */
export interface Kit {
  ac: BaseAudioContext;
  to: Record<BusName, AudioNode>;
  buf: Buffers;
}

// ---------------------------------------------------------------- procedural buffers

/** A mono buffer that loops without a seam: its tail is cross-faded (equal power) into its head. */
function loopBuffer(ac: BaseAudioContext, seconds: number, seed: number, gen: (rnd: () => number) => () => number): AudioBuffer {
  const sr = ac.sampleRate;
  const n = Math.round(seconds * sr);
  const x = Math.round(0.05 * sr);
  const next = gen(mulberry32(seed));
  const raw = new Float32Array(n + x);
  for (let i = 0; i < raw.length; i++) raw[i] = next();
  const b = ac.createBuffer(1, n, sr);
  const d = b.getChannelData(0);
  d.set(raw.subarray(0, n));
  for (let i = 0; i < x; i++) {
    const a = (i / x) * (Math.PI / 2);
    d[i] = raw[i] * Math.sin(a) + raw[n + i] * Math.cos(a);
  }
  return b;
}

const white = (rnd: () => number) => () => rnd() * 2 - 1;

/** Pink noise (Paul Kellet's filter): the hush of air and wind. */
const pink = (rnd: () => number) => {
  let b0 = 0;
  let b1 = 0;
  let b2 = 0;
  let b3 = 0;
  let b4 = 0;
  let b5 = 0;
  let b6 = 0;
  return () => {
    const w = rnd() * 2 - 1;
    b0 = 0.99886 * b0 + w * 0.0555179;
    b1 = 0.99332 * b1 + w * 0.0750759;
    b2 = 0.969 * b2 + w * 0.153852;
    b3 = 0.8665 * b3 + w * 0.3104856;
    b4 = 0.55 * b4 + w * 0.5329522;
    b5 = -0.7616 * b5 - w * 0.016898;
    const p = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362;
    b6 = w * 0.115926;
    return p * 0.11;
  };
};

/** Brown noise (leaky integral of white): the body of a rumble. */
const brown = (rnd: () => number) => {
  let l = 0;
  return () => {
    l = (l + 0.02 * (rnd() * 2 - 1)) / 1.02;
    return l * 3.5;
  };
};

/**
 * Electric crackle: sparse clicks (a sharp alternating ring each), arriving at random, often in
 * tight bursts like an arc spitting — the grain of every spark in the network.
 */
const crackle = (sr: number) => (rnd: () => number) => {
  let wait = 0;
  let amp = 0;
  return () => {
    if (--wait <= 0) {
      amp = (0.15 + 0.85 * Math.pow(rnd(), 2.5)) * (rnd() < 0.5 ? -1 : 1);
      wait = rnd() < 0.35 ? 1 + rnd() * 0.002 * sr : (-Math.log(1 - rnd()) * sr) / 220;
    }
    const v = amp;
    amp *= -0.55;
    return v;
  };
};

export function makeBuffers(ac: BaseAudioContext): Buffers {
  return {
    white: loopBuffer(ac, 2, 11, white),
    pink: loopBuffer(ac, 6, 12, pink),
    brown: loopBuffer(ac, 6, 13, brown),
    crackle: loopBuffer(ac, 4, 14, crackle(ac.sampleRate)),
  };
}

/**
 * A synthetic reverb tail: decorrelated noise per ear under an exponential decay (rt60), passed
 * through a one-pole low-pass that closes as it fades — high frequencies die first, as in a real room.
 */
export function impulse(ac: BaseAudioContext, seconds: number, rt60: number, pre: number, bright: number, seed: number): AudioBuffer {
  const sr = ac.sampleRate;
  const n = Math.round(seconds * sr);
  const b = ac.createBuffer(2, n, sr);
  for (let c = 0; c < 2; c++) {
    const rnd = mulberry32(seed + c * 101);
    const d = b.getChannelData(c);
    let lp = 0;
    for (let i = 0; i < n; i++) {
      const t = i / sr - pre;
      if (t < 0) continue;
      const k = Math.min(1, bright * Math.exp(-t / (rt60 * 0.3)) + 0.03);
      lp += k * (rnd() * 2 - 1 - lp);
      d[i] = lp * Math.exp((-6.9 * t) / rt60) * Math.min(1, t / 0.003);
    }
  }
  return b;
}

/** A soft, warm saw (harmonics fall as n^-1.5): the pad's voice before the filter shapes it. */
export function warmWave(ac: BaseAudioContext): PeriodicWave {
  const n = 24;
  const re = new Float32Array(n + 1);
  const im = new Float32Array(n + 1);
  for (let h = 1; h <= n; h++) im[h] = Math.pow(h, -1.5);
  return ac.createPeriodicWave(re, im);
}

// ---------------------------------------------------------------- building blocks

export function gain(ac: BaseAudioContext, v: number): GainNode {
  const g = ac.createGain();
  g.gain.value = v;
  return g;
}

export function bq(ac: BaseAudioContext, type: BiquadFilterType, f: number, q = 0.7): BiquadFilterNode {
  const b = ac.createBiquadFilter();
  b.type = type;
  b.frequency.value = f;
  b.Q.value = q;
  return b;
}

export function panner(ac: BaseAudioContext, x: number, to?: AudioNode): StereoPannerNode {
  const p = ac.createStereoPanner();
  p.pan.value = Math.max(-1, Math.min(1, x));
  if (to) p.connect(to);
  return p;
}

/** Percussive gain: a linear rise to `peak`, then an 80 dB exponential fall over `decay`. */
export function perc(ac: BaseAudioContext, at: number, peak: number, attack: number, decay: number): GainNode {
  const g = ac.createGain();
  const p = Math.max(peak, 1e-6);
  g.gain.setValueAtTime(0, at);
  g.gain.linearRampToValueAtTime(p, at + attack);
  g.gain.exponentialRampToValueAtTime(p * 1e-4, at + attack + decay);
  g.gain.setValueAtTime(0, at + attack + decay);
  return g;
}

/** An oscillator for one note, from `at` to `end`. */
export function osc(ac: BaseAudioContext, f: number, at: number, end: number, type: OscillatorType = 'sine'): OscillatorNode {
  const o = ac.createOscillator();
  o.type = type;
  o.frequency.setValueAtTime(f, at);
  o.start(at);
  o.stop(end);
  return o;
}

/** A looped buffer from a seeded offset, so no two grains begin alike. */
export function grain(ac: BaseAudioContext, buf: AudioBuffer, at: number, end: number, seed: number): AudioBufferSourceNode {
  const s = ac.createBufferSource();
  s.buffer = buf;
  s.loop = true;
  s.start(at, hash01(seed) * buf.duration);
  s.stop(end);
  return s;
}

/** A free-running oscillator (a sine, or a custom wave) whose parameters a lane will drive. */
export function drone(ac: BaseAudioContext, f: number, wave?: PeriodicWave): OscillatorNode {
  const o = ac.createOscillator();
  if (wave) o.setPeriodicWave(wave);
  o.frequency.value = f;
  o.start();
  return o;
}

/** A free-running noise loop for a lane-driven bed. */
export function bed(ac: BaseAudioContext, buf: AudioBuffer, seed: number): AudioBufferSourceNode {
  const s = ac.createBufferSource();
  s.buffer = buf;
  s.loop = true;
  s.start(0, hash01(seed) * buf.duration);
  return s;
}

/** Two decorrelated noise loops spread left and right: a stereo bed for air and wind. */
export function stereoBed(ac: BaseAudioContext, buf: AudioBuffer, seed: number, width = 0.7): GainNode {
  const sum = gain(ac, 1);
  bed(ac, buf, seed).connect(panner(ac, -width, sum));
  bed(ac, buf, seed + 1).connect(panner(ac, width, sum));
  return sum;
}

// ---------------------------------------------------------------- one-shots: the creature

/** A soft felt foot on paper: a falling thump with a breath of scuff. */
export function step(k: Kit, at: number, v: { gain: number; pan: number; pitch: number; seed: number }): void {
  const { ac } = k;
  const out = panner(ac, v.pan, k.to.paper);
  const o = osc(ac, 230 * v.pitch, at, at + 0.24);
  o.frequency.exponentialRampToValueAtTime(92 * v.pitch, at + 0.07);
  o.connect(perc(ac, at, 0.5 * v.gain, 0.004, 0.18)).connect(out);
  grain(ac, k.buf.white, at, at + 0.08, v.seed)
    .connect(bq(ac, 'bandpass', 1500 * v.pitch, 0.8))
    .connect(perc(ac, at, 0.1 * v.gain, 0.002, 0.05))
    .connect(out);
}

/** Both feet meet the ground, one a hair after the other. */
export function land(k: Kit, at: number, v: { gain: number; pan: number; seed: number }): void {
  step(k, at, { gain: v.gain, pan: v.pan - 0.05, pitch: 0.8, seed: v.seed });
  step(k, at + 0.014, { gain: 0.8 * v.gain, pan: v.pan + 0.05, pitch: 0.87, seed: v.seed + 1 });
}

/** "叮" — a small clear bell: a pure fundamental (a slow-beating pair), a soft octave and a glassy glint. */
export function ding(k: Kit, at: number, v: { f: number; gain: number; pan: number; decay: number }): void {
  const { ac } = k;
  const out = panner(ac, v.pan, k.to.glass);
  const end = at + v.decay + 0.05;
  const partials: [number, number, number][] = [
    [1, 0.5, 1],
    [1.0011, 0.5, 0.9],
    [2, 0.3, 0.45],
    [3, 0.07, 0.3],
    [5.43, 0.15, 0.06],
  ];
  for (const [r, a, d] of partials) osc(ac, v.f * r, at, end).connect(perc(ac, at, a * v.gain, 0.0015, v.decay * d)).connect(out);
}

/**
 * The hop. In the squash the spring winds down; at take-off it lets go — pitch and a resonant
 * formant leap up and wobble ("b-oi-oing"), then ring away through the fall.
 */
export function boing(k: Kit, v: { tA: number; tUp: number; tLand: number; lift: number; gain: number; pan: number }): void {
  const { ac } = k;
  const { tA, tUp, tLand } = v;
  const top = tUp + (tLand - tUp) * 0.45;
  const end = tLand + 0.3;
  const f0 = 150 * v.lift;
  const o = osc(ac, f0 * 0.95, tA, end, 'triangle');
  const fr = o.frequency;
  fr.exponentialRampToValueAtTime(f0 * 0.8, tUp);
  fr.exponentialRampToValueAtTime(f0 * 2.6, top);
  fr.exponentialRampToValueAtTime(f0 * 2.2, end);
  // the spring's wobble
  const wob = osc(ac, 13, tA, end);
  const depth = ac.createGain();
  depth.gain.setValueAtTime(0, tA);
  depth.gain.linearRampToValueAtTime(f0 * 0.2, tUp + 0.02);
  depth.gain.exponentialRampToValueAtTime(f0 * 0.02, end);
  wob.connect(depth).connect(fr);
  const lp = bq(ac, 'lowpass', f0 * 3, 8);
  lp.frequency.setValueAtTime(f0 * 3, tA);
  lp.frequency.exponentialRampToValueAtTime(f0 * 2.5, tUp);
  lp.frequency.exponentialRampToValueAtTime(f0 * 10, top);
  lp.frequency.exponentialRampToValueAtTime(f0 * 5, end);
  const g = ac.createGain();
  g.gain.setValueAtTime(0, tA);
  g.gain.linearRampToValueAtTime(0.12 * v.gain, tUp);
  g.gain.linearRampToValueAtTime(v.gain, tUp + 0.015);
  g.gain.setTargetAtTime(0, top, (tLand - top) * 0.45);
  g.gain.setValueAtTime(0, end);
  o.connect(lp).connect(g).connect(panner(ac, v.pan, k.to.paper));
}

// ---------------------------------------------------------------- one-shots: the network

/** The spark bursts into a network: a falling fizz, a shower of crackle and a soft electric thump. */
export function burst(k: Kit, at: number, v: { gain: number; pan: number }): void {
  const { ac } = k;
  const out = panner(ac, v.pan, k.to.air);
  const bp = bq(ac, 'bandpass', 5500, 1.6);
  bp.frequency.setValueAtTime(5500, at);
  bp.frequency.exponentialRampToValueAtTime(1400, at + 0.6);
  grain(ac, k.buf.white, at, at + 0.9, 7).connect(bp).connect(perc(ac, at, 0.14 * v.gain, 0.006, 0.8)).connect(out);
  grain(ac, k.buf.crackle, at, at + 1, 11)
    .connect(bq(ac, 'highpass', 1600))
    .connect(perc(ac, at, 0.45 * v.gain, 0.002, 0.95))
    .connect(out);
  const o = osc(ac, 160, at, at + 0.5);
  o.frequency.exponentialRampToValueAtTime(55, at + 0.35);
  o.connect(perc(ac, at, 0.22 * v.gain, 0.004, 0.42)).connect(out);
}

/** A trio of sparks flicked off a travelling signal: a few milliseconds of band-passed crackle. */
export function flick(k: Kit, at: number, v: { gain: number; pan: number; seed: number }): void {
  const { ac } = k;
  const dur = 0.035 + 0.04 * hash01(v.seed * 7 + 1);
  grain(ac, k.buf.crackle, at, at + dur + 0.01, v.seed)
    .connect(bq(ac, 'bandpass', 2600 + 3800 * hash01(v.seed * 13 + 5), 1.3))
    .connect(perc(ac, at, v.gain, 0.001, dur))
    .connect(panner(ac, v.pan, k.to.air));
}

/** A junction flaring as a signal passes: a resonant "tsip" of noise, higher for smaller nodes. */
export function flare(k: Kit, at: number, v: { gain: number; pan: number; f: number; seed: number }): void {
  const { ac } = k;
  grain(ac, k.buf.white, at, at + 0.07, v.seed)
    .connect(bq(ac, 'bandpass', v.f, 14))
    .connect(perc(ac, at, v.gain, 0.001, 0.05))
    .connect(panner(ac, v.pan, k.to.air));
}

/** A satellite cell popping alight: a quick falling plip over its own crackle. */
export function plip(k: Kit, at: number, v: { gain: number; pan: number; f: number; seed: number }): void {
  const { ac } = k;
  const o = osc(ac, v.f * 1.9, at, at + 0.3);
  o.frequency.exponentialRampToValueAtTime(v.f, at + 0.045);
  o.connect(perc(ac, at, 0.1 * v.gain, 0.003, 0.22)).connect(panner(ac, v.pan, k.to.air));
  flick(k, at, { gain: 0.15 * v.gain, pan: v.pan, seed: v.seed });
}

// ---------------------------------------------------------------- one-shots: prism, tunnel, finale

/**
 * White light strikes the glass: a bright click and the prism's own ring — inharmonic glass modes,
 * each a close pair so the ring shimmers.
 */
export function strike(k: Kit, at: number, v: { gain: number; pan: number }): void {
  const { ac } = k;
  const out = panner(ac, v.pan, k.to.glass);
  grain(ac, k.buf.white, at, at + 0.05, 3)
    .connect(bq(ac, 'highpass', 3200))
    .connect(perc(ac, at, 0.3 * v.gain, 0.0006, 0.02))
    .connect(out);
  const f0 = 1318.51; // E6
  const modes: [number, number, number][] = [
    [1, 0.2, 2.2],
    [2.32, 0.12, 0.9],
    [4.25, 0.07, 0.45],
    [6.63, 0.045, 0.22],
  ];
  for (const [r, a, d] of modes) {
    for (const beat of [-0.6, 0.6]) {
      osc(ac, f0 * r + beat * r, at, at + d + 0.05)
        .connect(perc(ac, at, 0.5 * a * v.gain, 0.001, d))
        .connect(out);
    }
  }
}

/** A ring rushing past the camera: a swell of air that falls in pitch as it goes by. */
export function gust(k: Kit, at: number, v: { gain: number; pan: number; bright: number; seed: number }): void {
  const { ac } = k;
  const t0 = at - 0.06;
  const bp = bq(ac, 'bandpass', 2600 * v.bright, 1.4);
  bp.frequency.setValueAtTime(2600 * v.bright, t0);
  bp.frequency.exponentialRampToValueAtTime(700 * v.bright, at + 0.25);
  const g = ac.createGain();
  g.gain.setValueAtTime(0, t0);
  g.gain.linearRampToValueAtTime(v.gain, at);
  g.gain.exponentialRampToValueAtTime(v.gain * 1e-3, at + 0.26);
  g.gain.setValueAtTime(0, at + 0.26);
  grain(ac, k.buf.pink, t0, at + 0.27, v.seed).connect(bp).connect(g).connect(panner(ac, v.pan, k.to.air));
}

/** The little orb meeting the ground: a round glassy note over a tiny paper thump. */
export function tink(k: Kit, at: number, v: { f: number; gain: number; pan: number }): void {
  const { ac } = k;
  const out = panner(ac, v.pan, k.to.glass);
  const partials: [number, number, number][] = [
    [1, 1, 0.9],
    [2.01, 0.22, 0.35],
    [4.1, 0.06, 0.12],
  ];
  for (const [r, a, d] of partials) osc(ac, v.f * r, at, at + d + 0.05).connect(perc(ac, at, a * v.gain, 0.003, d)).connect(out);
  const o = osc(ac, 300, at, at + 0.08);
  o.frequency.exponentialRampToValueAtTime(140, at + 0.04);
  o.connect(perc(ac, at, 0.4 * v.gain, 0.002, 0.06)).connect(out);
}
