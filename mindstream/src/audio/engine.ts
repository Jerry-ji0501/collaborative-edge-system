// The soundtrack engine: Web Audio, synthesised live and slaved to the film clock.
//  · One-shots (footsteps, bells, crackle…) are queued a little ahead on the audio clock, each timed
//    to the moment its frame reaches the screen.
//  · Lanes (pad, wind, rumble…) are AudioParams set from pure functions of film time, once per frame.
// Pausing, seeking and looping just re-read the score from the new playhead: nothing accumulates.
import { DURATION, FPS, TOTAL_FRAMES } from '../core/timeline';
import { BusName, Kit, bq, gain, impulse, makeBuffers } from './synth';
import { Cue, buildScore } from './score';

const LOOKAHEAD = 0.25; // film seconds kept queued ahead of the playhead
const DISPLAY_LAG = 0.025; // a frame's clock time → light: the rAF, the next vsync, scan-out
const BUSES: BusName[] = ['paper', 'glass', 'space', 'air', 'sub'];
/** Dry level and reverb sends per bus: the paper world is close and small; space is vast. */
const SENDS: Record<BusName, [dry: number, room: number, hall: number]> = {
  paper: [1, 0.22, 0.06],
  glass: [1, 0.08, 0.42],
  space: [0.9, 0, 0.75],
  air: [1, 0.05, 0.28],
  sub: [1, 0, 0],
};

const wrap = (u: number): number => ((u % DURATION) + DURATION) % DURATION;

/** An AudioParam following a function of film time: one point per frame, unchanged values skipped. */
class Lane {
  private last = NaN;
  private lastAt = 0;
  private held = false;
  constructor(
    private param: AudioParam,
    private fn: (t: number) => number,
  ) {}
  reset(t: number, at: number): void {
    const v = this.fn(t);
    this.param.cancelScheduledValues(at);
    this.param.setValueAtTime(v, at);
    this.last = v;
    this.lastAt = at;
    this.held = false;
  }
  to(t: number, at: number): void {
    const v = this.fn(t);
    if (v === this.last) {
      this.lastAt = at;
      this.held = true;
      return;
    }
    // after a flat stretch, pin the ramp's start so it spans one frame, not the whole stretch
    if (this.held) this.param.setValueAtTime(this.last, this.lastAt);
    this.param.linearRampToValueAtTime(v, at);
    this.last = v;
    this.lastAt = at;
    this.held = false;
  }
}

/** Everything that sounds in one audio context: buses, reverbs, the score's instruments. */
class Mixer {
  readonly out: GainNode; // the fader that mute and pause ride on
  readonly cues: Cue[];
  private readonly lanes: Lane[];
  private readonly bus = {} as Record<BusName, GainNode>;
  private readonly kit: Kit;
  private shotGains = {} as Record<BusName, GainNode>;
  shots: Kit; // one-shots of the current run, dropped together on pause or seek

  constructor(readonly ac: BaseAudioContext) {
    const comp = ac.createDynamicsCompressor();
    comp.threshold.value = -14;
    comp.knee.value = 12;
    comp.ratio.value = 3;
    comp.attack.value = 0.004;
    comp.release.value = 0.25;
    const master = bq(ac, 'highpass', 22);
    this.out = gain(ac, 0);
    master.connect(comp).connect(this.out).connect(ac.destination);
    // synthetic rooms: a small warm one for the paper world, a long dark hall for space
    const room = ac.createConvolver();
    room.buffer = impulse(ac, 0.9, 0.7, 0.006, 0.8, 71);
    const hall = ac.createConvolver();
    hall.buffer = impulse(ac, 3.6, 3.8, 0.022, 0.55, 97);
    room.connect(master);
    hall.connect(master);
    for (const name of BUSES) {
      const [dry, r, h] = SENDS[name];
      const b = gain(ac, 1);
      b.connect(gain(ac, dry)).connect(master);
      if (r) b.connect(gain(ac, r)).connect(room);
      if (h) b.connect(gain(ac, h)).connect(hall);
      this.bus[name] = b;
    }
    this.kit = { ac, to: this.bus, buf: makeBuffers(ac) };
    const score = buildScore(this.kit);
    this.cues = score.cues;
    this.lanes = score.lanes.map((l) => new Lane(l.param, l.at));
    this.shots = this.freshShots();
  }

  private freshShots(): Kit {
    this.shotGains = {} as Record<BusName, GainNode>;
    for (const name of BUSES) {
      this.shotGains[name] = gain(this.ac, 1);
      this.shotGains[name].connect(this.bus[name]);
    }
    return { ...this.kit, to: { ...this.shotGains } };
  }

  /** Silence the one-shots already queued — they belong to where we were — and start a fresh set. */
  dropShots(at: number): void {
    const old = this.shotGains;
    for (const name of BUSES) {
      old[name].gain.setValueAtTime(1, at);
      old[name].gain.linearRampToValueAtTime(0, at + 0.03);
    }
    setTimeout(() => BUSES.forEach((n) => old[n].disconnect()), 300);
    this.shots = this.freshShots();
  }

  /** Ramp the fader to `v` from wherever it is now. */
  fade(v: number, at: number, dur: number): void {
    const p = this.out.gain;
    p.cancelScheduledValues(at);
    p.setValueAtTime(p.value, at);
    p.linearRampToValueAtTime(v, at + dur);
  }

  resetLanes(t: number, at: number): void {
    for (const l of this.lanes) l.reset(t, at);
  }

  /** Queue the film-time window (from, until]. Times are unwrapped (loops keep counting up). */
  schedule(from: number, until: number, at: (u: number) => number): void {
    for (let n = Math.floor(from / DURATION); n * DURATION <= until; n++) {
      const base = n * DURATION;
      for (const c of this.cues) {
        const u = base + c.t;
        if (u > from && u <= until) c.play(this.shots, at(u));
      }
    }
    for (let f = Math.floor(from * FPS) + 1; f <= Math.floor(until * FPS + 1e-9); f++) {
      const t = (((f % TOTAL_FRAMES) + TOTAL_FRAMES) % TOTAL_FRAMES) / FPS;
      const T = at(f / FPS);
      for (const l of this.lanes) l.to(t, T);
    }
  }
}

/**
 * The live soundtrack. Browsers only allow audio after a user gesture, so the context is created by
 * enable(); from then on it follows the film's transport (playing, and the performance.now() at which
 * film time 0 plays), stays silent while muted or hidden, and rests the audio device while paused.
 */
export class Soundtrack {
  private ac: AudioContext | null = null;
  private mix: Mixer | null = null;
  private playing = false;
  private origin = 0;
  private hidden = false;
  private muted: boolean;
  private live = false;
  private gen = 0;
  private done = 0; // unwrapped film time queued up to
  private offset = 0; // audio clock − performance clock, seconds (smoothed)
  /** Called when the sound turns audible or silent, e.g. to update a hint. */
  onchange: (audible: boolean) => void = () => {};

  constructor(muted = false) {
    this.muted = muted;
  }

  get audible(): boolean {
    return !!this.ac && !this.muted;
  }

  /** From a user gesture. Returns true if this gesture is what turned the sound on. */
  enable(): boolean {
    if (this.ac) {
      if (this.wanted() && this.ac.state !== 'running') this.apply();
      return false;
    }
    const Ctor = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return false;
    this.ac = new Ctor();
    this.mix = new Mixer(this.ac);
    window.setInterval(() => this.pump(), 40);
    this.apply();
    this.onchange(this.audible);
    return this.audible;
  }

  /** Sound on (the first time) or mute / unmute. */
  toggle(): void {
    if (!this.ac) {
      this.muted = false;
      this.enable();
      return;
    }
    this.muted = !this.muted;
    this.apply();
    this.onchange(this.audible);
  }

  setHidden(hidden: boolean): void {
    this.hidden = hidden;
    this.apply();
  }

  /** The film clock changed: playing or not, and when (performance.now(), ms) film time 0 plays. */
  transport(playing: boolean, origin: number): void {
    this.playing = playing;
    this.origin = origin;
    this.apply();
  }

  /** Keep LOOKAHEAD of film queued. Runs on a timer and on every animation frame. */
  pump(): void {
    const { ac, mix } = this;
    if (!this.live || !ac || !mix) return;
    this.measure(false);
    const now = this.filmAt(ac.currentTime);
    if (this.done < now) this.done = now; // fell behind (a stalled main thread): never replay the past
    const until = now + LOOKAHEAD;
    if (until <= this.done) return;
    mix.schedule(this.done, until, (u) => this.ctxAt(u));
    this.done = until;
  }

  private wanted(): boolean {
    return !!this.ac && this.playing && !this.muted && !this.hidden;
  }

  private apply(): void {
    const { ac, mix } = this;
    if (!ac || !mix) return;
    const gen = ++this.gen;
    if (this.live) {
      // a quick dip hides the seam; whatever was queued for the old playhead goes with it
      mix.fade(0, ac.currentTime, 0.03);
      mix.dropShots(ac.currentTime + 0.03);
      this.live = false;
    }
    if (!this.wanted()) {
      window.setTimeout(() => {
        if (gen === this.gen && ac.state === 'running') ac.suspend();
      }, 150);
      return;
    }
    const start = (): void => {
      if (gen !== this.gen) return;
      this.measure(true);
      const t0 = ac.currentTime + 0.05;
      this.done = this.filmAt(t0);
      mix.resetLanes(wrap(this.done), t0);
      mix.fade(1, t0, 0.08);
      this.live = true;
      this.pump();
    };
    if (ac.state === 'running') start();
    else ac.resume().then(start, () => {});
  }

  // Film clock ↔ audio clock. A frame's time u (s, unwrapped) is shown at performance time
  // origin + 1000u (+ DISPLAY_LAG); `offset` maps performance time to the audio sample heard then.
  private ctxAt(u: number): number {
    return this.origin / 1000 + u + this.offset + DISPLAY_LAG;
  }
  private filmAt(T: number): number {
    return T - this.offset - DISPLAY_LAG - this.origin / 1000;
  }

  /** The output timestamp says which audio time is leaving the speakers at which performance time. */
  private measure(snap: boolean): void {
    const ac = this.ac!;
    const now = performance.now();
    const ts = ac.getOutputTimestamp?.();
    let off = ac.currentTime - now / 1000 - (ac.outputLatency || ac.baseLatency || 0);
    // trust a stamp only while it is fresh: just after a resume it still describes the moment we paused
    if (ts?.contextTime && ts.performanceTime && now - ts.performanceTime < 150) off = ts.contextTime - ts.performanceTime / 1000;
    // follow slow clock drift gently; jump on a real change (resume, a new output device)
    this.offset = snap || Math.abs(off - this.offset) > 0.03 ? off : this.offset + (off - this.offset) * 0.05;
  }
}

/** The whole soundtrack rendered offline — the same score, sample-exact on film time (for video export). */
export async function renderSoundtrack(sampleRate = 48000): Promise<AudioBuffer> {
  const ac = new OfflineAudioContext(2, Math.round(DURATION * sampleRate), sampleRate);
  const mix = new Mixer(ac);
  mix.out.gain.value = 1;
  mix.resetLanes(0, 0);
  mix.schedule(-1e-6, DURATION, (u) => u);
  return ac.startRendering();
}

/** 16-bit PCM WAV bytes of a rendered buffer. */
export function wav(b: AudioBuffer): Uint8Array {
  const ch = b.numberOfChannels;
  const n = b.length;
  const v = new DataView(new ArrayBuffer(44 + n * ch * 2));
  const str = (o: number, s: string): void => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  str(0, 'RIFF');
  v.setUint32(4, 36 + n * ch * 2, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, ch, true);
  v.setUint32(24, b.sampleRate, true);
  v.setUint32(28, b.sampleRate * ch * 2, true);
  v.setUint16(32, ch * 2, true);
  v.setUint16(34, 16, true);
  str(36, 'data');
  v.setUint32(40, n * ch * 2, true);
  const data = Array.from({ length: ch }, (_, c) => b.getChannelData(c));
  for (let i = 0, o = 44; i < n; i++) {
    for (let c = 0; c < ch; c++, o += 2) v.setInt16(o, Math.round(Math.max(-1, Math.min(1, data[c][i])) * 32767), true);
  }
  return new Uint8Array(v.buffer);
}
