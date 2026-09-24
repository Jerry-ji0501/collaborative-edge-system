// Boot: DPR-aware 16:9 canvas, deterministic 30 fps clock, minimal keyboard controls, capture hooks.
import { renderFilm } from './film';
import { buildTextures } from './core/texture';
import { DURATION, FPS, SECTIONS, TOTAL_FRAMES } from './core/timeline';
import { W } from './core/math';
import './style.css';

const canvas = document.getElementById('film') as HTMLCanvasElement;
const ctx = canvas.getContext('2d', { alpha: false })!;
const bar = document.getElementById('progress') as HTMLDivElement | null;
const params = new URLSearchParams(location.search);
const capture = params.has('capture');

const clock = document.getElementById('clock');
const chapterHost = document.getElementById('chapters');
const chapterName = document.getElementById('chapter-name');

let scale = 1;
function resize(): void {
  let vw = window.innerWidth;
  let vh = window.innerHeight;
  // embedded layout: fit the column the stage sits in, leaving room for the caption rows
  if (canvas.dataset.fit === 'column') {
    const host = canvas.parentElement?.parentElement;
    vw = host ? host.clientWidth : vw;
    vh = Math.max(200, vh - Number(canvas.dataset.reserve ?? 120));
  }
  const cssW = capture ? W / (window.devicePixelRatio || 1) : Math.min(vw, (vh * 16) / 9);
  const cssH = (cssW * 9) / 16;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  let bw = Math.round(cssW * dpr);
  if (bw > 3840) bw = 3840;
  const bh = Math.round((bw * 9) / 16);
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  scale = bw / W;
  lastFrame = -1;
}

const tex = buildTextures(ctx);

const fmt = (t: number): string => {
  const s = Math.floor(t);
  return `00:${String(s).padStart(2, '0')}.${Math.floor((t - s) * 10)}`;
};
const chapterBtns: HTMLButtonElement[] = [];
let activeChapter = -1;

function draw(t: number): void {
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  renderFilm(ctx, t, tex);
  if (bar) bar.style.transform = `scaleX(${t / DURATION})`;
  if (clock) clock.textContent = `${fmt(t)} / ${fmt(DURATION)}`;
  if (chapterBtns.length) {
    let idx = 0;
    for (let i = 0; i < SECTIONS.length; i++) if (t >= SECTIONS[i][0]) idx = i;
    if (idx !== activeChapter) {
      chapterBtns.forEach((b, i) => b.setAttribute('aria-current', i === idx ? 'true' : 'false'));
      if (chapterName) chapterName.textContent = SECTIONS[idx][1];
      activeChapter = idx;
    }
    chapterHost?.style.setProperty('--played', String(t / DURATION));
  }
}

// ---- clock: wall time quantised to whole frames so every playback shows identical frames
let playing = !capture;
let origin = performance.now();
let frame = Math.max(0, Math.min(TOTAL_FRAMES - 1, Math.round(parseFloat(params.get('t') ?? '0') * FPS) || 0));
origin -= (frame / FPS) * 1000;
let lastFrame = -1;

function tick(now: number): void {
  if (playing) frame = Math.floor(((now - origin) / 1000) * FPS) % TOTAL_FRAMES;
  if (frame !== lastFrame) {
    draw(frame / FPS);
    lastFrame = frame;
  }
  requestAnimationFrame(tick);
}

function setPlaying(p: boolean): void {
  playing = p;
  if (p) origin = performance.now() - (frame / FPS) * 1000;
  document.body.classList.toggle('paused', !p);
}

window.addEventListener('resize', resize);
window.addEventListener('keydown', (e) => {
  if (e.code === 'Space') {
    setPlaying(!playing);
    e.preventDefault();
  } else if (e.code === 'ArrowRight' || e.code === 'ArrowLeft') {
    const step = e.shiftKey ? FPS : 1;
    frame = (frame + (e.code === 'ArrowRight' ? step : -step) + TOTAL_FRAMES) % TOTAL_FRAMES;
    setPlaying(false);
  } else if (e.key === 'r' || e.key === 'R') {
    frame = 0;
    setPlaying(true);
  } else if (e.key === 'f' || e.key === 'F') {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen?.();
  }
});
canvas.addEventListener('click', () => setPlaying(!playing));

// optional chapter strip (used by the embedded player page)
if (chapterHost) {
  SECTIONS.forEach(([start, name], i) => {
    const end = i + 1 < SECTIONS.length ? SECTIONS[i + 1][0] : DURATION;
    const b = document.createElement('button');
    b.type = 'button';
    b.id = `chapter-${i}`;
    b.style.flexGrow = String(end - start);
    b.innerHTML = `<span>${name}</span>`;
    b.title = `${name} · ${start.toFixed(1)}s`;
    b.addEventListener('click', () => {
      frame = Math.round(start * FPS);
      setPlaying(true);
    });
    chapterHost.appendChild(b);
    chapterBtns.push(b);
  });
}

// idle UI: the hairline only shows while the pointer moves
let hideTimer = 0;
window.addEventListener('pointermove', () => {
  document.body.classList.add('ui');
  clearTimeout(hideTimer);
  hideTimer = window.setTimeout(() => document.body.classList.remove('ui'), 1600);
});

// hooks for headless frame capture
declare global {
  interface Window {
    __film: { renderAt: (t: number) => number; duration: number; fps: number };
  }
}
window.__film = {
  renderAt: (t: number) => {
    const t0 = performance.now();
    draw(t);
    return performance.now() - t0;
  },
  duration: DURATION,
  fps: FPS,
};

resize();
setPlaying(playing);
requestAnimationFrame(tick);
