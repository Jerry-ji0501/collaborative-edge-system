// Procedural paper tooth + film grain. Generated once at start-up; no bitmaps are loaded.
import { makeCanvas } from './draw';
import { fbm2, noise2 } from './noise';
import { RNG } from './rng';
import { TAU } from './math';

export interface Textures {
  paper: HTMLCanvasElement;
  grain: CanvasPattern[];
}

export function buildTextures(ctx: CanvasRenderingContext2D): Textures {
  const pw = 800;
  const ph = 450;
  const paper = makeCanvas(pw, ph);
  const g = paper.getContext('2d')!;
  const img = g.createImageData(pw, ph);
  const d = img.data;
  const rng = new RNG(1234);
  for (let y = 0; y < ph; y++) {
    for (let x = 0; x < pw; x++) {
      const i = (y * pw + x) * 4;
      const blot = fbm2(x / 95, y / 95, 3, 11); // slow brightness drift
      const fib = noise2(x / 26, y / 4.2, 9); // stretched fibres
      const tooth = noise2(x / 2.2, y / 2.2, 5); // fine tooth
      const v = 128 + blot * 30 + fib * 6 + tooth * 6 + (rng.next() - 0.5) * 7;
      d[i] = d[i + 1] = d[i + 2] = v;
      d[i + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  g.lineCap = 'round';
  for (let k = 0; k < 520; k++) {
    const x = rng.range(0, pw);
    const y = rng.range(0, ph);
    const len = rng.range(4, 16);
    const a = rng.range(0, TAU);
    g.strokeStyle = rng.chance(0.55) ? 'rgba(255,255,255,0.16)' : 'rgba(0,0,0,0.12)';
    g.lineWidth = rng.range(0.35, 0.8);
    g.beginPath();
    g.moveTo(x, y);
    g.quadraticCurveTo(
      x + Math.cos(a + 0.6) * len * 0.5,
      y + Math.sin(a + 0.6) * len * 0.5,
      x + Math.cos(a) * len,
      y + Math.sin(a) * len,
    );
    g.stroke();
  }
  for (let k = 0; k < 220; k++) {
    g.fillStyle = `rgba(40,30,20,${rng.range(0.08, 0.2).toFixed(2)})`;
    g.beginPath();
    g.arc(rng.range(0, pw), rng.range(0, ph), rng.range(0.3, 0.9), 0, TAU);
    g.fill();
  }

  const grain: CanvasPattern[] = [];
  for (let n = 0; n < 4; n++) {
    const s = 256;
    const c = makeCanvas(s, s);
    const gc = c.getContext('2d')!;
    const im = gc.createImageData(s, s);
    for (let i = 0; i < s * s; i++) {
      const v = 128 + (rng.next() + rng.next() - 1) * 80;
      im.data[i * 4] = im.data[i * 4 + 1] = im.data[i * 4 + 2] = v;
      im.data[i * 4 + 3] = 255;
    }
    gc.putImageData(im, 0, 0);
    grain.push(ctx.createPattern(c, 'repeat')!);
  }
  return { paper, grain };
}
