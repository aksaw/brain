// Forms with structure of their own, steered by the slow brain measures (AlphaFeatures)
// rather than drawn from the raw trace. Pure geometry; sculpture.js renders it.
import { FEATURE_RATE, SPECTRUM_BINS } from './alpha-features.js';

// ---- Harmonograph --------------------------------------------------------------------
// A 3D harmonograph: each axis is the sum of two pendulums, as in the Victorian drawing
// machines. The band powers set their amplitudes (delta, theta, alpha, beta swinging
// slowly through the figure) and frontal asymmetry shifts one pendulum's phase. The
// frequencies are slightly out of tune with each other, so the figure keeps precessing
// instead of closing on itself. Returned as a curve, like the others, sampled
// HARMONOGRAPH_RATE times a second.
const HARMONOGRAPH_RATE = 64;
// Slow, like real harmonograph pendulums: one to four seconds a swing.
const PENDULUMS = [
  [{ band: 'theta', hz: 0.25 }, { band: 'alpha', hz: 0.7535, shift: true }],
  [{ band: 'alpha', hz: 0.5 }, { band: 'beta', hz: 1.0065 }],
  [{ band: 'delta', hz: 0.3765 }, { band: 'theta', hz: 0.6245 }],
];

export function harmonographCurve(features) {
  if (features.first === null) return null;
  const k = FEATURE_RATE / HARMONOGRAPH_RATE; // measurements per curve sample
  const at = (t) => {
    const seconds = t / HARMONOGRAPH_RATE;
    const bands = features.bandsAt(t * k);
    const shift = (Math.PI / 2) * features.headAt(t * k).asymmetry;
    return PENDULUMS.map((pair) =>
      pair.reduce((sum, { band, hz, shift: shifted }) => {
        const amplitude = 0.27 + 0.2 * bands[band]; // 0.07 to 0.47: a quiet band barely swings
        return sum + amplitude * Math.sin(2 * Math.PI * hz * seconds + (shifted ? shift : 0));
      }, 0),
    );
  };
  return {
    at,
    first: Math.ceil(features.first / k),
    end: Math.floor((features.end - 1) / k), // keeps t·k inside the stored measurements
    lookback: 0,
    tau: HARMONOGRAPH_RATE / 2, // recurrence threads look for returns a second apart
    rate: HARMONOGRAPH_RATE,
  };
}

// ---- Spectral vessel ------------------------------------------------------------------
// A pot thrown from the spectrum: height is time (the newest measurement at the rim),
// angle is frequency, and the wall bulges where that frequency has more power than usual
// (whitened against each bin's baseline, so it's changes that show, not the 1/f slope).
// The left hemisphere wraps one half of the pot, 1 Hz to 30 Hz, and the right hemisphere
// the other half back again, so the halves meet at matching frequencies and a lopsided
// pot means lopsided activity. Underneath it is a plain cylinder, so the outline is all
// data.
export const VESSEL_COLUMNS = 2 * SPECTRUM_BINS;
const VESSEL_RADIUS = 0.45;
export const VESSEL_MAX_ROWS = 240;

export function vesselGrid(features, seconds, relief, out) {
  if (features.first === null) return 0;
  const span = Math.round(seconds * FEATURE_RATE);
  const start = Math.max(features.first, features.end - span);
  const count = features.end - start;
  if (count < 2) return 0;
  const step = Math.max(1, Math.ceil(count / VESSEL_MAX_ROWS));
  // Rows on a grid tied to the data, so they don't shimmer as time passes, plus the newest.
  const rows = [];
  for (let i = Math.ceil(start / step) * step; i < features.end - 1; i += step) rows.push(i);
  rows.push(features.end - 1);
  // Each row averages the spectra around its time (at least half a second's worth) and
  // blends each frequency with its neighbours (about ±1 Hz), so the wall carries smooth
  // bands and ridges rather than bristling with single-bin noise.
  const window = Math.max(step, FEATURE_RATE / 2);
  const KERNEL = [1, 4, 6, 4, 1].map((w) => w / 16);
  const sums = [new Float64Array(SPECTRUM_BINS), new Float64Array(SPECTRUM_BINS)];
  const value = new Float32Array(VESSEL_COLUMNS);
  rows.forEach((i, r) => {
    const h = rows.length > 1 ? r / (rows.length - 1) : 1; // 0 at the foot, 1 at the rim
    const from = Math.max(features.first, Math.round(i - window / 2));
    const to = Math.min(features.end - 1, Math.round(i + window / 2));
    for (let side = 0; side < 2; side++) {
      sums[side].fill(0);
      for (let m = from; m <= to; m++) {
        const spectrum = features.spectrumAt(m, side);
        for (let b = 0; b < SPECTRUM_BINS; b++) sums[side][b] += spectrum[b] / (to - from + 1);
      }
    }
    for (let c = 0; c < VESSEL_COLUMNS; c++) {
      const bin = c < SPECTRUM_BINS ? c : VESSEL_COLUMNS - 1 - c;
      const spectrum = sums[c < SPECTRUM_BINS ? 0 : 1];
      let v = 0;
      for (let o = -2; o <= 2; o++) v += KERNEL[o + 2] * spectrum[Math.max(0, Math.min(SPECTRUM_BINS - 1, bin + o))];
      value[c] = v;
    }
    const profile = VESSEL_RADIUS; // a plain cylinder: only the spectrum shapes the wall
    const y = -0.95 + 1.9 * h;
    for (let c = 0; c < VESSEL_COLUMNS; c++) {
      const angle = (2 * Math.PI * c) / VESSEL_COLUMNS;
      const radius = Math.max(0.05, profile + relief * value[c]);
      out.set([radius * Math.cos(angle), y, radius * Math.sin(angle)], (r * VESSEL_COLUMNS + c) * 3);
    }
  });
  return rows.length;
}

// ---- Grown shell ----------------------------------------------------------------------
// A seashell grown by accretion (Raup's model): an opening that widens exponentially as
// it turns around a coiling axis (and, for most shells, slides along it, so older growth
// spirals into a spire). Two rings are added a second, each shaped by the brain at that
// moment:
// - posterior alpha sets how fast that ring widens (calm, alpha-rich stretches flare);
// - frontal asymmetry squashes the opening one way or the other;
// - alpha peak frequency sets the depth of the ornament (ribs, growth lines or knobs).
// Ring positions are fixed once grown; the newest ring is the lip, kept at a constant
// size by scaling the whole shell down as it grows, so older whorls recede into the apex.
//
// SHELL_SHAPES are classic shell types, each a set of Raup-style proportions:
// - flare: how much of the Flare setting it uses (augers widen slowly, conchs fast);
// - opening: the opening's size as a share of its distance from the axis; or null for a
//   flat spiral, where it's sized so each whorl meets the one inside it, side by side
//   (r - ρ = (r + ρ) / whorl);
// - stacked: the coil steps down its axis just enough for each whorl to sit on the one
//   below (a spire); otherwise it stays flat;
// - overlap: how far each whorl wraps over the one it meets (1: they just touch);
// - aperture: the opening's [outward, along-axis] proportions;
// - tilt: how the coiling axis is turned to face the viewer (radians about x).
export const SHELL_COLUMNS = 48;
export const SHELL_MAX_ROWS = 601;
const RING_RATE = 2; // rings per second
const RINGS_PER_TURN = 40; // a whorl every 20 seconds
const SIZE = 0.85; // the shell is scaled to this many plot radii from its centre

export const SHELL_SHAPES = {
  // A flat spiral, each whorl wrapping over the last, a tall oval opening, growth lines.
  nautilus: { flare: 1, opening: null, stacked: false, overlap: 1.35, aperture: [0.8, 1.3], tilt: Math.PI / 2, ornament: 'growth' },
  // A tall, slender spire of many slowly widening whorls with fine spiral ridges.
  auger: { flare: 0.2, opening: 0.7, stacked: true, overlap: 1.08, aperture: [1, 0.9], tilt: 0, ornament: 'spiral' },
  // A big body whorl with a long opening, a pointed spire, and knobs on the shoulder.
  conch: { flare: 1, opening: 0.8, stacked: true, overlap: 1.1, aperture: [0.85, 1.5], tilt: -0.35, ornament: 'knobs' },
};

// The ornament's relief at a point of the opening (angle phi) on a given ring.
function ornament(kind, phi, ring) {
  if (kind === 'spiral') return Math.cos(14 * phi); // ridges running along the whorls
  if (kind === 'growth') return Math.cos((2 * Math.PI * ring) / 5); // a growth line every few seconds
  // Knobs: on the shoulder (the outer, upper side of the opening), every few seconds.
  const shoulder = Math.max(0, Math.cos(phi - Math.PI / 4)) ** 6;
  return 6 * shoulder * Math.max(0, Math.cos((2 * Math.PI * ring) / 6)) ** 3;
}

export function shellGrid(features, seconds, expansion, shapeName, out) {
  if (features.first === null) return 0;
  const shape = SHELL_SHAPES[shapeName] ?? SHELL_SHAPES.nautilus;
  const k = FEATURE_RATE / RING_RATE; // measurements per ring
  const newest = (features.end - 1) / k; // fractional: the lip grows smoothly
  const oldest = Math.max(features.first / k, newest - Math.min(seconds * RING_RATE, SHELL_MAX_ROWS - 1));
  if (newest - oldest < 1) return 0;
  const rings = [];
  for (let j = Math.ceil(oldest); j < newest; j++) rings.push(j);
  rings.push(newest);
  const n = rings.length;
  const params = rings.map((j) => features.headAt(j * k));

  // Size of each ring relative to the lip: widening between consecutive rings, summed
  // back from the newest. A whorl widens by `whorl` on average.
  const whorl = 1 + (expansion - 1) * shape.flare;
  const perRing = Math.log(whorl) / RINGS_PER_TURN;
  const [outwardShape, upShape] = shape.aperture;
  // A flat spiral's opening meets the next whorl out; a spire's whorls sit on each other,
  // stepping down a turn's height: twice the opening's height, less the widening.
  const opening = shape.opening ?? (shape.overlap * (whorl - 1)) / (whorl + 1) / outwardShape;
  const descent = shape.stacked ? (2 * opening * upShape) / (1 - 1 / whorl) / shape.overlap : 0;
  const logSize = new Float64Array(n);
  for (let r = n - 2; r >= 0; r--) {
    const growth = perRing * (1 + 0.6 * params[r + 1].posterior) * (rings[r + 1] - rings[r]);
    logSize[r] = logSize[r + 1] - growth;
  }

  const [tc, ts] = [Math.cos(shape.tilt), Math.sin(shape.tilt)];
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let r = 0; r < n; r++) {
    const size = Math.exp(logSize[r]);
    const theta = (2 * Math.PI * rings[r]) / RINGS_PER_TURN;
    const [cos, sin] = [Math.cos(theta), Math.sin(theta)];
    const { asymmetry, peak } = params[r];
    const squash = 1 + 0.35 * asymmetry;
    const depth = 0.06 * (0.5 + 0.5 * peak);
    const cy = -descent * size; // older (smaller) rings sit higher, up the spire
    for (let c = 0; c < SHELL_COLUMNS; c++) {
      const phi = (2 * Math.PI * c) / SHELL_COLUMNS;
      const radius = size * opening * (1 + depth * ornament(shape.ornament, phi, rings[r]));
      const outward = radius * Math.cos(phi) * outwardShape * squash; // away from the axis
      const up = (radius * Math.sin(phi) * upShape) / squash;
      const x = size * cos + outward * cos;
      const z = size * sin + outward * sin;
      const y = cy + up;
      // Turn the coiling axis to face the viewer.
      const p = [x, y * tc - z * ts, y * ts + z * tc];
      out.set(p, (r * SHELL_COLUMNS + c) * 3);
      for (let a = 0; a < 3; a++) {
        if (p[a] < min[a]) min[a] = p[a];
        if (p[a] > max[a]) max[a] = p[a];
      }
    }
  }

  // Centre the shell and scale it to a constant size in view.
  const centre = [0, 1, 2].map((a) => (min[a] + max[a]) / 2);
  const extent = Math.max(...[0, 1, 2].map((a) => max[a] - min[a])) / 2 || 1;
  const fit = SIZE / extent;
  for (let i = 0; i < n * SHELL_COLUMNS * 3; i++) out[i] = (out[i] - centre[i % 3]) * fit;
  return n;
}
