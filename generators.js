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
