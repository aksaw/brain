// Slow alpha-band measures, as 3D curves.
//
// Whole head, one curve:
// - x, posterior alpha: relative alpha (8-13 Hz over 4-30 Hz power) averaged over TP9 and
//   TP10, the electrodes nearest the back of the head, where alpha is strongest. Relative
//   power is much less sensitive to electrode contact than absolute power, and leaving out
//   1-4 Hz keeps blinks (large, slow, and strong at TP through the Fpz reference) from
//   swamping it.
// - y, frontal asymmetry: ln(alpha at AF8) - ln(alpha at AF7).
// - z, peak: the alpha peak frequency (centre of gravity of 7-13 Hz power at TP9/TP10).
//   Only meaningful while alpha is present, so it barely moves when alpha is weak.
//
// Per hemisphere, one curve each from that side's own two electrodes (left: AF7, TP9;
// right: AF8, TP10), on the same three axes:
// - x, back alpha: relative alpha at the ear electrode;
// - y, front alpha: relative alpha at the forehead electrode;
// - z, peak: the alpha peak frequency at the ear electrode, followed while alpha is clear.
// Both hemispheres share one baseline per axis, so a steady difference between them is
// kept (the shell between them opens) instead of being normalized away.
//
// Powers are measured FEATURE_RATE times a second over the last 2 s, smoothed over about a
// second, then expressed against a running baseline (the last minute or so, as a z-score
// / 2.5, clamped to -1..1), so the forms fill their space for any wearer and session.
// Values are stored already normalized, so older parts of a trail keep their shape as the
// baseline drifts.
import { EEG_CHANNELS, EEG_SAMPLE_RATE } from './muse.js';
import { fft } from './bands.js';

export const FEATURE_RATE = 16; // measurements per second
const HOP = EEG_SAMPLE_RATE / FEATURE_RATE; // raw samples between measurements
const WINDOW = 2 * EEG_SAMPLE_RATE; // raw samples per spectrum
const HISTORY_S = 600; // seconds of measurements kept
const SMOOTH_S = 1;
const BASELINE_S = 60;
const WARMUP = 3 * FEATURE_RATE; // measurements before the baseline is trusted
const SPREAD = 2.5; // z-scores mapped to the edge of the space

const channel = (name) => EEG_CHANNELS.indexOf(name);
const TP = [channel('TP9'), channel('TP10')];
const AF7 = channel('AF7');
const AF8 = channel('AF8');
const HEMISPHERES = [
  { name: 'left', back: channel('TP9'), front: AF7 },
  { name: 'right', back: channel('TP10'), front: AF8 },
];

const DF = EEG_SAMPLE_RATE / WINDOW; // Hz per spectrum bin
const bins = (lo, hi) => [Math.ceil(lo / DF), Math.ceil(hi / DF)]; // [first, end)
const ALPHA = bins(8, 13);
const WIDE = bins(4, 30);
const PEAK_RANGE = bins(7, 13);
const HANN = Float64Array.from({ length: WINDOW }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / WINDOW));

const HEAD_AXES = ['posterior', 'asymmetry', 'peak'];
const SIDE_AXES = ['back', 'front', 'peak'];

const power = (spectrum, [lo, hi]) => {
  let sum = 0;
  for (let k = lo; k < hi; k++) sum += spectrum[k];
  return sum + 1e-12;
};
const relativeAlpha = (spectrum) => power(spectrum, ALPHA) / power(spectrum, WIDE);

// Centre of gravity of 7-13 Hz power, summed over the given spectra.
function peakFrequency(spectra) {
  let weighted = 0;
  let total = 0;
  for (let k = PEAK_RANGE[0]; k < PEAK_RANGE[1]; k++) {
    const p = spectra.reduce((s, spectrum) => s + spectrum[k], 0);
    weighted += p * k * DF;
    total += p;
  }
  return total > 0 ? weighted / total : 10;
}

// How clearly alpha is present (0..1) from relative alpha, for gating the peak frequency.
const clarity = (relative) => Math.max(0, Math.min(1, (relative - 0.2) / 0.2));

const SMOOTH = 1 - Math.exp(-1 / (FEATURE_RATE * SMOOTH_S));
const BASELINE = 1 - Math.exp(-1 / (FEATURE_RATE * BASELINE_S));

// Exponential smoothing of a set of named values; `gate` scales how fast each follows.
function smoothInto(state, raw, gate = {}) {
  if (!state) return { ...raw };
  for (const axis in raw) state[axis] += SMOOTH * (gate[axis] ?? 1) * (raw[axis] - state[axis]);
  return state;
}

// A running mean and variance, quick at first, then over about a minute.
class Baseline {
  constructor() {
    this.mean = 0;
    this.variance = 0;
    this.count = 0;
  }

  add(value) {
    const w = Math.max(1 / ++this.count, BASELINE);
    const d = value - this.mean;
    this.mean += w * d;
    this.variance = (1 - w) * (this.variance + w * d * d);
  }

  normalize(value) {
    const z = (value - this.mean) / Math.sqrt(this.variance + 1e-12);
    return Math.max(-1, Math.min(1, z / SPREAD));
  }
}

export class AlphaFeatures {
  constructor(stream) {
    this.stream = stream;
    this.ring = HISTORY_S * FEATURE_RATE;
    this.reset();
  }

  reset() {
    const rings = (axes) => Object.fromEntries(axes.map((axis) => [axis, new Float32Array(this.ring)]));
    this.head = { values: rings(HEAD_AXES), smoothed: null };
    this.sides = HEMISPHERES.map(() => ({ values: rings(SIDE_AXES), smoothed: null }));
    this.headBaseline = Object.fromEntries(HEAD_AXES.map((axis) => [axis, new Baseline()]));
    this.sideBaseline = Object.fromEntries(SIDE_AXES.map((axis) => [axis, new Baseline()])); // shared
    this.count = 0;
    this.next = null; // the next measurement index to compute
    this.first = null; // first stored (normalized) measurement index
    this.end = 0; // one past the newest
  }

  // Computes every measurement whose window of raw samples is now complete. Measurement f
  // covers raw samples [f·HOP - WINDOW, f·HOP). Catches up on at most what the raw ring
  // still holds (if frames were paused for longer, the gap is skipped).
  update() {
    const channels = this.stream.channels;
    if (channels.some((c) => c.first === null)) return;
    const first = Math.max(...channels.map((c) => c.first));
    const end = Math.min(...channels.map((c) => c.end));
    const oldest = Math.max(first, end - this.stream.ring + HOP);
    let f = Math.max(this.next ?? 0, Math.ceil((oldest + WINDOW) / HOP));
    for (; f * HOP <= end; f++) this.measure(f);
    this.next = f;
  }

  measure(f) {
    const spectra = EEG_CHANNELS.map((_, ch) => this.spectrum(ch, f * HOP));

    // Whole head.
    const posterior = TP.reduce((s, ch) => s + relativeAlpha(spectra[ch]), 0) / TP.length;
    const head = {
      posterior,
      asymmetry: Math.log(power(spectra[AF8], ALPHA)) - Math.log(power(spectra[AF7], ALPHA)),
      peak: peakFrequency(TP.map((ch) => spectra[ch])),
    };
    this.head.smoothed = smoothInto(this.head.smoothed, head, { peak: clarity(posterior) });

    // Each hemisphere.
    HEMISPHERES.forEach(({ back, front }, s) => {
      const backAlpha = relativeAlpha(spectra[back]);
      const raw = { back: backAlpha, front: relativeAlpha(spectra[front]), peak: peakFrequency([spectra[back]]) };
      this.sides[s].smoothed = smoothInto(this.sides[s].smoothed, raw, { peak: clarity(backAlpha) });
    });

    for (const axis of HEAD_AXES) this.headBaseline[axis].add(this.head.smoothed[axis]);
    for (const axis of SIDE_AXES) for (const side of this.sides) this.sideBaseline[axis].add(side.smoothed[axis]);
    if (++this.count < WARMUP) return;

    const at = f % this.ring;
    for (const axis of HEAD_AXES) this.head.values[axis][at] = this.headBaseline[axis].normalize(this.head.smoothed[axis]);
    for (const side of this.sides) {
      for (const axis of SIDE_AXES) side.values[axis][at] = this.sideBaseline[axis].normalize(side.smoothed[axis]);
    }
    this.first ??= f;
    this.first = Math.max(this.first, f + 1 - this.ring);
    this.end = f + 1;
  }

  // Power per frequency bin (arbitrary units) of channel ch's full-band signal over the
  // WINDOW raw samples ending at `end`, Hann-windowed.
  spectrum(ch, end) {
    const wide = this.stream.channels[ch].wide;
    const ring = this.stream.ring;
    const re = new Float64Array(WINDOW);
    const im = new Float64Array(WINDOW);
    let mean = 0;
    for (let i = 0; i < WINDOW; i++) mean += re[i] = wide[(end - WINDOW + i) % ring];
    mean /= WINDOW;
    for (let i = 0; i < WINDOW; i++) re[i] = (re[i] - mean) * HANN[i];
    fft(re, im);
    const out = new Float64Array(WINDOW / 2);
    for (let k = 0; k < out.length; k++) out[k] = re[k] * re[k] + im[k] * im[k];
    return out;
  }

  // Three stored axes as a curve over measurement indices (fractional ones interpolate).
  // tau is one second, so recurrence threads look for returns at least two seconds apart.
  #curve(values, axes) {
    const rings = axes.map((axis) => values[axis]);
    const ring = this.ring;
    const at = (t) => {
      const i = Math.floor(t);
      const f = t - i;
      return rings.map((a) => a[i % ring] * (1 - f) + a[(i + 1) % ring] * f);
    };
    return { at, first: this.first, end: this.end, lookback: 0, tau: FEATURE_RATE, rate: FEATURE_RATE };
  }

  // The whole-head curve, or null before the baseline has warmed up.
  curve() {
    return this.first === null ? null : this.#curve(this.head.values, HEAD_AXES);
  }

  // [left, right] hemisphere curves, or nulls before the baseline has warmed up.
  hemisphereCurves() {
    if (this.first === null) return [null, null];
    return this.sides.map((side) => this.#curve(side.values, SIDE_AXES));
  }
}
