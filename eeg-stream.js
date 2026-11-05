// Filtered EEG stream and delay embeddings of each side's (x, y) trace, shared by pages
// that draw the traces in 3D.
import { EEG_CHANNELS, EEG_SAMPLE_RATE, EEG_SAMPLES_PER_PACKET } from './muse.js';
import { highpass, lowpass, notch } from './filters.js';

// Each side of the head: its forehead channel is x, its ear channel y.
export const SIDES = [
  { name: 'left', x: 'AF7', y: 'TP9' },
  { name: 'right', x: 'AF8', y: 'TP10' },
].map((s) => ({ ...s, xi: EEG_CHANNELS.indexOf(s.x), yi: EEG_CHANNELS.indexOf(s.y) }));

// Samples are stored by absolute sample index (packet sequence number x 12 + position), so
// x and y come from the same instant even though each channel arrives in its own packets.
// Every channel gets a high-pass and mains notch, then a low-pass to smooth the trace.
export class EegStream {
  constructor({ seconds = 16, highpassHz = 1, mainsHz = 60, lowpassHz = 15 } = {}) {
    this.ring = seconds * EEG_SAMPLE_RATE;
    this.options = { highpassHz, mainsHz, lowpassHz };
    this.reset();
  }

  reset() {
    const rate = EEG_SAMPLE_RATE;
    const { highpassHz, mainsHz, lowpassHz } = this.options;
    this.channels = EEG_CHANNELS.map(() => ({
      filters: [
        highpass(highpassHz, rate),
        notch(mainsHz, rate),
        lowpass(lowpassHz, rate),
        lowpass(lowpassHz, rate),
      ],
      values: new Float32Array(this.ring),
      first: null, // index of the first sample received
      end: 0, // one past the newest sample index
      lastSeq: null,
      wraps: 0,
    }));
  }

  add(ch, seq, samples) {
    const c = this.channels[ch];
    if (c.lastSeq !== null && seq < c.lastSeq - 32768) c.wraps++; // 16-bit counter wrapped
    c.lastSeq = seq;
    const firstIndex = (c.wraps * 65536 + seq) * EEG_SAMPLES_PER_PACKET;
    samples.forEach((s, k) => {
      for (const filter of c.filters) s = filter.process(s);
      c.values[(firstIndex + k) % this.ring] = s;
    });
    c.first ??= firstIndex;
    c.end = Math.max(c.end, firstIndex + samples.length);
  }

  // The sample range both of a side's channels have: { first, end }, or null before data.
  range(side) {
    const cx = this.channels[side.xi];
    const cy = this.channels[side.yi];
    if (cx.first === null || cy.first === null) return null;
    return { first: Math.max(cx.first, cy.first), end: Math.min(cx.end, cy.end) };
  }

  // Channel ch at a fractional sample index, by linear interpolation. Indices before the
  // first sample (rounding can land a hair below it) read as 0 rather than undefined.
  read(ch, t) {
    const values = this.channels[ch].values;
    const i = Math.floor(t);
    const f = t - i;
    const at = (k) => (k < 0 ? 0 : values[k % this.ring]);
    return at(i) * (1 - f) + at(i + 1) * f;
  }
}

// Ways to lift a side's (x, y) trace into 3D. signal(x, y) is the single signal the
// automatic delay is measured on; span is how many delays back a point reaches.
export const EMBEDDINGS = {
  xyx: { signal: (x, y) => x, span: 1, point: (x, y, tau) => (t) => [x(t), y(t), x(t - tau)] },
  xyy: { signal: (x, y) => y, span: 1, point: (x, y, tau) => (t) => [x(t), y(t), y(t - tau)] },
  diff: {
    signal: (x, y) => (x - y) / Math.SQRT2,
    span: 2,
    point: (x, y, tau) => {
      const s = (t) => (x(t) - y(t)) / Math.SQRT2;
      return (t) => [s(t), s(t - tau), s(t - 2 * tau)];
    },
  },
  sum: {
    signal: (x, y) => (x + y) / 2,
    span: 2,
    point: (x, y, tau) => {
      const s = (t) => (x(t) + y(t)) / 2;
      return (t) => [s(t), s(t - tau), s(t - 2 * tau)];
    },
  },
};

// A quarter period of `hz`, in samples: the delay that turns a rhythm at hz into a circle.
export function bandDelay(hz) {
  return EEG_SAMPLE_RATE / Number(hz) / 4;
}

// Automatic delay: the first lag at which the signal's autocorrelation reaches zero (a
// quarter period for a pure rhythm), interpolated between samples. Re-measured twice a
// second over the last 2 s and glided towards, so the shape morphs instead of jumping.
const AUTO_WINDOW = 2 * EEG_SAMPLE_RATE;
const AUTO_EVERY_MS = 500;
const AUTO_GLIDE_S = 2;
const AUTO_MIN = 2;
const AUTO_MAX = 64; // 250 ms

export class AutoDelay {
  constructor() {
    this.tau = bandDelay(10);
    this.target = this.tau;
    this.measuredAt = 0;
    this.frameAt = null;
  }

  // signal(i) is the signal at whole sample i; data runs from `first` to `end`.
  update(signal, first, end, now) {
    if (now - this.measuredAt >= AUTO_EVERY_MS && end - first >= AUTO_WINDOW) {
      this.measuredAt = now;
      const n = AUTO_WINDOW;
      const s = new Float64Array(n);
      for (let k = 0; k < n; k++) s[k] = signal(end - n + k);
      const mean = s.reduce((a, b) => a + b, 0) / n;
      for (let k = 0; k < n; k++) s[k] -= mean;
      const r = (lag) => {
        let sum = 0;
        for (let k = lag; k < n; k++) sum += s[k] * s[k - lag];
        return sum;
      };
      let lag = 1;
      let before = r(0);
      let after = r(1);
      while (lag < AUTO_MAX && after > 0) {
        before = after;
        after = r(++lag);
      }
      const crossing = after > 0 || before <= 0 ? lag : lag - after / (after - before);
      this.target = Math.max(AUTO_MIN, Math.min(AUTO_MAX, crossing));
    }
    const dt = this.frameAt === null ? 0 : (now - this.frameAt) / 1000;
    this.tau += (this.target - this.tau) * (1 - Math.exp(-dt / AUTO_GLIDE_S));
    this.frameAt = now;
    return this.tau;
  }
}
