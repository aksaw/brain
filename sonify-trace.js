// Sound modes driven by the trace as the plot draws it: the same smoothed signal, in the
// plot's own coordinates (the Scale setting's ±µV mapped to ±1), played with as little
// delay as the incoming packets allow. (The plucks are found on the page instead, from the
// very points drawn, and only played here.) Both are tuned to A minor pentatonic, so whatever
// they play together stays consonant.

import { EEG_RATE } from './sonify-core.js';

const SCALE = [0, 3, 5, 7, 10]; // A minor pentatonic, in semitones above A
const scaleNote = (degree) => {
  const octave = Math.floor(degree / SCALE.length);
  return 12 * octave + SCALE[degree - octave * SCALE.length];
};
const TAU = 2 * Math.PI;
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const smoothstep = (lo, hi, v) => {
  const t = clamp01((v - lo) / (hi - lo));
  return t * t * (3 - 2 * t);
};

// Plays back one side's (x, y) samples at the EEG rate on the audio clock. Samples arrive
// in bursts of 12; a small backlog (TARGET) absorbs that, and playback runs a little faster
// or slower to hold it, so the sound stays about 60 ms behind the newest data.
const TARGET = 16;
const RING = 1024;

export class TracePlayer {
  constructor() {
    this.xs = new Float32Array(RING);
    this.ys = new Float32Array(RING);
    this.written = 0;
    this.read = null;
    this.x = 0;
    this.y = 0;
    this.vx = 0; // velocity, in input units per second
    this.vy = 0;
  }

  push(x, y) {
    this.xs[this.written % RING] = x;
    this.ys[this.written % RING] = y;
    this.written++;
    this.read ??= this.written - 1;
  }

  // Advances one audio sample; returns false until there is data.
  next(rate) {
    if (this.read === null) return false;
    const newest = this.written - 1;
    let backlog = newest - this.read;
    if (backlog > RING / 2) {
      this.read = newest - TARGET; // fell far behind (the tab was hidden): catch up
      backlog = TARGET;
    }
    if (backlog > 0) {
      const speed = Math.min(2, Math.max(0.5, 1 + (backlog - TARGET) / (2 * TARGET)));
      this.read = Math.min(this.read + (speed * EEG_RATE) / rate, newest);
    }
    const i = Math.floor(this.read);
    const f = this.read - i;
    const a = i % RING;
    const b = (i + 1) % RING;
    const x = f > 0 ? this.xs[a] + f * (this.xs[b] - this.xs[a]) : this.xs[a];
    const y = f > 0 ? this.ys[a] + f * (this.ys[b] - this.ys[a]) : this.ys[a];
    this.vx = (x - this.x) * rate;
    this.vy = (y - this.y) * rate;
    this.x = x;
    this.y = y;
    return true;
  }
}

// Drone: follows the shape of the trail on screen rather than the moving point, which
// circles too fast (an alpha rhythm goes round ten times a second) to follow by ear.
// Over a window as long as the trail it keeps the trace's spread (an ellipse) and its net
// turning, and maps:
// - the ellipse's long axis to a note: turning the axis half a turn anticlockwise climbs
//   one octave of the scale, and on round again without end (a Shepard tone: each note is
//   a stack of octaves under a fixed bell-shaped envelope, so no octave is the top);
// - its size to loudness and brightness;
// - its roundness to fullness: a flat line is a single tone, a round loop adds the fifth;
// - net turning to shimmer (two slightly detuned copies beating faster), anticlockwise a
//   little brighter and clockwise a little darker.
const F0 = 27.5; // A0
const SLOTS = 7; // octaves in each Shepard tone
const BLOCK = 64; // samples between control updates

export class ShapeDrone {
  constructor() {
    this.mx = 0;
    this.my = 0;
    this.cxx = 0;
    this.cyy = 0;
    this.cxy = 0;
    this.rr = 0;
    this.rot = 0;
    this.axis = 0; // long-axis angle, unwrapped (period pi)
    this.degree = 0; // scale degree, unwrapped
    this.pitch = 0; // semitones above A, gliding
    this.amp = 0;
    this.bright = 0;
    this.full = 0;
    this.detune = 0.001;
    this.count = 0;
    // Per voice (root, fifth) and slot: two phases, increments and a weight.
    const n = 2 * SLOTS;
    this.phase = [new Float64Array(n), new Float64Array(n)];
    this.inc = [new Float64Array(n), new Float64Array(n)];
    this.weight = new Float64Array(n);
    this.gain = 0;
  }

  next(x, y, vx, vy, window, rate) {
    const a = 1 / (window * rate);
    this.mx += a * (x - this.mx);
    this.my += a * (y - this.my);
    const dx = x - this.mx;
    const dy = y - this.my;
    this.cxx += a * (dx * dx - this.cxx);
    this.cyy += a * (dy * dy - this.cyy);
    this.cxy += a * (dx * dy - this.cxy);
    this.rr += a * (x * x + y * y - this.rr);
    this.rot += a * (x * vy - y * vx - this.rot);
    if (this.count++ % BLOCK === 0) this.#control(rate);

    let s = 0;
    for (let v = 0; v < 2; v++) {
      const ph = this.phase[v];
      const inc = this.inc[v];
      for (let k = 0; k < SLOTS; k++) {
        const j = v * SLOTS + k;
        ph[k] += inc[k];
        ph[k + SLOTS] += inc[k + SLOTS];
        if (ph[k] > TAU) ph[k] -= TAU;
        if (ph[k + SLOTS] > TAU) ph[k + SLOTS] -= TAU;
        const w = this.weight[j];
        if (w > 1e-3) s += w * (Math.sin(ph[k]) + Math.sin(ph[k + SLOTS]));
      }
    }
    return this.gain * s;
  }

  #control(rate) {
    const tr = this.cxx + this.cyy;
    const size = Math.sqrt(Math.max(tr, 0));
    const spread = Math.sqrt((this.cxx - this.cyy) ** 2 + 4 * this.cxy ** 2);
    const elong = tr > 1e-9 ? spread / tr : 0; // 0 round, 1 a line

    // Only a clearly elongated trail has an axis to follow.
    if (elong > 0.2 && size > 0.02) {
      const phi = 0.5 * Math.atan2(2 * this.cxy, this.cxx - this.cyy);
      let d = phi - this.axis;
      d -= Math.PI * Math.round(d / Math.PI);
      this.axis += d;
      const D = (this.axis / Math.PI) * SCALE.length;
      if (Math.abs(D - this.degree) > 0.6) this.degree = Math.round(D);
    }
    const W = this.rr > 1e-9 ? this.rot / this.rr : 0; // net turning, rad/s
    const spin = clamp01(Math.abs(W) / (TAU * 8));

    const ease = (tau) => 1 - Math.exp(-BLOCK / (rate * tau));
    this.pitch += (scaleNote(this.degree) - this.pitch) * ease(0.15);
    this.amp += (0.3 * smoothstep(0.02, 0.35, size) - this.amp) * ease(0.08);
    const bright = clamp01(1.5 * size + 0.15 * Math.sign(W) * spin);
    this.bright += (bright - this.bright) * ease(0.2);
    this.full += (1 - elong - this.full) * ease(0.3);
    this.detune += (0.001 + 0.006 * spin - this.detune) * ease(0.3);

    const centre = 2.3 + 2.2 * this.bright; // in octaves above A0: about 135 to 620 Hz
    const sigma = 0.9 + 0.4 * this.bright;
    let total = 0;
    for (let v = 0; v < 2; v++) {
      const p = this.pitch + 7 * v;
      const level = v === 0 ? 1 : 0.6 * this.full;
      for (let k = 0; k < SLOTS; k++) {
        let e = (p / 12 + k) % SLOTS;
        if (e < 0) e += SLOTS;
        const f = F0 * 2 ** e;
        const w = level * Math.exp(-((e - centre) ** 2) / (2 * sigma * sigma));
        this.weight[v * SLOTS + k] = w;
        this.inc[v][k] = (TAU * f) / rate;
        this.inc[v][k + SLOTS] = (TAU * f * (1 + this.detune)) / rate;
        total += w;
      }
    }
    this.gain = total > 0 ? this.amp / (2 * total) : 0;
  }
}

// Marimba: soft, marimba-like notes for the plucks. The page decides when and what to
// strike (see plucks.js); this only plays them. Each note is three decaying partials at a
// marimba bar's stretched ratios, with a 2 ms attack.
const PARTIALS = [
  { ratio: 1, amp: 1, decay: 0.9 },
  { ratio: 3.93, amp: 0.25, decay: 0.22 },
  { ratio: 9.2, amp: 0.08, decay: 0.07 },
];
const MAX_VOICES = 6;
export class Marimba {
  constructor() {
    this.voices = [];
  }

  strike(freq, velocity, rate) {
    const shorten = Math.sqrt(220 / freq); // higher notes ring shorter
    this.voices.push({
      t: 0,
      phase: PARTIALS.map(() => 0),
      inc: PARTIALS.map((p) => (TAU * freq * p.ratio) / rate),
      amp: PARTIALS.map((p) => velocity * p.amp),
      decay: PARTIALS.map((p) => p.decay * shorten),
    });
    if (this.voices.length > MAX_VOICES) this.voices.shift();
  }

  next(rate) {
    let s = 0;
    for (const voice of this.voices) {
      const t = voice.t;
      voice.t += 1 / rate;
      const attack = Math.min(1, t / 0.002);
      for (let k = 0; k < PARTIALS.length; k++) {
        voice.phase[k] += voice.inc[k];
        s += voice.amp[k] * Math.exp(-t / voice.decay[k]) * attack * Math.sin(voice.phase[k]);
      }
    }
    if (this.voices.length && this.voices[0].t > 4 * this.voices[0].decay[0]) this.voices.shift();
    return 0.3 * s;
  }
}
