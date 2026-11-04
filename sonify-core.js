// Sonification DSP, shared by the audio worklet and by tests. EEG arrives at 256 Hz and
// is turned into audio at the AudioContext's sample rate.

export const EEG_RATE = 256;

// Scales a signal to a typical amplitude of 0.25 by its own running RMS, so the sound
// stays in range across fits and sessions.
export class Normalizer {
  constructor(tau = 2) {
    this.weight = 1 - Math.exp(-1 / (tau * EEG_RATE));
    this.meanSquare = 100; // start as if the signal were 10 µV RMS
  }

  process(x) {
    this.meanSquare += this.weight * (x * x - this.meanSquare);
    return (0.25 * x) / Math.sqrt(this.meanSquare + 1e-6);
  }
}

// Audification as a wavetable: the last second of EEG, mean removed and tapered at both
// ends so the loop joins smoothly, is played as one cycle of a tone at `pitch` Hz. With a
// one-second table an EEG rhythm at n Hz becomes the tone's nth harmonic, so the EEG
// spectrum is heard as the tone's timbre.
export class Wavetable {
  constructor(length = EEG_RATE) {
    this.table = new Float32Array(length);
    this.taper = tukey(length, 0.25);
    this.write = 0; // the oldest sample; the next one is written here
    this.sum = 0;
    this.phase = 0;
  }

  push(x) {
    this.sum += x - this.table[this.write];
    this.table[this.write] = x;
    this.write = (this.write + 1) % this.table.length;
  }

  next(pitch, sampleRate) {
    const n = this.table.length;
    const mean = this.sum / n;
    const pos = this.phase * n; // one cycle runs from the oldest sample to the newest
    const i = Math.floor(pos);
    const f = pos - i;
    const a = (this.table[(this.write + i) % n] - mean) * this.taper[i];
    const b = i + 1 < n ? (this.table[(this.write + i + 1) % n] - mean) * this.taper[i + 1] : 0;
    this.phase += pitch / sampleRate;
    if (this.phase >= 1) this.phase -= 1;
    return a + (b - a) * f;
  }
}

// Frequency shifting: the EEG's analytic signal (from a Hilbert FIR) is upsampled and
// rotated by a complex oscillator, so a rhythm at f Hz is heard at shift + f Hz, in real
// time and without the mirror image below the shift that plain multiplication would add.
export class Heterodyne {
  constructor(halfTaps = 31) {
    this.half = halfTaps; // the FIR delays the signal by this many EEG samples (~120 ms)
    this.taps = hilbertTaps(halfTaps);
    this.history = new Float32Array(2 * halfTaps + 1);
    this.pos = 0;
    this.filled = 0;
    // Analytic samples at the EEG rate, waiting to be played.
    this.inphase = [];
    this.quadrature = [];
    this.read = 0; // fractional position in the queue
    this.playing = false;
    this.theta = 0;
  }

  push(x) {
    const len = this.history.length;
    this.history[this.pos] = x;
    this.pos = (this.pos + 1) % len;
    if (++this.filled < len) return;
    // Sample n-j sits at pos-1-j. The centre of the window, n-half, is the in-phase part.
    const at = (j) => this.history[(this.pos - 1 - j + 2 * len) % len];
    let q = 0;
    for (let k = -this.half; k <= this.half; k++) q += this.taps[k + this.half] * at(this.half + k);
    this.inphase.push(at(this.half));
    this.quadrature.push(q);
    // Nothing is consumed while another mode plays; keep the queue from growing forever.
    if (this.inphase.length > 4 * EEG_RATE) this.#drop(this.inphase.length - 2 * EEG_RATE);
  }

  next(shift, sampleRate) {
    const target = 0.2 * EEG_RATE; // queued EEG samples to aim for, to absorb bursty packets
    const ahead = this.inphase.length - 1 - this.read;
    if (!this.playing) {
      if (ahead < target) return 0; // (re)fill the buffer before starting
      this.playing = true;
    }
    if (ahead < 1) {
      this.playing = false; // ran dry: wait for the buffer to refill
      return 0;
    }
    if (ahead > 3 * target) this.read = this.inphase.length - 1 - target; // too far behind: skip

    const k = Math.floor(this.read);
    const f = this.read - k;
    const i = this.inphase[k] + (this.inphase[k + 1] - this.inphase[k]) * f;
    const q = this.quadrature[k] + (this.quadrature[k + 1] - this.quadrature[k]) * f;
    const y = i * Math.cos(this.theta) - q * Math.sin(this.theta);

    this.theta += (2 * Math.PI * shift) / sampleRate;
    if (this.theta > 2 * Math.PI) this.theta -= 2 * Math.PI;
    // Read up to 1% faster or slower depending on the fill, to follow the headband's clock.
    const drift = Math.max(-1, Math.min(1, (ahead - target) / target));
    this.read += ((1 + 0.01 * drift) * EEG_RATE) / sampleRate;
    if (this.read > 2 * EEG_RATE) this.#drop(Math.floor(this.read));
    return y;
  }

  #drop(count) {
    this.inphase.splice(0, count);
    this.quadrature.splice(0, count);
    this.read = Math.max(0, this.read - count);
  }
}

// Hilbert transformer taps for k = -half..half: 2/(pi k) for odd k, Blackman-windowed.
export function hilbertTaps(half) {
  const taps = new Float32Array(2 * half + 1);
  for (let k = -half; k <= half; k++) {
    if (k % 2 === 0) continue;
    const j = k + half;
    const window =
      0.42 - 0.5 * Math.cos((Math.PI * j) / half) + 0.08 * Math.cos((2 * Math.PI * j) / half);
    taps[j] = (2 / (Math.PI * k)) * window;
  }
  return taps;
}

// Tukey window: flat in the middle, with cosine tapers covering `fraction` of the length.
export function tukey(length, fraction) {
  const w = new Float32Array(length);
  const edge = (fraction * (length - 1)) / 2;
  for (let i = 0; i < length; i++) {
    const d = Math.min(i, length - 1 - i);
    w[i] = d >= edge ? 1 : 0.5 - 0.5 * Math.cos((Math.PI * d) / edge);
  }
  return w;
}
