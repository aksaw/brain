// Sonification DSP, shared by the audio worklet and by tests. EEG arrives at 256 Hz and
// is turned into audio at the AudioContext's sample rate.
//
// Most modes hear each side's x-y trace as one complex signal, z = x + iy (AF7 + i TP9 on
// the left, AF8 + i TP10 on the right), the same point the plot draws. Motion turning
// anticlockwise at f Hz is a component at +f, clockwise at -f, so the direction of
// rotation becomes audible as above or below a note.

import { BANDS, fft } from './bands.js';
import { highpass, lowpass } from './filters.js';

export const EEG_RATE = 256;

// Seconds of EEG kept queued ahead of playback, to ride out bursty Bluetooth packets
// (arrival gaps are typically under 90 ms). Shorter means less delay behind the plots.
const BUFFER_S = 0.12;

// Scales a signal to a typical amplitude of 0.25 by its own running RMS, so the sound
// stays in range across fits and sessions. The average is slow (30 s), so real changes,
// such as alpha swelling with the eyes closed, are heard rather than levelled out.
export class Normalizer {
  constructor(tau = 30) {
    this.weight = 1 - Math.exp(-1 / (tau * EEG_RATE));
    this.meanSquare = 0;
    this.count = 0;
  }

  // The gain to apply, given the newest sample's power (x² for a signal, x² + y² for a point).
  gain(power) {
    // A plain running mean until the slow average has had time to settle.
    this.count++;
    this.meanSquare += Math.max(1 / this.count, this.weight) * (power - this.meanSquare);
    return 0.25 / Math.sqrt(this.meanSquare + 1e-6);
  }

  process(x) {
    return x * this.gain(x * x);
  }
}

// Running FFT peak analysis for the spectral map. The last
// `size` samples of a signal (complex x + iy, or real with y = 0) are windowed and
// transformed 32 times a second. Only spectral peaks are kept: one sine spreads over
// neighbouring bins, and separate oscillators for them would beat against each other.
// The phase change since the previous frame gives each peak's exact frequency, to a small
// fraction of a bin (the phase vocoder). Frames queue up and are read back in step with
// the audio, buffered against bursty packets and following the headband's clock.
class PeakAnalyser {
  constructor(size, hop = 8) {
    this.size = size; // delay is about half the window
    this.hop = hop; // a new frame every 8 EEG samples, 32 per second
    this.frameRate = EEG_RATE / hop;
    this.window = hann(size);
    this.historyX = new Float32Array(size);
    this.historyY = new Float32Array(size);
    this.pos = 0; // the oldest sample; the next one is written here
    this.count = 0;
    this.lastPhase = new Float32Array(size);
    this.frames = []; // { amp, freq } per bin, freq signed in EEG Hz, waiting to be played
    this.read = 0; // fractional position in the frame queue
    this.playing = false;
  }

  push(x, y) {
    this.historyX[this.pos] = x;
    this.historyY[this.pos] = y;
    this.pos = (this.pos + 1) % this.size;
    this.count++;
    if (this.count >= this.size && (this.count - this.size) % this.hop === 0) {
      this.frames.push(this.#analyse());
    }
    // Nothing is consumed while another mode plays; keep the queue from growing forever.
    if (this.frames.length > 4 * this.frameRate) {
      this.#drop(this.frames.length - 2 * this.frameRate);
    }
  }

  // Moves the read position on by one audio block and returns the peaks there: each bin's
  // amplitude and signed frequency, interpolated between frames. Null while the buffer
  // (re)fills.
  advance(blockLength, sampleRate) {
    const target = BUFFER_S * this.frameRate; // queued frames to aim for
    const ahead = this.frames.length - 1 - this.read;
    if (!this.playing) {
      if (ahead < target) return null; // (re)fill the buffer before starting
      this.playing = true;
    }
    if (ahead < 1) {
      this.playing = false; // ran dry: wait for the buffer to refill
      return null;
    }
    if (ahead > 3 * target) this.read = this.frames.length - 1 - target; // too far behind: skip

    // Read up to 1% faster or slower depending on the fill, to follow the headband's clock.
    const drift = Math.max(-1, Math.min(1, (ahead - target) / target));
    this.read += ((1 + 0.01 * drift) * this.frameRate * blockLength) / sampleRate;
    this.read = Math.min(this.read, this.frames.length - 1);
    const peaks = this.#peaksAt(this.read);
    if (this.read > 2 * this.frameRate) this.#drop(Math.floor(this.read));
    return peaks;
  }

  #peaksAt(pos) {
    const i = Math.floor(pos);
    const f = pos - i;
    const a = this.frames[i];
    const b = this.frames[Math.min(i + 1, this.frames.length - 1)];
    const amp = new Float32Array(this.size);
    const freq = new Float32Array(this.size);
    for (let k = 1; k < this.size; k++) {
      amp[k] = a.amp[k] + (b.amp[k] - a.amp[k]) * f;
      if (amp[k] === 0) continue;
      freq[k] =
        a.amp[k] > 0 && b.amp[k] > 0
          ? a.freq[k] + (b.freq[k] - a.freq[k]) * f
          : a.amp[k] > 0
            ? a.freq[k]
            : b.freq[k];
    }
    return { amp, freq };
  }

  #analyse() {
    const n = this.size;
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const j = (this.pos + i) % n;
      re[i] = this.historyX[j] * this.window[i];
      im[i] = this.historyY[j] * this.window[i];
    }
    fft(re, im);

    const mag = new Float32Array(n);
    for (let k = 0; k < n; k++) mag[k] = Math.hypot(re[k], im[k]);
    const amp = new Float32Array(n);
    const freq = new Float32Array(n);
    for (let k = 1; k < n; k++) {
      if (k === n / 2) continue; // the Nyquist bin has no direction
      const signed = k < n / 2 ? k : k - n; // bins above n/2 are negative (clockwise)
      const phase = Math.atan2(im[k], re[k]);
      let deviation = phase - this.lastPhase[k] - (2 * Math.PI * signed * this.hop) / n;
      deviation -= 2 * Math.PI * Math.round(deviation / (2 * Math.PI));
      this.lastPhase[k] = phase;
      if (mag[k] < mag[k - 1] || mag[k] <= mag[(k + 1) % n]) continue;
      amp[k] = (2 * mag[k]) / n; // a Hann-windowed rotation of radius A peaks at A n / 2
      freq[k] = ((signed + (deviation * n) / (2 * Math.PI * this.hop)) * EEG_RATE) / n;
    }
    return { amp, freq };
  }

  #drop(count) {
    this.frames.splice(0, count);
    this.read = Math.max(0, this.read - count);
  }
}

// Granular: the EEG itself, played back sped up. Many short overlapping grains each play
// the last half-second of signal at `speed` times real time, so a rhythm at f Hz is heard
// at speed x f Hz (alpha at x50 sits around 500 Hz). Grains start at random moments and
// from slightly different points in the recent past, so there is no repeating loop and so
// no fixed pitch to buzz: what remains is the EEG's own spectrum, as textured noise.
const GRAIN_EEG_S = 0.5; // EEG each grain plays
const GRAIN_SCATTER_S = 0.25; // grains start up to this much further back, at random
const GRAIN_MARGIN_S = 0.05; // stay clear of the newest samples, which arrive in bursts
const GRAIN_OVERLAP = 4; // grains sounding at once, on average

export class Granular {
  constructor() {
    this.ring = new Float32Array(4 * EEG_RATE);
    this.written = 0; // EEG samples received so far
    this.grains = []; // { pos: EEG sample position (fractional), age: audio samples }
    this.untilNext = 0; // audio samples until the next grain starts
  }

  push(x) {
    this.ring[this.written % this.ring.length] = x;
    this.written++;
  }

  next(speed, sampleRate) {
    const reach = (GRAIN_EEG_S + GRAIN_SCATTER_S + GRAIN_MARGIN_S) * EEG_RATE;
    if (this.written < reach) return 0;
    const length = Math.round((GRAIN_EEG_S / speed) * sampleRate); // a grain, in audio samples
    if (--this.untilNext <= 0) {
      const back = (GRAIN_EEG_S + GRAIN_MARGIN_S + Math.random() * GRAIN_SCATTER_S) * EEG_RATE;
      this.grains.push({ pos: this.written - back, age: 0 });
      this.untilNext = (length / GRAIN_OVERLAP) * (0.5 + Math.random()); // irregular spacing
    }

    const step = (speed * EEG_RATE) / sampleRate; // EEG samples per audio sample
    const n = this.ring.length;
    let out = 0;
    for (let g = this.grains.length - 1; g >= 0; g--) {
      const grain = this.grains[g];
      const i = Math.floor(grain.pos);
      const f = grain.pos - i;
      const a = this.ring[i % n];
      const b = this.ring[(i + 1) % n];
      const envelope = 0.5 - 0.5 * Math.cos((2 * Math.PI * grain.age) / length); // Hann
      out += envelope * (a + (b - a) * f);
      grain.pos += step;
      if (++grain.age >= length) this.grains.splice(g, 1); // finished
    }
    // Overlapping grains are unrelated, so their power adds: scale back to the input level.
    return out / Math.sqrt(GRAIN_OVERLAP * 0.375);
  }
}

// Bitcrusher, optional on the granular sound: the signal is held at a lower sample rate
// (each value repeated, so frequencies above half that rate fold back down as aliasing)
// and rounded to a few bits, for a gritty, stepped, lo-fi sound.
const CRUSH_BITS = 6; // 64 levels across -1..1
const CRUSH_RATE = 8000; // Hz

export class Bitcrusher {
  constructor() {
    this.held = 0;
    this.countdown = 0;
  }

  process(x, sampleRate) {
    if (--this.countdown <= 0) {
      const step = 2 / 2 ** CRUSH_BITS;
      this.held = step * Math.round(x / step);
      this.countdown += sampleRate / CRUSH_RATE;
    }
    return this.held;
  }
}

// Spectral map: the trace's spectrum is re-synthesised across the audible range. The peak
// analyser finds its rhythms, positive (anticlockwise) and negative (clockwise); a
// mirrored log map sends anticlockwise motion up from the centre of the audio range and
// clockwise motion down from it. Each mapped peak then snaps to the nearest harmonic of
// 55 Hz, and each harmonic fades in and out at its fixed pitch: a consonant shimmer.
const HARMONIC_BASE = 55;
const HARMONIC_GLIDE_S = 0.15;

export class SpectralMap {
  constructor() {
    this.peaks = new PeakAnalyser(EEG_RATE); // 1 s window: 1 Hz bins, about 0.5 s of delay
    this.levels = new Float32Array(512); // current loudness of each harmonic of 55 Hz
    this.phases = new Float64Array(512);
  }

  push(x, y) {
    this.peaks.push(x, y);
  }

  // Adds this side's sound for one audio block into `out`.
  render(out, map, sampleRate) {
    const peaks = this.peaks.advance(out.length, sampleRate);
    if (!peaks) return;

    // Every peak is mapped into the audio range and its amplitude added to the nearest
    // harmonic of 55 Hz.
    const top = Math.min(map.audioHigh, sampleRate / 2);
    const targets = new Float32Array(Math.floor(top / HARMONIC_BASE) + 1);
    for (let k = 1; k < peaks.amp.length; k++) {
      if (peaks.amp[k] === 0) continue;
      const speed = Math.abs(peaks.freq[k]);
      if (speed < map.eegLow || speed > map.eegHigh) continue;
      const harmonic = Math.max(1, Math.round(mapRotation(peaks.freq[k], map) / HARMONIC_BASE));
      if (harmonic >= targets.length) continue;
      // EEG amplitude falls roughly as 1/f; boosting by f evens out the mapped spectrum
      // so the far ends of the audio range aren't nearly silent.
      targets[harmonic] += peaks.amp[k] * (speed / 10);
    }

    // Each harmonic's loudness glides towards its target across the block.
    const len = out.length;
    const glide = 1 - Math.exp(-len / (HARMONIC_GLIDE_S * sampleRate));
    for (let h = 1; h < targets.length; h++) {
      const a0 = this.levels[h];
      const a1 = a0 + glide * (targets[h] - a0);
      this.levels[h] = a1;
      if (a0 < 1e-5 && a1 < 1e-5) continue;
      const inc = (2 * Math.PI * h * HARMONIC_BASE) / sampleRate;
      let phase = this.phases[h];
      for (let n = 0; n < len; n++) {
        phase += inc;
        out[n] += (a0 + ((a1 - a0) * n) / len) * Math.sin(phase);
      }
      this.phases[h] = phase % (2 * Math.PI);
    }
  }
}

// Mirrored log map for signed EEG frequencies: anticlockwise (positive) frequencies rise
// from the centre of the audio range (the geometric mean of its ends) to its top, and
// clockwise (negative) ones fall from the centre to its bottom. Ratios stay ratios.
export function mapRotation(eeg, { eegLow, eegHigh, audioLow, audioHigh }) {
  const centre = Math.sqrt(audioLow * audioHigh);
  const exponent = Math.log(audioHigh / centre) / Math.log(eegHigh / eegLow);
  const ratio = (Math.abs(eeg) / eegLow) ** exponent;
  return eeg > 0 ? centre * ratio : centre / ratio;
}

export function hann(length) {
  const w = new Float32Array(length);
  for (let i = 0; i < length; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / length);
  return w;
}

// Frequency shift of a complex signal: the point x + iy, queued at the EEG rate, is
// upsampled and rotated by a complex oscillator at the carrier, so motion turning
// anticlockwise at f Hz is heard at carrier + f and clockwise at carrier - f. The trace is
// already complex, so no Hilbert filter (and none of its delay) is needed.
export class ComplexShift {
  constructor() {
    this.re = [];
    this.im = [];
    this.read = 0; // fractional position in the queue
    this.playing = false;
    this.theta = 0;
  }

  push(x, y) {
    this.re.push(x);
    this.im.push(y);
    // Nothing is consumed while another mode plays; keep the queue from growing forever.
    if (this.re.length > 4 * EEG_RATE) this.#drop(this.re.length - 2 * EEG_RATE);
  }

  next(carrier, sampleRate) {
    const target = BUFFER_S * EEG_RATE; // queued EEG samples to aim for
    const ahead = this.re.length - 1 - this.read;
    if (!this.playing) {
      if (ahead < target) return 0; // (re)fill the buffer before starting
      this.playing = true;
    }
    if (ahead < 1) {
      this.playing = false; // ran dry: wait for the buffer to refill
      return 0;
    }
    if (ahead > 3 * target) this.read = this.re.length - 1 - target; // too far behind: skip

    const k = Math.floor(this.read);
    const f = this.read - k;
    const x = this.re[k] + (this.re[k + 1] - this.re[k]) * f;
    const y = this.im[k] + (this.im[k + 1] - this.im[k]) * f;
    const out = x * Math.cos(this.theta) - y * Math.sin(this.theta); // Re(z e^(i theta))

    this.theta += (2 * Math.PI * carrier) / sampleRate;
    if (this.theta > 2 * Math.PI) this.theta -= 2 * Math.PI;
    // Read up to 1% faster or slower depending on the fill, to follow the headband's clock.
    const drift = Math.max(-1, Math.min(1, (ahead - target) / target));
    this.read += ((1 + 0.01 * drift) * EEG_RATE) / sampleRate;
    if (this.read > 2 * EEG_RATE) this.#drop(Math.floor(this.read));
    return out;
  }

  #drop(count) {
    this.re.splice(0, count);
    this.im.splice(0, count);
    this.read = Math.max(0, this.read - count);
  }
}

// The chord the layered shift plays on: A in five octaves (A1 to A5), one per band from
// delta at the bottom to gamma at the top. Each band fills both sides of its note, so the
// notes are an octave apart to give every layer its own register instead of crowding.
export const BAND_PITCHES = [55, 110, 220, 440, 880];

// Layered shift: each EEG band of the trace is filtered out and frequency-shifted onto its
// note of the chord above, keeping its own windy texture: the band's anticlockwise motion
// sits just above the note and its clockwise motion just below. Each layer is scaled by
// its own slow running level, so the layers balance on average and each swells and fades
// with its band. Blinks live in the delta layer, around the bottom note.
const LAYER_GAIN = 0.5;

export class LayeredShift {
  constructor() {
    // Band-pass from two high-pass and two low-pass stages (24 dB per octave each side),
    // one set for x and one for y.
    const bandpass = (band) => [
      highpass(band.lo, EEG_RATE),
      highpass(band.lo, EEG_RATE),
      lowpass(band.hi, EEG_RATE),
      lowpass(band.hi, EEG_RATE),
    ];
    this.layers = BANDS.map((band, b) => ({
      filtersX: bandpass(band),
      filtersY: bandpass(band),
      normalizer: new Normalizer(),
      shift: new ComplexShift(),
      carrier: BAND_PITCHES[b],
    }));
  }

  push(x, y) {
    for (const layer of this.layers) {
      let bx = x;
      let by = y;
      for (const filter of layer.filtersX) bx = filter.process(bx);
      for (const filter of layer.filtersY) by = filter.process(by);
      const g = layer.normalizer.gain(bx * bx + by * by);
      layer.shift.push(g * bx, g * by);
    }
  }

  next(sampleRate) {
    let out = 0;
    for (const layer of this.layers) out += LAYER_GAIN * layer.shift.next(layer.carrier, sampleRate);
    return out;
  }
}
