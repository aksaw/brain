// Streaming second-order IIR filters (RBJ audio EQ cookbook), transposed direct form II.

export class Biquad {
  constructor(b0, b1, b2, a0, a1, a2) {
    this.b0 = b0 / a0;
    this.b1 = b1 / a0;
    this.b2 = b2 / a0;
    this.a1 = a1 / a0;
    this.a2 = a2 / a0;
    this.s1 = 0;
    this.s2 = 0;
    this.primed = false;
  }

  process(x) {
    if (!this.primed) {
      // Start as if the first sample had been constant forever, so a large DC
      // offset doesn't make the filter ring when streaming begins.
      const gain = (this.b0 + this.b1 + this.b2) / (1 + this.a1 + this.a2);
      this.s2 = (this.b2 - this.a2 * gain) * x;
      this.s1 = (this.b1 - this.a1 * gain) * x + this.s2;
      this.primed = true;
    }
    const y = this.b0 * x + this.s1;
    this.s1 = this.b1 * x - this.a1 * y + this.s2;
    this.s2 = this.b2 * x - this.a2 * y;
    return y;
  }
}

// Butterworth high-pass: removes slow drift below `cutoff` Hz.
export function highpass(cutoff, rate) {
  const w = (2 * Math.PI * cutoff) / rate;
  const cos = Math.cos(w);
  const alpha = Math.sin(w) / (2 * Math.SQRT1_2);
  return new Biquad((1 + cos) / 2, -(1 + cos), (1 + cos) / 2, 1 + alpha, -2 * cos, 1 - alpha);
}

// Butterworth low-pass: removes content above `cutoff` Hz.
export function lowpass(cutoff, rate) {
  const w = (2 * Math.PI * cutoff) / rate;
  const cos = Math.cos(w);
  const alpha = Math.sin(w) / (2 * Math.SQRT1_2);
  return new Biquad((1 - cos) / 2, 1 - cos, (1 - cos) / 2, 1 + alpha, -2 * cos, 1 - alpha);
}

// Narrow notch at `freq` Hz for mains interference; higher q means narrower.
export function notch(freq, rate, q = 20) {
  const w = (2 * Math.PI * freq) / rate;
  const cos = Math.cos(w);
  const alpha = Math.sin(w) / (2 * q);
  return new Biquad(1, -2 * cos, 1, 1 + alpha, -2 * cos, 1 - alpha);
}
