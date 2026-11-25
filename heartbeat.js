// Heartbeat detection from the Muse 2's PPG (light reflected from the forehead, 64 Hz).
// Each beat shows as a quick swing in the light level followed by a slow return. The
// signal is band-passed to the heart-rate range, and a beat is called when its slope
// crosses half its recent peak, in the direction the quick swing goes (which depends on
// the sensor, so it's found from the data).

import { highpass, lowpass } from './filters.js';

const WARMUP_S = 3; // let the filters and envelopes settle first
const MIN_INTERVAL_S = 0.3; // 200 bpm
const MAX_INTERVAL_S = 2; // 30 bpm; longer gaps don't count towards the rate
const THRESHOLD = 0.5; // fraction of the recent peak slope that calls a beat

export class HeartbeatDetector {
  constructor(rate) {
    this.rate = rate;
    this.filters = [highpass(0.7, rate), lowpass(4, rate), lowpass(4, rate)];
    this.decay = 0.5 ** (1 / (1.5 * rate)); // peak envelopes halve in 1.5 s
    this.follow = 1 / (4 * rate); // the polarity estimate follows over about 4 s
    this.reset();
  }

  reset() {
    this.filters.forEach((f) => (f.primed = false));
    this.prev = null;
    this.rise = 0; // decaying peak of the slope going up, and of it going down
    this.fall = 0;
    this.riseAvg = 0;
    this.fallAvg = 0;
    this.armed = false;
    this.count = 0;
    this.sinceBeat = Infinity; // samples
    this.intervals = [];
    this.bpm = null;
  }

  // Feed one sample; returns true when it completes a beat.
  process(x) {
    for (const f of this.filters) x = f.process(x);
    const prev = this.prev;
    this.prev = x;
    if (prev === null) return false;
    const slope = x - prev;
    this.count++;
    this.sinceBeat++;

    this.rise = Math.max(slope, this.rise * this.decay);
    this.fall = Math.max(-slope, this.fall * this.decay);
    this.riseAvg += (this.rise - this.riseAvg) * this.follow;
    this.fallAvg += (this.fall - this.fallAvg) * this.follow;
    // The quick swing has the steeper slope.
    const up = this.riseAvg >= this.fallAvg;
    const s = up ? slope : -slope;
    const peak = up ? this.rise : this.fall;

    if (s <= 0) this.armed = true;
    if (this.count < WARMUP_S * this.rate || !this.armed || peak <= 0) return false;
    // A beat can't come sooner than 0.3 s, or than 60% of the usual interval, which keeps
    // the smaller second swing of each pulse (the dicrotic notch) from counting.
    const usual = this.bpm ? (60 / this.bpm) * this.rate : 0;
    const refractory = Math.max(MIN_INTERVAL_S * this.rate, 0.6 * usual);
    if (s < THRESHOLD * peak || this.sinceBeat < refractory) return false;

    if (this.sinceBeat <= MAX_INTERVAL_S * this.rate) {
      this.intervals.push(this.sinceBeat / this.rate);
      if (this.intervals.length > 5) this.intervals.shift();
      const sorted = [...this.intervals].sort((a, b) => a - b);
      this.bpm = 60 / sorted[sorted.length >> 1];
    } else {
      this.intervals = [];
      this.bpm = null;
    }
    this.armed = false;
    this.sinceBeat = 0;
    return true;
  }
}
