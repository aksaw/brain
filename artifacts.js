// Rule-based artifact detection for the Muse's four EEG channels and motion sensors.
// Each detector runs on its own filtered copy of the raw signal, and thresholds are
// relative to each channel's own recent level. The constants are starting points to
// tune against real recordings, not calibrated values.
import { highpass, lowpass, notch } from './filters.js';

const EARS = [0, 3]; // TP9, TP10 in EEG_CHANNELS order
const WARMUP_MS = 3000; // let filters and baselines settle before flagging anything
const KEEP_MS = 60000;

// Blinks: every channel is measured against the reference at Fpz, just above the eyes.
// A blink drives the forehead positive, so it largely cancels on AF7/AF8 (which rise with
// Fpz) and shows as a sharp negative dip on both ear channels together (which don't).
// Because it enters through the reference, a blink is marked on all four channels.
// Times the usual mean absolute level of the 0.5-10 Hz signal. Picked by replaying a real
// recording: 3 caught 11 of 13 blinks with 1 false alarm; 2 added 6 false alarms.
const BLINK_THRESHOLD = 3;
const BLINK_MIN_UV = 40;
const BLINK_MIN_MS = 40; // time spent above threshold
const BLINK_MAX_MS = 600;
const BLINK_PAIR_MS = 100; // max gap between the TP9 and TP10 peaks, in sample time
const BLINK_PAD_BEFORE_MS = 150;
const BLINK_PAD_AFTER_MS = 400; // covers the slow return to baseline after the lid opens
// The detection filter rebounds after a blink with the opposite sign, which would
// otherwise read as a second blink; real blinks are much further apart than this.
const BLINK_REFRACTORY_MS = 500;
const BLINK_SCALE_TAU_S = 10;

// Muscle: broadband power above 30 Hz rising well above the channel's own baseline.
const MUSCLE_POWER_TAU_S = 0.1;
const MUSCLE_BASELINE_TAU_S = 20;
const MUSCLE_ON_DB = 8;
const MUSCLE_OFF_DB = 4;
const MUSCLE_PAD_MS = 100;

// Movement: head rotation or a change in acceleration, flagged on every channel.
const GYRO_DPS = 20;
const ACCEL_CHANGE_G = 0.1;
const ACCEL_MEAN_TAU_S = 1;
const ACCEL_RATE = 52; // Muse 2 IMU samples per second
const MOVEMENT_PAD_MS = 250;

// Sideways eye movements: the eye is a dipole (cornea positive), so a glance left raises
// the left-side channels (TP9, AF7) and lowers the right-side ones (AF8, TP10), and the
// shift holds while the gaze does. Detected as a step in left - right that is opposite on
// the two sides and persists, which a blink (same sign on both sides, brief) does not.
const SACCADE_THRESHOLD = 4; // times the usual size of the step measure
const SACCADE_MIN_UV = 30;
// The shift is re-measured over the last 100 ms of a 400 ms hold. The heartbeat also
// moves the two sides in opposite directions, but it has swung back by then.
const SACCADE_HOLD_MS = 400;
const SACCADE_CHECK_MS = 100;
const SACCADE_KEEP_FRACTION = 0.7; // of the threshold, after the hold
const SACCADE_REFRACTORY_MS = 300;
const SACCADE_BALANCE = 1 / 3; // the smaller side's shift, as a fraction of the larger
const SACCADE_SCALE_TAU_S = 10;
const LOOK_LEFT_SIGN = 1; // flip to -1 if the eyes look the wrong way on your headband
const HEOG_RING = 1024; // samples of each filtered channel kept for step measurements
const SIDES = [
  [0, 1], // left of the head: TP9, AF7
  [2, 3], // right of the head: AF8, TP10
];

// Electrode pops: a near-instant jump on one channel, far bigger than sample-to-sample noise.
const POP_THRESHOLD = 10; // times the usual sample-to-sample change
const POP_MIN_UV = 150;
const POP_BEFORE_MS = 100;
const POP_AFTER_MS = 500; // the electrode takes a moment to settle
const POP_SCALE_TAU_S = 10;

// Exponential smoothing weight for a time constant `tau` (s) at `rate` samples/s.
const smoothing = (tau, rate) => 1 - Math.exp(-1 / (tau * rate));
// Baselines start from a filter's first output (zero), so they track fast during warm-up
// and only slow down once detection starts.
const WARMUP_TAU_S = 0.3;

export class ArtifactDetector {
  constructor(rate, channelCount) {
    this.rate = rate;
    this.channels = [...Array(channelCount).keys()];
    this.startTime = null;
    // Finished artifacts: { type, channels, start, end } with times in ms, where type is
    // 'blink' | 'eye' | 'muscle' | 'movement' | 'pop'.
    this.intervals = [];
    this.blinkTimes = [];
    this.lastBlinkEndIndex = -Infinity; // sample index (shared across channels)
    this.onBlink = () => {};
    this.onEyeMove = (_direction) => {}; // 'left' | 'right'

    // Packets arrive per channel, in bursts and not in step across channels, so AF7 and
    // AF8 are lined up by sample index (from the packet sequence number), not arrival time.
    this.sequence = this.channels.map(() => ({ last: null, wraps: 0 }));

    this.blink = EARS.map(() => ({
      chain: [highpass(0.5, rate), lowpass(10, rate)],
      scale: null,
      above: false,
      start: 0,
      startIndex: 0,
      peak: 0,
      peakTime: 0,
      peakIndex: 0,
      sign: 0,
      candidates: [],
    }));

    // Window lengths for the step measure, in samples: the mean over the last 50 ms
    // minus the mean over 150-300 ms earlier. That difference already ignores slow
    // drift, so there's no high-pass here: one would make a held glance decay, and the
    // decay would look like a glance back.
    const samples = (ms) => Math.round((ms * rate) / 1000);
    this.heog = {
      chains: this.channels.map(() => [lowpass(10, rate)]),
      values: this.channels.map(() => new Float32Array(HEOG_RING)),
      times: new Float64Array(HEOG_RING),
      first: this.channels.map(() => null), // first and next-unseen sample index per channel
      ends: this.channels.map(() => 0),
      next: null,
      post: samples(50),
      preFrom: samples(300),
      preTo: samples(150),
      hold: samples(SACCADE_HOLD_MS),
      check: samples(SACCADE_CHECK_MS),
      scale: null,
      pending: null,
      lastEndIndex: -Infinity,
    };
    this.pop = this.channels.map(() => ({ last: null, scale: null, lastAt: -Infinity }));
    this.muscle = this.channels.map(() => ({
      chain: [highpass(30, rate), highpass(30, rate), notch(60, rate), notch(120, rate)],
      power: null,
      baseline: null,
      active: false,
      start: 0,
    }));
    this.accelMean = null;
    this.movementUntil = -Infinity;
  }

  // `samples` are raw µV values; `time` is when the packet arrived (performance.now());
  // `seq` is the packet's 16-bit sequence number, shared across channels.
  addEeg(ch, samples, time, seq) {
    this.startTime ??= time;
    const n = samples.length;
    const e = EARS.indexOf(ch);
    const firstIndex = this.#unwrap(ch, seq) * n;
    for (let k = 0; k < n; k++) {
      const t = time - ((n - 1 - k) * 1000) / this.rate;
      const warm = t - this.startTime > WARMUP_MS;
      this.#popSample(ch, samples[k], t, warm);
      this.#muscleSample(ch, samples[k], t, warm);
      if (e >= 0) this.#blinkSample(e, samples[k], t, firstIndex + k, warm);
      this.#heogSample(ch, samples[k], t, firstIndex + k);
    }
    this.#processSaccades();
  }

  #unwrap(ch, seq) {
    const s = this.sequence[ch];
    if (s.last !== null && seq < s.last - 32768) s.wraps++;
    s.last = seq;
    return s.wraps * 65536 + seq;
  }

  addGyro(samples, time) {
    for (const { x, y, z } of samples) {
      if (Math.hypot(x, y, z) > GYRO_DPS) this.#movement(time);
    }
  }

  addAccel(samples, time) {
    const a = smoothing(ACCEL_MEAN_TAU_S, ACCEL_RATE);
    for (const { x, y, z } of samples) {
      const g = Math.hypot(x, y, z);
      this.accelMean ??= g;
      if (Math.abs(g - this.accelMean) > ACCEL_CHANGE_G) this.#movement(time);
      this.accelMean += a * (g - this.accelMean);
    }
  }

  // Finished artifacts plus muscle bursts still in progress (which end at `now`).
  intervalsUntil(now) {
    const ongoing = this.channels
      .filter((ch) => this.muscle[ch].active)
      .map((ch) => ({
        type: 'muscle',
        channels: [ch],
        start: this.muscle[ch].start - MUSCLE_PAD_MS,
        end: now,
      }));
    return [...this.intervals, ...ongoing];
  }

  // Channels with an ongoing muscle burst, and whether the head is moving now.
  activeMuscle() {
    return this.channels.filter((ch) => this.muscle[ch].active);
  }

  moving(now) {
    return now < this.movementUntil;
  }

  blinksInLast(ms, now) {
    return this.blinkTimes.filter((t) => t > now - ms).length;
  }

  #muscleSample(ch, x, t, warm) {
    const m = this.muscle[ch];
    for (const filter of m.chain) x = filter.process(x);
    const a = smoothing(MUSCLE_POWER_TAU_S, this.rate);
    m.power = m.power === null ? x * x : m.power + a * (x * x - m.power);
    const db = 10 * Math.log10(m.power + 1e-9);
    m.baseline ??= db;
    // A pop's sudden jump is broadband too; don't call it muscle or learn from it.
    if (t - this.pop[ch].lastAt < POP_AFTER_MS) return;

    if (warm && !m.active && db - m.baseline > MUSCLE_ON_DB) {
      m.active = true;
      m.start = t;
    } else if (m.active && db - m.baseline < MUSCLE_OFF_DB) {
      m.active = false;
      this.#push({ type: 'muscle', channels: [ch], start: m.start - MUSCLE_PAD_MS, end: t });
    }
    if (!m.active) {
      // Track slowly once warm, so bursts don't raise the baseline.
      const tau = warm ? MUSCLE_BASELINE_TAU_S : WARMUP_TAU_S;
      m.baseline += smoothing(tau, this.rate) * (db - m.baseline);
    }
  }

  #blinkSample(f, x, t, index, warm) {
    const b = this.blink[f];
    for (const filter of b.chain) x = filter.process(x);
    const level = Math.abs(x);
    b.scale ??= level;
    const threshold = Math.max(BLINK_MIN_UV, BLINK_THRESHOLD * b.scale);

    // Durations and gaps use sample indices: packets often arrive in bursts, so arrival
    // times can squeeze a whole blink into a few milliseconds. Arrival times (`t`) are
    // only used to place the shading on screen.
    if (!b.above) {
      if (warm && level > threshold) {
        Object.assign(b, {
          above: true,
          start: t,
          startIndex: index,
          peak: level,
          peakTime: t,
          peakIndex: index,
          sign: Math.sign(x),
        });
      } else {
        const tau = warm ? BLINK_SCALE_TAU_S : WARMUP_TAU_S;
        b.scale += smoothing(tau, this.rate) * (level - b.scale);
      }
    } else if (level > b.peak) {
      Object.assign(b, { peak: level, peakTime: t, peakIndex: index, sign: Math.sign(x) });
    } else if (level < threshold / 2) {
      b.above = false;
      const duration = this.#ms(index - b.startIndex);
      const refractory = this.#ms(b.startIndex - this.lastBlinkEndIndex) < BLINK_REFRACTORY_MS;
      if (duration >= BLINK_MIN_MS && duration <= BLINK_MAX_MS && !refractory) {
        b.candidates.push({
          start: b.start,
          end: t,
          endIndex: index,
          peakTime: b.peakTime,
          peakIndex: b.peakIndex,
          sign: b.sign,
        });
        this.#pairBlink(f);
      }
    }
  }

  #ms(sampleCount) {
    return (sampleCount * 1000) / this.rate;
  }

  // A blink needs a same-sign candidate on the other ear channel at about the same time.
  #pairBlink(f) {
    const mine = this.blink[f].candidates;
    const theirs = this.blink[1 - f].candidates;
    const c = mine[mine.length - 1];
    const pairSamples = (BLINK_PAIR_MS * this.rate) / 1000;
    const other = theirs.find(
      (o) => o.sign === c.sign && Math.abs(o.peakIndex - c.peakIndex) < pairSamples,
    );
    // Forget candidates that are too old to pair with anything.
    for (const list of [mine, theirs]) {
      while (list.length && this.#ms(c.endIndex - list[0].endIndex) > 1000) list.shift();
    }
    if (!other) return;
    theirs.splice(theirs.indexOf(other), 1);
    mine.splice(mine.indexOf(c), 1);

    this.lastBlinkEndIndex = Math.max(c.endIndex, other.endIndex);
    const start = Math.min(c.start, other.start) - BLINK_PAD_BEFORE_MS;
    const end = Math.max(c.end, other.end) + BLINK_PAD_AFTER_MS;
    if (this.movementUntil > start) return; // likely a head movement, not a blink
    this.#push({ type: 'blink', channels: [...this.channels], start, end });
    this.blinkTimes.push(c.peakTime);
    this.onBlink();
  }

  #heogSample(ch, x, t, index) {
    const h = this.heog;
    for (const filter of h.chains[ch]) x = filter.process(x);
    h.values[ch][index % HEOG_RING] = x;
    h.times[index % HEOG_RING] = t;
    h.first[ch] ??= index;
    h.ends[ch] = Math.max(h.ends[ch], index + 1);
  }

  // Channels arrive in separate packets; step through sample indices all of them have.
  #processSaccades() {
    const h = this.heog;
    if (h.first.includes(null)) return;
    h.next ??= Math.max(...h.first) + h.preFrom;
    const ready = Math.min(...h.ends);
    h.next = Math.max(h.next, ready - HEOG_RING + h.preFrom); // skip ahead if far behind
    for (; h.next < ready; h.next++) this.#saccadeStep(h.next);
  }

  #saccadeStep(i) {
    const h = this.heog;
    // Mean over samples from..to, averaged over the channels on one side of the head.
    const mean = (side, from, to) => {
      let sum = 0;
      for (const ch of SIDES[side]) {
        for (let k = from; k <= to; k++) sum += h.values[ch][k % HEOG_RING];
      }
      return sum / ((to - from + 1) * SIDES[side].length);
    };
    const t = h.times[i % HEOG_RING];

    if (h.pending) {
      if (i === h.pending.i + h.hold) this.#confirmSaccade(i, t, mean);
      return;
    }

    const pre = SIDES.map((_, s) => mean(s, i - h.preFrom, i - h.preTo));
    const post = SIDES.map((_, s) => mean(s, i - h.post + 1, i));
    const step = post[0] - pre[0] - (post[1] - pre[1]);
    const level = Math.abs(step);
    h.scale ??= level;
    const threshold = Math.max(SACCADE_MIN_UV, SACCADE_THRESHOLD * h.scale);
    const warm = t - this.startTime > WARMUP_MS;

    if (level > threshold) {
      // Big steps never feed the scale, or each glance would raise the bar for the next.
      if (warm && this.#ms(i - h.lastEndIndex) > SACCADE_REFRACTORY_MS) {
        h.pending = { i, start: t, sign: Math.sign(step), pre, threshold };
      }
    } else {
      const tau = warm ? SACCADE_SCALE_TAU_S : WARMUP_TAU_S;
      h.scale += smoothing(tau, this.rate) * (level - h.scale);
    }
  }

  // At the end of the hold, the shift must still be there, opposite on the two sides,
  // with no blink or head movement around it.
  #confirmSaccade(i, t, mean) {
    const h = this.heog;
    const p = h.pending;
    h.pending = null;
    const shift = SIDES.map((_, s) => mean(s, i - h.check + 1, i) - p.pre[s]);
    const step = shift[0] - shift[1];
    const persists =
      Math.sign(step) === p.sign && Math.abs(step) > SACCADE_KEEP_FRACTION * p.threshold;
    // Both sides must move, in opposite directions and by comparable amounts; a shift on
    // one side alone (or one electrode) is a contact problem, not the eyes.
    const [small, large] = [Math.abs(shift[0]), Math.abs(shift[1])].sort((a, b) => a - b);
    const opposite =
      Math.sign(shift[0]) === -Math.sign(shift[1]) && small >= SACCADE_BALANCE * large;
    const blinked = this.#ms(p.i - this.lastBlinkEndIndex) < 300;
    const moved = this.movementUntil > p.start - MOVEMENT_PAD_MS;
    if (!persists || !opposite || blinked || moved) return;

    h.lastEndIndex = i;
    this.#push({ type: 'eye', channels: [...this.channels], start: p.start - 150, end: t + 100 });
    this.onEyeMove(step * LOOK_LEFT_SIGN > 0 ? 'left' : 'right');
  }

  #popSample(ch, x, t, warm) {
    const p = this.pop[ch];
    if (p.last !== null) {
      const jump = Math.abs(x - p.last);
      p.scale ??= jump;
      const threshold = Math.max(POP_MIN_UV, POP_THRESHOLD * p.scale);
      if (warm && jump > threshold && t - p.lastAt > POP_AFTER_MS) {
        p.lastAt = t;
        this.#push({ type: 'pop', channels: [ch], start: t - POP_BEFORE_MS, end: t + POP_AFTER_MS });
      } else {
        p.scale += smoothing(POP_SCALE_TAU_S, this.rate) * (jump - p.scale);
      }
    }
    p.last = x;
  }

  #movement(time) {
    const last = this.intervals.findLast((i) => i.type === 'movement');
    if (last && last.end >= time - MOVEMENT_PAD_MS) {
      last.end = time + MOVEMENT_PAD_MS;
    } else {
      this.#push({
        type: 'movement',
        channels: [...this.channels],
        start: time - MOVEMENT_PAD_MS,
        end: time + MOVEMENT_PAD_MS,
      });
    }
    this.movementUntil = time + MOVEMENT_PAD_MS;
  }

  #push(interval) {
    this.intervals.push(interval);
    const cutoff = interval.end - KEEP_MS;
    while (this.intervals.length && this.intervals[0].end < cutoff) this.intervals.shift();
    while (this.blinkTimes.length && this.blinkTimes[0] < cutoff) this.blinkTimes.shift();
  }
}
