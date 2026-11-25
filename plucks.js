// Plucks: imaginary marimba bars in a shell around the plot's centre, struck when the trace
// crosses into one on its way out. The shell is a sphere divided into chunks: ten
// around (longitude about the z axis, the note: two octaves of A minor pentatonic rising
// anticlockwise from the right, A3 to G5) by three bands of latitude (the octave: lower
// band one down, upper band one up). A flat (2D) trace stays on the equator, so there only
// the middle band's ring can be struck. As with a mallet, how fast the trace is moving
// outwards as it crosses sets how hard.
//
// Runs on the page, on the points as the plot draws them (plot radii, each axis -1..1),
// and posts each strike to the audio thread.

// The shell's size is a performer's control (the Sphere size slider): its inner surface,
// in plot radii, which the trace strikes by crossing outwards. Smaller means more hits.
export const DEFAULT_RADIUS = 0.5;
export const THICKNESS = 0.12; // the outer surface, for drawing, is this much further out
const REST = 0.6; // re-arms once the trace comes back inside this fraction of the radius
const MIN_GAP = 26; // samples between strikes (about 0.1 s)
const RATE = 256; // samples per second
const SPEED_SPAN = 3; // outward speed is measured over this many samples (about 12 ms)
// Outward speed (plot radii per second) to velocity: 0.25 for the slowest drift up towards
// 1, with SPEED_REF the speed at which it reaches about 0.72. A 10 Hz swing that would peak
// at 0.8 crosses at about 39 radii/s (0.96); a 3 Hz swing peaking at 0.6, about 6 (0.55).
const SPEED_REF = 12;
export const SECTORS = 10;
export const BANDS = [
  { from: -Math.PI / 2, to: -Math.PI / 6, octave: -1 },
  { from: -Math.PI / 6, to: Math.PI / 6, octave: 0 },
  { from: Math.PI / 6, to: Math.PI / 2, octave: 1 },
];
export const EQUATOR = 1;
export const GLOW_S = 0.35; // a struck chunk's light fades with this time constant

const SCALE = [0, 3, 5, 7, 10];
const TAU = 2 * Math.PI;

export function chunkIndex(sector, band) {
  return band * SECTORS + sector;
}

export class PluckDetector {
  constructor() {
    this.next = null; // the next sample index to examine; null until started
    this.glow = Array.from({ length: SECTORS * BANDS.length }, () => ({ at: -Infinity, velocity: 0 }));
    this.reset();
  }

  reset() {
    this.next = null;
    this.armed = true;
    this.previous = null; // the last sample's point and radius
    this.radii = []; // the last SPEED_SPAN + 1 radii, oldest first
    this.lastStrike = -Infinity;
  }

  // Set every frame from the slider.
  radius = DEFAULT_RADIUS;

  // Examines sample i at point [x, y, z]; returns a strike or null.
  process(i, point) {
    const r = Math.hypot(...point);
    this.radii.push(r);
    if (this.radii.length > SPEED_SPAN + 1) this.radii.shift();
    const previous = this.previous;
    this.previous = { point, r };
    if (!previous) return null;

    let strike = null;
    const inner = this.radius;
    if (this.armed && previous.r < inner && r >= inner) {
      // Crossed the shell outwards. Strike where it crossed, between the two samples.
      const f = (inner - previous.r) / (r - previous.r);
      const at = point.map((v, k) => previous.point[k] + f * (v - previous.point[k]));
      const speed = ((r - this.radii[0]) * RATE) / (this.radii.length - 1);
      if (i - this.lastStrike >= MIN_GAP) strike = this.#strike(i, at, speed);
      this.armed = false;
    } else if (!this.armed && r < REST * inner) {
      this.armed = true;
    }
    return strike;
  }

  #strike(i, [x, y, z], speed) {
    const turn = ((Math.atan2(y, x) / TAU) % 1 + 1) % 1;
    const sector = Math.min(SECTORS - 1, Math.floor(turn * SECTORS));
    const latitude = Math.atan2(z, Math.hypot(x, y));
    let band = BANDS.findIndex((b) => latitude < b.to);
    if (band < 0) band = BANDS.length - 1;
    const octave = Math.floor(sector / SCALE.length) + BANDS[band].octave;
    const semitones = 12 * octave + SCALE[sector % SCALE.length];
    this.lastStrike = i;
    return {
      sector,
      band,
      freq: 220 * 2 ** (semitones / 12),
      velocity: 0.25 + 0.75 * (1 - Math.exp(-Math.max(0, speed) / SPEED_REF)),
    };
  }

  light(strike, now) {
    this.glow[chunkIndex(strike.sector, strike.band)] = { at: now, velocity: strike.velocity };
  }

  // How brightly each chunk is lit at time `now` (ms), 0..1.
  glowAt(index, now) {
    const g = this.glow[index];
    return g.velocity * Math.exp(-(now - g.at) / 1000 / GLOW_S);
  }
}
