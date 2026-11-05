import { Muse, EEG_CHANNELS, EEG_SAMPLE_RATE, EEG_SAMPLES_PER_PACKET } from './muse.js';
import { highpass, lowpass, notch } from './filters.js';

// Each plot traces one side of the head: its forehead channel on x, its ear channel on y.
const PLOTS = [
  { canvas: 'left-plot', x: 'AF7', y: 'TP9' },
  { canvas: 'right-plot', x: 'AF8', y: 'TP10' },
];
const HIGHPASS_HZ = 1;
const MAINS_HZ = 60;
const DISPLAY_LOWPASS_HZ = 15; // smooths the traces only; the sound gets the full band
const SOUND_TOP_HZ = 45; // top of the EEG band the spectral map spreads over the audio range
const RING = 4 * EEG_SAMPLE_RATE; // filtered samples kept per channel
const FADE_STEPS = 16; // the trail is drawn in this many pieces, older ones fainter

const connectButton = document.getElementById('connect');
const scaleSelect = document.getElementById('scale');
const trailSelect = document.getElementById('trail');

const plots = PLOTS.map((p) => {
  const canvas = document.getElementById(p.canvas);
  return {
    ...p,
    canvas,
    ctx: canvas.getContext('2d'),
    xi: EEG_CHANNELS.indexOf(p.x),
    yi: EEG_CHANNELS.indexOf(p.y),
  };
});

// Samples are stored by absolute sample index (packet sequence number x 12 + position),
// so x and y come from the same instant even though each channel arrives in its own
// packets, sometimes late or in bursts.
let channels = [];

// Every channel gets the high-pass and mains notch; `values` (fed to the sound) stop
// there. The traces then also pass a low-pass, two stages in series (24 dB per octave),
// into `display`, to strip high-frequency noise such as muscle.
function resetChannels() {
  const rate = EEG_SAMPLE_RATE;
  channels = EEG_CHANNELS.map(() => ({
    filters: [highpass(HIGHPASS_HZ, rate), notch(MAINS_HZ, rate)],
    smoothing: [lowpass(DISPLAY_LOWPASS_HZ, rate), lowpass(DISPLAY_LOWPASS_HZ, rate)],
    values: new Float32Array(RING),
    display: new Float32Array(RING),
    first: null, // index of the first sample received
    end: 0, // one past the newest sample index
    lastSeq: null,
    wraps: 0,
  }));
  sentEnd = null;
}

function addSamples(ch, seq, samples) {
  const c = channels[ch];
  if (c.lastSeq !== null && seq < c.lastSeq - 32768) c.wraps++; // 16-bit counter wrapped
  c.lastSeq = seq;
  const firstIndex = (c.wraps * 65536 + seq) * EEG_SAMPLES_PER_PACKET;
  samples.forEach((s, k) => {
    const i = (firstIndex + k) % RING;
    for (const filter of c.filters) s = filter.process(s);
    c.values[i] = s;
    for (const filter of c.smoothing) s = filter.process(s);
    c.display[i] = s;
  });
  c.first ??= firstIndex;
  c.end = Math.max(c.end, firstIndex + samples.length);
  feedAudio();
}

// Sound: each ear gets one side's trace, the same x (forehead) and y (ear) the plot draws,
// but without the plot's low-pass, so the sound keeps the high frequencies.
const SIDES = [
  { x: EEG_CHANNELS.indexOf('AF7'), y: EEG_CHANNELS.indexOf('TP9') },
  { x: EEG_CHANNELS.indexOf('AF8'), y: EEG_CHANNELS.indexOf('TP10') },
];
const soundSelect = document.getElementById('sound');
const speedSelect = document.getElementById('speed');
const crushCheckbox = document.getElementById('crush');
const rangeSelect = document.getElementById('range');
const volumeInput = document.getElementById('volume');
const reverbInput = document.getElementById('reverb');
let audio = null; // { context, node, gain } once sound has been switched on
let sentEnd = null; // one past the newest sample index sent to the audio thread

resetChannels();

// Post every sample index that all four channels now have.
function feedAudio() {
  if (channels.some((c) => c.first === null)) return;
  const end = Math.min(...channels.map((c) => c.end));
  const oldest = Math.max(...channels.map((c) => c.first), end - RING + 1);
  sentEnd = Math.max(sentEnd ?? oldest, oldest);
  if (!audio) {
    sentEnd = end; // nothing to feed yet; start from fresh data when sound comes on
    return;
  }
  if (end <= sentEnd) return;
  const sides = SIDES.map(({ x, y }) => {
    const side = { x: new Float32Array(end - sentEnd), y: new Float32Array(end - sentEnd) };
    for (let i = sentEnd; i < end; i++) {
      side.x[i - sentEnd] = channels[x].values[i % RING];
      side.y[i - sentEnd] = channels[y].values[i % RING];
    }
    return side;
  });
  const buffers = sides.flatMap(({ x, y }) => [x.buffer, y.buffer]);
  audio.node.port.postMessage({ type: 'eeg', sides }, buffers);
  sentEnd = end;
}

async function startAudio() {
  const context = new AudioContext();
  await context.audioWorklet.addModule('sonify-worklet.js');
  const node = new AudioWorkletNode(context, 'sonifier', {
    numberOfInputs: 0,
    outputChannelCount: [2],
  });
  // Dry and reverberated paths are mixed into the volume control.
  const dry = context.createGain();
  const wet = context.createGain();
  const reverb = context.createConvolver();
  reverb.buffer = reverbImpulse(context);
  const gain = context.createGain();
  node.connect(dry).connect(gain);
  node.connect(reverb).connect(wet).connect(gain);
  gain.connect(context.destination);
  audio = { context, node, gain, dry, wet };
}

// A generated room: decaying noise, falling 60 dB over `seconds`, independent for each
// ear so each side's reverb stays on its own side.
function reverbImpulse(context, seconds = 2.5) {
  const length = Math.round(seconds * context.sampleRate);
  const impulse = context.createBuffer(2, length, context.sampleRate);
  for (let ch = 0; ch < 2; ch++) {
    const data = impulse.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.exp((-6.9 * i) / length);
    }
  }
  return impulse;
}

function sendSoundSettings() {
  const mode = soundSelect.value;
  document.getElementById('speed-control').hidden = mode !== 'granular';
  document.getElementById('crush-control').hidden = mode !== 'granular';
  document.getElementById('range-control').hidden = mode !== 'spectral';
  if (!audio) return;
  // The spectral map spreads the EEG band from the high-pass up to SOUND_TOP_HZ over the
  // chosen audio range.
  const map = {
    eegLow: HIGHPASS_HZ,
    eegHigh: SOUND_TOP_HZ,
    audioLow: 55,
    audioHigh: Number(rangeSelect.value),
  };
  audio.node.port.postMessage({
    type: 'settings',
    settings: { mode, speed: Number(speedSelect.value), crush: crushCheckbox.checked, map },
  });
  // Volume on a squared curve, which sounds more even across the slider.
  const now = audio.context.currentTime;
  audio.gain.gain.setTargetAtTime(Number(volumeInput.value) ** 2, now, 0.05);
  const reverb = Number(reverbInput.value);
  audio.wet.gain.setTargetAtTime(reverb, now, 0.05);
  audio.dry.gain.setTargetAtTime(1 - 0.5 * reverb, now, 0.05);
}

soundSelect.addEventListener('change', async () => {
  // The browser only allows audio to start from a user action, such as this change.
  if (soundSelect.value !== 'off' && !audio) await startAudio();
  sendSoundSettings();
  if (!audio) return;
  if (soundSelect.value === 'off') audio.context.suspend();
  else audio.context.resume();
});
for (const control of [speedSelect, rangeSelect, volumeInput, reverbInput]) {
  control.addEventListener('input', sendSoundSettings);
}
crushCheckbox.addEventListener('change', sendSoundSettings);

let muse = null;

if (!navigator.bluetooth) {
  document.getElementById('unsupported').hidden = false;
  connectButton.disabled = true;
}

// The button is the only status shown: its label and whether it can be clicked.
function setButton(label, { busy = false } = {}) {
  connectButton.textContent = label;
  connectButton.disabled = busy;
}

connectButton.addEventListener('click', async () => {
  if (muse) {
    setButton('Disconnecting…', { busy: true });
    const closing = muse;
    await closing.disconnect();
    // If the link had already dropped, no disconnect event comes; settle the button here.
    if (muse === closing && !closing.device?.gatt.connected) closing.onDisconnect();
    return;
  }
  resetChannels();
  resetPca();
  resetHead();
  const current = new Muse();
  muse = current;
  muse.onEeg = addSamples;
  muse.onGyro = addGyro;
  muse.onDisconnect = () => {
    if (muse !== current) return; // a late event from an earlier connection
    muse = null;
    setButton('Connect');
  };
  try {
    setButton('Connecting…', { busy: true });
    await muse.connect();
    setButton('Disconnect');
  } catch (err) {
    console.error(err);
    muse = null;
    setButton('Connect');
  }
});

// Views:
// - 2D: each side's forehead channel (x) against its ear channel (y).
// - 3D x, y, x (or y) delayed: each side's x, y and x (or y) a moment earlier, which opens a mixture of
//   rhythms out into a 3D shape; a single pure rhythm stays a flat loop.
// - 3D Takens embeddings: one signal per side (x - y, or (x + y) / 2)
//   against itself tau and 2 tau earlier.
// - 3D PCA: all four channels projected onto their three main directions of variation
//   (principal components), one trace for the whole head.
// The delay views take their delay from the Delay setting: a quarter period of a band's
// typical frequency (which turns that rhythm into a circle), or automatic (see autoDelay).
// Overlay (every view but PCA) puts both hemispheres in one plot, drawn additively, so where
// their fading trails cross the light adds up. Neither is mirrored: activity common to
// both sides moves both traces the same way, fusing the figures.
const viewSelect = document.getElementById('view');
const delaySelect = document.getElementById('delay');
const overlayCheckbox = document.getElementById('overlay');
const overlayCanvas = document.getElementById('overlay-plot');
const overlay = { canvas: overlayCanvas, ctx: overlayCanvas.getContext('2d') };
const DELAY_VIEWS = new Set(['delay', 'delayY', 'takens', 'takensSum']);
const DELAY_SAMPLES = EEG_SAMPLE_RATE / 10 / 4; // quarter of an alpha period, the starting guess

// The delay in samples for the Delay setting: a band's frequency in Hz, or 'auto'.
function bandDelay(hz) {
  return EEG_SAMPLE_RATE / Number(hz) / 4;
}

function updateLayout() {
  const view = viewSelect.value;
  const wholeHead = view === 'pca';
  const single = wholeHead || overlayCheckbox.checked;
  for (const figure of document.querySelectorAll('.xy .split')) figure.hidden = single;
  document.getElementById('overlay-figure').hidden = !single;
  document.getElementById('overlay-control').hidden = wholeHead; // always one plot there
  document.getElementById('motion-control').hidden = view === '2d';
  document.getElementById('delay-control').hidden = !DELAY_VIEWS.has(view);
  plotArea.classList.toggle('draggable', view !== '2d' && motionSelect.value === 'mouse');
}
viewSelect.addEventListener('change', updateLayout);
overlayCheckbox.addEventListener('change', updateLayout);

// Keep each canvas's pixel buffer matched to its on-screen size.
for (const target of [...plots, overlay]) {
  new ResizeObserver(() => {
    const dpr = window.devicePixelRatio || 1;
    target.canvas.width = target.canvas.clientWidth * dpr;
    target.canvas.height = target.canvas.clientHeight * dpr;
    target.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }).observe(target.canvas);
}

// 3D views are seen in perspective, nearer parts brighter, turned around the vertical axis
// (yaw) and tilted towards the viewer (pitch). Coordinates are in plot radii, each axis
// clamped to -1..1. Motion sets the turn and tilt:
// - Auto-rotate: one slow turn every SPIN_S seconds, at a fixed tilt.
// - Mouse: drag on the plot; the figure stays where it's left.
// - Head: the headband's gyroscope, integrated into angles and amplified, so turning the
//   head turns the figure and nodding tilts it.
const SPIN_S = 25; // seconds per turn
const TILT = 0.35; // radians
const CAMERA = 3.2; // viewing distance, in plot radii
const FIT_3D = 0.62; // keeps a turning cube's corners inside the plot
const DRAG_RADIANS_PER_PX = 0.01;
const MAX_PITCH = Math.PI / 2;

const motionSelect = document.getElementById('motion');
const plotArea = document.querySelector('.xy');
const dragged = { yaw: 0, pitch: TILT };
motionSelect.addEventListener('change', () => {
  // Head angles build up from connecting onwards; start from front-on whenever Head is
  // chosen, so earlier movement (putting the headband on, looking around) doesn't show.
  if (motionSelect.value === 'head') Object.assign(head, { yaw: 0, pitch: 0 });
  updateLayout();
});

let drag = null; // { x, y } of the last pointer position while dragging
plotArea.addEventListener('pointerdown', (e) => {
  if (!plotArea.classList.contains('draggable')) return;
  drag = { x: e.clientX, y: e.clientY };
  plotArea.setPointerCapture(e.pointerId);
  plotArea.classList.add('dragging');
});
plotArea.addEventListener('pointermove', (e) => {
  if (!drag) return;
  dragged.yaw += (e.clientX - drag.x) * DRAG_RADIANS_PER_PX;
  dragged.pitch += (e.clientY - drag.y) * DRAG_RADIANS_PER_PX;
  dragged.pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, dragged.pitch));
  drag = { x: e.clientX, y: e.clientY };
});
for (const type of ['pointerup', 'pointercancel']) {
  plotArea.addEventListener(type, () => {
    drag = null;
    plotArea.classList.remove('dragging');
  });
}

// Head motion. The gyroscope reports rotation rates (degrees per second), so they're
// integrated into angles. An uncorrected gyroscope drifts, so its resting offset is tracked
// while the head is still and subtracted. Which axis is which is assumed (z for turning,
// y for nodding); swap GYRO_AXES or flip a GYRO_SIGNS entry if the figure moves the wrong way.
const GYRO_RATE = 52; // samples per second
const GYRO_AXES = { yaw: 'z', pitch: 'y' };
const GYRO_SIGNS = { yaw: 1, pitch: 1 };
const GYRO_GAIN = 3; // a 15 degree head turn turns the figure 45 degrees
const GYRO_STILL_DPS = 3; // slower than this counts as still, for tracking the offset
const GYRO_OFFSET_TAU_S = 5;
const head = { yaw: 0, pitch: 0, offset: { x: 0, y: 0, z: 0 }, stillCount: 0 };

function resetHead() {
  Object.assign(head, { yaw: 0, pitch: 0, offset: { x: 0, y: 0, z: 0 }, stillCount: 0 });
}

function addGyro(samples) {
  const slow = 1 - Math.exp(-1 / (GYRO_OFFSET_TAU_S * GYRO_RATE));
  const toRadians = Math.PI / 180;
  for (const rate of samples) {
    // Still means slow relative to the offset learned so far.
    const moving = Math.hypot(rate.x - head.offset.x, rate.y - head.offset.y, rate.z - head.offset.z);
    if (moving < GYRO_STILL_DPS) {
      // A plain average of the first still readings, so the offset is known almost at once.
      const follow = Math.max(1 / ++head.stillCount, slow);
      for (const axis of ['x', 'y', 'z']) head.offset[axis] += follow * (rate[axis] - head.offset[axis]);
    }
    for (const angle of ['yaw', 'pitch']) {
      const axis = GYRO_AXES[angle];
      head[angle] += (GYRO_SIGNS[angle] * (rate[axis] - head.offset[axis]) * toRadians) / GYRO_RATE;
    }
  }
}

// The turn and tilt for this frame.
function orientation(now) {
  if (motionSelect.value === 'mouse') return [dragged.yaw, dragged.pitch];
  if (motionSelect.value === 'head') {
    const pitch = TILT + GYRO_GAIN * head.pitch;
    return [GYRO_GAIN * head.yaw, Math.max(-MAX_PITCH, Math.min(MAX_PITCH, pitch))];
  }
  return [(2 * Math.PI * now) / 1000 / SPIN_S, TILT];
}

function flat([x, y]) {
  return [x, y, 1];
}

function perspective([yaw, pitch]) {
  const [ca, sa, ct, st] = [Math.cos(yaw), Math.sin(yaw), Math.cos(pitch), Math.sin(pitch)];
  return ([x, y, z]) => {
    const x1 = x * ca + z * sa;
    const z1 = -x * sa + z * ca;
    const y2 = y * ct - z1 * st;
    const z2 = y * st + z1 * ct; // towards the viewer
    const scale = (FIT_3D * CAMERA) / (CAMERA - z2);
    const nearness = Math.max(0, Math.min(1, (z2 + 1.7) / 3.4));
    return [x1 * scale, y2 * scale, 0.35 + 0.65 * nearness];
  };
}

// Draws one trace into `target`: point(i) gives sample i's coordinates, project maps them
// to the plane plus a brightness. Oldest pieces first and faintest.
function drawTrace(target, point, start, end, project) {
  const { ctx } = target;
  const size = target.canvas.clientWidth;
  const mid = size / 2;
  const radius = mid - 12;
  const at = (i) => {
    const [x, y, brightness] = project(point(i));
    return [mid + x * radius, mid - y * radius, brightness];
  };

  ctx.strokeStyle = '#fff';
  ctx.fillStyle = '#fff';
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  const step = Math.max(1, Math.ceil((end - start) / FADE_STEPS));
  for (let from = start; from < end - 1; from += step) {
    const to = Math.min(end - 1, from + step);
    const age = 0.08 + 0.92 * ((to - start) / (end - 1 - start));
    const [x0, y0, b0] = at(from);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    let b = b0;
    for (let i = from + 1; i <= to; i++) {
      const [x, y, bi] = at(i);
      ctx.lineTo(x, y);
      b = bi;
    }
    ctx.globalAlpha = age * (b0 + b) / 2;
    ctx.stroke();
  }

  // The newest point.
  const [hx, hy, hb] = at(end - 1);
  ctx.globalAlpha = hb;
  ctx.beginPath();
  ctx.arc(hx, hy, 3.5, 0, 2 * Math.PI);
  ctx.fill();
  ctx.globalAlpha = 1;
}

// Automatic delay for the delay views: the first lag at which the signal's autocorrelation
// reaches zero, i.e. where s(t - tau) has become as unrelated to s(t) as a lag can make it
// (a quarter period for a pure rhythm). Re-measured twice a second over the last 2 s and
// glided towards, so the shape morphs instead of jumping. Kept per side and per view.
const AUTO_WINDOW = 2 * EEG_SAMPLE_RATE;
const AUTO_EVERY_MS = 500;
const AUTO_GLIDE_S = 2;
const AUTO_MIN = 2;
const AUTO_MAX = 64; // 250 ms
const autoDelays = new Map(); // `${canvas}:${view}` -> { tau, target, measuredAt, frameAt }
function autoDelay(key, signal, end, first, now) {
  let d = autoDelays.get(key);
  if (!d) autoDelays.set(key, (d = { tau: DELAY_SAMPLES, target: DELAY_SAMPLES, measuredAt: 0, frameAt: now }));
  if (now - d.measuredAt >= AUTO_EVERY_MS && end - first >= AUTO_WINDOW) {
    d.measuredAt = now;
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
    // First lag where r drops to zero, interpolated between the samples either side.
    let lag = 1;
    let before = r(0);
    let after = r(1);
    while (lag < AUTO_MAX && after > 0) {
      before = after;
      after = r(++lag);
    }
    const crossing = after > 0 || before <= 0 ? lag : lag - after / (after - before);
    d.target = Math.max(AUTO_MIN, Math.min(AUTO_MAX, crossing));
  }
  d.tau += (d.target - d.tau) * (1 - Math.exp(-(now - d.frameAt) / 1000 / AUTO_GLIDE_S));
  d.frameAt = now;
  return d.tau;
}

// A ring-buffer read at a fractional sample index, by linear interpolation.
// Indices before the first sample (rounding can land a hair below it) read as 0.
function readAt(ring, t) {
  const i = Math.floor(t);
  const f = t - i;
  const at = (k) => (k < 0 ? 0 : ring[k % RING]);
  return at(i) * (1 - f) + at(i + 1) * f;
}

// One side's trace in any per-side view:
// - 2d: (x, y)
// - delay: (x, y, x tau earlier)
// - delayY: (x, y, y tau earlier)
// - takens: (s, s tau earlier, s 2 tau earlier) for the side's single bipolar signal
//   s = (x - y) / sqrt 2, a proper delay embedding of one measurement
// - takensSum: the same for the side's average, s = (x + y) / 2, the activity the forehead
//   and ear channels share
// tau comes from the Delay setting; when automatic, it's measured on the embedded signal.
function drawSide(target, p, view, scale, trail, project, now) {
  const cx = channels[p.xi];
  const cy = channels[p.yi];
  if (cx.first === null || cy.first === null) return;
  const end = Math.min(cx.end, cy.end);
  const first = Math.max(cx.first, cy.first);
  const norm = (v) => Math.max(-1, Math.min(1, v / scale));
  const x = (t) => readAt(cx.display, t);
  const y = (t) => readAt(cy.display, t);
  // signal(i) is the embedded signal at whole sample i, for measuring an automatic delay.
  const delay = (signal) =>
    delaySelect.value === 'auto'
      ? autoDelay(`${p.canvas.id}:${view}`, signal, end, first, now)
      : bandDelay(delaySelect.value);

  let lookback = 0;
  let point;
  if (view === 'delay') {
    const tau = delay((i) => cx.display[i % RING]);
    lookback = Math.ceil(tau);
    point = (i) => [norm(x(i)), norm(y(i)), norm(x(i - tau))];
  } else if (view === 'delayY') {
    const tau = delay((i) => cy.display[i % RING]);
    lookback = Math.ceil(tau);
    point = (i) => [norm(x(i)), norm(y(i)), norm(y(i - tau))];
  } else if (view === 'takens') {
    const s = (t) => (x(t) - y(t)) / Math.SQRT2;
    const tau = delay((i) => (cx.display[i % RING] - cy.display[i % RING]) / Math.SQRT2);
    lookback = Math.ceil(2 * tau);
    point = (i) => [norm(s(i)), norm(s(i - tau)), norm(s(i - 2 * tau))];
  } else if (view === 'takensSum') {
    const s = (t) => (x(t) + y(t)) / 2;
    const tau = delay((i) => (cx.display[i % RING] + cy.display[i % RING]) / 2);
    lookback = Math.ceil(2 * tau);
    point = (i) => [norm(s(i)), norm(s(i - tau)), norm(s(i - 2 * tau))];
  } else {
    point = (i) => [norm(x(i)), norm(y(i)), 0];
  }

  const start = Math.max(end - trail, first + lookback);
  if (end - start < 2) return;
  drawTrace(target, point, start, end, project);
}

// Whole head: a running covariance of the four smoothed channels (about 20 s), whose top
// three eigenvectors are the target axes. Targets are recomputed often and matched to the
// shown axes in order and sign; the shown axes then glide towards them every frame (and
// are kept perpendicular), so the figure turns gently instead of jumping when the
// directions shift, as they do after a blink or when two carry similar variance.
const PCA_TAU_S = 20;
const PCA_EVERY_MS = 200;
const PCA_GLIDE_S = 2;
const pca = {};

function resetPca() {
  Object.assign(pca, {
    covariance: new Float64Array(16),
    count: 0,
    end: null,
    target: null,
    axes: null, // the axes drawn, gliding towards the target
    solvedAt: 0,
    glidedAt: null,
  });
}
resetPca();

function updatePca(now) {
  if (channels.some((c) => c.first === null)) return;
  const end = Math.min(...channels.map((c) => c.end));
  pca.end ??= Math.max(...channels.map((c) => c.first));
  pca.end = Math.max(pca.end, end - RING + 1);
  const slow = 1 - Math.exp(-1 / (PCA_TAU_S * EEG_SAMPLE_RATE));
  for (; pca.end < end; pca.end++) {
    const v = channels.map((c) => c.display[pca.end % RING]);
    const w = Math.max(1 / ++pca.count, slow);
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) pca.covariance[r * 4 + c] += w * (v[r] * v[c] - pca.covariance[r * 4 + c]);
    }
  }
  if (pca.count < EEG_SAMPLE_RATE) return;
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

  if (now - pca.solvedAt >= PCA_EVERY_MS) {
    pca.solvedAt = now;
    const { values, vectors } = eigenSymmetric(pca.covariance, 4);
    const order = [0, 1, 2, 3].sort((a, b) => values[b] - values[a]);
    const fresh = order.slice(0, 3).map((k) => vectors[k]);
    if (!pca.axes) {
      pca.axes = fresh;
      pca.target = fresh;
    } else {
      // Match each shown axis to the most similar new one, flipping its sign if needed.
      const unused = new Set(fresh);
      pca.target = pca.axes.map((shown) => {
        let best = null;
        for (const v of unused) if (!best || Math.abs(dot(shown, v)) > Math.abs(dot(shown, best))) best = v;
        unused.delete(best);
        return dot(shown, best) < 0 ? best.map((x) => -x) : best;
      });
    }
  }

  // Glide the shown axes towards the target, then straighten them (Gram-Schmidt) so they
  // stay perpendicular unit vectors.
  const dt = pca.glidedAt === null ? 0 : (now - pca.glidedAt) / 1000;
  pca.glidedAt = now;
  const w = 1 - Math.exp(-dt / PCA_GLIDE_S);
  const previous = pca.axes;
  const moved = previous.map((axis, k) => axis.map((x, c) => x + w * (pca.target[k][c] - x)));
  pca.axes = [];
  for (const [k, v] of moved.entries()) {
    let u = v;
    for (const done of pca.axes) {
      const d = dot(u, done);
      u = u.map((x, c) => x - d * done[c]);
    }
    const length = Math.hypot(...u);
    pca.axes.push(length > 1e-6 ? u.map((x) => x / length) : previous[k]); // degenerate: hold
  }
}

function drawWholeHead(scale, trail, project) {
  if (!pca.axes) return;
  const end = Math.min(...channels.map((c) => c.end));
  const start = Math.max(end - trail, ...channels.map((c) => c.first));
  if (end - start < 2) return;
  const norm = (v) => Math.max(-1, Math.min(1, v / scale));
  const point = (i) => {
    const v = channels.map((c) => c.display[i % RING]);
    return pca.axes.map((axis) => norm(axis.reduce((s, a, c) => s + a * v[c], 0)));
  };
  drawTrace(overlay, point, start, end, project);
}

// Eigen-decomposition of a symmetric n x n matrix (row-major) by cyclic Jacobi rotations.
// Returns the eigenvalues and the matching unit eigenvectors.
function eigenSymmetric(matrix, n) {
  const a = Float64Array.from(matrix);
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
    if (off < 1e-18) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-30) continue;
        const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k * n + p];
          const akq = a[k * n + q];
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p * n + k];
          const aqk = a[q * n + k];
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k * n + p];
          const vkq = v[k * n + q];
          v[k * n + p] = c * vkp - s * vkq;
          v[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const values = Array.from({ length: n }, (_, i) => a[i * n + i]);
  const vectors = Array.from({ length: n }, (_, k) => Array.from({ length: n }, (_, r) => v[r * n + k]));
  return { values, vectors };
}

function clear(target) {
  target.ctx.clearRect(0, 0, target.canvas.clientWidth, target.canvas.clientHeight);
}

function draw() {
  const now = performance.now();
  const scale = Number(scaleSelect.value);
  const trail = Math.round(Number(trailSelect.value) * EEG_SAMPLE_RATE);
  const view = viewSelect.value;
  const project = view === '2d' ? flat : perspective(orientation(now));
  updatePca(now);
  if (view === 'pca') {
    clear(overlay);
    drawWholeHead(scale, trail, project);
  } else if (overlayCheckbox.checked) {
    clear(overlay);
    overlay.ctx.globalCompositeOperation = 'lighter'; // overlapping light adds up
    for (const p of plots) drawSide(overlay, p, view, scale, trail, project, now);
    overlay.ctx.globalCompositeOperation = 'source-over';
  } else {
    for (const p of plots) {
      clear(p);
      drawSide(p, p, view, scale, trail, project, now);
    }
  }
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
