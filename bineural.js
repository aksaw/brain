import { Muse, EEG_CHANNELS, EEG_SAMPLE_RATE, EEG_SAMPLES_PER_PACKET } from './muse.js';
import { highpass, lowpass, notch } from './filters.js';

// Each plot traces one side of the head: its forehead channel on x, its ear channel on y.
const PLOTS = [
  { canvas: 'left-plot', x: 'AF7', y: 'TP9', color: '#fff' },
  { canvas: 'right-plot', x: 'AF8', y: 'TP10', color: '#fff' },
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
  channels = EEG_CHANNELS.map(() => ({
    filters: [highpass(HIGHPASS_HZ, EEG_SAMPLE_RATE), notch(MAINS_HZ, EEG_SAMPLE_RATE)],
    smoothing: [
      lowpass(DISPLAY_LOWPASS_HZ, EEG_SAMPLE_RATE),
      lowpass(DISPLAY_LOWPASS_HZ, EEG_SAMPLE_RATE),
    ],
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
    for (const filter of c.filters) s = filter.process(s);
    c.values[(firstIndex + k) % RING] = s;
    for (const filter of c.smoothing) s = filter.process(s);
    c.display[(firstIndex + k) % RING] = s;
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
  const current = new Muse();
  muse = current;
  muse.onEeg = addSamples;
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

// Keep each canvas's pixel buffer matched to its on-screen size.
for (const p of plots) {
  new ResizeObserver(() => {
    const dpr = window.devicePixelRatio || 1;
    p.canvas.width = p.canvas.clientWidth * dpr;
    p.canvas.height = p.canvas.clientHeight * dpr;
    p.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }).observe(p.canvas);
}

function drawPlot(p, scale, trail) {
  const { ctx } = p;
  const size = p.canvas.clientWidth;
  const mid = size / 2;
  const radius = mid - 12;
  // µV to pixels, clamped to the edge so large swings stay visible.
  const toPx = (v) => (Math.max(-scale, Math.min(scale, v)) / scale) * radius;

  ctx.clearRect(0, 0, size, size);

  const cx = channels[p.xi];
  const cy = channels[p.yi];
  if (cx.first === null || cy.first === null) return;
  const end = Math.min(cx.end, cy.end);
  const start = Math.max(end - trail, cx.first, cy.first);
  if (end - start < 2) return;

  const point = (i) => [
    mid + toPx(cx.display[i % RING]),
    mid - toPx(cy.display[i % RING]),
  ];

  // Oldest pieces first and faintest; each piece starts where the last one ended.
  ctx.strokeStyle = p.color;
  ctx.lineWidth = 1.5;
  ctx.lineJoin = 'round';
  const step = Math.max(1, Math.ceil((end - start) / FADE_STEPS));
  for (let from = start; from < end - 1; from += step) {
    const to = Math.min(end - 1, from + step);
    ctx.globalAlpha = 0.08 + 0.92 * ((to - start) / (end - 1 - start));
    ctx.beginPath();
    ctx.moveTo(...point(from));
    for (let i = from + 1; i <= to; i++) ctx.lineTo(...point(i));
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  // The newest point.
  const [hx, hy] = point(end - 1);
  ctx.fillStyle = p.color;
  ctx.beginPath();
  ctx.arc(hx, hy, 3.5, 0, 2 * Math.PI);
  ctx.fill();
}

function draw() {
  const scale = Number(scaleSelect.value);
  const trail = Math.round(Number(trailSelect.value) * EEG_SAMPLE_RATE);
  for (const p of plots) drawPlot(p, scale, trail);
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
