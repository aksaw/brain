import { Muse, EEG_CHANNELS, EEG_SAMPLE_RATE, EEG_SAMPLES_PER_PACKET } from './muse.js';
import { highpass, notch } from './filters.js';

// Each plot traces one side of the head: its forehead channel on x, its ear channel on y.
const PLOTS = [
  { canvas: 'left-plot', x: 'AF7', y: 'TP9', color: '#fff' },
  { canvas: 'right-plot', x: 'AF8', y: 'TP10', color: '#fff' },
];
const HIGHPASS_HZ = 1;
const MAINS_HZ = 60;
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
function resetChannels() {
  channels = EEG_CHANNELS.map(() => ({
    filters: [highpass(HIGHPASS_HZ, EEG_SAMPLE_RATE), notch(MAINS_HZ, EEG_SAMPLE_RATE)],
    values: new Float32Array(RING),
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
  });
  c.first ??= firstIndex;
  c.end = Math.max(c.end, firstIndex + samples.length);
  feedAudio();
}

// Sound: each ear gets one side's bipolar signal, forehead minus ear channel. Both are
// measured against the Fpz reference, so the difference cancels what the reference picks up.
const [TP9, AF7, AF8, TP10] = ['TP9', 'AF7', 'AF8', 'TP10'].map((n) => EEG_CHANNELS.indexOf(n));
const soundSelect = document.getElementById('sound');
const pitchSelect = document.getElementById('pitch');
const shiftSelect = document.getElementById('shift');
const volumeInput = document.getElementById('volume');
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
  const left = new Float32Array(end - sentEnd);
  const right = new Float32Array(end - sentEnd);
  for (let i = sentEnd; i < end; i++) {
    const v = (ch) => channels[ch].values[i % RING];
    left[i - sentEnd] = v(AF7) - v(TP9);
    right[i - sentEnd] = v(AF8) - v(TP10);
  }
  audio.node.port.postMessage({ type: 'eeg', sides: [left, right] }, [left.buffer, right.buffer]);
  sentEnd = end;
}

async function startAudio() {
  const context = new AudioContext();
  await context.audioWorklet.addModule('sonify-worklet.js');
  const node = new AudioWorkletNode(context, 'sonifier', {
    numberOfInputs: 0,
    outputChannelCount: [2],
  });
  const gain = context.createGain();
  node.connect(gain).connect(context.destination);
  audio = { context, node, gain };
}

function sendSoundSettings() {
  const mode = soundSelect.value;
  document.getElementById('pitch-control').hidden = mode !== 'audify';
  document.getElementById('shift-control').hidden = mode !== 'shift';
  if (!audio) return;
  audio.node.port.postMessage({
    type: 'settings',
    settings: { mode, pitch: Number(pitchSelect.value), shift: Number(shiftSelect.value) },
  });
  // Volume on a squared curve, which sounds more even across the slider.
  audio.gain.gain.setTargetAtTime(Number(volumeInput.value) ** 2, audio.context.currentTime, 0.05);
}

soundSelect.addEventListener('change', async () => {
  // The browser only allows audio to start from a user action, such as this change.
  if (soundSelect.value !== 'off' && !audio) await startAudio();
  sendSoundSettings();
  if (!audio) return;
  if (soundSelect.value === 'off') audio.context.suspend();
  else audio.context.resume();
});
for (const control of [pitchSelect, shiftSelect, volumeInput]) {
  control.addEventListener('input', sendSoundSettings);
}

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
    mid + toPx(cx.values[i % RING]),
    mid - toPx(cy.values[i % RING]),
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
