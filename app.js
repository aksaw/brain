import { Muse, EEG_CHANNELS, EEG_SAMPLE_RATE, EEG_SAMPLES_PER_PACKET } from './muse.js';
import { BANDS, bandPowers } from './bands.js';
import { highpass, notch } from './filters.js';
import { ArtifactDetector } from './artifacts.js';

const WINDOW_SECONDS = 5;
const WINDOW = WINDOW_SECONDS * EEG_SAMPLE_RATE;
const COLORS = ['#437bbd', '#48886c', '#b28942', '#b8555b'];

const FFT_SIZE = 256; // 1 s segments, 1 Hz resolution
const FFT_HOP = 64; // segments overlap by 75%
const BAND_HISTORY = 3 * EEG_SAMPLE_RATE; // look for clean segments in the last 3 s
const BAND_UPDATE_MS = 250;
const BAND_SMOOTHING = 0.3; // weight of each new estimate; ~1 s time constant
const DB_MIN = -10;
const DB_MAX = 40;

const HIGHPASS_HZ = 1;
const MAINS_HZ = 60;
const RAIL_UV = 999; // the 12-bit samples span roughly ±1000 µV

// Contact quality from the robust spread (scaled median absolute deviation) of the
// last 2 s of filtered signal, which a brief blink barely moves. The thresholds are
// rough rules of thumb for dry electrodes, not calibrated values.
const QUALITY_WINDOW = 2 * EEG_SAMPLE_RATE;
const QUALITY_LEVELS = [
  { label: 'Good', className: 'q-good', maxSpread: 20, opacity: 1 },
  { label: 'Fair', className: 'q-fair', maxSpread: 50, opacity: 0.65 },
  { label: 'Poor', className: 'q-poor', maxSpread: Infinity, opacity: 0.35 },
];
const NO_SIGNAL = { label: 'No signal', className: 'q-poor', opacity: 0.15 };
const NO_DATA = { label: '–', className: '', opacity: 1 };

const MARK_LABELS = {
  blink: 'blink',
  eye: 'eye movement',
  muscle: 'muscle',
  movement: 'movement',
  pop: 'pop',
};

const connectButton = document.getElementById('connect');
const recordButton = document.getElementById('record');
const calibrateButton = document.getElementById('calibrate');
const statusEl = document.getElementById('status');
const scaleSelect = document.getElementById('scale');
const highpassCheckbox = document.getElementById('highpass');
const notchCheckbox = document.getElementById('notch');
const canvas = document.getElementById('eeg');
const ctx = canvas.getContext('2d');
const bandCanvas = document.getElementById('band-chart');
const bandCtx = bandCanvas.getContext('2d');
const statsBody = document.querySelector('#stats tbody');

// One ring buffer per channel holding the last WINDOW samples.
const buffers = EEG_CHANNELS.map(() => new Float32Array(WINDOW));
const writeIndex = EEG_CHANNELS.map(() => 0);
const packetCounts = EEG_CHANNELS.map(() => 0);
const lastPacketAt = EEG_CHANNELS.map(() => -Infinity);
const railedAt = EEG_CHANNELS.map(() => -Infinity);

// Smoothed power per channel per band, in µV², and whether it's paused for an artifact.
const bandPower = EEG_CHANNELS.map(() => BANDS.map(() => 0));
const bandHeld = EEG_CHANNELS.map(() => false);

// One chain of filters per channel; rebuilt (with fresh state) when the settings change.
function makeFilterChains() {
  return EEG_CHANNELS.map(() => {
    const chain = [];
    if (highpassCheckbox.checked) chain.push(highpass(HIGHPASS_HZ, EEG_SAMPLE_RATE));
    if (notchCheckbox.checked) chain.push(notch(MAINS_HZ, EEG_SAMPLE_RATE));
    return chain;
  });
}
let filterChains = makeFilterChains();
highpassCheckbox.addEventListener('change', () => (filterChains = makeFilterChains()));
notchCheckbox.addEventListener('change', () => (filterChains = makeFilterChains()));

const statCells = EEG_CHANNELS.map((name, ch) => {
  const row = statsBody.insertRow();
  row.insertCell().innerHTML = `<span style="color:${COLORS[ch]}">${name}</span>`;
  return { quality: row.insertCell(), std: row.insertCell(), packets: row.insertCell() };
});

// 10-20 sites as [degrees down from the vertex, degrees clockwise from the nose].
// Inner sites (F3, P3, ...) are approximate; the rest sit on the 10% grid.
const SITES = {
  Fpz: [72, 0], Fp1: [72, -18], Fp2: [72, 18], AF7: [72, -36], AF8: [72, 36],
  F7: [72, -54], F8: [72, 54], T7: [72, -90], T8: [72, 90], P7: [72, -126], P8: [72, 126],
  O1: [72, -162], O2: [72, 162], Oz: [72, 180],
  F3: [46, -39], F4: [46, 39], C3: [36, -90], C4: [36, 90], P3: [46, -141], P4: [46, 141],
  Fz: [36, 0], Cz: [0, 0], Pz: [36, 180],
  TP9: [90, -108], TP10: [90, 108],
};
const REFERENCE = 'Fpz';

function drawMontage(svg) {
  const R = 100; // radius of the 90-degree (ear level) circle
  const project = ([theta, phi]) => {
    const r = (theta / 90) * R;
    const a = (phi * Math.PI) / 180;
    return [r * Math.sin(a), -r * Math.cos(a)];
  };
  let html = `
    <path class="head" d="M -10 -99.5 L 0 -114 L 10 -99.5" />
    <ellipse class="head" cx="-104" cy="0" rx="5" ry="13" />
    <ellipse class="head" cx="104" cy="0" rx="5" ry="13" />
    <circle class="head" r="${R}" />
    <circle class="ring" r="${0.8 * R}" />`;
  for (const [name, pos] of Object.entries(SITES)) {
    const [x, y] = project(pos);
    const ch = EEG_CHANNELS.indexOf(name);
    if (ch >= 0) {
      html += `<g data-channel="${ch}">
        <circle r="10" cx="${x}" cy="${y}" fill="${COLORS[ch]}" />
        <text class="muse-label" x="${x}" y="${y}" text-anchor="middle" dominant-baseline="central">${name}</text>
      </g>`;
    } else {
      html += `<circle class="${name === REFERENCE ? 'ref' : 'site'}" r="8" cx="${x}" cy="${y}" />
        <text class="site-label" x="${x}" y="${y}" text-anchor="middle" dominant-baseline="central">${name}</text>`;
    }
  }
  svg.innerHTML = html;
}
drawMontage(document.querySelector('#montage svg'));
const montageSites = EEG_CHANNELS.map((_, ch) =>
  document.querySelector(`#montage [data-channel="${ch}"]`),
);

let muse = null;
let artifacts = null;

const eyes = document.querySelectorAll('#eyes .eye');
const gazeGroups = document.querySelectorAll('#eyes .gaze');
const GAZE_SHIFT = 13; // SVG units the irises move sideways
const GAZE_RETURN_MS = 2500; // drift back to centre if no glance back is seen

// -1 left, 0 centre, 1 right. A glance moves one step; the eyes act like a mirror.
let gaze = 0;
let gazeTimer = null;

function setGaze(position) {
  gaze = position;
  for (const g of gazeGroups) g.style.transform = `translateX(${gaze * GAZE_SHIFT}px)`;
}

function lookTowards(direction) {
  setGaze(Math.max(-1, Math.min(1, gaze + (direction === 'left' ? -1 : 1))));
  clearTimeout(gazeTimer);
  gazeTimer = setTimeout(() => setGaze(0), GAZE_RETURN_MS);
}

// Squash the eyes shut and open again; each call starts a fresh animation.
function blinkEyes() {
  for (const eye of eyes) {
    eye.animate(
      [
        { transform: 'scaleY(1)' },
        { transform: 'scaleY(0.06)', offset: 0.4 },
        { transform: 'scaleY(0.06)', offset: 0.6 },
        { transform: 'scaleY(1)' },
      ],
      { duration: 220, easing: 'ease-in-out' },
    );
  }
}

if (!navigator.bluetooth) {
  document.getElementById('unsupported').hidden = false;
  connectButton.disabled = true;
}

connectButton.addEventListener('click', async () => {
  if (muse) {
    await muse.disconnect();
    return;
  }
  muse = new Muse();
  artifacts = new ArtifactDetector(EEG_SAMPLE_RATE, EEG_CHANNELS.length);
  artifacts.onBlink = () => {
    blinkEyes();
    recording?.events.push({ time: performance.now(), label: 'blink' });
  };
  artifacts.onEyeMove = (direction) => {
    lookTowards(direction);
    recording?.events.push({ time: performance.now(), label: `eye ${direction}` });
  };
  muse.onEeg = (ch, seq, samples) => {
    const buf = buffers[ch];
    const now = performance.now();
    artifacts.addEeg(ch, samples, now, seq);
    recording?.eeg[ch].push({ seq, time: now, samples: Array.from(samples) });
    for (let s of samples) {
      if (Math.abs(s) >= RAIL_UV) railedAt[ch] = now;
      for (const filter of filterChains[ch]) s = filter.process(s);
      buf[writeIndex[ch]] = s;
      writeIndex[ch] = (writeIndex[ch] + 1) % WINDOW;
    }
    packetCounts[ch]++;
    lastPacketAt[ch] = now;
  };
  muse.onTelemetry = (t) => {
    document.getElementById('battery').textContent =
      `${t.battery.toFixed(0)}% (${(t.voltage / 1000).toFixed(2)} V)`;
  };
  muse.onAccel = (s) => {
    artifacts.addAccel(s, performance.now());
    document.getElementById('accel').textContent = formatXyz(s[s.length - 1], 2);
  };
  muse.onGyro = (s) => {
    artifacts.addGyro(s, performance.now());
    document.getElementById('gyro').textContent = formatXyz(s[s.length - 1], 1);
  };
  muse.onDisconnect = () => {
    muse = null;
    if (recording) stopRecording();
    recordButton.disabled = true;
    calibrateButton.disabled = true;
    setStatus('Disconnected');
    connectButton.textContent = 'Connect';
  };

  try {
    setStatus('Connecting…');
    await muse.connect();
    setStatus(`Streaming from ${muse.name}`);
    connectButton.textContent = 'Disconnect';
    recordButton.disabled = false;
    calibrateButton.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`);
    muse = null;
  }
});

// Recording: raw EEG packets plus detector events, saved as a CSV when stopped.
let recording = null;

recordButton.addEventListener('click', () => (recording ? stopRecording() : startRecording()));

function startRecording() {
  recording = { eeg: EEG_CHANNELS.map(() => []), events: [] };
  recordButton.textContent = 'Stop & save';
}

function stopRecording({ save = true, name = 'muse' } = {}) {
  const finished = recording;
  recording = null;
  recordButton.textContent = 'Record';
  if (!save) return;
  const csv = recordingCsv(finished);
  const stamp = new Date().toISOString().slice(0, 19).replace(/[-:]/g, '').replace('T', '-');
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  link.download = `${name}-${stamp}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

// Gaze calibration: on-screen cues move the eyes left, centre and right in a fixed rhythm
// while recording, and each cue is written into the CSV so glances can be lined up later.
const CALIBRATION_STEPS = [
  { at: 'centre', seconds: 5, text: 'Keep your head still and look at the centre dot' },
  ...Array.from({ length: 4 }, () => [
    { at: 'left', seconds: 2, text: 'Look at the left dot' },
    { at: 'centre', seconds: 3, text: 'Back to the centre' },
    { at: 'right', seconds: 2, text: 'Look at the right dot' },
    { at: 'centre', seconds: 3, text: 'Back to the centre' },
  ]).flat(),
];
const cueOverlay = document.getElementById('cue-overlay');
const cueText = document.getElementById('cue-text');
let calibration = null; // { cancelled } while a run is in progress

calibrateButton.addEventListener('click', runCalibration);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && calibration) calibration.cancelled = true;
});

async function runCalibration() {
  const run = (calibration = { cancelled: false });
  startRecording();
  cueOverlay.hidden = false;
  for (const step of CALIBRATION_STEPS) {
    for (const dot of cueOverlay.querySelectorAll('.cue-dot')) {
      dot.classList.toggle('active', dot.dataset.at === step.at);
    }
    cueText.textContent = step.text;
    recording?.events.push({ time: performance.now(), label: `cue ${step.at}` });
    const end = performance.now() + step.seconds * 1000;
    while (performance.now() < end && !run.cancelled && recording) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    if (run.cancelled || !recording) break;
  }
  cueOverlay.hidden = true;
  calibration = null;
  // A disconnect mid-run has already saved the recording; Esc discards it.
  if (recording) stopRecording({ save: !run.cancelled, name: 'muse-gaze' });
}

// One row per sample with the raw µV of each channel. Packets with the same sequence
// number on every channel cover the same 12 samples, so rows are aligned by that.
function recordingCsv({ eeg, events }) {
  const bySeq = new Map();
  eeg.forEach((packets, ch) => {
    let wraps = 0;
    let last = null;
    for (const p of packets) {
      if (last !== null && p.seq < last - 32768) wraps++;
      last = p.seq;
      const key = wraps * 65536 + p.seq;
      if (!bySeq.has(key)) bySeq.set(key, { time: p.time, channels: [] });
      bySeq.get(key).channels[ch] = p.samples;
    }
  });

  const rows = ['time_ms,' + EEG_CHANNELS.join(',') + ',event'];
  const pending = [...events].sort((a, b) => a.time - b.time);
  for (const key of [...bySeq.keys()].sort((a, b) => a - b)) {
    const { time, channels } = bySeq.get(key);
    if (EEG_CHANNELS.some((_, ch) => !channels[ch])) continue; // a channel's packet was lost
    for (let k = 0; k < EEG_SAMPLES_PER_PACKET; k++) {
      const t = time - ((EEG_SAMPLES_PER_PACKET - 1 - k) * 1000) / EEG_SAMPLE_RATE;
      let event = '';
      while (pending.length && pending[0].time <= t) {
        const label = pending.shift().label;
        event = event ? `${event};${label}` : label;
      }
      rows.push(`${t.toFixed(1)},${channels.map((c) => c[k].toFixed(2)).join(',')},${event}`);
    }
  }
  return rows.join('\n') + '\n';
}

function setStatus(text) {
  statusEl.textContent = text;
}

function formatXyz({ x, y, z }, digits) {
  return [x, y, z].map((v) => v.toFixed(digits).padStart(7)).join(' ');
}

// Keep each canvas's pixel buffer matched to its on-screen size.
for (const [c, cx] of [[canvas, ctx], [bandCanvas, bandCtx]]) {
  new ResizeObserver(() => {
    const dpr = window.devicePixelRatio || 1;
    c.width = c.clientWidth * dpr;
    c.height = c.clientHeight * dpr;
    cx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }).observe(c);
}

// Welch-style estimate with artifact rejection: the last few seconds are cut into
// overlapping segments, any segment touching an artifact is dropped (a single blink can
// outweigh seconds of EEG in the low bands), and the clean ones are averaged. A channel's
// bars hold their last value only when no segment is clean.
function updateBandPowers() {
  const segment = new Float32Array(FFT_SIZE);
  const segmentMs = (FFT_SIZE * 1000) / EEG_SAMPLE_RATE;
  const marks = artifacts ? artifacts.intervalsUntil(performance.now()) : [];
  for (let ch = 0; ch < EEG_CHANNELS.length; ch++) {
    if (packetCounts[ch] * EEG_SAMPLES_PER_PACKET < BAND_HISTORY) continue;
    const sum = BANDS.map(() => 0);
    let clean = 0;
    for (let offset = 0; offset + FFT_SIZE <= BAND_HISTORY; offset += FFT_HOP) {
      const segmentEnd = lastPacketAt[ch] - (offset * 1000) / EEG_SAMPLE_RATE;
      const touched = marks.some(
        (m) => m.channels.includes(ch) && m.end > segmentEnd - segmentMs && m.start < segmentEnd,
      );
      if (touched) continue;
      const start = writeIndex[ch] - offset - FFT_SIZE + WINDOW;
      for (let i = 0; i < FFT_SIZE; i++) segment[i] = buffers[ch][(start + i) % WINDOW];
      bandPowers(segment, EEG_SAMPLE_RATE).forEach((p, b) => (sum[b] += p));
      clean++;
    }
    bandHeld[ch] = clean === 0;
    if (bandHeld[ch]) continue;
    sum.forEach((p, b) => {
      bandPower[ch][b] += BAND_SMOOTHING * (p / clean - bandPower[ch][b]);
    });
  }
}

function updateArtifactReadouts() {
  if (!artifacts) return;
  const now = performance.now();
  document.getElementById('blinks').textContent =
    `${artifacts.blinksInLast(60000, now)} in the last minute`;
  const muscle = artifacts.activeMuscle().map((ch) => EEG_CHANNELS[ch]);
  document.getElementById('muscle').textContent = muscle.length ? muscle.join(', ') : '–';
}

function contactQuality(ch) {
  const now = performance.now();
  if (now - lastPacketAt[ch] > 1000) return NO_DATA;
  if (now - railedAt[ch] < 1000) return NO_SIGNAL;
  if (packetCounts[ch] * EEG_SAMPLES_PER_PACKET < QUALITY_WINDOW) return NO_DATA;

  const recent = new Float32Array(QUALITY_WINDOW);
  const start = writeIndex[ch] - QUALITY_WINDOW + WINDOW;
  for (let i = 0; i < QUALITY_WINDOW; i++) recent[i] = buffers[ch][(start + i) % WINDOW];
  const median = (values) => values.sort()[values.length >> 1];
  const center = median(recent.slice());
  const spread = 1.4826 * median(recent.map((v) => Math.abs(v - center)));

  if (spread < 0.5) return NO_SIGNAL; // flat: the electrode isn't picking anything up
  return QUALITY_LEVELS.find((level) => spread <= level.maxSpread);
}

function updateContactQuality() {
  for (let ch = 0; ch < EEG_CHANNELS.length; ch++) {
    const quality = contactQuality(ch);
    statCells[ch].quality.textContent = quality.label;
    statCells[ch].quality.className = quality.className;
    montageSites[ch].style.opacity = quality.opacity;
  }
}

setInterval(() => {
  updateBandPowers();
  updateContactQuality();
  updateArtifactReadouts();
}, BAND_UPDATE_MS);

// Grouped bars: one group per band, one bar per channel, log scale.
function drawBands() {
  const w = bandCanvas.clientWidth;
  const h = bandCanvas.clientHeight;
  const left = 36;
  const top = 8;
  const bottom = 36;
  const plotH = h - top - bottom;
  const yOf = (db) => top + plotH * (1 - (db - DB_MIN) / (DB_MAX - DB_MIN));

  bandCtx.clearRect(0, 0, w, h);
  bandCtx.font = '11px system-ui, sans-serif';
  bandCtx.lineWidth = 1;

  bandCtx.strokeStyle = '#2a2a2a';
  bandCtx.fillStyle = '#8f8f8f';
  bandCtx.textAlign = 'right';
  bandCtx.textBaseline = 'middle';
  bandCtx.beginPath();
  for (let db = DB_MIN; db <= DB_MAX; db += 10) {
    const y = Math.round(yOf(db)) + 0.5;
    bandCtx.moveTo(left, y);
    bandCtx.lineTo(w, y);
    bandCtx.fillText(db, left - 6, y);
  }
  bandCtx.stroke();

  const groupW = (w - left) / BANDS.length;
  const barW = (groupW * 0.7) / EEG_CHANNELS.length;
  bandCtx.textAlign = 'center';
  bandCtx.textBaseline = 'top';
  BANDS.forEach((band, b) => {
    const x0 = left + groupW * b + groupW * 0.15;
    for (let ch = 0; ch < EEG_CHANNELS.length; ch++) {
      const db = 10 * Math.log10(bandPower[ch][b]);
      if (!Number.isFinite(db)) continue;
      const y = yOf(Math.min(DB_MAX, Math.max(DB_MIN, db)));
      bandCtx.fillStyle = COLORS[ch];
      bandCtx.globalAlpha = bandHeld[ch] ? 0.35 : 1;
      bandCtx.fillRect(x0 + barW * ch, y, barW - 1, yOf(DB_MIN) - y);
      bandCtx.globalAlpha = 1;
    }
    const cx = left + groupW * (b + 0.5);
    bandCtx.fillStyle = '#e8e8e8';
    bandCtx.fillText(`${band.symbol} ${band.name}`, cx, h - bottom + 6);
    bandCtx.fillStyle = '#8f8f8f';
    bandCtx.fillText(`${band.lo}–${band.hi} Hz`, cx, h - bottom + 20);
  });
}

function draw() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const laneH = h / EEG_CHANNELS.length;
  const scale = Number(scaleSelect.value);

  ctx.clearRect(0, 0, w, h);

  // Vertical grid line every second.
  ctx.strokeStyle = '#2a2a2a';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let s = 1; s < WINDOW_SECONDS; s++) {
    const x = Math.round((s / WINDOW_SECONDS) * w) + 0.5;
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
  }
  ctx.stroke();

  const marks = artifacts ? artifacts.intervalsUntil(performance.now()) : [];

  for (let ch = 0; ch < EEG_CHANNELS.length; ch++) {
    const buf = buffers[ch];
    const mid = laneH * (ch + 0.5);

    // Shade artifacts in this lane. The newest sample (right edge) arrived at lastPacketAt.
    const toX = (t) => w - ((lastPacketAt[ch] - t) / (WINDOW_SECONDS * 1000)) * w;
    ctx.font = '10px system-ui, sans-serif';
    for (const mark of marks) {
      if (!mark.channels.includes(ch)) continue;
      const x0 = Math.max(0, toX(mark.start));
      const x1 = Math.min(w, toX(mark.end));
      if (x1 <= x0) continue;
      ctx.fillStyle = 'rgba(255, 255, 255, 0.07)';
      ctx.fillRect(x0, laneH * ch, x1 - x0, laneH);
      if (x1 - x0 > 40) {
        ctx.fillStyle = '#8f8f8f';
        ctx.fillText(MARK_LABELS[mark.type], x0 + 4, laneH * (ch + 1) - 6);
      }
    }

    // Remove the DC offset so traces stay centred in their lane.
    let mean = 0;
    for (let i = 0; i < WINDOW; i++) mean += buf[i];
    mean /= WINDOW;
    let variance = 0;
    for (let i = 0; i < WINDOW; i++) variance += (buf[i] - mean) ** 2;
    const std = Math.sqrt(variance / WINDOW);

    ctx.strokeStyle = '#2a2a2a';
    ctx.beginPath();
    ctx.moveTo(0, Math.round(mid) + 0.5);
    ctx.lineTo(w, Math.round(mid) + 0.5);
    ctx.stroke();

    // Oldest sample on the left, newest on the right.
    ctx.strokeStyle = COLORS[ch];
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    for (let i = 0; i < WINDOW; i++) {
      const v = buf[(writeIndex[ch] + i) % WINDOW] - mean;
      const x = (i / (WINDOW - 1)) * w;
      const y = mid - (Math.max(-scale, Math.min(scale, v)) / scale) * (laneH / 2);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = COLORS[ch];
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText(EEG_CHANNELS[ch], 8, laneH * ch + 16);

    statCells[ch].std.textContent = std.toFixed(1);
    statCells[ch].packets.textContent = packetCounts[ch];
  }

  drawBands();
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
