import { Muse, EEG_CHANNELS, EEG_SAMPLE_RATE, EEG_SAMPLES_PER_PACKET } from './muse.js';
import { BANDS, bandPowers } from './bands.js';
import { highpass, notch } from './filters.js';

const WINDOW_SECONDS = 5;
const WINDOW = WINDOW_SECONDS * EEG_SAMPLE_RATE;
const COLORS = ['#437bbd', '#48886c', '#b28942', '#b8555b'];

const FFT_SIZE = 512; // 2 s at 256 Hz, 0.5 Hz resolution
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

const connectButton = document.getElementById('connect');
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

// Smoothed power per channel per band, in µV².
const bandPower = EEG_CHANNELS.map(() => BANDS.map(() => 0));

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
  muse.onEeg = (ch, _seq, samples) => {
    const buf = buffers[ch];
    const now = performance.now();
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
    document.getElementById('accel').textContent = formatXyz(s[s.length - 1], 2);
  };
  muse.onGyro = (s) => {
    document.getElementById('gyro').textContent = formatXyz(s[s.length - 1], 1);
  };
  muse.onDisconnect = () => {
    muse = null;
    setStatus('Disconnected');
    connectButton.textContent = 'Connect';
  };

  try {
    setStatus('Connecting…');
    await muse.connect();
    setStatus(`Streaming from ${muse.name}`);
    connectButton.textContent = 'Disconnect';
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`);
    muse = null;
  }
});

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

function updateBandPowers() {
  const segment = new Float32Array(FFT_SIZE);
  for (let ch = 0; ch < EEG_CHANNELS.length; ch++) {
    if (packetCounts[ch] * EEG_SAMPLES_PER_PACKET < FFT_SIZE) continue;
    const start = writeIndex[ch] - FFT_SIZE + WINDOW;
    for (let i = 0; i < FFT_SIZE; i++) segment[i] = buffers[ch][(start + i) % WINDOW];
    bandPowers(segment, EEG_SAMPLE_RATE).forEach((p, b) => {
      bandPower[ch][b] += BAND_SMOOTHING * (p - bandPower[ch][b]);
    });
  }
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
      bandCtx.fillRect(x0 + barW * ch, y, barW - 1, yOf(DB_MIN) - y);
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

  for (let ch = 0; ch < EEG_CHANNELS.length; ch++) {
    const buf = buffers[ch];
    const mid = laneH * (ch + 0.5);

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
