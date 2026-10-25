import { Muse, EEG_CHANNELS, EEG_SAMPLE_RATE } from './muse.js';

const WINDOW_SECONDS = 5;
const WINDOW = WINDOW_SECONDS * EEG_SAMPLE_RATE;
const COLORS = ['#4f9cf9', '#3ecf8e', '#f5b544', '#ef6b73'];

const connectButton = document.getElementById('connect');
const statusEl = document.getElementById('status');
const scaleSelect = document.getElementById('scale');
const canvas = document.getElementById('eeg');
const ctx = canvas.getContext('2d');
const statsBody = document.querySelector('#stats tbody');

// One ring buffer per channel holding the last WINDOW samples.
const buffers = EEG_CHANNELS.map(() => new Float32Array(WINDOW));
const writeIndex = EEG_CHANNELS.map(() => 0);
const packetCounts = EEG_CHANNELS.map(() => 0);

const statCells = EEG_CHANNELS.map((name, ch) => {
  const row = statsBody.insertRow();
  row.insertCell().innerHTML = `<span style="color:${COLORS[ch]}">${name}</span>`;
  return { std: row.insertCell(), packets: row.insertCell() };
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
      html += `<circle r="10" cx="${x}" cy="${y}" fill="${COLORS[ch]}" />
        <text class="muse-label" x="${x}" y="${y}" text-anchor="middle" dominant-baseline="central">${name}</text>`;
    } else {
      html += `<circle class="${name === REFERENCE ? 'ref' : 'site'}" r="8" cx="${x}" cy="${y}" />
        <text class="site-label" x="${x}" y="${y}" text-anchor="middle" dominant-baseline="central">${name}</text>`;
    }
  }
  svg.innerHTML = html;
}
drawMontage(document.querySelector('#montage svg'));

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
    for (const s of samples) {
      buf[writeIndex[ch]] = s;
      writeIndex[ch] = (writeIndex[ch] + 1) % WINDOW;
    }
    packetCounts[ch]++;
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

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

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

  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
