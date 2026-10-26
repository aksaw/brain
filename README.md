# Muse 2 web viewer

A dependency-free page that connects to a Muse 2 headband over Web Bluetooth and plots
the four EEG channels (TP9, AF7, AF8, TP10) live, plus battery, accelerometer and gyro.

## Run

Web Bluetooth needs Chrome or Edge and a secure context (`localhost` counts). Serve this
folder with any static server, for example:

```bash
python -m http.server 8000 -d web
```

Then open http://localhost:8000, turn on the headband, and click **Connect**.

## Files

- `muse.js` – BLE client: GATT UUIDs, command framing, packet decoding.
- `app.js` – ring buffers, contact quality, and canvas rendering (5 s window, DC offset removed per channel).
- `filters.js` – streaming high-pass and mains notch filters applied to incoming samples.
- `bands.js` – FFT band power (delta to gamma) for the band chart.
- `artifacts.js` – detection of blinks, sideways eye movements, muscle, head movement and
  electrode pops. Drives the shading on the traces, the pausing of band powers, and the eyes.
