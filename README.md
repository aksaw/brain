# Muse 2 web viewer

A dependency-free page that connects to a Muse 2 headband over Web Bluetooth and plots
the four EEG channels (TP9, AF7, AF8, TP10) live, plus battery, accelerometer and gyro.

## Run

Web Bluetooth needs Chrome or Edge and a secure context (`localhost` counts). Serve this
folder with any static server, for example:

```bash
python -m http.server 8000 -d web
```

Then open http://localhost:8000 (it lists the apps) or go straight to one, turn on the
headband, and click **Connect**:

- http://localhost:8000/dashboard.html – traces, contact quality, band powers and artifacts.
- http://localhost:8000/bineural.html – two parametric (x-y) plots, one per side of the head:
  AF7 against TP9 and AF8 against TP10, after the 1 Hz high-pass and 60 Hz notch.

The headband accepts one connection at a time, so disconnect one page before connecting
another.

## Files

- `muse.js` – BLE client: GATT UUIDs, command framing, packet decoding.
- `app.js` – the dashboard: ring buffers, contact quality, and canvas rendering (5 s window, DC offset removed per channel).
- `bineural.js` – the x-y plots: filters each channel and pairs samples by packet sequence number.
  Also feeds the sound: AF7 − TP9 to the left ear and AF8 − TP10 to the right.
- `sonify-worklet.js` – audio-thread processor that turns the two signals into stereo sound.
- `sonify-core.js` – the sound modes: audification (the last second of EEG played as one
  cycle of a tone, so a rhythm at n Hz becomes the nth harmonic) and frequency shift
  (Hilbert-transform shift, so a rhythm at f Hz is heard at shift + f Hz).
- `filters.js` – streaming high-pass and mains notch filters applied to incoming samples.
- `bands.js` – FFT band power (delta to gamma) for the band chart.
- `artifacts.js` – detection of blinks, sideways eye movements, muscle, head movement and
  electrode pops. Drives the shading on the traces, the pausing of band powers, and the eyes.
