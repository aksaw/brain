# Muse 2 web viewer

A dependency-free page that connects to a Muse 2 headband over Web Bluetooth and plots
the four EEG channels (TP9, AF7, AF8, TP10) live, plus battery, accelerometer and gyro.

## Run

Web Bluetooth needs Chrome or Edge and a secure context (`localhost` counts). Serve this
folder with the included server, which tells the browser not to cache, so edits to the
JavaScript (including the audio worklet) are always picked up on reload:

```bash
python web/serve.py
```

Then open http://localhost:8000 (it lists the apps) or go straight to one, turn on the
headband, and click **Connect**:

- http://localhost:8000/dashboard.html – traces, contact quality, band powers and artifacts.
- http://localhost:8000/bineural.html – two parametric (x-y) plots, one per side of the head:
  AF7 against TP9 and AF8 against TP10, after the 1 Hz high-pass and 60 Hz notch.
- http://localhost:8000/sculpture.html – Bineural's 3D curves built as solid sculpture in a
  black space. Needs an internet connection the first time, for Three.js from a CDN.

The headband accepts one connection at a time, so disconnect one page before connecting
another.

## Files

- `muse.js` – BLE client: GATT UUIDs, command framing, packet decoding.
- `app.js` – the dashboard: ring buffers, contact quality, and canvas rendering (5 s window, DC offset removed per channel).
- `bineural.js` – the x-y plots: filters each channel and pairs samples by packet sequence number.
  Also feeds the sound each side's trace, AF7/TP9 to the left ear and AF8/TP10 to the right.
  The plots pass a 15 Hz low-pass to smooth the traces; the sound does not, so it keeps
  the full EEG band. Views: 2D; 3D x, y, x delayed by τ; 3D x, y, y delayed by τ; 3D Takens embeddings (one signal
  per side against itself τ and 2τ earlier, for the bipolar x − y or the average
  (x + y)/2); 3D PCA (all four channels on their three principal components). For
  the delay views the Delay setting picks τ: a quarter period of a band's typical
  frequency (delta 2 Hz to gamma 40 Hz), or automatic, the first zero of the embedded
  signal's autocorrelation, re-measured twice a second and glided. The 3D views are shown in
  perspective and turned by the Motion setting: auto-rotate, mouse drag, or head movement
  from the headband's gyroscope (integrated, drift-corrected, amplified 3×). Overlay puts
  both hemispheres in one plot, drawn additively.
- `sculpture.js` – the sculpture page (Three.js). Forms:
  - tube: swept along each side's trail, tapering to nothing at its oldest end (thickness
    None: a plain line);
  - shell between hemispheres: lofted between the left and right curves, with tubes along
    the edges (None: bare surface);
  - shrink-wrap: metaballs spaced evenly along the trail, fusing where it passes close to
    itself. Live controls: Ball size (radius); Blend (how far each ball's field reaches:
    separate strands at low settings, one liquid skin at high); Spacing (distance between
    balls along the curve, in ball radii: smooth cord when close, beads when far); Grid
    (how finely the surface is sampled: faceted at 24, smooth at 72, which is also much
    slower). Balls are thinned out automatically when there'd be too many to compute;
  - recurrence threads: a thread wherever the trajectory returns close to where it was at
    least half a cycle earlier (a recurrence plot in 3D); the Closeness slider sets how
    close counts, live.
  Materials: marble, bronze, white bronze, chrome, concrete, white stone (matte,
  unpatterned). Textures are laid out by distance along the form, so they aren't
  stretched, and anchored to the data, so they don't slide. Settings sit in a panel on the
  right, grouped into Signal, Form, Look and Save; the button in the top-right corner hides it
  (remembered between visits). Orbit with the mouse; Freeze
  holds the current form; Save writes the visible sculpture as .glb or .stl (.stl holds
  surfaces only, not lines; tube and shrink-wrap are closed solids; the shell has no
  thickness, so it needs thickening before printing).
- `eeg-stream.js` – the filtered stream, delay embeddings and automatic delay used by the
  sculpture page.
- `sonify-worklet.js` – audio-thread processor that turns the two traces into stereo sound.
- `sonify-core.js` – the sound modes. Except for granular, they treat each side's point as
  a complex signal z = x + iy, so the direction the trace turns is audible:
  - granular: the bipolar difference x − y played back sped up (×25 to ×100) as many short,
    randomly timed, overlapping grains, so a rhythm at f Hz is heard as a band of textured
    noise around speed × f Hz, with no repeating loop; optionally bitcrushed (6 bits,
    held at 8 kHz);
  - layered shift: each band of z rotated onto its note of A in five octaves (55 Hz for
    delta up to 880 Hz for gamma), anticlockwise motion just above the note and clockwise
    just below, each layer at its own level;
  - spectral map: complex FFT peaks mapped log-to-log out from the centre of the audio range,
    anticlockwise up and clockwise down, snapped to harmonics of 55 Hz that fade in and out.

  Levels follow a slow (30 s) running average, so real swells are heard. The page adds a
  generated stereo reverb with an amount slider. The plots always show the newest data;
  the sound runs behind it (layered shift about 0.12 s, spectral map about 0.6 s).
- `filters.js` – streaming high-pass and mains notch filters applied to incoming samples.
- `bands.js` – FFT band power (delta to gamma) for the band chart.
- `artifacts.js` – detection of blinks, sideways eye movements, muscle, head movement and
  electrode pops. Drives the shading on the traces, the pausing of band powers, and the eyes.
