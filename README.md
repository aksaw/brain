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
    close counts, live;
  - long exposure: every point the curve passes through is splatted into a slowly fading
    density field, drawn as a surface around where it lingers (Exposure: how long a pass
    lingers; Level: how dense a region must be; Clear starts afresh). A new exposure
    starts from the history there is, older passes faded;
  - spectral vessel (from the measures, not a curve): a pot whose height is time (newest
    at the rim), angle is frequency (left hemisphere 1–30 Hz round one half, right back
    round the other) and wall bulges where a frequency is stronger than usual (Relief);
  - grown shell (from the measures): a seashell grown by accretion, two rings a second;
    posterior alpha sets how fast each ring widens, asymmetry squashes the opening, peak
    frequency deepens the ornament. Shape: nautilus (a flat spiral of wrapping whorls with
    growth lines), auger (a tall slender spire with spiral ridges) or conch (a big body
    whorl under a pointed spire, knobbed shoulder); Flare: how fast whorls widen.
  Materials: marble, bronze, white bronze, chrome, concrete, white stone (matte,
  unpatterned). Textures are laid out by distance along the form, so they aren't
  stretched, and anchored to the data, so they don't slide. Settings sit in a panel on the
  right, grouped into Signal, Form, Look and Save; the button in the top-right corner hides it
  (remembered between visits). Orbit with the mouse; Freeze
  holds the current form; Save writes the visible sculpture as .glb or .stl (.stl holds
  surfaces only, not lines; tube and shrink-wrap are closed solids; the shell has no
  thickness, so it needs thickening before printing).
- `eeg-stream.js` – the filtered stream (smoothed for the traces, full band for band
  power), delay embeddings and automatic delay used by the sculpture page.
- `alpha-features.js` – the slow alpha curves on the sculpture page's Curve menu:
  - whole head, one curve: x is posterior alpha (relative 8–13 Hz power at TP9/TP10), y
    is frontal asymmetry (ln alpha AF8 − ln alpha AF7), z is the alpha peak frequency
    (followed only while alpha is clear); Side and the shell between hemispheres don't
    apply;
  - per hemisphere, one curve each from that side's own electrodes: x is back alpha
    (relative alpha at TP9 or TP10), y is front alpha (AF7 or AF8), z is that side's
    alpha peak frequency. Both sides share one baseline per axis, so a difference between
    them is kept: the shell between hemispheres widens where they disagree.
  Measured 16 times a second over 2 s windows, smoothed over about a second, and scaled
  against a running one-minute baseline. Their trails run from 5 s to 5 min; Scale and
  Delay don't apply. Also kept: band powers (delta, theta, alpha, beta) and a whitened
  spectrum per hemisphere, for the harmonograph and the spectral vessel.
- `generators.js` – geometry for forms with structure of their own, steered by those
  measures: the harmonograph (a Curve option: three pairs of slow, slightly detuned
  pendulums whose amplitudes follow the band powers and whose phase follows frontal
  asymmetry), the spectral vessel and the grown shell.
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
