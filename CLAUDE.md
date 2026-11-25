# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Browser pages that read a Muse 2 EEG headband over Web Bluetooth and turn the signal into plots, sound, 3D sculpture and a webcam portrait piece. Plain HTML + ES modules: no build step, no package manager, no bundler, no tests, no linter. `README.md` is the detailed description of every page, view, form and sound mode — keep it updated when behaviour changes (commits so far have done so).

## Running

```bash
python serve.py        # optional port argument; serves this folder on http://localhost:8000
```

(The README and serve.py docstring say `python web/serve.py` — a leftover from an older layout; the script is at the repo root.) `serve.py` sends `Cache-Control: no-store` so edited modules, especially the audio worklet, are reloaded; don't swap it for a plain static server.

Requirements: Chrome or Edge (Web Bluetooth, and WebGPU for fast depth on Portrait), a secure context (`localhost`), internet access for CDN dependencies. The headband allows one connection at a time, so only one page can be connected. Verifying changes generally means loading the page in the browser; without a headband, only loading/UI/console errors can be checked.

## Architecture

Four pages, each an HTML file plus one entry module: `dashboard.html`→`app.js`, `bineural.html`→`bineural.js`, `sculpture.html`→`sculpture.js`, `portrait.html`→`portrait.js`. `index.html` just links them. `style.css` is shared.

Shared module layers (imported by relative path):

- **Device**: `muse.js` — BLE client, GATT UUIDs, packet decoding; exports `EEG_CHANNELS` (`TP9, AF7, AF8, TP10`), `EEG_SAMPLE_RATE` (256), `EEG_SAMPLES_PER_PACKET` (12).
- **DSP primitives**: `filters.js` (biquad high-pass/low-pass/notch, streaming), `bands.js` (`fft`, band powers).
- **Derived streams**: `eeg-stream.js` (`EegStream`: per-side filtered streams — smoothed for drawing, full band for power — plus delay embeddings and `AutoDelay`), `alpha-features.js` (`AlphaFeatures`: slow alpha curves, band powers and whitened spectra at `FEATURE_RATE` = 16 Hz against a running baseline), `generators.js` (harmonograph and spectral-vessel geometry from those features), `artifacts.js` (blink/eye/muscle/motion/pop detection, used by the dashboard).
- **Sound**: `bineural.js` loads `sonify-worklet.js` via `audioWorklet.addModule`; the worklet imports the DSP from `sonify-core.js` (which itself imports `bands.js`/`filters.js`), so everything reachable from the worklet must run on the audio thread (no DOM).

Note `app.js` and `bineural.js` predate `eeg-stream.js` and do their own filtering/pairing; Sculpture and Portrait share `EegStream`/`AlphaFeatures`.

External dependencies come only from jsDelivr, pinned by version:
- Three.js 0.169.0 via an `importmap` in `sculpture.html` and `portrait.html` (`three`, `three/addons/`).
- MediaPipe tasks-vision and Transformers.js (Depth Anything V2) imported by full URL in `portrait.js`. Everything runs in the browser; no video leaves the machine.

Sculpture and Portrait use a collapsible right-hand settings panel whose state is persisted in `localStorage`.

## Repo conventions

- `IDEAS.md` is a private, gitignored notes file — never commit it.
- Commit messages are short imperative sentences describing the user-visible change (e.g. "Add structured forms to Sculpture").
