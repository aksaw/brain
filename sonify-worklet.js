import { Normalizer, Granular, Bitcrusher, SpectralMap, LayeredShift, Thump } from './sonify-core.js';
import { TracePlayer, ShapeDrone, Marimba } from './sonify-trace.js';

// Audio-thread side of the sonification. Each side of the head arrives as the plot's x and
// y (left AF7 and TP9, right AF8 and TP10) and becomes one audio channel. Granular
// plays the bipolar difference x - y; the other modes treat the point x + iy as a complex
// signal, so the trace's direction of rotation is audible. Drone and plucks instead follow
// the trace as the plot draws it (dx, dy: smoothed, in µV; divided by the plot's scale
// here). The page posts EEG in small batches and settings as they change.
class Sonifier extends AudioWorkletProcessor {
  constructor() {
    super();
    this.settings = {
      mode: 'off',
      speed: 50,
      crush: false,
      scale: 50, // µV at the plot's edge
      trail: 1, // seconds of trail on screen
      droneLayer: 0, // level of the drone under the plucks, 0..about 2
      map: { eegLow: 1, eegHigh: 45, audioLow: 55, audioHigh: 7040 },
    };
    this.sides = [0, 1].map(() => ({
      bipolar: new Normalizer(),
      trace: new Normalizer(),
      granular: new Granular(),
      crusher: new Bitcrusher(),
      spectral: new SpectralMap(),
      layers: new LayeredShift(),
      player: new TracePlayer(),
      drone: new ShapeDrone(),
      marimba: new Marimba(),
    }));
    this.traceOut = [new Float32Array(128), new Float32Array(128)];
    this.droneLayerGain = 0; // eased towards settings.droneLayer, so moving it doesn't click
    // The heartbeat thump, the same in both ears; the page posts a beat when the PPG sees one.
    this.thump = new Thump();
    this.thumpLevel = 0.45;
    this.thumpBuffer = new Float32Array(128);
    this.port.onmessage = ({ data }) => {
      if (data.type === 'beat') this.thump.trigger();
      // A pluck found on the page, for one side or both (the whole-head view).
      if (data.type === 'pluck') {
        for (const s of data.sides) this.sides[s].marimba.strike(data.freq, data.velocity, sampleRate);
      }
      if (data.type === 'settings') Object.assign(this.settings, data.settings);
      if (data.type === 'eeg') {
        data.sides.forEach(({ x, y, dx, dy }, s) => {
          const side = this.sides[s];
          for (let i = 0; i < dx.length; i++) side.player.push(dx[i], dy[i]);
          for (let i = 0; i < x.length; i++) {
            side.granular.push(side.bipolar.process(x[i] - y[i]));
            const g = side.trace.gain(x[i] * x[i] + y[i] * y[i]);
            side.spectral.push(g * x[i], g * y[i]);
            side.layers.push(g * x[i], g * y[i]);
          }
        });
      }
    };
  }

  process(_inputs, outputs) {
    const { mode, speed, crush, map } = this.settings;
    const length = outputs[0][0].length;
    if (this.thumpBuffer.length !== length) this.thumpBuffer = new Float32Array(length);
    for (let n = 0; n < length; n++) this.thumpBuffer[n] = this.thumpLevel * this.thump.next(sampleRate);
    this.#renderTrace(mode, length);
    outputs[0].forEach((channel, s) => {
      const side = this.sides[s];
      channel.fill(0);
      if (mode === 'granular') {
        for (let n = 0; n < channel.length; n++) {
          const y = side.granular.next(speed, sampleRate);
          channel[n] = crush ? side.crusher.process(y, sampleRate) : y;
        }
      } else if (mode === 'layers') {
        for (let n = 0; n < channel.length; n++) channel[n] = side.layers.next(sampleRate);
      } else if (mode === 'spectral') {
        side.spectral.render(channel, map, sampleRate);
      } else if (mode === 'drone' || mode === 'plucks') {
        // Each ear mostly its own side, with a little of the other, which is easier on the
        // ears for tones than hard panning.
        const own = this.traceOut[s];
        const other = this.traceOut[1 - s];
        for (let n = 0; n < channel.length; n++) channel[n] = own[n] + 0.2 * other[n];
      }
      // Soft limit, so blinks and pops can't blast. The thump goes in before it, so it
      // presses the rest down a touch as it lands.
      for (let n = 0; n < channel.length; n++) {
        channel[n] = Math.tanh(channel[n] + this.thumpBuffer[n]);
      }
    });
    return true;
  }

  // Advances both sides' trace playback (always, so it stays current) and renders into
  // traceOut the drone, or the marimba's plucks with the drone as a layer underneath at its
  // own level.
  #renderTrace(mode, length) {
    const { scale, trail, droneLayer } = this.settings;
    const plucks = mode === 'plucks';
    const ease = 1 - Math.exp(-1 / (0.05 * sampleRate));
    const gains = new Float32Array(length);
    for (let n = 0; n < length; n++) {
      this.droneLayerGain += ((plucks ? droneLayer : 0) - this.droneLayerGain) * ease;
      gains[n] = this.droneLayerGain;
    }
    this.sides.forEach((side, s) => {
      if (this.traceOut[s].length !== length) this.traceOut[s] = new Float32Array(length);
      const out = this.traceOut[s];
      const { player } = side;
      side.drone.shift = plucks ? -1 : 0; // under the plucks it sits an octave lower
      for (let n = 0; n < length; n++) {
        out[n] = plucks ? side.marimba.next(sampleRate) : 0;
        if (!player.next(sampleRate)) continue;
        if (mode !== 'drone' && !plucks) continue;
        const x = player.x / scale;
        const y = player.y / scale;
        // The drone keeps following the trail under the plucks even when turned down, so
        // it's already in tune with the shape when turned up.
        const drone = side.drone.next(x, y, player.vx / scale, player.vy / scale, trail, sampleRate);
        out[n] += plucks ? gains[n] * drone : drone;
      }
    });
  }
}

registerProcessor('sonifier', Sonifier);
