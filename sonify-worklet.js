import { Normalizer, Granular, Bitcrusher, SpectralMap, LayeredShift } from './sonify-core.js';

// Audio-thread side of the sonification. Each side of the head arrives as the plot's x and
// y (left AF7 and TP9, right AF8 and TP10) and becomes one audio channel. Granular
// plays the bipolar difference x - y; the other modes treat the point x + iy as a complex
// signal, so the trace's direction of rotation is audible. The page posts EEG in small
// batches and settings as they change.
class Sonifier extends AudioWorkletProcessor {
  constructor() {
    super();
    this.settings = {
      mode: 'off',
      speed: 50,
      crush: false,
      map: { eegLow: 1, eegHigh: 45, audioLow: 55, audioHigh: 7040 },
    };
    this.sides = [0, 1].map(() => ({
      bipolar: new Normalizer(),
      trace: new Normalizer(),
      granular: new Granular(),
      crusher: new Bitcrusher(),
      spectral: new SpectralMap(),
      layers: new LayeredShift(),
    }));
    this.port.onmessage = ({ data }) => {
      if (data.type === 'settings') Object.assign(this.settings, data.settings);
      if (data.type === 'eeg') {
        data.sides.forEach(({ x, y }, s) => {
          const side = this.sides[s];
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
      }
      // Soft limit, so blinks and pops can't blast.
      for (let n = 0; n < channel.length; n++) channel[n] = Math.tanh(channel[n]);
    });
    return true;
  }
}

registerProcessor('sonifier', Sonifier);
