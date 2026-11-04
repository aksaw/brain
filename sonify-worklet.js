import { Normalizer, Wavetable, Heterodyne } from './sonify-core.js';

// Audio-thread side of the sonification: turns the two bipolar EEG signals (left side
// AF7 - TP9, right side AF8 - TP10) into the left and right audio channels. The page
// posts EEG in small batches and settings as they change.
class Sonifier extends AudioWorkletProcessor {
  constructor() {
    super();
    this.settings = { mode: 'off', pitch: 110, shift: 200 };
    this.sides = [0, 1].map(() => ({
      normalizer: new Normalizer(),
      wavetable: new Wavetable(),
      heterodyne: new Heterodyne(),
    }));
    this.port.onmessage = ({ data }) => {
      if (data.type === 'settings') Object.assign(this.settings, data.settings);
      if (data.type === 'eeg') {
        data.sides.forEach((samples, s) => {
          const side = this.sides[s];
          for (const x of samples) {
            const v = side.normalizer.process(x);
            side.wavetable.push(v);
            side.heterodyne.push(v);
          }
        });
      }
    };
  }

  process(_inputs, outputs) {
    const { mode, pitch, shift } = this.settings;
    outputs[0].forEach((channel, s) => {
      const side = this.sides[s];
      for (let n = 0; n < channel.length; n++) {
        let y = 0;
        if (mode === 'audify') y = side.wavetable.next(pitch, sampleRate);
        else if (mode === 'shift') y = side.heterodyne.next(shift, sampleRate);
        channel[n] = Math.tanh(y); // soft limit, so blinks and pops can't blast
      }
    });
    return true;
  }
}

registerProcessor('sonifier', Sonifier);
