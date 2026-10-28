// Minimal Muse 2 client over Web Bluetooth.
// Protocol notes: https://github.com/urish/muse-js (reverse-engineered GATT layout).

const SERVICE = 0xfe8d;
const uuid = (short) => `273e${short}-4c4d-454d-96be-f03bac821358`;

const CONTROL = uuid('0001');
const TELEMETRY = uuid('000b');
const GYRO = uuid('0009');
const ACCEL = uuid('000a');

export const EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10'];
const EEG_CHARACTERISTICS = ['0003', '0004', '0005', '0006'].map(uuid);

export const EEG_SAMPLE_RATE = 256;
export const EEG_SAMPLES_PER_PACKET = 12;

// Commands are framed as [length, ...ascii, '\n'].
function encodeCommand(cmd) {
  const bytes = new TextEncoder().encode(`X${cmd}\n`);
  bytes[0] = bytes.length - 1;
  return bytes;
}

// 18 bytes -> 12 unsigned 12-bit samples, converted to microvolts.
function decodeEeg(view) {
  const samples = new Float32Array(EEG_SAMPLES_PER_PACKET);
  for (let i = 0, j = 0; i < 18; i += 3) {
    const b0 = view.getUint8(2 + i);
    const b1 = view.getUint8(3 + i);
    const b2 = view.getUint8(4 + i);
    samples[j++] = 0.48828125 * (((b0 << 4) | (b1 >> 4)) - 0x800);
    samples[j++] = 0.48828125 * ((((b1 & 0x0f) << 8) | b2) - 0x800);
  }
  return samples;
}

// 3 samples of (x, y, z) int16, big-endian, after the 2-byte sequence number.
function decodeImu(view, scale) {
  const samples = [];
  for (let k = 0; k < 3; k++) {
    const o = 2 + k * 6;
    samples.push({
      x: view.getInt16(o) * scale,
      y: view.getInt16(o + 2) * scale,
      z: view.getInt16(o + 4) * scale,
    });
  }
  return samples;
}

function decodeTelemetry(view) {
  return {
    battery: view.getUint16(2) / 512, // percent
    voltage: view.getUint16(4) * 2.2, // mV
    temperature: view.getUint16(8),
  };
}

export class Muse {
  constructor() {
    this.device = null;
    this.control = null;
    // Callbacks; assign before connect().
    this.onEeg = (_channel, _sequence, _samples) => {};
    this.onAccel = (_samples) => {};
    this.onGyro = (_samples) => {};
    this.onTelemetry = (_telemetry) => {};
    this.onDisconnect = () => {};
  }

  async connect() {
    this.device = await navigator.bluetooth.requestDevice({
      filters: [{ services: [SERVICE] }],
    });
    this.device.addEventListener('gattserverdisconnected', () => this.onDisconnect());

    const server = await this.device.gatt.connect();
    const service = await server.getPrimaryService(SERVICE);
    this.control = await service.getCharacteristic(CONTROL);

    await this.#subscribe(service, TELEMETRY, (v) => this.onTelemetry(decodeTelemetry(v)));
    await this.#subscribe(service, ACCEL, (v) => this.onAccel(decodeImu(v, 0.0000610352)));
    await this.#subscribe(service, GYRO, (v) => this.onGyro(decodeImu(v, 0.0074768)));
    for (let ch = 0; ch < EEG_CHARACTERISTICS.length; ch++) {
      await this.#subscribe(service, EEG_CHARACTERISTICS[ch], (v) =>
        this.onEeg(ch, v.getUint16(0), decodeEeg(v)),
      );
    }

    await this.#send('h'); // halt
    await this.#send('p21'); // preset: 4 EEG channels, no aux
    await this.#send('s'); // status
    await this.#send('d'); // start streaming
  }

  async disconnect() {
    if (!this.device?.gatt.connected) return;
    try {
      await this.#send('h');
    } catch {
      // Device may already be going away.
    }
    this.device.gatt.disconnect();
  }

  get name() {
    return this.device?.name ?? '';
  }

  async #send(cmd) {
    await this.control.writeValue(encodeCommand(cmd));
  }

  async #subscribe(service, id, handler) {
    const characteristic = await service.getCharacteristic(id);
    characteristic.addEventListener('characteristicvaluechanged', (e) => handler(e.target.value));
    await characteristic.startNotifications();
  }
}
