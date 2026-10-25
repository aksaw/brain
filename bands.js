// Band power estimates from a window of EEG samples.

export const BANDS = [
  { symbol: 'δ', name: 'Delta', lo: 1, hi: 4 },
  { symbol: 'θ', name: 'Theta', lo: 4, hi: 8 },
  { symbol: 'α', name: 'Alpha', lo: 8, hi: 13 },
  { symbol: 'β', name: 'Beta', lo: 13, hi: 30 },
  { symbol: 'γ', name: 'Gamma', lo: 30, hi: 44 },
];

// In-place iterative radix-2 FFT; the length must be a power of two.
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = (-2 * Math.PI) / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const cr = Math.cos(angle * k);
        const ci = Math.sin(angle * k);
        const a = i + k;
        const b = a + half;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
      }
    }
  }
}

// Power in each of BANDS, in µV², for `samples` (oldest first) taken at `rate` Hz.
// Uses a Hann-windowed periodogram, so a sine of amplitude A reads as A²/2.
export function bandPowers(samples, rate) {
  const n = samples.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += samples[i];
  mean /= n;

  const re = new Float64Array(n);
  const im = new Float64Array(n);
  let windowEnergy = 0;
  for (let i = 0; i < n; i++) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
    re[i] = (samples[i] - mean) * w;
    windowEnergy += w * w;
  }
  fft(re, im);

  // One-sided PSD (µV²/Hz) summed over each band's bins, times the bin width.
  const df = rate / n;
  return BANDS.map(({ lo, hi }) => {
    let sum = 0;
    for (let k = Math.ceil(lo / df); k < hi / df; k++) sum += re[k] ** 2 + im[k] ** 2;
    return ((2 * sum) / (rate * windowEnergy)) * df;
  });
}
