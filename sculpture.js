// Sculpture: each side's 3D trace (the delay embeddings from Bineural) built as geometry
// with real materials, floating in black: a tube swept along the trail, a shell lofted
// between the two hemispheres' curves, a metaball shrink-wrap, or recurrence threads.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { STLExporter } from 'three/addons/exporters/STLExporter.js';
import { Muse, EEG_SAMPLE_RATE } from './muse.js';
import { EegStream, SIDES, EMBEDDINGS, AutoDelay, bandDelay } from './eeg-stream.js';
import { AlphaFeatures } from './alpha-features.js';

const MAX_POINTS = 640; // points per curve; longer trails are sampled more sparsely
const RADIAL = 16; // sides of the tube's cross-section
const ACROSS = 6; // strips across a shell
const TAPER = 0.3; // the oldest part of the trail (this fraction) tapers to nothing
const RIM = 0.35; // a shell's edge tubes, as a fraction of the thickness

const $ = (id) => document.getElementById(id);
const connectButton = $('connect');
const controlsIn = {
  scale: $('scale'),
  trail: $('trail'),
  slowTrail: $('slow-trail'),
  form: $('form'),
  curve: $('curve'),
  delay: $('delay'),
  side: $('side'),
  material: $('material'),
  thickness: $('thickness'),
  ballSize: $('ball-size'),
  blend: $('blend'),
  spacing: $('spacing'),
  grid: $('grid'),
  closeness: $('closeness'),
  rotate: $('rotate'),
  freeze: $('freeze'),
};

// ---- Data --------------------------------------------------------------------------

const stream = new EegStream({ seconds: 16 });
const alpha = new AlphaFeatures(stream);// The slow alpha curves: one whole-head curve, or one per hemisphere.
const isAlphaCurve = () => controlsIn.curve.value.startsWith('alpha');
const isWholeHeadCurve = () => controlsIn.curve.value === 'alphaPeak';
const autoDelays = new Map(); // `${side}:${curve}` -> AutoDelay

let muse = null;

if (!navigator.bluetooth) {
  $('unsupported').hidden = false;
  connectButton.disabled = true;
}

// The button is the only status shown: its label and whether it can be clicked.
function setButton(label, { busy = false } = {}) {
  connectButton.textContent = label;
  connectButton.disabled = busy;
}

connectButton.addEventListener('click', async () => {
  if (muse) {
    setButton('Disconnecting…', { busy: true });
    const closing = muse;
    await closing.disconnect();
    // If the link had already dropped, no disconnect event comes; settle the button here.
    if (muse === closing && !closing.device?.gatt.connected) closing.onDisconnect();
    return;
  }
  stream.reset();
  alpha.reset();
  autoDelays.clear();
  const current = new Muse();
  muse = current;
  muse.onEeg = (ch, seq, samples) => stream.add(ch, seq, samples);
  muse.onDisconnect = () => {
    if (muse !== current) return; // a late event from an earlier connection
    muse = null;
    setButton('Connect');
  };
  try {
    setButton('Connecting…', { busy: true });
    await muse.connect();
    setButton('Disconnect');
  } catch (err) {
    console.error(err);
    muse = null;
    setButton('Connect');
  }
});

// A side's curve as a function of (fractional) sample index, in plot units: each axis
// divided by the scale and clamped to -1..1, so loud stretches flatten against the faces
// of a cube. Also returns the delay used and how far back a point reaches.
function sideCurve(side, now) {
  const range = stream.range(side);
  if (!range) return null;
  const embedding = EMBEDDINGS[controlsIn.curve.value];
  const xRaw = (t) => stream.read(side.xi, t);
  const yRaw = (t) => stream.read(side.yi, t);
  let tau;
  if (controlsIn.delay.value === 'auto') {
    const key = `${side.name}:${controlsIn.curve.value}`;
    if (!autoDelays.has(key)) autoDelays.set(key, new AutoDelay());
    const values = (ch) => stream.channels[ch].values;
    const signal = (i) => embedding.signal(values(side.xi)[i % stream.ring], values(side.yi)[i % stream.ring]);
    tau = autoDelays.get(key).update(signal, range.first, range.end, now);
  } else {
    tau = bandDelay(controlsIn.delay.value);
  }
  const raw = embedding.point(xRaw, yRaw, tau);
  const scale = Number(controlsIn.scale.value);
  const norm = (v) => Math.max(-1, Math.min(1, v / scale));
  const at = (t) => raw(t).map(norm);
  return { at, tau, lookback: embedding.span * tau, rate: EEG_SAMPLE_RATE, ...range };
}

// Sample indices along the trail, on a grid tied to the data (so points don't shimmer as
// the trail advances), ending exactly at the newest sample.
function trailSamples(start, end) {
  const step = Math.max(1, Math.ceil((end - start) / (MAX_POINTS - 2)));
  const samples = [];
  for (let i = Math.ceil(start / step) * step; i < end - 1; i += step) samples.push(i);
  samples.push(end - 1);
  return samples;
}

// How thick the trail is at each point: tapering in over its oldest part.
function taper(k, n) {
  const age = n > 1 ? k / (n - 1) : 1;
  const u = Math.min(1, age / TAPER);
  return u * u * (3 - 2 * u);
}

// ---- Geometry ----------------------------------------------------------------------

// A tube swept along a curve, its radius varying along it. Cross-sections follow
// parallel-transported frames, so the tube doesn't twist. Buffers are allocated once.
class Tube {
  constructor(material) {
    const rings = MAX_POINTS;
    const ringSize = RADIAL + 1; // the seam vertex is repeated for the texture
    this.positions = new Float32Array(rings * ringSize * 3);
    this.normals = new Float32Array(rings * ringSize * 3);
    this.uvs = new Float32Array(rings * ringSize * 2);
    const index = new Uint32Array((rings - 1) * RADIAL * 6);
    let n = 0;
    for (let i = 0; i < rings - 1; i++) {
      for (let j = 0; j < RADIAL; j++) {
        const a = i * ringSize + j;
        const b = a + ringSize;
        index.set([a, a + 1, b, b, a + 1, b + 1], n); // facing outwards
        n += 6;
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute('normal', new THREE.BufferAttribute(this.normals, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute('uv', new THREE.BufferAttribute(this.uvs, 2).setUsage(THREE.DynamicDrawUsage));
    geometry.setIndex(new THREE.BufferAttribute(index, 1));
    geometry.setDrawRange(0, 0);
    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.mesh.frustumCulled = false;
  }

  // points: flat [x, y, z, ...]; radii and u (texture coordinate along) per point.
  update(points, radii, u, count) {
    const { positions, normals, uvs } = this;
    const ringSize = RADIAL + 1;
    const p = (k) => [points[3 * k], points[3 * k + 1], points[3 * k + 2]];
    let tangent = [1, 0, 0];
    let normal = null;
    for (let k = 0; k < count; k++) {
      const ahead = p(Math.min(count - 1, k + 1));
      const behind = p(Math.max(0, k - 1));
      const d = sub(ahead, behind);
      const length = Math.hypot(...d);
      if (length > 1e-9) tangent = scaleVec(d, 1 / length); // else keep the last direction
      if (normal === null) normal = perpendicular(tangent);
      else {
        normal = sub(normal, scaleVec(tangent, dot(normal, tangent)));
        const n = Math.hypot(...normal);
        normal = n > 1e-6 ? scaleVec(normal, 1 / n) : perpendicular(tangent);
      }
      const binormal = cross(tangent, normal);
      const centre = p(k);
      for (let j = 0; j <= RADIAL; j++) {
        const angle = (2 * Math.PI * j) / RADIAL;
        const [c, s] = [Math.cos(angle), Math.sin(angle)];
        const out = [0, 1, 2].map((a) => c * normal[a] + s * binormal[a]);
        const v = (k * ringSize + j) * 3;
        for (let a = 0; a < 3; a++) {
          positions[v + a] = centre[a] + radii[k] * out[a];
          normals[v + a] = out[a];
        }
        uvs[(k * ringSize + j) * 2] = u[k];
        uvs[(k * ringSize + j) * 2 + 1] = j / RADIAL;
      }
    }
    this.commit(Math.max(0, count - 1) * RADIAL * 6);
  }

  commit(indexCount) {
    const geometry = this.mesh.geometry;
    geometry.setDrawRange(0, indexCount);
    for (const name of ['position', 'normal', 'uv']) geometry.attributes[name].needsUpdate = true;
  }
}

// A ruled shell lofted between two curves a and b (same number of points): straight
// lines join a[k] to b[k]. Normals are computed from the surface's two directions.
class Shell {
  constructor(material) {
    const rows = MAX_POINTS;
    const rowSize = ACROSS + 1;
    this.positions = new Float32Array(rows * rowSize * 3);
    this.normals = new Float32Array(rows * rowSize * 3);
    this.uvs = new Float32Array(rows * rowSize * 2);
    const index = new Uint32Array((rows - 1) * ACROSS * 6);
    let n = 0;
    for (let i = 0; i < rows - 1; i++) {
      for (let j = 0; j < ACROSS; j++) {
        const a = i * rowSize + j;
        const b = a + rowSize;
        index.set([a, b, a + 1, b, b + 1, a + 1], n);
        n += 6;
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute('normal', new THREE.BufferAttribute(this.normals, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute('uv', new THREE.BufferAttribute(this.uvs, 2).setUsage(THREE.DynamicDrawUsage));
    geometry.setIndex(new THREE.BufferAttribute(index, 1));
    geometry.setDrawRange(0, 0);
    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.mesh.frustumCulled = false;
  }

  // u: texture coordinate along, per row; spans: texture repeats across, per row.
  update(a, b, u, spans, count) {
    const { positions, normals, uvs } = this;
    const rowSize = ACROSS + 1;
    const at = (arr, k) => [arr[3 * k], arr[3 * k + 1], arr[3 * k + 2]];
    let last = [0, 1, 0];
    for (let k = 0; k < count; k++) {
      const [a0, b0] = [at(a, k), at(b, k)];
      const k1 = Math.min(count - 1, k + 1);
      const k0 = Math.max(0, k - 1);
      const across = sub(b0, a0);
      for (let j = 0; j <= ACROSS; j++) {
        const f = j / ACROSS;
        const lerp = (k_) => {
          const [p, q] = [at(a, k_), at(b, k_)];
          return [0, 1, 2].map((i) => p[i] + f * (q[i] - p[i]));
        };
        const point = lerp(k);
        const along = sub(lerp(k1), lerp(k0));
        let normal = cross(along, across);
        const n = Math.hypot(...normal);
        normal = n > 1e-9 ? scaleVec(normal, 1 / n) : last;
        last = normal;
        const v = (k * rowSize + j) * 3;
        positions.set(point, v);
        normals.set(normal, v);
        uvs[(k * rowSize + j) * 2] = u[k];
        uvs[(k * rowSize + j) * 2 + 1] = f * spans[k];
      }
    }
    const geometry = this.mesh.geometry;
    geometry.setDrawRange(0, Math.max(0, count - 1) * ACROSS * 6);
    for (const name of ['position', 'normal', 'uv']) geometry.attributes[name].needsUpdate = true;
  }
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const scaleVec = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

// Any unit vector perpendicular to the unit vector t.
function perpendicular(t) {
  const axis = Math.abs(t[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const p = cross(t, axis);
  return scaleVec(p, 1 / Math.hypot(...p));
}

// ---- Materials ---------------------------------------------------------------------

// Tileable value noise, summed over octaves (fractal Brownian motion), in 0..1.
function fbmTexture(size, octaves, fill) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(size, size);
  const lattices = [];
  for (let o = 0; o < octaves; o++) {
    const cells = 4 << o;
    lattices.push({ cells, values: Float32Array.from({ length: cells * cells }, Math.random) });
  }
  const noise = ({ cells, values }, x, y) => {
    const [fx, fy] = [x * cells, y * cells];
    const [ix, iy] = [Math.floor(fx), Math.floor(fy)];
    const [tx, ty] = [fx - ix, fy - iy];
    const [sx, sy] = [tx * tx * (3 - 2 * tx), ty * ty * (3 - 2 * ty)];
    const v = (i, j) => values[((j % cells) * cells) + (i % cells)];
    const top = v(ix, iy) * (1 - sx) + v(ix + 1, iy) * sx;
    const bottom = v(ix, iy + 1) * (1 - sx) + v(ix + 1, iy + 1) * sx;
    return top * (1 - sy) + bottom * sy;
  };
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let sum = 0;
      let weight = 0;
      lattices.forEach((lattice, o) => {
        const w = 0.5 ** o;
        sum += w * noise(lattice, x / size, y / size);
        weight += w;
      });
      const [r, g, b] = fill(sum / weight, x / size, y / size);
      image.data.set([r, g, b, 255], (y * size + x) * 4);
    }
  }
  ctx.putImageData(image, 0, 0);
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

function makeMaterials() {
  // Marble: pale stone with soft grey veins following a turbulent stripe pattern.
  const marble = fbmTexture(512, 6, (n, x) => {
    const vein = Math.abs(Math.sin(2 * Math.PI * (2 * x + 3 * n)));
    const dark = 1 - Math.pow(vein, 0.18) * 0.92;
    const base = 236 + 14 * n;
    const c = base - 120 * dark;
    return [c, c * 0.985, c * 0.965];
  });
  marble.colorSpace = THREE.SRGBColorSpace;
  marble.repeat.set(0.3, 1); // broad veins: one repeat spans about three tube-widths along
  // Concrete: fine grain and pits, used for both colour and bump.
  const concrete = fbmTexture(512, 7, (n) => {
    const pit = Math.random() < 0.015 ? 0.55 : 1; // small scattered air holes
    const c = (112 + 80 * n) * pit;
    return [c, c * 0.98, c * 0.95];
  });
  concrete.colorSpace = THREE.SRGBColorSpace;
  // Bronze: patchy roughness, as in a hand-finished cast.
  const patina = fbmTexture(256, 5, (n) => {
    const c = 80 + 140 * n;
    return [c, c, c];
  });

  // White stone: plain colour; only a fine, even grain in the surface (bump), no pattern.
  const grain = fbmTexture(512, 3, (n) => {
    const v = 255 * (0.5 + 0.4 * (Math.random() - 0.5) + 0.1 * (n - 0.5));
    return [v, v, v];
  });

  const side = THREE.DoubleSide; // shells are seen from both faces
  return {
    marble: new THREE.MeshPhysicalMaterial({
      map: marble, roughness: 0.32, clearcoat: 0.5, clearcoatRoughness: 0.25, side,
    }),
    bronze: new THREE.MeshPhysicalMaterial({
      color: 0xb4804a, metalness: 1, roughness: 0.38, roughnessMap: patina, side,
    }),
    // The same cast finish as bronze, in a white metal.
    whiteBronze: new THREE.MeshPhysicalMaterial({
      color: 0xe4e4e4, metalness: 1, roughness: 0.38, roughnessMap: patina, side,
    }),
    chrome: new THREE.MeshPhysicalMaterial({ color: 0xffffff, metalness: 1, roughness: 0.04, side }),
    concrete: new THREE.MeshStandardMaterial({
      map: concrete, bumpMap: concrete, bumpScale: 1, roughness: 0.95, side,
    }),
    // Matte white stone, like limestone or plaster: no gloss, no pattern.
    stone: new THREE.MeshStandardMaterial({
      color: 0xe9e7e2, roughness: 0.92, bumpMap: grain, bumpScale: 0.4, side,
    }),
  };
}

// ---- Scene -------------------------------------------------------------------------

const stage = $('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
stage.append(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.55;

const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
camera.position.set(0, 0.6, 4.4);
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.autoRotateSpeed = 1.2;
orbit.minDistance = 2;
orbit.maxDistance = 12;

// The sculpture floats in black. A key light casts shadows (the form onto itself), and a
// cooler light from behind picks out the rim.
const key = new THREE.DirectionalLight(0xfff4e8, 2.2);
key.position.set(3, 5, 4);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
Object.assign(key.shadow.camera, { left: -2.5, right: 2.5, top: 2.5, bottom: -2.5, near: 1, far: 15 });
key.shadow.bias = -0.0005;
key.shadow.normalBias = 0.02;
scene.add(key);
const rim = new THREE.DirectionalLight(0xdde6ff, 1.1);
rim.position.set(-4, 2, -3);
scene.add(rim);


const materials = makeMaterials();
const sculpture = new THREE.Group();
scene.add(sculpture);

// At zero thickness the tube becomes a bare line, in the material's colour, fading in
// over the trail's oldest part as the tube tapers.
const LINE_COLORS = { marble: 0xf0ece4, bronze: 0xc89058, whiteBronze: 0xe6e6e6, chrome: 0xffffff, concrete: 0xa09c95, stone: 0xe9e7e2 };
const lineMaterial = new THREE.LineBasicMaterial({ vertexColors: true });

class Line {
  constructor() {
    this.positions = new Float32Array(MAX_POINTS * 3);
    this.colors = new Float32Array(MAX_POINTS * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute('color', new THREE.BufferAttribute(this.colors, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setDrawRange(0, 0);
    this.mesh = new THREE.Line(geometry, lineMaterial);
    this.mesh.frustumCulled = false;
  }

  update(points, fade, color, count) {
    this.positions.set(points.subarray(0, 3 * count));
    for (let k = 0; k < count; k++) {
      this.colors[3 * k] = color.r * fade[k];
      this.colors[3 * k + 1] = color.g * fade[k];
      this.colors[3 * k + 2] = color.b * fade[k];
    }
    const geometry = this.mesh.geometry;
    geometry.setDrawRange(0, count);
    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;
  }
}

// Separate straight lines (threads), each end with its own brightness, in the material's
// line colour.
const MAX_SEGMENTS = 4096;

class Segments {
  constructor() {
    this.positions = new Float32Array(MAX_SEGMENTS * 6);
    this.colors = new Float32Array(MAX_SEGMENTS * 6);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute('color', new THREE.BufferAttribute(this.colors, 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setDrawRange(0, 0);
    this.mesh = new THREE.LineSegments(geometry, lineMaterial);
    this.mesh.frustumCulled = false;
    this.count = 0;
  }

  begin(color) {
    this.count = 0;
    this.color = color;
  }

  get full() {
    return this.count >= MAX_SEGMENTS;
  }

  add(a, b, fadeA = 1, fadeB = 1) {
    if (this.full) return;
    const { r, g, b: blue } = this.color;
    const at = this.count * 6;
    this.positions.set(a, at);
    this.positions.set(b, at + 3);
    this.colors.set([r * fadeA, g * fadeA, blue * fadeA, r * fadeB, g * fadeB, blue * fadeB], at);
    this.count++;
  }

  end() {
    const geometry = this.mesh.geometry;
    geometry.setDrawRange(0, this.count * 2);
    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;
    this.mesh.visible = this.count > 0;
  }
}

// Metaballs: a field of soft balls along the trail whose surface (marching cubes) fuses
// wherever the trail passes close to itself. The field spans ±BLOB_SPAN plot radii, in a
// grid of `resolution` cells a side (the Grid menu; changing it rebuilds the metaballs).
const BLOB_SPAN = 1.4;
const MAX_BALLS = 400;

function makeBlobs(material, resolution = 40) {
  const maxTriangles = Math.round(40000 * (resolution / 40) ** 2); // surfaces grow with the grid
  const blobs = new MarchingCubes(resolution, material, true, false, maxTriangles);
  blobs.resolution = resolution;
  blobs.scale.setScalar(BLOB_SPAN);
  blobs.castShadow = true;
  blobs.receiveShadow = true;
  blobs.frustumCulled = false;
  return blobs;
}

// One set of pieces per side: a tube and its rounded head (or a line), a shell with its
// two rims, line segments, and metaballs.
const pieces = SIDES.map(() => {
  const tube = new Tube(materials.marble);
  const head = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), materials.marble);
  head.castShadow = true;
  const line = new Line();
  const shell = new Shell(materials.marble);
  const rims = [new Tube(materials.marble), new Tube(materials.marble)];
  const segments = new Segments();
  const blobs = makeBlobs(materials.marble);
  for (const mesh of [tube.mesh, head, line.mesh, shell.mesh, ...rims.map((r) => r.mesh), segments.mesh, blobs]) {
    sculpture.add(mesh);
  }
  return { tube, head, line, shell, rims, segments, blobs };
});

new ResizeObserver(() => {
  const { clientWidth: w, clientHeight: h } = stage;
  renderer.setSize(w, h);
  camera.aspect = w / Math.max(1, h);
  camera.updateProjectionMatrix();
}).observe(stage);

// ---- Building the sculpture each frame -----------------------------------------------

const scratch = {
  a: new Float32Array(MAX_POINTS * 3),
  b: new Float32Array(MAX_POINTS * 3),
  radii: new Float32Array(MAX_POINTS),
  u: new Float32Array(MAX_POINTS),
  shellU: new Float32Array(MAX_POINTS),
  width: new Float32Array(MAX_POINTS),
};
const rimScratch = [new Float32Array(MAX_POINTS * 3), new Float32Array(MAX_POINTS * 3)];

function hideAll() {
  for (const piece of pieces) {
    piece.tube.mesh.visible = false;
    piece.head.visible = false;
    piece.line.mesh.visible = false;
    piece.shell.mesh.visible = false;
    for (const r of piece.rims) r.mesh.visible = false;
    piece.segments.mesh.visible = false;
    piece.blobs.visible = false;
  }
}

// Distance along a curve, for texture coordinates that keep the material's grain the same
// size along a form as across it (so it isn't stretched into streaks). Distances are
// anchored to the data: each frame starts from the distance a sample already had last
// frame, so the material stays put on the form instead of sliding as the trail advances.
class ArcLength {
  constructor() {
    this.previous = new Map(); // sample index -> distance
  }

  // Writes the distance of each of the n points (flat [x, y, z, ...]) into out.
  measure(samples, points, n, out) {
    out[0] = 0;
    for (let k = 1; k < n; k++) {
      const [dx, dy, dz] = [0, 1, 2].map((c) => points[3 * k + c] - points[3 * (k - 1) + c]);
      out[k] = out[k - 1] + Math.hypot(dx, dy, dz);
    }
    let offset = 0;
    for (let k = 0; k < n; k++) {
      if (this.previous.has(samples[k])) {
        offset = this.previous.get(samples[k]) - out[k];
        break;
      }
    }
    offset -= Math.floor(offset / 4096) * 4096; // keep numbers small; textures repeat
    const next = new Map();
    for (let k = 0; k < n; k++) {
      out[k] += offset;
      next.set(samples[k], out[k]);
    }
    this.previous = next;
  }
}

const SHELL_TILE = 0.6; // a shell's texture repeats every this many plot radii

function buildTube(piece, curve, trail, thickness) {
  const start = Math.max(curve.end - trail, curve.first + Math.ceil(curve.lookback));
  if (curve.end - start < 3) return;
  const samples = trailSamples(start, curve.end);
  const n = samples.length;
  samples.forEach((i, k) => {
    scratch.a.set(curve.at(i), 3 * k);
    scratch.radii[k] = thickness * taper(k, n);
  });
  // One texture repeat around the tube, and the same distance per repeat along it.
  piece.arcs ??= { tube: new ArcLength(), shell: new ArcLength() };
  piece.arcs.tube.measure(samples, scratch.a, n, scratch.u);
  if (thickness > 0) for (let k = 0; k < n; k++) scratch.u[k] /= 2 * Math.PI * thickness;
  if (thickness === 0) {
    samples.forEach((_, k) => (scratch.radii[k] = taper(k, n)));
    piece.line.update(scratch.a, scratch.radii, new THREE.Color(LINE_COLORS[controlsIn.material.value]), n);
    piece.line.mesh.visible = true;
    return;
  }
  piece.tube.update(scratch.a, scratch.radii, scratch.u, n);
  piece.tube.mesh.visible = true;
  piece.head.position.fromArray(scratch.a, 3 * (n - 1));
  piece.head.scale.setScalar(thickness);
  piece.head.visible = true;
}

// A shell from curve a to curve b at the same samples (b given as a function of the
// sample index), narrowing to a line over its oldest part, with tubes along both edges
// unless the thickness is zero.
function buildShell(piece, samples, aAt, bAt, thickness) {
  const n = samples.length;
  if (n < 3) return;
  samples.forEach((i, k) => {
    const [p, q] = [aAt(i), bAt(i)];
    const w = taper(k, n);
    for (let c = 0; c < 3; c++) {
      const mid = (p[c] + q[c]) / 2;
      scratch.a[3 * k + c] = mid + (p[c] - mid) * w;
      scratch.b[3 * k + c] = mid + (q[c] - mid) * w;
    }
    scratch.radii[k] = RIM * thickness * w;
    // The shell's midline, for measuring distance along it; and its width.
    for (let c = 0; c < 3; c++) rimScratch[0][3 * k + c] = (scratch.a[3 * k + c] + scratch.b[3 * k + c]) / 2;
    scratch.width[k] = Math.hypot(...[0, 1, 2].map((c) => scratch.b[3 * k + c] - scratch.a[3 * k + c]));
  });
  piece.arcs ??= { tube: new ArcLength(), shell: new ArcLength() };
  piece.arcs.shell.measure(samples, rimScratch[0], n, scratch.u);
  for (let k = 0; k < n; k++) {
    scratch.shellU[k] = scratch.u[k] / SHELL_TILE;
    scratch.width[k] /= SHELL_TILE;
  }
  piece.shell.update(scratch.a, scratch.b, scratch.shellU, scratch.width, n);
  piece.shell.mesh.visible = true;
  if (thickness === 0) return;
  for (let k = 0; k < n; k++) scratch.u[k] /= 2 * Math.PI * RIM * thickness;
  rimScratch[0].set(scratch.a.subarray(0, 3 * n));
  rimScratch[1].set(scratch.b.subarray(0, 3 * n));
  piece.rims.forEach((r, e) => {
    r.update(rimScratch[e], scratch.radii, scratch.u, n);
    r.mesh.visible = true;
  });
}

// ---- Forms from the trail's points ---------------------------------------------------

// The trail's points (into scratch.a) and their sample indices, or null before data.
function trailPoints(curve, trail) {
  const start = Math.max(curve.end - trail, curve.first + Math.ceil(curve.lookback));
  if (curve.end - start < 4) return null;
  const samples = trailSamples(start, curve.end);
  samples.forEach((i, k) => scratch.a.set(curve.at(i), 3 * k));
  return samples;
}

const pointAt = (k) => [scratch.a[3 * k], scratch.a[3 * k + 1], scratch.a[3 * k + 2]];
const lineColor = () => new THREE.Color(LINE_COLORS[controlsIn.material.value]);

// The trail itself as a thin line, fading in over its oldest part.
function drawTrailLine(piece, n) {
  for (let k = 0; k < n; k++) scratch.radii[k] = taper(k, n);
  piece.line.update(scratch.a, scratch.radii, lineColor(), n);
  piece.line.mesh.visible = true;
}

// Shrink-wrap: metaballs along the trail, fusing into one skin where it passes close to
// itself. The Ball size slider sets their radius (plot radii); they shrink over the
// trail's oldest part. The Blend slider sets how far each ball's field reaches beyond its
// surface, in ball radii: about 1.3 keeps balls as separate beads, 3 melts neighbouring
// loops into one liquid skin. Balls are spaced evenly by distance along the curve (the
// Spacing slider, in ball radii: close spacing fuses into a smooth cord, wide spacing
// leaves beads), at multiples of that distance as measured by the anchored ArcLength, so
// they stay put on the curve as it advances instead of sliding. When there would be too
// many to compute in time (more balls, bigger balls and wider reach all cost more), the
// spacing doubles (keeping every other ball) rather than changing smoothly, which would
// make every ball shift.
const BLOB_BUDGET = 1.5e6; // field cells updated per side per frame, under 10 ms

function buildBlobs(piece, curve, trail) {
  const samples = trailPoints(curve, trail);
  if (!samples) return;
  const n = samples.length;
  piece.arcs ??= { tube: new ArcLength(), shell: new ArcLength() };
  piece.arcs.blobs ??= new ArcLength();
  const arc = scratch.u;
  piece.arcs.blobs.measure(samples, scratch.a, n, arc);

  const blobs = piece.blobs;
  const size = Number(controlsIn.ballSize.value);
  const radius = size / (2 * BLOB_SPAN); // in the field's 0..1 units
  // A ball's field is strength / d² - subtract, cut off at zero. Its surface sits where the
  // field equals the isolation level (at `radius`) and its field ends at reach × radius,
  // which gives subtract = isolation / (reach² - 1).
  const reach = Number(controlsIn.blend.value);
  const subtract = blobs.isolation / (reach * reach - 1);
  const strength = (blobs.isolation + subtract) * radius * radius;
  const ballCells = (2 * reach * radius * blobs.resolution) ** 3; // cells each ball touches
  const most = Math.min(MAX_BALLS, BLOB_BUDGET / ballCells);
  let spacing = Number(controlsIn.spacing.value) * size; // the Spacing slider, in ball radii
  while ((arc[n - 1] - arc[0]) / spacing > most) spacing *= 2;
  const addBall = (p, k) => {
    const w = taper(k, n);
    if (w < 0.02) return;
    const [x, y, z] = p.map((v) => (v / BLOB_SPAN + 1) / 2);
    blobs.addBall(x, y, z, strength * w, subtract);
  };
  blobs.reset();
  for (let k = 1; k < n; k++) {
    const [a, b] = [arc[k - 1], arc[k]];
    if (b <= a) continue;
    const [p, q] = [pointAt(k - 1), pointAt(k)];
    for (let m = Math.ceil(a / spacing) * spacing; m < b; m += spacing) {
      const f = (m - a) / (b - a);
      addBall([0, 1, 2].map((c) => p[c] + f * (q[c] - p[c])), k - 1 + f);
    }
  }
  addBall(pointAt(n - 1), n - 1); // the head
  blobs.update();
  blobs.visible = true;
}

// Recurrence threads: a thread wherever the trajectory comes back close to where it was
// at least half a cycle earlier (a recurrence plot, made physical). A steady rhythm laces
// its loops together; changing activity leaves few threads. How close counts as a return
// is the Closeness slider (plot radii).
function buildRecurrence(piece, curve, trail) {
  const samples = trailPoints(curve, trail);
  if (!samples) return;
  const n = samples.length;
  const step = n > 2 ? samples[1] - samples[0] : 1;
  const apart = Math.max(3, Math.ceil((2 * curve.tau) / step)); // half a cycle, in grid points
  const r2 = Number(controlsIn.closeness.value) ** 2;
  const p = scratch.a;
  drawTrailLine(piece, n);
  piece.segments.begin(lineColor());
  for (let k = n - 1; k >= 0 && !piece.segments.full; k--) {
    for (let j = k - apart; j >= 0; j--) {
      const dx = p[3 * k] - p[3 * j];
      const dy = p[3 * k + 1] - p[3 * j + 1];
      const dz = p[3 * k + 2] - p[3 * j + 2];
      if (dx * dx + dy * dy + dz * dz < r2) piece.segments.add(pointAt(k), pointAt(j), taper(k, n), taper(j, n));
    }
  }
  piece.segments.end();
}

function build(now) {
  const thickness = Number(controlsIn.thickness.value);
  const form = controlsIn.form.value;
  const sideChoice = controlsIn.side.value;
  hideAll();

  // A curve per side ([left, right]); the whole-head alpha curve fills only the first slot.
  // The alpha curves are sampled FEATURE_RATE times a second and have their own long trails.
  const wholeHead = isWholeHeadCurve();
  let curves;
  if (wholeHead) curves = [alpha.curve(), null];
  else if (isAlphaCurve()) curves = alpha.hemisphereCurves();
  else curves = SIDES.map((side) => sideCurve(side, now));
  const seconds = Number((isAlphaCurve() ? controlsIn.slowTrail : controlsIn.trail).value);
  const trailOf = (curve) => Math.round(seconds * curve.rate);

  if (form === 'hemispheres') {
    // One shell spanning the left curve and the right curve at each instant.
    const [left, right] = curves;
    if (!left || !right) return;
    const trail = trailOf(left);
    const end = Math.min(left.end, right.end);
    const first = Math.max(left.first + left.lookback, right.first + right.lookback);
    const start = Math.max(end - trail, Math.ceil(first));
    if (end - start < 3) return;
    buildShell(pieces[0], trailSamples(start, end), left.at, right.at, thickness);
    return;
  }

  SIDES.forEach((side, s) => {
    const curve = curves[s];
    if (!curve || (!wholeHead && sideChoice !== 'both' && sideChoice !== side.name)) return;
    const trail = trailOf(curve);
    if (form === 'tube') buildTube(pieces[s], curve, trail, thickness);
    else if (form === 'blobs') buildBlobs(pieces[s], curve, trail);
    else if (form === 'recurrence') buildRecurrence(pieces[s], curve, trail);
  });
}

// The Grid menu: rebuild each side's metaballs at the new resolution.
controlsIn.grid.addEventListener('change', () => {
  const resolution = Number(controlsIn.grid.value);
  for (const piece of pieces) {
    const old = piece.blobs;
    piece.blobs = makeBlobs(old.material, resolution);
    piece.blobs.visible = old.visible;
    sculpture.remove(old);
    old.geometry.dispose();
    sculpture.add(piece.blobs);
  }
});

function applyMaterial() {
  const material = materials[controlsIn.material.value];
  for (const mesh of sculpture.children) if (mesh.isMesh) mesh.material = material;
}
controlsIn.material.addEventListener('change', applyMaterial);
applyMaterial();

// The settings panel collapses to give the sculpture the whole window. Whether it was open
// is remembered for next time, where the browser allows it.
const PANEL_KEY = 'sculpture.panelCollapsed';
const panelToggle = $('panel-toggle');

function setPanelCollapsed(collapsed) {
  document.body.classList.toggle('collapsed', collapsed);
  panelToggle.textContent = collapsed ? '☰' : '›';
  panelToggle.title = collapsed ? 'Show settings' : 'Hide settings';
  panelToggle.setAttribute('aria-label', panelToggle.title);
  panelToggle.setAttribute('aria-expanded', String(!collapsed));
  try {
    localStorage.setItem(PANEL_KEY, collapsed ? '1' : '0');
  } catch {
    // Storage unavailable (private window, blocked site data): just don't remember.
  }
}

panelToggle.addEventListener('click', () => setPanelCollapsed(!document.body.classList.contains('collapsed')));
try {
  setPanelCollapsed(localStorage.getItem(PANEL_KEY) === '1');
} catch {
  setPanelCollapsed(false);
}

function updateControls() {
  // The alpha curves are already scaled and have no delay, so Scale and Delay don't apply;
  // they get the long trail menu. The whole-head one has no sides, so neither Side nor the
  // shell between hemispheres applies to it.
  const slow = isAlphaCurve();
  const wholeHead = isWholeHeadCurve();
  const hemispheres = controlsIn.form.querySelector('option[value="hemispheres"]');
  hemispheres.disabled = wholeHead;
  if (wholeHead && controlsIn.form.value === 'hemispheres') controlsIn.form.value = 'tube';
  const form = controlsIn.form.value;
  $('side-control').hidden = form === 'hemispheres' || wholeHead;
  $('scale-control').hidden = slow;
  $('delay-control').hidden = slow;
  $('trail-control').hidden = slow;
  $('slow-trail-control').hidden = !slow;
  $('thickness-control').hidden = form === 'blobs' || form === 'recurrence';
  $('ball-control').hidden = form !== 'blobs';
  $('blend-control').hidden = form !== 'blobs';
  $('spacing-control').hidden = form !== 'blobs';
  $('grid-control').hidden = form !== 'blobs';
  $('closeness-control').hidden = form !== 'recurrence';
  orbit.autoRotate = controlsIn.rotate.checked;
}
controlsIn.form.addEventListener('change', updateControls);
controlsIn.curve.addEventListener('change', updateControls);
controlsIn.rotate.addEventListener('change', updateControls);
updateControls();

function frame() {
  const now = performance.now();
  alpha.update(); // always, so the alpha history is there when its curves are chosen
  if (!controlsIn.freeze.checked) build(now);
  orbit.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ---- Saving ------------------------------------------------------------------------

// A copy of the visible sculpture with only the drawn part of each object, for export.
// Lines go into .glb only; .stl holds surfaces alone.
function snapshot() {
  const group = new THREE.Group();
  for (const object of sculpture.children) {
    if (!object.visible) continue;
    const source = object.geometry;
    const { count } = source.drawRange;
    const geometry = new THREE.BufferGeometry();
    if (source.index) {
      const used = Number.isFinite(count) ? count : source.index.count;
      if (used === 0) continue;
      geometry.setIndex(new THREE.BufferAttribute(source.index.array.slice(0, used), 1));
      for (const [name, attribute] of Object.entries(source.attributes)) geometry.setAttribute(name, attribute.clone());
    } else {
      const used = Number.isFinite(count) ? count : source.attributes.position.count;
      if (used === 0) continue;
      for (const [name, attribute] of Object.entries(source.attributes)) {
        const size = attribute.itemSize;
        geometry.setAttribute(name, new THREE.BufferAttribute(attribute.array.slice(0, used * size), size));
      }
    }
    const Kind = object.isLineSegments ? THREE.LineSegments : object.isLine ? THREE.Line : THREE.Mesh;
    const copy = new Kind(geometry, object.material);
    copy.position.copy(object.position);
    copy.quaternion.copy(object.quaternion);
    copy.scale.copy(object.scale);
    group.add(copy);
  }
  return group;
}

function download(data, name, type) {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const link = Object.assign(document.createElement('a'), { href: url, download: name });
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const stamp = () => new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

$('save-glb').addEventListener('click', () => {
  new GLTFExporter().parse(
    snapshot(),
    (glb) => download(glb, `sculpture-${stamp()}.glb`, 'model/gltf-binary'),
    (err) => console.error(err),
    { binary: true },
  );
});

$('save-stl').addEventListener('click', () => {
  const stl = new STLExporter().parse(snapshot(), { binary: true });
  download(stl, `sculpture-${stamp()}.stl`, 'model/stl');
});
