// Portrait: a live webcam image of the sitter against black, whose head is replaced by
// the surface between their hemispheres (the per-hemisphere alpha curves, lofted into a
// metal shell) following the head as it moves. Until there's enough brain data, or when
// no face is found, the real head stays.
//
// With depth estimated from the camera image (Depth Anything V2, in the browser on the
// GPU), the body is lit as relief by the same light as the shell, blurs with distance
// from the face (depth of field), fades into the dark, and anything clearly in front of
// the face (a raised hand) passes in front of the shell (occlusion).
//
// Faces and the person/hair/skin regions come from MediaPipe. Nothing leaves the
// machine. The scene is computed in unmirrored camera space and the canvas is flipped in
// CSS, so it reads as a mirror.
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { FilesetResolver, FaceLandmarker, ImageSegmenter } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
import { Muse } from './muse.js';
import { EegStream } from './eeg-stream.js';
import { AlphaFeatures, FEATURE_RATE } from './alpha-features.js';

const MAX_TRAIL_S = 60; // longest Trail setting
const SEGMENT_WIDTH = 512; // the segmenter sees a copy of each frame this wide (faster)
const MASK_SETTLE = 0.5; // each new mask blends this much into the last, so edges don't chatter
const DEPTH_SETTLE = 0.5; // likewise for each new depth map, so the relief doesn't shimmer
const OCCLUDE_MARGIN = 0.04; // how much nearer than the face (in depth units) occludes

const MEDIAPIPE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const SEGMENT_MODEL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite';
const TRANSFORMERS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';
const DEPTH_MODEL = 'onnx-community/depth-anything-v2-small';
// The segmenter's classes.
const HAIR = 1;
const FACE_SKIN = 3;
const ACCESSORY = 5; // glasses, hats, and probably the headband

const $ = (id) => document.getElementById(id);
const settings = {
  get trail() { return Number($('trail').value); },
  get material() { return $('material').value; },
  get headSize() { return Number($('head-size').value); },
  get headLift() { return Number($('head-lift').value); },
  get colour() { return Number($('colour').value); },
  get depthOn() { return $('depth-on').checked; },
  get depthSize() { return Number($('depth-size').value); },
  get relief() { return Number($('relief').value); },
  get occlusion() { return $('occlusion').checked; },
  get reach() { return Number($('reach').value); },
  get dof() { return Number($('dof').value); },
  get fade() { return Number($('fade').value); },
  get showDepth() { return $('show-depth').checked; },
};

function setButton(button, label, busy = false) {
  button.textContent = label;
  button.disabled = busy;
}

// ---- Brain data ----------------------------------------------------------------------

const stream = new EegStream({ seconds: 16 });
const alpha = new AlphaFeatures(stream);
const connectButton = $('connect');
let muse = null;

if (!navigator.bluetooth) {
  $('unsupported').hidden = false;
  connectButton.disabled = true;
}

connectButton.addEventListener('click', async () => {
  if (muse) {
    setButton(connectButton, 'Disconnecting…', true);
    const closing = muse;
    await closing.disconnect();
    if (muse === closing && !closing.device?.gatt.connected) closing.onDisconnect();
    return;
  }
  stream.reset();
  alpha.reset();
  const current = new Muse();
  muse = current;
  muse.onEeg = (ch, seq, samples) => stream.add(ch, seq, samples);
  muse.onDisconnect = () => {
    if (muse !== current) return;
    muse = null;
    stream.reset(); // no data: the real head comes back
    alpha.reset();
    setButton(connectButton, 'Connect');
  };
  try {
    setButton(connectButton, 'Connecting…', true);
    await muse.connect();
    setButton(connectButton, 'Disconnect');
  } catch (err) {
    console.error(err);
    muse = null;
    setButton(connectButton, 'Connect');
  }
});

// ---- The surface between hemispheres ---------------------------------------------------
// A ruled shell lofted between the left and right hemisphere curves at the same moments,
// narrowing to a line over its oldest part. Positions are in the curves' own space
// (-1..1 on each axis); the head group scales and turns them onto the head.

const MAX_ROWS = Math.ceil(MAX_TRAIL_S * FEATURE_RATE) + 2;
const ACROSS = 6;
const TAPER = 0.3;

function taper(k, n) {
  const u = Math.min(1, (n > 1 ? k / (n - 1) : 1) / TAPER);
  return u * u * (3 - 2 * u);
}

class Shell {
  constructor(material) {
    const rowSize = ACROSS + 1;
    this.positions = new Float32Array(MAX_ROWS * rowSize * 3);
    this.normals = new Float32Array(MAX_ROWS * rowSize * 3);
    this.uvs = new Float32Array(MAX_ROWS * rowSize * 2);
    const index = new Uint32Array((MAX_ROWS - 1) * ACROSS * 6);
    let n = 0;
    for (let i = 0; i < MAX_ROWS - 1; i++) {
      for (let j = 0; j < ACROSS; j++) {
        const a = i * rowSize + j;
        const b = a + rowSize;
        index.set([a, b, a + 1, b, b + 1, a + 1], n);
        n += 6;
      }
    }
    const geometry = new THREE.BufferGeometry();
    for (const [name, array, size] of [['position', this.positions, 3], ['normal', this.normals, 3], ['uv', this.uvs, 2]]) {
      geometry.setAttribute(name, new THREE.BufferAttribute(array, size).setUsage(THREE.DynamicDrawUsage));
    }
    geometry.setIndex(new THREE.BufferAttribute(index, 1));
    geometry.setDrawRange(0, 0);
    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.frustumCulled = false;
  }

  // Lofts between curves a and b (functions of a measurement index) over the given rows.
  update(a, b, rows) {
    const n = rows.length;
    const rowSize = ACROSS + 1;
    const A = [];
    const B = [];
    rows.forEach((i, k) => {
      const [p, q] = [a(i), b(i)];
      const w = taper(k, n);
      const mid = p.map((v, c) => (v + q[c]) / 2);
      A.push(p.map((v, c) => mid[c] + (v - mid[c]) * w));
      B.push(q.map((v, c) => mid[c] + (v - mid[c]) * w));
    });
    const at = (k, f) => A[k].map((v, c) => v + f * (B[k][c] - v));
    let last = [0, 1, 0];
    for (let k = 0; k < n; k++) {
      const across = B[k].map((v, c) => v - A[k][c]);
      for (let j = 0; j <= ACROSS; j++) {
        const f = j / ACROSS;
        const p = at(k, f);
        const ahead = at(Math.min(n - 1, k + 1), f);
        const behind = at(Math.max(0, k - 1), f);
        const along = ahead.map((v, c) => v - behind[c]);
        let normal = [
          along[1] * across[2] - along[2] * across[1],
          along[2] * across[0] - along[0] * across[2],
          along[0] * across[1] - along[1] * across[0],
        ];
        const length = Math.hypot(...normal);
        normal = length > 1e-9 ? normal.map((v) => v / length) : last;
        last = normal;
        const v = (k * rowSize + j) * 3;
        this.positions.set(p, v);
        this.normals.set(normal, v);
        this.uvs[(k * rowSize + j) * 2] = k * 0.05;
        this.uvs[(k * rowSize + j) * 2 + 1] = f;
      }
    }
    const geometry = this.mesh.geometry;
    geometry.setDrawRange(0, Math.max(0, n - 1) * ACROSS * 6);
    for (const name of ['position', 'normal', 'uv']) geometry.attributes[name].needsUpdate = true;
    // Its bounding box, for fitting it to the head.
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n * rowSize * 3; i++) {
      const c = i % 3;
      min[c] = Math.min(min[c], this.positions[i]);
      max[c] = Math.max(max[c], this.positions[i]);
    }
    this.centre = [0, 1, 2].map((c) => (min[c] + max[c]) / 2);
    this.extent = Math.max(...[0, 1, 2].map((c) => max[c] - min[c])) / 2;
  }
}

// ---- Materials -------------------------------------------------------------------------

// A tileable greyscale texture: value noise summed over octaves (`fill` maps 0..1 to grey),
// or per-pixel grain when octaves is empty.
function noiseTexture(octaveCells, fill) {
  const size = 256;
  const canvas = Object.assign(document.createElement('canvas'), { width: size, height: size });
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(size, size);
  const octaves = octaveCells.map((cells) => ({ cells, values: Float32Array.from({ length: cells * cells }, Math.random) }));
  const smooth = (t) => t * t * (3 - 2 * t);
  const noise = ({ cells, values }, x, y) => {
    const [fx, fy] = [x * cells, y * cells];
    const [ix, iy] = [Math.floor(fx), Math.floor(fy)];
    const [sx, sy] = [smooth(fx - ix), smooth(fy - iy)];
    const v = (i, j) => values[(j % cells) * cells + (i % cells)];
    return (v(ix, iy) * (1 - sx) + v(ix + 1, iy) * sx) * (1 - sy) + (v(ix, iy + 1) * (1 - sx) + v(ix + 1, iy + 1) * sx) * sy;
  };
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let n = Math.random();
      if (octaves.length) {
        let sum = 0;
        let weight = 0;
        octaves.forEach((octave, o) => {
          sum += 0.5 ** o * noise(octave, x / size, y / size);
          weight += 0.5 ** o;
        });
        n = sum / weight;
      }
      const c = fill(n);
      image.data.set([c, c, c, 255], (y * size + x) * 4);
    }
  }
  ctx.putImageData(image, 0, 0);
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

const patina = noiseTexture([4, 8, 16, 32, 64], (n) => 80 + 140 * n); // a cast's patchy roughness
const grain = noiseTexture([], (n) => 128 + 50 * (n - 0.5)); // stone's fine grain
const side = THREE.DoubleSide;
const materials = {
  whiteBronze: new THREE.MeshPhysicalMaterial({ color: 0xe4e4e4, metalness: 1, roughness: 0.38, roughnessMap: patina, side }),
  stone: new THREE.MeshStandardMaterial({ color: 0xe9e7e2, roughness: 0.92, bumpMap: grain, bumpScale: 0.4, side }),
  chrome: new THREE.MeshPhysicalMaterial({ color: 0xffffff, metalness: 1, roughness: 0.04, side }),
  bronze: new THREE.MeshPhysicalMaterial({ color: 0xb4804a, metalness: 1, roughness: 0.38, roughnessMap: patina, side }),
};

// ---- Scene -----------------------------------------------------------------------------

const stage = $('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.autoClear = false;
stage.append(renderer.domElement);

const FOV = 40;
const DEPTH = 5; // the head sits at this distance from the camera
const camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 100);
const LIGHT = new THREE.Vector3(2, 4, 5).normalize(); // the key light, also lighting the body's relief

// Data textures for the masks and depth, re-created when their size changes.
function dataTexture(channels) {
  const format = channels === 2 ? THREE.RGFormat : THREE.RedFormat;
  const texture = new THREE.DataTexture(new Uint8Array(channels), 1, 1, format);
  texture.magFilter = texture.minFilter = THREE.LinearFilter;
  texture.unpackAlignment = 1; // rows needn't be 4-byte aligned
  return texture;
}
function resizeTexture(texture, data, width, height) {
  texture.dispose(); // its storage is sized once; a new size needs a new one
  texture.image = { data, width, height };
}

// Background: the camera image of the person (minus their head, once it's replaced) against
// black, lit as relief, blurred and faded by depth, and writing its depth where it's in
// front of the face, so it can hide the shell.
const video = Object.assign(document.createElement('video'), { muted: true, playsInline: true });
const videoTexture = new THREE.VideoTexture(video);
videoTexture.colorSpace = THREE.NoColorSpace; // passed straight through
const maskTexture = dataTexture(2);
const depthTexture = dataTexture(1);

const backdrop = new THREE.ShaderMaterial({
  uniforms: {
    videoMap: { value: videoTexture },
    maskMap: { value: maskTexture },
    depthMap: { value: depthTexture },
    cover: { value: new THREE.Vector4(1, 1, 0, 0) }, // video UV = screen UV * xy + zw
    videoAspect: { value: 16 / 9 },
    maskTexel: { value: new THREE.Vector2(1, 1) },
    depthTexel: { value: new THREE.Vector2(1, 1) },
    hideHead: { value: 0 },
    colour: { value: 0 },
    useDepth: { value: 0 },
    relief: { value: 0 },
    dof: { value: 0 },
    fade: { value: 0 },
    showDepth: { value: 0 },
    occlusion: { value: 0 },
    reach: { value: 3 },
    faceDepth: { value: 0.5 },
    headDistance: { value: DEPTH },
    projection: { value: new THREE.Vector2() }, // the camera's depth terms: z' = x·z + y, w' = -z
    light: { value: LIGHT },
  },
  vertexShader: `
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
  `,
  fragmentShader: `
    uniform sampler2D videoMap;
    uniform sampler2D maskMap;
    uniform sampler2D depthMap;
    uniform vec4 cover;
    uniform float videoAspect;
    uniform vec2 maskTexel;
    uniform vec2 depthTexel;
    uniform float hideHead, colour, useDepth, relief, dof, fade, showDepth, occlusion, reach, faceDepth, headDistance;
    uniform vec2 projection;
    uniform vec3 light;
    varying vec2 vUv;

    // The masks and depth map are stored top row first; the video bottom row first.
    float depthAt(vec2 uv) { return texture2D(depthMap, vec2(uv.x, 1.0 - uv.y)).r; }

    void main() {
      vec2 uv = vUv * cover.xy + cover.zw;
      if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        gl_FragDepth = 1.0;
        return;
      }
      vec3 rgb = texture2D(videoMap, uv).rgb;

      // The mask is coarser than the image, so refine it against the image itself (a joint
      // bilateral filter): average the mask over neighbouring mask pixels, weighting each
      // by how close it is and how similar its colour is to this pixel's. The edge then
      // follows the real outline of shoulders and hair rather than the mask's grid.
      vec2 muv = vec2(uv.x, 1.0 - uv.y);
      vec2 sum = vec2(0.0);
      float weights = 0.0;
      for (int j = -2; j <= 2; j++) {
        for (int i = -2; i <= 2; i++) {
          vec2 o = vec2(float(i), float(j));
          vec2 offset = o * maskTexel;
          vec3 d = texture2D(videoMap, uv + vec2(offset.x, -offset.y)).rgb - rgb;
          float w = exp(-dot(o, o) / 4.5 - dot(d, d) / 0.015);
          sum += w * texture2D(maskMap, muv + offset).rg;
          weights += w;
        }
      }
      vec2 mask = sum / weights; // person, head
      float person = smoothstep(0.3, 0.7, mask.r);
      // A lower threshold for the head, so at its edge with the background the head (cut
      // away) wins and no faint rim of hair is left behind.
      float headPart = smoothstep(0.08, 0.4, mask.g);
      float shown = person * (1.0 - hideHead * headPart);

      float depth = useDepth > 0.5 ? depthAt(uv) : faceDepth; // 1 is nearest
      float away = abs(depth - faceDepth);

      // Depth of field: the body blurs with distance from the face (16 samples on a disc).
      if (useDepth > 0.5 && dof > 0.0) {
        float radius = dof * clamp(away * 3.0, 0.0, 1.0) * 0.012; // in video heights
        if (radius > 0.0004) {
          vec3 blurred = rgb;
          for (int k = 0; k < 16; k++) {
            float angle = float(k) * 2.39996; // the golden angle spreads them evenly
            vec2 o = vec2(cos(angle) / videoAspect, sin(angle)) * radius * sqrt((float(k) + 0.5) / 16.0);
            blurred += texture2D(videoMap, uv + o).rgb;
          }
          rgb = blurred / 17.0;
        }
      }

      vec3 colourOut = mix(vec3(dot(rgb, vec3(0.299, 0.587, 0.114))), rgb, colour);

      if (useDepth > 0.5) {
        // Relief: light the body by the slope of its depth, from the shell's key light.
        float dx = depthAt(uv + vec2(depthTexel.x, 0.0)) - depthAt(uv - vec2(depthTexel.x, 0.0));
        float dy = depthAt(uv + vec2(0.0, depthTexel.y)) - depthAt(uv - vec2(0.0, depthTexel.y));
        vec3 normal = normalize(vec3(-dx * 8.0, -dy * 8.0, 1.0));
        float shade = 0.25 + 0.95 * max(dot(normal, light), 0.0);
        colourOut *= mix(1.0, shade, relief);
        // Fade: the farther parts of the body sink into the dark.
        shown *= mix(1.0, smoothstep(0.0, 0.7, depth), fade);
        if (showDepth > 0.5) colourOut = vec3(depth);
      }
      gl_FragColor = vec4(colourOut * shown, 1.0);

      // Occlusion: where the body is clearly nearer than the face (a raised hand), write
      // its depth, so the shell behind it is hidden. Elsewhere, write the far plane.
      float fragDepth = 1.0;
      if (occlusion > 0.5 && useDepth > 0.5 && shown > 0.5 && depth > faceDepth + ${OCCLUDE_MARGIN.toFixed(3)}) {
        float z = -(headDistance - reach * (depth - faceDepth));
        fragDepth = 0.5 * (projection.x * z + projection.y) / -z + 0.5;
      }
      gl_FragDepth = fragDepth;
    }
  `,
  depthTest: true,
  depthWrite: true,
  depthFunc: THREE.AlwaysDepth, // it sets the depth; nothing to test against yet
});
const backdropScene = new THREE.Scene();
backdropScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), backdrop));
const backdropCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

// Foreground: the surface, lit like the sculptures, on the head.
const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.55;
const key = new THREE.DirectionalLight(0xfff4e8, 2.2);
key.position.copy(LIGHT).multiplyScalar(6);
scene.add(key);
const rim = new THREE.DirectionalLight(0xdde6ff, 1.1);
rim.position.set(-4, 2, -3);
scene.add(rim);

const head = new THREE.Group();
const shell = new Shell(materials.whiteBronze);
head.add(shell.mesh);
head.visible = false;
scene.add(head);

function resize() {
  const { innerWidth: w, innerHeight: h } = window;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  const e = camera.projectionMatrix.elements; // column-major: [10] and [14] map z to depth
  backdrop.uniforms.projection.value.set(e[10], e[14]);
}
window.addEventListener('resize', resize);
resize();

// Where the video sits on screen: scaled to cover it, centred, cropping the overflow.
function coverFit() {
  const W = window.innerWidth;
  const H = window.innerHeight;
  const w = video.videoWidth || 1;
  const h = video.videoHeight || 1;
  const s = Math.max(W / w, H / h);
  return { W, H, s, x0: (W - w * s) / 2, y0: (H - h * s) / 2, w, h };
}

// A point in video coordinates (0..1, y down) as a world point at DEPTH.
function videoToWorld(u, v, fit) {
  const x = fit.x0 + u * fit.w * fit.s;
  const y = fit.y0 + v * fit.h * fit.s;
  const halfH = DEPTH * Math.tan(THREE.MathUtils.degToRad(FOV / 2));
  const halfW = halfH * camera.aspect;
  return new THREE.Vector3((2 * x / fit.W - 1) * halfW, (1 - 2 * y / fit.H) * halfH, -DEPTH);
}
const worldPerPixel = () => (2 * DEPTH * Math.tan(THREE.MathUtils.degToRad(FOV / 2))) / window.innerHeight;

// ---- Smoothing ---------------------------------------------------------------------------
// One Euro filter: steady when the head is still, quick when it moves.
class OneEuro {
  constructor(minCutoff = 1.2, beta = 0.02, dCutoff = 1) {
    Object.assign(this, { minCutoff, beta, dCutoff, x: null, dx: 0, t: null });
  }

  filter(value, t) {
    if (this.x === null) {
      this.x = value;
      this.t = t;
      return value;
    }
    const dt = Math.max(1e-3, (t - this.t) / 1000);
    this.t = t;
    const a = (cutoff) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
    const dx = (value - this.x) / dt;
    this.dx += a(this.dCutoff) * (dx - this.dx);
    this.x += a(this.minCutoff + this.beta * Math.abs(this.dx)) * (value - this.x);
    return this.x;
  }
}
const filters = { u: new OneEuro(), v: new OneEuro(), size: new OneEuro(0.8, 0.005) };
const smoothedTurn = new THREE.Quaternion();
let haveTurn = false;

// ---- Camera and vision --------------------------------------------------------------------

const cameraButton = $('camera');
let faceLandmarker = null;
let segmenter = null;
let face = null; // { u, v, height (video fraction), box: [u0, v0, u1, v1], turn: Quaternion }

async function loadVision() {
  const fileset = await FilesetResolver.forVisionTasks(MEDIAPIPE);
  [faceLandmarker, segmenter] = await Promise.all([
    FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: FACE_MODEL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFacialTransformationMatrixes: true,
    }),
    ImageSegmenter.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: SEGMENT_MODEL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      outputCategoryMask: false,
      outputConfidenceMasks: true, // soft 0-1 maps per class, for smooth edges
    }),
  ]);
}

cameraButton.addEventListener('click', async () => {
  try {
    setButton(cameraButton, 'Starting…', true);
    const [media] = await Promise.all([
      navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false }),
      faceLandmarker ? null : loadVision(),
    ]);
    video.srcObject = media;
    await video.play();
    setButton(cameraButton, 'Camera on', true);
    loadDepth(); // in the background; the portrait runs without depth until it's ready
  } catch (err) {
    console.error(err);
    setButton(cameraButton, 'Camera');
  }
});

// The face's position, size and turn from the landmarker.
function readFace(result) {
  const landmarks = result.faceLandmarks?.[0];
  if (!landmarks) return null;
  let [u0, v0, u1, v1] = [1, 1, 0, 0];
  for (const { x, y } of landmarks) {
    u0 = Math.min(u0, x);
    v0 = Math.min(v0, y);
    u1 = Math.max(u1, x);
    v1 = Math.max(v1, y);
  }
  // The transformation matrix: its layout is detected from where the translation sits
  // (a face is tens of centimetres in front of the camera, so |z| is large).
  const data = result.facialTransformationMatrixes?.[0]?.data;
  const turn = new THREE.Quaternion();
  if (data) {
    const m = new THREE.Matrix4().fromArray(data);
    if (Math.abs(data[11]) > Math.abs(data[14])) m.transpose(); // row-major after all
    m.decompose(new THREE.Vector3(), turn, new THREE.Vector3());
  }
  return { u: (u0 + u1) / 2, v: (v0 + v1) / 2, height: v1 - v0, box: [u0, v0, u1, v1], turn };
}

// Builds the person and head masks from the segmenter's per-class confidences (soft, 0..1):
// the person is anything not background; the head is hair and face skin near the face,
// plus accessories on it (glasses, a hat, the headband). Each new mask is blended into
// the last, so the edges settle instead of chattering frame to frame.
const masks = { width: 0, height: 0, data: null, settled: null };
function readMasks(confidence) {
  const { width, height } = confidence[0];
  const [background, hair, faceSkin, gear] = [0, HAIR, FACE_SKIN, ACCESSORY].map((c) => confidence[c].getAsFloat32Array());
  if (width !== masks.width || height !== masks.height) {
    Object.assign(masks, { width, height, data: new Uint8Array(width * height * 2), settled: new Float32Array(width * height * 2) });
    resizeTexture(maskTexture, masks.data, width, height);
    backdrop.uniforms.maskTexel.value.set(1 / width, 1 / height);
  }
  // Around the face (in mask pixels): generous for hair, tighter for accessories.
  const near = (box, grow) => {
    const [u0, v0, u1, v1] = box;
    const [du, dv] = [(u1 - u0) * grow, (v1 - v0) * grow];
    return [(u0 - du) * width, (v0 - dv * 1.4) * height, (u1 + du) * width, (v1 + dv * 0.2) * height];
  };
  const hairBox = face ? near(face.box, 0.6) : null;
  const gearBox = face ? near(face.box, 0.25) : null;
  const inside = (x, y, b) => b !== null && x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3];
  const { data, settled } = masks;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const person = 1 - background[i];
      let headPart = inside(x, y, hairBox) ? hair[i] + faceSkin[i] : 0;
      if (inside(x, y, gearBox)) headPart += gear[i];
      settled[i * 2] += MASK_SETTLE * (person - settled[i * 2]);
      settled[i * 2 + 1] += MASK_SETTLE * (Math.min(1, headPart) - settled[i * 2 + 1]);
      data[i * 2] = 255 * settled[i * 2];
      data[i * 2 + 1] = 255 * settled[i * 2 + 1];
    }
  }
  maskTexture.needsUpdate = true;
}

// ---- Depth ---------------------------------------------------------------------------------
// Depth Anything V2 (small) estimates relative depth from each image: nearer is larger.
// It runs as fast as it can, apart from drawing (each estimate waits for the last), on the
// GPU through WebGPU where the browser has it. Each estimate is normalised (its 2nd to 98th
// percentiles to 0..1) and blended into the last, so the relief doesn't shimmer.

const depth = {
  pipe: null, RawImage: null, busy: false, loading: false, size: 0,
  width: 0, height: 0, data: null, settled: null, values: null,
  ms: 0, rate: 0, count: 0, countedAt: 0, device: '',
};
const depthInput = document.createElement('canvas');
const depthContext = depthInput.getContext('2d', { willReadFrequently: true });
let faceDepth = 0.5; // the depth map's value at the face, smoothed

async function loadDepth() {
  if (depth.pipe || depth.loading) return;
  depth.loading = true;
  try {
    const { pipeline, RawImage } = await import(TRANSFORMERS);
    depth.RawImage = RawImage;
    const gpu = 'gpu' in navigator && (await navigator.gpu.requestAdapter().catch(() => null));
    const attempts = gpu ? [['webgpu', 'fp16'], ['webgpu', 'fp32'], ['wasm', 'q8']] : [['wasm', 'q8']];
    for (const [device, dtype] of attempts) {
      try {
        depth.pipe = await pipeline('depth-estimation', DEPTH_MODEL, { device, dtype });
        depth.device = `${device}, ${dtype}`;
        break;
      } catch (err) {
        console.warn(`Depth model on ${device} (${dtype}) failed`, err);
      }
    }
  } catch (err) {
    console.error(err);
  }
  depth.loading = false;
}

async function estimateDepth() {
  if (!depth.pipe || depth.busy || !settings.depthOn || video.readyState < 2) return;
  depth.busy = true;
  try {
    const size = settings.depthSize;
    if (size !== depth.size) {
      // The model's input size (a multiple of 14), keeping the frame's shape. (Its image
      // processor is `feature_extractor` in this version of Transformers.js.)
      const processor = depth.pipe.processor.image_processor ?? depth.pipe.processor.feature_extractor;
      processor.size = { width: size, height: size };
      depth.size = size;
    }
    depthInput.width = size;
    depthInput.height = Math.round((size * video.videoHeight) / video.videoWidth);
    depthContext.drawImage(video, 0, 0, depthInput.width, depthInput.height);
    const started = performance.now();
    const { predicted_depth: result } = await depth.pipe(depth.RawImage.fromCanvas(depthInput));
    const ms = performance.now() - started;
    depth.ms = depth.ms ? depth.ms + 0.2 * (ms - depth.ms) : ms;
    readDepth(result);
    depth.count++;
  } catch (err) {
    console.error(err);
  }
  depth.busy = false;
}

function readDepth(tensor) {
  const [height, width] = tensor.dims.slice(-2);
  const raw = tensor.data;
  if (width !== depth.width || height !== depth.height) {
    Object.assign(depth, { width, height, data: new Uint8Array(width * height), settled: null });
    resizeTexture(depthTexture, depth.data, width, height);
    backdrop.uniforms.depthTexel.value.set(1 / width, 1 / height);
  }
  // Normalise robustly: the 2nd and 98th percentiles of a sample of values become 0 and 1.
  const sample = [];
  for (let i = 0; i < raw.length; i += 7) sample.push(raw[i]);
  sample.sort((a, b) => a - b);
  const lo = sample[Math.floor(sample.length * 0.02)];
  const hi = sample[Math.floor(sample.length * 0.98)];
  const span = Math.max(1e-6, hi - lo);
  const fresh = !depth.settled;
  if (fresh) depth.settled = new Float32Array(width * height);
  for (let i = 0; i < raw.length; i++) {
    const v = Math.max(0, Math.min(1, (raw[i] - lo) / span));
    depth.settled[i] = fresh ? v : depth.settled[i] + DEPTH_SETTLE * (v - depth.settled[i]);
    depth.data[i] = 255 * depth.settled[i];
  }
  depthTexture.needsUpdate = true;
}

// The depth at the face (averaged over a small patch at its centre), smoothed.
function updateFaceDepth() {
  if (!face || !depth.settled) return;
  const cx = Math.round(face.u * depth.width);
  const cy = Math.round(face.v * depth.height);
  let sum = 0;
  let n = 0;
  for (let y = cy - 2; y <= cy + 2; y++) {
    for (let x = cx - 2; x <= cx + 2; x++) {
      if (x < 0 || y < 0 || x >= depth.width || y >= depth.height) continue;
      sum += depth.settled[y * depth.width + x];
      n++;
    }
  }
  if (n) faceDepth += 0.2 * (sum / n - faceDepth);
}

// ---- Each frame ------------------------------------------------------------------------

let lastVideoTime = -1;
const segmentInput = document.createElement('canvas');
const segmentContext = segmentInput.getContext('2d', { willReadFrequently: true });

function track(now) {
  if (!faceLandmarker || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  face = readFace(faceLandmarker.detectForVideo(video, now));
  // The segmenter works on a smaller copy of the frame: faster, and its masks are coarse
  // anyway (the shader refines their edges against the full image).
  segmentInput.width = SEGMENT_WIDTH;
  segmentInput.height = Math.round((SEGMENT_WIDTH * video.videoHeight) / video.videoWidth);
  segmentContext.drawImage(video, 0, 0, segmentInput.width, segmentInput.height);
  segmenter.segmentForVideo(segmentInput, now, (result) => {
    if (result.confidenceMasks) readMasks(result.confidenceMasks);
    for (const mask of result.confidenceMasks ?? []) mask.close();
  });
}

function placeHead(now, fit) {
  if (!face) {
    haveTurn = false;
    for (const f of Object.values(filters)) f.x = null;
    return false;
  }
  const u = filters.u.filter(face.u, now);
  const v = filters.v.filter(face.v, now);
  const faceHeight = filters.size.filter(face.height, now) * fit.h * fit.s * worldPerPixel();
  if (!haveTurn) smoothedTurn.copy(face.turn);
  else smoothedTurn.slerp(face.turn, 0.35);
  haveTurn = true;
  head.position.copy(videoToWorld(u, v, fit));
  head.quaternion.copy(smoothedTurn);
  head.scale.setScalar(settings.headSize * faceHeight);
  return true;
}

// The surface from the per-hemisphere alpha curves over the last Trail seconds.
function buildSurface() {
  const [left, right] = alpha.hemisphereCurves();
  if (!left || !right) return false;
  const end = Math.min(left.end, right.end);
  const start = Math.max(Math.max(left.first, right.first), end - Math.round(settings.trail * FEATURE_RATE));
  if (end - start < 4) return false;
  const rows = [];
  for (let i = start; i < end; i++) rows.push(i);
  shell.update(left.at, right.at, rows);
  fitSurface();
  return true;
}

// Wherever the surface sits in the curves' space, it's centred on the head and scaled to
// head size. Both ease over a couple of seconds, so it settles rather than jumps as the
// data moves. (fit.scale is how much the surface is magnified; it's capped so a nearly
// flat or tiny surface isn't blown up.)
const surfaceFit = { centre: null, scale: 1 };
const FIT_EASE = 0.03; // per frame
function fitSurface() {
  const target = 1 / Math.max(0.25, shell.extent);
  if (!surfaceFit.centre) {
    surfaceFit.centre = [...shell.centre];
    surfaceFit.scale = target;
  } else {
    for (let c = 0; c < 3; c++) surfaceFit.centre[c] += FIT_EASE * (shell.centre[c] - surfaceFit.centre[c]);
    surfaceFit.scale += FIT_EASE * (target - surfaceFit.scale);
  }
  const { centre, scale } = surfaceFit;
  shell.mesh.scale.setScalar(scale);
  shell.mesh.position.set(-centre[0] * scale, settings.headLift / settings.headSize - centre[1] * scale, -centre[2] * scale);
}

// Frames drawn, and depth estimates, per second, for the readout.
const stats = { frames: 0, since: performance.now(), fps: 0 };
function updateStats(now) {
  stats.frames++;
  if (now - stats.since < 1000) return;
  const seconds = (now - stats.since) / 1000;
  stats.fps = stats.frames / seconds;
  depth.rate = depth.count / seconds;
  stats.frames = 0;
  depth.count = 0;
  stats.since = now;
  let depthLine = 'Depth: off';
  if (settings.depthOn) {
    if (depth.loading) depthLine = 'Depth: loading model…';
    else if (!depth.pipe) depthLine = video.readyState >= 2 ? 'Depth: unavailable' : 'Depth: waiting for camera';
    else depthLine = `Depth: ${depth.rate.toFixed(1)} per s, ${Math.round(depth.ms)} ms each\n(${depth.device}, ${depth.width}×${depth.height})`;
  }
  $('stats').textContent = `Drawing: ${stats.fps.toFixed(0)} frames per s\n${depthLine}`;
}

function frame(now) {
  alpha.update();
  track(now);
  estimateDepth(); // starts an estimate if none is running; it finishes in the background
  updateFaceDepth();

  const fit = coverFit();
  const u = backdrop.uniforms;
  u.cover.value.set(fit.W / (fit.w * fit.s), fit.H / (fit.h * fit.s), -fit.x0 / (fit.w * fit.s), (fit.y0 + fit.h * fit.s - fit.H) / (fit.h * fit.s));
  u.videoAspect.value = fit.w / fit.h;
  const placed = placeHead(now, fit);
  const ready = placed && buildSurface();
  head.visible = ready;
  shell.mesh.material = materials[settings.material];
  const useDepth = settings.depthOn && depth.settled !== null;
  u.hideHead.value = ready ? 1 : 0; // keep the real head until the surface can replace it
  u.colour.value = settings.colour;
  u.useDepth.value = useDepth ? 1 : 0;
  u.relief.value = settings.relief;
  u.dof.value = settings.dof;
  u.fade.value = settings.fade;
  u.showDepth.value = settings.showDepth ? 1 : 0;
  u.occlusion.value = settings.occlusion ? 1 : 0;
  u.reach.value = settings.reach;
  u.faceDepth.value = faceDepth;

  renderer.clear();
  renderer.render(backdropScene, backdropCamera);
  renderer.render(scene, camera);
  updateStats(now);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ---- Interface ---------------------------------------------------------------------------
// The settings panel hides with the corner button (remembered between visits). The panel,
// its button and the cursor fade after a few still seconds; F toggles full screen.
const PANEL_KEY = 'portrait.panelCollapsed';
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
    // Storage unavailable: just don't remember.
  }
}
panelToggle.addEventListener('click', () => setPanelCollapsed(!document.body.classList.contains('collapsed')));
try {
  setPanelCollapsed(localStorage.getItem(PANEL_KEY) === '1');
} catch {
  setPanelCollapsed(false);
}

let idleTimer = null;
function wake() {
  document.body.classList.remove('idle');
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => document.body.classList.add('idle'), 3000);
}
window.addEventListener('pointermove', wake);
window.addEventListener('keydown', (event) => {
  wake();
  if ((event.key === 'f' || event.key === 'F') && !(event.target instanceof HTMLSelectElement)) {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen().catch((err) => console.error(err));
  }
});
wake();
