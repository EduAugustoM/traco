// Carregamento do OpenCV.js e das sessões do onnxruntime-web (backend WASM, 100% no navegador).

const ORT_VERSION = "1.22.0";
const OPENCV_URLS = [
  "https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js",
  "https://docs.opencv.org/4.10.0/opencv.js",
];

export const MODELS = {
  cursive: { url: "models/cursive_crnn.onnx", T: 64, K: 27, input: [1, 1, 32, 256] },
  print: { url: "models/emnist_cnn.onnx", classes: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt" },
};

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.async = true;
    s.crossOrigin = "anonymous";
    s.onload = resolve;
    s.onerror = () => reject(new Error(`falha ao carregar ${src}`));
    document.head.appendChild(s);
  });
}

function waitCv(timeoutMs = 60000) {
  return new Promise((resolve, reject) => {
    const t0 = performance.now();
    const finish = (m) => {
      // Remove .then do objeto Emscripten: senão Promise.resolve(cv) entra em loop infinito!
      try { delete m.then; } catch { m.then = undefined; }
      window.cv = m;
      resolve(m);
    };
    const tick = () => {
      const cv = window.cv;
      if (cv && typeof cv.Mat === "function") return finish(cv);
      if (cv && typeof cv.then === "function" && !cv.__waiting) {
        cv.__waiting = true;
        cv.then((m) => { if (m && typeof m.Mat === "function") finish(m); });
      }
      if (performance.now() - t0 > timeoutMs) return reject(new Error("OpenCV.js não inicializou"));
      setTimeout(tick, 60);
    };
    tick();
  });
}

let cvPromise = null;
export function loadOpenCV() {
  if (cvPromise) return cvPromise;
  cvPromise = (async () => {
    let lastErr;
    for (const url of OPENCV_URLS) {
      try {
        await loadScript(url);
        return await waitCv();
      } catch (e) { lastErr = e; }
    }
    throw lastErr;
  })();
  return cvPromise;
}

const sessions = {};
export function loadModel(name) {
  if (sessions[name]) return sessions[name];
  const ort = window.ort;
  if (!ort) return Promise.reject(new Error("onnxruntime-web não carregou (verifique a conexão)"));
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  sessions[name] = ort.InferenceSession.create(MODELS[name].url, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  return sessions[name];
}

/** CRNN: Float32Array [32*256] -> Float32Array [T*K] (probabilidades). */
export async function runCursive(input) {
  const ort = window.ort;
  const sess = await loadModel("cursive");
  const feeds = { [sess.inputNames[0]]: new ort.Tensor("float32", input, MODELS.cursive.input) };
  const out = await sess.run(feeds);
  return out[sess.outputNames[0]].data;
}

/** CNN EMNIST: N letras 28x28 -> Float32Array [N*47] (probabilidades). */
export async function runPrint(batch, n) {
  const ort = window.ort;
  const sess = await loadModel("print");
  const feeds = { [sess.inputNames[0]]: new ort.Tensor("float32", batch, [n, 1, 28, 28]) };
  const out = await sess.run(feeds);
  return out[sess.outputNames[0]].data;
}
