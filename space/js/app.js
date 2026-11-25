import { InkPad } from "./canvas.js";
import { renderStrokes, cursiveLineWidth } from "./render.js";
import {
  binarize, findBoxes, dropNoise, unionBox, groupLetters, wordTensor, letterTensor, printLineWidth,
  CRNN_H, CRNN_W, PAD,
} from "./preprocess.js";
import { greedy, spanBounds, lexiconBeam, ctcLogProb } from "./ctc.js";
import { loadLexicon } from "./lexicon.js";
import { loadOpenCV, loadModel, runCursive, runPrint, MODELS } from "./models.js";

const $ = (id) => document.getElementById(id);
const els = {
  page: $("page"), pad: $("pad"), guides: $("guides"), marks: $("marks"),
  undo: $("undo"), clear: $("clear"), recognize: $("recognize"), status: $("status"), hint: $("hint"),
  word: $("word"), note: $("word-note"), letters: $("letters"), useLex: $("use-lexicon"),
  alts: $("alts"), altsList: $("alts-list"), options: $("options"),
  stRender: $("st-render"), stBinary: $("st-binary"), stInput: $("st-input"), stOutput: $("st-output"),
  stInputLabel: $("st-input-label"), stOutputLabel: $("st-output-label"), pipelineDesc: $("pipeline-desc"),
};

const state = {
  mode: "cursive",
  busy: false,
  stale: false,
  last: null,
  cvReady: false,
};

const LOW = 0.5; // abaixo disso a letra é marcada como incerta
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const pct = (p) => `${Math.round(p * 100)}%`;

// ------------------------------------------------------------------------------------------ coleta (declarado antes do setup para evitar TDZ)
const WORDS = [
  "casa", "bola", "gato", "livro", "escola", "amigo", "tempo", "verde", "janela", "cidade",
  "mundo", "feliz", "noite", "cursiva", "letra", "palavra", "caderno", "brasil", "sol", "agua",
  "futebol", "musica", "praia", "chuva", "flor", "maquina", "rede", "neural", "dados", "modelo",
  "hello", "world", "machine", "learning", "deep", "house", "water", "money", "people", "school",
  "light", "green", "quick", "brown", "fox", "jumps", "zebra", "queijo", "xadrez", "kiwi",
  "yoga", "whisky", "banana", "minimum", "summer", "little", "coffee", "apple", "pizza", "jazz",
  "amor", "vida", "terra", "fogo", "ar", "mesa", "porta", "caneta", "papel", "lapis",
  "quero", "hoje", "ontem", "sempre", "nunca", "juntos", "viagem", "exame", "professor", "aluno",
  "the", "and", "you", "mind", "night", "dream", "voice", "black", "white", "orange",
  "uva", "kilo", "wifi", "boxe", "ninja", "lua", "mar", "rio", "ilha", "trem",
];

const collect = {
  active: new URLSearchParams(location.search).has("coletar"),
  key: "traco-amostras-v1",
  section: $("collect"), word: $("collect-word"), save: $("collect-save"),
  skip: $("collect-skip"), exportBtn: $("collect-export"), count: $("collect-count"),
  samples: [],
  idx: 0,
  init() {
    if (!this.active) return;
    this.section.hidden = false;
    try { this.samples = JSON.parse(localStorage.getItem(this.key) || "[]"); } catch { this.samples = []; }
    this.idx = this.samples.length % WORDS.length;
    this.save.addEventListener("click", () => this.saveSample());
    this.skip.addEventListener("click", () => { this.idx = (this.idx + 1) % WORDS.length; this.render(); });
    this.exportBtn.addEventListener("click", () => this.download());
    this.render();
  },
  render() {
    this.word.textContent = WORDS[this.idx];
    this.count.textContent = `${this.samples.length} ${this.samples.length === 1 ? "amostra salva" : "amostras salvas"}`;
    this.exportBtn.disabled = !this.samples.length;
  },
  async saveSample() {
    if (pad.isEmpty || state.busy) return;
    const cv = await cvP;
    const inp = cursiveInput(cv, pad.strokes);
    inp.thresh.delete(); inp.contours.delete();
    const q = new Uint8Array(inp.data.length);
    for (let i = 0; i < q.length; i++) q[i] = Math.round(inp.data[i] * 255);
    let bin = "";
    for (let i = 0; i < q.length; i++) bin += String.fromCharCode(q[i]);
    this.samples.push({
      label: WORDS[this.idx], strokes: pad.snapshot(), w: Math.round(pad.width), h: Math.round(pad.height),
      input_u8_b64: btoa(bin), ts: new Date().toISOString(), ua: navigator.userAgent,
    });
    try { localStorage.setItem(this.key, JSON.stringify(this.samples)); } catch { setStatus("Armazenamento cheio: baixe as amostras.", true); }
    this.idx = (this.idx + 1) % WORDS.length;
    pad.clear();
    this.render();
    setStatus(`Amostra salva (${this.samples.length}). Próxima palavra: ${WORDS[this.idx]}.`);
  },
  download() {
    const blob = new Blob([JSON.stringify({ version: 1, app: "traco", samples: this.samples })], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `amostras_cursivas_${this.samples.length}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  },
};

// ------------------------------------------------------------------------------------------ setup
const pad = new InkPad({ ink: els.pad, guides: els.guides, page: els.page, onChange: onInkChange });
document.documentElement.dataset.mode = state.mode;
collect.init();

const cvP = loadOpenCV().then((cv) => { state.cvReady = true; updateButtons(); return cv; });
const cursiveP = loadModel("cursive");
Promise.all([cvP, cursiveP])
  .then(() => setStatus("Pronto. Escreva uma palavra."))
  .catch((e) => setStatus(`Não consegui carregar os modelos (${e.message}). Recarregue a página.`, true));
// a CNN do modo letra de forma carrega quando o navegador estiver ocioso.
// Atenção: requestIdleCallback recebe { timeout } (objeto), não um número como o setTimeout;
// passar 2500 lançava TypeError e abortava o módulo inteiro (botões e troca de modo sem eventos).
const preloadPrint = () => { try { loadModel("print").catch(() => {}); } catch { /* carrega sob demanda */ } };
if (window.requestIdleCallback) window.requestIdleCallback(preloadPrint, { timeout: 2500 });
else setTimeout(preloadPrint, 2500);

function setStatus(msg, error = false) {
  els.status.textContent = msg;
  els.status.classList.toggle("error", error);
}

function updateButtons() {
  const empty = pad.isEmpty;
  els.undo.disabled = empty || state.busy;
  els.clear.disabled = empty || state.busy;
  els.recognize.disabled = empty || state.busy;
  if (collect.active) collect.save.disabled = empty || state.busy || !state.cvReady;
}

function onInkChange(kind) {
  updateButtons();
  if (state.last) {
    state.stale = true;
    els.marks.innerHTML = "";
    els.word.style.opacity = "0.45";
    els.note.textContent = pad.isEmpty
      ? "Folha limpa. Escreva outra palavra."
      : "O desenho mudou. Toque em Reconhecer para ler de novo.";
  }
  if (kind === "clear") resetResult();
}

function resetResult() {
  state.last = null;
  state.stale = false;
  els.marks.innerHTML = "";
  els.word.textContent = "—";
  els.word.classList.add("is-empty");
  els.word.style.opacity = "";
  els.note.textContent = "O resultado aparece aqui depois de você escrever e tocar em Reconhecer.";
  els.letters.innerHTML = "";
  els.alts.hidden = true;
}

// ------------------------------------------------------------------------------------------ eventos
els.undo.addEventListener("click", () => pad.undo());
els.clear.addEventListener("click", () => pad.clear());
els.recognize.addEventListener("click", () => recognize());
els.useLex.addEventListener("change", () => { if (state.last && !state.stale) recognize(); });

document.addEventListener("keydown", (e) => {
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea") return;
  if (e.key === "Enter" && tag !== "button") { e.preventDefault(); recognize(); }
  else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); if (!pad.isEmpty) pad.undo(); }
  else if (e.key === "Escape") { if (!pad.isEmpty) pad.clear(); }
  else if (collect.active && e.key.toLowerCase() === "s" && !e.ctrlKey && !e.metaKey) { e.preventDefault(); collect.saveSample(); }
});

for (const btn of document.querySelectorAll(".mode-btn")) {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
  btn.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      const other = document.querySelector(`.mode-btn:not([data-mode="${btn.dataset.mode}"])`);
      other.focus();
      setMode(other.dataset.mode);
    }
  });
}

function setMode(mode) {
  if (mode === state.mode) return;
  state.mode = mode;
  document.documentElement.dataset.mode = mode;
  for (const b of document.querySelectorAll(".mode-btn")) {
    const on = b.dataset.mode === mode;
    b.setAttribute("aria-checked", String(on));
    b.tabIndex = on ? 0 : -1;
  }
  const cursive = mode === "cursive";
  els.hint.innerHTML = cursive
    ? "Escreva <strong>uma palavra em letras minúsculas</strong>, emendando as letras, entre as linhas azuis."
    : "Escreva <strong>uma palavra em letra de forma</strong>, com as letras separadas umas das outras.";
  els.options.hidden = !cursive;
  els.stInputLabel.innerHTML = cursive ? "<b>3.</b> Entrada da rede 32×256" : "<b>3.</b> Recortes 28×28 (padrão EMNIST)";
  els.stOutputLabel.innerHTML = cursive ? "<b>4.</b> Probabilidades por quadro (CTC)" : "<b>4.</b> Três classes mais prováveis por letra";
  els.pipelineDesc.textContent = cursive
    ? "Cada etapa na mesma escala: o traço é normalizado, binarizado com Otsu e entregue inteiro à CRNN; a CTC indica onde cada letra está."
    : "Cada etapa na mesma escala: o traço é binarizado com Otsu, os contornos separam as letras e cada recorte vai para a CNN treinada no EMNIST.";
  pad.drawGuides();
  pad.redraw();
  if (!pad.isEmpty && state.last) recognize();
  else resetResult();
}

// ------------------------------------------------------------------------------------------ reconhecimento
async function recognize() {
  if (pad.isEmpty || state.busy) return;
  state.busy = true;
  updateButtons();
  setStatus(state.cvReady ? "Lendo…" : "Carregando o OpenCV.js…");
  const t0 = performance.now();
  try {
    const cv = await cvP;
    const res = state.mode === "cursive" ? await readCursive(cv) : await readPrint(cv);
    state.last = res;
    state.stale = false;
    showResult(res);
    setStatus(`Lido em ${Math.round(performance.now() - t0)} ms, no seu navegador.`);
  } catch (e) {
    console.error(e);
    setStatus(`Algo deu errado ao ler o desenho: ${e.message}`, true);
  } finally {
    state.busy = false;
    updateButtons();
  }
}

/** Traços -> tensor da CRNN (usado também pela coleta de amostras). */
function cursiveInput(cv, strokes) {
  const r = renderStrokes(strokes, cursiveLineWidth());
  const thresh = binarize(cv, r.imageData);
  const { boxes, contours } = findBoxes(cv, thresh);
  const kept = dropNoise(boxes);
  if (!kept.length) { thresh.delete(); contours.delete(); throw new Error("nenhum traço encontrado"); }
  const group = unionBox(kept);
  const t = wordTensor(cv, thresh, contours, group);
  return { r, thresh, contours, group, kept, ...t };
}

async function readCursive(cv) {
  const inp = cursiveInput(cv, pad.strokes);
  const { r, thresh, contours, group, data, nw, sx } = inp;
  try {
    const { T, K } = MODELS.cursive;
    const probs = await runCursive(data);
    const g = greedy(probs, T, K);
    const toCss = (xn) => r.toCssX(group.x + (xn - PAD) / sx);
    const letters = spanBounds(g.spans, PAD, PAD + nw).map((s) => ({
      ch: s.ch, p: s.p, alts: s.alts.slice(1),
      xa: toCss(s.xa), xb: toCss(s.xb), cx: toCss(s.cx),
    }));
    let word = g.text, alts = [], corrected = false;
    if (els.useLex.checked && g.text) {
      const trie = await loadLexicon();
      const cands = lexiconBeam(probs, T, K, trie);
      const rawLp = ctcLogProb(probs, T, K, g.text);
      if (cands.length && !trie.has(g.text) && cands[0].logp > rawLp - 9) {
        word = cands[0].word;
        corrected = word !== g.text;
      }
      alts = cands.map((c) => c.word).filter((w) => w !== word).slice(0, 4);
      if (word !== g.text && !alts.includes(g.text)) alts.unshift(g.text);
    }
    drawStages({ r, thresh, rects: [group], input: { data, w: CRNN_W, h: CRNN_H }, probs, T, K });
    return {
      mode: "cursive", raw: g.text, word, corrected, alts, letters,
      bbox: { top: r.toCssY(group.y), bottom: r.toCssY(group.y + group.h) },
    };
  } finally {
    thresh.delete();
    contours.delete();
  }
}

async function readPrint(cv) {
  // 1ª passada: acha as letras para medir o tamanho típico e escolher a espessura do traço
  const r1 = renderStrokes(pad.strokes, 10);
  const t1 = binarize(cv, r1.imageData);
  const f1 = findBoxes(cv, t1);
  const lw = printLineWidth(groupLetters(dropNoise(f1.boxes)));
  t1.delete(); f1.contours.delete();
  // 2ª passada: espessura parecida com a do EMNIST (~17% do tamanho da letra)
  const r = renderStrokes(pad.strokes, lw);
  const thresh = binarize(cv, r.imageData);
  const { boxes, contours } = findBoxes(cv, thresh);
  try {
    const groups = groupLetters(dropNoise(boxes));
    if (!groups.length) throw new Error("nenhuma letra encontrada");
    const n = groups.length;
    const batch = new Float32Array(n * 784);
    groups.forEach((g, i) => batch.set(letterTensor(cv, thresh, contours, g), i * 784));
    const probs = await runPrint(batch, n);
    const C = MODELS.print.classes;
    const letters = groups.map((g, i) => {
      const row = Array.from(probs.subarray(i * C.length, (i + 1) * C.length));
      const order = row.map((p, k) => [p, k]).sort((a, b) => b[0] - a[0]).slice(0, 3);
      return {
        ch: C[order[0][1]], p: order[0][0], alts: order.slice(1).map(([p, k]) => ({ ch: C[k], p })),
        x0: r.toCssX(g.x), x1: r.toCssX(g.x + g.w), y0: r.toCssY(g.y), y1: r.toCssY(g.y + g.h),
      };
    });
    drawStages({ r, thresh, rects: groups, crops: { batch, n }, letters });
    const word = letters.map((l) => l.ch).join("");
    return { mode: "print", raw: word, word, corrected: false, alts: [], letters };
  } finally {
    thresh.delete();
    contours.delete();
  }
}

// ------------------------------------------------------------------------------------------ resultado
function showResult(res) {
  els.word.textContent = res.word || "?";
  els.word.classList.toggle("is-empty", !res.word);
  els.word.style.opacity = "";
  els.word.classList.remove("pop");
  void els.word.offsetWidth;
  els.word.classList.add("pop");

  if (res.mode === "cursive") {
    els.note.innerHTML = res.corrected
      ? `A rede leu <b>${res.raw}</b>; o dicionário corrigiu para a palavra mais próxima.`
      : els.useLex.checked
        ? "Leitura da rede (CRNN + CTC), conferida no dicionário."
        : "Leitura direta da rede (CRNN + CTC), sem dicionário.";
  } else {
    els.note.textContent = "Cada letra foi recortada pelos contornos (OpenCV) e classificada pela CNN treinada no EMNIST.";
  }

  els.letters.innerHTML = "";
  if (!res.letters.length) {
    els.letters.innerHTML = '<li class="letters-empty">Nenhuma letra encontrada. Tente escrever maior.</li>';
  }
  res.letters.forEach((l, i) => {
    const li = document.createElement("li");
    li.className = "letter" + (l.p < LOW ? " low" : "");
    li.style.setProperty("--delay", `${0.05 + i * 0.06}s`);
    const alt = l.alts.filter((a) => a.p >= 0.01).map((a) => `${a.ch} ${pct(a.p)}`).join(" · ");
    li.innerHTML = `<span class="ch">${l.ch}</span><span class="p">${pct(l.p)}</span>`
      + `<span class="bar"><i style="width:${Math.max(4, l.p * 100)}%"></i></span>`
      + (alt ? `<span class="alt">${alt}</span>` : "");
    li.setAttribute("aria-label", `Letra ${i + 1}: ${l.ch}, ${pct(l.p)}${alt ? `; alternativas ${alt}` : ""}`);
    els.letters.appendChild(li);
  });

  els.alts.hidden = !res.alts.length;
  els.altsList.innerHTML = "";
  for (const w of res.alts) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "alt-chip";
    b.textContent = w;
    b.addEventListener("click", () => { els.word.textContent = w; });
    els.altsList.appendChild(b);
  }
  drawMarks(res);
}

/** Marcas de correção em caneta vermelha sobre a folha: a mesma geometria dos chips. */
function drawMarks(res) {
  const svg = els.marks;
  const W = pad.width, H = pad.height;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const NS = "http://www.w3.org/2000/svg";
  const parts = [];
  const add = (tag, attrs, cls, delay, len) => {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    el.setAttribute("class", cls);
    el.style.setProperty("--delay", `${delay}s`);
    if (len) el.style.setProperty("--len", len);
    parts.push(el);
    return el;
  };
  const label = (x, y, l, delay) => {
    const t = add("text", { x, y }, "mark-label appear", delay);
    t.textContent = l.ch;
    const p = add("text", { x, y: y + 14 }, "mark-prob appear", delay);
    p.textContent = pct(l.p);
  };
  if (res.mode === "cursive") {
    const { top, bottom } = res.bbox;
    const labelY = bottom + 26 < H - 14 ? bottom + 24 : Math.max(18, top - 22);
    res.letters.forEach((l, i) => {
      const d = 0.08 + i * 0.07;
      if (i > 0) {
        const x = l.xa;
        add("line", { x1: x, y1: top - 4, x2: x, y2: bottom + 4 }, "mark-tick draw", d, bottom - top + 10);
      }
      if (l.p < LOW) {
        const rx = Math.max(10, (l.xb - l.xa) / 2), ry = (bottom - top) / 2 + 6;
        add("ellipse", { cx: (l.xa + l.xb) / 2, cy: (top + bottom) / 2, rx, ry }, "mark-circle draw", d + 0.1, 2 * Math.PI * Math.max(rx, ry));
      }
      label((l.xa + l.xb) / 2, labelY, l, d + 0.05);
    });
  } else {
    res.letters.forEach((l, i) => {
      const d = 0.08 + i * 0.07;
      const pad4 = 6;
      const w = l.x1 - l.x0 + 2 * pad4, h = l.y1 - l.y0 + 2 * pad4;
      add("rect", { x: l.x0 - pad4, y: l.y0 - pad4, width: w, height: h, rx: 8 }, "mark-box draw", d, 2 * (w + h));
      const below = l.y1 + 28 < H - 14;
      label((l.x0 + l.x1) / 2, below ? l.y1 + 26 : l.y0 - 24, l, d + 0.05);
    });
  }
  svg.replaceChildren(...parts);
}

// ------------------------------------------------------------------------------------------ pipeline
function prepCanvas(c) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = c.clientWidth || 240, h = c.clientHeight || 112;
  c.width = Math.round(w * dpr);
  c.height = Math.round(h * dpr);
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

/** Desenha `src` (canvas/imagem) contido na área, alinhado pela base (linha de base comum). */
function fit(ctx, src, sw, sh, w, h, smooth) {
  const m = 10;
  const s = Math.min((w - 2 * m) / sw, (h - 2 * m) / sh);
  const dw = sw * s, dh = sh * s;
  const x = (w - dw) / 2, y = h - m - dh;
  ctx.imageSmoothingEnabled = smooth;
  ctx.drawImage(src, x, y, dw, dh);
  return { x, y, s };
}

function tensorCanvas(data, w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = 1 - data[i];
    img.data[i * 4] = 28 + v * 224; img.data[i * 4 + 1] = 34 + v * 219; img.data[i * 4 + 2] = 48 + v * 207; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

function drawStages({ r, thresh, rects, input, probs, T, K, crops, letters }) {
  const red = css("--correct");
  // 1. traço normalizado
  { const { ctx, w, h } = prepCanvas(els.stRender); fit(ctx, r.canvas, r.width, r.height, w, h, true); }
  // 2. binarização (tinta escura sobre papel) + caixas dos contornos
  {
    const { ctx, w, h } = prepCanvas(els.stBinary);
    const c = document.createElement("canvas");
    c.width = thresh.cols; c.height = thresh.rows;
    const img = c.getContext("2d").createImageData(c.width, c.height);
    const d = thresh.data;
    for (let i = 0; i < d.length; i++) { const v = d[i] ? 28 : 252; img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v + 3; img.data[i * 4 + 3] = 255; }
    c.getContext("2d").putImageData(img, 0, 0);
    const { x, y, s } = fit(ctx, c, c.width, c.height, w, h, true);
    ctx.strokeStyle = red;
    ctx.lineWidth = 1.5;
    for (const b of rects) ctx.strokeRect(x + b.x * s, y + b.y * s, b.w * s, b.h * s);
  }
  // 3. entrada da rede
  {
    const { ctx, w, h } = prepCanvas(els.stInput);
    if (input) {
      fit(ctx, tensorCanvas(input.data, input.w, input.h), input.w, input.h, w, h, false);
    } else if (crops) {
      const strip = document.createElement("canvas");
      strip.width = crops.n * 30 - 2; strip.height = 28;
      const sctx = strip.getContext("2d");
      sctx.fillStyle = css("--sheet"); sctx.fillRect(0, 0, strip.width, 28);
      for (let i = 0; i < crops.n; i++) sctx.drawImage(tensorCanvas(crops.batch.subarray(i * 784, (i + 1) * 784), 28, 28), i * 30, 0);
      fit(ctx, strip, strip.width, 28, w, h, false);
    }
  }
  // 4. saída
  {
    const { ctx, w, h } = prepCanvas(els.stOutput);
    if (probs && T) {
      // mapa de calor: colunas = quadros de tempo, linhas = classes (blank em cima)
      const c = document.createElement("canvas");
      c.width = T; c.height = K;
      const cctx = c.getContext("2d");
      const img = cctx.createImageData(T, K);
      for (let t = 0; t < T; t++) for (let k = 0; k < K; k++) {
        const p = probs[t * K + k];
        const o = (k * T + t) * 4;
        const blank = k === 0;
        img.data[o] = blank ? 252 - p * 120 : 252 - p * 223;
        img.data[o + 1] = blank ? 252 - p * 110 : 252 - p * 189;
        img.data[o + 2] = blank ? 255 - p * 60 : 255 - p * 80;
        img.data[o + 3] = 255;
      }
      cctx.putImageData(img, 0, 0);
      fit(ctx, c, T * 4, K * 1.6, w, h, false);
    } else if (letters) {
      ctx.font = "600 12px Figtree, sans-serif";
      ctx.textBaseline = "middle";
      const colW = Math.min(64, (w - 16) / letters.length);
      letters.forEach((l, i) => {
        const x = 8 + i * colW;
        const top3 = [{ ch: l.ch, p: l.p }, ...l.alts];
        top3.forEach((a, j) => {
          const y = h - 14 - (2 - j) * 30;
          const bw = Math.max(2, (colW - 22) * a.p);
          ctx.fillStyle = j === 0 ? css("--ink") : "#9fb1d6";
          ctx.fillRect(x + 16, y - 5, bw, 10);
          ctx.fillStyle = css("--graphite");
          ctx.fillText(a.ch, x + 2, y);
        });
      });
    }
  }
}

updateButtons();
