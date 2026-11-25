// Teste de paridade JS x Python para ctc.js, lexicon.js e (com @techstark/opencv-js) preprocess.js.
// Uso: node tests/parity.test.mjs

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { greedy, spanBounds, lexiconBeam, ctcLogProb } from "../space/js/ctc.js";
import { Trie } from "../space/js/lexicon.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

const fixPath = path.join(__dirname, "fixtures", "cursive_fixtures.json");
const fix = JSON.parse(fs.readFileSync(fixPath, "utf8"));

// Carrega léxico real
const lexTxt = fs.readFileSync(path.join(ROOT, "space", "data", "lexicon.txt"), "utf8");
const trie = new Trie();
let total = 0;
const rows = [];
for (const line of lexTxt.split("\n")) {
  const [w, c] = line.split("\t");
  if (!w) continue;
  const n = Math.sqrt(Number(c) || 1);
  rows.push([w.trim(), n]);
  total += n;
}
for (const [w, n] of rows) trie.add(w, Math.log(n / total));
assert.ok(trie.size > 40000, `Léxico deve ter >40k palavras (tem ${trie.size})`);

let passed = 0;
for (const c of fix.cases) {
  const probs = new Float32Array(c.probs_f32);
  const g = greedy(probs, 64, 27);
  assert.equal(g.text, c.pred, `Greedy JS (${g.text}) != Python (${c.pred}) para ${c.word}`);

  // Verifica spans (letra, quadro inicial e final)
  assert.equal(g.spans.length, c.spans.length);
  for (let i = 0; i < g.spans.length; i++) {
    assert.equal(g.spans[i].ch, c.spans[i][0]);
    assert.equal(g.spans[i].t0, c.spans[i][1]);
    assert.equal(g.spans[i].t1, c.spans[i][2]);
  }

  // Fronteiras monotônicas dentro da faixa [2, 254]
  const bounds = spanBounds(g.spans, 2, 254);
  for (let i = 0; i < bounds.length; i++) {
    assert.ok(bounds[i].xa < bounds[i].xb, "xa < xb");
    if (i > 0) assert.ok(Math.abs(bounds[i].xa - bounds[i - 1].xb) < 1e-6, "fronteiras contíguas");
  }

  // Beam search com léxico recupera a palavra verdadeira no top-3
  const beam = lexiconBeam(probs, 64, 27, trie);
  assert.ok(beam.length > 0, "beam não vazio");
  const topWords = beam.map((b) => b.word);
  assert.ok(topWords.includes(c.word), `Beam [${topWords.join(", ")}] deveria conter '${c.word}'`);

  // ctcLogProb finito e razoável
  const lp = ctcLogProb(probs, 64, 27, c.word);
  assert.ok(Number.isFinite(lp) && lp > -50, `logProb(${c.word}) = ${lp}`);
  passed++;
}

console.log(`OK: ${passed}/${fix.cases.length} fixtures verificadas (greedy + spans + beam search + logProb).`);
