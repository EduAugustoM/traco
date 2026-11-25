// Decodificação CTC: greedy com alinhamento (segmentação em letras) e beam search restrito a um léxico.

export const ALPHABET = "abcdefghijklmnopqrstuvwxyz"; // classe 0 = blank
export const FRAME_W = 4; // cada passo de tempo cobre 4 px da entrada 256

function topk(row, k, offset = 1) {
  const idx = [];
  for (let i = offset; i < row.length; i++) idx.push(i);
  idx.sort((a, b) => row[b] - row[a]);
  return idx.slice(0, k).map((i) => ({ ch: ALPHABET[i - 1], p: row[i] }));
}

/**
 * probs: Float32Array [T*K] (softmax). Retorna { text, spans } onde cada span é uma letra emitida:
 * { ch, t0, t1, tPeak, p, alts }. p = probabilidade máxima da letra nos quadros em que ela aparece.
 */
export function greedy(probs, T, K) {
  const spans = [];
  let prev = 0;
  for (let t = 0; t < T; t++) {
    const row = probs.subarray(t * K, (t + 1) * K);
    let best = 0;
    for (let k = 1; k < K; k++) if (row[k] > row[best]) best = k;
    if (best !== 0 && best !== prev) {
      spans.push({ ch: ALPHABET[best - 1], k: best, t0: t, t1: t, tPeak: t, p: row[best], alts: topk(row, 3) });
    } else if (best !== 0 && best === prev) {
      const s = spans[spans.length - 1];
      s.t1 = t;
      if (row[best] > s.p) { s.p = row[best]; s.tPeak = t; s.alts = topk(row, 3); }
    }
    prev = best;
  }
  return { text: spans.map((s) => s.ch).join(""), spans };
}

/**
 * Converte os picos da CTC em fronteiras de letras (x na entrada 256): cada letra vai do meio
 * entre o seu pico e o anterior até o meio entre o seu pico e o próximo.
 */
export function spanBounds(spans, x0, x1) {
  const centers = spans.map((s) => ((s.t0 + s.t1) / 2) * FRAME_W + FRAME_W / 2);
  return spans.map((s, i) => ({
    ...s,
    cx: centers[i],
    xa: i === 0 ? x0 : (centers[i - 1] + centers[i]) / 2,
    xb: i === spans.length - 1 ? x1 : (centers[i] + centers[i + 1]) / 2,
  }));
}

const logsum = (a, b) => (a === -Infinity ? b : b === -Infinity ? a : Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b))));

/**
 * Beam search CTC restrito a prefixos do léxico (trie). Pontuação final:
 * log P_ctc(palavra) + alpha * log P_lexico(palavra).
 * Retorna até `n` candidatos { word, score, logp }.
 */
export function lexiconBeam(probs, T, K, trie, { beam = 16, alpha = 0.25, n = 5, prune = 1e-4 } = {}) {
  let beams = new Map([["", { pb: 0, pnb: -Infinity, node: trie.root }]]);
  const logp = new Float32Array(probs.length);
  for (let i = 0; i < probs.length; i++) logp[i] = Math.log(Math.max(probs[i], 1e-12));
  for (let t = 0; t < T; t++) {
    const next = new Map();
    const get = (key, node) => {
      let b = next.get(key);
      if (!b) { b = { pb: -Infinity, pnb: -Infinity, node }; next.set(key, b); }
      return b;
    };
    const row = t * K;
    const cand = [];
    for (let k = 1; k < K; k++) if (probs[row + k] > prune) cand.push(k);
    for (const [prefix, b] of beams) {
      const total = logsum(b.pb, b.pnb);
      // blank
      const nb = get(prefix, b.node);
      nb.pb = logsum(nb.pb, total + logp[row]);
      const last = prefix.length ? ALPHABET.indexOf(prefix[prefix.length - 1]) + 1 : -1;
      for (const k of cand) {
        const ch = ALPHABET[k - 1];
        const lp = logp[row + k];
        if (k === last) {
          // repetição sem blank: mesma letra continua
          const same = get(prefix, b.node);
          same.pnb = logsum(same.pnb, b.pnb + lp);
          // letra dupla (ex.: "ss") exige um blank no meio
          const child = b.node.c[ch];
          if (child) {
            const ext = get(prefix + ch, child);
            ext.pnb = logsum(ext.pnb, b.pb + lp);
          }
        } else {
          const child = b.node.c[ch];
          if (!child) continue;
          const ext = get(prefix + ch, child);
          ext.pnb = logsum(ext.pnb, total + lp);
        }
      }
    }
    beams = new Map([...next.entries()]
      .sort((a, b) => logsum(b[1].pb, b[1].pnb) - logsum(a[1].pb, a[1].pnb))
      .slice(0, beam));
  }
  const out = [];
  for (const [word, b] of beams) {
    if (!word || b.node.w === undefined) continue;
    const lp = logsum(b.pb, b.pnb);
    out.push({ word, logp: lp, score: lp + alpha * b.node.w });
  }
  return out.sort((a, b) => b.score - a.score).slice(0, n);
}

/** Log-probabilidade CTC de uma palavra específica (forward algorithm). */
export function ctcLogProb(probs, T, K, word) {
  const lab = [0];
  for (const ch of word) { lab.push(ALPHABET.indexOf(ch) + 1, 0); }
  const S = lab.length;
  let a = new Float64Array(S).fill(-Infinity);
  const lp = (t, k) => Math.log(Math.max(probs[t * K + k], 1e-12));
  a[0] = lp(0, lab[0]);
  if (S > 1) a[1] = lp(0, lab[1]);
  for (let t = 1; t < T; t++) {
    const b = new Float64Array(S).fill(-Infinity);
    for (let s = 0; s < S; s++) {
      let v = a[s];
      if (s > 0) v = logsum(v, a[s - 1]);
      if (s > 1 && lab[s] !== 0 && lab[s] !== lab[s - 2]) v = logsum(v, a[s - 2]);
      b[s] = v + lp(t, lab[s]);
    }
    a = b;
  }
  return S > 1 ? logsum(a[S - 1], a[S - 2]) : a[S - 1];
}
