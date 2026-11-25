// Léxico (português sem acentos + inglês) em uma trie, com log-frequência relativa por palavra.

export class Trie {
  constructor() { this.root = { c: {} }; this.size = 0; }
  add(word, w) {
    let n = this.root;
    for (const ch of word) n = n.c[ch] || (n.c[ch] = { c: {} });
    if (n.w === undefined) this.size++;
    n.w = n.w === undefined ? w : Math.max(n.w, w);
  }
  has(word) {
    let n = this.root;
    for (const ch of word) { n = n.c[ch]; if (!n) return false; }
    return n.w !== undefined;
  }
}

let cached = null;

/** Carrega `data/lexicon.txt` (palavra<TAB>contagem) uma única vez. */
export function loadLexicon(url = "data/lexicon.txt") {
  if (cached) return cached;
  cached = fetch(url)
    .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.text(); })
    .then((txt) => {
      const trie = new Trie();
      const lines = txt.split("\n");
      let total = 0;
      const rows = [];
      for (const line of lines) {
        const [w, c] = line.split("\t");
        if (!w) continue;
        const n = Math.sqrt(Number(c) || 1); // suaviza a frequência, igual ao treino
        rows.push([w.trim(), n]);
        total += n;
      }
      for (const [w, n] of rows) trie.add(w, Math.log(n / total));
      return trie;
    });
  return cached;
}
