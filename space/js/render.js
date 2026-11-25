// Re-renderiza os traços (vetores) numa imagem de trabalho com escala e espessura normalizadas.
// Espelhado em training/app_pipeline.py (render_strokes): mudanças aqui devem ir para lá.

export const RENDER = {
  targetH: 112,       // altura da bbox dos pontos na imagem de trabalho
  margin: 24,
  cursiveWidth: 2.2 / 28, // espessura final ≈ 2,2 px quando a palavra vira 28 px de altura
  maxW: 3000,
};

export function strokesBBox(strokes) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const s of strokes) for (const [x, y] of s) {
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  return { x0, y0, x1, y1, w: x1 - x0, h: y1 - y0 };
}

/**
 * strokes: [[ [x,y,p], ... ], ...] em px CSS.
 * lineWidth: espessura em px da imagem de trabalho.
 * Retorna { imageData, width, height, toCss(xWork) } com tinta preta em fundo branco.
 */
export function renderStrokes(strokes, lineWidth) {
  const bb = strokesBBox(strokes);
  // a bbox vira targetH px; a espessura fica proporcional (igual ao treino, que normaliza pela bbox)
  const h = Math.max(bb.h, 8, bb.w / 25);
  let s = RENDER.targetH / h;
  if (bb.w * s > RENDER.maxW) s = RENDER.maxW / bb.w;
  const M = RENDER.margin;
  const W = Math.ceil(bb.w * s + 2 * M);
  const H = Math.ceil(bb.h * s + 2 * M);
  const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(W, H) : Object.assign(document.createElement("canvas"), { width: W, height: H });
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = "#000";
  ctx.fillStyle = "#000";
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const tx = (x) => (x - bb.x0) * s + M;
  const ty = (y) => (y - bb.y0) * s + M;
  for (const st of strokes) {
    if (st.length === 1) {
      ctx.beginPath();
      ctx.arc(tx(st[0][0]), ty(st[0][1]), lineWidth / 2, 0, Math.PI * 2);
      ctx.fill();
      continue;
    }
    ctx.beginPath();
    ctx.moveTo(tx(st[0][0]), ty(st[0][1]));
    for (let i = 1; i < st.length; i++) ctx.lineTo(tx(st[i][0]), ty(st[i][1]));
    ctx.stroke();
  }
  return {
    canvas,
    imageData: ctx.getImageData(0, 0, W, H),
    width: W,
    height: H,
    scale: s,
    toCssX: (xw) => (xw - M) / s + bb.x0,
    toCssY: (yw) => (yw - M) / s + bb.y0,
  };
}

export function cursiveLineWidth() {
  return RENDER.targetH * RENDER.cursiveWidth; // 8,8 px -> ~2 px depois de normalizar para 28 px
}
