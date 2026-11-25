// Pré-processamento com OpenCV.js, seguindo o enunciado:
// cinza -> GaussianBlur(5x5) -> threshold(BINARY_INV + OTSU) -> findContours(RETR_EXTERNAL) -> boundingRect.
// Espelhado em training/app_pipeline.py. Funciona no navegador e no Node (testes de paridade).

export const CRNN_H = 32, CRNN_W = 256, INNER_H = 28, PAD = 2;
export const EMNIST_SIDE = 22; // lado maior da letra no quadro 28x28 (EMNIST: ~23 px, mediana medida)

/** imageData RGBA -> { thresh: cv.Mat (0/255, tinta=255) }. Quem chama libera com .delete(). */
export function binarize(cv, imageData) {
  const src = cv.matFromImageData(imageData);
  const gray = new cv.Mat();
  const blur = new cv.Mat();
  const thresh = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0);
  cv.threshold(blur, thresh, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
  src.delete(); gray.delete(); blur.delete();
  return thresh;
}

/** Contornos externos -> [{x,y,w,h,area,idx}] e o vetor de contornos (liberar depois). */
export function findBoxes(cv, thresh) {
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  const tmp = thresh.clone();
  cv.findContours(tmp, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  const boxes = [];
  for (let i = 0; i < contours.size(); i++) {
    const c = contours.get(i);
    const r = cv.boundingRect(c);
    boxes.push({ x: r.x, y: r.y, w: r.width, h: r.height, area: cv.contourArea(c), idx: [i] });
    c.delete();
  }
  tmp.delete(); hierarchy.delete();
  return { boxes, contours };
}

/** Remove ruído (respingos muito pequenos em relação ao componente mais alto). Pingos de i/j ficam. */
export function dropNoise(boxes) {
  if (!boxes.length) return boxes;
  const maxH = Math.max(...boxes.map((b) => b.h));
  return boxes.filter((b) => Math.max(b.w, b.h) >= maxH * 0.04 && b.w * b.h >= 12);
}

export function unionBox(boxes) {
  const x0 = Math.min(...boxes.map((b) => b.x)), y0 = Math.min(...boxes.map((b) => b.y));
  const x1 = Math.max(...boxes.map((b) => b.x + b.w)), y1 = Math.max(...boxes.map((b) => b.y + b.h));
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0, idx: boxes.flatMap((b) => b.idx) };
}

/**
 * Letra de forma: agrupa componentes que se sobrepõem bastante no eixo x
 * (pingo do i/j, letras feitas em dois traços que não se tocam) e ordena da esquerda p/ direita.
 */
export function groupLetters(boxes) {
  const sorted = [...boxes].sort((a, b) => a.x - b.x);
  const groups = [];
  for (const b of sorted) {
    const g = groups[groups.length - 1];
    if (g) {
      const ov = Math.min(g.x + g.w, b.x + b.w) - Math.max(g.x, b.x);
      if (ov > 0.5 * Math.min(g.w, b.w)) {
        groups[groups.length - 1] = unionBox([g, b]);
        continue;
      }
    }
    groups.push({ ...b });
  }
  return groups;
}

/** Máscara só com os contornos do grupo (preenchidos) AND thresh: evita pedaços de vizinhas. */
function groupMask(cv, thresh, contours, group) {
  const mask = cv.Mat.zeros(thresh.rows, thresh.cols, cv.CV_8UC1);
  for (const i of group.idx) cv.drawContours(mask, contours, i, new cv.Scalar(255), -1);
  const out = new cv.Mat();
  cv.bitwise_and(thresh, mask, out);
  mask.delete();
  return out;
}

/**
 * Cursiva: recorte da palavra -> tensor Float32 [32*256] (altura útil 28 px, alinhado à esquerda).
 * Retorna também a escala para mapear quadros da CTC de volta para a imagem.
 */
export function wordTensor(cv, thresh, contours, group) {
  const masked = groupMask(cv, thresh, contours, group);
  const roi = masked.roi(new cv.Rect(group.x, group.y, group.w, group.h));
  const f = new cv.Mat();
  roi.convertTo(f, cv.CV_32F, 1 / 255);
  let nw = Math.max(1, Math.round((group.w * INNER_H) / group.h));
  nw = Math.min(nw, CRNN_W - 2 * PAD);
  const small = new cv.Mat();
  cv.resize(f, small, new cv.Size(nw, INNER_H), 0, 0, cv.INTER_AREA);
  const out = new Float32Array(CRNN_H * CRNN_W);
  const d = small.data32F;
  for (let y = 0; y < INNER_H; y++) {
    for (let x = 0; x < nw; x++) out[(y + PAD) * CRNN_W + x + PAD] = Math.min(1, Math.max(0, d[y * nw + x]));
  }
  masked.delete(); roi.delete(); f.delete(); small.delete();
  return { data: out, nw, sx: nw / group.w };
}

/** Letra de forma: recorte -> tensor 28x28 no padrão EMNIST (lado maior 22 px, centralizado). */
export function letterTensor(cv, thresh, contours, group) {
  const masked = groupMask(cv, thresh, contours, group);
  const roi = masked.roi(new cv.Rect(group.x, group.y, group.w, group.h));
  const f = new cv.Mat();
  roi.convertTo(f, cv.CV_32F, 1 / 255);
  const s = EMNIST_SIDE / Math.max(group.w, group.h);
  const tw = Math.max(1, Math.round(group.w * s)), th = Math.max(1, Math.round(group.h * s));
  const small = new cv.Mat();
  cv.resize(f, small, new cv.Size(tw, th), 0, 0, cv.INTER_AREA);
  const out = new Float32Array(28 * 28);
  const ox = Math.floor((28 - tw) / 2), oy = Math.floor((28 - th) / 2);
  const d = small.data32F;
  for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) out[(y + oy) * 28 + x + ox] = Math.min(1, Math.max(0, d[y * tw + x]));
  masked.delete(); roi.delete(); f.delete(); small.delete();
  return out;
}

/** Espessura de traço alvo p/ letra de forma: ~17% do lado maior típico da letra (EMNIST: 4/23). */
export function printLineWidth(groups) {
  if (!groups.length) return 12;
  const sides = groups.map((g) => Math.max(g.w, g.h)).sort((a, b) => a - b);
  const med = sides[Math.floor(sides.length / 2)];
  return Math.min(22, Math.max(6, 0.17 * med));
}
