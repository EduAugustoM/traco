"""Espelho em Python do pipeline do app (space/js/render.js + preprocess.js).

Usado para: (1) avaliar as amostras reais coletadas no app; (2) gerar fixtures dos testes
de paridade JS x Python; (3) fine-tuning com amostras reais.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np

try:
    from .models import CRNN_H, CRNN_W
except ImportError:
    from models import CRNN_H, CRNN_W

INNER_H, PAD = CRNN_H - 4, 2
TARGET_H, MARGIN, MAX_W = 112, 24, 3000
CURSIVE_LW = TARGET_H * 2.2 / 28
EMNIST_SIDE = 22


# ----------------------------------------------------------------------------- render (render.js)
def render_strokes(strokes, line_width: float) -> np.ndarray:
    """strokes: lista de traços [[x,y,(p)],...] em px CSS -> imagem RGBA-like (H,W) uint8, tinta preta."""
    pts = np.array([p[:2] for s in strokes for p in s], np.float64)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    w, h = x1 - x0, y1 - y0
    hh = max(h, 8, w / 25)
    s = TARGET_H / hh
    if w * s > MAX_W:
        s = MAX_W / w
    W, H = int(np.ceil(w * s + 2 * MARGIN)), int(np.ceil(h * s + 2 * MARGIN))
    # desenha em 4x e reduz (antialias parecido com o canvas)
    k = 4
    img = np.full((H * k, W * k), 255, np.uint8)
    lw = max(1, int(round(line_width * k)))
    for st in strokes:
        p = np.array([[(q[0] - x0) * s + MARGIN, (q[1] - y0) * s + MARGIN] for q in st]) * k
        p = np.round(p).astype(np.int32)
        if len(p) == 1:
            cv2.circle(img, tuple(p[0]), lw // 2, 0, -1, cv2.LINE_AA)
        else:
            cv2.polylines(img, [p.reshape(-1, 1, 2)], False, 0, lw, cv2.LINE_AA)
            for q in (p[0], p[-1]):
                cv2.circle(img, tuple(q), lw // 2, 0, -1, cv2.LINE_AA)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------- preprocess.js
def binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def find_boxes(thresh):
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        boxes.append(dict(x=x, y=y, w=w, h=h, idx=[i]))
    return boxes, contours


def drop_noise(boxes):
    if not boxes:
        return boxes
    max_h = max(b["h"] for b in boxes)
    return [b for b in boxes if max(b["w"], b["h"]) >= max_h * 0.04 and b["w"] * b["h"] >= 12]


def union_box(boxes):
    x0 = min(b["x"] for b in boxes); y0 = min(b["y"] for b in boxes)
    x1 = max(b["x"] + b["w"] for b in boxes); y1 = max(b["y"] + b["h"] for b in boxes)
    return dict(x=x0, y=y0, w=x1 - x0, h=y1 - y0, idx=[i for b in boxes for i in b["idx"]])


def group_letters(boxes):
    groups = []
    for b in sorted(boxes, key=lambda b: b["x"]):
        if groups:
            g = groups[-1]
            ov = min(g["x"] + g["w"], b["x"] + b["w"]) - max(g["x"], b["x"])
            if ov > 0.5 * min(g["w"], b["w"]):
                groups[-1] = union_box([g, b])
                continue
        groups.append(dict(b))
    return groups


def _group_mask(thresh, contours, group):
    mask = np.zeros_like(thresh)
    for i in group["idx"]:
        cv2.drawContours(mask, contours, i, 255, -1)
    return cv2.bitwise_and(thresh, mask)


def word_tensor(thresh, contours, group) -> np.ndarray:
    m = _group_mask(thresh, contours, group)
    roi = m[group["y"]:group["y"] + group["h"], group["x"]:group["x"] + group["w"]].astype(np.float32) / 255
    nw = max(1, int(round(group["w"] * INNER_H / group["h"])))
    nw = min(nw, CRNN_W - 2 * PAD)
    small = cv2.resize(roi, (nw, INNER_H), interpolation=cv2.INTER_AREA)
    out = np.zeros((CRNN_H, CRNN_W), np.float32)
    out[PAD:PAD + INNER_H, PAD:PAD + nw] = np.clip(small, 0, 1)
    return out


def letter_tensor(thresh, contours, group) -> np.ndarray:
    m = _group_mask(thresh, contours, group)
    roi = m[group["y"]:group["y"] + group["h"], group["x"]:group["x"] + group["w"]].astype(np.float32) / 255
    s = EMNIST_SIDE / max(group["w"], group["h"])
    tw, th = max(1, int(round(group["w"] * s))), max(1, int(round(group["h"] * s)))
    small = cv2.resize(roi, (tw, th), interpolation=cv2.INTER_AREA)
    out = np.zeros((28, 28), np.float32)
    ox, oy = (28 - tw) // 2, (28 - th) // 2
    out[oy:oy + th, ox:ox + tw] = np.clip(small, 0, 1)
    return out


def print_line_width(groups) -> float:
    if not groups:
        return 12.0
    sides = sorted(max(g["w"], g["h"]) for g in groups)
    return float(min(22, max(6, 0.17 * sides[len(sides) // 2])))


# ----------------------------------------------------------------------------- pipelines completos
def cursive_from_gray(gray: np.ndarray) -> np.ndarray:
    thresh = binarize(gray)
    boxes, contours = find_boxes(thresh)
    kept = drop_noise(boxes)
    return word_tensor(thresh, contours, union_box(kept))


def cursive_from_strokes(strokes) -> np.ndarray:
    return cursive_from_gray(render_strokes(strokes, CURSIVE_LW))


def print_from_strokes(strokes):
    g1 = render_strokes(strokes, 10)
    t1 = binarize(g1)
    b1, _ = find_boxes(t1)
    lw = print_line_width(group_letters(drop_noise(b1)))
    gray = render_strokes(strokes, lw)
    thresh = binarize(gray)
    boxes, contours = find_boxes(thresh)
    groups = group_letters(drop_noise(boxes))
    return np.stack([letter_tensor(thresh, contours, g) for g in groups]) if groups else np.zeros((0, 28, 28), np.float32)


def decode_app_input(b64: str) -> np.ndarray:
    """Tensor 32x256 salvo pelo modo de coleta do app (uint8 em base64)."""
    return np.frombuffer(base64.b64decode(b64), np.uint8).reshape(CRNN_H, CRNN_W).astype(np.float32) / 255
