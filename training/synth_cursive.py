"""Gerador de palavras cursivas sintéticas + normalização compartilhada com o app.

A normalização (`normalize_word`) é espelhada em space/js/preprocess.js: qualquer mudança
aqui precisa ser replicada lá (os testes de paridade em tests/ verificam isso).

Pipeline de uma amostra:
  fonte cursiva (Google Fonts) -> render em alta resolução -> slant/rotação/estiramento
  -> distorção elástica -> ESQUELETIZAÇÃO + redesenho com espessura uniforme (imita a caneta
  do canvas) -> cortes ocasionais de traço (caneta levantada) -> recorte + resize 32x256.
"""
from __future__ import annotations

import random
import string
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

try:
    from .models import CRNN_H, CRNN_W, CURSIVE_ALPHABET
except ImportError:
    from models import CRNN_H, CRNN_W, CURSIVE_ALPHABET

ROOT = Path(__file__).resolve().parents[1]
FONTS_DIR = ROOT / "training" / "fonts"
LEXICON = ROOT / "data" / "lexicon_pt_en.txt"

# Fontes reservadas para validação (o modelo nunca as vê no treino)
VAL_FONTS = {"allura", "courgette", "satisfy", "meaculpa", "norican"}
# Fontes descartadas após inspeção visual (letras desconectadas ou ilegíveis)
EXCLUDED_FONTS = {"comforterbrush", "hurricane", "herrvonmuellerhoff", "babylonica", "kristi", "waterbrush",
                  "puppiesplay", "ole", "licorice", "engagement"}

INNER_H = CRNN_H - 4  # 28 px de altura útil, 2 px de margem em cima/embaixo
PAD = 2


# ----------------------------------------------------------------------------- normalização
def normalize_word(ink: np.ndarray) -> np.ndarray:
    """ink: imagem binária/float (tinta>0) em qualquer resolução.
    Retorna float32 [32,256] em [0,1]: recorte pela bbox da tinta, altura 28 px mantendo a
    proporção (largura máx. 252 px, comprimindo se necessário), alinhado à esquerda."""
    ink = (ink > 0).astype(np.float32) if ink.dtype != np.float32 else ink
    ys, xs = np.nonzero(ink > 0)
    out = np.zeros((CRNN_H, CRNN_W), np.float32)
    if len(ys) == 0:
        return out
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    nw = max(1, int(round(w * INNER_H / h)))
    nw = min(nw, CRNN_W - 2 * PAD)
    img = cv2.resize(crop, (nw, INNER_H), interpolation=cv2.INTER_AREA)
    out[PAD:PAD + INNER_H, PAD:PAD + nw] = np.clip(img, 0, 1)
    return out


def encode(word: str) -> list[int]:
    return [CURSIVE_ALPHABET.index(c) + 1 for c in word]


# ----------------------------------------------------------------------------- assets
def list_fonts(split: str = "train") -> list[Path]:
    fonts = []
    for p in sorted(FONTS_DIR.glob("*.ttf")):
        fam = p.name.split("__")[0]
        if fam in EXCLUDED_FONTS:
            continue
        if (split == "val") == (fam in VAL_FONTS):
            fonts.append(p)
    return fonts


def load_lexicon(max_len: int = 14) -> tuple[list[str], np.ndarray]:
    words, freqs = [], []
    for line in LEXICON.read_text(encoding="utf-8").splitlines():
        w, c = line.split("\t")
        if 1 <= len(w) <= max_len:
            words.append(w)
            freqs.append(float(c))
    f = np.sqrt(np.array(freqs))  # suaviza: palavras raras também aparecem
    return words, f / f.sum()


@lru_cache(maxsize=256)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


# ----------------------------------------------------------------------------- render + aug
def render(word: str, font_path: str, size: int = 72) -> np.ndarray:
    font = _font(font_path, size)
    l, t, r, b = font.getbbox(word)
    m = size
    img = Image.new("L", (r - l + 2 * m, b - t + 2 * m), 0)
    ImageDraw.Draw(img).text((m - l, m - t), word, font=font, fill=255)
    return (np.asarray(img) > 48).astype(np.uint8)  # limiar baixo: preserva traços finos


def _affine(ink: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = ink.shape
    shear = rng.uniform(-0.45, 0.35)       # inclinação (slant)
    sx = rng.uniform(0.75, 1.35)           # estreita/alarga as letras
    ang = np.radians(rng.uniform(-4, 4))
    c, s = np.cos(ang), np.sin(ang)
    A = np.array([[c, -s], [s, c]]) @ np.array([[sx, shear], [0, 1]])
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float64) @ A.T
    mn, mx = corners.min(0), corners.max(0)
    M = np.hstack([A, -mn[:, None]]).astype(np.float32)
    size = (int(np.ceil(mx[0] - mn[0])) + 1, int(np.ceil(mx[1] - mn[1])) + 1)
    return cv2.warpAffine(ink, M, size, flags=cv2.INTER_NEAREST)


def _elastic(ink: np.ndarray, rng: random.Random, size: int) -> np.ndarray:
    h, w = ink.shape
    nrs = np.random.default_rng(rng.randrange(1 << 30))
    alpha = size * rng.uniform(0.04, 0.12)
    sigma = size * rng.uniform(0.25, 0.5)
    dx = cv2.GaussianBlur(nrs.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(nrs.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    dx *= alpha / (np.abs(dx).max() + 1e-6)
    dy *= alpha / (np.abs(dy).max() + 1e-6)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(ink, gx + dx, gy + dy, cv2.INTER_NEAREST)


def _crop(ink: np.ndarray, margin: int = 4) -> np.ndarray:
    ys, xs = np.nonzero(ink)
    if len(ys) == 0:
        return ink
    return ink[max(0, ys.min() - margin):ys.max() + margin + 1, max(0, xs.min() - margin):xs.max() + margin + 1]


def _restroke(ink: np.ndarray, rng: random.Random) -> np.ndarray:
    """Esqueletiza e redesenha com espessura uniforme. A espessura é sorteada no espaço FINAL
    (altura 28 px), por isso convertemos para a resolução atual."""
    ink = _crop(ink)
    h = max(1, int(np.ptp(np.nonzero(ink)[0])) + 1) if ink.any() else ink.shape[0]
    scale = INNER_H / h
    t_final = rng.uniform(1.1, 3.2)
    r = max(0.6, t_final / (2 * scale))
    if rng.random() < 0.85:
        sk = skeletonize(ink > 0).astype(np.uint8)
        dist = cv2.distanceTransform((1 - sk).astype(np.uint8), cv2.DIST_L2, 3)
        return (dist <= r).astype(np.uint8)
    # 15%: mantém o traço da fonte, só engrossando/afinando
    k = max(1, int(round(r * 0.6)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return cv2.dilate(ink, kernel) if rng.random() < 0.3 else ink


def _pen_lifts(ink: np.ndarray, rng: random.Random) -> np.ndarray:
    """Cortes verticais finos ocasionais: simula quem levanta a caneta entre letras."""
    if rng.random() > 0.25:
        return ink
    ink = ink.copy()
    h, w = ink.shape
    for _ in range(rng.randint(1, 3)):
        x = rng.randrange(w)
        cw = max(1, int(w * rng.uniform(0.004, 0.012)))
        band = ink[:, x:x + cw]
        ys = np.nonzero(band.any(1))[0]
        if len(ys):  # corta só a parte inferior (ligaduras ficam embaixo)
            y0 = int(ys.min() + (ys.max() - ys.min()) * 0.45)
            ink[y0:, x:x + cw] = 0
    return ink


def synth_sample(word: str, font_path: str, rng: random.Random, augment: bool = True) -> np.ndarray:
    size = rng.randint(56, 88) if augment else 72
    ink = render(word, font_path, size)
    if augment:
        ink = _affine(ink, rng)
        if rng.random() < 0.8:
            ink = _elastic(ink, rng, size)
        ink = _restroke(ink, rng)
        ink = _pen_lifts(ink, rng)
    else:
        ink = _restroke(ink, random.Random(0))
    x = normalize_word(ink.astype(np.float32))
    if augment and rng.random() < 0.15:
        x = cv2.GaussianBlur(x, (3, 3), rng.uniform(0.3, 0.8))
    return x


# ----------------------------------------------------------------------------- dataset
class CursiveDataset:
    """Dataset on-the-fly (compatível com torch DataLoader). Cada índice é determinístico
    dado (seed, epoch), então a validação é sempre a mesma."""

    def __init__(self, split: str = "train", n: int = 100_000, seed: int = 0, augment: bool = True,
                 random_frac: float = 0.2):
        self.fonts = [str(p) for p in list_fonts(split)]
        self.words, self.p = load_lexicon()
        self.cum = np.cumsum(self.p)
        self.n, self.seed, self.augment, self.random_frac = n, seed, augment, random_frac
        self.epoch = 0

    def __len__(self):
        return self.n

    def sample_word(self, rng: random.Random) -> str:
        r = rng.random()
        if r < self.random_frac:  # sequência aleatória: evita decorar o vocabulário
            return "".join(rng.choices(string.ascii_lowercase, k=rng.randint(1, 10)))
        idx = int(np.searchsorted(self.cum, rng.random() * self.cum[-1]))
        return self.words[min(idx, len(self.words) - 1)]

    def __getitem__(self, i):
        rng = random.Random(hash((self.seed, self.epoch, i)) & 0xFFFFFFFF)
        for _ in range(5):
            word = self.sample_word(rng)
            font = rng.choice(self.fonts)
            try:
                x = synth_sample(word, font, rng, self.augment)
                if x.any():
                    return x[None], np.array(encode(word), np.int64), word
            except Exception:  # noqa: BLE001 - glifo ausente / imagem vazia
                continue
        return np.zeros((1, CRNN_H, CRNN_W), np.float32), np.array(encode("a"), np.int64), "a"


def collate(batch):
    import torch

    xs, ys, words = zip(*batch)
    x = torch.from_numpy(np.stack(xs))
    targets = torch.from_numpy(np.concatenate(ys))
    lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    return x, targets, lengths, list(words)


def imwrite(path, img):
    """cv2.imwrite falha com acentos no caminho (Windows); usa imencode."""
    Path(path).write_bytes(cv2.imencode(Path(path).suffix, img)[1].tobytes())


def preview_grid(path: str | Path, n: int = 24, split: str = "train", seed: int = 1):
    """Salva uma grade de amostras para inspeção visual."""
    ds = CursiveDataset(split, n=n, seed=seed)
    rows = []
    for i in range(n):
        x, _, w = ds[i]
        img = (255 - x[0] * 255).astype(np.uint8)
        img = cv2.copyMakeBorder(img, 0, 14, 0, 0, cv2.BORDER_CONSTANT, value=255)
        cv2.putText(img, w, (2, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
        rows.append(cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=180))
    half = (len(rows) + 1) // 2
    left = np.vstack(rows[:half])
    right = np.vstack(rows[half:] + [np.full_like(rows[0], 255)] * (half - len(rows[half:])))
    imwrite(path, np.hstack([left, right]))


if __name__ == "__main__":
    import sys

    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    preview_grid(out / "synth_preview_train.png", split="train")
    preview_grid(out / "synth_preview_val.png", split="val")
    print("Fontes treino:", len(list_fonts("train")), "| validação:", len(list_fonts("val")))
    if "--fonts" in sys.argv:  # uma linha por fonte, para inspeção
        rows = []
        for p in sorted(FONTS_DIR.glob("*.ttf")):
            ink = render("minimum cursive", str(p), 48)
            img = (255 - normalize_word(_restroke(ink, random.Random(0)).astype(np.float32)) * 255).astype(np.uint8)
            img = cv2.copyMakeBorder(img, 0, 0, 0, 200, cv2.BORDER_CONSTANT, value=255)
            cv2.putText(img, p.name.split("__")[0], (262, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
            rows.append(img)
        imwrite(out / "fonts_sheet.png", np.vstack(rows))
