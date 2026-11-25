"""Avaliação "traço a traço": palavras escritas com fontes Hershey de traço único (cursivas),
passadas pelo MESMO pipeline do app (app_pipeline.py == preprocess.js) e pelo modelo ONNX.

As fontes Hershey nunca foram vistas no treino e, por serem vetoriais de traço único, imitam
bem uma palavra desenhada com o mouse.
Uso:  python training/eval_hershey.py [--onnx space/models/cursive_crnn.onnx] [--n 300]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
from HersheyFonts import HersheyFonts

try:
    from . import app_pipeline as ap
    from .models import ctc_greedy
    from .synth_cursive import load_lexicon
    from .train_crnn import cer_wer
except ImportError:
    import app_pipeline as ap
    from models import ctc_greedy
    from synth_cursive import load_lexicon
    from train_crnn import cer_wer

ROOT = Path(__file__).resolve().parents[1]
FONTS = ["scripts", "scriptc", "cursive"]


def hershey_strokes(word: str, font: str, rng: random.Random, jitter: float = 0.6):
    f = HersheyFonts()
    f.load_default_font(font)
    f.normalize_rendering(100)
    raw = [list(s) for s in f.strokes_for_text(word)]
    # junta segmentos contíguos num traço só (como a caneta que não levanta)
    strokes = []
    for s in raw:
        if strokes and np.hypot(strokes[-1][-1][0] - s[0][0], strokes[-1][-1][1] - s[0][1]) < 1e-6:
            strokes[-1].extend(s[1:])
        else:
            strokes.append(list(s))
    slant = rng.uniform(-0.15, 0.15)
    out = []
    for s in strokes:
        pts = []
        for (x0, y0), (x1, y1) in zip(s[:-1], s[1:]):  # densifica, como o pointermove
            n = max(1, int(np.hypot(x1 - x0, y1 - y0) / 2))
            for t in range(n):
                x, y = x0 + (x1 - x0) * t / n, y0 + (y1 - y0) * t / n
                pts.append([x + slant * y + rng.gauss(0, jitter), -y + rng.gauss(0, jitter), 0.5])
        pts.append([s[-1][0] + slant * s[-1][1], -s[-1][1], 0.5])
        out.append(pts)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default=str(ROOT / "space" / "models" / "cursive_crnn.onnx"))
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--save", default=str(ROOT / "artifacts" / "eval_hershey.json"))
    a = p.parse_args()
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    words, prob = load_lexicon(12)
    rng = random.Random(123)
    sample = [words[i] for i in np.random.default_rng(5).choice(len(words), a.n, p=prob, replace=False)]
    res = {}
    for font in FONTS:
        preds = []
        for w in sample:
            x = ap.cursive_from_strokes(hershey_strokes(w, font, rng))
            probs = sess.run(None, {"image": x[None, None]})[0][0]
            preds.append(ctc_greedy(probs)[0])
        cer, wer = cer_wer(preds, sample)
        res[font] = {"cer": cer, "wer": wer, "examples": list(zip(sample[:12], preds[:12]))}
        print(f"{font:8s} CER {cer:.3f}  WER {wer:.3f}  ex: " + ", ".join(f"{g}->{q}" for g, q in zip(sample[:6], preds[:6])))
    Path(a.save).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
