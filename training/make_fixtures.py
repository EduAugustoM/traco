"""Gera fixtures em tests/fixtures/ para os testes de paridade Node (space/js) x Python (training)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort

try:
    from . import app_pipeline as ap
    from .eval_hershey import hershey_strokes
    from .models import ctc_greedy
except ImportError:
    import app_pipeline as ap
    from eval_hershey import hershey_strokes
    from models import ctc_greedy

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"


def main():
    FIX.mkdir(parents=True, exist_ok=True)
    sess = ort.InferenceSession(str(ROOT / "space" / "models" / "cursive_crnn.onnx"), providers=["CPUExecutionProvider"])
    cases = []
    rng = random.Random(42)
    for word, font in [("casa", "scripts"), ("machine", "scriptc"), ("minimum", "cursive"), ("escola", "scripts")]:
        st = hershey_strokes(word, font, rng)
        gray = ap.render_strokes(st, ap.CURSIVE_LW)
        h, w = gray.shape
        x = ap.cursive_from_gray(gray)
        probs = sess.run(None, {"image": x[None, None]})[0][0]
        pred, spans = ctc_greedy(probs)
        cases.append({
            "word": word,
            "font": font,
            "gray_w": int(w),
            "gray_h": int(h),
            "gray_u8": gray.flatten().tolist(),
            "tensor_f32": np.round(x.flatten(), 5).tolist(),
            "probs_f32": np.round(probs.flatten(), 5).tolist(),
            "pred": pred,
            "spans": spans,
        })
        print(f"fixture {word} ({font}): pred={pred}  W={w} H={h}")
    (FIX / "cursive_fixtures.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")


if __name__ == "__main__":
    main()
