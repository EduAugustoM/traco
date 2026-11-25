"""Exporta os modelos para ONNX (para o onnxruntime-web) e verifica a paridade com o PyTorch.

Uso:  python training/export_onnx.py [--crnn artifacts/crnn_best.pt] [--emnist artifacts/emnist_final.pt]
Saídas: space/models/emnist_cnn.onnx, space/models/cursive_crnn.onnx, space/models/labels.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

try:
    from .models import CRNN, CRNN_H, CRNN_W, CURSIVE_ALPHABET, EMNIST_CLASSES, CRNNExport, EmnistCNN, WithSoftmax
except ImportError:
    from models import CRNN, CRNN_H, CRNN_W, CURSIVE_ALPHABET, EMNIST_CLASSES, CRNNExport, EmnistCNN, WithSoftmax

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "space" / "models"


def _check(path: Path, torch_model, x: torch.Tensor) -> float:
    onnx.checker.check_model(onnx.load(str(path)))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    with torch.no_grad():
        ref = torch_model(x).numpy()
    return float(np.abs(got - ref).max())


def export_emnist(ckpt: Path, out: Path = OUT / "emnist_cnn.onnx") -> float:
    m = EmnistCNN(dropout=0.25)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m = WithSoftmax(m).eval()
    x = torch.rand(3, 1, 28, 28)
    torch.onnx.export(m, x, str(out), input_names=["image"], output_names=["probs"],
                      dynamic_axes={"image": {0: "n"}, "probs": {0: "n"}}, opset_version=17, dynamo=False)
    err = _check(out, m, torch.rand(5, 1, 28, 28))
    print(f"EMNIST -> {out.name} ({out.stat().st_size / 1e6:.1f} MB)  max|Δ| = {err:.2e}")
    return err


def export_crnn(ckpt: Path, out: Path = OUT / "cursive_crnn.onnx") -> float:
    m = CRNN()
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m = CRNNExport(m.eval()).eval()
    x = torch.rand(1, 1, CRNN_H, CRNN_W)
    torch.onnx.export(m, x, str(out), input_names=["image"], output_names=["probs"], opset_version=17, dynamo=False)
    err = _check(out, m, torch.rand(1, 1, CRNN_H, CRNN_W))
    print(f"CRNN   -> {out.name} ({out.stat().st_size / 1e6:.1f} MB)  max|Δ| = {err:.2e}")
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emnist", default=str(ROOT / "artifacts" / "emnist_final.pt"))
    ap.add_argument("--crnn", default=str(ROOT / "artifacts" / "crnn_best.pt"))
    ap.add_argument("--only", choices=["emnist", "crnn"], default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "emnist"):
        export_emnist(Path(a.emnist))
    if a.only in (None, "crnn"):
        export_crnn(Path(a.crnn))
    (OUT / "labels.json").write_text(json.dumps({
        "emnist_classes": EMNIST_CLASSES, "cursive_alphabet": CURSIVE_ALPHABET, "ctc_blank": 0,
        "crnn_input": [1, 1, CRNN_H, CRNN_W], "crnn_frames": CRNN_W // 4,
    }, indent=1))


if __name__ == "__main__":
    main()
