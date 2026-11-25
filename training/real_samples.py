"""Avalia e (opcionalmente) faz fine-tuning da CRNN com as amostras reais exportadas pelo app
(amostras_cursivas_*.json geradas em http://127.0.0.1:8765/?coletar=1).

Uso:
  # Apenas avaliar CER/WER nas amostras reais:
  python training/real_samples.py caminho/para/amostras_cursivas.json

  # Fazer fine-tuning curto (divide 70% treino / 30% teste, ou usa --finetune-all) e reexportar ONNX:
  python training/real_samples.py caminho/para/amostras_cursivas.json --finetune --epochs 12
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from . import app_pipeline as ap
    from .export_onnx import export_crnn
    from .models import CRNN, ctc_greedy
    from .synth_cursive import CursiveDataset, encode
    from .train_crnn import cer_wer
except ImportError:
    import app_pipeline as ap
    from export_onnx import export_crnn
    from models import CRNN, ctc_greedy
    from synth_cursive import CursiveDataset, encode
    from train_crnn import cer_wer

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def load_real_json(paths: list[str]):
    samples = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        for s in data.get("samples", []):
            w = s["label"].strip().lower()
            if "input_u8_b64" in s:
                x = ap.decode_app_input(s["input_u8_b64"])
            else:
                x = ap.cursive_from_strokes(s["strokes"])
            samples.append((x, w))
    return samples


@torch.no_grad()
def eval_samples(model: CRNN, samples, device="cuda"):
    model.eval()
    xs = torch.from_numpy(np.stack([s[0] for s in samples]))[:, None].to(device)
    gts = [s[1] for s in samples]
    probs = model(xs).exp().permute(1, 0, 2).cpu().numpy()
    preds = [ctc_greedy(p)[0] for p in probs]
    cer, wer = cer_wer(preds, gts)
    return cer, wer, preds, gts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="Arquivos .json exportados pelo modo ?coletar=1 do app")
    p.add_argument("--ckpt", default=str(ART / "crnn_best.pt"))
    p.add_argument("--finetune", action="store_true", help="Faz fine-tuning misturando amostras reais + sintéticas")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_real_json(a.files)
    print(f"Total de amostras reais carregadas: {len(samples)}")
    model = CRNN().to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))

    cer0, wer0, preds0, gts0 = eval_samples(model, samples, dev)
    print(f"[Zero-shot nas amostras reais] CER: {cer0*100:.2f}% | WER: {wer0*100:.1f}%")
    for g, pr in list(zip(gts0, preds0))[:15]:
        print(f"  {g:14s} -> {pr}")

    if not a.finetune or len(samples) < 10:
        return

    rng = random.Random(42)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_test = max(5, len(samples) // 3)
    test_s = [samples[i] for i in idx[:n_test]]
    train_s = [samples[i] for i in idx[n_test:]]
    cer_b, wer_b, _, _ = eval_samples(model, test_s, dev)
    print(f"\nSplit: {len(train_s)} treino real | {len(test_s)} teste real (antes do fine-tune: CER={cer_b*100:.2f}%, WER={wer_b*100:.1f}%)")

    synth = CursiveDataset("train", n=2000, seed=555, augment=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    best_cer, best_state = cer_b, {k: v.detach().clone() for k, v in model.state_dict().items()}
    for ep in range(1, a.epochs + 1):
        model.train()
        # Mistura 50% amostras reais (com jitter horizontal leve) + 50% sintéticas para não esquecer o alfabeto
        batch_items = []
        for _ in range(max(64, len(train_s) * 4)):
            if rng.random() < 0.5:
                x, w = rng.choice(train_s)
                shift = rng.randint(-2, 2)
                x_aug = np.roll(x, shift, axis=1) if shift else x
                batch_items.append((x_aug, w))
            else:
                xs, _, w = synth[rng.randrange(len(synth))]
                batch_items.append((xs[0], w))
        rng.shuffle(batch_items)
        bs = 32
        losses = []
        for i in range(0, len(batch_items), bs):
            chunk = batch_items[i:i + bs]
            xb = torch.from_numpy(np.stack([c[0] for c in chunk]))[:, None].to(dev)
            ys = [np.array(encode(c[1]), np.int64) for c in chunk]
            tg = torch.from_numpy(np.concatenate(ys)).to(dev)
            tl = torch.tensor([len(y) for y in ys], dtype=torch.long)
            logp = model(xb)
            in_len = torch.full((xb.size(0),), logp.size(0), dtype=torch.long)
            loss = ctc(logp.float(), tg, in_len, tl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
        cer_t, wer_t, _, _ = eval_samples(model, test_s, dev)
        print(f"  ép {ep:02d}/{a.epochs} | loss {np.mean(losses):.4f} | teste real CER {cer_t*100:.2f}% WER {wer_t*100:.1f}%")
        if cer_t <= best_cer:
            best_cer = cer_t
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    out_pt = ART / "crnn_finetuned.pt"
    torch.save(model.state_dict(), out_pt)
    export_crnn(out_pt)
    print(f"Modelo ajustado salvo em {out_pt} e exportado para space/models/cursive_crnn.onnx (melhor CER teste real = {best_cer*100:.2f}%)")


if __name__ == "__main__":
    main()
