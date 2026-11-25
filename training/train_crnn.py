"""Treino da CRNN-CTC para palavras cursivas (dados sintéticos on-the-fly).

Uso:  python training/train_crnn.py --epochs 30 --samples 100000
Saídas em artifacts/: crnn_best.pt, crnn_last.pt, crnn_history.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .models import CRNN, ctc_greedy
    from .synth_cursive import CursiveDataset, collate
except ImportError:
    from models import CRNN, ctc_greedy
    from synth_cursive import CursiveDataset, collate

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer_wer(preds: list[str], gts: list[str]) -> tuple[float, float]:
    errs = sum(levenshtein(p, g) for p, g in zip(preds, gts))
    chars = sum(len(g) for g in gts)
    wer = float(np.mean([p != g for p, g in zip(preds, gts)]))
    return errs / max(1, chars), wer


@torch.no_grad()
def evaluate(model, loader, device="cuda"):
    model.eval()
    preds, gts = [], []
    for x, _, _, words in loader:
        probs = model(x.to(device)).exp().permute(1, 0, 2).cpu().numpy()  # [N,T,K]
        preds += [ctc_greedy(p)[0] for p in probs]
        gts += words
    return cer_wer(preds, gts), preds, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples", type=int, default=100_000, help="amostras por época")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    ART.mkdir(exist_ok=True)
    dev = "cuda"
    train_ds = CursiveDataset("train", n=args.samples, seed=1)
    val_seen = CursiveDataset("train", n=2000, seed=999, augment=True)   # fontes do treino, palavras novas
    val_unseen = CursiveDataset("val", n=2000, seed=777, augment=True)   # fontes nunca vistas
    kw = dict(batch_size=args.bs, num_workers=args.workers, collate_fn=collate, persistent_workers=True)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, prefetch_factor=4, **kw)
    vs_dl = DataLoader(val_seen, shuffle=False, **{**kw, "num_workers": 4})
    vu_dl = DataLoader(val_unseen, shuffle=False, **{**kw, "num_workers": 4})

    model = CRNN().to(dev)
    print(f"Parâmetros: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.08)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    hist = {"epoch": [], "train_loss": [], "cer_seen": [], "wer_seen": [], "cer_unseen": [], "wer_unseen": [], "lr": []}
    start_ep, best = 1, math.inf
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        hist, start_ep, best = ck["hist"], ck["epoch"] + 1, ck["best"]

    for ep in range(start_ep, args.epochs + 1):
        train_ds.epoch = ep
        model.train()
        t0, tl, n = time.time(), 0.0, 0
        for it, (x, tg, tl_len, _) in enumerate(train_dl):
            x, tg = x.to(dev, non_blocking=True), tg.to(dev)
            logp = model(x)  # [T,N,K]
            in_len = torch.full((x.size(0),), logp.size(0), dtype=torch.long)
            loss = ctc(logp.float(), tg, in_len, tl_len)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tl += loss.item(); n += 1
            if it % 200 == 0:
                print(f"  ép {ep} it {it}/{len(train_dl)} loss {tl / n:.3f} ({time.time() - t0:.0f}s)", flush=True)
        (cs, ws), _, _ = evaluate(model, vs_dl, dev)
        (cu, wu), pu, gu = evaluate(model, vu_dl, dev)
        for k, v in zip(hist, (ep, tl / n, cs, ws, cu, wu, sched.get_last_lr()[0])):
            hist[k].append(v)
        print(f"Época {ep:02d}  loss {tl / n:.4f}  CER vistas {cs:.4f} WER {ws:.4f} | "
              f"CER novas {cu:.4f} WER {wu:.4f}  ({time.time() - t0:.0f}s)  ex: "
              + ", ".join(f"{g}->{p}" for p, g in list(zip(pu, gu))[:4]), flush=True)
        if cu < best:
            best = cu
            torch.save(model.state_dict(), ART / "crnn_best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "hist": hist, "epoch": ep, "best": best}, ART / "crnn_last.pt")
        (ART / "crnn_history.json").write_text(json.dumps(hist, indent=1))
    print("Melhor CER (fontes novas):", best)


if __name__ == "__main__":
    main()
