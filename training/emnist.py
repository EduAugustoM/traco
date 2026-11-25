"""Treino e avaliação da CNN EMNIST (Balanced, 47 classes).

Uso direto:  python training/emnist.py   (treina as duas variantes e salva em artifacts/)
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .models import EMNIST_CLASSES, EmnistCNN
except ImportError:  # execução como script
    from models import EMNIST_CLASSES, EmnistCNN

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "artifacts"


def load_emnist(device="cuda"):
    """Carrega EMNIST Balanced como tensores [N,1,28,28] em [0,1] já TRANSPOSTOS
    (o EMNIST original vem espelhado/rotacionado; transpor aqui evita transpor no app)."""
    from torchvision.datasets import EMNIST

    out = {}
    for split, train in (("train", True), ("test", False)):
        ds = EMNIST(str(DATA / "emnist"), split="balanced", train=train, download=True)
        x = ds.data.unsqueeze(1).float().div(255.0).transpose(2, 3).contiguous()
        y = ds.targets.long()
        out[split] = (x.to(device), y.to(device))
    return out


def gpu_augment(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Augmentation em lote na GPU: afim aleatória (rotação, escala, shear, translação)
    + variação de espessura do traço (dilatação/erosão), para aproximar o traço do canvas."""
    n = x.shape[0]
    dev = x.device
    ang = (torch.rand(n, device=dev) * 2 - 1) * math.radians(12) * strength
    sc = 1 + (torch.rand(n, device=dev) * 2 - 1) * 0.12 * strength
    sh = (torch.rand(n, device=dev) * 2 - 1) * 0.25 * strength
    tx = (torch.rand(n, device=dev) * 2 - 1) * 0.10 * strength
    ty = (torch.rand(n, device=dev) * 2 - 1) * 0.10 * strength
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([
        torch.stack([cos, -sin + sh, tx], 1),
        torch.stack([sin, cos, ty], 1),
    ], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    r = torch.rand(n, device=dev)
    dil = F.max_pool2d(x, 3, 1, 1)
    ero = -F.max_pool2d(-x, 3, 1, 1)
    x = torch.where((r < 0.25).view(-1, 1, 1, 1), dil, x)
    x = torch.where(((r >= 0.25) & (r < 0.35)).view(-1, 1, 1, 1), ero, x)
    return x.clamp(0, 1)


@torch.no_grad()
def predict(model, x, bs=2048):
    model.eval()
    outs = [F.softmax(model(x[i:i + bs]), -1) for i in range(0, len(x), bs)]
    return torch.cat(outs)


def train_emnist(variant: str = "final", epochs: int = 20, bs: int = 256, lr: float = 2e-3,
                 data=None, seed: int = 42, device: str = "cuda", log=print):
    """variant='enunciado': arquitetura do enunciado, sem augmentation/dropout.
    variant='final': mesma arquitetura + augmentation na GPU + dropout 0.25 (usado no app)."""
    torch.manual_seed(seed)
    data = data or load_emnist(device)
    x, y = data["train"]
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(len(x), generator=g).to(device)
    n_val = len(x) // 10
    xv, yv = x[perm[:n_val]], y[perm[:n_val]]
    xt, yt = x[perm[n_val:]], y[perm[n_val:]]

    aug = variant == "final"
    model = EmnistCNN(dropout=0.25 if aug else 0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(xt) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    hist = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        idx = torch.randperm(len(xt), device=device)
        tl, tc, tn = 0.0, 0, 0
        for i in range(0, len(xt), bs):
            b = idx[i:i + bs]
            xb, yb = xt[b], yt[b]
            if aug:
                xb = gpu_augment(xb)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.05 if aug else 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(b)
            tc += (logits.argmax(1) == yb).sum().item()
            tn += len(b)
        pv = predict(model, xv)
        vl = F.nll_loss(pv.clamp_min(1e-9).log(), yv).item()
        va = (pv.argmax(1) == yv).float().mean().item()
        for k, v in zip(hist, (ep, tl / tn, tc / tn, vl, va)):
            hist[k].append(v)
        if va > best:
            best, best_state = va, {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f"[{variant}] época {ep:02d}/{epochs}  loss {tl / tn:.4f}  acc {tc / tn:.4f}  "
            f"val_loss {vl:.4f}  val_acc {va:.4f}  ({time.time() - t0:.1f}s)")
    model.load_state_dict(best_state)
    return model, hist


def main():
    ART.mkdir(exist_ok=True)
    data = load_emnist()
    xs, ys = data["test"]
    for variant, epochs in (("enunciado", 12), ("final", 25)):
        model, hist = train_emnist(variant, epochs=epochs, data=data)
        acc = (predict(model, xs).argmax(1) == ys).float().mean().item()
        print(f"[{variant}] acurácia no teste: {acc:.4f}")
        torch.save(model.state_dict(), ART / f"emnist_{variant}.pt")
        hist["test_acc"] = acc
        (ART / f"emnist_{variant}_history.json").write_text(json.dumps(hist, indent=1))


if __name__ == "__main__":
    main()
