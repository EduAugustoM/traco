"""Modelos do desafio: CNN EMNIST (arquitetura do enunciado) e CRNN-CTC para cursiva."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- EMNIST
# EMNIST Balanced: 47 classes (dígitos, 26 maiúsculas, 11 minúsculas distintas)
EMNIST_CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"


class EmnistCNN(nn.Module):
    """Porta fiel para PyTorch da CNN do enunciado (Keras):
    Conv(32,5x5,tanh) -> MaxPool -> Conv(48,5x5,tanh) -> MaxPool -> Conv(64,5x5,tanh)
    -> Flatten -> Dense(512,tanh) -> Dense(84,tanh) -> Dense(47,softmax)
    `dropout` > 0 adiciona regularização (variante comparada no notebook).
    """

    def __init__(self, n_classes: int = 47, dropout: float = 0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.Tanh(), nn.MaxPool2d(2),
            nn.Conv2d(32, 48, 5, padding=2), nn.Tanh(), nn.MaxPool2d(2),
            nn.Conv2d(48, 64, 5, padding=2), nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(64 * 7 * 7, 512), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(512, 84), nn.Tanh(),
            nn.Linear(84, n_classes),
        )

    def forward(self, x):  # x: [N,1,28,28] em [0,1], tinta=1
        return self.classifier(self.features(x))  # logits


class WithSoftmax(nn.Module):
    """Embute o softmax no grafo exportado para ONNX (o app recebe probabilidades)."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x):
        return F.softmax(self.m(x), dim=-1)


# ----------------------------------------------------------------------------- CRNN
CURSIVE_ALPHABET = "abcdefghijklmnopqrstuvwxyz"  # índice 0 = blank do CTC
CRNN_H, CRNN_W = 32, 256


def _block(cin, cout, pool):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if pool is not None:
        layers.append(nn.MaxPool2d(pool))
    return layers


class CRNN(nn.Module):
    """CNN (estilo VGG) + BiLSTM + Linear -> log-probs por passo de tempo.
    Entrada [N,1,32,256] -> saída [T=64, N, 27] (log_softmax).
    """

    def __init__(self, n_classes: int = len(CURSIVE_ALPHABET) + 1, hidden: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            *_block(1, 32, (2, 2)),        # 16 x 128
            *_block(32, 64, (2, 2)),       # 8 x 64
            *_block(64, 128, None),
            *_block(128, 128, (2, 1)),     # 4 x 64
            *_block(128, 192, None),
            *_block(192, 192, (2, 1)),     # 2 x 64
            *_block(192, 256, (2, 1)),     # 1 x 64
        )
        self.dropout = nn.Dropout(0.2)
        self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        f = self.cnn(x)                    # [N,C,1,T]
        f = f.squeeze(2).permute(2, 0, 1)  # [T,N,C]
        f, _ = self.rnn(self.dropout(f))
        return F.log_softmax(self.fc(f), dim=-1)  # [T,N,K]


class CRNNExport(nn.Module):
    """Versão para exportação: saída [N,T,K] com probabilidades (softmax)."""

    def __init__(self, m: CRNN):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).exp().permute(1, 0, 2)


# ----------------------------------------------------------------------------- CTC util
def ctc_greedy(probs_tk, alphabet: str = CURSIVE_ALPHABET):
    """probs_tk: tensor/array [T,K]. Retorna (texto, spans[(char, t0, t1, p)])."""
    import numpy as np

    p = np.asarray(probs_tk)
    best = p.argmax(-1)
    out, spans, prev = [], [], 0
    for t, k in enumerate(best):
        if k != 0 and k != prev:
            spans.append([alphabet[k - 1], t, t, float(p[t, k])])
        elif k != 0 and k == prev and spans:
            spans[-1][2] = t
            spans[-1][3] = max(spans[-1][3], float(p[t, k]))
        prev = k
    return "".join(s[0] for s in spans), spans
