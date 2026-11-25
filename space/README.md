---
title: Traço · escrita cursiva
emoji: ✍️
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Reconhece palavras em letra cursiva no navegador (ONNX)
---

# Traço: reconhecimento de escrita cursiva

Prova de conceito do **Desafio Final do Bootcamp Arquiteto(a) de Deep Learning (XP Educação)**.
Você escreve uma palavra no canvas (com mouse, dedo ou caneta) e o sistema reconhece a palavra
**letra por letra**, direto no navegador. Não há servidor nem Gradio: o app usa HTML, JS,
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) e [OpenCV.js](https://docs.opencv.org/4.10.0/opencv.js).

## Dois modos

| Modo | Modelo | Segmentação em letras |
|---|---|---|
| **Cursiva** (padrão) | CRNN (CNN + BiLSTM) com perda CTC, treinada com palavras cursivas sintéticas (45 fontes do Google Fonts, esqueletizadas e redesenhadas com traço uniforme + distorções) | alinhamento da CTC: o pico de cada letra na saída localiza a letra na imagem |
| **Letra de forma** | CNN do enunciado treinada no **EMNIST Balanced** (47 classes) | OpenCV: `GaussianBlur` → `threshold(OTSU)` → `findContours` → `boundingRect`, com as caixas ordenadas por x |

## Pipeline no navegador

1. Os traços são guardados como vetores e re-renderizados com **escala e espessura normalizadas**.
2. OpenCV.js: cinza → blur 5×5 → Otsu invertido → contornos externos → caixas.
3. Cursiva: o recorte da palavra vira um tensor 32×256 → CRNN → 64 quadros × 27 classes → decodificação CTC (greedy + beam search com dicionário PT/EN opcional).
   Letra de forma: cada letra vira um recorte 28×28 no padrão EMNIST → CNN → top-3.
4. A palavra é reconstruída e exibida com a probabilidade de cada letra. As marcas em vermelho mostram onde a rede separou as letras.

## Arquivos

- `index.html`, `css/`, `js/`: a aplicação
- `models/cursive_crnn.onnx`, `models/emnist_cnn.onnx`: os modelos exportados do PyTorch
- `data/lexicon.txt`: o léxico usado na correção (frequências do OpenSubtitles via *hermitdave/FrequencyWords*)
- `notebook/desafio_final.ipynb`: o treino, as métricas e as respostas do desafio
