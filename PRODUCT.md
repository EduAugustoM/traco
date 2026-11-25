# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Static HTML + CSS + vanilla JS (ES modules, no bundler), hosted as a Hugging Face **static** Space (`sdk: static`). Inference runs entirely in the browser with onnxruntime-web (WASM). Image preprocessing with OpenCV.js. No Gradio, no Gradio-Lite, no server.

## Users

Primary: the bootcamp professor/grader (Prof. Pedro Calais) opening the public Space link to test the app quickly, on desktop (mouse) and on a phone/tablet (touch). Secondary: bootcamp colleagues. They must understand what to do without instructions and see the pipeline the challenge asks for: draw → segment into letters → model per letter → reconstructed word.

## Product Purpose

**Traço** is a proof of concept of handwritten word recognition (Desafio Final, Bootcamp Arquiteto(a) de Deep Learning, XP Educação). The user writes a word on a canvas and the system recognizes it. Success = the grader draws a word, presses recognize, and sees the correct word with per-letter segmentation and probabilities.

## Positioning

Recognizes **cursive (joined) handwriting**, not just block letters. Two modes:
- **Cursiva** (default): CRNN + CTC trained on synthetic cursive words; letter segmentation comes from the CTC alignment.
- **Letra de forma**: CNN trained on EMNIST Balanced (47 classes, the challenge's architecture); segmentation via OpenCV contours, exactly as the challenge statement.

## Operating Context

Graded academic delivery: `.ipynb` notebook + public HF Space link. The grader tests by drawing words with mouse or finger. Language: Portuguese (Brazil) UI; words in Portuguese and English, lowercase a–z only in cursive mode (no accents).

## Capabilities and Constraints

- Canvas drawing (mouse, touch, pen), undo, clear, recognize.
- Shows recognized word, per-letter chips `[m][a][c]…` with probabilities, letter boxes over the drawing, optional dictionary correction, and an inspectable pipeline (binarized image, model input, crops).
- Data collection mode for the author to draw real samples (test set) and export them.
- Models: `emnist_cnn.onnx` (~7 MB fp32), `cursive_crnn.onnx` (int8, few MB). Everything loads client-side.
- Cursive mode: lowercase only (v1).

## Evidence on Hand

Real metrics come from the notebook (EMNIST test accuracy, confusion matrix; cursive CER/WER on held-out fonts and on real samples drawn by the author). No testimonials, users or benchmarks beyond these must be invented.

## Product Principles

1. The canvas is the product: drawing must feel immediate and precise.
2. Show the work: every recognition exposes how it was reached (segmentation + probabilities), because the audience is evaluating the pipeline.
3. Honest uncertainty: low-confidence letters look low-confidence; never fake certainty.
4. Zero setup: opens and works on first visit, desktop and mobile, no login.

## Accessibility & Inclusion

WCAG AA contrast, visible focus, keyboard shortcuts (Enter = recognize, Ctrl+Z = undo, Esc = clear), usable with touch on small screens, respects `prefers-reduced-motion`.
