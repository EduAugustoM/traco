---
name: Traço
description: Reconhecedor de escrita cursiva e letra de forma inspirado em um caderno de caligrafia escolar brasileiro.
colors:
  cover: "#1a3696"
  cover-deep: "#12276f"
  cover-ink: "#f3f6ff"
  ground: "#edf1f8"
  sheet: "#fcfdff"
  rule: "#b3c7ea"
  rule-strong: "#7d9bd2"
  margin: "#e4576a"
  ink: "#1d3faf"
  ink-deep: "#142e85"
  ink-wash: "#e3eafb"
  correct: "#c81f3a"
  correct-wash: "#fde8ec"
  graphite: "#1c2230"
  graphite-2: "#465066"
  muted: "#586276"
  line: "#d5ddec"
  ok: "#1d7a4c"
typography:
  display:
    fontFamily: "Figtree, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "4.4rem"
    fontWeight: 750
    lineHeight: 1.05
    letterSpacing: "-0.025em"
  headline:
    fontFamily: "Playwrite BR, Figtree, cursive"
    fontSize: "1.95rem"
    fontWeight: 300
    lineHeight: 1.5
    letterSpacing: "0em"
  title:
    fontFamily: "Figtree, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "1.35rem"
    fontWeight: 700
    lineHeight: 1.25
    letterSpacing: "-0.015em"
  body:
    fontFamily: "Figtree, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Figtree, system-ui, -apple-system, Segoe UI, sans-serif"
    fontSize: "0.86rem"
    fontWeight: 600
    lineHeight: 1.3
rounded:
  sm: "7px"
  md: "10px"
  pill: "999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "32px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "#ffffff"
    rounded: "{rounded.sm}"
    padding: "9px 20px"
  button-primary-hover:
    backgroundColor: "{colors.ink-deep}"
  button-ghost:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.graphite}"
    rounded: "{rounded.sm}"
    padding: "9px 16px"
  letter-chip:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.graphite}"
    rounded: "{rounded.sm}"
    padding: "6px 4px"
  letter-chip-low:
    backgroundColor: "{colors.correct-wash}"
    textColor: "{colors.correct}"
    rounded: "{rounded.sm}"
    padding: "6px 4px"
---

# Design System: Traço

## Overview

**Creative North Star: "O Caderno de Caligrafia"**

Traço transforma uma ferramenta técnica de reconhecimento de escrita manual em uma folha viva de caderno de caligrafia escolar brasileiro. Em vez de isolar o desenho dentro de um retângulo cinza genérico ao lado de cartões empilhados, a própria folha pautada — com linha de base firme, altura-x tracejada, ascendentes, descendentes e margem vertical vermelha — é o protagonista da experiência. As guias da pauta cumprem dupla função: ambientam o usuário imediatamente e induzem a escrita na proporção vertical exata que a rede neural CRNN-CTC espera.

Quando o reconhecimento acontece, a resposta volta como a correção de um professor sobre a própria página: traços verticais em vermelho-correção marcam onde o alinhamento CTC dividiu as letras ligadas, rótulos tipográficos identificam cada caractere sob a linha de base e elipses vermelhas circulam letras de baixa confiança (<50%). Abaixo da dobra, a tira de inspeção do pipeline mantém cada estágio (traço normalizado, binarização de Otsu com contornos OpenCV, tensor de entrada 32×256 ou 28×28 e mapa de calor CTC) na mesma linha de base para leitura técnica imediata pelo avaliador.

**Key Characteristics:**
- Folha pautada funcional com linhas de caligrafia (`--rule`, `--rule-strong`, `--margin`) que orientam a escala do traço sem contaminar o tensor binarizado.
- Tinta esferográfica azul (`--ink`) para o traço do usuário e ações primárias; vermelho-correção (`--correct`) reservado exclusivamente para marcas de segmentação e incerteza.
- Contraste tipográfico entre a letra cursiva escolar brasileira (`Playwrite BR`) na marca/exemplos e a sem-serifa limpa (`Figtree`) com numerais tabulares para probabilidades e palavra decodificada.
- Uma única geometria de segmentação compartilhada entre as marcas SVG sobre o canvas, os chips de letra por letra e as caixas da tira de pipeline.

## Colors

A paleta combina o azul profundo da capa de caderno universitário, o branco frio do papel sulfite pautado, o azul de caneta esferográfica e o vermelho de caneta corretora.

### Primary
- **Ballpoint Ink Blue** (`#1d3faf`): Cor do traço desenhado pelo usuário no canvas, do botão primário `Reconhecer`, das barras de confiança e dos links ativos.
- **Deep Ink Navy** (`#142e85`): Estado hover do botão primário e cor da palavra final reconhecida em tamanho display.
- **Notebook Cover Blue** (`#1a3696`): Fundo da barra superior (`topbar`), evocando a capa dura do caderno, apoiado pela borda inferior **Deep Cover** (`#12276f`) e texto **Cover Chalk** (`#f3f6ff`).

### Secondary
- **Correction Red** (`#c81f3a`): Usado exclusivamente para as marcas de segmentação da caneta corretora sobre o canvas, alertas de erro e destaque de letras com probabilidade abaixo de 50%.
- **Correction Wash** (`#fde8ec`): Fundo suave dos chips de letra quando a confiança é baixa (<50%).

### Tertiary
- **Calligraphy Ruling Blue** (`#b3c7ea`) & **Baseline Ruling Blue** (`#7d9bd2`): Linhas horizontais da pauta de caligrafia (ascendente, altura-x tracejada, linha de base sólida e descendente).
- **Margin Rule Rose** (`#e4576a`): Linha vertical de margem esquerda da folha pautada.
- **Ink Wash** (`#e3eafb`): Seleção de texto e chips de leituras alternativas do dicionário.
- **Verified Green** (`#1d7a4c`): Indicadores pontuais de confirmação.

### Neutral
- **Desk Ground** (`#edf1f8`): Fundo geral da página que faz a folha branca destacar-se com leveza.
- **Sulfite Sheet** (`#fcfdff`): Superfície da folha de escrita, dos chips de letra e dos quadros do pipeline.
- **Lead Graphite** (`#1c2230`): Texto principal, títulos de seção e caracteres reconhecidos nos chips.
- **Secondary Graphite** (`#465066`) & **Muted Slate** (`#586276`): Instruções acima da folha, rótulos de estágio e notas explicativas (todos com contraste ≥4.5:1).
- **Hairline Border** (`#d5ddec`): Bordas de 1px nos chips de letras, botões fantasma e quadros de inspeção do pipeline.

### Named Rules
**The Teacher's Red Pen Rule.** O vermelho (`#c81f3a`) nunca é usado como cor decorativa ou de ação primária: ele pertence somente às marcas de segmentação sobre a folha e à sinalização honesta de baixa confiança (<50%).
**The Clean Tensor Rule.** As linhas azuis e vermelhas da pauta vivem em um `<canvas class="guides">` separado do `<canvas class="ink">`, garantindo que a rede neural e o limiar de Otsu recebam apenas a tinta do usuário.

## Typography

**Display & UI Font:** `Figtree` (com fallback `system-ui, -apple-system, Segoe UI, sans-serif`)
**Handwriting & Brand Font:** `Playwrite BR` (cursiva escolar brasileira, pesos 200–400)

**Character:** `Playwrite BR` traz o traço autêntico da alfabetização cursiva brasileira para o logotipo, o convite "escreva aqui" e as palavras-modelo da coleta, enquanto `Figtree` oferece leitura nítida e numerais tabulares (`tabular-nums`) para a palavra decodificada e as probabilidades por letra.

### Hierarchy
- **Display Word** (`750`, `clamp(2.6rem, 5.4vw, 4.4rem)`, `line-height: 1.05`, `letter-spacing: -0.025em`): A palavra reconstruída pela rede neural no painel de leitura.
- **Handwritten Brand / Headline** (`300`, `1.95rem`, `line-height: 1.5`): Logotipo `Traço` no cabeçalho e palavras-modelo no modo de coleta (`1.8rem`).
- **Section Title** (`700`, `1.35rem`, `line-height: 1.25`, `letter-spacing: -0.015em`): Títulos principais (`Palavra reconhecida`, `Como o Traço leu`) e subtítulo `Letra por letra` (`1.05rem`).
- **Body** (`400`, `1rem`, `line-height: 1.5`, medida máxima `65–70ch`): Instruções de uso e explicação de cada modo.
- **Label & Numeric Probabilities** (`600`, `0.78rem–0.86rem`, `font-variant-numeric: tabular-nums`): Percentuais de probabilidade por letra, alternativas top-3 e legendas numeradas do pipeline.

### Named Rules
**The Tabular Confidence Rule.** Toda porcentagem de probabilidade, tempo de inferência em milissegundos e contagem de amostras usa `font-variant-numeric: tabular-nums` para que os dígitos nunca dancem entre leituras.

## Layout

O layout principal (`.layout`, largura máxima `1320px`, padding lateral `clamp(1rem, 4vw, 2.5rem)`) usa um grid assimétrico de duas colunas em telas `≥1000px`: a coluna esquerda (`1.7fr`) abriga a folha pautada e a barra de ferramentas; a coluna direita (`minmax(300px, 1fr)`) apresenta a palavra reconhecida, a lista horizontal de chips letra por letra e o seletor de dicionário. Abaixo das duas colunas (`grid-column: 1 / -1`), a seção **Como o Traço leu** alinha os quatro estágios do pipeline em uma grade responsiva (`repeat(auto-fit, minmax(220px, 1fr))`) compartilhando a mesma altura (`112px`) e a mesma linha de base. Em telas menores que `1000px`, tudo empilha em coluna única com a folha no topo.

## Elevation & Depth

O sistema é majoritariamente plano e estruturado por linhas de 1px (`#d5ddec`), reservando sombra difusa apenas para destacar a folha de papel sobre a mesa (`--ground`) e dar peso tátil ao botão primário `Reconhecer`.

### Shadow Vocabulary
- **Sheet Elevation** (`box-shadow: 0 1px 2px rgb(18 39 111 / 0.07), 0 10px 28px -10px rgb(18 39 111 / 0.22)`): Aplicada exclusivamente à folha de escrita (`.page`) e ao painel de coleta de amostras.
- **Primary Ink Button** (`box-shadow: 0 1px 2px rgb(20 46 133 / 0.25), 0 6px 14px -6px rgb(20 46 133 / 0.5)`): Dá presença física ao botão `Reconhecer`.

## Shapes

- **Folha e painéis maiores:** cantos suavemente arredondados (`10px`, `--radius`).
- **Botões, chips de letras e quadros do pipeline:** cantos técnicos compactos (`7px`, `--radius-sm`) com borda de `1px solid #d5ddec`.
- **Seletor de modo (`Cursiva` / `Letra de forma`) e chips de palavras alternativas:** cápsulas completas (`999px`).

## Components

### Buttons
- **Shape:** Cantos arredondados de `7px` (`--radius-sm`), altura mínima de `44px` para toque confortável.
- **Primary (`.btn.primary`):** Fundo `#1d3faf`, texto branco, padding `0.55rem 1.25rem`, ícone SVG em traço `1.8px` e dica de tecla `<kbd>Enter</kbd>`.
- **Hover / Focus:** Muda para `#142e85` no hover; `:active` desloca `1px` para baixo; foco visível com anel `2.5px solid #1d3faf` e `outline-offset: 3px`.
- **Ghost (`.btn.ghost`):** Fundo `#fcfdff`, texto `#1c2230`, borda `1px solid #d5ddec` que escurece para `#7d9bd2` no hover (`Desfazer`, `Limpar`).

### Chips (`.letter` & `.alt-chip`)
- **Letter Chip (`.letter`):** Bloco vertical de `3.4rem` de largura em `#fcfdff` com borda `1px solid #d5ddec`, exibindo a letra em `1.55rem` negrito, a probabilidade em `0.78rem` tabular, uma barra de confiança de `3px` e as alternativas top-3 abaixo.
- **Low-Confidence State (`.letter.low`):** Quando a probabilidade é `<50%`, a borda muda para `#f1a9b5`, o fundo para `#fde8ec` e a letra/barra para `#c81f3a`.
- **Dictionary Alternative Chip (`.alt-chip`):** Cápsula (`999px`) em `#e3eafb` com texto `#142e85`; ao clicar, substitui a palavra exibida.

### Signature Component: Ruled Calligraphy Sheet & Red-Pen Overlay (`.page`)
A área de escrita sobrepõe três camadas sincronizadas:
1. `canvas.guides`: Desenha a pauta azul escolar (ascendente, altura-x tracejada, linha de base mais espessa, descendente) e a margem vertical vermelha.
2. `canvas.ink`: Captura Pointer Events (mouse, toque ou caneta com pressão) em azul-caneta (`#1d3faf`) com suavização por curvas quadráticas.
3. `svg.marks`: Após a inferência, desenha da esquerda para a direita (`stroke-dashoffset` com `cubic-bezier(0.16, 1, 0.3, 1)`) os cortes verticais da CTC (modo cursiva) ou caixas arredondadas (modo forma), circulando letras incertas e rotulando cada segmento com letra e porcentagem.

## Do's and Don'ts

### Do:
- **Do** manter as três camadas da folha (`guides`, `ink`, `marks`) alinhadas no mesmo sistema de coordenadas CSS para que os cortes vermelhos coincidam exatamente com os traços do usuário.
- **Do** exibir sempre os quatro estágios da tira `Como o Traço leu` na mesma altura (`112px`) e alinhados pela base.
- **Do** destacar letras com probabilidade `<50%` tanto no chip (`.letter.low`) quanto com uma elipse vermelha sobre a folha.

### Don't:
- **Don't** misturar o canvas da pauta com o canvas da tinta; a binarização de Otsu jamais deve enxergar as linhas-guia.
- **Don't** usar vermelho (`#c81f3a`) em botões primários, cabeçalhos ou elementos decorativos fora da semântica de correção/erro.
- **Don't** ocultar a leitura crua da rede quando o dicionário corrigir uma palavra: mostre sempre o que a CRNN leu (`res.raw`) junto da palavra corrigida.
