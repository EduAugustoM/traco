// Área de escrita: folha pautada (guias) + camada de tinta. Guarda os traços como vetores
// (coordenadas em px CSS), o que permite re-renderizar com espessura normalizada para a rede.

export class InkPad {
  constructor({ ink, guides, page, onChange }) {
    this.ink = ink;
    this.guides = guides;
    this.page = page;
    this.onChange = onChange || (() => {});
    this.strokes = [];
    this.current = null;
    this.ctx = ink.getContext("2d");
    this.gctx = guides.getContext("2d");
    this.width = 0;
    this.height = 0;
    this._bind();
    new ResizeObserver(() => this.resize()).observe(page);
    this.resize();
  }

  get isEmpty() { return this.strokes.length === 0; }

  /** Geometria da pauta (px CSS): ascendente, altura-x, linha de base, descendente. */
  geometry() {
    const h = this.height;
    const u = h * 0.215;
    const asc = h * 0.17;
    return { asc, xTop: asc + u, base: asc + 2 * u, desc: asc + 3 * u, unit: u, margin: this.width < 520 ? 22 : 54 };
  }

  resize() {
    const r = this.page.getBoundingClientRect();
    if (!r.width || !r.height) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const sx = this.width ? r.width / this.width : 1;
    const sy = this.height ? r.height / this.height : 1;
    if (this.width && (sx !== 1 || sy !== 1)) {
      // mantém o desenho proporcional quando a folha muda de tamanho
      for (const s of this.strokes) for (const p of s) { p[0] *= sx; p[1] *= sy; }
    }
    this.width = r.width;
    this.height = r.height;
    for (const c of [this.ink, this.guides]) {
      c.width = Math.round(r.width * dpr);
      c.height = Math.round(r.height * dpr);
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.gctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.drawGuides();
    this.redraw();
  }

  drawGuides() {
    const g = this.gctx;
    const css = getComputedStyle(document.documentElement);
    const rule = css.getPropertyValue("--rule").trim();
    const strong = css.getPropertyValue("--rule-strong").trim();
    const margin = css.getPropertyValue("--margin").trim();
    const { asc, xTop, base, desc, unit, margin: mx } = this.geometry();
    g.clearRect(0, 0, this.width, this.height);
    const line = (y, color, w, dash) => {
      g.beginPath();
      g.setLineDash(dash || []);
      g.strokeStyle = color;
      g.lineWidth = w;
      g.moveTo(0, Math.round(y) + 0.5);
      g.lineTo(this.width, Math.round(y) + 0.5);
      g.stroke();
    };
    // pautas vizinhas, bem leves, sugerindo a folha do caderno
    g.globalAlpha = 0.35;
    for (let y = asc - unit; y > 0; y -= unit) line(y, rule, 1);
    for (let y = desc + unit; y < this.height; y += unit) line(y, rule, 1);
    g.globalAlpha = 1;
    line(asc, rule, 1);
    line(xTop, rule, 1, [6, 5]);
    line(base, strong, 1.6);
    line(desc, rule, 1);
    g.setLineDash([]);
    g.beginPath();
    g.strokeStyle = margin;
    g.lineWidth = 1.5;
    g.moveTo(mx + 0.5, 0);
    g.lineTo(mx + 0.5, this.height);
    g.stroke();
  }

  _penWidth(p, type) {
    if (type === "pen" && p > 0) return 2.6 + 3.2 * p;
    return 4.2;
  }

  _bind() {
    const c = this.ink;
    c.addEventListener("pointerdown", (e) => {
      if (e.button !== undefined && e.button > 0) return;
      e.preventDefault();
      c.setPointerCapture(e.pointerId);
      const pt = this._pt(e);
      this.current = { pts: [pt], type: e.pointerType };
      this.strokes.push(this.current.pts);
      this._drawDot(pt, e.pointerType);
      this.page.classList.add("has-ink");
    });
    c.addEventListener("pointermove", (e) => {
      if (!this.current) return;
      e.preventDefault();
      const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
      for (const ev of evs.length ? evs : [e]) {
        const pt = this._pt(ev);
        const pts = this.current.pts;
        const last = pts[pts.length - 1];
        if (Math.hypot(pt[0] - last[0], pt[1] - last[1]) < 0.8) continue;
        pts.push(pt);
        this._drawSegment(pts, this.current.type);
      }
    });
    const end = (e) => {
      if (!this.current) return;
      this.current = null;
      try { c.releasePointerCapture(e.pointerId); } catch { /* já liberado */ }
      this.onChange("stroke");
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
    c.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  _pt(e) {
    const r = this.ink.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top, e.pressure || 0.5];
  }

  _style(type, p) {
    const ctx = this.ctx;
    // lê a cor uma vez por traço (getComputedStyle a cada pointermove força recálculo de estilo e trava o desenho)
    if (!this._ink || !this.current) this._ink = getComputedStyle(document.documentElement).getPropertyValue("--ink").trim() || "#1d3faf";
    ctx.strokeStyle = this._ink;
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = this._penWidth(p, type);
  }

  _drawDot(pt, type) {
    this._style(type, pt[2]);
    this.ctx.beginPath();
    this.ctx.arc(pt[0], pt[1], this.ctx.lineWidth / 2, 0, Math.PI * 2);
    this.ctx.fill();
  }

  _drawSegment(pts, type) {
    const n = pts.length;
    const ctx = this.ctx;
    this._style(type, pts[n - 1][2]);
    ctx.beginPath();
    if (n < 3) {
      ctx.moveTo(pts[0][0], pts[0][1]);
      ctx.lineTo(pts[n - 1][0], pts[n - 1][1]);
    } else {
      const [a, b, c] = [pts[n - 3], pts[n - 2], pts[n - 1]];
      ctx.moveTo((a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
      ctx.quadraticCurveTo(b[0], b[1], (b[0] + c[0]) / 2, (b[1] + c[1]) / 2);
    }
    ctx.stroke();
  }

  redraw() {
    this.ctx.clearRect(0, 0, this.width, this.height);
    const ctx = this.ctx;
    for (const s of this.strokes) {
      this._drawDot(s[0], "mouse");
      if (s.length < 2) continue;
      this._style("mouse", 0.5);
      ctx.beginPath();
      ctx.moveTo(s[0][0], s[0][1]);
      if (s.length === 2) ctx.lineTo(s[1][0], s[1][1]);
      for (let i = 1; i < s.length - 1; i++) {
        const mx = (s[i][0] + s[i + 1][0]) / 2, my = (s[i][1] + s[i + 1][1]) / 2;
        ctx.quadraticCurveTo(s[i][0], s[i][1], mx, my);
      }
      ctx.stroke();
    }
    this.page.classList.toggle("has-ink", !this.isEmpty);
  }

  undo() {
    this.strokes.pop();
    this.redraw();
    this.onChange("undo");
  }

  clear() {
    this.strokes = [];
    this.redraw();
    this.onChange("clear");
  }

  /** Cópia serializável dos traços (para a coleta de amostras). */
  snapshot() {
    return this.strokes.map((s) => s.map(([x, y, p]) => [+x.toFixed(1), +y.toFixed(1), +p.toFixed(2)]));
  }
}
