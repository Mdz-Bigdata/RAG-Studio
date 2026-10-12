import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, Download, FileText, GitBranch, Hash, Info, Loader2, Maximize2, Minimize2, Minus, Network, Plus, Scan, Search, Shapes, X } from 'lucide-react';
import type { GraphEdge, GraphNode, KbGraph } from '../api/types';
import { Badge, Button } from './ui';

type Pt = { x: number; y: number };
type View = { k: number; tx: number; ty: number };

const EDGE_LEN: Record<GraphEdge['type'], number> = { contains: 70, defines: 62, mentions: 92, relation: 110 };
const EDGE_K: Record<GraphEdge['type'], number> = { contains: 0.09, defines: 0.07, mentions: 0.025, relation: 0.05 };
const CHARGE: Record<GraphNode['type'], number> = { doc: 5, section: 1.8, entity: 1.5 };
const TYPE_LABEL: Record<GraphNode['type'], string> = { doc: '文档', section: '章节', entity: '实体' };
const EDGE_LABEL: Record<GraphEdge['type'], string> = { contains: '包含', defines: '定义', mentions: '提及', relation: '关系' };

/** 节点半径：文档最大；实体按连接数放大，越"中心"的实体越显眼 */
function radius(n: GraphNode): number {
  if (n.type === 'doc') return 12 + Math.min(8, Math.sqrt(n.degree) * 1.6);
  if (n.type === 'section') return n.level === 1 ? 7 : 5;
  return 5 + Math.min(7, n.degree * 0.7);
}
function fill(n: GraphNode): string {
  if (n.type === 'doc') return 'var(--ink)';
  if (n.type === 'section') return n.level === 1 ? 'var(--c-dense)' : 'var(--c-mmr)';
  return `var(--et-${n.entityType ?? 'concept'})`;
}
/** 关系边粗细：三档（屏幕像素，随缩放按 √k 变化）。默认按"对推理的信息量"：因果 / 依赖最粗，结构性关系最细 */
type WidthLevel = 1 | 2 | 3;
const WIDTH_PX: Record<WidthLevel, number> = { 1: 1.1, 2: 2, 3: 3.4 };
const WIDTH_LABEL: Record<WidthLevel, string> = { 1: '细', 2: '中', 3: '粗' };
const DEFAULT_REL_WIDTH: Record<string, WidthLevel> = { cause: 3, require: 3, constraint: 2, process: 2, replace: 2, usage: 2, structure: 1, other: 1 };
const REL_WIDTH_KEY = 'rag.graph.relWidths';
function readRelWidths(): Record<string, WidthLevel> {
  try {
    const v = JSON.parse(localStorage.getItem(REL_WIDTH_KEY) ?? 'null');
    if (v && typeof v === 'object') return { ...DEFAULT_REL_WIDTH, ...Object.fromEntries(Object.entries(v).filter(([, l]) => l === 1 || l === 2 || l === 3)) } as Record<string, WidthLevel>;
  } catch { /* 读不到就用默认 */ }
  return { ...DEFAULT_REL_WIDTH };
}
/** 实体节点大小：三档半径（世界坐标）。默认：制度 / 系统是知识库的主干最大，数值类是细节最小 */
type SizeLevel = 1 | 2 | 3;
const SIZE_R: Record<SizeLevel, number> = { 1: 4.5, 2: 7, 3: 10.5 };
const SIZE_LABEL: Record<SizeLevel, string> = { 1: '小', 2: '中', 3: '大' };
const DEFAULT_ENTITY_SIZE: Record<string, SizeLevel> = { policy: 3, system: 3, process: 2, product: 2, role: 2, concept: 2, metric: 1, other: 1 };
const ENTITY_SIZE_KEY = 'rag.graph.entitySizes';
function readEntitySizes(): Record<string, SizeLevel> {
  try {
    const v = JSON.parse(localStorage.getItem(ENTITY_SIZE_KEY) ?? 'null');
    if (v && typeof v === 'object') return { ...DEFAULT_ENTITY_SIZE, ...Object.fromEntries(Object.entries(v).filter(([, l]) => l === 1 || l === 2 || l === 3)) } as Record<string, SizeLevel>;
  } catch { /* 读不到就用默认 */ }
  return { ...DEFAULT_ENTITY_SIZE };
}
/** 第 i 篇文档的颜色：前 10 篇用主题色板（浅 / 深各一套），之后按黄金角生成色相 */
const docColorAt = (i: number) => (i < 10 ? `var(--doc-${i})` : `hsl(${Math.round((i * 137.508) % 360)} 62% 52%)`);
type ColorMode = 'type' | 'doc';
const COLOR_MODE_KEY = 'rag.graph.colorMode';
const readColorMode = (): ColorMode => { try { return localStorage.getItem(COLOR_MODE_KEY) === 'doc' ? 'doc' : 'type'; } catch { return 'type'; } };
type EdgeColorMode = 'group' | 'doc' | 'mono';
const EDGE_COLOR_KEY = 'rag.graph.edgeColorMode';
const readEdgeColorMode = (): EdgeColorMode => { try { const v = localStorage.getItem(EDGE_COLOR_KEY); return v === 'doc' || v === 'mono' ? v : 'group'; } catch { return 'group'; } };
type SizeMode = 'type' | 'doc' | 'degree';
const SIZE_MODE_KEY = 'rag.graph.sizeMode';
const readSizeMode = (): SizeMode => { try { const v = localStorage.getItem(SIZE_MODE_KEY); return v === 'doc' || v === 'degree' ? v : 'type'; } catch { return 'type'; } };
/** 按文档显示大小时，文档 / 章节节点的缩放倍数（实体直接用 SIZE_R 三档） */
const DOC_SIZE_FACTOR: Record<SizeLevel, number> = { 1: 0.75, 2: 1, 3: 1.35 };
const DOC_SIZE_KEY = 'rag.graph.docSizes';
function readDocSizes(): Record<string, SizeLevel> {
  try {
    const v = JSON.parse(localStorage.getItem(DOC_SIZE_KEY) ?? 'null');
    if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).filter(([, l]) => l === 1 || l === 2 || l === 3)) as Record<string, SizeLevel>;
  } catch { /* 读不到就用默认 */ }
  return {};
}
type EdgeWidthMode = 'group' | 'doc' | 'type' | 'uniform';
const EDGE_WIDTH_MODE_KEY = 'rag.graph.edgeWidthMode';
const readEdgeWidthMode = (): EdgeWidthMode => { try { const v = localStorage.getItem(EDGE_WIDTH_MODE_KEY); return v === 'doc' || v === 'type' || v === 'uniform' ? v : 'group'; } catch { return 'group'; } };
const TYPE_EDGE_WIDTH_KEY = 'rag.graph.typeEdgeWidths';
/** 按文档显示粗细时，结构 / 定义 / 提及边的缩放倍数（关系边直接用 WIDTH_PX 三档） */
const DOC_WIDTH_FACTOR: Record<WidthLevel, number> = { 1: 0.7, 2: 1, 3: 1.6 };
const DOC_EDGE_WIDTH_KEY = 'rag.graph.docEdgeWidths';
const ENTITY_TYPE_LABEL: Record<string, string> = { policy: '制度政策', process: '流程操作', product: '产品服务', system: '系统组件', role: '组织角色', metric: '指标数值', concept: '概念术语', other: '其他' };

type Box = { x0: number; y0: number; x1: number; y1: number };
type PlacedLabel = { key: string; text: string; x: number; y: number; anchor: 'start' | 'end' | 'middle'; cls: string; size: number; color?: string };
const LABEL_PX: Record<GraphNode['type'], number> = { doc: 12.5, section: 11.5, entity: 11.5 };
const MAX_LABEL = 14;

/** 估算文字宽度：中日韩字符约 1em，其余约 0.58em */
function textWidth(text: string, px: number): number {
  let w = 0;
  for (const ch of text) w += /[\u2e80-\uffff]/.test(ch) ? px : px * 0.58;
  return w;
}
const hits = (a: Box, b: Box) => a.x0 < b.x1 && a.x1 > b.x0 && a.y0 < b.y1 && a.y1 > b.y0;

/** 标签避让：按优先级依次尝试「右 / 左 / 下 / 上」四个位置，四个都和已放的标签或别的节点重叠就不显示。
 *  forced（选中 / 悬停 / 搜索命中）的标签一定显示，并且最先摆放。 */
function placeLabels(
  candidates: { key: string; text: string; sx: number; sy: number; r: number; cls: string; size: number; forced: boolean }[],
  circles: { key: string; box: Box }[],
  edgeLabels: { key: string; text: string; sx: number; sy: number; color?: string }[],
  size: Size,
): PlacedLabel[] {
  const placed: Box[] = [];
  const out: PlacedLabel[] = [];
  const pad = 2;
  const blocked = (b: Box, self: string) => placed.some((o) => hits(b, o)) || circles.some((c) => c.key !== self && hits(b, c.box));
  const onScreen = (b: Box) => b.x1 > 0 && b.x0 < size.w && b.y1 > 0 && b.y0 < size.h;
  for (const c of candidates) {
    const w = textWidth(c.text, c.size);
    const h = c.size * 1.25;
    const gap = c.r + 4;
    const spots: [Box, PlacedLabel['anchor'], number, number][] = [
      [{ x0: c.sx + gap, y0: c.sy - h / 2, x1: c.sx + gap + w, y1: c.sy + h / 2 }, 'start', c.sx + gap, c.sy],
      [{ x0: c.sx - gap - w, y0: c.sy - h / 2, x1: c.sx - gap, y1: c.sy + h / 2 }, 'end', c.sx - gap, c.sy],
      [{ x0: c.sx - w / 2, y0: c.sy + c.r + 2, x1: c.sx + w / 2, y1: c.sy + c.r + 2 + h }, 'middle', c.sx, c.sy + c.r + 2 + h / 2],
      [{ x0: c.sx - w / 2, y0: c.sy - c.r - 2 - h, x1: c.sx + w / 2, y1: c.sy - c.r - 2 }, 'middle', c.sx, c.sy - c.r - 2 - h / 2],
    ];
    let pick = spots.find(([b]) => onScreen(b) && !blocked(b, c.key));
    if (!pick && c.forced) {
      // 必须显示的标签（悬停节点的邻居等）：四个默认位置都被占了，就往外多推几圈再找空位，实在没有才允许重叠
      for (const d of [10, 22, 36, 52]) {
        const shifted: typeof spots = [
          [{ x0: c.sx + gap + d, y0: c.sy - h / 2, x1: c.sx + gap + d + w, y1: c.sy + h / 2 }, 'start', c.sx + gap + d, c.sy],
          [{ x0: c.sx - gap - d - w, y0: c.sy - h / 2, x1: c.sx - gap - d, y1: c.sy + h / 2 }, 'end', c.sx - gap - d, c.sy],
          [{ x0: c.sx - w / 2, y0: c.sy + c.r + 2 + d, x1: c.sx + w / 2, y1: c.sy + c.r + 2 + d + h }, 'middle', c.sx, c.sy + c.r + 2 + d + h / 2],
          [{ x0: c.sx - w / 2, y0: c.sy - c.r - 2 - d - h, x1: c.sx + w / 2, y1: c.sy - c.r - 2 - d }, 'middle', c.sx, c.sy - c.r - 2 - d - h / 2],
          [{ x0: c.sx + gap, y0: c.sy - d - h, x1: c.sx + gap + w, y1: c.sy - d }, 'start', c.sx + gap, c.sy - d - h / 2],
          [{ x0: c.sx + gap, y0: c.sy + d, x1: c.sx + gap + w, y1: c.sy + d + h }, 'start', c.sx + gap, c.sy + d + h / 2],
          [{ x0: c.sx - gap - w, y0: c.sy - d - h, x1: c.sx - gap, y1: c.sy - d }, 'end', c.sx - gap, c.sy - d - h / 2],
          [{ x0: c.sx - gap - w, y0: c.sy + d, x1: c.sx - gap, y1: c.sy + d + h }, 'end', c.sx - gap, c.sy + d + h / 2],
        ];
        pick = shifted.find(([b]) => onScreen(b) && !blocked(b, c.key));
        if (pick) break;
      }
      pick ??= spots.find(([b]) => onScreen(b)) ?? spots[0];
    }
    if (!pick) continue;
    if (!pick) continue;
    const [b, anchor, x, y] = pick;
    placed.push({ x0: b.x0 - pad, y0: b.y0 - pad, x1: b.x1 + pad, y1: b.y1 + pad });
    out.push({ key: c.key, text: c.text, x, y, anchor, cls: c.cls, size: c.size });
  }
  for (const e of edgeLabels) {
    const px = 10.5;
    const w = textWidth(e.text, px);
    const b = { x0: e.sx - w / 2, y0: e.sy - px * 0.65, x1: e.sx + w / 2, y1: e.sy + px * 0.65 };
    if (!onScreen(b) || placed.some((o) => hits(b, o))) continue;
    placed.push(b);
    out.push({ key: e.key, text: e.text, x: e.sx, y: e.sy, anchor: 'middle', cls: 'graph-edge-label', size: px, color: e.color });
  }
  return out;
}

const STYLE_PROPS = ['fill', 'fill-opacity', 'stroke', 'stroke-width', 'stroke-opacity', 'stroke-dasharray', 'stroke-linecap', 'stroke-linejoin', 'opacity', 'font-size', 'font-family', 'font-weight', 'paint-order', 'text-anchor', 'dominant-baseline'];
const FONT_PROPS = new Set(['font-size', 'font-family', 'font-weight', 'text-anchor', 'dominant-baseline', 'paint-order']);
/** 与 SVG 默认值相同的样式不写进导出文件，体积能小一半 */
const STYLE_DEFAULTS: Record<string, string> = { 'fill-opacity': '1', 'stroke-opacity': '1', opacity: '1', 'stroke-dasharray': 'none', 'stroke-linecap': 'butt', 'stroke-linejoin': 'miter', 'paint-order': 'normal', 'text-anchor': 'start', 'dominant-baseline': 'auto', 'font-weight': '400', stroke: 'none' };
const HEADER_H = 56;

/** 生成独立的 SVG 文档：逐个元素内联计算后的样式（CSS 变量、class 样式离开页面就失效）→ 加背景与标题栏 */
function buildExportSvg(svg: SVGSVGElement, size: Size, header: { title: string; sub: string }): { xml: string; w: number; h: number } {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  const live = [svg, ...svg.querySelectorAll('*')];
  const copies = [clone, ...clone.querySelectorAll('*')];
  live.forEach((el, i) => {
    const cs = getComputedStyle(el);
    const target = copies[i] as SVGElement;
    target.removeAttribute('class');
    // 引用 CSS 变量的展示属性（fill="var(--ink)" 等）在页面外无意义，而且 Figma / Illustrator 不认，删掉，以内联样式为准
    for (const attr of [...target.attributes]) if (attr.value.includes('var(')) target.removeAttribute(attr.name);
    const isText = el.tagName === 'text';
    const decl = STYLE_PROPS.flatMap((prop) => {
      if (!isText && FONT_PROPS.has(prop)) return [];
      const v = cs.getPropertyValue(prop).trim();
      return !v || STYLE_DEFAULTS[prop] === v ? [] : [`${prop}:${v}`];
    });
    if (decl.length) target.setAttribute('style', decl.join(';'));
  });
  const root = getComputedStyle(document.documentElement);
  const bg = root.getPropertyValue('--bg-1').trim() || '#ffffff';
  const fg = root.getPropertyValue('--text-1').trim() || '#14171d';
  const muted = root.getPropertyValue('--text-3').trim() || '#858d9c';
  const border = root.getPropertyValue('--border').trim() || 'rgba(0,0,0,0.1)';
  const font = getComputedStyle(document.body).fontFamily;
  const W = size.w;
  const H = size.h + HEADER_H;
  const ns = 'http://www.w3.org/2000/svg';
  const out = document.createElementNS(ns, 'svg');
  out.setAttribute('xmlns', ns);
  out.setAttribute('width', String(W));
  out.setAttribute('height', String(H));
  out.setAttribute('viewBox', `0 0 ${W} ${H}`);
  // 属性值里会出现字体名的双引号（"IBM Plex Sans", …），必须转义，否则 SVG 解析失败
  const esc = (t: string) => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  out.innerHTML = `<rect width="${W}" height="${H}" fill="${bg}"/>`
    + `<text x="20" y="24" style="font-family:${esc(font)};font-size:15px;font-weight:600;fill:${fg}">${esc(header.title)}</text>`
    + `<text x="20" y="44" style="font-family:${esc(font)};font-size:11.5px;fill:${muted}">${esc(header.sub)}</text>`
    + `<line x1="0" y1="${HEADER_H - 0.5}" x2="${W}" y2="${HEADER_H - 0.5}" stroke="${border}"/>`;
  const body = document.createElementNS(ns, 'g');
  body.setAttribute('transform', `translate(0,${HEADER_H})`);
  while (clone.firstChild) body.appendChild(clone.firstChild);
  out.appendChild(body);
  const titleEl = document.createElementNS(ns, 'title');
  titleEl.textContent = header.title;
  out.insertBefore(titleEl, out.firstChild);
  return { xml: '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(out), w: W, h: H };
}

/** 栅格化：SVG → <img> → canvas → PNG */
async function svgToPng(exported: { xml: string; w: number; h: number }, ratio: number): Promise<Blob> {
  const { xml, w: W, h: H } = exported;
  const url = URL.createObjectURL(new Blob([xml], { type: 'image/svg+xml;charset=utf-8' }));
  try {
    const img = new Image();
    img.decoding = 'async';
    await new Promise<void>((resolve, reject) => { img.onload = () => resolve(); img.onerror = () => reject(new Error('SVG 渲染失败')); img.src = url; });
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(W * ratio);
    canvas.height = Math.round(H * ratio);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('浏览器不支持 canvas');
    ctx.scale(ratio, ratio);
    ctx.drawImage(img, 0, 0, W, H);
    return await new Promise<Blob>((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('PNG 编码失败'))), 'image/png'));
  } finally {
    URL.revokeObjectURL(url);
  }
}

/** 可复现的伪随机：同一张图每次布局一致，刷新不会"跳" */
function seeded(seed: number) {
  let s = seed % 2147483647 || 1;
  return () => (s = (s * 16807) % 2147483647) / 2147483647;
}
function hash(str: string): number {
  let h = 0;
  for (let i = 0; i < str.length; i++) h = (h * 31 + str.charCodeAt(i)) | 0;
  return Math.abs(h) + 1;
}

/** 力导向布局（同步跑完再渲染）：节点互斥 + 边弹簧 + 向心力。300 个节点以内几十毫秒。 */
function layout(nodes: GraphNode[], edges: GraphEdge[]): Map<string, Pt> {
  const rnd = seeded(hash(nodes.map((n) => n.id).join('|')));
  const idx = new Map(nodes.map((n, i) => [n.id, i]));
  const docs = nodes.filter((n) => n.type === 'doc');
  const pos: Pt[] = nodes.map(() => ({ x: 0, y: 0 }));
  const vel: Pt[] = nodes.map(() => ({ x: 0, y: 0 }));
  // 初始位置：文档排成一圈，其余节点放在第一个相连节点附近
  const ring = Math.max(120, docs.length * 55);
  docs.forEach((d, i) => {
    const a = (i / Math.max(1, docs.length)) * Math.PI * 2;
    pos[idx.get(d.id)!] = { x: Math.cos(a) * ring, y: Math.sin(a) * ring };
  });
  const firstLink = new Map<string, string>();
  for (const e of edges) {
    if (!firstLink.has(e.target)) firstLink.set(e.target, e.source);
    if (!firstLink.has(e.source)) firstLink.set(e.source, e.target);
  }
  const placed = new Set(docs.map((d) => d.id));
  for (let pass = 0; pass < 4; pass++) {
    for (const n of nodes) {
      if (placed.has(n.id)) continue;
      const anchor = firstLink.get(n.id);
      if (pass < 3 && (!anchor || !placed.has(anchor))) continue;
      const base = anchor && placed.has(anchor) ? pos[idx.get(anchor)!] : { x: 0, y: 0 };
      pos[idx.get(n.id)!] = { x: base.x + (rnd() - 0.5) * 80, y: base.y + (rnd() - 0.5) * 80 };
      placed.add(n.id);
    }
  }
  const links = edges.map((e) => ({ a: idx.get(e.source)!, b: idx.get(e.target)!, len: EDGE_LEN[e.type], k: EDGE_K[e.type] })).filter((l) => l.a !== undefined && l.b !== undefined);
  const charge = nodes.map((n) => CHARGE[n.type]);
  const iters = nodes.length > 300 ? 220 : 320;
  for (let it = 0; it < iters; it++) {
    const alpha = 1 - it / iters;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        let dx = pos[j].x - pos[i].x;
        let dy = pos[j].y - pos[i].y;
        let d2 = dx * dx + dy * dy;
        if (d2 > 250000) continue;                       // 500px 以外的斥力忽略不计
        if (d2 < 1) { dx = rnd() - 0.5; dy = rnd() - 0.5; d2 = 1; }
        const f = (900 * charge[i] * charge[j]) / d2 * alpha;
        const d = Math.sqrt(d2);
        vel[i].x -= (dx / d) * f; vel[i].y -= (dy / d) * f;
        vel[j].x += (dx / d) * f; vel[j].y += (dy / d) * f;
      }
    }
    for (const l of links) {
      const dx = pos[l.b].x - pos[l.a].x;
      const dy = pos[l.b].y - pos[l.a].y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const f = (d - l.len) * l.k * alpha;
      vel[l.a].x += (dx / d) * f; vel[l.a].y += (dy / d) * f;
      vel[l.b].x -= (dx / d) * f; vel[l.b].y -= (dy / d) * f;
    }
    for (let i = 0; i < nodes.length; i++) {
      vel[i].x -= pos[i].x * 0.006 * alpha;
      vel[i].y -= pos[i].y * 0.006 * alpha;
      vel[i].x *= 0.62; vel[i].y *= 0.62;
      const sp = Math.hypot(vel[i].x, vel[i].y);
      if (sp > 30) { vel[i].x *= 30 / sp; vel[i].y *= 30 / sp; }
      pos[i].x += vel[i].x; pos[i].y += vel[i].y;
    }
  }
  return new Map(nodes.map((n, i) => [n.id, pos[i]]));
}

type Size = { w: number; h: number };

/** 适应画布。core > 0 时只框住中间那部分节点（按坐标分位数裁掉外围），大图默认看得清 */
function fitView(points: Pt[], size: Size, core = 0): View {
  if (!points.length) return { k: 1, tx: size.w / 2, ty: size.h / 2 };
  const xs = points.map((p) => p.x).sort((a, b) => a - b);
  const ys = points.map((p) => p.y).sort((a, b) => a - b);
  const at = (arr: number[], q: number) => arr[Math.min(arr.length - 1, Math.max(0, Math.round(q * (arr.length - 1))))];
  const [x0, x1, y0, y1] = [at(xs, core), at(xs, 1 - core), at(ys, core), at(ys, 1 - core)];
  const k = Math.min(2, Math.max(0.15, Math.min((size.w - 80) / Math.max(1, x1 - x0), (size.h - 80) / Math.max(1, y1 - y0))));
  return { k, tx: size.w / 2 - ((x0 + x1) / 2) * k, ty: size.h / 2 - ((y0 + y1) / 2) * k };
}
/** 默认视图：节点多时聚焦核心区域（裁掉四周各 15% 的外围节点），少时全部显示 */
const initialView = (points: Pt[], size: Size) => fitView(points, size, points.length > 120 ? 0.15 : points.length > 60 ? 0.08 : 0);
const clampK = (k: number) => Math.min(4, Math.max(0.12, k));

export function EntityGraph({ graph, title, onOpenList, onExported }: { graph: KbGraph; title?: string; onOpenList?: (entity: string) => void; onExported?: (fileName: string, error: string | null) => void }) {
  const [showSections, setShowSections] = useState(true);
  const [showMentions, setShowMentions] = useState(true);
  const [edgeColorMode, setEdgeColorModeState] = useState<EdgeColorMode>(readEdgeColorMode);
  const setEdgeColorMode = (m: EdgeColorMode) => { setEdgeColorModeState(m); try { localStorage.setItem(EDGE_COLOR_KEY, m); } catch { /* 只在本次生效 */ } };
  const colorRelations = edgeColorMode === 'group';
  const [edgeWidthMode, setEdgeWidthModeState] = useState<EdgeWidthMode>(readEdgeWidthMode);
  const setEdgeWidthMode = (m: EdgeWidthMode) => { setEdgeWidthModeState(m); try { localStorage.setItem(EDGE_WIDTH_MODE_KEY, m); } catch { /* 只在本次生效 */ } };
  const widthByGroup = edgeWidthMode === 'group';
  const [relWidths, setRelWidths] = useState<Record<string, WidthLevel>>(readRelWidths);
  const setRelWidth = (group: string, level: WidthLevel | null) => setRelWidths((cur) => {
    const next = level === null ? { ...DEFAULT_REL_WIDTH } : { ...cur, [group]: level };
    try {
      const custom = Object.fromEntries(Object.entries(next).filter(([g, l]) => DEFAULT_REL_WIDTH[g] !== l));
      if (Object.keys(custom).length) localStorage.setItem(REL_WIDTH_KEY, JSON.stringify(custom)); else localStorage.removeItem(REL_WIDTH_KEY);
    } catch { /* 无痕模式等不可写时只在本次生效 */ }
    return next;
  });
  /** 某个关系类别的线宽（屏幕像素，缩放为 1 时） */
  const groupWidth = useCallback((group: string) => (widthByGroup ? WIDTH_PX[relWidths[group] ?? 2] : 1.6), [widthByGroup, relWidths]);
  const relWidthsCustomized = Object.entries(relWidths).some(([g, l]) => DEFAULT_REL_WIDTH[g] !== l);
  const [colorMode, setColorModeState] = useState<ColorMode>(readColorMode);
  const setColorMode = (m: ColorMode) => { setColorModeState(m); try { localStorage.setItem(COLOR_MODE_KEY, m); } catch { /* 只在本次生效 */ } };
  // 文档颜色按完整图里的文档顺序分配；实体归属：定义它的文档权重 3、提到它的文档权重 1，取权重最高的为主文档
  const docColors = useMemo(() => new Map(graph.nodes.filter((n) => n.type === 'doc').map((n, i) => [n.id, docColorAt(i)])), [graph]);
  const entityDocs = useMemo(() => {
    const docOfNode = new Map(graph.nodes.filter((n) => n.type !== 'entity').map((n) => [n.id, `doc:${n.docId}`]));
    const weights = new Map<string, Map<string, number>>();
    for (const e of graph.edges) {
      if (e.type !== 'defines' && e.type !== 'mentions') continue;
      const d = docOfNode.get(e.source);
      if (!d) continue;
      if (!weights.has(e.target)) weights.set(e.target, new Map());
      const w = weights.get(e.target)!;
      w.set(d, (w.get(d) ?? 0) + (e.type === 'defines' ? 3 : 1));
    }
    return new Map([...weights].map(([id, w]) => [id, [...w].sort((a, b) => b[1] - a[1]).map(([d]) => d)]));
  }, [graph]);
  /** 节点颜色：按类型（默认）或按所属文档 */
  const nodeFill = useCallback((n: GraphNode) => {
    if (colorMode === 'type') return fill(n);
    const doc = n.type === 'entity' ? entityDocs.get(n.id)?.[0] : `doc:${n.docId}`;
    return (doc && docColors.get(doc)) || 'var(--text-3)';
  }, [colorMode, entityDocs, docColors]);
  const [sizeMode, setSizeModeState] = useState<SizeMode>(readSizeMode);
  const setSizeMode = (m: SizeMode) => { setSizeModeState(m); try { localStorage.setItem(SIZE_MODE_KEY, m); } catch { /* 只在本次生效 */ } };
  const sizeByType = sizeMode === 'type';
  // 每篇文档定义 / 提到的实体数（完整图），用于文档列表与默认大小
  const docEntityCounts = useMemo(() => {
    const docOfNode = new Map(graph.nodes.filter((n) => n.type !== 'entity').map((n) => [n.id, `doc:${n.docId}`]));
    const ents = new Map<string, Set<string>>();
    for (const e of graph.edges) {
      if (e.type !== 'defines' && e.type !== 'mentions') continue;
      const d = docOfNode.get(e.source);
      if (!d) continue;
      if (!ents.has(d)) ents.set(d, new Set());
      ents.get(d)!.add(e.target);
    }
    return new Map(graph.nodes.filter((n) => n.type === 'doc').map((n) => [n.id, ents.get(n.id)?.size ?? 0]));
  }, [graph]);
  const defaultDocSizes = useMemo(() => {
    const ranked = [...docEntityCounts].sort((a, b) => b[1] - a[1]).map(([id]) => id);
    const n = ranked.length;
    return new Map(ranked.map((id, i) => [id, (n < 2 ? 2 : i < n / 3 ? 3 : i < (2 * n) / 3 ? 2 : 1) as SizeLevel]));
  }, [docEntityCounts]);
  const [docSizeOverrides, setDocSizeOverrides] = useState<Record<string, SizeLevel>>(readDocSizes);
  const docSize = useCallback((docId: string | undefined): SizeLevel => (docId ? docSizeOverrides[docId] ?? defaultDocSizes.get(docId) ?? 2 : 2), [docSizeOverrides, defaultDocSizes]);
  const setDocSize = (docId: string, level: SizeLevel | null) => setDocSizeOverrides((cur) => {
    const next = { ...cur };
    if (level === null || level === defaultDocSizes.get(docId)) delete next[docId]; else next[docId] = level;
    try { if (Object.keys(next).length) localStorage.setItem(DOC_SIZE_KEY, JSON.stringify(next)); else localStorage.removeItem(DOC_SIZE_KEY); } catch { /* 只在本次生效 */ }
    return next;
  });
  const resetDocSizes = () => { setDocSizeOverrides({}); try { localStorage.removeItem(DOC_SIZE_KEY); } catch { /* 只在本次生效 */ } };
  const docSizesCustomized = [...defaultDocSizes.keys()].some((id) => docSizeOverrides[id] !== undefined);
  const [docWidthOverrides, setDocWidthOverrides] = useState<Record<string, WidthLevel>>(() => {
    try {
      const v = JSON.parse(localStorage.getItem(DOC_EDGE_WIDTH_KEY) ?? 'null');
      if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).filter(([, l]) => l === 1 || l === 2 || l === 3)) as Record<string, WidthLevel>;
    } catch { /* 读不到就用默认 */ }
    return {};
  });
  const docWidth = useCallback((docId: string | undefined): WidthLevel => (docId ? docWidthOverrides[docId] ?? (defaultDocSizes.get(docId) as WidthLevel | undefined) ?? 2 : 2), [docWidthOverrides, defaultDocSizes]);
  const setDocWidth = (docId: string, level: WidthLevel) => setDocWidthOverrides((cur) => {
    const next = { ...cur };
    if (level === defaultDocSizes.get(docId)) delete next[docId]; else next[docId] = level;
    try { if (Object.keys(next).length) localStorage.setItem(DOC_EDGE_WIDTH_KEY, JSON.stringify(next)); else localStorage.removeItem(DOC_EDGE_WIDTH_KEY); } catch { /* 只在本次生效 */ }
    return next;
  });
  const resetDocWidths = () => { setDocWidthOverrides({}); try { localStorage.removeItem(DOC_EDGE_WIDTH_KEY); } catch { /* 只在本次生效 */ } };
  const docWidthsCustomized = [...defaultDocSizes.keys()].some((id) => docWidthOverrides[id] !== undefined);
  const [entitySizes, setEntitySizes] = useState<Record<string, SizeLevel>>(readEntitySizes);
  const setEntitySize = (type: string, level: SizeLevel | null) => setEntitySizes((cur) => {
    const next = level === null ? { ...DEFAULT_ENTITY_SIZE } : { ...cur, [type]: level };
    try {
      const custom = Object.fromEntries(Object.entries(next).filter(([t, l]) => DEFAULT_ENTITY_SIZE[t] !== l));
      if (Object.keys(custom).length) localStorage.setItem(ENTITY_SIZE_KEY, JSON.stringify(custom)); else localStorage.removeItem(ENTITY_SIZE_KEY);
    } catch { /* 无痕模式等不可写时只在本次生效 */ }
    return next;
  });
  const entitySizesCustomized = Object.entries(entitySizes).some(([t, l]) => DEFAULT_ENTITY_SIZE[t] !== l);
  const [typeWidthOverrides, setTypeWidthOverrides] = useState<Record<string, WidthLevel>>(() => {
    try {
      const v = JSON.parse(localStorage.getItem(TYPE_EDGE_WIDTH_KEY) ?? 'null');
      if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).filter(([, l]) => l === 1 || l === 2 || l === 3)) as Record<string, WidthLevel>;
    } catch { /* 读不到就用默认 */ }
    return {};
  });
  const typeWidth = useCallback((t: string | undefined): WidthLevel => typeWidthOverrides[t ?? 'concept'] ?? (DEFAULT_ENTITY_SIZE[t ?? 'concept'] as WidthLevel | undefined) ?? 2, [typeWidthOverrides]);
  const setTypeWidth = (t: string, level: WidthLevel | null) => setTypeWidthOverrides((cur) => {
    const next = { ...cur };
    if (level === null) { for (const k of Object.keys(next)) delete next[k]; } else if (level === DEFAULT_ENTITY_SIZE[t]) delete next[t]; else next[t] = level;
    try { if (Object.keys(next).length) localStorage.setItem(TYPE_EDGE_WIDTH_KEY, JSON.stringify(next)); else localStorage.removeItem(TYPE_EDGE_WIDTH_KEY); } catch { /* 只在本次生效 */ }
    return next;
  });
  const typeWidthsCustomized = Object.keys(typeWidthOverrides).length > 0;
  /** 节点半径：实体按类型（开启时）或按连接数；文档 / 章节不变 */
  const nodeR = useCallback((n: GraphNode) => {
    if (sizeMode === 'type') return n.type === 'entity' ? SIZE_R[entitySizes[n.entityType ?? 'concept'] ?? 2] : radius(n);
    if (sizeMode === 'doc') {
      const level = docSize(n.type === 'entity' ? entityDocs.get(n.id)?.[0] : `doc:${n.docId}`);
      return n.type === 'entity' ? SIZE_R[level] : radius(n) * DOC_SIZE_FACTOR[level];
    }
    return radius(n);
  }, [sizeMode, entitySizes, docSize, entityDocs]);
  const [query, setQuery] = useState('');
  const [hover, setHover] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [showInfo, setShowInfo] = useState(false);

  // ---- 按文档筛选：null = 全部文档
  const docList = useMemo(() => graph.nodes.filter((n) => n.type === 'doc').map((n) => {
    const color = colorMode === 'doc' || edgeColorMode === 'doc' ? docColors.get(n.id) : undefined;
    const level = docSize(n.id);
    const wLevel = docWidth(n.id);
    const sizeSteps = sizeMode === 'doc' ? (
      <span className="width-steps" role="radiogroup" aria-label={`${n.label}的节点大小`} title="节点大小">
        {([1, 2, 3] as SizeLevel[]).map((l) => (
          <button key={l} role="radio" aria-checked={level === l} className={level === l ? 'on' : ''} title={`节点${SIZE_LABEL[l]}`} onClick={() => setDocSize(n.id, l)}>
            <i className="size-dot" style={{ width: 3 + l * 2.5, height: 3 + l * 2.5 }} />
          </button>
        ))}
      </span>
    ) : null;
    const widthSteps = edgeWidthMode === 'doc' ? (
      <span className="width-steps" role="radiogroup" aria-label={`${n.label}的边粗细`} title="边粗细">
        {([1, 2, 3] as WidthLevel[]).map((l) => (
          <button key={l} role="radio" aria-checked={wLevel === l} className={wLevel === l ? 'on' : ''} title={`边${WIDTH_LABEL[l]}`} onClick={() => setDocWidth(n.id, l)}>
            <span style={{ borderTopWidth: WIDTH_PX[l] }} />
          </button>
        ))}
      </span>
    ) : null;
    return {
      id: n.id, label: n.label, count: docEntityCounts.get(n.id) ?? 0, color,
      swatch: sizeMode === 'doc' ? <span className="size-swatch"><span style={{ background: color ?? 'var(--text-2)', width: SIZE_R[level] * 1.4, height: SIZE_R[level] * 1.4 }} /></span>
        : edgeWidthMode === 'doc' ? <span className="rel-swatch" style={{ borderTopColor: color ?? 'var(--text-2)', borderTopWidth: WIDTH_PX[wLevel] }} /> : undefined,
      extra: sizeSteps || widthSteps ? <>{sizeSteps}{widthSteps}</> : undefined,
    };
  }).sort((a, b) => b.count - a.count), [graph, colorMode, edgeColorMode, docColors, docEntityCounts, sizeMode, docSize, edgeWidthMode, docWidth]); // eslint-disable-line
  const [docFilter, setDocFilter] = useState<Set<string> | null>(null);
  // 换了知识库（文档集合变了）：去掉已不存在的文档，筛选为空就回到全部
  useEffect(() => {
    setDocFilter((cur) => {
      if (!cur) return cur;
      const next = new Set([...cur].filter((id) => docList.some((d) => d.id === id)));
      return next.size && next.size < docList.length ? next : null;
    });
  }, [docList]);
  const onlyDoc = (id: string) => setDocFilter(docList.length > 1 ? new Set([id]) : null);

  // 作用域：选中文档的章节 + 它们定义 / 提到的实体 + 与这些实体直接相关的实体（跨文档关系不至于断头）
  const docScoped = useMemo(() => {
    if (!docFilter) return graph;
    const keep = new Set(graph.nodes.filter((n) => n.type !== 'entity' && docFilter.has(`doc:${n.docId}`)).map((n) => n.id));
    for (const e of graph.edges) if ((e.type === 'defines' || e.type === 'mentions') && keep.has(e.source)) keep.add(e.target);
    const core = new Set(keep);
    for (const e of graph.edges) {
      if (e.type !== 'relation') continue;
      if (core.has(e.source)) keep.add(e.target);
      if (core.has(e.target)) keep.add(e.source);
    }
    return { ...graph, nodes: graph.nodes.filter((n) => keep.has(n.id)), edges: graph.edges.filter((e) => keep.has(e.source) && keep.has(e.target)) };
  }, [graph, docFilter]);

  // ---- 按实体类型筛选（数量按当前文档范围统计）；文档 / 章节骨架始终保留
  const [typeFilter, setTypeFilter] = useState<Set<string> | null>(null);
  const typeList = useMemo(() => {
    const counts = new Map<string, number>();
    for (const n of docScoped.nodes) if (n.type === 'entity') counts.set(n.entityType ?? 'concept', (counts.get(n.entityType ?? 'concept') ?? 0) + 1);
    const order = (graph.entityTypes ?? []).map((t) => t.id);
    const ids = [...new Set([...order, ...counts.keys()])].filter((id) => counts.get(id));
    return ids.map((id) => ({
      id, label: ENTITY_TYPE_LABEL[id] ?? id, count: counts.get(id) ?? 0, color: `var(--et-${id})`,
      swatch: sizeByType ? <span className="size-swatch"><span style={{ background: `var(--et-${id})`, width: SIZE_R[entitySizes[id] ?? 2] * 1.4, height: SIZE_R[entitySizes[id] ?? 2] * 1.4 }} /></span>
        : edgeWidthMode === 'type' ? <span className="rel-swatch" style={{ borderTopColor: `var(--et-${id})`, borderTopWidth: WIDTH_PX[typeWidth(id)] }} /> : undefined,
      extra: sizeByType || edgeWidthMode === 'type' ? (
        <>
          {sizeByType ? (
            <span className="width-steps" role="radiogroup" aria-label={`${ENTITY_TYPE_LABEL[id] ?? id}的大小`} title="节点大小">
              {([1, 2, 3] as SizeLevel[]).map((l) => (
                <button key={l} role="radio" aria-checked={(entitySizes[id] ?? 2) === l} className={(entitySizes[id] ?? 2) === l ? 'on' : ''} title={`节点${SIZE_LABEL[l]}`} onClick={() => setEntitySize(id, l)}>
                  <i className="size-dot" style={{ width: 3 + l * 2.5, height: 3 + l * 2.5 }} />
                </button>
              ))}
            </span>
          ) : null}
          {edgeWidthMode === 'type' ? (
            <span className="width-steps" role="radiogroup" aria-label={`${ENTITY_TYPE_LABEL[id] ?? id}的边粗细`} title="边粗细">
              {([1, 2, 3] as WidthLevel[]).map((l) => (
                <button key={l} role="radio" aria-checked={typeWidth(id) === l} className={typeWidth(id) === l ? 'on' : ''} title={`边${WIDTH_LABEL[l]}`} onClick={() => setTypeWidth(id, l)}>
                  <span style={{ borderTopWidth: WIDTH_PX[l] }} />
                </button>
              ))}
            </span>
          ) : null}
        </>
      ) : undefined,
    }));
  }, [docScoped, graph, sizeByType, entitySizes, edgeWidthMode, typeWidth]); // eslint-disable-line
  useEffect(() => {
    setTypeFilter((cur) => {
      if (!cur) return cur;
      const next = new Set([...cur].filter((id) => typeList.some((t) => t.id === id)));
      return next.size && next.size < typeList.length ? next : null;
    });
  }, [typeList]);
  const typeScoped = useMemo(() => {
    if (!typeFilter) return docScoped;
    const drop = new Set(docScoped.nodes.filter((n) => n.type === 'entity' && !typeFilter.has(n.entityType ?? 'concept')).map((n) => n.id));
    return { ...docScoped, nodes: docScoped.nodes.filter((n) => !drop.has(n.id)), edges: docScoped.edges.filter((e) => !drop.has(e.source) && !drop.has(e.target)) };
  }, [docScoped, typeFilter]);

  // ---- 按关系类别筛选：只保留选中类别的关系边，以及参与这些关系的实体；文档 / 章节骨架保留
  const [relFilter, setRelFilter] = useState<Set<string> | null>(null);
  const relList = useMemo(() => {
    const counts = new Map<string, Map<string, number>>();
    for (const e of typeScoped.edges) {
      if (e.type !== 'relation') continue;
      const g = e.group ?? 'other';
      if (!counts.has(g)) counts.set(g, new Map());
      counts.get(g)!.set(e.label, (counts.get(g)!.get(e.label) ?? 0) + 1);
    }
    return (graph.relationGroups ?? []).filter((g) => counts.has(g.id)).map((g) => {
      const labels = [...counts.get(g.id)!.entries()].sort((a, b) => b[1] - a[1]);
      const color = colorRelations ? `var(--rg-${g.id})` : 'var(--text-2)';
      return {
        id: g.id, label: g.label, count: labels.reduce((n, [, c]) => n + c, 0), hint: labels.slice(0, 4).map(([l]) => l).join('、'),
        swatch: colorRelations || widthByGroup ? <span className="rel-swatch" style={{ borderTopColor: color, borderTopWidth: groupWidth(g.id) }} /> : undefined,
        extra: widthByGroup ? (
          <span className="width-steps" role="radiogroup" aria-label={`${g.label}的粗细`}>
            {([1, 2, 3] as WidthLevel[]).map((l) => (
              <button key={l} role="radio" aria-checked={(relWidths[g.id] ?? 2) === l} className={(relWidths[g.id] ?? 2) === l ? 'on' : ''} title={WIDTH_LABEL[l]} onClick={() => setRelWidth(g.id, l)}>
                <span style={{ borderTopWidth: WIDTH_PX[l] }} />
              </button>
            ))}
          </span>
        ) : undefined,
      };
    });
  }, [typeScoped, graph, colorRelations, widthByGroup, relWidths, groupWidth]); // eslint-disable-line
  useEffect(() => {
    setRelFilter((cur) => {
      if (!cur) return cur;
      const next = new Set([...cur].filter((id) => relList.some((r) => r.id === id)));
      return next.size && next.size < relList.length ? next : null;
    });
  }, [relList]);
  const scoped = useMemo(() => {
    if (!relFilter) return typeScoped;
    const keptRel = typeScoped.edges.filter((e) => e.type === 'relation' && relFilter.has(e.group ?? 'other'));
    const inRel = new Set(keptRel.flatMap((e) => [e.source, e.target]));
    const nodes = typeScoped.nodes.filter((n) => n.type !== 'entity' || inRel.has(n.id));
    const ids = new Set(nodes.map((n) => n.id));
    return { ...typeScoped, nodes, edges: typeScoped.edges.filter((e) => ids.has(e.source) && ids.has(e.target) && (e.type !== 'relation' || relFilter.has(e.group ?? 'other'))) };
  }, [typeScoped, relFilter]);
  const EXPORT_HEADER = useMemo(() => {
    const count = (t: GraphNode['type']) => scoped.nodes.filter((n) => n.type === t).length;
    const rel = scoped.edges.filter((e) => e.type === 'relation').length;
    const scope = (docFilter ? ` · 仅含：${docList.filter((d) => docFilter.has(d.id)).map((d) => d.label).join('、')}` : '')
      + (typeFilter ? ` · 实体类型：${[...typeFilter].map((t) => ENTITY_TYPE_LABEL[t] ?? t).join('、')}` : '')
      + (relFilter ? ` · 关系：${relList.filter((r) => relFilter.has(r.id)).map((r) => r.label).join('、')}` : '');
    return {
      title: `${title || '知识库'} · 实体织网`,
      sub: `${count('doc')} 文档 · ${count('section')} 章节 · ${count('entity')} 实体 · ${rel} 关系${scope} · 导出于 ${new Date().toLocaleString('zh-CN', { hour12: false })}`,
    };
  }, [title, scoped, docFilter, docList, typeFilter, relFilter, relList]);

  // 过滤后的图：不显示章节时，文档直接连到实体（保留"实体属于哪篇文档"）
  const { nodes, edges } = useMemo(() => {
    let ns = scoped.nodes;
    let es = scoped.edges.filter((e) => showMentions || e.type !== 'mentions');
    if (!showSections) {
      const docOf = new Map(scoped.nodes.filter((n) => n.type === 'section').map((n) => [n.id, `doc:${n.docId}`]));
      ns = ns.filter((n) => n.type !== 'section');
      const seen = new Set<string>();
      es = es.flatMap((e) => {
        if (e.type === 'contains') return [];
        const s = docOf.get(e.source) ?? e.source;
        const t = docOf.get(e.target) ?? e.target;
        const key = `${s}>${t}>${e.type}>${e.label}`;
        if (s === t || seen.has(key)) return [];
        seen.add(key);
        return [{ ...e, source: s, target: t }];
      });
    }
    return { nodes: ns, edges: es };
  }, [scoped, showSections, showMentions]);

  // ---- 画布尺寸跟随容器（普通 / 全屏），viewBox 与像素 1:1，不会留白边
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [size, setSize] = useState<Size>({ w: 1200, h: 640 });
  const sizeRef = useRef(size);
  useLayoutEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const apply = () => {
      const r = el.getBoundingClientRect();
      const next = { w: Math.max(320, Math.round(r.width)), h: Math.max(320, Math.round(r.height)) };
      const prev = sizeRef.current;
      if (next.w === prev.w && next.h === prev.h) return;
      sizeRef.current = next;
      setSize(next);
      // 尺寸变了（进出全屏 / 窗口缩放）：保持画面中心不动
      setView((v) => ({ ...v, tx: v.tx + (next.w - prev.w) / 2, ty: v.ty + (next.h - prev.h) / 2 }));
    };
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    // 兜底：某些环境（后台标签页等不出帧时）ResizeObserver 不回调；窗口缩放 / 进出全屏时再量一次
    const late = () => requestAnimationFrame(apply);
    window.addEventListener('resize', apply);
    document.addEventListener('fullscreenchange', late);
    return () => { ro.disconnect(); window.removeEventListener('resize', apply); document.removeEventListener('fullscreenchange', late); };
  }, []);

  const base = useMemo(() => layout(nodes, edges), [nodes, edges]);
  const [pos, setPos] = useState(base);
  const [view, setView] = useState<View>(() => initialView([...base.values()], sizeRef.current));
  useLayoutEffect(() => { setPos(base); setView(initialView([...base.values()], sizeRef.current)); }, [base]);

  const byId = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  useEffect(() => { setSelected((cur) => (cur && !byId.has(cur) ? null : cur)); setHover(null); }, [byId]);
  const neighbors = useMemo(() => {
    const m = new Map<string, Set<string>>();
    for (const e of edges) {
      if (!m.has(e.source)) m.set(e.source, new Set());
      if (!m.has(e.target)) m.set(e.target, new Set());
      m.get(e.source)!.add(e.target);
      m.get(e.target)!.add(e.source);
    }
    return m;
  }, [edges]);

  const q = query.trim().toLowerCase();
  const matches = useMemo(() => (q ? new Set(nodes.filter((n) => n.label.toLowerCase().includes(q) || (n.definitions ?? []).some((d) => d.text.toLowerCase().includes(q))).map((n) => n.id)) : null), [nodes, q]);
  const focus = hover ?? selected;
  // 选中节点 → 所属文档的路径：沿「包含 / 定义」边往上找；没有定义的实体退而用「提及」，再不行用实体关系。
  // 同一实体在多处定义时，所有最短路径都高亮。
  const pathInfo = useMemo(() => {
    const start = selected && byId.get(selected);
    if (!start || start.type === 'doc') return null;
    const incoming = new Map<string, GraphEdge[]>();
    const outgoing = new Map<string, GraphEdge[]>();
    for (const e of edges) {
      if (!incoming.has(e.target)) incoming.set(e.target, []);
      if (!outgoing.has(e.source)) outgoing.set(e.source, []);
      incoming.get(e.target)!.push(e);
      outgoing.get(e.source)!.push(e);
    }
    const tiers: GraphEdge['type'][][] = [['contains', 'defines'], ['contains', 'defines', 'mentions'], ['contains', 'defines', 'mentions', 'relation']];
    for (const allowed of tiers) {
      const dist = new Map<string, number>([[start.id, 0]]);
      const via = new Map<string, { to: string; edge: GraphEdge }[]>();      // 节点 → 它下一步通向（离选中节点更近的）节点
      const docs: string[] = [];
      const queue = [start.id];
      while (queue.length) {
        const id = queue.shift()!;
        if (byId.get(id)?.type === 'doc') { docs.push(id); continue; }
        const steps = [
          ...(incoming.get(id) ?? []).filter((e) => allowed.includes(e.type)).map((e) => ({ next: e.source, edge: e })),
          // 关系边没有"上下级"，两个方向都能走
          ...(allowed.includes('relation') ? (outgoing.get(id) ?? []).filter((e) => e.type === 'relation').map((e) => ({ next: e.target, edge: e })) : []),
        ];
        for (const { next, edge } of steps) {
          const d = dist.get(id)! + 1;
          if (!dist.has(next)) { dist.set(next, d); queue.push(next); }
          if (dist.get(next) === d) {
            if (!via.has(next)) via.set(next, []);
            via.get(next)!.push({ to: id, edge });
          }
        }
      }
      if (!docs.length) continue;
      const nodeSet = new Set<string>([start.id]);
      const edgeSet = new Map<string, GraphEdge>();
      const paths: string[][] = [];
      const walk = (id: string, trail: string[]) => {
        nodeSet.add(id);
        if (id === start.id) { if (paths.length < 6) paths.push(trail); return; }
        for (const v of via.get(id) ?? []) {
          edgeSet.set(`${v.edge.source}>${v.edge.target}>${v.edge.type}>${v.edge.label}`, v.edge);
          walk(v.to, [...trail, v.to]);
        }
      };
      for (const d of docs) walk(d, [d]);
      return { nodes: nodeSet, edges: [...edgeSet.values()], paths, viaMentions: !allowed.every((t) => t === 'contains' || t === 'defines') };
    }
    return { nodes: new Set([start.id]), edges: [] as GraphEdge[], paths: [] as string[][], viaMentions: false };
  }, [selected, edges, byId]);

  const lit = useMemo(() => {
    if (focus) {
      const s = new Set([focus, ...(neighbors.get(focus) ?? [])]);
      if (pathInfo && focus === selected) pathInfo.nodes.forEach((id) => s.add(id));
      return s;
    }
    return matches;
  }, [focus, neighbors, matches, pathInfo, selected]);

  // ---- 全屏：优先浏览器原生全屏；不支持 / 被拒绝时用铺满窗口的浮层兜底
  const [nativeFs, setNativeFs] = useState(false);
  const [overlayFs, setOverlayFs] = useState(false);
  const fullscreen = nativeFs || overlayFs;
  useEffect(() => {
    const onChange = () => setNativeFs(document.fullscreenElement === wrapRef.current);
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);
  useEffect(() => {
    if (!overlayFs) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOverlayFs(false); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [overlayFs]);
  useEffect(() => {
    if (!selected) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !document.fullscreenElement && !overlayFs && !(e.target instanceof HTMLInputElement)) setSelected(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selected, overlayFs]);
  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement) { document.exitFullscreen().catch(() => undefined); return; }
    if (overlayFs) { setOverlayFs(false); return; }
    const el = wrapRef.current;
    if (el?.requestFullscreen) el.requestFullscreen().catch(() => setOverlayFs(true));
    else setOverlayFs(true);
  }, [overlayFs]);

  // ---- 交互：拖动画布平移、拖动节点、滚轮 / 触控板 / 双指手势
  const toSvg = useCallback((clientX: number, clientY: number): Pt => {
    const r = svgRef.current!.getBoundingClientRect();
    return { x: ((clientX - r.left) / r.width) * sizeRef.current.w, y: ((clientY - r.top) / r.height) * sizeRef.current.h };
  }, []);
  const zoomAt = useCallback((p: Pt, factor: number) => setView((v) => {
    const k = clampK(v.k * factor);
    return { k, tx: p.x - ((p.x - v.tx) / v.k) * k, ty: p.y - ((p.y - v.ty) / v.k) * k };
  }), []);
  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = (ev: WheelEvent) => {
      ev.preventDefault();                                // React 的 onWheel 是 passive，拦不住页面滚动，只能原生监听
      const p = toSvg(ev.clientX, ev.clientY);
      // 触控板捏合（浏览器会带 ctrlKey）与鼠标滚轮 → 缩放；触控板双指滑动（像素级、带横向分量）→ 平移
      const mouseWheel = ev.deltaMode !== 0 || (ev.deltaX === 0 && Math.abs(ev.deltaY) >= 50 && Number.isInteger(ev.deltaY));
      if (ev.ctrlKey || ev.metaKey) zoomAt(p, Math.exp(-ev.deltaY * 0.01));
      else if (mouseWheel) zoomAt(p, Math.exp(-ev.deltaY * 0.0015));
      else setView((v) => ({ ...v, tx: v.tx - ev.deltaX, ty: v.ty - ev.deltaY }));
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [toSvg, zoomAt]);

  const pointers = useRef(new Map<number, Pt>());
  const pinch = useRef<{ dist: number; k: number; mid: Pt; view: View } | null>(null);
  const drag = useRef<{ kind: 'pan' | 'node'; id?: string; start: Pt; origin: Pt; moved: boolean } | null>(null);
  const onPointerDown = (ev: React.PointerEvent, id?: string) => {
    ev.stopPropagation();
    svgRef.current?.setPointerCapture?.(ev.pointerId);
    const p = toSvg(ev.clientX, ev.clientY);
    pointers.current.set(ev.pointerId, p);
    if (pointers.current.size === 2) {
      // 第二根手指落下：从拖动切换到双指缩放
      const [a, b] = [...pointers.current.values()];
      pinch.current = { dist: Math.hypot(a.x - b.x, a.y - b.y) || 1, k: view.k, mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, view };
      drag.current = null;
      return;
    }
    drag.current = id ? { kind: 'node', id, start: p, origin: pos.get(id)!, moved: false } : { kind: 'pan', start: p, origin: { x: view.tx, y: view.ty }, moved: false };
  };
  const onPointerMove = (ev: React.PointerEvent) => {
    if (!pointers.current.has(ev.pointerId)) return;
    const p = toSvg(ev.clientX, ev.clientY);
    pointers.current.set(ev.pointerId, p);
    const pc = pinch.current;
    if (pc && pointers.current.size >= 2) {
      const [a, b] = [...pointers.current.values()];
      const k = clampK(pc.k * (Math.hypot(a.x - b.x, a.y - b.y) / pc.dist));
      const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      const wx = (pc.mid.x - pc.view.tx) / pc.view.k;          // 手势开始时双指中点对应的世界坐标
      const wy = (pc.mid.y - pc.view.ty) / pc.view.k;
      setView({ k, tx: mid.x - wx * k, ty: mid.y - wy * k });
      return;
    }
    const d = drag.current;
    if (!d) return;
    const dx = p.x - d.start.x;
    const dy = p.y - d.start.y;
    if (Math.hypot(dx, dy) > 5) d.moved = true;
    if (d.kind === 'pan') setView((v) => ({ ...v, tx: d.origin.x + dx, ty: d.origin.y + dy }));
    else if (d.id) setPos((m) => new Map(m).set(d.id!, { x: d.origin.x + dx / view.k, y: d.origin.y + dy / view.k }));
  };
  const lastTap = useRef<{ id: string; t: number } | null>(null);
  const onPointerUp = (ev: React.PointerEvent) => {
    pointers.current.delete(ev.pointerId);
    if (pointers.current.size < 2) pinch.current = null;
    const d = drag.current;
    drag.current = null;
    if (!d || d.moved) return;
    if (d.kind !== 'node' || !d.id) { setSelected(null); setHover(null); lastTap.current = null; return; }
    // 单击切换选中；双击的第二下不取消选中（否则双击节点会"打开又关上"详情）
    const now = performance.now();
    const second = lastTap.current?.id === d.id && now - lastTap.current.t < 400;
    lastTap.current = second ? null : { id: d.id, t: now };
    setSelected(second ? d.id : d.id === selected ? null : d.id);
  };
  const zoom = (f: number) => zoomAt({ x: size.w / 2, y: size.h / 2 }, f);
  const centerOn = (id: string) => {
    const p = pos.get(id);
    if (!p) return;
    setSelected(id);
    setView((v) => {
      const k = Math.max(v.k, 1.1);
      return { k, tx: size.w / 2 - p.x * k, ty: size.h / 2 - p.y * k };
    });
  };

  // 关系边颜色：按类别着色（可关），关掉时统一用紫色
  /** 边所属的文档：关系边取抽取出它的第一篇文档；结构 / 定义 / 提及边取起点（文档或章节）所在的文档 */
  const edgeDoc = useCallback((e: GraphEdge) => {
    if (e.type === 'relation') return e.docIds?.length ? `doc:${e.docIds[0]}` : undefined;
    const src = byId.get(e.source);
    return src && src.type !== 'entity' ? `doc:${src.docId}` : undefined;
  }, [byId]);
  const edgeColor = useCallback((e: GraphEdge) => {
    if (edgeColorMode === 'doc') return docColors.get(edgeDoc(e) ?? '') ?? 'var(--text-3)';
    if (e.type === 'relation') return edgeColorMode === 'group' ? `var(--rg-${e.group ?? 'other'})` : 'var(--c-hyde)';
    return e.type === 'contains' ? 'var(--border-strong)' : 'var(--c-weaved)';
  }, [edgeColorMode, docColors, edgeDoc]);
  const relColor = edgeColor;
  // 箭头 marker：按（颜色, 线宽）组合各建一个，只建当前图里用得到的
  /** 边线宽（屏幕像素，缩放为 1 时） */
  const typeEdgeLevel = useCallback((e: GraphEdge): WidthLevel | null => {
    if (e.type === 'contains') return null;
    const t = byId.get(e.target);
    if (e.type === 'relation') return Math.max(typeWidth(byId.get(e.source)?.entityType), typeWidth(t?.entityType)) as WidthLevel;
    return typeWidth(t?.entityType);
  }, [byId, typeWidth]);
  const edgeWidth = useCallback((e: GraphEdge) => {
    if (edgeWidthMode === 'type') {
      const l = typeEdgeLevel(e);
      return l === null ? 1.1 : e.type === 'relation' ? WIDTH_PX[l] : DOC_WIDTH_FACTOR[l];
    }
    if (edgeWidthMode === 'doc') {
      const l = docWidth(edgeDoc(e));
      return e.type === 'relation' ? WIDTH_PX[l] : (e.type === 'contains' ? 1.1 : 1) * DOC_WIDTH_FACTOR[l];
    }
    if (e.type === 'relation') return groupWidth(e.group ?? 'other');
    return e.type === 'contains' ? 1.1 : 1;
  }, [edgeWidthMode, docWidth, edgeDoc, groupWidth, typeEdgeLevel]);
  const markerKey = (e: GraphEdge) => {
    const c = edgeColorMode === 'group' ? e.group ?? 'other' : edgeColorMode === 'doc' ? `d${[...docColors.keys()].indexOf(edgeDoc(e) ?? '')}` : 'm';
    const w = edgeWidthMode === 'group' ? e.group ?? 'other' : edgeWidthMode === 'doc' ? `w${docWidth(edgeDoc(e))}` : edgeWidthMode === 'type' ? `t${typeEdgeLevel(e) ?? 0}` : 'u';
    return `${c}-${w}`;
  };
  const markerId = (e: GraphEdge, interactive: boolean) => `eg-arrow${interactive ? '' : '-x'}-${markerKey(e)}`;
  const renderMarkers = (interactive: boolean) => {
    const seen = new Map<string, GraphEdge>();
    for (const e of edges) if (e.type === 'relation' && !seen.has(markerKey(e))) seen.set(markerKey(e), e);
    return (
      <defs>
        {[...seen].map(([key, e]) => (
          // 箭头尺寸默认随线宽放大；粗线上按 √ 收一点，免得箭头过大
          <marker key={key} id={`eg-arrow${interactive ? '' : '-x'}-${key}`} viewBox="0 0 10 10" refX="10" refY="5" markerWidth={6 * Math.sqrt(1.6 / edgeWidth(e))} markerHeight={6 * Math.sqrt(1.6 / edgeWidth(e))} orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill={edgeColor(e)} />
          </marker>
        ))}
      </defs>
    );
  };
  // 标签：屏幕坐标、固定字号，逐个避让。有焦点（悬停 / 选中 / 搜索）时只给相关节点标注
  const computeLabels = useCallback((v: View, sz: Size) => {
    const { k, tx, ty } = v;
    const sp = (p: Pt) => ({ sx: p.x * k + tx, sy: p.y * k + ty });
    // 悬停 / 选中节点的邻居标签一定显示（排在其他标签之前，优先占到好位置）
    const forcedIds = new Set<string>([...(focus ? [focus, ...(neighbors.get(focus) ?? [])] : []), ...(selected ? [selected] : []), ...(matches ?? []), ...(pathInfo && focus === selected ? pathInfo.nodes : [])]);
    const rank = (n: GraphNode) => (forcedIds.has(n.id) ? 0 : n.type === 'doc' ? 1 : n.type === 'section' ? (n.level === 1 ? 2 : 4) : 3);
    const eligible = nodes.filter((n) => {
      if (lit) return lit.has(n.id);
      if (n.type === 'doc') return true;
      if (n.type === 'section') return n.level === 1 ? k > 0.45 : k > 0.9;
      return k > 0.55 || n.degree >= 5;
    });
    const candidates = eligible
      .map((n) => {
        const p = pos.get(n.id);
        if (!p) return null;
        const text = n.label.length > MAX_LABEL ? `${n.label.slice(0, MAX_LABEL - 1)}…` : n.label;
        // 文档节点数量少、又是图的骨架：标签总是显示（周围挤时会往外找空位）
        return { key: n.id, text, ...sp(p), r: nodeR(n) * k, cls: `graph-label ${n.type}`, size: LABEL_PX[n.type], forced: forcedIds.has(n.id) || n.type === 'doc', rank: rank(n), degree: n.degree };
      })
      .filter((c): c is NonNullable<typeof c> => c !== null)
      .sort((a, b) => a.rank - b.rank || b.degree - a.degree);
    // 聚焦时其余节点已经变淡，不再当作障碍物，给邻居标签腾位置
    const circles = nodes.flatMap((n) => {
      const p = pos.get(n.id);
      if (!p || (lit && !lit.has(n.id))) return [];
      const { sx, sy } = sp(p);
      const r = nodeR(n) * k + 1;
      return [{ key: n.id, box: { x0: sx - r, y0: sy - r, x1: sx + r, y1: sy + r } }];
    });
    // 关系标签优先级最低：只在聚焦相关节点或放大到一定程度时尝试
    const edgeLabels = edges.flatMap((e, i) => {
      if (e.type !== 'relation' || !e.label) return [];
      if (lit ? !(lit.has(e.source) && lit.has(e.target)) : k < 1.3) return [];
      const a = pos.get(e.source);
      const b = pos.get(e.target);
      if (!a || !b) return [];
      return [{ key: `e${i}`, text: e.label, color: relColor(e), ...sp({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }) }];
    });
    return placeLabels(candidates, circles, edgeLabels, sz);
  }, [nodes, edges, pos, lit, focus, selected, matches, neighbors, pathInfo, relColor, nodeR]);
  const labels = useMemo(() => computeLabels(view, size), [computeLabels, view, size]);
  const renderScene = (v: View, lbls: PlacedLabel[], interactive: boolean) => (
    <>
    <g transform={`translate(${v.tx},${v.ty}) scale(${v.k})`}>
      {edges.map((e, i) => {
        const a = pos.get(e.source);
        const b = pos.get(e.target);
        const nb = byId.get(e.target);
        if (!a || !b || !nb) return null;
        const on = !lit || (lit.has(e.source) && lit.has(e.target));
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const d = Math.hypot(dx, dy) || 1;
        const r = nodeR(nb) + 2;
        const x2 = b.x - (dx / d) * r;
        const y2 = b.y - (dy / d) * r;
        const stroke = edgeColor(e);
        return (
          <g key={i} opacity={on ? 1 : 0.08}>
            <line x1={a.x} y1={a.y} x2={e.type === 'relation' ? x2 : b.x} y2={e.type === 'relation' ? y2 : b.y} stroke={stroke} strokeWidth={edgeWidth(e) / Math.sqrt(v.k)} strokeDasharray={e.type === 'mentions' ? '4 3' : undefined} strokeOpacity={e.type === 'mentions' ? 0.55 : e.type === 'contains' && edgeColorMode === 'doc' ? 0.45 : 0.9} markerEnd={e.type === 'relation' ? `url(#${markerId(e, interactive)})` : undefined} />
          </g>
        );
      })}
      {pathInfo && (!hover || hover === selected || pathInfo.nodes.has(hover)) ? pathInfo.edges.map((e) => {
        const a = pos.get(e.source);
        const b = pos.get(e.target);
        if (!a || !b) return null;
        return (
          <g key={`path:${e.source}>${e.target}>${e.type}`} className="graph-path">
            <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} className="graph-path-glow" strokeWidth={8 / v.k} />
            <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} className="graph-path-line" strokeWidth={2.4 / v.k} />
          </g>
        );
      }) : null}
      {nodes.map((n) => {
        const p = pos.get(n.id);
        if (!p) return null;
        const on = !lit || lit.has(n.id);
        const onPath = !!pathInfo?.nodes.has(n.id) && n.id !== selected;
        const r = nodeR(n);
        return (
          <g key={n.id} data-node-id={n.id} transform={`translate(${p.x},${p.y})`} opacity={on ? 1 : 0.15} className="graph-node" {...(interactive ? { onPointerDown: (e: React.PointerEvent) => onPointerDown(e, n.id), onPointerEnter: () => setHover(n.id), onPointerLeave: () => setHover(null) } : {})}>
            {onPath ? <circle r={r + 4 / v.k} fill="none" className="graph-path-ring" strokeWidth={2 / v.k} /> : null}
            {colorMode === 'doc' && n.type === 'entity' && (entityDocs.get(n.id)?.length ?? 0) > 1 ? <circle r={r + 2.6 / v.k} fill="none" stroke="var(--text-2)" strokeWidth={1.2 / v.k} strokeDasharray={`${3 / v.k} ${2 / v.k}`} /> : null}
            <circle r={r} fill={nodeFill(n)} stroke={n.id === selected ? 'var(--accent)' : matches?.has(n.id) ? 'var(--warning)' : 'var(--bg-2)'} strokeWidth={n.id === selected || matches?.has(n.id) ? 3 : 1.5} />
            {interactive ? <title>{`${TYPE_LABEL[n.type]}：${n.path ?? n.label}`}</title> : null}
          </g>
        );
      })}
    </g>
    <g className="graph-labels" aria-hidden="true">
      {lbls.map((l) => <text key={l.key} x={l.x} y={l.y} textAnchor={l.anchor} dominantBaseline="central" className={l.cls} fontSize={l.size} style={l.color ? { fill: l.color } : undefined}>{l.text}</text>)}
    </g>
    </>
  );

  // ---- 导出 PNG：当前视图（所见即所得）或完整图谱（全部节点、标签可读的缩放）
  const exportRef = useRef<SVGSVGElement>(null);
  type ExportFormat = 'png' | 'svg';
  const [exportJob, setExportJob] = useState<{ mode: 'view' | 'full'; format: ExportFormat; view: View; size: Size; labels: PlacedLabel[]; ratio: number } | null>(null);
  const [exportMenu, setExportMenu] = useState(false);
  const [exportFormat, setExportFormat] = useState<ExportFormat>('png');
  const startExport = (mode: 'view' | 'full') => {
    setExportMenu(false);
    const format = exportFormat;
    if (mode === 'view') { setExportJob({ mode, format, view, size, labels, ratio: 2 }); return; }
    const pts = [...pos.values()];
    if (!pts.length) return;
    const xs = pts.map((p) => p.x);
    const ys = pts.map((p) => p.y);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const padX = 160;                                   // 右侧留出标签宽度
    const padY = 50;
    let k = 1.15;
    const longest = Math.max(x1 - x0, y1 - y0) * k;
    if (longest > 5600) k *= 5600 / longest;            // 画布尺寸上限，避免超出浏览器 canvas 限制
    const sz = { w: Math.round((x1 - x0) * k + padX * 2), h: Math.round((y1 - y0) * k + padY * 2) };
    const v = { k, tx: padX - x0 * k, ty: padY - y0 * k };
    setExportJob({ mode, format, view: v, size: sz, labels: computeLabels(v, sz), ratio: Math.max(1, Math.min(2, 9000 / Math.max(sz.w, sz.h))) });
  };
  useEffect(() => {
    if (!exportJob || !exportRef.current) return;
    let cancelled = false;
    const run = async () => {
      try {
        const built = buildExportSvg(exportRef.current!, exportJob.size, EXPORT_HEADER);
        const blob = exportJob.format === 'svg' ? new Blob([built.xml], { type: 'image/svg+xml;charset=utf-8' }) : await svgToPng(built, exportJob.ratio);
        if (cancelled) return;
        const stamp = new Date();
        const pad = (n: number) => String(n).padStart(2, '0');
        const name = `实体织网-${(title || '知识库').replace(/[\\/:*?"<>|]/g, '_')}-${exportJob.mode === 'full' ? '完整' : '视图'}-${stamp.getFullYear()}${pad(stamp.getMonth() + 1)}${pad(stamp.getDate())}-${pad(stamp.getHours())}${pad(stamp.getMinutes())}.${exportJob.format}`;
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = name;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 30000);
        onExported?.(name, null);
      } catch (e) {
        onExported?.('', e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setExportJob(null);
      }
    };
    run();
    return () => { cancelled = true; };
  }, [exportJob]); // eslint-disable-line

  const sel = selected ? byId.get(selected) : undefined;

  return (
    <div ref={wrapRef} className={`graph-wrap ${fullscreen ? 'is-fullscreen' : ''} ${overlayFs ? 'is-overlay' : ''}`}>
      <div className="row between wrap gap-8 graph-toolbar">
        <div className="row gap-8 wrap">
          <div className="row gap-6 graph-search">
            <Search size={13} className="faint" />
            <input value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && matches?.size) centerOn([...matches][0]); }} placeholder="搜索实体 / 章节，回车定位" />
            {query ? <button className="btn ghost icon xs" onClick={() => setQuery('')} aria-label="清空搜索"><X size={12} /></button> : null}
          </div>
          {docList.length > 1 ? <ScopeMenu icon={<FileText size={11} />} allLabel="全部文档" title="文档" unit="实体" items={docList} selected={docFilter} onChange={setDocFilter}
            footer={sizeMode === 'doc' || edgeWidthMode === 'doc' ? (
              <>
                <span className="faint">{sizeMode === 'doc' && edgeWidthMode === 'doc' ? '左：节点大小　右：边粗细；' : sizeMode === 'doc' ? '节点大小：' : '边粗细：'}默认按实体数，多的大、少的小</span>
                <span className="row gap-8">
                  {sizeMode === 'doc' && docSizesCustomized ? <button className="link-btn" onClick={resetDocSizes}>恢复默认大小</button> : null}
                  {edgeWidthMode === 'doc' && docWidthsCustomized ? <button className="link-btn" onClick={resetDocWidths}>恢复默认粗细</button> : null}
                </span>
              </>
            ) : undefined} /> : null}
          {typeList.length > 0 ? <ScopeMenu icon={<Shapes size={11} />} allLabel="全部类型" title="实体类型" unit="个" items={typeList} selected={typeFilter} onChange={setTypeFilter}
            footer={sizeByType || edgeWidthMode === 'type' ? (
              <>
                <span className="faint">{sizeByType && edgeWidthMode === 'type' ? '左：节点大小　右：边粗细；' : sizeByType ? '节点大小：' : '边粗细：'}制度 / 系统默认最大，数值最小{edgeWidthMode === 'type' ? '；关系边取两端较粗的一档' : ''}</span>
                <span className="row gap-8">
                  {sizeByType && entitySizesCustomized ? <button className="link-btn" onClick={() => setEntitySize('', null)}>恢复默认大小</button> : null}
                  {edgeWidthMode === 'type' && typeWidthsCustomized ? <button className="link-btn" onClick={() => setTypeWidth('', null)}>恢复默认粗细</button> : null}
                </span>
              </>
            ) : undefined} /> : null}
          {relList.length > 0 ? <ScopeMenu icon={<GitBranch size={11} />} allLabel="全部关系" title="关系类别" unit="条" items={relList} selected={relFilter} onChange={setRelFilter}
            footer={widthByGroup ? <><span className="faint">线宽：因果 / 依赖默认最粗，结构关系最细</span>{relWidthsCustomized ? <button className="link-btn" onClick={() => setRelWidth('', null)}>恢复默认粗细</button> : null}</> : undefined}
            note={`类别由${(graph.relationGroupSource?.rule ?? 0) > 0 ? ` LLM 抽取时给出（${graph.relationGroupSource?.rule} 条旧关系按关键词推断，重建索引后更新）` : ' LLM 抽取时给出'}。筛选后只显示参与这些关系的实体；文档 / 章节结构保留`} /> : null}
          <label className="row gap-6 xs muted graph-check"><input type="checkbox" checked={showSections} onChange={(e) => setShowSections(e.target.checked)} />章节结构</label>
          <label className="row gap-6 xs muted graph-check"><input type="checkbox" checked={showMentions} onChange={(e) => setShowMentions(e.target.checked)} />提及关系</label>
          <span className="row gap-6 xs muted" title="边的颜色表示什么：关系类别（只影响实体之间的关系边）/ 所属文档（所有边）/ 单色">
            边着色
            <span className="color-mode" role="radiogroup" aria-label="边着色">
              {([['group', '类别'], ['doc', '文档'], ['mono', '单色']] as const).map(([m, l]) => <button key={m} role="radio" aria-checked={edgeColorMode === m} className={edgeColorMode === m ? 'on' : ''} onClick={() => setEdgeColorMode(m)}>{l}</button>)}
            </span>
          </span>
          <span className="row gap-6 xs muted" title="边粗细表示什么：关系类别（在「全部关系」菜单里逐类调整，只影响关系边）/ 所属文档（在「全部文档」菜单里逐篇调整，所有边）/ 实体类型（在「全部类型」菜单里逐类调整；关系边取两端较粗的一档）/ 统一">
            边粗细
            <span className="color-mode" role="radiogroup" aria-label="边粗细">
              {([['group', '类别'], ['doc', '文档'], ['type', '类型'], ['uniform', '统一']] as const).map(([m, l]) => <button key={m} role="radio" aria-checked={edgeWidthMode === m} className={edgeWidthMode === m ? 'on' : ''} onClick={() => setEdgeWidthMode(m)}>{l}</button>)}
            </span>
          </span>
          <span className="row gap-6 xs muted" title="节点大小表示什么：实体类型（在「全部类型」菜单里逐类调整）/ 所属文档（在「全部文档」菜单里逐篇调整）/ 连接数">
            节点大小
            <span className="color-mode" role="radiogroup" aria-label="节点大小">
              {([['type', '类型'], ['doc', '文档'], ['degree', '连接数']] as const).map(([m, l]) => <button key={m} role="radio" aria-checked={sizeMode === m} className={sizeMode === m ? 'on' : ''} onClick={() => setSizeMode(m)}>{l}</button>)}
            </span>
          </span>
          <span className="row gap-6 xs muted" title="节点颜色表示什么">
            节点着色
            <span className="color-mode" role="radiogroup" aria-label="节点着色">
              {([['type', '类型'], ['doc', '文档']] as const).map(([m, l]) => <button key={m} role="radio" aria-checked={colorMode === m} className={colorMode === m ? 'on' : ''} onClick={() => setColorMode(m)}>{l}</button>)}
            </span>
          </span>
          {matches ? <span className="xs faint">匹配 {matches.size} 个</span> : null}
        </div>
        <div className="row gap-6 wrap xs muted graph-legend">
          {colorMode === 'type' ? (
            <>
              <span className="row gap-4"><i className="dot-lg" style={{ background: 'var(--ink)' }} />文档</span>
              <span className="row gap-4"><i className="dot-lg" style={{ background: 'var(--c-dense)' }} />章节</span>
              <span className="row gap-4"><i className="dot-lg" style={{ background: 'var(--c-mmr)' }} />小节</span>
              <span className="row gap-4" title="实体按类型着色，颜色见「全部类型」菜单"><i className="dot-lg multi" />实体（按类型着色）</span>
            </>
          ) : (
            <>
              <span className="row gap-4" title="文档、章节与实体按所属文档着色，颜色见「全部文档」菜单"><i className="dot-lg docs" />节点（按文档着色）</span>
              <span className="row gap-4" title="实体被多篇文档定义或提到；颜色取定义它的文档"><i className="dot-lg ring" />跨文档实体</span>
            </>
          )}
          {edgeColorMode === 'group' ? <span className="row gap-4" title="关系边按类别着色，颜色见「全部关系」菜单"><i className="line-lg multi" />关系（按类别着色）</span>
            : edgeColorMode === 'doc' ? <span className="row gap-4" title="所有边按所属文档着色，颜色见「全部文档」菜单"><i className="line-lg docs" />边（按文档着色）</span>
            : <span className="row gap-4"><i className="line-lg" style={{ borderColor: 'var(--c-hyde)' }} />关系</span>}
          <span className="row gap-4"><i className="line-lg dashed" style={{ borderColor: 'var(--c-weaved)' }} />提及</span>
          {fullscreen ? <Button size="xs" onClick={toggleFullscreen}><Minimize2 size={11} />退出全屏</Button> : null}
        </div>
      </div>

      <div
        ref={canvasRef}
        className="graph-canvas"
        onDoubleClick={(e) => {
          e.preventDefault();
          // 双击节点 = 选中并居中；只有双击空白处才进出全屏
          const nodeEl = (e.target as Element).closest?.('[data-node-id]');
          if (nodeEl) centerOn(nodeEl.getAttribute('data-node-id')!);
          else toggleFullscreen();
        }}
      >
        <svg
          ref={svgRef}
          width={size.w}
          height={size.h}
          viewBox={`0 0 ${size.w} ${size.h}`}
          onPointerDown={(e) => onPointerDown(e)}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onPointerLeave={() => setHover(null)}
          role="img"
          aria-label="知识库结构拓扑图"
        >
          {renderMarkers(true)}
          {renderScene(view, labels, true)}
        </svg>

        <div className="graph-zoom" onDoubleClick={(e) => e.stopPropagation()}>
          <button className="btn icon xs" onClick={() => zoom(1.25)} title="放大" aria-label="放大"><Plus size={13} /></button>
          <button className="btn icon xs" onClick={() => zoom(0.8)} title="缩小" aria-label="缩小"><Minus size={13} /></button>
          <button className="btn icon xs" onClick={() => { setView(fitView([...pos.values()], size)); setSelected(null); }} title="显示全部节点" aria-label="适应画布"><Scan size={12} /></button>
          <button className="btn icon xs" onClick={toggleFullscreen} title={fullscreen ? '退出全屏（Esc）' : '全屏（双击画布空白处）'} aria-label={fullscreen ? '退出全屏' : '全屏'}>{fullscreen ? <Minimize2 size={12} /> : <Maximize2 size={12} />}</button>
          <button className={`btn icon xs ${showInfo ? 'active' : ''}`} onClick={() => setShowInfo((v) => !v)} title="图例说明" aria-label="图例说明"><Info size={12} /></button>
          <div className="graph-export">
            <button className={`btn icon xs ${exportMenu ? 'active' : ''}`} onClick={() => setExportMenu((v) => !v)} disabled={!!exportJob} title="导出图片（PNG / SVG）" aria-label="导出图片" aria-expanded={exportMenu}>
              {exportJob ? <Loader2 size={12} className="spin" /> : <Download size={12} />}
            </button>
            {exportMenu ? (
              <div className="graph-export-menu" role="menu">
                <div className="graph-export-format" role="radiogroup" aria-label="导出格式">
                  {(['png', 'svg'] as const).map((f) => (
                    <button key={f} role="radio" aria-checked={exportFormat === f} className={exportFormat === f ? 'active' : ''} onClick={() => setExportFormat(f)}>{f.toUpperCase()}</button>
                  ))}
                </div>
                <div className="xs faint graph-export-note">{exportFormat === 'png' ? '位图，适合贴进文档 / 聊天' : '矢量图，无限放大不失真，文字可搜索、可在 Figma / Illustrator 里编辑'}</div>
                <button role="menuitem" onClick={() => startExport('view')}><span className="strong">当前视图</span><span className="xs faint">所见即所得，含高亮{exportFormat === 'png' ? ' · 2 倍清晰度' : ''}</span></button>
                <button role="menuitem" onClick={() => startExport('full')}><span className="strong">完整图谱</span><span className="xs faint">全部 {nodes.length} 个节点，标签可读的缩放</span></button>
              </div>
            ) : null}
          </div>
        </div>
        {exportJob ? (
          <svg ref={exportRef} className="graph-export-stage" width={exportJob.size.w} height={exportJob.size.h} viewBox={`0 0 ${exportJob.size.w} ${exportJob.size.h}`} aria-hidden="true">
            {renderMarkers(false)}
            {renderScene(exportJob.view, exportJob.labels, false)}
          </svg>
        ) : null}

        {sel ? (
          <aside className="graph-float" onDoubleClick={(e) => e.stopPropagation()} onWheel={(e) => e.stopPropagation()}>
            <NodeDetail node={sel} graph={graph} byId={byId} paths={pathInfo?.paths ?? []} pathViaMentions={!!pathInfo?.viaMentions} onPick={centerOn} onClose={() => setSelected(null)} onOpenList={onOpenList} onOnlyDoc={docList.length > 1 ? (id) => onlyDoc(id) : undefined} docFiltered={!!docFilter} onShowAllDocs={() => setDocFilter(null)} entityDocs={sel.type === 'entity' ? (entityDocs.get(sel.id) ?? []).map((d) => ({ id: d, label: graph.nodes.find((n) => n.id === d)?.label ?? d, color: docColors.get(d) ?? 'var(--text-3)' })) : undefined} />
          </aside>
        ) : showInfo ? (
          <aside className="graph-float" onDoubleClick={(e) => e.stopPropagation()}>
            <div className="col gap-8 xs muted">
              <div className="row between">
                <span className="small strong row gap-6" style={{ color: 'var(--text-1)' }}><Network size={13} />结构拓扑</span>
                <button className="btn ghost icon xs" onClick={() => setShowInfo(false)} aria-label="关闭说明"><X size={12} /></button>
              </div>
              <div>骨架来自切片的标题路径：<b>文档 → 章节 → 小节</b>。实体挂在定义它的章节下；<b>紫色箭头</b>是实体间的三元组关系；<b>虚线</b>表示其他章节提到了这个实体——检索时召回了这些章节、却没召回定义块，UnWeaver 会把定义自动织入上下文。</div>
              <div>点击节点查看定义、关系与来源。</div>
            </div>
          </aside>
        ) : null}

        <div className="graph-hint xs faint">拖动平移 · 滚轮 / 双指捏合缩放 · 触控板双指滑动平移 · 拖动节点调整位置 · 单击空白处取消选中 · 双击空白处{fullscreen ? '退出' : '进入'}全屏</div>
      </div>
    </div>
  );
}

type ScopeItem = { id: string; label: string; count: number; color?: string; hint?: string; swatch?: React.ReactNode; extra?: React.ReactNode };
/** 多选筛选下拉：勾选多项 / 「只看」某一项 / 显示全部；至少保留一项。selected 为 null 表示全部。 */
function ScopeMenu({ icon, allLabel, title, unit, items, selected, onChange, note, footer }: { icon: React.ReactNode; allLabel: string; title: string; unit: string; items: ScopeItem[]; selected: Set<string> | null; onChange: (next: Set<string> | null) => void; note?: string; footer?: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, [open]);
  const ids = items.map((i) => i.id);
  const toggle = (id: string) => {
    const next = new Set(selected ?? ids);
    if (next.has(id)) { if (next.size === 1) return; next.delete(id); } else next.add(id);
    onChange(next.size === ids.length ? null : next);
  };
  const only = (id: string) => { onChange(ids.length > 1 ? new Set([id]) : null); setOpen(false); };
  const picked = selected ? items.filter((i) => selected.has(i.id)) : [];
  return (
    <div className="graph-docs" ref={ref}>
      <button className={`btn xs ${selected ? 'active' : ''}`} onClick={() => setOpen((v) => !v)} aria-expanded={open} aria-haspopup="true">
        {icon}{selected ? (picked.length === 1 ? picked[0].label : `${picked.length} / ${items.length} ${title}`) : `${allLabel}（${items.length}）`}<ChevronDown size={11} />
      </button>
      {open ? (
        <div className="graph-docs-menu" role="menu">
          <div className="row between xs graph-docs-head">
            <span className="faint">按{title}筛选</span>
            <button className="link-btn" onClick={() => { onChange(null); setOpen(false); }} disabled={!selected}>显示全部</button>
          </div>
          {note ? <div className="xs faint graph-docs-note">{note}</div> : null}
          {items.map((d) => {
            const on = !selected || selected.has(d.id);
            return (
              <div key={d.id} className={`graph-docs-row ${on ? 'on' : ''}`}>
                <label className="row gap-6 flex-1" style={{ minWidth: 0, cursor: 'pointer' }}>
                  <input type="checkbox" checked={on} onChange={() => toggle(d.id)} />
                  {d.swatch ?? (d.color ? <span className="type-dot" style={{ background: d.color }} /> : null)}
                  {d.hint ? (
                    <span className="col" style={{ minWidth: 0 }}>
                      <span className="ellipsis small" title={d.label}>{d.label}</span>
                      <span className="ellipsis xs faint" title={d.hint}>{d.hint}</span>
                    </span>
                  ) : <span className="ellipsis small" title={d.label}>{d.label}</span>}
                </label>
                {d.extra}
                <span className="xs faint">{d.count} {unit}</span>
                <button className="link-btn xs" onClick={() => only(d.id)}>只看</button>
              </div>
            );
          })}
          {footer ? <div className="graph-docs-foot xs">{footer}</div> : null}
        </div>
      ) : null}
    </div>
  );
}

function NodeDetail({ node, graph, byId, paths, pathViaMentions, onPick, onClose, onOpenList, onOnlyDoc, docFiltered, onShowAllDocs, entityDocs }: { node: GraphNode; graph: KbGraph; byId: Map<string, GraphNode>; paths: string[][]; pathViaMentions: boolean; onPick: (id: string) => void; onClose: () => void; onOpenList?: (entity: string) => void; onOnlyDoc?: (docNodeId: string) => void; docFiltered?: boolean; onShowAllDocs?: () => void; entityDocs?: { id: string; label: string; color: string }[] }) {
  // 用完整图（不受过滤影响）列出关系
  const all = useMemo(() => new Map(graph.nodes.map((n) => [n.id, n])), [graph]);
  const out = graph.edges.filter((e) => e.source === node.id);
  const inc = graph.edges.filter((e) => e.target === node.id);
  const link = (id: string) => {
    const n = all.get(id);
    if (!n) return null;
    return <button className="link-btn" onClick={() => byId.has(id) && onPick(id)} title={n.path ?? n.label} disabled={!byId.has(id)}>{n.label}</button>;
  };
  const group = (title: string, items: React.ReactNode[]) => (items.length ? (
    <div className="col gap-4">
      <div className="xs faint">{title}（{items.length}）</div>
      <div className="col gap-4">{items.slice(0, 30)}</div>
      {items.length > 30 ? <div className="xs faint">… 另有 {items.length - 30} 项</div> : null}
    </div>
  ) : null);

  return (
    <div className="col gap-12">
      <div className="row between gap-6" style={{ alignItems: 'flex-start' }}>
        <div style={{ minWidth: 0 }}>
          <div className="row gap-6 xs faint">{node.type === 'doc' ? <FileText size={11} /> : node.type === 'section' ? <Hash size={11} /> : <Network size={11} />}{TYPE_LABEL[node.type]}{node.type === 'section' && node.level ? ` · ${node.level} 级` : ''}</div>
          <div className="strong" style={{ wordBreak: 'break-word' }}>{node.label}</div>
          {node.path && node.path !== node.label ? <div className="xs muted" style={{ wordBreak: 'break-word' }}>{node.path}</div> : null}
        </div>
        <button className="btn ghost icon xs" onClick={onClose} aria-label="关闭详情"><X size={12} /></button>
      </div>
      <div className="row gap-6 wrap">
        <Badge>连接 {node.degree}</Badge>
        {node.chunks ? <Badge>{node.chunks} 个段落</Badge> : null}
        {node.type === 'entity' && node.entityType ? (
          <span className="badge" title={node.entityTypeSource === 'llm' ? '类型由 LLM 抽取时给出' : '类型按名称 / 定义规则推断（重建索引后由 LLM 给出）'} style={{ borderColor: `var(--et-${node.entityType})`, color: `var(--et-${node.entityType})` }}>
            <span className="type-dot" style={{ background: `var(--et-${node.entityType})`, width: 7, height: 7, borderRadius: '50%', display: 'inline-block' }} />
            {ENTITY_TYPE_LABEL[node.entityType] ?? node.entityType}{node.entityTypeSource === 'rule' ? ' · 推断' : ''}
          </span>
        ) : null}
        {(node.sources ?? []).map((s) => <Badge key={s} tone={s === 'llm' ? 'info' : 'weaved'}>{s === 'llm' ? 'LLM 抽取' : '正则定义'}</Badge>)}
      </div>
      {node.type !== 'doc' ? (
        <div className="col gap-4">
          <div className="xs faint row gap-6"><span className="graph-path-swatch" />来源路径{paths.length > 1 ? `（${paths.length} 条）` : ''}{pathViaMentions ? ' · 无定义，经提及 / 关系' : ''}</div>
          {paths.length ? paths.map((trail, i) => (
            <div key={i} className="graph-crumbs xs">
              {trail.map((id, j) => (
                <span key={id + j} className="row gap-4">
                  {j > 0 ? <span className="faint">›</span> : null}
                  {id === node.id ? <span className="strong">{all.get(id)?.label ?? id}</span> : link(id)}
                </span>
              ))}
            </div>
          )) : <div className="xs faint">当前视图里找不到通往文档的路径。</div>}
        </div>
      ) : null}
      {node.type === 'entity' ? (
        <>
          {group('所属文档', (entityDocs ?? []).map((d, i) => (
            <div key={d.id} className="xs row gap-6">
              <span className="doc-chip"><i style={{ background: d.color }} />{byId.has(d.id) ? <button className="link-btn" onClick={() => onPick(d.id)}>{d.label}</button> : d.label}</span>
              {i === 0 && (entityDocs?.length ?? 0) > 1 ? <span className="faint">主文档</span> : null}
            </div>
          )))}
          {group('定义', (node.definitions ?? []).map((d, i) => <div key={i} className="chunk xs" style={{ padding: '6px 8px' }}>{d.text}</div>))}
          {group('关系', [
            ...out.filter((e) => e.type === 'relation').map((e, i) => <div key={`o${i}`} className="xs row gap-4 wrap"><span className="strong">{node.label}</span><Badge tone="info">{e.label}</Badge>{link(e.target)}</div>),
            ...inc.filter((e) => e.type === 'relation').map((e, i) => <div key={`i${i}`} className="xs row gap-4 wrap">{link(e.source)}<Badge tone="info">{e.label}</Badge><span className="strong">{node.label}</span></div>),
          ])}
          {group('定义所在', inc.filter((e) => e.type === 'defines').map((e, i) => <div key={i} className="xs">{link(e.source)}</div>))}
          {group('被这些章节提到', inc.filter((e) => e.type === 'mentions').map((e, i) => <div key={i} className="xs">{link(e.source)}</div>))}
          {onOpenList ? <Button size="xs" onClick={() => onOpenList(node.label)}>在列表中查看</Button> : null}
        </>
      ) : (
        <>
          {group('下级章节', out.filter((e) => e.type === 'contains').map((e, i) => <div key={i} className="xs">{link(e.target)}</div>))}
          {group('定义的实体', out.filter((e) => e.type === 'defines').map((e, i) => <div key={i} className="xs">{link(e.target)}</div>))}
          {group('提到的实体', out.filter((e) => e.type === 'mentions').map((e, i) => <div key={i} className="xs">{link(e.target)}</div>))}
          {group('上级', inc.filter((e) => e.type === 'contains').map((e, i) => <div key={i} className="xs">{link(e.source)}</div>))}
          {!out.length && !inc.length ? <div className="xs faint">没有连接。</div> : null}
          {node.type === 'doc' && onOnlyDoc ? (
            <div className="row gap-6 wrap">
              <Button size="xs" onClick={() => onOnlyDoc(node.id)}><FileText size={11} />只看这篇文档</Button>
              {docFiltered && onShowAllDocs ? <Button size="xs" variant="ghost" onClick={onShowAllDocs}>显示全部文档</Button> : null}
            </div>
          ) : null}
          <div className="xs faint">{EDGE_LABEL.contains}关系来自标题路径；章节下的实体来自 UnWeaver 抽取。</div>
        </>
      )}
    </div>
  );
}
