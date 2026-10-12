"""数据清洗与预处理："垃圾进，垃圾出"的第一道闸。

覆盖五类问题：
1. 页眉 / 页脚 / 页码 / 水印：MinerU 已打标的直接丢（首页上只出现一次的"页眉"多半是文档标题，保留）；
   未打标的按"同一行出现在多数页面"识别
2. PDF 断行：一行一个换行的半截句子拼回完整段落
3. 乱码 / 不可打印字符：占比超阈值的块丢弃
4. 重复内容：精确重复 + 近似重复（字符 shingle Jaccard），避免权重虚高
5. PDF 文本层的字形问题（在真实 PDF 上踩到的）：
   - 康熙部首：Chrome / Skia 导出的 PDF 常把"音 手 一 文"映射成 ⾳ ⼿ ⼀ ⽂（U+2F00 段），肉眼一样，
     但"语⾳"≠"语音"，BM25 与向量都匹配不上 → 归一化回统一汉字；
   - 图标字体项目符号：Font Awesome 等私用区字符（U+E000-F8FF）直接删掉会把列表项粘成一行 → 转成 `- `；
   - OCR 把空心圆点 ○ 认成句号"。"，出现在行首 → 同样按项目符号处理。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from ..config import CleaningConfig
from ..schemas import Block, BlockType, ParsedDocument

_TERMINALS = "。！？!?；;：:…”\"』」）)]】》"
_LIST_MARK = re.compile(r"^\s*(?:[-*•▪]|\d+[.)、]|[（(]\d+[)）]|[一二三四五六七八九十]+[、.]|[①-⑳]|第[一二三四五六七八九十百]+[条章节])")
_CJK = re.compile(r"[一-鿿㐀-䶿]")
_ALLOWED = re.compile(r"[一-鿿㐀-䶿　-〿＀-￯ -⁯A-Za-z0-9\s\.,;:!?'\"()\[\]{}<>/\\|@#$%^&*_+=~`\-–—…·•■□●○★☆→←↑↓×÷±°℃‰€£¥₩§¶†‡]")
_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"[ \t　]+")
_RADICALS = re.compile(r"[\u2e80-\u2eff\u2f00-\u2fdf]")          # CJK 部首补充 + 康熙部首
_PUA = re.compile(r"[\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd]")   # 私用区：图标字体字形
_PUA_LINE_START = re.compile(r"^(\s*)" + _PUA.pattern + r"\s*")
_PUA_INLINE = re.compile(r"[ \t]*" + _PUA.pattern + r"[ \t]*")
_SOLID_BULLETS = "•●▪·・"
_HOLLOW_BULLETS = "。○◦"                              # 行首的"。"是 OCR 对空心圆点的误识别
_BULLET_LINE = re.compile(rf"^(\s*)(?:[-*]\s+)?([{_SOLID_BULLETS}{_HOLLOW_BULLETS}])\s*(?=\S)")
_TEXTUAL = (BlockType.TEXT, BlockType.LIST, BlockType.CAPTION, BlockType.FOOTNOTE)


def clean_document(doc: ParsedDocument, cfg: CleaningConfig) -> ParsedDocument:
    blocks = _rescue_first_page_header(doc.blocks)
    blocks = [b for b in blocks if b.type not in (BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER)]
    blocks = [_normalize_block(b, cfg) for b in blocks]
    blocks = [b for b in blocks if b.type == BlockType.TITLE or len(b.text.strip()) >= cfg.min_block_chars]
    if cfg.drop_headers_footers:
        blocks = drop_repeated_page_lines(blocks, cfg.header_footer_min_page_ratio)
    blocks = [b for b in blocks if not is_garbled(b.text, cfg.garbled_ratio_threshold)]
    if cfg.dedup_near_duplicates:
        blocks = dedup_blocks(blocks, cfg.dedup_jaccard_threshold)
    doc.blocks = blocks
    return doc


def _rescue_first_page_header(blocks: list[Block]) -> list[Block]:
    """版面模型会把首页顶部的文档标题标成页眉。真正的页眉会跨页重复；只在首页出现一次的，当正文留下。"""
    headers = [b for b in blocks if b.type == BlockType.HEADER and b.text.strip()]
    pages = [b.page for b in blocks if b.page is not None]
    if not headers or not pages:
        return blocks
    first = min(pages)
    seen_on: dict[str, set[int | None]] = {}
    for h in headers:
        seen_on.setdefault(_page_line_key(h.text), set()).add(h.page)
    out: list[Block] = []
    for b in blocks:
        if b.type == BlockType.HEADER and b.page == first and len(b.text.strip()) >= 4 and seen_on.get(_page_line_key(b.text)) == {first}:
            b = b.model_copy(update={"type": BlockType.TEXT, "meta": {**b.meta, "rescued_header": True}})
        out.append(b)
    return out


# CJK 部首补充区里"整字形"的部首：NFKC 只覆盖康熙部首（214 个全部有映射），这一段要自己列。
# 只收录与简体 / 通用汉字一一对应、在 PDF 文本层里确实会出现的（Unicode EquivalentUnifiedIdeograph）。
_RADICAL_SUPPLEMENT = {
    0x2E9D: "月", 0x2E9F: "母", 0x2EA0: "民", 0x2EC1: "虎", 0x2EC4: "西", 0x2EC5: "见", 0x2EC6: "角", 0x2EC8: "讠", 0x2EC9: "贝", 0x2ECB: "车",
    0x2ED0: "钅", 0x2ED1: "長", 0x2ED3: "长", 0x2ED4: "门", 0x2ED7: "雨", 0x2ED8: "青", 0x2ED9: "韦", 0x2EDA: "页", 0x2EDB: "风", 0x2EDC: "飞",
    0x2EDD: "食", 0x2EE0: "饣", 0x2EE2: "马", 0x2EE3: "骨", 0x2EE4: "鬼", 0x2EE5: "鱼", 0x2EE6: "鸟", 0x2EE7: "卤", 0x2EE8: "麦", 0x2EE9: "黄",
    0x2EEA: "黾", 0x2EEC: "齐", 0x2EEE: "齿", 0x2EF0: "龙", 0x2EF3: "龟",
}


def _radical_to_ideograph(m: re.Match[str]) -> str:
    ch = m.group(0)
    return _RADICAL_SUPPLEMENT.get(ord(ch)) or unicodedata.normalize("NFKC", ch)


def normalize_glyphs(text: str) -> str:
    """部首字形 → 统一汉字；私用区图标字形 → 项目符号（行首 / 重复出现）或空格。"""
    text = _RADICALS.sub(_radical_to_ideograph, text)
    if not _PUA.search(text):
        return text
    listy = len(_PUA.findall(text)) >= 2
    lines: list[str] = []
    for ln in text.split("\n"):
        m = _PUA_LINE_START.match(ln)
        if m:
            ln = f"{m.group(1)}- {ln[m.end():]}"
        # 行中的图标字形：成组出现时是被压到一行里的列表项，拆行；孤立的一个只留空格，避免把前后词粘在一起
        ln = _PUA_INLINE.sub("\n- " if listy else " ", ln)
        lines.extend(x.rstrip() for x in ln.split("\n"))
    return "\n".join(lines)


def normalize_bullets(text: str) -> str:
    """行首的 • ● ○ 。等统一成 `- `（空心的缩进一级），`- • 第1集` 这种双重符号只留一个。"""
    out: list[str] = []
    for ln in text.split("\n"):
        m = _BULLET_LINE.match(ln)
        if m:
            indent = m.group(1) + ("  " if m.group(2) in _HOLLOW_BULLETS else "")
            ln = f"{indent}- {ln[m.end():]}"
        out.append(ln)
    return "\n".join(out)


def _squeeze(ln: str, keep_indent: bool) -> str:
    body = _WS.sub(" ", ln.strip())
    if not keep_indent or not body:
        return body
    return " " * (len(ln) - len(ln.lstrip(" "))) + body


def _normalize_block(b: Block, cfg: CleaningConfig) -> Block:
    text = unicodedata.normalize("NFC", b.text).replace("　", " ").replace("\xa0", " ")
    text = normalize_glyphs(text)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    if b.type in _TEXTUAL:
        text = normalize_bullets(text)
    if b.type in _TEXTUAL or b.type == BlockType.TITLE:
        if cfg.fix_pdf_linebreaks and b.type != BlockType.LIST:
            text = fix_linebreaks(text)
        # 列表保留行首缩进（嵌套层级），其余类型去掉
        text = "\n".join(_squeeze(ln, keep_indent=b.type == BlockType.LIST) for ln in text.split("\n"))
    b.text = text.strip("\n").rstrip() if b.type == BlockType.LIST else text.strip()
    return b


def fix_linebreaks(text: str) -> str:
    """把 PDF 抽取出的硬换行拼回句子：上一行不以终止标点结尾、下一行不是列表项则合并。
    中文之间直接拼接，英文之间补空格，连字符断词（"informa-\\ntion"）去掉连字符。"""
    lines = [ln.strip() for ln in text.split("\n")]
    out: list[str] = []
    for ln in lines:
        if not ln:
            if out and out[-1]:
                out.append("")
            continue
        if not out or not out[-1] or _LIST_MARK.match(ln) or out[-1][-1] in _TERMINALS:
            out.append(ln)
            continue
        prev = out[-1]
        if prev.endswith("-") and ln[:1].isalpha():
            out[-1] = prev[:-1] + ln
        elif _CJK.search(prev[-1:]) or _CJK.search(ln[:1]):
            out[-1] = prev + ln
        else:
            out[-1] = prev + " " + ln
    return "\n".join(out).strip()


def is_garbled(text: str, threshold: float) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    bad = sum(1 for ch in stripped if not _ALLOWED.match(ch))
    return bad / len(stripped) > threshold


def drop_repeated_page_lines(blocks: list[Block], min_ratio: float) -> list[Block]:
    """跨页重复出现、且位于页面边缘的短行（公司名、文档名、"第 x 页"）视为页眉页脚。

    两个条件缺一不可：数字归一化后的文本出现在 ≥ min_ratio 的页面上；块位于页面顶部 / 底部
    （有 bbox 时按 y 坐标判断，否则取该页的首块 / 末块）。只看重复会误伤"仅数字不同"的正文行。
    需要块带页码，且总页数 ≥ 3。
    """
    pages = {b.page for b in blocks if b.page is not None}
    if len(pages) < 3:
        return blocks
    edge_ids = _page_edge_block_ids(blocks)
    seen_on: dict[str, set[int]] = {}
    for b in blocks:
        if b.id in edge_ids and _is_furniture_candidate(b):
            seen_on.setdefault(_page_line_key(b.text), set()).add(b.page)
    threshold = max(3, int(len(pages) * min_ratio))
    repeated = {k for k, ps in seen_on.items() if len(ps) >= threshold}
    if not repeated:
        return blocks
    return [b for b in blocks if not (b.id in edge_ids and _is_furniture_candidate(b) and _page_line_key(b.text) in repeated)]


def _is_furniture_candidate(b: Block) -> bool:
    return b.page is not None and b.type in (BlockType.TEXT, BlockType.OTHER, BlockType.TITLE) and len(b.text) <= 80 and "\n" not in b.text


def _page_edge_block_ids(blocks: list[Block]) -> set[str]:
    """页面顶部 / 底部的块：bbox y0 < 12% 或 y1 > 88%（兼容 0-1 与 0-1000 坐标）；无 bbox 时取每页首末块。"""
    by_page: dict[int, list[Block]] = {}
    for b in blocks:
        if b.page is not None:
            by_page.setdefault(b.page, []).append(b)
    edge: set[str] = set()
    for page_blocks in by_page.items():
        pb = page_blocks[1]
        with_bbox = [b for b in pb if b.bbox and len(b.bbox) == 4]
        if with_bbox:
            scale = 1000.0 if max(max(b.bbox) for b in with_bbox) > 1.5 else 1.0
            for b in with_bbox:
                y0, y1 = b.bbox[1] / scale, b.bbox[3] / scale
                if y0 < 0.12 or y1 > 0.88:
                    edge.add(b.id)
        else:
            edge.add(pb[0].id)
            edge.add(pb[-1].id)
    return edge


def _page_line_key(text: str) -> str:
    return _DIGITS.sub("#", _WS.sub("", text)).lower()


def dedup_blocks(blocks: list[Block], jaccard_threshold: float, shingle: int = 5, near_dup_limit: int = 4000) -> list[Block]:
    out: list[Block] = []
    exact: set[str] = set()
    shingles: list[set[int]] = []
    for b in blocks:
        norm = _WS.sub("", b.text)
        if b.type == BlockType.TITLE:
            out.append(b)
            continue
        h = hashlib.md5(norm.encode("utf-8")).hexdigest()
        if h in exact:
            continue
        exact.add(h)
        if len(norm) >= 30 and len(out) < near_dup_limit:
            s = {hash(norm[i:i + shingle]) for i in range(len(norm) - shingle + 1)}
            dup = False
            for other in shingles:
                if other and _jaccard(s, other) >= jaccard_threshold:
                    dup = True
                    break
            if dup:
                continue
            shingles.append(s)
        out.append(b)
    return out


def _jaccard(a: set[int], b: set[int]) -> float:
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)
