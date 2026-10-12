"""图片 → 文本（路线一：文本描述法）。OCR 与 VLM 分工：

- OCR 提取图里的**精确文字**（表格数字、坐标轴标签、编号）——让 LLM 看图读数会"四舍五入"，数值以 OCR 为准；
- VLM 负责**语义理解**（"这是一张展示 Q3 各区域销售额的柱状图"）。

两者拼成"图片文本摘要"后进入描述索引（向量 + BM25），CLIP 不擅长的细粒度查询就靠它。
VLM 结果按图片内容哈希缓存，重建索引不重复付费。
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

from ..config import MultimodalConfig
from .embedders import image_to_base64

logger = logging.getLogger(__name__)

CAPTION_PROMPT = """请描述这张图片，供知识库检索使用。要求：
1. 先说明图片类型（柱状图 / 折线图 / 饼图 / 流程图 / 架构图 / 表格截图 / 界面截图 / 照片 / 示意图等）和主题；
2. 列出图中可见的关键文字、标签、图例；数值只写你能清楚看到的，看不清就不要写，不要估算；
3. 概括数据趋势或要表达的关键信息；
4. 用一段连贯的中文输出，不要用列表，不超过 200 字。
{hint}
描述："""

_OCR_MAX_SIDE = 2000
_WORD_GAP = 1.5   # 相邻两字的中心距（按字宽归一化后）超过本行中位数的 1.5 倍，视为英文单词之间的空格
_MIN_LATIN = 8    # 只处理至少 8 个拉丁字母、且模型一个空格都没输出的行
# 字宽（em 的近似值）：M / W 这类宽字母会把相邻字的中心距撑大，不归一化会拆出 "M ale""W hite""Im ageNet"
_NARROW, _SEMI, _WIDE = set("ijl'.,:;!|"), set("ftrI()-/[]"), set("mwMW%@")
_MEANINGFUL = re.compile(r"[A-Za-z0-9一-鿿]")
_OCR_ORDER = ("rapidocr", "paddleocr", "tesseract")
_OCR_MODULES = {"rapidocr": ("rapidocr_onnxruntime", "rapidocr"), "paddleocr": ("paddleocr",), "tesseract": ("pytesseract",)}


def available_ocr(preference: str = "auto") -> str | None:
    """返回可用的 OCR 引擎名；none 或都没装时返回 None。"""
    if preference == "none":
        return None
    names = _OCR_ORDER if preference == "auto" else (preference,)
    for name in names:
        if any(importlib.util.find_spec(m) is not None for m in _OCR_MODULES.get(name, ())):
            return name
    return None


class OCREngine:
    """rapidocr（onnxruntime，CPU 友好，推荐）/ paddleocr / pytesseract 的薄封装，懒加载。"""

    def __init__(self, preference: str = "auto", min_chars: int = 2) -> None:
        self.name = available_ocr(preference)
        self.min_chars = min_chars
        self._engine: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._engine is not None:
            return self._engine
        if self.name == "rapidocr":
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError:
                from rapidocr import RapidOCR
            self._engine = RapidOCR()
        elif self.name == "paddleocr":
            from paddleocr import PaddleOCR

            self._engine = PaddleOCR(use_angle_cls=True, lang="ch")
        elif self.name == "tesseract":
            import pytesseract

            self._engine = pytesseract
        return self._engine

    def read(self, image: Any) -> str:
        if self.name is None:
            return ""
        if max(image.size) > _OCR_MAX_SIDE:   # 超大图（整页扫描 / 手机原图）缩到 OCR 的有效分辨率：更快，精度不降
            image = image.copy()
            image.thumbnail((_OCR_MAX_SIDE, _OCR_MAX_SIDE))
        try:
            with self._lock:
                text = self._read(image)
        except Exception as exc:
            logger.warning("OCR 失败（%s）：%s", self.name, exc)
            return ""
        text = " ".join(text.split())
        # 装饰图 / 背景图的 OCR 结果往往是几个无意义字符，过滤掉避免污染 BM25
        return text if len(_MEANINGFUL.findall(text)) >= self.min_chars else ""

    def _read(self, image: Any) -> str:
        import numpy as np

        engine = self._load()
        arr = np.asarray(image.convert("RGB"))
        boxes: list[tuple[float, float, float, float, str]] = []
        if self.name == "rapidocr":
            try:
                res = engine(arr, return_word_box=True)   # 带逐字框：用来补回英文单词间被吞掉的空格
            except TypeError:
                res = engine(arr)
            if hasattr(res, "txts"):  # rapidocr >= 2
                for box, txt in zip(res.boxes if res.boxes is not None else [], res.txts or []):
                    boxes.append(_box(box, txt))
            else:
                result = res[0] if isinstance(res, tuple) else res
                boxes = [_box(line[0], (_respace(line[4], line[3]) if len(line) >= 5 else None) or line[1]) for line in (result or [])]
        elif self.name == "paddleocr":
            result = engine.ocr(arr, cls=True)
            boxes = [_box(w[0], w[1][0]) for line in (result or []) if line for w in line]
        else:
            return engine.image_to_string(image, lang="chi_sim+eng")
        return layout_ocr(boxes, image.size[0])


def _char_width(c: str) -> float:
    if c in _NARROW:
        return 0.28
    if c in _SEMI:
        return 0.36
    if c in _WIDE:
        return 0.88
    return 0.68 if c.isupper() else 0.55


def _respace(chars: Any, char_boxes: Any) -> str | None:
    """按逐字框补回英文单词之间的空格。rapidocr 的中文识别模型经常整行不输出空格：
    CLIP 论文的图实测得到 "(1)Contrastivepre-training""Pepperthe"，英文关键词检索全部落空。
    字框中心来自 CTC 解码位置；相邻两字的中心距按两字的大致字宽归一化，超过本行中位数 1.5 倍的地方补空格。
    只处理模型一个空格都没输出、且至少 8 个拉丁字母的行（已有空格的行信模型）；只在字母 / 数字之间补
    （以及")"之后、"("之前），两个数字之间永远不拆。
    标准答案测试（论文图注渲染成图，3 种字体 × 3 种字号，1260 行）：单词召回 0.850 → 0.882，精度 0.900 → 0.906；
    中文图（300 张合成图里的 120 张）的输出一字不变。"""
    try:
        chars = [str(c) for c in chars]
        centers = [(min(float(p[0]) for p in b) + max(float(p[0]) for p in b)) / 2 for b in char_boxes]
    except (TypeError, ValueError, IndexError):
        return None
    if len(chars) < 3 or len(chars) != len(centers) or any(c.isspace() for c in chars):
        return None
    if sum(c.isascii() and c.isalpha() for c in chars) < _MIN_LATIN:
        return None
    steps = [(centers[i] - centers[i - 1]) / ((_char_width(chars[i - 1]) + _char_width(chars[i])) / 2) for i in range(1, len(centers))]
    median = sorted(steps)[len(steps) // 2]
    if median <= 0:
        return None
    word = lambda c: c.isascii() and c.isalnum()  # noqa: E731
    out = [chars[0]]
    for i in range(1, len(chars)):
        a, b = chars[i - 1], chars[i]
        if steps[i - 1] > _WORD_GAP * median and (word(a) or a == ")") and (word(b) or b == "(") and not (a.isdigit() and b.isdigit()):
            out.append(" ")
        out.append(b)
    return "".join(out)


def _box(points: Any, text: Any) -> tuple[float, float, float, float, str]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return (min(xs), min(ys), max(xs), max(ys), str(text))


def layout_ocr(boxes: list[tuple[float, float, float, float, str]], width: float) -> str:
    """把 OCR 文本框按版面拼回文字，保住"哪个数值属于哪个标签"。

    OCR 默认按从上到下的阅读顺序输出。对柱状图这是错的：数值标签的高度随柱子高低变化，
    实测得到 "450 380 300 210 华北区 华东区 华南区 西部区"，LLM 按顺序配对就把华东区读成了 380。
    这里把竖直方向对齐的文本框归成一列（柱子上方的数值 + 下方的类目），一列一组输出：
    "300 华北区 | 450 华东区 | 380 华南区 | 210 西部区"。标题这类跨度很大的文本框、以及没有形成列的
    零散文本，按阅读顺序放在前后。没有列结构的图（截图、扫描件）保持原来的阅读顺序。"""
    items = [b for b in boxes if b[4].strip()]
    if len(items) < 4:
        return " ".join(b[4] for b in _reading_order(items))
    # 标题 / 长句不参与分列：跨度超过图宽 40%，或比典型标签宽 2.5 倍以上
    widths = sorted(b[2] - b[0] for b in items)
    typical = widths[len(widths) // 2]
    is_wide = lambda b: (b[2] - b[0]) > 0.4 * width or (b[2] - b[0]) > 2.5 * typical  # noqa: E731
    wide = [b for b in items if is_wide(b)]
    narrow = sorted((b for b in items if not is_wide(b)), key=lambda b: (b[0] + b[2]) / 2)
    groups: list[list[tuple[float, float, float, float, str]]] = []
    for b in narrow:
        cx = (b[0] + b[2]) / 2
        if groups:
            g = groups[-1]
            gx0, gx1 = min(x[0] for x in g), max(x[2] for x in g)
            gcx = (gx0 + gx1) / 2
            if gx0 <= cx <= gx1 and b[0] <= gcx <= b[2]:   # 互相覆盖对方的中心：同一列
                g.append(b)
                continue
        groups.append([b])
    columns = [g for g in groups if len(g) >= 2]
    if len(columns) < 2:
        return " ".join(b[4] for b in _reading_order(items))
    top = min(b[1] for g in columns for b in g)
    loose = wide + [b for g in groups if len(g) < 2 for b in g]
    head = [b for b in loose if b[3] <= top]
    tail = [b for b in loose if b[3] > top]
    parts = [" ".join(b[4] for b in _reading_order(head))] if head else []
    parts.append(" | ".join(" ".join(b[4] for b in sorted(g, key=lambda x: x[1])) for g in columns))
    if tail:
        parts.append(" ".join(b[4] for b in _reading_order(tail)))
    return " ".join(p for p in parts if p)


def _reading_order(items: list[tuple[float, float, float, float, str]]) -> list[tuple[float, float, float, float, str]]:
    """按行（纵向中心相近）再按横向排序。"""
    rows: list[list[tuple[float, float, float, float, str]]] = []
    for b in sorted(items, key=lambda x: (x[1] + x[3]) / 2):
        cy, h = (b[1] + b[3]) / 2, b[3] - b[1]
        if rows and abs((rows[-1][0][1] + rows[-1][0][3]) / 2 - cy) <= max(h, 1) * 0.5:
            rows[-1].append(b)
        else:
            rows.append([b])
    return [b for row in rows for b in sorted(row, key=lambda x: x[0])]


class ImageDescriber:
    def __init__(self, cfg: MultimodalConfig, vlm=None, cache_path: Path | None = None) -> None:
        self.cfg = cfg
        self.vlm = vlm if (cfg.vlm_caption and vlm is not None and getattr(vlm, "supports_vision", False)) else None
        self.ocr = OCREngine(cfg.ocr, cfg.ocr_min_chars)
        self.cache_path = cache_path
        self._cache: dict[str, str] | None = None

    @property
    def capabilities(self) -> dict[str, Any]:
        return {"ocr": self.ocr.name, "vlm": getattr(self.vlm, "model", None) if self.vlm is not None else None}

    # ----------------------------------------------------------------------------------
    def describe(self, image: Any, sha1: str, *, hint: str = "") -> tuple[str, str]:
        """返回 (ocr_text, description)。任何一项不可用时为空串。"""
        ocr_text = self.ocr.read(image)
        description = self.caption(image, sha1, hint=hint or ocr_text[:200])
        return ocr_text, description

    def caption(self, image: Any, sha1: str, *, hint: str = "") -> str:
        if self.vlm is None:
            return ""
        key = f"{getattr(self.vlm, 'model', 'vlm')}:{sha1}"
        cache = self._load_cache()
        if key in cache:
            return cache[key]
        prompt = CAPTION_PROMPT.format(hint=f"（参考信息：{hint}）\n" if hint else "")
        try:
            data_url = "data:image/jpeg;base64," + image_to_base64(image, max_side=self.cfg.caption_max_side)
            text = self.vlm.chat([self.vlm.user_message(prompt, data_url)], max_tokens=600).text.strip()
        except Exception as exc:
            logger.warning("VLM 图片描述失败：%s", exc)
            return ""
        cache[key] = text
        self._save_cache()
        return text

    def _load_cache(self) -> dict[str, str]:
        if self._cache is None:
            self._cache = {}
            if self.cache_path is not None and self.cache_path.exists():
                try:
                    self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    self._cache = {}
        return self._cache

    def _save_cache(self) -> None:
        if self.cache_path is None or self._cache is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            logger.debug("描述缓存写入失败：%s", exc)
